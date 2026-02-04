# Storage Architecture

What gets stored when a failure happens.

## Episode Data

When an agent fails, Vritti captures:

### Core Context
```python
goal: str              # What was being attempted
error_trace: str       # Full error message
error_class: str       # Type: configuration/network/permission/resource/etc
severity: int          # 1=critical, 5=minor
resolution: str        # How it was fixed (optional)
tags: list[str]        # Keywords for search
```

### Execution Trace
```python
tool_chain: list[str]     # Tools used: ["docker", "kubectl"]
actions_taken: list[str]  # Commands run before failure
```

### Environment
```python
environment_info: dict  # OS, versions, config
```

**Example**:
```json
{
  "goal": "Deploy to K8s",
  "error_trace": "ImagePullBackOff: image not found",
  "error_class": "resource_error",
  "severity": 2,
  "tool_chain": ["kubectl"],
  "actions_taken": ["kubectl apply -f deployment.yaml"],
  "environment_info": {"k8s_version": "1.28"}
}
```

## AI-Generated Reflection

LLM analyzes each failure and generates:

```python
root_cause: str                 # Why it failed
resolution_strategy: str        # How to fix it
preconditions: dict[str, str]   # When this applies (structured key/value context)
environment_factors: list[str]  # Runtime/env constraints
affected_components: list[str]  # Components touched by the failure
confidence_score: float        # Reliability (0-1)
```

**Example**:
```json
{
  "root_cause": "Image tag 'latest' doesn't exist in registry",
  "resolution_strategy": "Use specific version tag like v1.2.3",
  "preconditions": {"tool": "kubectl", "image_tag": "latest"},
  "environment_factors": ["kubernetes", "private-registry"],
  "confidence_score": 0.95
}
```

Compatibility note:
- Readers/writers should accept both structured dict form and legacy `["key=value"]` lists.
- Normalize internally to a canonical representation before matching/ranking.

## Storage Backend: KyroDB

**Two Instances Required**:
- **Text** (port 50051): 384-dim or 768-dim embeddings
- **Image** (port 50052): 512-dim CLIP embeddings

### Episode Organization

```
customer-123:failures/
├── text embeddings (384-dim)
│   └── episode_456_text_vector
└── image embeddings (512-dim)
    └── episode_456_image_vector
```

Each episode stored with:
- Vector embeddings (text + optional image)
- Metadata (JSON)
- Reflection (if generated)

## Search Process

When checking if action is safe:

1. **Embed query** - Convert action to vector
2. **Vector search** - Find similar episodes (cosine similarity)
3. **Precondition match** - Filter by context
4. **Rank results** - Combine vector + precondition scores
5. **Return top matches** - With reflections

Archived episodes are excluded from search by default (`include_archived=false`).

**Example Search**:
```
Query: "kubectl apply failed"
→ Vector search finds 20 candidates
→ Precondition filter: tool=kubectl, cluster=prod
→ Top 5 matches returned with solutions
```

## Cost Efficiency

### Free Tier (Default - $0/month)
- 90% episodes: OpenRouter free models
- 10% critical: Multi-perspective (free tier)
- **Total**: $0.00/month for LLM costs

### Vector Storage
- KyroDB: Self-hosted, no cost
- Storage: ~1KB per episode
- 10,000 episodes ≈ 10MB

## Performance Notes
- Sustained mixed-load benchmark (30s, 200 concurrency): ~899 RPS, ~357ms overall P99.
- Ingestion-only load (1000 requests, 100 concurrency): ~1121 RPS, ~132ms P99.
- Vector similarity: cosine distance (L2-normalized embeddings).

## Data Retention

Episodes stored indefinitely unless:
- Manually deleted via API
- Decay policy removes unused episodes (configurable)
- Archive old episodes to cold storage

See `BEST_PRACTICES.md` for optimization tips.

## Verification Checklist

When changing precondition schema, verify all serializer/parser call sites:

```bash
rg "preconditions" src/models src/ingestion src/retrieval
```

Required checks:
- Episode/Reflection serializers accept and emit structured preconditions.
- Retrieval precondition matcher can read normalized dict semantics.
- Any migration/compat logic is covered by tests.
