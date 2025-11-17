# Episodic Memory for AI Agents

Application-layer episodic memory system built on top of [KyroDB](https://github.com/KyroDB/KyroDB) - a high-performance vector database optimized for RAG workloads and AI agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Episodic Memory Service (Python FastAPI)                    │
│ - Multi-modal ingestion (text/code/images)                 │
│ - Precondition-aware retrieval                              │
│ - Automated hygiene (decay & promotion)                     │
└─────────────────────────────────────────────────────────────┘
                        ↓ (gRPC)
┌─────────────────────────────────────────────────────────────┐
│ KyroDB (2 instances)                                        │
│ - Text/code embeddings (384-dim)                           │
│ - Image embeddings (512-dim via CLIP)                      │
│ - 3-tier caching (71.7% hit rate)                          │
│ - <1ms P99 vector search                                    │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-modal storage**: Text, code, and image embeddings
- **Namespace-based collections**: failures, skills, semantic_rules
- **Precondition matching**: LLM-powered relevance filtering
- **Temporal queries**: Filter by timestamp ranges
- **Automated hygiene**: Time-based decay and usage-based pruning
- **Pattern promotion**: Episodic → Semantic memory via clustering

## Performance Targets

- **<50ms P99** retrieval latency
- **10K-100K** episodes per collection
- **Multi-modal search** across text + images

## Quick Start

### Prerequisites

- Python 3.11+
- KyroDB running (2 instances: text + images)
- Redis (for Celery background jobs)

### Installation

```bash
# Clone repository
git clone https://github.com/KyroDB/EpisodicMemory.git
cd EpisodicMemory

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with KyroDB connection details and OpenAI API key
```

### Running Locally

```bash
# Terminal 1: Start KyroDB (text embeddings)
cd ../ProjectKyro/engine
./target/release/kyrodb_server --port 50051 --data-dir ./data/kyrodb_text

# Terminal 2: Start KyroDB (image embeddings)
./target/release/kyrodb_server --port 50052 --data-dir ./data/kyrodb_images

# Terminal 3: Start Episodic Memory service
cd ../../EpisodicMemory
./run_dev.sh
# Or manually:
# source venv/bin/activate
# python -m uvicorn src.main:app --reload --port 8000
```

Access the API:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Statistics**: http://localhost:8000/stats

### API Usage

```python
import requests

# Capture an episode (failure)
response = requests.post("http://localhost:8000/api/v1/capture", json={
    "episode_type": "failure",
    "goal": "Deploy web application to production",
    "tool_chain": ["kubectl", "docker"],
    "actions_taken": [
        "Built Docker image",
        "Pushed to registry",
        "Applied Kubernetes manifest"
    ],
    "error_trace": "ImagePullBackOff: failed to resolve image 'myapp:latest'",
    "error_class": "resource_error",
    "code_state_diff": "# git diff output",
    "screenshot_path": "./screenshots/deploy_error.png",
    "environment_info": {
        "os": "Darwin",
        "kubectl_version": "1.28.0",
        "cluster": "production"
    },
    "tags": ["production", "deployment", "critical"],
    "severity": 1
})
print(f"Episode captured: {response.json()['episode_id']}")

# Search for relevant failures
response = requests.post("http://localhost:8000/api/v1/search", json={
    "goal": "Deploy application with kubectl getting ImagePullBackOff",
    "current_state": {
        "tool": "kubectl",
        "error_class": "ImagePullBackOff",
        "environment": {"os": "Darwin", "kubectl_version": "1.28"},
        "components": ["kubernetes", "docker"],
        "goal_keywords": ["deploy", "production"]
    },
    "collection": "failures",
    "k": 5,
    "min_similarity": 0.6,
    "precondition_threshold": 0.5,
    "ranking_weights": {
        "similarity_weight": 0.5,
        "precondition_weight": 0.3,
        "recency_weight": 0.1,
        "usage_weight": 0.1
    }
})

results = response.json()
print(f"Found {len(results['results'])} relevant episodes")
print(f"Search latency: {results['search_latency_ms']:.2f}ms")

for result in results['results']:
    print(f"Rank {result['rank']}: Episode {result['episode']['episode_id']}")
    print(f"  Combined score: {result['scores']['combined']:.3f}")
    print(f"  Explanation: {result['similarity_explanation']}")
```

## Project Status

- **Phase 0-1**: Core infrastructure ✅ **COMPLETED**
  - Configuration management with Pydantic Settings
  - Pydantic models for episodes and search
  - KyroDB dual-instance client with retry logic
  - Multi-modal embedding service (text + CLIP)
  - PII redaction utilities
  - Snowflake ID generation
  - LLM reflection service with GPT-4

- **Phase 1**: Ingestion & Retrieval Pipelines ✅ **COMPLETED**  - Episode ingestion with async reflection
  - Precondition matching engine (heuristic-based)
  - Weighted ranking system (similarity + precondition + recency + usage)
  - Search orchestrator with latency breakdown
  - FastAPI application with lifespan management

- **Phase 2**: Testing & Validation 🔄 **IN PROGRESS**
  - Integration tests
  - Load testing
  - Performance validation (<50ms P99 target)

- **Phase 3 (Future)**: Background Hygiene
  - Time-based decay and pruning
  - Usage-based pattern promotion
  - Episodic → Semantic memory clustering

## Development

```bash
# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Format code
black src/ tests/
ruff check src/ tests/

# Type checking (optional)
mypy src/ --ignore-missing-imports
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

BSL License - see [LICENSE](LICENSE) for details.

## Related Projects

- [KyroDB](https://github.com/KyroDB/KyroDB) - High-performance vector database
