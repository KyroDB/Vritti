# LLM Configuration

Vritti uses LLMs to analyze failures and generate solutions.

## Setup

### Get an API Key

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Add $5-10 credits
3. Generate API key
4. Add to `.env`:

```bash
LLM_OPENROUTER_API_KEY=sk-or-v1-your-api-key
```

### Optional Settings

```bash
# LLM CONFIGURATION (OpenRouter)
# Unified API for all LLM calls
LLM_OPENROUTER_API_KEY=sk-or-v1-your-api-key
LLM_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_CHEAP_MODEL=x-ai/grok-4.1-fast:free
LLM_CONSENSUS_MODEL_1=kwaipilot/kat-coder-pro:free
LLM_CONSENSUS_MODEL_2=tngtech/deepseek-r1t2-chimera:free

# Generation parameters
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1500

# Budget limits
LLM_DAILY_WARNING_USD=10.0
LLM_DAILY_LIMIT_USD=50.0
```

## How It Works

### Tier System

Vritti uses 3 tiers to balance cost and quality:

| Tier | When Used | Cost | Model |
|------|-----------|------|-------|
| Cached | Cluster template match (experimental) | $0.00 | N/A |
| Cheap | Most failures | ~$0.00 | OpenRouter free tier |
| Premium | Complex failures | ~$0.00 | Multi-perspective (free tier) |

### Automatic Tier Selection

**Cached tier** if:
- Clustering is enabled and a matching cluster template exists
- Otherwise cached tier is disabled and the system falls back to cheap/premium

#### Enable Clustering (Cached Tier)

Clustering/cached-tier is **off by default** because it requires you to persist cluster templates first.

1) Enable clustering in `.env`:

```bash
# Cached tier toggle (experimental)
CLUSTERING_ENABLED=true

# Matching behavior
CLUSTERING_TEMPLATE_MATCH_MIN_SIMILARITY=0.85
CLUSTERING_TEMPLATE_MATCH_K=5
```

2) (Optional) Tune clustering parameters (used by the offline clustering job / template generation):

```bash
CLUSTERING_MIN_CLUSTER_SIZE=5
CLUSTERING_MIN_SAMPLES=3
CLUSTERING_METRIC=cosine
```

#### What is a “matching cluster template”?

A **cluster template** is a saved reflection that Vritti can reuse at $0 cost when a new episode’s
embedding is sufficiently similar to the template’s embedding.

**Storage convention (required):**
- Templates are stored in KyroDB text instance under the namespace: `{customer_id}:cluster_templates`
- Each template must have:
  - `doc_id`: the `cluster_id` (int)
  - `embedding`: a 384-dim text embedding (typically the cluster centroid embedding)
  - `metadata.template_reflection_json`: JSON string of a `Reflection` payload

When clustering is enabled, Vritti searches this namespace using the incoming episode embedding and
selects cached tier when the best match score ≥ `CLUSTERING_TEMPLATE_MATCH_MIN_SIMILARITY`.

#### Where do templates come from?

Templates can be created in two ways:
1) **Generated** by an offline clustering + template-generation workflow (recommended for production).
2) **Seeded** manually for known recurring failures (useful in early pilots).

If you don’t have templates yet, cached tier will never trigger (it will fall back to cheap/premium).

#### Troubleshooting

If cached tier is not triggering:
1) Confirm `CLUSTERING_ENABLED=true`.
2) Confirm templates exist in KyroDB under `{customer_id}:cluster_templates`.
3) Confirm template embeddings are 384-dim and normalized.
4) Lower `CLUSTERING_TEMPLATE_MATCH_MIN_SIMILARITY` temporarily (e.g., `0.80`) to validate wiring.
5) Check logs for “Selected CACHED tier” / “No cluster match”.

**Premium tier** if:
- `error_class` is one of critical classes (`data_loss`, `security_breach`, `production_outage`, `corruption`)
- `environment_info.retry_count > 0` (agent is retrying after prior failure)
- `tags` includes `premium_reflection`

**Cheap tier**:
- Everything else (default)

### LLM Validation Settings

```bash
# In .env (search LLM validation thresholds)
SEARCH_LLM_SIMILARITY_THRESHOLD=0.92
```

Higher threshold = fewer LLM validations but better precision.

## Check Budget

```bash
curl http://localhost:8000/admin/budget \
  -H "X-Admin-API-Key: your-admin-key"
```

Response:

```json
{
  "daily_cost_usd": 12.50,
  "warning_threshold_usd": 10.0,
  "limit_threshold_usd": 50.0,
  "premium_tier_blocked": false,
  "cost_by_tier": {
    "cheap": 2.50,
    "premium": 10.00
  }
}
```

## Cost Optimization

1. **Use cheap tier by default** - Most failures use cheap tier
2. **Limit premium tier usage** - Keep premium for complex failures
3. **Set budget limits** - Prevents runaway costs
4. **Monitor daily spend** - Check budget endpoint

Typical costs:
- 1000 reflections/day = ~$1-2/day
- 10,000 reflections/day = ~$10-20/day

## Fallback Behavior

### No API Key Set

Startup behavior is controlled by `SERVICE_REQUIRE_LLM_REFLECTION`:
- `true`: startup fails fast if LLM configuration is missing or placeholder.
- `false`: service starts with reflection generation disabled.

Note: `.env.production.example` ships with `SERVICE_REQUIRE_LLM_REFLECTION=false` for safe bootstrap.
Set it to `true` before production rollout once valid LLM credentials are configured.

### Rate Limits

OpenRouter limits:

| Account | Rate Limit |
|---------|------------|
| Free | 10 req/min |
| Pay-as-you-go | 60 req/min |
| Enterprise | Custom |

System retries with exponential backoff (max 3 attempts).

### Budget Exceeded

When daily limit reached:
- Cheap tier continues
- Premium tier blocked
- Warning logged


## Troubleshooting

### "Reflection generation disabled"

Check:
1. `LLM_OPENROUTER_API_KEY` is set in `.env`
2. API key is valid
3. Account has credits

### "Premium tier blocked"

Daily budget exceeded. Options:
1. Increase `LLM_DAILY_LIMIT_USD`
2. Wait for next day (resets at midnight UTC)

### "Rate limit exceeded"

Options:
1. Upgrade OpenRouter account
2. Reduce request rate
3. Enable more aggressive caching

### Slow reflections

Check:
1. OpenRouter status page
2. Switch to faster free model (Grok Fast)
3. Check network latency

## Monitoring

Check logs for LLM issues:

```bash
grep -E "(reflection|llm|openrouter)" logs/app.log
grep "budget_warning\|limit_exceeded" logs/app.log
```

Key signals live in logs (reflection success/failure, tier selection, and cost warnings).

## Security

1. Rotate API keys monthly
2. Use separate keys for dev/prod
3. Set `ADMIN_API_KEY` to protect budget endpoints
4. Monitor for cost anomalies
