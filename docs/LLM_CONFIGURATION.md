# LLM Configuration

Vritti uses LLMs to analyze failures and generate solutions.

## Setup

### Get an API Key

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Add $5-10 credits
3. Generate API key
4. Add to `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-api-key
```

### Optional Settings

```bash
# LLM CONFIGURATION (OpenRouter - Free Tier)
# OpenRouter unified API for all LLM calls
LLM_OPENROUTER_API_KEY=your-openrouter-api-key
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
| Cached | Identical failure seen before | $0.00 | N/A |
| Cheap | Most failures | ~$0.00 | OpenRouter free tier |
| Premium | Complex failures | ~$0.00 | Multi-perspective (free tier) |

### Automatic Tier Selection

**Cached tier** if:
- Similar failure exists (>92% match)
- Uses cached reflection (free)

**Premium tier** if:
- Authentication errors
- Database connection failures
- Complexity score >0.7

**Cheap tier**:
- Everything else (default)

### Cache Settings

```bash
# In .env
SEARCH_LLM_SIMILARITY_THRESHOLD=0.92
```

Cache hit when episode similarity >0.92. Higher threshold = fewer cache hits but better accuracy.

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

1. **Enable caching** - Saves 30-40% on costs
2. **Use cheap tier** - 80% of failures use cheap tier
3. **Set budget limits** - Prevents runaway costs
4. **Monitor daily spend** - Check budget endpoint

Typical costs:
- 1000 reflections/day = ~$1-2/day
- 10,000 reflections/day = ~$10-20/day

## Fallback Behavior

### No API Key Set

Episodes stored but no reflection generated:
- Storage works normally
- Search returns episodes
- Gating uses precondition matching only

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
1. `OPENROUTER_API_KEY` is set in `.env`
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
