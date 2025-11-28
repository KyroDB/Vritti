# Troubleshooting Guide

## Common Issues

### 401 Unauthorized
- **Cause**: Invalid or missing `X-API-Key`.
- **Fix**: Check your API key in `.env` or headers. Ensure it matches the hashed key in the database.

### 429 Too Many Requests
- **Cause**: Rate limit exceeded (default: 100 req/min).
- **Fix**: Implement exponential backoff in your client. Contact support to increase limits.

### "No candidates found"
- **Cause**:
  - Vector search threshold too high (`min_similarity`).
  - Preconditions filtered out all candidates.
  - Empty database.
- **Fix**:
  - Lower `min_similarity` (vector similarity threshold, range 0.0-1.0, default: 0.8) to 0.6 or 0.7.
  - Check `current_state` matches episode preconditions.
  - Ensure episodes are being ingested correctly.

### Slow Search Response (>200ms)
- **Cause**:
  - Cold start (model loading).
  - Large `k` value (e.g., k=100).
  - Network latency to KyroDB.
- **Fix**:
  - Keep the service warm.
  - Use `k=5` or `k=10` (number of results to return, default: 20, acceptable range: 1-100).
  - Check KyroDB connection.

## Health Checks

Check the system status:

```bash
# Liveness (is the service running?)
curl http://localhost:8000/health/liveness

# Readiness (is the service ready to accept traffic?)
curl http://localhost:8000/health/readiness

# Comprehensive health check
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "components": [
    {"name": "kyrodb", "status": "healthy"},
    {"name": "database", "status": "healthy"}
  ]
}
```

**Component Statuses**:
- `healthy`: Service is working
- `unhealthy`: Service is unreachable
- `degraded`: Service is partially available

**Troubleshooting disconnected components**:
1. **Verify KyroDB is running**:
   ```bash
   lsof -i :50051
   ```
2. **Check service logs**:
   ```bash
   grep ERROR logs/app.log | tail -20
   ```
3. **Confirm network/.env configuration**:
   - Check connection strings in `.env`
   - Verify firewall rules
   - Test network connectivity

## Logs

Logs are structured JSON. Look for `level="ERROR"` or `level="WARNING"`.

```bash
# Tail logs
tail -f logs/app.log | grep "ERROR"
```
