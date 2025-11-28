# Vritti Incident Response Runbooks

This document contains step-by-step procedures for handling common incidents in the Vritti Episodic Memory Service.

---

## Table of Contents

1. [High Error Rate](#1-high-error-rate)
2. [High Latency (P99 > 50ms)](#2-high-latency-p99--50ms)
3. [KyroDB Connection Failure](#3-kyrodb-connection-failure)
4. [LLM Budget Exceeded](#4-llm-budget-exceeded)
5. [Reflection Generation Failures](#5-reflection-generation-failures)
6. [Dead Letter Queue Growing](#6-dead-letter-queue-growing)
7. [Customer Quota Exceeded](#7-customer-quota-exceeded)
8. [Memory/CPU Exhaustion](#8-memorycpu-exhaustion)
9. [API Key Compromise](#9-api-key-compromise)
10. [Data Recovery Procedures](#10-data-recovery-procedures)

---

## 1. High Error Rate

### Symptoms
- `episodic_memory_errors_total` counter increasing rapidly
- Error rate > 1% of total requests
- HTTP 5xx responses in logs

### Severity
- **P1 (Critical)**: Error rate > 10%
- **P2 (High)**: Error rate 5-10%
- **P3 (Medium)**: Error rate 1-5%

### Diagnosis Steps

```bash
# 1. Check current error rate
curl -s http://localhost:8000/metrics | grep episodic_memory_errors_total

# 2. Check error breakdown by type
curl -s http://localhost:8000/metrics | grep "error_type"

# 3. Check recent logs for errors
grep -i error logs/app.log | tail -50

# 4. Check service health
curl -s http://localhost:8000/health | jq
```

### Resolution Steps

1. **Identify Error Type**:
   - `validation`: Check request payload schemas
   - `authentication`: Check API key configuration
   - `kyrodb`: See [KyroDB Connection Failure](#3-kyrodb-connection-failure)
   - `llm`: See [Reflection Generation Failures](#5-reflection-generation-failures)

2. **If Validation Errors**:
   ```bash
   # Check for schema changes in recent commits
   git log --oneline -10 src/models/
   ```

3. **If Authentication Errors**:
   ```bash
   # Check API key cache health
   curl -s http://localhost:8000/metrics | grep api_key_cache
   
   # Clear cache by restarting service
   ./scripts/restart_api.sh
   # Or manually: pkill -TERM -f "uvicorn.*app.main:app" && sleep 5 && uvicorn app.main:app ...
   ```

4. **Escalation**: If error rate persists > 15 minutes, page on-call engineer.

---

## 2. High Latency (P99 > 50ms)

### Symptoms
- P99 latency exceeds 50ms SLO
- `episodic_memory_http_request_duration_seconds` histogram showing high values
- Slow query warnings in logs

### Severity
- **P2 (High)**: P99 > 100ms for > 5 minutes
- **P3 (Medium)**: P99 50-100ms for > 10 minutes

### Diagnosis Steps

```bash
# 1. Check current latency percentiles
curl -s http://localhost:8000/metrics | grep http_request_duration

# 2. Check endpoint-specific latency
curl -s http://localhost:8000/metrics | grep -E "endpoint.*capture|search"

# 3. Check KyroDB latency
curl -s http://localhost:8000/metrics | grep kyrodb_operation_duration

# 4. Check embedding generation latency
curl -s http://localhost:8000/metrics | grep embedding_generation_duration

# 5. Check active connections
curl -s http://localhost:8000/metrics | grep http_requests_active
```

### Resolution Steps

1. **If KyroDB is slow**:
   ```bash
   # Check KyroDB instance health
   curl -s http://localhost:8000/health | jq '.components[] | select(.name=="kyrodb")'
   
   # Check KyroDB metrics
   grpcurl -plaintext localhost:50051 kyrodb.KyroDB/Health
   ```

2. **If Embedding is slow**:
   ```bash
   # Check if GPU is available (should be using CPU)
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Check embedding model warmup
   curl -s http://localhost:8000/stats | jq '.ingestion_stats'
   ```

3. **If Traffic Spike**:
   ```bash
   # Check rate limiting
   curl -s http://localhost:8000/metrics | grep rate_limit_exceeded
   
   # Consider running multiple workers
   uvicorn src.main:app --workers 4
   ```

4. **Temporary Mitigation**:
   - Reduce `SEARCH_DEFAULT_K` to lower search result count
   - Disable reflection generation temporarily: `LLM_API_KEY=`

---

## 3. KyroDB Connection Failure

### Symptoms
- Health check shows `kyrodb: unhealthy`
- `episodic_memory_kyrodb_connection_healthy` gauge = 0
- All capture/search requests failing with 503

### Severity
- **P1 (Critical)**: Complete service outage

### Diagnosis Steps

```bash
# 1. Check KyroDB health via router
curl -s http://localhost:8000/health | jq '.components[] | select(.name=="kyrodb")'

# 2. Check KyroDB process is running
lsof -i :50051

# 3. Test gRPC connectivity
grpcurl -plaintext localhost:50051 list
```

### Resolution Steps

1. **If KyroDB is Down**:
   ```bash
   # Restart KyroDB server
   ./kyrodb_server --config kyrodb_config.toml
   
   # Wait for health
   sleep 10 && curl http://localhost:8000/health/readiness
   ```

2. **If Connection Timeout**:
   ```bash
   # Check network connectivity
   nc -zv localhost 50051
   
   # Check firewall rules
   sudo ufw status | grep 50051
   ```

3. **Recovery Validation**:
   ```bash
   # Insert test episode
   curl -X POST http://localhost:8000/api/v1/capture \
     -H "X-API-Key: test-key" \
     -H "Content-Type: application/json" \
     -d '{"goal":"test","error_trace":"test","episode_type":"failure",...}'
   
   # Verify search works 
   curl -X POST http://localhost:8000/api/v1/search \
     -H "X-API-Key: test-key" \
     -H "Content-Type: application/json" \
     -d '{"goal":"test"}'
   ```

---

## 4. LLM Budget Exceeded

### Symptoms
- `episodic_memory_daily_cost_usd` > $50
- Premium tier blocked (`/admin/budget` shows `premium_tier_blocked: true`)
- Reflections failing or falling back to cheap tier

### Severity
- **P3 (Medium)**: Service degraded but functional

### Diagnosis Steps

```bash
# 1. Check current budget status
curl -s http://localhost:8000/admin/budget | jq

# 2. Check cost breakdown by tier
curl -s http://localhost:8000/admin/reflection/stats | jq

# 3. Check for cost spike source
curl -s http://localhost:8000/metrics | grep reflection_cost_by_tier
```

### Resolution Steps

1. **Immediate Mitigation** (wait for daily reset at midnight UTC):
   ```bash
   # Reflections will automatically use cheap tier only
   # No action needed - circuit breaker is working as designed
   ```

2. **If Budget Increase Needed**:
   ```bash
   # Update config in .env and restart
   export REFLECTION_MAX_COST_PER_DAY_USD=100
   export REFLECTION_MAX_COST_PER_DAY_USD=100
   # Restart the uvicorn process
   ./scripts/restart_api.sh
   ```

3. **Investigate High Spend**:
   ```bash
   # Check tier distribution
   curl -s http://localhost:8000/admin/reflection/stats | jq '.percentage_by_tier'
   
   # If premium > 15%, check error class distribution
   curl -s http://localhost:8000/metrics | grep skill_promotions_total
   ```

4. **Long-term Fix**:
   - Review `REFLECTION_PREMIUM_ERROR_CLASSES` configuration
   - Tune `REFLECTION_MIN_CHEAP_CONFIDENCE` to reduce fallbacks

---

## 5. Reflection Generation Failures

### Symptoms
- `episodic_memory_reflection_failure_total` counter increasing
- Reflections not appearing on episodes
- LLM provider errors in logs

### Severity
- **P3 (Medium)**: Reflections degraded, core service functional

### Diagnosis Steps

```bash
# 1. Check failure breakdown by reason
curl -s http://localhost:8000/metrics | grep reflection_failure_total

# 2. Check LLM provider success rates
curl -s http://localhost:8000/metrics | grep llm_call_total

# 3. Check OpenRouter status
curl -s https://status.openrouter.ai/api/v2/status.json | jq

# 4. Check logs for LLM errors
grep -i "openrouter\|llm\|reflection" logs/app.log | tail -50
```

### Resolution Steps

1. **If All Models Failing**:
   ```bash
   # Check API key validity
   curl https://openrouter.ai/api/v1/models \
     -H "Authorization: Bearer $OPENROUTER_API_KEY" | head -20
   
   # If key invalid, rotate key
   export OPENROUTER_API_KEY=sk-or-v1-new-key-here
   # Restart the service
   export OPENROUTER_API_KEY=sk-or-v1-new-key-here
   # Restart the service
   ./scripts/restart_api.sh
   ```

2. **If Single Model Failing**:
   ```bash
   # Check which model
   curl -s http://localhost:8000/metrics | grep 'llm_call_total{model_name'
   
   # Update to different model in config
   export LLM_CONSENSUS_MODEL_1=anthropic/claude-3-haiku
   export LLM_CONSENSUS_MODEL_1=anthropic/claude-3-haiku
   # Restart the uvicorn process
   ./scripts/restart_api.sh
   ```

3. **If Timeout Errors**:
   ```bash
   # Increase timeout in .env
   export LLM_TIMEOUT_SECONDS=90
   export LLM_TIMEOUT_SECONDS=90
   # Restart the uvicorn process
   ./scripts/restart_api.sh
   ```

---

## 6. Dead Letter Queue Growing

### Symptoms
- `episodic_memory_reflection_dead_letter_total` > 0
- Failed reflections accumulating in `data/failed_reflections.log`

### Severity
- **P4 (Low)**: Background task, no immediate user impact

### Diagnosis Steps

```bash
# 1. Check DLQ count
curl -s http://localhost:8000/metrics | grep dead_letter

# 2. Check DLQ file
wc -l data/failed_reflections.log
tail -5 data/failed_reflections.log | jq

# 3. Check failure reasons
jq -r '.failure_reason' data/failed_reflections.log | sort | uniq -c
```

### Resolution Steps

1. **Manual Recovery**:
   ```bash
   # Process DLQ entries
   python scripts/process_dead_letter_queue.py
   
   # Or retry via API (if implemented)
   curl -X POST http://localhost:8000/admin/dlq/retry
   ```

2. **If Persistence Failing Consistently**:
   - Check KyroDB health (see [KyroDB Connection Failure](#3-kyrodb-connection-failure))
   - Check if namespace exists
   - Verify customer_id is valid

3. **Archive Old Entries**:
   ```bash
   # Archive and clear
   mv data/failed_reflections.log data/failed_reflections.$(date +%Y%m%d).log
   touch data/failed_reflections.log
   ```

---

## 7. Customer Quota Exceeded

### Symptoms
- Customer receiving 403 responses
- `episodic_memory_customer_quota_usage_ratio` > 1.0
- Rate limit warnings in logs

### Severity
- **P4 (Low)**: Single customer affected

### Diagnosis Steps

```bash
# 1. Check customer quota
curl -s http://localhost:8000/metrics | grep customer_quota_usage_ratio

# 2. Check customer's current usage
sqlite3 data/customers.db "SELECT credits_used_current_month, monthly_credit_limit FROM customers WHERE customer_id='$CUSTOMER_ID'"
```

### Resolution Steps

1. **If Legitimate Usage**:
   ```bash
   # Upgrade customer tier
   curl -X PATCH http://localhost:8000/api/v1/customers/$CUSTOMER_ID \
     -H "X-Admin-API-Key: $ADMIN_KEY" \
     -d '{"subscription_tier": "pro"}'
   ```

2. **If Abuse**:
   ```bash
   # Suspend customer
   curl -X PATCH http://localhost:8000/api/v1/customers/$CUSTOMER_ID \
     -H "X-Admin-API-Key: $ADMIN_KEY" \
     -d '{"status": "suspended"}'
   ```

---

## 8. Memory/CPU Exhaustion

### Symptoms
- OOM kills in container logs
- CPU > 90% sustained
- Service unresponsive

### Severity
- **P1 (Critical)**: Service outage

### Diagnosis Steps

```bash
# 1. Check resource usage
# macOS:
top -l 1 | grep uvicorn
# Linux:
# top -bn1 | grep uvicorn

# 2. Check for memory leaks
curl -s http://localhost:8000/metrics | grep process_

# 3. Check embedding model memory
python -c "import psutil; print(psutil.Process().memory_info().rss / 1024**3, 'GB')"
```

### Resolution Steps

1. **Immediate**: Restart service to clear memory
   ```bash
   # Kill and restart the uvicorn process
   # Kill and restart the uvicorn process
   ./scripts/restart_api.sh
   ```

2. **If Recurring**:
   - Profile for memory leaks
   - Consider running with fewer workers

3. **Embedding Model Memory**:
   ```bash
   # Use smaller model if memory constrained
   export EMBEDDING_TEXT_MODEL_NAME=paraphrase-MiniLM-L3-v2
   ```

---

## 9. API Key Compromise

### Symptoms
- Unusual traffic patterns from unknown IPs
- Requests from unexpected customer_ids
- Spike in credit usage

### Severity
- **P1 (Critical)**: Security incident

### Immediate Actions

1. **Identify Compromised Key**:
   ```bash
   # Check logs for suspicious activity (DO NOT LOG API KEYS IN PRODUCTION)
   grep "customer_id" logs/app.log | sort | uniq -c | sort -rn | head -20
   ```

2. **Revoke Key Immediately**:
   ```bash
   # Deactivate customer
   curl -X PATCH http://localhost:8000/api/v1/customers/$CUSTOMER_ID \
     -H "X-Admin-API-Key: $ADMIN_KEY" \
     -d '{"status": "suspended"}'
   
   # Generate new key
   curl -X POST http://localhost:8000/api/v1/customers/$CUSTOMER_ID/rotate-key \
     -H "X-Admin-API-Key: $ADMIN_KEY"
   ```

3. **Block IP if Identified**:
   ```bash
   # Add to firewall (nginx/cloudflare)
   iptables -A INPUT -s $MALICIOUS_IP -j DROP
   ```

4. **Notify Customer** with new API key

5. **Post-Incident**:
   - Rotate admin API key
   - Audit access logs
   - Review security controls

---

## 10. Data Recovery Procedures

### Episode Recovery from KyroDB

```bash
# 1. List episodes for customer
grpcurl -plaintext -d '{"namespace":"customer-id:failures","k":100}' \
  localhost:50051 kyrodb.KyroDB/List

# 2. Export episodes to JSON
python scripts/export_episodes.py --customer-id=$CUSTOMER_ID --output=backup.json
```

### Reflection Recovery from DLQ

```bash
# 1. Parse DLQ entries
jq -c 'select(.customer_id=="$CUSTOMER_ID")' data/failed_reflections.log > customer_dlq.json

# 2. Retry failed reflections
python scripts/retry_reflections.py --input=customer_dlq.json
```

### Customer Database Recovery

```bash
# 1. Backup current state
cp data/customers.db data/customers.db.bak

# 2. Restore from backup (ensure path is correct)
# Example: cp /mnt/backups/vritti/customers.db.$(date -d yesterday +%Y%m%d) data/customers.db
cp ${BACKUP_DIR}/customers.db.$(date -d yesterday +%Y%m%d) data/customers.db

# 3. Verify integrity
sqlite3 data/customers.db "PRAGMA integrity_check"
```

---

## Escalation Matrix

| Severity | Response Time | Escalation Path |
|----------|---------------|-----------------|
| P1 (Critical) | 15 minutes | On-call → Team Lead → VP Engineering |
| P2 (High) | 1 hour | On-call → Team Lead |
| P3 (Medium) | 4 hours | On-call via ticket |
| P4 (Low) | 24 hours | Normal ticket queue |

## Contact Information

- **On-Call Rotation**: See PagerDuty schedule
- **Slack Channel**: #vritti-incidents
- **Status Page**: status.vritti.example.com

---

