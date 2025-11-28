# Structured Logging

Production-grade structured logging with JSON output for log aggregation .

## Overview

EpisodicMemory uses **structlog** for structured logging with:

- **JSON output** for log aggregation (ELK, Loki, CloudWatch)
- **Request context** propagation (customer_id, request_id, trace_id)
- **Correlation IDs** for distributed tracing
- **PII redaction** for GDPR/SOC2 compliance
- **Performance-optimized** (<5μs overhead per log call)

## Architecture

```
FastAPI Request
      │
      ▼
StructuredLoggingMiddleware ──> Sets request context (request_id, customer_id, trace_id)
      │
      ▼
Application Code ──────────────> Uses get_logger(__name__).info("...", key=value)
      │
      ▼
structlog Processors ──────────> Add timestamp, service metadata, redact PII
      │
      ▼
JSON Output ───────────────────> {"timestamp": "...", "level": "info", "event": "...", "request_id": "..."}
      │
      ▼
Log Aggregation (ELK/Loki)
```

## Configuration

### Environment Variables

```bash
# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOGGING_LEVEL=INFO

# Output format (true for JSON, false for console)
LOGGING_JSON_OUTPUT=true

# Colorize console output (only for development)
LOGGING_COLORIZED=false

# Slow request thresholds
LOGGING_SLOW_REQUEST_WARNING_MS=100.0
LOGGING_SLOW_REQUEST_ERROR_MS=500.0

# Service metadata (for log aggregation)
LOGGING_SERVICE_NAME=episodic-memory
LOGGING_SERVICE_VERSION=0.1.0
LOGGING_ENVIRONMENT=production
```

### Configuration in Code

```python
from src.config import get_settings
from src.observability.logging import configure_logging

settings = get_settings()
configure_logging(
    log_level=settings.logging.level,
    json_output=settings.logging.json_output,
    colorized=settings.logging.colorized,
)
```

## Usage

### Basic Logging

```python
from src.observability.logging import get_logger

logger = get_logger(__name__)

# Simple log
logger.info("Episode captured successfully")

# Log with structured fields
logger.info(
    "Episode captured",
    episode_id=12345,
    latency_ms=45.2,
    credits_used=1.7,
)

# Warning
logger.warning("Slow query detected", latency_ms=150.5)

# Error with exception
try:
    result = risky_operation()
except Exception as e:
    logger.error("Operation failed", error=str(e), exc_info=True)
```

**JSON Output**:
```json
{
  "timestamp": "2025-01-15T10:30:45.123456Z",
  "level": "info",
  "event": "Episode captured",
  "episode_id": 12345,
  "latency_ms": 45.2,
  "credits_used": 1.7,
  "service": "episodic-memory",
  "version": "0.1.0",
  "environment": "production",
  "request_id": "req_abc123",
  "customer_id": "acme-corp",
  "trace_id": "trace_xyz789"
}
```

### Request Context

Request context is automatically set by `StructuredLoggingMiddleware` for all HTTP requests.

**Manual context** (for background tasks):

```python
from src.observability.logging import RequestContext, get_logger

logger = get_logger(__name__)

with RequestContext(customer_id="acme-corp"):
    logger.info("Processing background task")
    # request_id and trace_id are auto-generated
    # All logs in this context include customer_id
```

**Extract context values**:

```python
from src.observability.logging import (
    get_request_id,
    get_customer_id,
    get_trace_id,
)

# Get current context values
request_id = get_request_id()  # "req_abc123"
customer_id = get_customer_id()  # "acme-corp"
trace_id = get_trace_id()  # "trace_xyz789"
```

### Operation Context (Timing)

Automatically logs operation start, completion, and latency:

```python
from src.observability.logging import OperationContext

with OperationContext("kyrodb_insert", episode_id=12345, collection="failures"):
    await kyrodb.insert(...)
# Logs:
# - "kyrodb_insert started" (debug level)
# - "kyrodb_insert completed" with latency_ms (info level)
```

**On error**:
```python
with OperationContext("kyrodb_insert", episode_id=12345):
    raise ValueError("Connection timeout")
# Logs: "kyrodb_insert failed" with latency_ms and exception (error level)
```

### Bound Loggers (Persistent Context)

Add persistent context to all logs from a logger:

```python
logger = get_logger(__name__)

# Bind context that applies to all logs
batch_logger = logger.bind(operation="batch_ingestion", batch_size=100)

batch_logger.info("Batch started")  # Includes operation and batch_size
batch_logger.info("Processing record", record_id=5)  # Includes all fields
batch_logger.info("Batch complete", records_processed=50)
```

## Features

### 1. Request Context Propagation

Every HTTP request automatically gets:
- **request_id**: Unique ID for request tracking
- **customer_id**: From authenticated API key
- **trace_id**: For distributed tracing

These are injected into **all logs** within the request context.

**Headers**:
- `X-Request-ID`: Client-provided or auto-generated
- `X-Trace-ID`: For distributed tracing (OpenTelemetry compatible)

**Response headers**:
- `X-Request-ID`: Returned to client for correlation
- `X-Trace-ID`: Returned to client for correlation

### 2. PII Redaction

Sensitive fields are automatically redacted:

```python
logger.info(
    "User authenticated",
    email="user@example.com",  # Redacted to: ***@example.com
    api_key="em_live_abc123xyz",  # Redacted to: em_live_abc***xyz
    password="secret123",  # Redacted to: ***REDACTED***
    customer_id="acme-corp",  # NOT PII, shown fully
)
```

**Redacted fields**:
- `api_key`: Shows first 12 chars + last 3 chars
- `password`: Fully redacted
- `authorization`: Fully redacted
- `credit_card`: Shows last 4 digits only
- `email`: Redacted to domain only
- `secret`: Fully redacted
- `token`: Shows prefix only

### 3. Distributed Tracing

Trace IDs propagate across service boundaries:

```python
# Upstream service includes X-Trace-ID header
# Middleware extracts and uses it for all logs

with RequestContext(trace_id="trace_from_upstream"):
    logger.info("Processing request")  # trace_id: trace_from_upstream

    # Call downstream service (pass trace_id in headers)
    response = await call_downstream_service(headers={"X-Trace-ID": trace_id})
```

**Query across services**:
```
# All logs from the same distributed request
trace_id:"trace_from_upstream"
```

### 4. Slow Request Logging

Automatic warnings for slow requests:

```python
# Configured via environment:
# LOGGING_SLOW_REQUEST_WARNING_MS=100.0
# LOGGING_SLOW_REQUEST_ERROR_MS=500.0

# Request takes 150ms
# WARNING: "Slow request detected" with latency_ms=150.5

# Request takes 600ms
# ERROR: "Slow request detected" with latency_ms=600.2
```

### 5. Service Metadata

All logs include service metadata for aggregation:

```json
{
  "service": "episodic-memory",
  "version": "0.1.0",
  "environment": "production"
}
```

**Query by service**:
```
# ELK
service:"episodic-memory" AND environment:"production"

# Loki
{service="episodic-memory", environment="production"}
```

## Log Levels

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: Confirmation that things are working as expected
- **WARNING**: Indication that something unexpected happened (e.g., slow query)
- **ERROR**: Due to a more serious problem, the software has not been able to perform some function
- **CRITICAL**: A serious error, indicating the program itself may be unable to continue running

**Example**:

```python
logger.debug("Cache hit", key="episode_12345")  # Development only
logger.info("Episode captured", episode_id=12345)  # Normal operations
logger.warning("Slow query", latency_ms=150.5)  # Performance degradation
logger.error("Database connection failed", exc_info=True)  # Operation failure
logger.critical("Service unavailable", reason="out of memory")  # Service down
```

## Output Formats

### JSON Output (Production)

```json
{
  "timestamp": "2025-01-15T10:30:45.123456Z",
  "level": "info",
  "event": "Episode captured successfully",
  "service": "episodic-memory",
  "version": "0.1.0",
  "environment": "production",
  "request_id": "req_abc123",
  "customer_id": "acme-corp",
  "trace_id": "trace_xyz789",
  "episode_id": 12345,
  "latency_ms": 45.2,
  "credits_used": 1.7
}
```

### Console Output (Development)

```
2025-01-15 10:30:45 [info] Episode captured successfully
    request_id=req_abc123 customer_id=acme-corp trace_id=trace_xyz789
    episode_id=12345 latency_ms=45.2 credits_used=1.7
```

## Log Aggregation

### ELK Stack (Elasticsearch + Logstash + Kibana)

**Logstash configuration**:
```ruby
input {
  file {
    path => "/var/log/episodic-memory/*.log"
    codec => json
  }
}

filter {
  # Logs are already JSON, no parsing needed
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "episodic-memory-%{+YYYY.MM.dd}"
  }
}
```

**Kibana queries**:
```
# All logs for a specific customer
customer_id:"acme-corp"

# Errors in the last hour
level:"error" AND @timestamp:[now-1h TO now]

# Slow requests
latency_ms:>100

# Trace a distributed request
trace_id:"trace_xyz789"

# All logs for a specific request
request_id:"req_abc123"
```

### Grafana Loki

**Promtail configuration**:
```yaml
server:
  http_listen_port: 9080

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: episodic-memory
    static_configs:
      - targets:
          - localhost
        labels:
          job: episodic-memory
          __path__: /var/log/episodic-memory/*.log
    pipeline_stages:
      - json:
          expressions:
            level: level
            customer_id: customer_id
            request_id: request_id
```

**LogQL queries**:
```
# All logs for a service
{service="episodic-memory"}

# Errors only
{service="episodic-memory"} |= "level=error"

# Specific customer
{service="episodic-memory", customer_id="acme-corp"}

# Slow requests
{service="episodic-memory"} | json | latency_ms > 100

# Request trace
{service="episodic-memory"} | json | request_id="req_abc123"
```

### AWS CloudWatch Logs

**Configuration**:
```python
# Install watchtower: pip install watchtower
import watchtower
import logging

# Add CloudWatch handler
cloudwatch_handler = watchtower.CloudWatchLogHandler(
    log_group="/aws/episodic-memory/production",
    stream_name="api-server",
)
logging.getLogger().addHandler(cloudwatch_handler)
```

**CloudWatch Insights queries**:
```
# All errors
fields @timestamp, level, event, error, request_id
| filter level = "error"
| sort @timestamp desc
| limit 100

# Slow requests
fields @timestamp, latency_ms, method, path, customer_id
| filter latency_ms > 100
| sort latency_ms desc
| limit 20

# Customer activity
fields @timestamp, event, episode_id, credits_used
| filter customer_id = "acme-corp"
| stats count() by bin(5m)
```

## Performance

### Overhead

- **Per log call**: <5μs (JSON output), ~15μs (console with colors)
- **Context propagation**: <1μs (context variables are thread-local)
- **PII redaction**: <2μs (only if sensitive fields present)

### Optimization

**Lazy evaluation** (no overhead if log level disabled):
```python
# DEBUG logs are skipped entirely if level=INFO
logger.debug("Expensive calculation", result=expensive_function())
# expensive_function() is NOT called if DEBUG disabled
```

**Batch logging** (for high-throughput scenarios):
```python
# Batch operations use a single log per batch
batch_logger = logger.bind(batch_id="batch_123", batch_size=1000)
batch_logger.info("Batch started")
# ... process 1000 records ...
batch_logger.info("Batch complete", records_processed=1000)
```

## Security

### PII Compliance

- **GDPR**: PII automatically redacted from logs
- **SOC2**: Audit trail without sensitive data
- **HIPAA**: No PHI in logs (email redacted)

### Credential Protection

- **API keys**: Prefix shown only (em_live_***abc)
- **Passwords**: Fully redacted
- **Tokens**: Prefix shown only

### Audit Logging

All logs include:
- **customer_id**: WHO performed the action
- **request_id**: WHICH request
- **timestamp**: WHEN it happened
- **event**: WHAT happened

## Troubleshooting

### Logs not appearing

**Check log level**:
```bash
# Ensure log level allows INFO logs
LOGGING_LEVEL=INFO
```

**Check JSON parsing**:
```bash
# Validate JSON output
tail -f /var/log/episodic-memory/app.log | jq .
```

### Missing request context

**Verify middleware order**:
```python
# StructuredLoggingMiddleware must be OUTERMOST
app.add_middleware(StructuredLoggingMiddleware)  # First
app.add_middleware(PrometheusMiddleware)  # After
```

### PII not redacted

**Check field names**:
```python
# Field names must match redaction list
logger.info("User login", api_key="...")  # Redacted
logger.info("User login", apikey="...")  # NOT redacted (wrong name)
```

## Examples

See `examples/structured_logging_demo.py` for complete working examples.

## Next Steps

- Grafana dashboards and log aggregation integration
- Production log retention and archival policies

## References

- [structlog Documentation](https://www.structlog.org/)
- [Logging Best Practices](https://12factor.net/logs)
- [OpenTelemetry Logging](https://opentelemetry.io/docs/reference/specification/logs/)
