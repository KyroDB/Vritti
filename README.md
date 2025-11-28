# Vritti

**Stop AI agents from repeating the same mistakes.**

Vritti captures failures, analyzes patterns, and prevents AI coding assistants from making the same errors twice.

---

## The Problem

AI coding assistants make the same mistakes repeatedly:
- Deploy with wrong image tags
- Forget to run migrations
- Use incorrect API endpoints
- Skip prerequisite steps

Each mistake wastes developer time debugging and fixing the same issue.

## The Solution

Vritti learns from failures and blocks similar actions before they fail:

1. **Capture**: AI agent fails (wrong command, missing dependency, etc.)
2. **Analyze**: LLM generates root cause and solution
3. **Prevent**: Next time the agent tries something similar, Vritti says "Don't do that, do this instead"

---

## Quick Start

### Prerequisites

- Python 3.11+
- KyroDB server running (ports 50051 for text, 50052 for images)

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.production.example .env
```

Edit `.env` and set:
- `KYRODB_TEXT_HOST` - Your KyroDB instance for vector storage
- `OPENROUTER_API_KEY` - LLM API key for reflection generation

### Run

```bash
uvicorn src.main:app --port 8000
```

API documentation: `http://localhost:8000/docs`

---

## Usage

### 1. Capture a Failure

When your AI agent makes a mistake, send it to Vritti:

```bash
curl -X POST http://localhost:8000/api/v1/capture \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "episode_type": "failure",
    "goal": "Deploy application to production",
    "actions_taken": ["kubectl apply -f deployment.yaml"],
    "error_trace": "ImagePullBackOff: image not found in registry",
    "error_class": "resource_error",
    "tool_chain": ["kubectl"],
    "resolution": "Fixed image tag to match pushed version"
  }'
```

Vritti stores this failure and generates analysis.

### 2. Check Before Acting

Before your AI agent does something risky, ask Vritti:

```bash
curl -X POST http://localhost:8000/api/v1/reflect \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "proposed_action": "kubectl apply -f deployment.yaml",
    "goal": "Deploy to production",
    "tool": "kubectl",
    "context": "Deploying version 1.2.3",
    "current_state": {"cluster": "production"}
  }'
```

Response:

```json
{
  "recommendation": "block",
  "confidence": 0.92,
  "rationale": "Similar action failed: ImagePullBackOff error. Verify image exists in registry first.",
  "suggested_action": "Verify image exists in registry before applying"
}
```

Recommendations:
- **block**: Don't do this, it will fail
- **rewrite**: Do this differently (suggestion provided)
- **hint**: Be careful, check these things first
- **proceed**: No known issues, go ahead

### 3. Search Past Failures

Find similar problems you've solved before:

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "kubectl deployment image pull error",
    "k": 5,
    "min_similarity": 0.7
  }'
```

---

## How It Works

### Storage
Episodes stored in KyroDB (vector database) with 384-dimensional embeddings.

### Matching
Combines semantic similarity with precondition matching:
- "Using kubectl" + "production cluster" = high match
- Generic errors = low match

### LLM Analysis
Uses OpenRouter API (Grok, Deepseek, or other models) to generate:
- Root cause analysis
- Specific resolution steps
- Preconditions that must match for this solution to apply

### Caching
LLM responses cached for 5 minutes to reduce cost and latency.

---

## Development

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_gating.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format
black src/ tests/

# Lint
ruff check src/ --fix

# Type check
mypy src/
```

---

## Performance Targets

- Search latency: <50ms P99
- Gating decision: <100ms P99
- LLM cache hit rate: >80%

Current performance validated with 280+ tests.

---

## Configuration

Key settings in `.env`:

```bash
# Required
KYRODB_TEXT_HOST=localhost
KYRODB_TEXT_PORT=50051
OPENROUTER_API_KEY=sk-or-v1-...

# Optional
LOG_LEVEL=INFO
SEARCH_LLM_VALIDATION_ENABLED=true
SEARCH_LLM_SIMILARITY_THRESHOLD=0.92
```

Full configuration options in `src/config.py`.

---

## API Authentication

Generate API keys:

```bash
python scripts/generate_api_key.py
```

Include in requests:

```bash
curl -H "X-API-Key: em_live_..." http://localhost:8000/api/v1/search
```

---

## Monitoring

Prometheus metrics available at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

Key metrics:
- `episodic_memory_gating_decision_total` - Total gating decisions
- `episodic_memory_repeat_error_prevented_total` - Errors prevented
- `episodic_memory_gating_latency_seconds` - Decision latency

---

## Health Checks

```bash
# Liveness (is the service alive?)
curl http://localhost:8000/health/liveness

# Readiness (is the service ready to accept traffic?)
curl http://localhost:8000/health/readiness

# Comprehensive health check
curl http://localhost:8000/health
```

---

## Production Deployment

1. Set environment variables in `.env`
2. Start KyroDB instances (text on port 50051, image on port 50052)
3. Start with `uvicorn src.main:app --host 0.0.0.0 --port 8000`
4. Configure reverse proxy (nginx/caddy) for HTTPS
5. Set up Prometheus scraping for `/metrics`

---

## Documentation

- `docs/ARCHITECTURE.md` - System design
- `docs/API_GUIDE.md` - API integration guide
- `docs/KYRODB_SETUP.md` - KyroDB configuration
- `docs/LLM_CONFIGURATION.md` - LLM tier configuration
- `docs/VRITTI_INTEGRATION_GUIDE.md` - Integration examples

---

## License

Business Source License 1.1 - Free for non-production use.
Commercial use requires license: kishan@kyrodb.com

---

**Built with:** Python 3.9+ | FastAPI | KyroDB | OpenRouter
