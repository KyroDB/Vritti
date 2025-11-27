# Vriti

**Production-grade episodic memory for AI coding assistants**

Store and retrieve multi-modal failure episodes with semantic search, precondition matching, and intelligent ranking.

---

## What It Does

Captures failed actions (code errors, deployment failures) and retrieves relevant past solutions when similar situations occur.

**Example**: AI assistant fails to deploy → searches past deployment failures → finds matching error → applies proven solution

---

## Core Features

- **Multi-modal storage**: Text, code, images, error traces
- **Semantic search**: Vector similarity + precondition matching (\u003c50ms P99)
- **Smart ranking**: Recency, similarity, context compatibility
- **Multi-tenant**: API key authentication, namespace isolation
- **Production-ready**: Prometheus metrics, structured logging, Kubernetes

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Set KyroDB connection in .env

# Run
uvicorn src.main:app --port 8000
```

**API**: `http://localhost:8000/docs`

---

## Usage

### Capture Failure

```python
POST /api/v1/capture
Headers: {"X-API-Key": "your_key"}
Body: {
  "episode_type": "failure",
  "goal": "Deploy to production",
  "error_trace": "ImagePullBackOff: registry.io/app:v1.2.3",
  "error_class": "resource_error",
  "tool_chain": ["kubectl", "apply"]
}
```

### Search Similar Episodes

```python
POST /api/v1/search
Headers: {"X-API-Key": "your_key"}
Body: {
  "goal": "Deploy with kubectl",
  "current_state": {"cluster": "prod"},
  "k": 5,
  "min_similarity": 0.6
}
```

**Response**: Top-k relevant episodes with confidence scores

---

## Architecture

```
┌─────────────┐
│   FastAPI   │ ← API Layer (rate limiting, auth)
└──────┬──────┘
       │
┌──────▼──────┐
│  Retrieval  │ ← Search pipeline (semantic + precondition)
└──────┬──────┘
       │
┌──────▼──────┐
│   KyroDB    │ ← Vector database (384-dim embeddings)
└─────────────┘
```

**Storage**: KyroDB (fast vector search)  
**Embeddings**: Sentence-Transformers (384-dim)  
**Observability**: Prometheus + Grafana

---

## Deployment

### Docker Compose

```bash
docker-compose up -d
```

Includes: API, KyroDB, Prometheus, Grafana

### Kubernetes

```bash
kubectl apply -k k8s/production/
```

Auto-scaling, health checks, metrics included.

---

## Development

```bash
# Test
pytest tests/ -v --cov=src

# Format
black src/ tests/
ruff check src/ --fix

# Type check
mypy src/
```

---

## Performance

- **Search latency**: \u003c50ms P99 (10K episodes)
- **Throughput**: 1000 req/sec (single instance)
- **Storage**: ~500KB per episode (with reflection)

---

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)**: System design, data flow
- **[Deployment](docs/DEPLOYMENT.md)**: Production setup
- **[API Docs](http://localhost:8000/docs)**: Interactive OpenAPI

---

## License

Business Source License 1.1 - Free for non-production use  
Commercial use requires license - Contact: [kishan@kyrodb.com]

---

**Stack**: Python 3.11+ • FastAPI • KyroDB • Docker • Kubernetes
