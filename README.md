# EpisodicMemory

Production-ready episodic memory system for AI coding assistants. Multi-tenant SaaS providing intelligent episode storage and retrieval with enterprise-grade observability and security.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSL-blue.svg)](LICENSE)

---

## Overview

EpisodicMemory stores multi-modal episodes (text, code, images) and retrieves relevant context using semantic search, precondition matching, and intelligent ranking. Built on [KyroDB](https://github.com/KyroDB/KyroDB) for high-performance vector search.

**Target Customers**: AI coding tool companies (Cursor, Replit), enterprise AI teams, developer platforms

**Business Model**: Freemium (100 episodes/month) → Pro ($29/month, 10K episodes) → Enterprise (custom)

---

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with KyroDB connection details

# Run
uvicorn src.main:app --reload --port 8000
```

### Docker Compose

```bash
docker-compose up -d
```

Access:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Metrics: http://localhost:8000/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## Features

**Core**: Multi-modal storage, semantic search (<50ms P99), precondition matching, intelligent ranking, temporal queries

**Production**: Multi-tenancy, API key auth, rate limiting, Prometheus metrics, structured JSON logging, Kubernetes health checks, CI/CD, auto-scaling

**Observability**: 25+ Prometheus metrics, structured logs with request context, Grafana dashboards, distributed tracing

---

## API Usage

```python
import requests

# Capture episode
response = requests.post("http://localhost:8000/api/v1/capture",
    headers={"X-API-Key": "your_api_key"},
    json={
        "episode_type": "failure",
        "goal": "Deploy application",
        "error_trace": "ImagePullBackOff: failed to resolve image",
        "error_class": "resource_error",
        "tags": ["production", "deployment"]
    })

# Search episodes
response = requests.post("http://localhost:8000/api/v1/search",
    headers={"X-API-Key": "your_api_key"},
    json={
        "goal": "Deploy with kubectl",
        "k": 5,
        "min_similarity": 0.6
    })
```

---

## Deployment

### Kubernetes

```bash
# Build
./scripts/build.sh --prod --tag v1.0.0 --push

# Deploy
kubectl apply -k k8s/staging/
kubectl apply -k k8s/production/
```

### CI/CD

Automated via GitHub Actions:
- **Staging**: Auto-deploy on merge to `develop`
- **Production**: Auto-deploy on tag `v*.*.*` (manual approval required)

See [docs/CICD.md](docs/CICD.md) for details.

---

## Development

```bash
# Run tests
pytest tests/ -v --cov=src

# Format
black src/ tests/
ruff check src/ tests/ --fix

# Deploy manually
./scripts/deploy.sh --env staging --dry-run
```

---

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Deployment guide
- [CICD.md](docs/CICD.md) - CI/CD pipeline
- [BILLING.md](docs/BILLING.md) - Stripe billing integration
- [STRUCTURED_LOGGING.md](docs/STRUCTURED_LOGGING.md) - Logging guide

---

## Status

**Phase 1-4 Complete**: Multi-tenancy, auth, security, metrics, logging, health checks, containerization, CI/CD, Stripe billing

**Phase 5-6 Planned**: Performance optimization, launch preparation

---

## License

Business Source License 1.1 - Free for non-production use. See [LICENSE](LICENSE).

---

**Built with**: FastAPI, KyroDB, Prometheus, Grafana, Kubernetes, Docker
