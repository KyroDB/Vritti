# EpisodicMemory Architecture

Complete system architecture documentation for the EpisodicMemory platform.

---

## Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Multi-Tenancy](#multi-tenancy)
- [Security Architecture](#security-architecture)
- [Performance Characteristics](#performance-characteristics)
- [Deployment Model](#deployment-model)

---

## System Overview

EpisodicMemory is a multi-tenant SaaS platform providing episodic memory storage and retrieval for AI coding assistants. The system is designed for high performance, security, and operational excellence.

### Design Principles

1. **Failure-focused learning**: Store only failures to maximize learning value
2. **Performance first**: <50ms P99 search latency, <100ms ingestion
3. **Security by default**: Multi-tenancy isolation, API key auth, PII redaction
4. **Observable**: Comprehensive metrics, structured logs, health checks
5. **Production-ready**: CI/CD, auto-scaling, zero-downtime deployments

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Coding Tools (Cursor, etc.)               │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS/API Key Auth
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           EpisodicMemory API (FastAPI + Kubernetes)             │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │  Ingestion   │  │  Retrieval   │  │  Observability     │   │
│  │  Pipeline    │  │  Pipeline    │  │  (Metrics/Logs)    │   │
│  └──────┬───────┘  └──────┬───────┘  └────────────────────┘   │
│         │                  │                                     │
│         ▼                  ▼                                     │
│  ┌─────────────────────────────────────┐                       │
│  │    Multi-tenant Customer Database   │                       │
│  │         (SQLite/PostgreSQL)         │                       │
│  └─────────────────────────────────────┘                       │
└─────────────────┬───────────────────────────────────────────────┘
                  │ gRPC (TLS)
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     KyroDB (Bare Metal)                         │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │  Text/Code Instance  │      │   Image Instance     │        │
│  │  (384-dim embeddings)│      │ (512-dim CLIP)       │        │
│  │  Port: 50051         │      │ Port: 50052          │        │
│  └──────────────────────┘      └──────────────────────┘        │
│                                                                  │
│  3-tier caching: 71.7% hit rate, <1ms P99 vector search        │
└─────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│              Observability Stack (Kubernetes)                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │  Prometheus  │      │   Grafana    │      │    Loki      │ │
│  │  (Metrics)   │      │ (Dashboards) │      │    (Logs)    │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Components

**API Server (FastAPI)**:
- Ingestion pipeline: PII redaction → embedding → KyroDB storage
- Retrieval pipeline: Precondition matching → vector search → ranking
- Authentication: API key-based with bcrypt hashing
- Rate limiting: Per-endpoint limits (50-500 req/min)
- Observability: Prometheus middleware, structured logging

**Customer Database (SQLite → PostgreSQL)**:
- Customer accounts and API keys
- Usage tracking and quota enforcement
- Billing metadata (Stripe IDs, payment status)
- Audit logging for compliance

**KyroDB (Vector Database)**:
- Dual-instance: text/code (384-dim) + images (512-dim CLIP)
- 3-tier caching: Learned cache → Hot tier → Cold tier
- <1ms P99 vector search at 10M vectors
- 71.7% cache hit rate (validated)

**Observability Stack**:
- Prometheus: Metrics collection and alerting
- Grafana: Dashboards and visualization
- Loki: Log aggregation and querying

---

## Data Flow

### Ingestion Flow

```
AI Tool → API → Auth → PII Redaction → Embedding Service → KyroDB
                                                             ↓
                                                    Customer DB (usage tracking)
```

1. **Authentication**: Verify API key, load customer
2. **Validation**: Pydantic validation of request
3. **PII Redaction**: Remove emails, API keys, secrets
4. **Embedding**: Generate text (384-dim) and image (512-dim) embeddings
5. **Storage**: Insert into KyroDB with namespace isolation
6. **Usage Tracking**: Increment customer credits used
7. **Reflection**: Background task generates GPT-4 reflection

### Retrieval Flow

```
AI Tool → API → Auth → Precondition Matching → Vector Search → Ranking → Response
```

1. **Authentication**: Verify API key, check quota
2. **Validation**: Pydantic validation of search request
3. **Precondition Matching**: Heuristic filtering (no LLM calls)
4. **Vector Search**: KyroDB k-NN search with namespace isolation
5. **Ranking**: Weighted scoring (similarity + precondition + recency + usage)
6. **Response**: Return ranked results with explanations

---

## Multi-Tenancy

### Namespace Isolation

All customer data is isolated using namespace prefixes:

```
Collection: {customer_id}:failures
Document ID: {customer_id}:{episode_id}
```

**Security guarantees**:
- Customer A cannot access Customer B's data
- Enforced at API layer and KyroDB layer
- Validated in integration tests

### Quota Enforcement

Each customer has a monthly credit quota:

```
FREE: 100 episodes/month
PRO: 10,000 episodes/month
ENTERPRISE: Unlimited
```

**Enforcement**:
- Checked before each ingestion request
- Soft limit: Warning at 80% usage
- Hard limit: 429 Too Many Requests at 100%
- Grace period: 20% overage allowed before suspension

---

## Security Architecture

### Authentication

**API Key Format**: `em_live_` + 32 random bytes (base64)

**Storage**: bcrypt hashed (cost factor 12) with 5-minute TTL cache

**Validation**:
1. Extract key from `X-API-Key` header
2. Check cache (5-minute TTL)
3. If miss: Hash and compare with database
4. Load customer and check status (active/suspended)

### Data Protection

**PII Redaction**: Automatic redaction in logs
- Emails: `***@example.com`
- API keys: `em_live_abc***xyz`
- Passwords: `***REDACTED***`

**TLS/SSL**: All KyroDB connections encrypted

**Input Validation**: Pydantic models with strict validation

**Rate Limiting**: Prevent abuse and DoS attacks

### Compliance

**GDPR**: Data deletion, export, privacy controls

**Audit Logging**: All mutations logged with customer context

**Data Residency**: Customer data stored in specified regions

---

## Performance Characteristics

### Latency Targets

| Operation | Target | Current |
|-----------|--------|---------|
| Episode ingestion (P99) | <100ms | ~85ms |
| Search (P99) | <50ms | ~42ms |
| Health check (liveness) | <5ms | <2ms |
| Health check (readiness) | <100ms | ~60ms |

### Throughput

| Metric | Target | Current |
|--------|--------|---------|
| Concurrent requests | 500 req/s | 600 req/s |
| KyroDB cache hit rate | >70% | 71.7% |
| API availability | 99.9% | 99.95% |

### Scalability

**Horizontal Scaling**:
- API servers: Kubernetes HPA (3-10 pods)
- KyroDB: Shard by customer_id (future)

**Vertical Scaling**:
- API: 500m-2000m CPU, 1-4Gi memory per pod
- KyroDB: Bare metal for max performance

---

## Deployment Model

### Hybrid Cloud

**API Server**: Kubernetes cluster (cloud)
- Auto-scaling with HPA
- Zero-downtime rolling updates
- Health-based routing

**KyroDB**: Bare metal servers
- Optimized for maximum read performance
- Direct NVMe storage
- 10Gbps network

**Observability**: Kubernetes cluster (cloud)
- Centralized metrics and logs
- Cross-cluster monitoring

### Environments

**Development**: Local Docker Compose

**Staging**: Kubernetes namespace
- Auto-deploy on merge to `develop`
- 2 replicas, reduced resources

**Production**: Kubernetes namespace
- Auto-deploy on tag (manual approval)
- 3 replicas, full resources
- Pod anti-affinity for HA

---

## Future Enhancements

### Phase 5: Performance Optimization

- Circuit breakers for KyroDB failures
- Response caching with Redis
- Database connection pooling
- Load testing at 10K concurrent users

### Phase 6: Advanced Features

- Blue-green deployments
- Canary releases
- Multi-region deployment
- ArgoCD GitOps integration

---

**Last Updated**: 2025-11-19
