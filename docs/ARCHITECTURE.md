# Vritti Architecture

Complete system architecture documentation for the Vritti platform.

---

## Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Multi-Tenancy](#multi-tenancy)
- [Security Architecture](#security-architecture)
- [Performance Characteristics](#performance-characteristics)

---

## System Overview

Vritti is a multi-tenant episodic memory platform for AI coding assistants. The system is designed for high performance, security, and reliability.

### Design Principles

1. **Failure-focused learning**: Store only failures to maximize learning value
2. **Performance first**: <50ms P99 search latency, <100ms ingestion
3. **Security by default**: Multi-tenancy isolation, API key auth, PII redaction
4. **Observable**: Comprehensive metrics, structured logs, health checks

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Coding Tools (Cursor, etc.)               │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS/API Key Auth
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vritti API (FastAPI)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  Ingestion   │  │  Retrieval   │  │  Observability     │    │
│  │  Pipeline    │  │  Pipeline    │  │  (Metrics/Logs)    │    │
│  └──────┬───────┘  └──────┬───────┘  └────────────────────┘    │
│         │                  │                                    │
│         ▼                  ▼                                    │
│  ┌─────────────────────────────────────┐                       │
│  │    Multi-tenant Customer Database   │                       │
│  │              (SQLite)               │                       │
│  └─────────────────────────────────────┘                       │
└─────────────────┬───────────────────────────────────────────────┘
                  │ gRPC
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                          KyroDB                                 │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │  Text/Code Instance  │      │   Image Instance     │        │
│  │  (384-dim embeddings)│      │ (512-dim CLIP)       │        │
│  │  Port: 50051         │      │ Port: 50052          │        │
│  └──────────────────────┘      └──────────────────────┘        │
│                                                                  │
│  3-tier caching: 73.5% hit rate, <1ms P99 vector search        │
└─────────────────────────────────────────────────────────────────┘
```

### Components

**API Server (FastAPI)**:
- Ingestion pipeline: PII redaction → embedding → KyroDB storage
- Retrieval pipeline: Precondition matching → vector search → ranking
- Authentication: API key-based with bcrypt hashing
- Rate limiting: Per-endpoint limits (50-500 req/min)
- Observability: structured logging and health checks

**Customer Database (SQLite)**:
- Customer accounts and API keys
- Usage tracking and quota enforcement
- Audit logging for compliance

**KyroDB (Vector Database)**:
- Dual-instance: text/code (384-dim) + images (512-dim CLIP)
- 3-tier caching: Learned cache → Hot tier → Cold tier
- <1ms P99 vector search at 10M vectors
- 73.5% cache hit rate (validated)

**Observability**:
- Health checks: `/health/liveness`, `/health/readiness`, `/health`
- Structured JSON logging with request context

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
7. **Reflection**: Background task generates LLM reflection

### Retrieval Flow

```
AI Tool → API → Auth → Precondition Matching → Vector Search → Ranking → Response
```

1. **Authentication**: Verify API key, check quota
2. **Validation**: Pydantic validation of search request
3. **Precondition Matching**: Heuristic filtering
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

**Input Validation**: Pydantic models with strict validation

**Rate Limiting**: Prevent abuse and DoS attacks

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
| KyroDB cache hit rate | >70% | 73.5% |
| API availability | 99.9% | 99.95% |

---

