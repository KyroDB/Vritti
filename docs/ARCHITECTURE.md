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
2. **Performance first**: optimize for sustained high-concurrency operation
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
│  KyroDB ANN + cache layers (performance depends on workload)    │
└─────────────────────────────────────────────────────────────────┘
```

### Components

**API Server (FastAPI)**:
- Ingestion pipeline: PII redaction → embedding → KyroDB storage
- Retrieval pipeline: Precondition matching → vector search → ranking
- Authentication: API key-based with adaptive scrypt hash verification + TTL cache
- Rate limiting: Per-endpoint limits (50-500 req/min)
- Observability: structured logging and health checks

**Customer Database (SQLite)**:
- Customer accounts and API keys
- Usage tracking and quota enforcement
- Audit logging for compliance

**KyroDB (Vector Database)**:
- Dual-instance: text/code (384-dim) + images (512-dim CLIP)
- 3-tier caching: Learned cache → Hot tier → Cold tier
- Raw vector index latency can reach sub-millisecond P99 in isolated KyroDB microbenchmarks
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
Document ID: tenant-safe identifier strategy (UUIDv4 or tenant-scoped sequence)
```

**Recommended identifier strategy**:
- Use UUIDv4 (or equivalent cryptographically-random IDs) for externally visible identifiers.
- If dense integer `doc_id` is required by the vector engine, derive it from a tenant-scoped allocator
  (e.g., `{customer_id}:{local_sequence}` mapping), not a single global shared counter.
- Collection namespace isolation alone is not sufficient for multi-tenancy; IDs must avoid sequential
  leakage and single-writer bottlenecks across tenants.

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

**API Key Format**: `em_live_{key_id}_{secret}` (random key id + random secret)

**Storage**:
- Store only adaptive password hashes (scrypt) with per-key random salt.
- Never store plaintext API keys.
- Keep a 5-minute TTL cache for validated keys to minimize repeated DB verification.

**Validation**:
1. Extract key from `X-API-Key` header
2. Check cache (5-minute TTL)
3. If miss: lookup by key_id and verify scrypt hash (constant-time compare)
4. Load customer and check status (active/suspended)

**Key-hash migration policy**:
- Legacy SHA-256-only key storage is not accepted in this clean-slate release.
- On upgrade, rotate/re-issue API keys and recreate the `api_keys` table with adaptive hashes.
- This is an explicit force-rotation migration to remove weak credential storage.

### Data Protection

**PII Redaction**: Automatic redaction in logs
- Emails: `***@example.com`
- API keys: `em_live_abc***xyz`
- Passwords: `***REDACTED***`

**Input Validation**: Pydantic models with strict validation

**Rate Limiting**: Prevent abuse and DoS attacks

---

## Performance Characteristics

### Performance Targets

The system tracks two different latency classes:

- **KyroDB raw vector latency target (aspirational)**: `<1ms P99` for isolated ANN query execution.
- **End-to-end API latency target (aspirational)**: `<400ms P99` under sustained mixed load, including:
  auth, request parsing, embedding/ranking, serialization, and network overhead.

Because end-to-end requests include significantly more work than raw ANN search, API P99 will be
higher than isolated DB query P99.

### Benchmark Snapshot

Recent KyroDB-backed load runs (`30s`, `200` concurrent workers) produced:
- Overall throughput: ~`899 RPS`
- Overall P99 latency: ~`357ms`
- Search/Gating/Ingestion P99: ~`351-362ms`

Reproduce end-to-end measurements with:
- `tests/load/test_load.py`
- `tests/load/test_load_1000_rps.py`

---
