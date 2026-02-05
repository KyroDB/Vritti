# Changelog

All notable changes to Vritti will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2025-11-29

### Initial Release

First production-ready release of Vritti - episodic memory system for AI agents.

### Added

**Core Features**
- Episodic memory capture for AI agent failures
- Multi-perspective LLM reflection generation (consensus-based)
- Semantic search with precondition matching
- AI agent gating system (prevent repeated mistakes)
- Skills promotion from high-quality episodes
- Tiered reflection system (cached/cheap/premium)
- Multi-modal support (text + image embeddings)
- Customer isolation and multi-tenancy

**Integrations**
- KyroDB vector database (dual-instance: text + image)
- OpenRouter LLM API (free tier models)
- Observability improvements
- Structured JSON logging

**API Endpoints**
- `POST /api/v1/capture` - Capture failure episodes
- `POST /api/v1/reflect` - Pre-action reflection and gating (proceed/block/rewrite/hint)
- `POST /api/v1/search` - Semantic search
- `GET /health` - Health checks


**Documentation**
- Complete API guide
- Integration guide with setup instructions
- Best practices for AI agent integration
- Storage architecture documentation
- Observability and monitoring guide
- Troubleshooting runbooks
- Example integrations (generic + Aider-specific)

**Examples**
- Generic coding agent integration wrapper
- Aider-specific integration
- System validation test (no auth required)

**Infrastructure**
- FastAPI-based REST API
- SQLite customer database
- Dual KyroDB instances (text + image vectors)
- Automatic PII redaction
- Request/response validation
- Circuit breakers for resilience
- Health checks and readiness probes

### Technical Details

**Dependencies**
- Python 3.11+
- FastAPI, SQLAlchemy, Pydantic
- OpenRouter API (free tier)
- KyroDB v0.1.0
- Sentence Transformers (all-MiniLM-L6-v2)

**Performance Targets**
- Search latency: <50ms P99
- Reflection generation: asynchronous background task (cheap/cached/premium tiers)
- Episode capture: <200ms P99
- Health checks: <5ms

**Deployment Model**
- Local-first deployment 
- Two KyroDB instances required (text: 50051, image: 50052)
- API server on port 8000

### Security

- API key authentication (em_live_ prefix)
- Automatic PII redaction
- Customer namespace isolation
- API key auth via key_id lookup + adaptive scrypt hash verification + TTL cache
- Request validation and sanitization


### Migration Notes

**From Previous Versions**
- This is the first release - no migration needed

**API Keys**
- All API keys must start with `em_live_` prefix
- Register keys via customer API before use

**Configuration**
- Requires dual KyroDB setup (text + image instances)
- OpenRouter API key needed for reflections
- See docs/KYRODB_SETUP.md for configuration
