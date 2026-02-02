# Vritti v0.1.0 Release Notes


## 🎉 First Production Release!

Vritti v0.1.0 is here - stop your AI agents from making the same mistakes twice!

## What is Vritti?

Vritti is an episodic memory system that helps AI coding agents:
- **Learn from failures** - Capture every mistake with full context
- **Get intelligent solutions** - LLM-powered root cause analysis
- **Prevent repeats** - Block similar actions before they fail again

## Quick Start

### 1. Setup KyroDB (Two Instances Required)

```bash
# Text instance (port 50051)
./target/release/kyrodb_server --config kyrodb_config.toml --data-dir ./data/kyrodb

# Image instance (port 50052)  
./target/release/kyrodb_server --config kyrodb_config_images.toml
```

### 2. Configure Vritti

```bash
cd Vritti
cp .env.production.example .env
# Edit .env and set LLM_OPENROUTER_API_KEY
```

### 3. Start Vritti

```bash
uvicorn src.main:app --port 8000
```

### 4. Integrate with Your Agent

```python
from examples.coding_agent_integration import VrittiAgent

agent = VrittiAgent(api_key="em_live_your_key")

# Before risky action
if await agent.check_action("rm -rf /", "Clean files"):
    os.system("rm -rf /")  # Only if safe!
```

## Key Features

### 🧠 Episodic Memory
- Capture failures with full context (actions, errors, environment)
- Multi-modal support (text + screenshots)
- Customer-isolated storage

### 🤖 AI-Powered Analysis
- **Multi-perspective reflection**: Consensus from multiple LLMs
- **Root cause analysis**: Why it failed
- **Solution generation**: How to fix it
- **Prevention guidance**: How to avoid it

### 🛡️ Intelligent Gating
- Check actions before execution
- Recommendations: proceed, block, rewrite, hint
- Precondition matching for context-aware decisions
- <50ms latency (P99)

### 📊 Production-Ready

- Structured JSON logging
- Health checks and readiness probes
- Circuit breakers for resilience
- 51+ automated tests

### 💰 Zero Cost (Free Tier)
- OpenRouter free tier models
- Self-hosted KyroDB (no vector DB costs)
- **$0/month** for LLM + storage

## What's New in v0.1.0

### Core Capabilities
✅ Failure capture with multi-modal embeddings  
✅ Semantic search (cosine similarity + preconditions)  
✅ Tiered reflection system (cached/cheap/premium)  
✅ Skills promotion from high-quality episodes  
✅ AI agent gating (prevent repeated mistakes)  
✅ Multi-tenancy with customer isolation  

### Developer Experience
✅ Comprehensive documentation (70% more concise)  
✅ Integration examples for coding agents  
✅ Quick validation test (no auth needed)  
✅ Clear setup instructions  
✅ Troubleshooting runbooks  

### Technical Improvements
✅ 200+ outdated LLM references cleaned up  
✅ Removed Docker complexity (local-first)  
✅ Standardized on Python 3.9  
✅ All tests passing (51+)  
✅ Cost calculations corrected ($0 free tier)  

## Examples Included

### 1. Generic Integration
`examples/coding_agent_integration.py` - Works with any agent

### 2. Aider Integration
`examples/aider_integration.py` - Aider-specific wrapper

### 3. System Validation
`examples/test_no_auth.py` - Quick health check

## Documentation

- **[README](./README.md)** - Overview and quick start
- **[API Guide](./docs/API_GUIDE.md)** - API reference
- **[Integration Guide](./docs/VRITTI_INTEGRATION_GUIDE.md)** - Full setup
- **[Best Practices](./docs/BEST_PRACTICES.md)** - Tips and patterns
- **[Troubleshooting](./docs/TROUBLESHOOTING.md)** - Common issues

## Performance

- **Search**: <50ms P99
- **Gating**: <100ms P99
- **Capture**: <200ms P99
- **Uptime**: 99.9% target

## Requirements

- Python 3.9+
- KyroDB v0.1.0 (two instances)
- OpenRouter API key (free tier)
- 2GB RAM minimum
- Ports: 8000 (API), 50051 (text DB), 50052 (image DB)


## Upgrade Notes

This is the first release - no migration needed!

## What's Next (v0.2 Roadmap)

- Docker/Kubernetes deployment
- PostgreSQL support
- Full decay policy implementation
- Batch episode processing
- Enhanced skill promotion
- Dashboard UI

## Get Help

- **Email**: kishan@kyrodb.com
- **Documentation**: See `docs/` folder
- **Issues**: Check `docs/TROUBLESHOOTING.md`

## License

See LICENSE file for details.

---

**Ready to stop repeating mistakes?** Start with `examples/test_no_auth.py` to validate your setup!
