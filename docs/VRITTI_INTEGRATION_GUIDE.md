# Vritti Integration Guide

Quick guide to integrate Vritti with AI agents.

## Setup

### Prerequisites
- **Two KyroDB instances running**:
  - Text instance: `port 50051` (384-dim text embeddings)
  - Image instance: `port 50052` (512-dim CLIP embeddings)
- **Vritti service**: `port 8000`

### Start Services
```bash
# Terminal 1: Text instance
./target/release/kyrodb_server --config kyrodb_text.yaml

# Terminal 2: Image instance
./target/release/kyrodb_server --config kyrodb_image.yaml

# Terminal 3: Vritti
cd Vritti && uvicorn src.main:app --port 8000
```

### Get API Key
Register your agent and get an API key starting with `em_live_`

## Integration Pattern

```python
from examples.coding_agent_integration import VrittiAgent

agent = VrittiAgent(api_key="em_live_your_key")

# Before risky action
result = await agent.check_action(
    action="rm -rf /important/data",
    goal="Clean up files",
    tool="shell",
    context={"has_backup": False}
)

if result['safe']:
    os.system("rm -rf /important/data")  # Execute
else:
    print(f"BLOCKED: {result['reason']}")  # Don't execute
```

## When to Use Gating

**Call gating for**:
- File operations: `rm`, `mv`, `write`
- Database: `DROP`, `DELETE`, `UPDATE`
- Deployment: `kubectl apply`, `docker run`
- Network: API calls, downloads
- Git: `push`, `merge`, `rebase`

**Skip gating for**:
- Read operations: `ls`, `cat`, `SELECT`
- Safe commands: `echo`, `pwd`, `cd`

## After Failures

Always capture failures so Vritti learns:
```python
await agent.learn_from_failure(
    goal="Deploy application",
    actions=["kubectl apply -f deployment.yaml"],
    error="ImagePullBackOff: image not found",
    resolution="Changed image tag to v1.2.3"
)
```

## Complete Example

```python
class MyAgent:
    def __init__(self):
        self.vritti = VrittiAgent(api_key="em_live_key")
    
    async def execute(self, command, goal):
        # 1. Check if risky
        if self.is_risky(command):
            # 2. Ask Vritti
            check = await self.vritti.check_action(
                action=command,
                goal=goal,
                tool="shell"
            )
            
            # 3. Handle response
            if not check['safe']:
                print(f"❌ BLOCKED: {check['reason']}")
                if check['alternative']:
                    command = check['alternative']  # Use suggested fix
                else:
                    return  # Don't execute
        
        # 4. Execute
        try:
            result = os.system(command)
            if result != 0:
                # 5. Report failure
                await self.vritti.learn_from_failure(
                    goal=goal,
                    actions=[command],
                    error=f"Exit code: {result}",
                    resolution=None
                )
        except Exception as e:
            # 6. Report exceptions too
            await self.vritti.learn_from_failure(
                goal=goal,
                actions=[command],
                error=str(e),
                resolution=None
            )
```

## Configuration

### KyroDB Instances

KyroDB config schema matches `config.example.yaml` in the KyroDB repo.

**Text instance** (`kyrodb_text.yaml`):
```yaml
server:
  host: "127.0.0.1"
  port: 50051

hnsw:
  dimension: 384
  distance: cosine

persistence:
  # IMPORTANT: use a dedicated data_dir per instance
  data_dir: "./kyrodb_text_data"
```

**Image instance** (`kyrodb_image.yaml`):
```yaml
server:
  host: "127.0.0.1"
  port: 50052

hnsw:
  dimension: 512
  distance: cosine

persistence:
  data_dir: "./kyrodb_image_data"
```

### Vritti `.env`
```bash
# KyroDB connections
KYRODB_TEXT_HOST=localhost
KYRODB_TEXT_PORT=50051
KYRODB_IMAGE_HOST=localhost
KYRODB_IMAGE_PORT=50052

# LLM for reflections
LLM_OPENROUTER_API_KEY=sk-or-v1-your-key

# Models (free tier)
LLM_CHEAP_MODEL=x-ai/grok-4.1-fast:free
LLM_CONSENSUS_MODEL_1=kwaipilot/kat-coder-pro:free
LLM_CONSENSUS_MODEL_2=tngtech/deepseek-r1t2-chimera:free
```

## API Endpoints

See [`API_GUIDE.md`](./API_GUIDE.md) for full API reference.

**Core endpoints**:
- `POST /api/v1/capture` - Report failures
- `POST /api/v1/reflect` - Check before action
- `POST /api/v1/search` - Find similar failures
- `GET /health` - System status

## Examples

Working integration examples in `examples/`:
- `coding_agent_integration.py` - Generic wrapper (✅ use this)
- `aider_integration.py` - Aider-specific
- `test_no_auth.py` - System validation

## Troubleshooting

**"Cannot connect to Vritti"**
```bash
# Check Vritti is running
curl http://localhost:8000/health
```

**"Invalid API key"**
- API keys must start with `em_live_`
- Register your key in the system first

**"KyroDB unhealthy"**
```bash
# Check both instances are running
curl http://localhost:51051/health  # Text
curl http://localhost:51052/health  # Image
```

**Too slow**
- Enable client-side caching (5-min TTL)
- Reduce `k` parameter in search (default: 20)
- Only gate risky actions

## Support

Email: kishan@kyrodb.com
