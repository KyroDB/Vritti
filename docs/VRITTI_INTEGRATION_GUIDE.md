# Vritti Integration Guide for AI Agents

**Version**: v0.1
**Mission**: Enable AI agents that learn from mistakes and don't repeat them

---

## Table of Contents

1. [Overview](#overview)
2. [Integration Architecture](#integration-architecture)
3. [API Integration](#api-integration)
4. [Gating Flow](#gating-flow)
5. [Client-Side Caching](#client-side-caching)
6. [Mission Metrics](#mission-metrics)
7. [Example Implementations](#example-implementations)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Vritti provides two core capabilities for AI agents:

1. **Learning (Ingestion)**: Capture failures and generate reflections
2. **Prevention (Gating)**: Check if an action will repeat a past mistake

```
┌──────────────┐
│  AI Agent    │
└──────┬───────┘
       │
       ├─────► Ingestion API (when failure occurs)
       │         POST /api/v1/capture
       │         → Stores failure + generates reflection
       │
       └─────► Gating API (before risky actions)
                 POST /api/v1/reflect
                 → Returns BLOCK/REWRITE/HINT/PROCEED
```

---

## Integration Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Agent Execution Loop                                            │
│                                                                 │
│  1. Plan Action                                                │
│       ↓                                                         │
│  2. Check Vritti (if action is risky)                         │
│       ├─ BLOCK     → Don't execute, use suggested fix         │
│       ├─ REWRITE   → Use suggested alternative                │
│       ├─ HINT      → Show warning, let agent decide           │
│       └─ PROCEED   → Execute as planned                       │
│       ↓                                                         │
│  3. Execute Action                                             │
│       ├─ SUCCESS   → Continue                                  │
│       └─ FAILURE   → Report to Vritti (Ingestion)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### When to Call Gating

**IMPORTANT**: Do NOT call gating for every action. This adds overhead.

**Call gating for**:
- File system operations (`rm`, `mv`, `write`)
- Network operations (`curl`, `wget`, API calls)
- Database operations (`DROP`, `UPDATE`, `DELETE`)
- Deployment commands (`kubectl apply`, `docker run`)
- Git operations (`git push`, `git merge`)
- Package installations (`pip install`, `npm install`)

**Skip gating for**:
- Read-only operations (`ls`, `cat`, `SELECT`)
- Idempotent operations
- Low-risk commands (`echo`, `pwd`)

---

## API Integration

### 1. Ingestion API (Capture Failures)

**Endpoint**: `POST /api/v1/capture`

**When to call**: After any failure the agent wants to learn from

**Request**:
```json
{
  "episode_type": "failure",
  "goal": "Deploy app to Kubernetes",
  "actions_taken": [
    "kubectl apply -f deployment.yaml",
    "kubectl get pods"
  ],
  "error_class": "resource_error",
  "error_trace": "ImagePullBackOff: image myapp:latest not found",
  "tool_chain": ["kubectl"],
  "environment_info": {
    "os": "darwin",
    "cluster": "production",
    "kubectl_version": "1.28.0"
  },
  "resolution": "Changed image tag to v1.2.3",
  "tags": ["kubernetes", "deployment"],
  "severity": 1
}
```

**Response**:
```json
{
  "episode_id": 12345,
  "status": "captured",
  "reflection_queued": true
}
```

**Python Example**:
```python
import requests

def report_failure_to_vritti(goal, actions, error, resolution):
    """Report failure to Vritti for learning."""
    response = requests.post(
        "https://vritti.api.company.com/api/v1/capture",
        headers={
            "Authorization": f"Bearer {VRITTI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "episode_type": "failure",
            "goal": goal,
            "actions_taken": actions,
            "error_class": classify_error(error),
            "error_trace": str(error),
            "tool_chain": extract_tools(actions),
            "environment_info": get_environment(),
            "resolution": resolution,
            "severity": assess_severity(error)
        }
    )
    return response.json()
```

---

### 2. Gating API (Prevent Repeats)

**Endpoint**: `POST /api/v1/reflect`

**When to call**: Before executing risky actions

**Request**:
```json
{
  "proposed_action": "kubectl apply -f deployment.yaml",
  "goal": "Deploy app",
  "tool": "kubectl",
  "context": "Deploying to production with latest tag",
  "current_state": {
    "cluster": "production",
    "image_tag": "latest"
  }
}
```

**Response**:
```json
{
  "recommendation": "BLOCK",
  "confidence": 0.95,
  "rationale": "High risk: Similar action failed previously (0.96 similarity, 0.91 precondition match). Root cause: Image tag 'latest' not found in registry",
  "suggested_action": "Use specific version tag: myapp:v1.2.3",
  "hints": [
    "Previous failure: Deploy app to Kubernetes"
  ],
  "matched_failures": [...],
  "relevant_skills": [],
  "search_latency_ms": 12.3,
  "total_latency_ms": 45.7
}
```

**Recommendation Types**:

| Type | Confidence | Meaning | Action |
|------|-----------|---------|--------|
| **BLOCK** | >90% | Very likely to fail | DON'T execute, use suggested fix |
| **REWRITE** | 80-90% | Likely to fail | Use suggested alternative |
| **HINT** | 70-80% | Might fail | Show warning, let agent decide |
| **PROCEED** | <70% | No past failures | Execute as planned |

**Python Example**:
```python
def check_vritti_before_action(action, goal, tool, context):
    """Check Vritti gating before executing action."""
    response = requests.post(
        "https://vritti.api.company.com/api/v1/reflect",
        headers={
            "Authorization": f"Bearer {VRITTI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "proposed_action": action,
            "goal": goal,
            "tool": tool,
            "context": context,
            "current_state": get_current_state()
        },
        timeout=2  # 2s timeout
    )
    return response.json()

# Usage
result = check_vritti_before_action(
    action="kubectl apply -f deployment.yaml",
    goal="Deploy app",
    tool="kubectl",
    context="Deploying to production"
)

if result["recommendation"] == "BLOCK":
    print(f"Action blocked: {result['rationale']}")
    if result["suggested_action"]:
        print(f"Suggestion: {result['suggested_action']}")
    return False

elif result["recommendation"] == "REWRITE":
    print(f"Rewrite recommended: {result['rationale']}")
    print(f"Use instead: {result['suggested_action']}")
    # Agent decides whether to use suggestion

elif result["recommendation"] == "HINT":
    print(f"Hint: {result['rationale']}")
    # Agent sees warning but can proceed

return True  # PROCEED
```

---

## Gating Flow

### Complete Integration Example

```python
class VrittiAgent:
    """AI agent with Vritti integration."""

    def __init__(self, vritti_api_key):
        self.vritti_api_key = vritti_api_key
        self.cache = {}  # Client-side cache

    def execute_action(self, action, goal):
        """Execute action with Vritti gating."""

        # 1. Determine if action needs gating
        if not self.is_risky_action(action):
            return self.execute(action)

        # 2. Check cache first (5-min TTL)
        cache_key = self.get_cache_key(action, self.get_context())
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if cached_result["timestamp"] > time.time() - 300:
                print(f"Using cached gating result")
                return self.handle_gating_result(cached_result, action)

        # 3. Call Vritti gating
        try:
            gating_result = self.check_vritti(action, goal)
            self.cache[cache_key] = {
                **gating_result,
                "timestamp": time.time()
            }
        except Exception as e:
            print(f"Vritti unavailable: {e}. Proceeding with caution.")
            return self.execute(action)  # Fail open

        # 4. Handle recommendation
        return self.handle_gating_result(gating_result, action)

    def handle_gating_result(self, result, action):
        """Handle Vritti gating result."""
        recommendation = result["recommendation"]

        if recommendation == "BLOCK":
            print(f"BLOCKED: {result['rationale']}")
            if result.get("suggested_action"):
                print(f"Try instead: {result['suggested_action']}")
                # Optionally: auto-execute suggested action
                return self.execute(result["suggested_action"])
            return {"status": "blocked", "reason": result["rationale"]}

        elif recommendation == "REWRITE":
            print(f"REWRITE: {result['rationale']}")
            suggested = result.get("suggested_action")
            if suggested and self.confidence_threshold(result["confidence"]):
                print(f"Auto-rewriting to: {suggested}")
                return self.execute(suggested)
            else:
                # Ask user or use LLM to decide
                return self.ask_user_or_llm(action, suggested, result)

        elif recommendation == "HINT":
            print(f"HINT: {result['rationale']}")
            # Log warning but proceed
            return self.execute(action)

        else:  # PROCEED
            return self.execute(action)

    def report_failure(self, action, goal, error, resolution):
        """Report failure to Vritti for learning."""
        try:
            response = requests.post(
                "https://vritti.api.company.com/api/v1/capture",
                headers={"Authorization": f"Bearer {self.vritti_api_key}"},
                json={
                    "episode_type": "failure",
                    "goal": goal,
                    "actions_taken": [action],
                    "error_class": self.classify_error(error),
                    "error_trace": str(error),
                    "tool_chain": self.extract_tools(action),
                    "environment_info": self.get_environment(),
                    "resolution": resolution,
                    "severity": self.assess_severity(error)
                }
            )
            print(f"Failure reported to Vritti: episode_id={response.json()['episode_id']}")
        except Exception as e:
            print(f"Failed to report to Vritti: {e}")
```

---

## Client-Side Caching

**CRITICAL**: Implement client-side caching to reduce latency and cost.

### Caching Strategy

```python
import hashlib
import time

class VrittiCache:
    """Client-side cache for Vritti gating results."""

    def __init__(self, ttl_seconds=300):  # 5-min TTL
        self.cache = {}
        self.ttl = ttl_seconds

    def get_cache_key(self, action, context):
        """Generate cache key from action + context."""
        key_str = f"{action}|{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, action, context):
        """Get cached result if valid."""
        key = self.get_cache_key(action, context)
        if key in self.cache:
            entry = self.cache[key]
            if entry["expires_at"] > time.time():
                entry["hit_count"] += 1
                return entry["result"]
            else:
                del self.cache[key]  # Expired
        return None

    def set(self, action, context, result):
        """Cache result."""
        key = self.get_cache_key(action, context)
        self.cache[key] = {
            "result": result,
            "expires_at": time.time() + self.ttl,
            "hit_count": 0
        }

    def clear_expired(self):
        """Clear expired entries."""
        now = time.time()
        expired_keys = [k for k, v in self.cache.items() if v["expires_at"] <= now]
        for key in expired_keys:
            del self.cache[key]
```

**Expected Cache Hit Rate**: 70-80%

**Latency Comparison**:
- Cached: <1ms
- Uncached (with LLM): ~220ms
- Uncached (without LLM): ~50ms

---

## Mission Metrics

### Track Mission Success

Vritti exposes Prometheus metrics to track mission success:

```promql
# Total repeat errors prevented (24h)
increase(episodic_memory_repeat_error_prevented_total[24h])

# Mission success rate
(
  increase(episodic_memory_repeat_error_prevented_total[1h]) /
  increase(episodic_memory_gating_decision_total[1h])
) * 100

# Prevention rate by error class
topk(10, sum by (error_class) (
  increase(episodic_memory_repeat_error_prevented_total[24h])
))
```

### Grafana Dashboard

Import the dashboard:
```bash
# Import dashboard JSON
kubectl apply -f monitoring/grafana_dashboard_mission_metrics.json
```

Dashboard includes:
- Total repeat errors prevented
- Mission success rate (target: >80%)
- Real-time prevention gauge
- Prevention trends over time
- Top error classes prevented
- Estimated value delivered

---

## Example Implementations

### Example 1: Claude Code Integration

```python
# claude_code_vritti_integration.py

class ClaudeCodeWithVritti:
    """Claude Code with Vritti integration."""

    def __init__(self):
        self.vritti = VrittiAgent(os.getenv("VRITTI_API_KEY"))

    async def execute_bash_command(self, command, context):
        """Execute bash command with Vritti gating."""

        # Check if command is risky
        if self.is_write_operation(command):
            gating_result = self.vritti.check_vritti(
                action=command,
                goal=context.get("goal", "Execute command"),
                tool="bash",
                context=context
            )

            if gating_result["recommendation"] == "BLOCK":
                return {
                    "success": False,
                    "blocked_by_vritti": True,
                    "reason": gating_result["rationale"],
                    "suggestion": gating_result.get("suggested_action")
                }

        # Execute command
        try:
            result = subprocess.run(command, shell=True, capture_output=True, timeout=30)
            if result.returncode != 0:
                # Failure - report to Vritti
                self.vritti.report_failure(
                    action=command,
                    goal=context.get("goal"),
                    error=result.stderr.decode(),
                    resolution=None  # Will be filled when fixed
                )
            return {"success": result.returncode == 0, "output": result.stdout.decode()}
        except Exception as e:
            # Report exception to Vritti
            self.vritti.report_failure(
                action=command,
                goal=context.get("goal"),
                error=str(e),
                resolution=None
            )
            return {"success": False, "error": str(e)}
```

### Example 2: Agentic Framework (LangChain)

```python
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor

class VrittiGatingTool(BaseTool):
    """LangChain tool that checks Vritti before execution."""

    name = "vritti_gate"
    description = "Check if action is safe before executing"

    def _run(self, action: str, goal: str, tool: str) -> dict:
        """Check Vritti gating."""
        vritti = VrittiAgent(os.getenv("VRITTI_API_KEY"))
        return vritti.check_vritti(action, goal, tool, {})

# Agent with Vritti integration
tools = [VrittiGatingTool(), ...other tools...]
agent = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory
)

# Agent will check Vritti before risky actions
agent.run("Deploy the app to production")
```

---

## Troubleshooting

### Common Issues

#### 1. Gating Latency Too High

**Symptom**: P99 latency >100ms

**Solutions**:
- Implement client-side caching (should hit 70-80%)
- Check LLM validation cache hit rate (target >80%)
- Reduce `k` parameter in search (default: 20)
- Check KyroDB health

#### 2. Too Many False Blocks

**Symptom**: Actions blocked that should proceed

**Solutions**:
- Check similarity thresholds (may be too low)
- Verify precondition matching quality
- Review episode reflection quality
- Consider enabling LLM validation (already enabled in v0.1)

#### 3. Too Few Preventions

**Symptom**: Mission success rate <50%

**Solutions**:
- Check if episodes have quality reflections
- Verify gating is being called for risky actions
- Review similarity/precondition thresholds (may be too strict)
- Check if episode corpus is too small (need >100 episodes)

#### 4. Vritti API Unavailable

**Symptom**: Timeouts or 503 errors

**Solutions**:
- Implement fail-open strategy (proceed without gating)
- Use exponential backoff for retries
- Check Vritti service health: `GET /health`
- Verify API key is valid

---

## Support

- **Documentation**: https://docs.vritti.ai
- **API Reference**: https://api.vritti.ai/docs
- **Grafana Dashboard**: https://grafana.company.com/d/vritti-mission
- **Slack**: #vritti-support
- **Email**: support@vritti.ai

---

## Next Steps

1. **Implement Integration**: Follow examples above
2. **Enable Metrics**: Set up Grafana dashboard
3. **Monitor Mission Success**: Track prevention rate (target: >80%)
4. **Iterate**: Tune thresholds based on false positive/negative rates

**Mission**: Enable AI agents that learn from mistakes and don't repeat them.
