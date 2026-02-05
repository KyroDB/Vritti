# Vritti API Guide

Quick reference for integrating AI agents with Vritti.

## Authentication

Include API key in all requests:
```http
X-API-Key: em_live_your_api_key
```

## Core Endpoints

### 1. Capture Failure
`POST /api/v1/capture`

When your agent fails, capture it. Reflection is generated asynchronously when LLM reflection is enabled
(recommended; set `SERVICE_REQUIRE_LLM_REFLECTION=true` to fail fast if LLM config is missing):
```json
{
  "episode_type": "failure",
  "goal": "Deploy to Kubernetes",
  "tool_chain": ["kubectl"],
  "actions_taken": ["kubectl apply -f deployment.yaml"],
  "error_trace": "ImagePullBackOff: image not found",
  "error_class": "resource_error",
  "resolution": "Fixed image tag",
  "tags": ["kubernetes", "production"],
  "screenshot_base64": "<base64-encoded screenshot bytes>"
}
```

**Response**:
```json
{
  "episode_id": 12345,
  "collection": "failures",
  "ingestion_latency_ms": 42.7,
  "text_stored": true,
  "image_stored": true,
  "reflection_queued": true
}
```

### 2. Check Before Action (Gating)
`POST /api/v1/reflect`

Before risky actions, ask if it's safe:
```json
{
  "proposed_action": "kubectl apply -f deployment.yaml",
  "goal": "Deploy application",
  "tool": "kubectl",
  "context": "Deploying to production",
  "current_state": {"cluster": "prod"}
}
```

**Response**:
```json
{
  "recommendation": "block",
  "rationale": "Similar action failed before (96% match). Image tag 'latest' not found",
  "suggested_action": "Use specific tag: myapp:v1.2.3",
  "confidence": 0.95,
  "matched_failures": [],
  "hints": [],
  "relevant_skills": [],
  "search_latency_ms": 12.4,
  "total_latency_ms": 18.9
}
```

**Recommendations**:
- `proceed` - Safe to proceed
- `block` - Don't do it, use suggestion
- `rewrite` - Use alternative action
- `hint` - Warning, proceed with caution

### 3. Search Past Failures
`POST /api/v1/search`

Find similar problems you've solved:
```json
{
  "goal": "Fix ImagePullBackOff error",
  "k": 5
}
```

**Multi-modal search** (include `image_base64` to enable image search):
```json
{
  "goal": "UI screenshot error",
  "k": 5,
  "image_base64": "<base64-encoded image bytes>",
  "image_weight": 0.3,
  "include_archived": false
}
```

**Response**:
```json
{
  "results": [{
    "episode": {
      "episode_id": 987,
      "create_data": {
        "goal": "Deploy to Kubernetes"
      },
      "reflection": {
        "root_cause": "Image tag doesn't exist",
        "resolution_strategy": "Verify tag exists before applying",
        "confidence_score": 0.95
      }
    },
    "scores": {
      "similarity": 0.89,
      "combined": 0.91
    },
    "rank": 1
  }]
}
```

## Error Classes

Use correct classification for better search:
- `configuration_error` - Wrong config values
- `permission_error` - Access denied
- `network_error` - Connection issues  
- `resource_error` - File/image not found
- `dependency_error` - Missing packages
- `timeout_error` - Took too long
- `validation_error` - Invalid input
- `unknown` - Unclassified

## Python Example

```python
import httpx

async def check_action(action, goal):
    """Check if action is safe before executing."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/reflect",
            headers={"X-API-Key": "em_live_your_key"},
            json={
                "proposed_action": action,
                "goal": goal,
                "tool": "shell",
                "context": "",
                "current_state": {}
            }
        )
        result = response.json()
        
        if result['recommendation'] == 'block':
            print(f"BLOCKED: {result['rationale']}")
            return False
        
        return True

# Usage
if await check_action("rm -rf /", "Clean files"):
    os.system("rm -rf /")  # Only execute if safe
```

## Health Check

Verify Vritti is running:
```bash
curl http://localhost:8000/health
```

## API Documentation

Interactive API docs: `http://localhost:8000/docs`
