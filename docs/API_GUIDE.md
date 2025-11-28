# EpisodicMemory API Guide

This guide details how to integrate your AI agents with the EpisodicMemory system.

## Authentication

All API requests require an API Key in the header:

```http
X-API-Key: your_api_key_here
```

## Core Workflows

### 1. Capture a Failure Episode

Call this when your agent encounters an error or fails a task.

**Endpoint**: `POST /api/v1/capture`

**Request**:
```json
{
  "episode_type": "failure",
  "goal": "Deploy application to Kubernetes",
  "tool_chain": ["kubectl", "helm"],
  "actions_taken": [
    "kubectl apply -f deployment.yaml"
  ],
  "error_trace": "ImagePullBackOff: rpc error: code = Unknown desc = ...",
  "error_class": "resource_error",
  "tags": ["kubernetes", "deployment", "production"]
}
```

**Response**:
```json
{
  "episode_id": 12345,
  "status": "ingested",
  "reflection_status": "pending"
}
```

### 2. Search for Solutions

Call this before taking an action or when an error occurs, to see if it happened before.

**Endpoint**: `POST /api/v1/search`

**Request**:
```json
{
  "goal": "Fix ImagePullBackOff error",
  "current_state": {
    "cluster": "prod-us-east-1",
    "last_command": "kubectl apply"
  },
  "k": 5,
  "min_similarity": 0.7
}
```

**Response**:
```json
{
  "results": [
    {
      "episode_id": 987,
      "similarity": 0.89,
      "reflection": {
        "root_cause": "Image tag does not exist in registry",
        "resolution_strategy": "Verify image tag exists in ECR before applying",
        "confidence_score": 0.95
      }
    }
  ]
}
```

**Error Response**:
```json
{
  "error": "No candidates found",
  "details": "Try lowering min_similarity or check database connectivity"
}
```

### 3. Pre-Action Gating

Call this *before* executing a sensitive or complex action to get a Go/No-Go decision.

**Endpoint**: `POST /api/v1/reflect`

**Request**:
```json
{
  "goal": "Delete production database",
  "proposed_action": "DROP DATABASE users",
  "tool": "psql",
  "current_state": {"env": "production"}
}
```

**Response**:
```json
{
  "recommendation": "block",
  "reason": "High-risk action in production environment detected",
  "matched_failures": [
    {
      "episode_id": 456,
      "similarity": 0.92,
      "outcome": "data_loss"
    }
  ],
  "suggested_alternatives": ["Use soft delete", "Backup first"]
}
```

---

## Client Libraries

### Python

```python
import requests
from typing import Optional, List, Dict

class EpisodicClient:
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.headers = {"X-API-Key": api_key}
        self.base_url = base_url
        self.timeout = timeout

    def capture(
        self,
        goal: str,
        error: str,
        tool_chain: List[str],
        actions_taken: Optional[List[str]] = None,
        error_class: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict:
        """Capture a failure episode."""
        payload = {
            "episode_type": "failure",
            "goal": goal,
            "error_trace": error,
            "tool_chain": tool_chain
        }
        if actions_taken:
            payload["actions_taken"] = actions_taken
        if error_class:
            payload["error_class"] = error_class
        if tags:
            payload["tags"] = tags

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/capture",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to capture episode: {e}")
    
    def search(
        self,
        goal: str,
        k: int = 5,
        min_similarity: float = 0.7,
        current_state: Optional[Dict] = None
    ) -> Dict:
        """Search for similar episodes."""
        payload = {
            "goal": goal,
            "k": k,
            "min_similarity": min_similarity
        }
        if current_state:
            payload["current_state"] = current_state
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/search",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to search: {e}")
    
    def gate(
        self,
        goal: str,
        proposed_action: str,
        tool: str,
        current_state: Optional[Dict] = None
    ) -> Dict:
        """Get pre-action gating decision."""
        payload = {
            "goal": goal,
            "proposed_action": proposed_action,
            "tool": tool
        }
        if current_state:
            payload["current_state"] = current_state
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/reflect",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to gate action: {e}")

# Example usage
client = EpisodicClient(api_key="your_key_here")

# Capture
result = client.capture(
    goal="Deploy to prod",
    error="Connection timeout",
    tool_chain=["kubectl"]
)

# Search  
episodes = client.search(goal="Fix timeout", k=5)

# Gate
decision = client.gate(
    goal="Deploy to prod",
    proposed_action="kubectl delete namespace",
    tool="kubectl"
)
```

### cURL Examples

```bash
# Capture
curl -X POST https://${API_HOST}/api/v1/capture \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "episode_type": "failure",
    "goal": "Deploy app",
    "error_trace": "Connection timeout",
    "tool_chain": ["kubectl"]
  }'

# Search
curl -X POST https://${API_HOST}/api/v1/search \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Fix timeout",
    "k": 5,
    "min_similarity": 0.7
  }'

# Gate
curl -X POST https://${API_HOST}/api/v1/reflect \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Deploy to prod",
    "proposed_action": "kubectl delete ns",
    "tool": "kubectl",
    "current_state": {"env": "production"}
  }'
```

### Error Handling

Implement retry logic with exponential backoff for transient errors (429, 5xx):

```python
import time

def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [429, 500, 502, 503, 504]:
                wait = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded")
```
