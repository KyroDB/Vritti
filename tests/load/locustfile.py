"""
Vritti Load Testing Suite using Locust.

Targets:
- 100 RPS sustained load
- P99 latency < 50ms for search/capture
- P99 latency < 5s for reflection generation

Usage:
    # Install dependencies
    pip install locust

    # Run with web UI
    locust -f tests/load/locustfile.py --host http://localhost:8000

    # Run headless (CI/CD)
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless --users 50 --spawn-rate 5 --run-time 5m \
        --csv results/load_test

    # Run specific scenarios
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --tags search --users 100

Requirements:
    - Vritti API running at target host
    - Valid API key in VRITTI_API_KEY environment variable
    - KyroDB instances healthy
"""

import os
import random
import time
from datetime import datetime
from typing import Optional

from locust import HttpUser, task, between, tag, events
from locust.runners import MasterRunner


# Configuration
API_KEY = os.environ.get("VRITTI_API_KEY", "test-api-key-for-load-testing")
CUSTOMER_ID = os.environ.get("VRITTI_CUSTOMER_ID", "load-test-customer")

# Sample data for realistic payloads
SAMPLE_GOALS = [
    "Deploy Python application to Kubernetes cluster",
    "Fix database connection timeout in production",
    "Configure nginx reverse proxy for microservices",
    "Debug memory leak in Node.js application",
    "Set up CI/CD pipeline with GitHub Actions",
    "Migrate PostgreSQL database to new server",
    "Implement authentication with OAuth2",
    "Optimize Docker image size for faster deployments",
    "Configure basic health checks",
    "Debug intermittent test failures in pytest suite",
]

SAMPLE_ERROR_TRACES = [
    "TypeError: Cannot read property 'id' of undefined\n    at processUser (/app/src/users.js:45:12)",
    "ConnectionRefusedError: [Errno 111] Connection refused\n  File 'app.py', line 23, in connect_db",
    "FATAL:  password authentication failed for user 'postgres'",
    "Error: ENOENT: no such file or directory, open '/app/config.json'",
    "kubectl error: pods 'my-app-xyz' not found in namespace 'production'",
    "docker: Error response from daemon: Conflict. Container already exists",
    "nginx: [emerg] bind() to 0.0.0.0:80 failed (98: Address already in use)",
    "ImportError: No module named 'tensorflow.keras'",
    "java.lang.OutOfMemoryError: Java heap space",
    "SSL: CERTIFICATE_VERIFY_FAILED",
]

SAMPLE_TOOLS = [
    ["kubectl", "helm"],
    ["docker", "docker-compose"],
    ["python", "pip"],
    ["npm", "node"],
    ["git", "github-actions"],
    ["nginx", "certbot"],
    ["psql", "pgadmin"],
    ["terraform", "aws-cli"],
    ["pytest", "coverage"],
    ["redis-cli", "celery"],
]

SAMPLE_TAGS = [
    ["kubernetes", "deployment"],
    ["docker", "containerization"],
    ["python", "backend"],
    ["database", "postgresql"],
    ["ci-cd", "automation"],
    ["security", "authentication"],
    ["monitoring", "observability"],
    ["networking", "proxy"],
]


def generate_episode_payload() -> dict:
    """Generate realistic episode capture payload."""
    return {
        "goal": random.choice(SAMPLE_GOALS),
        "error_trace": random.choice(SAMPLE_ERROR_TRACES),
        "code_state_diff": "- old_config = {}\n+ new_config = {'timeout': 30}",
        "actions_taken": [
            "Checked logs for errors",
            "Verified configuration",
            "Attempted restart",
        ],
        "tool_chain": random.choice(SAMPLE_TOOLS),
        "episode_type": random.choice(["failure", "success"]),
        "error_class": random.choice([
            "configuration_error",
            "dependency_error",
            "environment_error",
            "tool_error",
            "validation_error",
        ]),
        "environment_info": {
            "os": "linux",
            "python_version": "3.11",
            "node_version": "18.17.0",
        },
        "tags": random.choice(SAMPLE_TAGS),
        "customer_id": CUSTOMER_ID,
    }


def generate_search_payload() -> dict:
    """Generate realistic search payload."""
    return {
        "goal": random.choice(SAMPLE_GOALS),
        "customer_id": CUSTOMER_ID,
        "k": random.choice([5, 10, 20]),
        "min_similarity": 0.5,
    }


def generate_reflect_payload() -> dict:
    """Generate pre-action gating payload."""
    return {
        "action": f"Run command: {random.choice(['kubectl apply', 'docker push', 'npm install'])}",
        "context": {
            "tool": random.choice(["kubectl", "docker", "npm"]),
            "environment": "production" if random.random() > 0.7 else "staging",
        },
        "customer_id": CUSTOMER_ID,
    }


class VrittiUser(HttpUser):
    """
    Simulates a typical AI agent using Vritti for episodic memory.
    
    Behavior pattern:
    - 60% search requests (checking for past failures)
    - 25% capture requests (storing new episodes)
    - 10% reflect requests (pre-action gating)
    - 5% health checks
    """
    
    wait_time = between(0.1, 0.5)  # 100-500ms between requests
    
    def on_start(self):
        """Initialize user session."""
        self.headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json",
        }
        self.captured_episode_ids = []
    
    @task(60)
    @tag("search")
    def search_episodes(self):
        """Search for similar episodes (most common operation)."""
        payload = generate_search_payload()
        
        with self.client.post(
            "/api/v1/search",
            json=payload,
            headers=self.headers,
            name="/api/v1/search",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    response.success()
                else:
                    response.failure("Missing results in response")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(25)
    @tag("capture")
    def capture_episode(self):
        """Capture a new episode."""
        payload = generate_episode_payload()
        
        with self.client.post(
            "/api/v1/capture",
            json=payload,
            headers=self.headers,
            name="/api/v1/capture",
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                data = response.json()
                if "episode_id" in data:
                    self.captured_episode_ids.append(data["episode_id"])
                    response.success()
                else:
                    response.failure("Missing episode_id in response")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(10)
    @tag("reflect")
    def reflect_before_action(self):
        """Pre-action gating check."""
        payload = generate_reflect_payload()
        
        with self.client.post(
            "/api/v1/reflect",
            json=payload,
            headers=self.headers,
            name="/api/v1/reflect",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "recommendation" in data:
                    response.success()
                else:
                    response.failure("Missing recommendation in response")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(5)
    @tag("health")
    def check_health(self):
        """Health check (lightweight)."""
        with self.client.get(
            "/health/liveness",
            name="/health/liveness",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class HighThroughputUser(HttpUser):
    """
    High-throughput user for stress testing.
    
    Simulates burst traffic with minimal wait time.
    Use for capacity testing.
    """
    
    wait_time = between(0.01, 0.05)  # 10-50ms between requests
    
    def on_start(self):
        self.headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json",
        }
    
    @task(80)
    @tag("stress", "search")
    def rapid_search(self):
        """Rapid search requests for stress testing."""
        payload = generate_search_payload()
        self.client.post(
            "/api/v1/search",
            json=payload,
            headers=self.headers,
            name="/api/v1/search [stress]",
        )
    
    @task(20)
    @tag("stress", "capture")
    def rapid_capture(self):
        """Rapid capture requests for stress testing."""
        payload = generate_episode_payload()
        self.client.post(
            "/api/v1/capture",
            json=payload,
            headers=self.headers,
            name="/api/v1/capture [stress]",
        )


class ReflectionLoadUser(HttpUser):
    """
    User focused on reflection generation load testing.
    
    Tests LLM integration under load.
    """
    
    wait_time = between(0.5, 2.0)  # Slower due to LLM calls
    
    def on_start(self):
        self.headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json",
        }
    
    @task(50)
    @tag("reflection")
    def capture_with_reflection(self):
        """Capture with reflection enabled."""
        payload = generate_episode_payload()
        
        with self.client.post(
            "/api/v1/capture?generate_reflection=true",
            json=payload,
            headers=self.headers,
            name="/api/v1/capture [with reflection]",
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                data = response.json()
                if data.get("reflection_queued"):
                    response.success()
                else:
                    response.failure("Reflection not queued")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(30)
    @tag("reflection")
    def capture_premium_tier(self):
        """Capture with premium tier reflection."""
        payload = generate_episode_payload()
        payload["error_class"] = "data_loss"  # Forces premium tier
        
        self.client.post(
            "/api/v1/capture?generate_reflection=true&tier=premium",
            json=payload,
            headers=self.headers,
            name="/api/v1/capture [premium tier]",
        )
    
    @task(20)
    @tag("reflection")
    def capture_cheap_tier(self):
        """Capture with cheap tier reflection."""
        payload = generate_episode_payload()
        
        self.client.post(
            "/api/v1/capture?generate_reflection=true&tier=cheap",
            json=payload,
            headers=self.headers,
            name="/api/v1/capture [cheap tier]",
        )


# Event hooks for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print(f"\n{'='*60}")
    print("Vritti Load Test Starting")
    print(f"Target Host: {environment.host}")
    print(f"API Key: {API_KEY[:10]}...")
    print(f"Customer ID: {CUSTOMER_ID}")
    print(f"{'='*60}\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print(f"\n{'='*60}")
    print("Load Test Complete")
    
    if environment.stats.total.num_requests > 0:
        print(f"Total Requests: {environment.stats.total.num_requests}")
        print(f"Failure Rate: {environment.stats.total.fail_ratio * 100:.2f}%")
        print(f"Avg Response Time: {environment.stats.total.avg_response_time:.2f}ms")
        print(f"P95 Response Time: {environment.stats.total.get_response_time_percentile(0.95):.2f}ms")
        print(f"P99 Response Time: {environment.stats.total.get_response_time_percentile(0.99):.2f}ms")
        print(f"RPS: {environment.stats.total.current_rps:.2f}")
    
    print(f"{'='*60}\n")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Track SLO violations."""
    # P99 SLO: 50ms for search/capture, 5000ms for reflection
    if "reflection" in name.lower() or "premium" in name.lower():
        slo_threshold = 5000
    else:
        slo_threshold = 50
    
    if response_time > slo_threshold and exception is None:
        print(f"[SLO VIOLATION] {name}: {response_time:.2f}ms > {slo_threshold}ms")
