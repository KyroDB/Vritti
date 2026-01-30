#!/usr/bin/env python3
"""
Simple Vritti System Test (No Authentication Required)
========================================================

Tests basic Vritti functionality without requiring API keys.
Perfect for quickly validating the system is running correctly.

Usage:
    python3 examples/test_no_auth.py
"""

import asyncio
import httpx
import json
from datetime import datetime


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print('=' * 60)


async def test_health_check():
    """Test the health check endpoint."""
    print_section("1. Testing Health Check")
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8000/health")
            
            if response.status_code == 200:
                health = response.json()
                
                print(f"Vritti is running!")
                print(f"   Status: {health['status']}")
                print(f"   Version: {health['version']}")
                print(f"   Uptime: {health['uptime_seconds']:.1f}s")
                
                print("\nComponents Status:")
                for component in health.get('components', []):
                    print(f"   {component['name']}: {component['status']}")
                    if component.get('message'):
                        print(f"      └─ {component['message']}")
                    if component.get('latency_ms') is not None:
                        print(f"      └─ Latency: {component['latency_ms']:.2f}ms")
                
                return True
            else:
                print(f" Health check failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f" Cannot connect to Vritti: {e}")
        print("\nMake sure Vritti is running:")
        print("   cd Vritti && uvicorn src.main:app --port 8000")
        return False

async def test_core_endpoints():
    """Validate core endpoints that do not require authentication."""
    print_section("2. Testing Core Endpoints")
    async with httpx.AsyncClient() as client:
        for path in ["/health/liveness", "/health/readiness", "/health"]:
            try:
                response = await client.get(f"http://localhost:8000{path}")
                print(f"{path}: {response.status_code}")
            except Exception as e:
                print(f"Error testing {path}: {e}")


async def test_api_docs():
    """Test that API documentation is available."""
    print_section("3. Testing API Documentation")
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8000/docs")
            
            if response.status_code == 200:
                print(f" API docs available at: http://localhost:8000/docs")
                print(f"   Interactive Swagger UI ready")
                return True
            else:
                print(f" API docs failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f" Cannot access API docs: {e}")
        return False


async def test_openapi_schema():
    """Test OpenAPI schema endpoint."""
    print_section("4. Testing OpenAPI Schema")
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8000/openapi.json")
            
            if response.status_code == 200:
                schema = response.json()
                
                print(f" OpenAPI schema available")
                print(f"   Title: {schema.get('info', {}).get('title')}")
                print(f"   Version: {schema.get('info', {}).get('version')}")
                
                # Count endpoints
                paths = schema.get('paths', {})
                endpoint_count = len(paths)
                
                print(f"\nAPI Endpoints: {endpoint_count}")
                for path in list(paths.keys())[:5]:
                    methods = list(paths[path].keys())
                    print(f"   {path} [{', '.join(m.upper() for m in methods)}]")
                
                if endpoint_count > 5:
                    print(f"   ... and {endpoint_count - 5} more")
                
                return True
            else:
                print(f" OpenAPI schema failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f" Cannot access OpenAPI schema: {e}")
        return False


async def test_kyrodb_connectivity():
    """Test that Vritti can connect to KyroDB instances."""
    print_section("5. Testing KyroDB Connectivity")
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8000/health")
            
            if response.status_code == 200:
                health = response.json()
                
                # Find KyroDB component
                kyrodb_component = None
                for component in health.get('components', []):
                    if component['name'] == 'kyrodb':
                        kyrodb_component = component
                        break
                
                if kyrodb_component:
                    if kyrodb_component['status'] == 'healthy':
                        metadata = kyrodb_component.get('metadata', {})
                        print(f" KyroDB instances connected")
                        print(f"   Text instance (50051): {'healthy' if metadata.get('text_healthy') else 'unhealthy'}")
                        print(f"   Image instance (50052): {'healthy' if metadata.get('image_healthy') else 'unhealthy'}")
                        print(f"   Latency: {kyrodb_component.get('latency_ms', 0):.2f}ms")
                        return True
                    else:
                        print(f"KyroDB status: {kyrodb_component['status']}")
                        print(f"   Message: {kyrodb_component.get('message')}")
                        return False
                else:
                    print(f" KyroDB component not found in health check")
                    return False
                    
    except Exception as e:
        print(f" Cannot check KyroDB connectivity: {e}")
        return False


async def run_all_tests():
    """Run all tests and show summary."""
    print("=" * 60)
    print("Vritti System Test (No Authentication)")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "Health Check": await test_health_check(),
        "Core Endpoints": await test_core_endpoints(),
        "API Documentation": await test_api_docs(),
        "OpenAPI Schema": await test_openapi_schema(),
        "KyroDB Connectivity": await test_kyrodb_connectivity(),
    }
    
    # Summary
    print_section("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed. Vritti is working correctly.")
    elif passed > 0:
        print(" Some tests passed. Check failed tests above.")
    else:
        print(" All tests failed. Check if Vritti is running.")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
