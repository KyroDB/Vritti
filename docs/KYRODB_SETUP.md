# KyroDB Setup for Vritti Integration Testing

This document describes how to set up KyroDB for Vritti integration testing.

## Overview

Vritti uses KyroDB as its vector database for storing and searching episodic memories. The integration tests require a running KyroDB instance with the correct embedding dimensions.

## Prerequisites

1. **KyroDB Binary**: The `kyrodb_server` binary must be built and available
2. **Python Environment**: Vritti's Python dependencies installed
3. **Network Ports**: Ports 50051 (text) and 50052 (image) available

## Quick Start

### 1. Build KyroDB (if not already built)

```bash
cd /path/to/ProjectKyro
cargo build --release -p kyrodb-engine --bin kyrodb_server
```

### 2. Create Test Configuration

Create `/tmp/kyrodb_test_config.toml`:

```toml
# KyroDB Test Configuration for Vritti Integration Tests
# Uses 384-dim embeddings to match all-MiniLM-L6-v2

[server]
port = 50051
max_connections = 100

[storage]
data_dir = "/tmp/kyrodb_test_data"
wal_sync = "immediate"

[hnsw]
# 384 dimensions for all-MiniLM-L6-v2 embeddings
dimension = 384
distance = "cosine"
M = 16
ef_construction = 100
ef_search = 50
max_elements = 100000

[cache]
enable_learned_cache = true
l1_cache_size_mb = 64
l2_cache_size_mb = 128

[logging]
level = "info"
```

### 3. Start KyroDB Server

```bash
# Create clean data directory
rm -rf /tmp/kyrodb_test_data
mkdir -p /tmp/kyrodb_test_data

# Start server with test config
./target/release/kyrodb_server --config /tmp/kyrodb_test_config.toml
```

### 4. Run Integration Tests

```bash
cd Vritti
python -m pytest tests/integration/test_kyrodb_real.py -v
```

## Test Configuration

### Embedding Dimensions

Vritti uses `all-MiniLM-L6-v2` from sentence-transformers, which produces **384-dimensional** embeddings. The KyroDB server must be configured to match:

```toml
[hnsw]
dimension = 384
```

### Dual Instance Architecture (Production)

In production, Vritti uses two KyroDB instances:
- **Text instance** (port 50051): 384-dim embeddings for text/code
- **Image instance** (port 50052): 512-dim embeddings for images

For testing, we use a single instance with 384-dim.

### Environment Variables

Set these in your `.env` file for Vritti:

```bash
# KyroDB connection (for tests)
KYRODB_TEXT_HOST=localhost
KYRODB_TEXT_PORT=50051
KYRODB_IMAGE_HOST=localhost
KYRODB_IMAGE_PORT=50052
KYRODB_ENABLE_TLS=false
KYRODB_REQUEST_TIMEOUT_SECONDS=30
```

## Integration Test Coverage

The following tests validate KyroDB integration:

| Test | Description | Status |
|------|-------------|--------|
| `test_health_check` | Verify server connectivity | PASS |
| `test_insert_and_query` | Insert and retrieve documents | PASS |
| `test_search` | k-NN vector search | PASS |
| `test_router_health` | Router health check (both instances) | PASS |
| `test_insert_episode` | Episode insertion via router | PASS |
| `test_search_episodes` | Episode search via router | PASS |
| `test_update_episode_reflection` | Reflection persistence | PASS |
| `test_insert_and_search_skill` | Skills storage and search | PASS |

## Troubleshooting

### Dimension Mismatch Error

```
grpc.aio._call.AioRpcError: Search failed: query dimension mismatch: expected 768 found 384
```

**Solution**: Ensure KyroDB is configured with `dimension = 384` in the HNSW config.

### Connection Refused

```
grpc._channel._InactiveRpcError: failed to connect to all addresses
```

**Solution**: 
1. Check if KyroDB is running: `lsof -i :50051`
2. Start the server if not running

### Health Check Degraded

The server may report "Degraded" status on first start (before any data is inserted). This is expected and tests will still pass.

## Performance Benchmarks

Target performance for integration tests:
- Insert: < 1ms per document
- Query: < 1ms P99
- Search: < 5ms P99 for k=10
- Batch operations: < 10ms for 100 documents

## Production Deployment

For production deployment, see:
- `docs/OPERATIONS.md` - Operational guidelines
- `docs/CONFIGURATION_MANAGEMENT.md` - Configuration best practices
- `docs/BACKUP_AND_RECOVERY.md` - Data durability

## Related Documentation

- [Architecture Overview](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [Observability](OBSERVABILITY.md)
