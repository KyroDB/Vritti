# KyroDB Setup

Vritti uses KyroDB as its vector database for storing episodes and running searches.

## Prerequisites

- KyroDB server binary
- Rust and Cargo (for building from source)
- Ports 50051 (text) and 50052 (image) available

## Quick Start

### 1. Build KyroDB

```bash
cd /path/to/KyroDB
cargo build --release
```

### 2. Create Configuration

Create `kyrodb_config.toml`:

```toml
[server]
port = 50051
max_connections = 100

[storage]
data_dir = "./kyrodb_data"
wal_sync = "immediate"

[hnsw]
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

### 3. Start KyroDB

```bash
./target/release/kyrodb_server --config kyrodb_config.toml
```

Verify it's running:

```bash
lsof -i :50051
```

### 4. Configure Vritti

Edit `.env`:

```bash
KYRODB_TEXT_HOST=localhost
KYRODB_TEXT_PORT=50051
KYRODB_IMAGE_HOST=localhost
KYRODB_IMAGE_PORT=50052
KYRODB_ENABLE_TLS=false
KYRODB_REQUEST_TIMEOUT_SECONDS=30
```

### 5. Test Connection

```bash
python -m pytest tests/integration/test_kyrodb_real.py -v
```

## Embedding Dimensions

Vritti uses `all-MiniLM-L6-v2` which produces 384-dimensional embeddings.

KyroDB must be configured to match:

```toml
[hnsw]
dimension = 384
```

Mismatch will cause errors:
```
grpc.aio._call.AioRpcError: query dimension mismatch: expected 768 found 384
```

## Dual Instance Setup (Production)

For production, run two KyroDB instances:

**Text instance** (port 50051):
```toml
[hnsw]
dimension = 384
```

**Image instance** (port 50052):
```toml
[hnsw]
dimension = 512
```

Start both:

```bash
# Terminal 1
./kyrodb_server --config text_config.toml

# Terminal 2
./kyrodb_server --config image_config.toml
```

## Troubleshooting

### Connection Refused

Check if KyroDB is running:

```bash
lsof -i :50051
```

Start the server if not running.

### Dimension Mismatch

Ensure KyroDB config has `dimension = 384` for text instance.

### Performance Issues

Expected performance:
- Insert: <1ms per document
- Search: <5ms P99 for k=10

If slower, check:
- Disk I/O (use SSD)
- Memory (increase cache sizes)
- Network latency (use localhost for testing)
