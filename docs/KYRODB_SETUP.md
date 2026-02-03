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

KyroDB supports YAML/TOML configs. This repo’s examples use **YAML** because it matches KyroDB’s
current documented config schema (`config.example.yaml` in the KyroDB repo).

Create `kyrodb_text.yaml` (384-dim text embeddings):

```yaml
server:
  host: "127.0.0.1"
  port: 50051
  max_connections: 10000

hnsw:
  dimension: 384
  distance: cosine
  max_elements: 200000
  m: 16
  ef_construction: 200
  ef_search: 50

persistence:
  data_dir: "./kyrodb_text_data"
  enable_wal: true
  fsync_policy: data_only

cache:
  capacity: 10000
  strategy: learned

logging:
  level: info
  format: text
```

### 3. Start KyroDB

```bash
./target/release/kyrodb_server --config kyrodb_text.yaml
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

**Image instance** (port 50052):

```yaml
server:
  port: 50052
hnsw:
  dimension: 512
persistence:
  data_dir: "./kyrodb_image_data"
```

Start both:

```bash
# Terminal 1
./kyrodb_server --config kyrodb_text.yaml

# Terminal 2
./kyrodb_server --config kyrodb_image.yaml
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
