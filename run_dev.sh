#!/bin/bash

# Development runner for Episodic Memory service
# Starts FastAPI with hot reload

set -e

echo "=== Episodic Memory Development Server ==="
echo ""
echo "Starting FastAPI with hot reload on http://localhost:8000"
echo "API docs available at: http://localhost:8000/docs"
echo "Health check: http://localhost:8000/health"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set Python path to project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start uvicorn with reload
python -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
