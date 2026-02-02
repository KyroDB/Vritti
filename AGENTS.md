# Repository Guidelines

## Project Structure & Module Organization

- `src/`: FastAPI service (entrypoint: `src/main.py`) and feature modules (`routers/`, `storage/`, `retrieval/`, `gating/`, etc.)
- `tests/`: pytest suite with shared fixtures in `tests/conftest.py` (see `tests/integration/`, `tests/load/`, `tests/chaos/`)
- `docs/`: architecture and integration docs (start with `docs/ARCHITECTURE.md`)
- `scripts/`: developer utilities (example: `scripts/generate_api_key.py`)
- `data/`, `examples/`: sample inputs and reference material

Always prioritise depth in implementation over breadth. Complete one feature fully before moving to the next. The code quality and logic should be SOTA level. If you feel the implementation is not going in depth, please raise a flag. Refrain from adding unnecessary bloat or deprecated code as well as unnecessary comments, only useful and required documentation should be added.

## Build, Test, and Development Commands

This repo targets Python 3.11 (see `pyproject.toml` / `.github/workflows/ci.yml`).

```bash
# Install
pip install -r requirements.txt
cp .env.production.example .env

# Run locally (hot reload)
./run_dev.sh
# or: uvicorn src.main:app --reload --port 8000
```

```bash
# Tests
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html

# Quality checks (CI)
ruff check src/ tests/ --fix
black src/ tests/
mypy src/
```

## Coding Style & Naming Conventions

- Formatting: Black + isort (configured in `pyproject.toml`). Prefer type hints; mypy is enabled for `src/`.
- Linting: Ruff (`ruff.toml`). Avoid hand-editing generated gRPC/protobuf files under `src/kyrodb/`.
- Naming: `snake_case` (functions/vars), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants).

## Testing Guidelines

- Follow pytest discovery conventions: `tests/test_*.py`, `test_*` functions.
- Use markers from `pytest.ini` (notably `unit`, `integration`, `requires_kyrodb`, `load`, `slow`).
- Keep unit tests offline; mark KyroDB-dependent tests with `@pytest.mark.requires_kyrodb`.
  - Example: `pytest -m "not requires_kyrodb"` to run without a KyroDB instance.

## Commit & Pull Request Guidelines

- Commit messages in Git history are short, imperative summaries (e.g., `Fix logging import bug`, `Add CI workflow`); avoid vague “update” commits.
- PRs should include: what/why, a test plan (commands + results), and doc updates for behavior/API changes.
- Security: never commit secrets. Use `.env.production.example` as the template and keep local `.env` out of Git.
