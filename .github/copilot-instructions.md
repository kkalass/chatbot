# Project Guidelines

## Overview
Python RAG chatbot using Chainlit + Ollama + Haystack + Qdrant.
Full spec in `doc/` (01-product-scope → 05-delivery-plan).

## Architecture
See [doc/03-architecture.md](../doc/03-architecture.md) for component boundaries, data flows, and auth sequence.
- Modules: `src/ui`, `src/app`, `src/retrieval`, `src/ingestion`, `src/tools`, `src/config`
- Core orchestration must not import infrastructure directly — use `Protocol`-based interfaces at retrieval and tool adapter boundaries
- Dependencies injected via constructor parameters; composition in a factory function at startup
- Auth credentials are session-scoped only (objects created in `on_chat_start`); never from env per-user, never from model output

## Build and Test
```
uv sync                        # install dependencies
uv run ruff check .            # lint
uv run ruff format .           # format
uv run pyright                 # type check (strict via pyproject.toml)
uv run pytest                  # tests
uv run chainlit run src/ui/app.py  # start app
```

## Code Quality (NFR-06)
- `pyright` strict mode via `typeCheckingMode = "strict"` in `pyproject.toml` — must pass clean
- `ruff` for linting and formatting — no separate black/isort/flake8
- `uv` exclusively for dependency and environment management — no pip, no poetry
- `structlog` for all logging in application code (`structlog.get_logger()`, never `logging.getLogger()`)
- Modern union syntax: `X | None`, `X | Y` — never `Optional[X]` or `Union[X, Y]`
- Pydantic v2 at system boundaries (tool schemas, external responses, env config via `pydantic-settings`)
- `@dataclass(frozen=True)` for internal value objects — no Pydantic internally
- `Protocol` for structural typing at module boundaries — not concrete base classes
- No `Any` without explicit justification in a comment

## Working Agreement
- **Never** unilaterally change, simplify, or reinterpret agreed requirements during implementation
- If a conflict or ambiguity is discovered, **stop and raise it** before proceeding
- Trade-off decisions require explicit human approval — never make them autonomously
- Agreed design is intentional; apparent conflicts are a signal to discuss, not to "fix" silently
