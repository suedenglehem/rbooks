# library-rag

A local-first RAG research application for a large personal book library
(PDF + EPUB, ~300 GB and growing). See `CLAUDE_PRD.md` for the full product
specification.

## Status

M0 (environment, skeleton, `doctor`) is complete. See `PROGRESS.md` for the
milestone log. Full-library ingestion is intentionally not yet wired up and is
gated behind pilot-configuration approval (PRD §15).

## Quick start

```sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev          # install runtime + dev tooling into .venv
uv run library-rag --help
uv run library-rag doctor    # read-only environment diagnostics
uv run library-rag doctor --json
```

## Commands

`library-rag --help` documents the whole intended CLI surface. M0 implements
`doctor`; the remaining commands are registered with their milestone so the
interface is stable from day one.

| Command    | Milestone | Purpose                                             |
| ---------- | --------- | --------------------------------------------------- |
| `doctor`   | M0        | environment diagnostics (distro/CPU/RAM/GPU/OCR/…)  |
| `init`     | M1        | create/verify state directory + database            |
| `scan`     | M1        | discover PDFs/EPUBs, register source revisions      |
| `ingest`   | M1–M4     | extraction → OCR → chunk → embed → publish pipeline |
| `pause`    | M1        | persist pause; stop new job claims                  |
| `resume`   | M1        | clear a persisted pause                             |
| `status`   | M1        | show ingestion/pipeline status + counts             |
| `retry`    | M1        | requeue retryable-failed jobs                       |
| `search`   | M4        | lexical + semantic search over the library          |
| `reconcile`| M1        | reconcile durable outputs and index/DB state        |
| `serve`    | M5        | FastAPI research app + reader                       |
| `evaluate` | M6        | retrieval/answer evaluation on a labeled set        |
| `backup`   | M7        | consistent backup of DB, Qdrant, manifests          |
| `restore`  | M7        | restore a backup into an isolated directory         |
| `verify`   | M7        | verify source links + search after restore          |

## Configuration

Copy `config.example.yaml` to `config.yaml` (or point `LIBRARY_RAG_CONFIG` at a
file) and fill in real paths. Paths are validated at load time: managed roots
must be mutually disjoint and disjoint from read-only source roots. Secrets go
in `.env` (see `.env.example`), primarily `LIBRARY_RAG_API_TOKEN` before any
non-loopback exposure.

## Tests

```sh
uv run ruff check .     # lint
uv run mypy             # type check (strict)
uv run pytest           # unit tests
```

## Services

`compose.yaml` runs the app with Qdrant (and an optional `llama` answer server).
GPU devices are selected by UUID in the app config, not by CUDA ordinal.
