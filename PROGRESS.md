# Progress Log

Milestone-by-milestone record of what has been delivered, the tests that were
actually run, and the next unfinished task. Updated at the end of each work
session (PRD §15).

## M0 — Environment, skeleton, `doctor`

**Status: gate passed** (ruff + mypy + pytest green; `doctor` handles no-GPU
and missing tesseract cleanly).

### Delivered
- Python project scaffold: `pyproject.toml`, hatchling build, `uv` lockfile.
- Package layout under `src/library_rag/`:
  - `config.py` — Pydantic config, path-overlap validation, env precedence.
  - `log.py` — structured logging (human/json) to stderr.
  - `doctor.py` — read-only environment diagnostics, fully defensive.
  - `cli.py` — argparse CLI; `doctor` implemented, all other commands
    registered as milestone-stamped stubs.
  - `__main__.py`, `__init__.py`.
- `config.example.yaml`, `.env.example`, `.gitignore`, `README.md`.
- `Dockerfile` (runtime image), `compose.yaml` (app + qdrant + optional llama,
  GPU overlay via `gpu` profile).
- `.github/workflows/ci.yml` (ruff + mypy + pytest).
- Tests: `tests/conftest.py`, `tests/test_config.py`, `tests/test_doctor.py`.

### Test commands + results
| Command                   | Result |
| ------------------------- | ------ |
| `uv run ruff check .`     | All checks passed! |
| `uv run mypy`             | Success: no issues found in 9 source files |
| `uv run pytest`           | 16 passed |

`uv run library-rag doctor` — runs cleanly on this host (text + `--json`):
3 GPUs detected, Tesseract reported unavailable without failing.

### Notes
- `uv` installed user-local at `~/.local/bin` (no apt/system packages).
- tesseract not installed (needs approval) — `doctor` reports OCR as
  unavailable without failing.
- GPU quirk encoded in config: assignments by UUID, not CUDA ordinal.
- `doctor` subparser registers `--config`; a missing/invalid config falls back
  to the built-in default rather than crashing.
- Minor: `detect_cpu` reports physical cores as 1 on this host (the /proc/cpuinfo
  physical-id parse falls back); cosmetic, non-gating — tracked for M0 polish.

### Next unfinished task
- Begin M1: catalog + identities, source archive (content-addressed),
  job table with SQLite-claimed leases + token fencing, artifact commit
  protocol (atomic rename + fsync), pause/resume, reconciliation.
