# library-rag

custom built (100% local claude + qwen 3.8 on 2 3090ies) RAG for my home library, you can run it on your books collection on linux box with direct accees to your books file tree and 
local llm (i used vllm / llama-server) with 2 models running simultaneously - embedder and decent llm for summaries and querying Qdrant. 
or just ask claude to adapt it to your config. 

more details in "doc"

---

Local-first RAG research application for a large personal book library
(~35,000 books, PDF + EPUB, read-only source mount). A detached ingestion
worker turns books into chunks and vectors; a FastAPI service with a web UI
answers questions over the library with page-level citations, keeps per-book
resumes, and browses the source filesystem.

Everything runs on a single GPU workstation with local models — no cloud
services.

## Where the documentation lives

| Path | What it is |
|---|---|
| `README.md` | this document — the current state of the project |
| `doc/RUNBOOK.md` | operator runbook for full-library ingestion (launch, stop, recovery, triage) |
| `doc/README.md` | the original M0-era README (frozen development artifact) |
| `doc/project/CLAUDE_PRD.md` | the original implementation PRD (milestones, contracts, gates) |
| `doc/project/PROGRESS.md` | the milestone-by-milestone development log (M0–M11 + session addenda) |
| `doc/project/M6_PILOT_REPORT.md` | measured pilot report (rates, capacity, tuning findings) — APPROVED |

Files under `doc/project/` are frozen development records — read them for
provenance and measured numbers, but this document is authoritative for the
current state.

## Architecture

```
                        /mnt/models_sas_ssd/books   (read-only source root)
                                        │  scan / discover (sentinel-gated, fast check)
                                        ▼
   ┌───────────────────────── ingest worker (detached, one at a time) ─────────────────────────┐
   │  extract (PDF text layer → tesseract OCR fallback; EPUB)                                  │
   │    → normalize/sanitize → chunk (256-word budget / 40 overlap)                            │
   │    → embed (bge-m3 1024-d, 8-way fleet, per-batch commit) → publish                       │
   └───────────────┬──────────────────────────────────────────────┬───────────────────────────┘
                   ▼                                              ▼
        SQLite: jobs, catalog,       Qdrant (embedded, file mode):
        meta, OCR, versions           dense + sparse (BM25) points
                   │                                              │
                   └──────────────  serve  (FastAPI, 0.0.0.0:8100) + web UI
                                        │  hybrid retrieval → cited context → LLM
                                        ▼
        answer models: vLLM qwen3.8-27b FP8 :8091 (weight 2) + ak:8080 (weight 1)
        embedding fleet: 8× llama.cpp bge-m3 Q8_0, 127.0.0.1:8081–8088
```

- **serve and the worker can never run at the same time**: embedded Qdrant
  holds an exclusive folder lock for the lifetime of its client.
  `serve.sh start` refuses to launch while a worker is up.
- The model fleet is **operator-managed and auto-restarts at boot**; the
  app only talks to it over HTTP.

## Components

### CLI — `library-rag`

Installed by `uv sync`; every subcommand takes `--config` (default
`$LIBRARY_RAG_CONFIG`, else `./config.yaml`).

| Group | Commands |
|---|---|
| Environment | `doctor` (read-only diagnostics, `--json`), `init` |
| Ingestion control | `scan`, `ingest` (worker; `--force` recovery mode), `discover` (continuous re-scan, detached), `pause`, `resume`, `stop`, `status`, `retry` |
| Library maintenance | `reconcile`, `coverage`, `gc`, `remove`, `migrate` (chunker/embedding-generation planner), `backup`, `restore`, `verify` |
| Research | `search` (hybrid), `answer` (cited, multi-endpoint), `resumes` (per-book summaries), `evaluate` (labeled-set evaluation) |
| App | `serve` (FastAPI + web UI) |
| Pilot tooling | `pilot {survey, sample, run, latency, report, annotate}` |

### Serve app + web UI

FastAPI on `0.0.0.0:8100` (`services.app_port`). Web bundle is built into
`web/dist` (TypeScript + Vite, single JS bundle; PDF.js reader).

Views:

- **Research** — question box; hybrid retrieval, cited answers, per-answer
  history (M9 answer log), long answers continue via resumes.
- **Summaries** — the per-book resume (extended summary) list (M8).
- **Browse** — read-only filesystem browse of the source root with a
  built-in PDF/EPUB reader (M10).
- **Rag** — ingestion status cards + the M11 **Diagnostics** block: catalog
  stats, on-disk book totals, serve/job logs, endpoint health, GPU (nvidia-smi)
  and CPU/RAM, and a shutdown button.

API surface (all under `http://<host>:8100`):

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /ready` | liveness; readiness (Qdrant / embedding / answer / browse probes) |
| `POST /search`, `POST /answer` | hybrid search; cited answering (weighted round-robin over configured LLM endpoints) |
| `GET /resumes`, `POST /resumes` | per-book summaries |
| browse endpoints | filesystem listing + raw book fetch for the reader |
| `GET /system/status` | endpoints + GPU + CPU status (the JSON behind Diagnostics and `serve.sh status`) |
| `GET /system/catalog`, logs, job logs | catalog stats and log access (M11) |
| `POST /system/shutdown` | graceful self-shutdown (SIGTERM via BackgroundTask) |

Auth: optional bearer token (`services.require_api_token` +
`LIBRARY_RAG_API_TOKEN`). The live pilot runs with `require_api_token: false`
on a private network; enable the token before any non-loopback exposure.

### Ingest worker

`library-rag ingest` runs detached (nohup). Durable execution contract:

- jobs live in SQLite (WAL) with **leases** (300 s TTL), **fenced commits**,
  and **pause** persisted in `meta`;
- a worker writes its own pidfile and runs a **reconcile pass on start**, so
  relaunching after any interruption is safe; committed units are never
  reprocessed (OCR per page, chunking by fingerprint, embedding per batch);
- a verbose per-job log is kept on disk **only if the job failed**;
- after a code upgrade the **version gate** refuses jobs stamped with an
  older software version until `--allow-version-mismatch` is confirmed
  (the ack is durable, per upgrade).

### Model fleet (operator-managed)

| Role | Model / runtime | Endpoints | Notes |
|---|---|---|---|
| Embedding | bge-m3 Q8_0 GGUF, llama.cpp | 127.0.0.1:8081–8088 (8 servers) | 1024-d, 8192 context; round-robin with failover on transient errors; launch via `models/embedder.sh` |
| Answer (primary) | qwen3.8-27b FP8, vLLM | 127.0.0.1:8091 | weight 2; docker container (internal :8000); launch via `models/qwen_on_lx.sh` |
| Answer (secondary) | (operator-managed remote) | ak:8080 | weight 1; dual-endpoint split (M9); restarted server-side only — never from the client |

The fleet auto-restarts at boot. The app treats every endpoint as
replaceable: health probes in `/ready` and `/system/status`, failover in the
answer pool.

## Data flow

### Ingestion (per book)

1. **scan/discover** — sentinel-gated walk of the source root (a zero-byte
   sentinel marks the mount available); size+mtime fast check makes
   re-scans of unchanged books nearly free.
2. **archive** — SHA-256 content store (the durable copy).
3. **extract** — PDF text layer when usable, tesseract OCR otherwise (one
   job per page); EPUB parsed natively; text sanitized/normalized.
4. **chunk** — 256-word budget, 40-word overlap (the word budget keeps each
   chunk under the embedder's per-input BPE ceiling; see the measured
   margins in `config.example.yaml`).
5. **embed** — 1024-d dense vectors across the 8-server fleet; vectors are
   written to the artifact store before the batch row commits.
6. **publish** — dense + sparse (BM25) points to Qdrant, catalog row,
   version record. `retrieval.stats_epoch: frozen` pins the BM25 corpus
   stats per campaign so publishes don't fan out O(N²) republishes (the M6
   pilot's 15 h publish tail, measured).

### Search & answer

Query → hybrid retrieval (BM25 + dense) → context assembled with page-level
source references → LLM (weighted round-robin over configured endpoints,
`max_tokens`/temperature per config) → answer with citations. Long answers
overflow into per-book **resumes** (continuations up to `resume.max_tokens`,
8192 in the live config).

## Storage layout (live pilot)

| Purpose | Path |
|---|---|
| Source books (read-only) | `/mnt/models_sas_ssd/books` |
| Source mount sentinel | `/mnt/models_sas_ssd/books/.library-rag-sentinel` |
| Archive (SHA-256 content store) | `/mnt/models_sas_ssd/library-rag/pilot-sandbox/archive` |
| Artifacts (vectors, OCR images) | `/mnt/models_sas_ssd/library-rag/pilot-sandbox/artifacts` |
| State (SQLite `library.db`, job logs) | `/mnt/models_sas_ssd/library-rag/pilot-sandbox/state` |
| Qdrant (embedded) | `/mnt/models_sas_ssd/library-rag/pilot-sandbox/qdrant` |
| Models | `/mnt/models_sas_ssd/library-rag/models` |
| Scratch (hard-capped) | `/mnt/models_sas_ssd/library-rag/pilot-sandbox/scratch` |

Full-run backups (`/mnt/models_sas_ssd/library-rag/backups`, see
`doc/RUNBOOK.md`) live on the same hardware as the data they protect — they
recover accidental loss, not disk failure.

## Configuration

`config.yaml` (or `$LIBRARY_RAG_CONFIG`); `config.example.yaml` documents
every key. Secrets via `.env` (`.env.example`), primarily
`LIBRARY_RAG_API_TOKEN`. When serve runs, `<config dir>/serve.env` is sourced
for the token environment (0600, values never echoed).

Key sections:

- `paths` — source roots (read-only) and managed roots (mutually disjoint,
  validated at load).
- `services` — app host/port (0.0.0.0:8100), answer endpoints
  (`endpoints` + `extra_endpoints`, each with weight/model/max_tokens/
  temperature/timeout), embedding endpoint(s), `require_api_token`,
  `show_path_to_original`.
- `embedding` — exact model + revision (changes the embedding key).
- `chunking` — word budget / overlap.
- `retrieval` — counts and `stats_epoch` (`live` vs `frozen`).
- `worker` — `max_concurrent_jobs` (2 in the live config), lease TTL.
- `browse` — enabled + root.
- `file_types` — top-level allow-list (`.pdf`, `.epub` live).
- `resume` — `max_tokens` (8192 live).

**Live pilot config**: `/mnt/models_sas_ssd/library-rag/pilot-sandbox/scratch/config.sandbox.yaml`
(sandbox state roots under `/mnt/models_sas_ssd/library-rag/pilot-sandbox/`).

## Operations

**Serve control** — `./serve.sh` at the repo root:

```sh
./serve.sh start     # detached serve; waits for /health; writes serve.pid
./serve.sh stop      # graceful POST /system/shutdown; SIGTERM/SIGKILL fallbacks
./serve.sh status    # configured paths + serve + worker + fleet (via /system/status when up)
./serve.sh ingest    # detached ingest worker (log: worker.log); kill -TERM <pid> to stop
./serve.sh ingest N  # bounded run: at most N new books end-to-end, then exits on its own
```

`serve.sh` owns `serve.pid`, starts/stops serve, and can start the ingest
worker; stopping the worker is `kill -TERM <pid>` (graceful) or
`library-rag stop`. The model fleet is reported, not managed. It refuses to
start while an ingest worker is running — embedded Qdrant is single-process —
and so does `ingest`.

**Full-library launch** — currently **HELD on explicit operator approval**.
The approved procedure lives in `doc/RUNBOOK.md` (M7 gate deliverable); it
covers:

- **Pre-flight**: `doctor`, health-check of all nine model endpoints, drive
  capacity.
- **Launch sequence**: `library-rag scan` (discover/register/enqueue — a
  re-scan is nearly free) → start the worker detached → verify with
  `status`.
- **Monitoring**: `status` plus periodic `coverage` (the full funnel); failed
  jobs keep their verbose log under `state/job_logs/`.
- **Stopping**: three levels — pause (process keeps running), graceful
  SIGTERM via `library-rag stop`, hard kill / power loss (lease expiry +
  idempotent replay).
- **Crash recovery and the version gate**: a restart is safe after any
  interruption; after a code upgrade, jobs stamped with an older software
  version need a one-time `--allow-version-mismatch` ack.
- **Failure triage table**, plus what is safe mid-run vs what needs the
  worker stopped (`backup`, `gc`, `remove`, `reconcile`).

Measured pilot rate implies ~41–45 days for the full ~34.4k-book library at
a serial worker; re-measure during the first hour of any real run.

**Post-reboot** — the fleet auto-restarts; serve does not (it is a nohup
process). Restore it with `./serve.sh start` and check `./serve.sh status`.

**Quality gates** — run before any commit that touches code:

```sh
uv run ruff check .    # lint
uv run mypy            # strict; 12 pre-existing test_browse errors are known
uv run pytest          # ~590 tests
```

## Development history

| Milestone | Scope |
|---|---|
| M0 | Environment, skeleton, `doctor` |
| M1 | Catalog + recovery foundation (jobs, leases, pause/stop, reconcile) |
| M2 | Extraction and source reader (PDF/EPUB) |
| M3 | Selective OCR and chunking |
| M4 | Index and search (Qdrant, hybrid retrieval) |
| M5 | Cited answering and UI |
| M6 | Pilot and capacity report (measured; APPROVED) |
| M7 | Full-run operations and maintenance (runbook, backup/restore/verify, gc, remove, migrate) |
| M8 | Extended per-book resumes + keyword search |
| M9 | Second LLM endpoint (dual-endpoint split) + worker concurrency |
| M10 | Filesystem browse tab (with in-app reader) |
| M11 | Rag-page Diagnostics block (system/status, logs, GPU/CPU, shutdown) |

Current pilot state (as of 2026-09-25): 344 books in the catalog / 335
processed / 54,270 chunks; 34,768 books on disk under the source root.
Full milestone log with commands and measured results:
`doc/project/PROGRESS.md`.

## Repository layout

```
src/library_rag/     Python package (pipeline, API, CLI)
web/                 TypeScript + Vite frontend (dist/ is committed)
models/              operator launch scripts for the model fleet
tests/               pytest suite
serve.sh             operator control script (serve start/stop/status, ingest [N])
compose.yaml         containerized deployment (Qdrant server + optional llama)
config.example.yaml  fully-commented configuration template
doc/                 project docs: RUNBOOK.md + the M0-era README
doc/project/         frozen development records (PRD, progress log, pilot report)
```
