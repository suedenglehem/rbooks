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

## M1 — Catalog + recovery foundation

**Status: gate passed.** PRD §15 M1 gate, verbatim: "duplicate/rename/change
tests pass; killed worker replay produces one logical output; stale worker
cannot commit; missing mount cannot delete content." All four groups pass in
`tests/test_catalog.py`, `tests/test_jobs.py`, and `tests/test_reconcile.py`.

### Delivered
- `identity.py` — content-anchored identity: `normalize_path`, `document_id`
  (uuid5 of the anchor SHA-256), `revision_id` (uuid5 of `doc:sha`),
  `make_task_key` (deterministic job identity: stage + input + config + range).
- `migrations.py` + `db.py` — versioned, idempotent schema runner; each
  migration applied in one short `BEGIN IMMEDIATE` transaction and recorded in
  `schema_migrations`. Durable PRAGMAs (WAL, `foreign_keys=ON`,
  `busy_timeout=5000`, `synchronous=FULL`). `Database.transaction()` brackets
  the commit with the before/after crash hooks: a before-commit failure rolls
  back; an after-commit failure leaves data durable.
- `catalog.py` — `register_source` resolution rules (exact duplicate → alias,
  rename → same revision/no re-embed, content change → new active revision,
  distinct editions never collapsed) plus read helpers.
- `archive.py` — content-addressed store `<root>/<2-hex>/<sha256>`; streamed
  SHA-256 copy (never a hardlink), checksum-verified, dedup no-op, atomic
  visibility via `commit_stream`; `verify_archived`.
- `artifacts.py` — atomic artifact commit: temp file in destination dir →
  write → flush → fsync → before-rename hook → atomic `os.rename` →
  after-rename hook → dir fsync.
- `crash.py` — process-global crash-injection hooks, empty in production;
  tests arm them to model death at each boundary.
- `jobs.py` — durable job queue: atomic claim (reclaims expired leases first),
  uuid4 lease fencing token embedded in every state-changing write
  (`heartbeat`/`succeed`/`fail` reject stale tokens with `StaleLeaseError`),
  pause/resume (no new claims while paused, active jobs finish), exponential
  backoff with jitter, retry (optionally requeueing permanent failures).
- `reconcile.py` — mount-aware reconciliation: a mount whose sentinel is
  missing is reported unavailable and its aliases are **not** pruned; a present
  mount prunes stale path aliases but never deletes documents/revisions or
  touches the archive.
- `cli.py` — `init`, `pause`, `resume`, `status` (text + `--json`), `retry`
  (`--include-permanent`), `reconcile` (`--json`) implemented; scan/ingest/
  search/serve/evaluate/backup/restore/verify remain milestone-stamped stubs.
- Tests: `test_identity.py`, `test_migrations.py`, `test_catalog.py`,
  `test_archive.py` (incl. crash-at-rename cases), `test_jobs.py`,
  `test_crash_db.py`, `test_reconcile.py`.

### Test commands + results
| Command               | Result                                    |
| --------------------- | ----------------------------------------- |
| `uv run ruff check .` | All checks passed!                        |
| `uv run mypy`         | Success: no issues found in 25 source files |
| `uv run pytest`       | 52 passed in 1.60s                        |

Gate cases (all passing):
- duplicate/rename/change → `test_catalog.py` (7 tests incl. the vertical
  slice: hash → archive → register → rename leaves archive untouched).
- killed worker replay → `test_killed_worker_replay_produces_one_logical_output`
  (at-least-once work, exactly one artifact on disk).
- stale worker cannot commit → `test_stale_worker_cannot_commit_after_reclaim`
  + `test_stale_token_rejected_even_mid_flight` (succeed/heartbeat/fail all
  fenced).
- missing mount cannot delete content → `test_missing_mount_cannot_delete_content`
  (catalog entry and archived original both survive a disconnected mount).

CLI smoke (temp dir, real commands): `init` → `status` → `pause` →
`status --json` → `resume` → `reconcile --json` all exit 0 with correct output.

### Notes / bugs found
- `Database.transaction()` was missing its `yield` (a `@contextmanager` that
  never yields commits before the body runs and never rolls back); fixed —
  this would have silently broken the crash-recovery guarantees.
- The M1 schema DDL is multi-statement; `sqlite3` `execute()` accepts one
  statement at a time, so `Migration` now carries `statements: tuple[str, ...]`
  applied in order inside the migration transaction.
- All fixes are strict-typing clean: mypy strict over `src` + `tests`.

### Next unfinished task
- Begin M2: scan (walk source roots, hash, archive, `register_source`,
  mount-sentinel verification on startup), the extraction stage for PDF
  (PyMuPDF) and EPUB (EbookLib) producing durable extraction artifacts, and
  the worker loop that claims jobs and runs the pipeline stage.

## M2 — Extraction and source reader

**Status: gate passed.** PRD §12 M2 gate, verbatim: "citations open correct
physical pages/sections on fixtures; no model required; path traversal/XSS/
ZIP-bomb limits tested." All three groups pass in `tests/test_reader.py`,
`tests/test_extraction_pdf.py`, `tests/test_extraction_epub.py`, and
`tests/test_worker.py`; no model or index is required anywhere.

### Delivered
- `scan.py` — walks the configured source roots for PDF/EPUB, verifies mount
  sentinels, hashes + archives each file, `register_source`, and enqueues an
  `extract` job per new revision (durable task key, dedup no-op).
- `extraction/` package:
  - `pdf.py` — PyMuPDF, one unit per physical page; zero-based position,
    one-based `ref`; `char_count` = non-whitespace chars; quality flags
    `no_text` / `sparse` (below `pdf.sparse_chars`, default 40) /
    `many_replacement`; page geometry (rotation/width/height) + text blocks in
    the artifact.
  - `epub.py` — spine-ordered section units with deterministic paragraph
    anchors (`a0000`, `a0001`, … in document order; the `<h1>` title is
    anchored first, title stored in the artifact payload, not the DB);
    `char_count` = sum of anchored paragraph lengths; flags `bad_html`
    (unparseable markup: nothing kept) and `no_text`.
  - `sanitize.py` — strict-XML XHTML sanitization: strips forbidden tags
    (script/style/link/meta/iframe/object/embed/form/base/template), `on*`
    attributes, and external/protocol-relative URIs in href/src/action/
    poster/data; keeps in-document `#fragment` links (anchors need them);
    raises `ET.ParseError` for unparseable input so callers classify.
  - `store.py` — `insert_unit` idempotent upsert + artifact commit (verified
    checksum, committed before the DB row).
  - `errors.py` — `ExtractionFailure` taxonomy; permanent categories
    `{encrypted, corrupt, invalid_source, missing_source}`.
  - `check_epub_safety` — archive-safety limits: not-a-zip, max entries, max
    uncompressed bytes, compression ratio (zip bomb), absolute paths, `..`
    traversal → all `invalid_source` (permanent).
- `worker.py` — claim/run loop: one job at a time, lease heartbeated between
  units (PDF every 10 pages, EPUB every 20 sections) via the extractor's
  `on_progress`; `StaleLeaseError` → drop the job and touch nothing;
  `ExtractionFailure` → `fail` with `transient=not is_permanent(...)` +
  `fail_run`; graceful SIGTERM (`stop_event`, in-flight work is durable and
  resumable); unknown stage → permanent failure (counted as terminal).
- `reader.py` — FastAPI app: revision manifest (`GET /books/{rev}`), ranged
  original delivery (`GET /books/{rev}/source`, 206/416/410), verified unit
  artifacts, and `pdfjs_page_for` (zero-based → one-based PDF.js page); EPUB
  units expose `(section, paragraph anchor)` — never a fabricated page.
- `config.py` — `ExtractionSettings` with `settings_sha()` (PDF/EPUB limits
  snapshotted into the deterministic run key).
- `migrations.py` — `extraction_runs` (state, unit_count, parser version,
  settings sha, failure info) and `source_units` (UNIQUE(run_id, kind,
  position), artifact relpath + sha256).
- `cli.py` — `scan` (discover/register/enqueue, `--json`) and `ingest`
  (worker loop, `--once`, `--lease-ttl`) implemented; `serve` stays an M5
  stub (the reader app is testable via `create_app` now).
- Tests: `test_extraction_pdf.py`, `test_extraction_epub.py`, `test_worker.py`,
  `test_scan.py`, `test_reader.py`, plus `tests/fixtures.py` (real PyMuPDF
  PDFs / ebooklib EPUBs built exactly as production consumes them, plus a
  raw-ZIP builder for the malicious-archive cases).

### Test commands + results
| Command               | Result                                     |
| --------------------- | ------------------------------------------ |
| `uv run ruff check .` | All checks passed!                         |
| `uv run mypy`         | Success: no issues found in 40 source files |
| `uv run pytest`       | 97 passed in 6.56s                         |

Gate cases (all passing):
- citations open correct physical pages → `test_reader.py`
  (`test_pdfjs_page_mapping`, `test_unit_page_carrys_pdfjs_page`, ranged
  original delivery 206/416, missing archive 410) + `test_extraction_pdf.py`
  (one unit per page, refs `"1","2","3"`, geometry in artifact).
- citations open correct sections → `test_reader.py`
  (`test_epub_section_unit_and_anchor`: anchor `a0001` → "Hello world.",
  unknown anchor 404, anchor endpoint on a page unit 409) +
  `test_extraction_epub.py` (deterministic anchors, title in artifact).
- path traversal / XSS / ZIP-bomb limits → `test_extraction_epub.py`
  (`test_epub_safety_rejects_zip_bomb`, `..._path_traversal` for `../` and
  absolute, `..._too_many_entries`, `..._too_large`, `..._not_a_zip`,
  `test_sanitize_strips_active_content` for script/onclick/javascript:/
  external-URI stripping) + `test_scan.py` (unscannable roots, sentinel
  behavior).
- no model required → the whole suite runs on PyMuPDF/ebooklib/SQLite only;
  no embedding/index dependency anywhere in the M2 path.

### Notes / bugs found
- **Production bug fixed in gate run:** `extraction/epub.py` fed the sanitizer
  `item.get_content()`, but ebooklib's `EpubHtml.get_content()` re-parses the
  chapter with lxml's lenient HTML parser and re-serializes it from a
  template — repairing malformed markup before the security layer could see
  it, making `bad_html` unreachable. The fix passes the raw archive bytes
  (`item.content` after `read_epub` is the verbatim zip entry) to `sanitize`.
- ebooklib prefixes archive entries with the OPF directory (`EPUB/ch1.xhtml`,
  not `ch1.xhtml`), and normalizes chapter content on *both* read and write —
  the malformed-chapter fixture must patch raw ZIP bytes after `write_epub`.
  Note: `write_epub` itself crashes (`Document is empty`) on near-empty
  chapter bodies, so the fixture gives every chapter a structurally valid
  full-HTML body before patching.
- MuPDF clips glyphs beyond the page edge out of `get_text`, so the exact
  `char_count` fixture is a single 48-char 11pt A4 line (no wrapping, no
  clipping).
- Worker: jobs are claimed unfiltered by stage, so unknown-stage jobs are
  failed as permanent `unknown_stage` and count toward `run_worker`'s
  terminal-state return.

### Next unfinished task
- Begin M3: selective OCR and chunking (OCR for scanned/image-only PDF pages
  flagged by extraction quality, chunking over the extracted units with the
  page/section anchors preserved for citations).
