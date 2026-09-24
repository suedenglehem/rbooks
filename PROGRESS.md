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

### ePUB "Open book" addendum (2026-09-22 evening)
- Operator report: "'open book' works with pdf, and doesn't
  work on epub". Root cause: `openReaderFor` opened
  `m.units[0]`, and for ALL 82 pilot ePUBs position 0 is the
  cover XHTML section — char_count 0, payload paragraphs [],
  html a bare `<img>` whose archive-relative src the browser
  cannot resolve → blank pane (live-confirmed on an ePUB test
  rev via `GET /books/{rev}`). DB proof of the asymmetry:
  zero-text-first units are epub 82/82 vs pdf 9/210; PDFs were
  unaffected because pages render from the source PDF even
  without a text layer. Fix (8378a45): open the first unit
  that is a page or carries text (`kind === "page" ||
  char_count > 0`), falling back to `units[0]` if no unit has
  text — only textless sections are skipped, so a textless-
  cover PDF still opens page 1. Live check after the rebuild:
  same ePUB rev, old pick `cover.html` (0 chars) → new pick
  `frontmatter.html` (1621 chars). Bundle index-DdRxYc1o.js;
  the operator must hard-refresh the browser (Ctrl/Cmd+Shift+R)
  — serve reads web/dist from disk, no restart. Gate: 507
  passed, ruff + mypy clean.

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

## M3 — Selective OCR and chunking

**Status: gate passed.** PRD §12 M3 gate, verbatim: "kill after page N resumes
from unfinished units; rotated OCR highlight mapping tested; no tokenizer
truncation; originals unchanged." All four groups pass, verified by running the
named tests this session.

### Delivered
- `extraction/routing.py` — page-route decision matrix: `assess_page` quality
  flags (already in M2's pdf.py) + `image_area_ratio` (embedded-image coverage)
  → `route_page` returns `ocr` / `reuse` / `skip`. Broken text layer is always
  worth re-reading; clean text is reused no matter the image coverage;
  no/sparse text is OCR'd only when an embedded image plausibly *is* the page
  (threshold `OcrSettings.image_area_threshold`, default 0.5).
- `extraction/ocr.py` — Tesseract TSV parsing (word boxes, `conf`, page/line/
  word ids; malformed rows dropped), and the render→OCR→map coordinate
  round-trip: rendered-page px ↔ page points for any rotation (0/90/180/270),
  crop, and downscale to `max_side_px`. `ocr_page` runs the binary, classifies
  failures (missing binary → permanent `ocr_unavailable`; timeout/non-zero/
  empty → retryable), and commits the word layer into the unit artifact under
  `ocr` (text, words with bboxes in **page points**).
- `normalization.py` — cross-unit header/footer removal (exact repeated lines,
  length-bounded, needs ≥2 units, disable-able), dehyphenation of line-end
  fragments (never across paragraphs or capitals), whitespace collapse — each
  with span round-trip mapping so surviving text still points at original
  character ranges. `settings_sha()`.
- `chunking.py` — word tokenizer (maximal non-whitespace runs), budget/overlap
  (`target_tokens`/`overlap_tokens`/`max_tokens`), cut only between tokens
  (no truncation), lookback that prefers line-end/unit-boundary cuts, maximal
  same-unit span runs with `_union_word_boxes` bboxes for OCR units, title
  prefix (first chunk only, consumes budget, stored in its own column, excluded
  from chunk text/spans), deterministic `chunk_id`. `settings_sha()`.
- `worker.py` — `ocr` and `chunk` stage handlers: OCR renders the page, runs
  tesseract, commits the artifact, marks `ocr_state=done`, cleans scratch;
  chunk normalizes all units, recomputes the fingerprint, chunks, and inserts
  with deterministic ids in one transaction. Worker-start reconciliation
  re-enqueues chunk jobs for runs whose chunk rows/fingerprint were lost.
- `identity.py` — `chunk_key`; `migrations.py` — `chunks` table +
  `source_units.route`/`ocr_state` + `extraction_runs.chunk_fingerprint`;
  `config.py` — `OcrSettings`, `ChunkingSettings` (both into
  `ExtractionSettings.settings_sha()`).
- `tests/fixtures.py` — `make_mixed_pdf` (text/scanned/blank/sparse/
  sparse_image page shapes via real PyMuPDF) and `make_fake_tesseract` (an
  env-steered shim emitting 3 words per page; PRD §1: no system installs).
- Tests: `test_routing.py`, `test_ocr.py`, `test_normalization.py`,
  `test_chunking.py`, `test_pipeline_m3.py` (full extract→OCR→chunk pipeline
  over real mixed PDF and real EPUB).

### Test commands + results
| Command | Result |
| ------- | ------ |
| `uv run ruff check .` | All checks passed! |
| `uv run mypy` | Success: no issues found in 49 source files |
| `uv run pytest` | 148 passed, 2 warnings in 16.49s |
| `uv run pytest tests/test_normalization.py tests/test_routing.py tests/test_chunking.py tests/test_worker.py tests/test_ocr.py tests/test_pipeline_m3.py --tb=short` | 56 passed in 10.52s |

(The 2 warnings are a third-party starlette testclient/anyio
DeprecationWarning, not from this codebase.)

Gate cases (all passing, run this session):
- kill after page N resumes from unfinished units →
  `test_ocr.py::test_ocr_kill_after_page_n_resumes`: 5-page scanned PDF,
  stop signal observed after OCR page 1 commits → first run handles 3 (extract
  + 2 OCR), resumed run handles 4 (3 OCR + chunk), 7 succeeded total; exactly
  5 tesseract calls (pages 0–4, no re-OCR of finished pages); one chunk, 5
  spans, each with bbox `[12.0, 24.0, 165.6, 36.0]`; no scratch files survive.
- rotated OCR highlight mapping tested →
  `test_ocr.py::test_coordinate_roundtrip_rotated[90|180|270]` (plus straight
  and cropped round-trips) mapping TSV word boxes back to page points, and the
  pipeline tests asserting span bboxes end up in page coordinates.
- no tokenizer truncation → `test_chunking.py::test_no_token_is_truncated`
  (cuts land only between whole tokens; every token of every unit appears in
  exactly the expected chunk).
- originals unchanged → `test_pipeline_m3.py::test_mixed_pdf_pipeline`: source
  file and its content-addressed archive copy are bit-for-bit unchanged after
  the full pipeline, and the run's scratch dir is empty.

### Notes / bugs found
- **Chunk-job key idempotency fix (`worker.py::_rekey_chunk_job`):** the chunk
  job's `task_key` embeds the fingerprint of the run's units, computed at
  *enqueue* time (pre-OCR). OCR then rewrites unit artifacts with new shas, so
  the post-OCR fingerprint differs; on a later re-extract replay,
  `on_extract_done` recomputed the post-OCR fp and `INSERT OR IGNORE` missed
  the settled row, inserting a fresh duplicate chunk job — unbounded job-row
  growth per replay of any OCR'd book (`test_reextract_is_a_job_noop` failed
  with 5 jobs instead of 4). Fix: in all three `_run_chunk` success paths the
  settled job row is re-keyed to the fingerprint of the state it actually
  chunked, guarded by `NOT EXISTS` so `UNIQUE(task_key)` can never be
  violated. Replays now hit the settled row; settings changes still produce a
  new fp → new key → new job; reconcile-after-lost-row still works.
- Test-expectation corrections (product behavior was correct, M2-pinned):
  native PDF pages store PyMuPDF's raw text layer verbatim, which keeps the
  per-line trailing newline (`text == sentence + "\n"`); EPUB `<h1>` headings
  are block-level paragraphs, so their text is *in* the chunk body and the
  promoted title is only the extra `title` column (test EPUB chunk: 13 body +
  2 title = 15 tokens).
- Deploy note (PRD §6): `OcrSettings` and normalization are now part of
  `ExtractionSettings.settings_sha()`, so the first M3 deploy re-keys every
  prior run (one re-extract/re-chunk cycle). Expected and harmless.

### Next unfinished task
- Begin M4: embeddings + Qdrant indexing (model wiring behind the service
  config, upsert chunks by deterministic id, delete-on-revision, and the
  evaluate harness per PRD §9).

## M4 — Index and search

**Status: gate passed.** PRD §12 M4 gate, verbatim: "crash at each
publication boundary never returns uncommitted or obsolete evidence;
re-upsert is idempotent; search works after restart; sparse degraded mode
works." All four clauses verified by running the named tests this session.

### Delivered
- `embeddings.py` — `Embedder` protocol (`encode_documents` /
  `encode_query`); `FakeEmbedder` (deterministic, hash-derived, revision
  label `fake-v1` — allowed for tests, refused by `make_embedder` when
  production config is otherwise active); `LlamaCppEmbedder` (real model
  adapter, gated behind `EmbeddingSettings.model_revision`); `embedding_sha`
  (canonical encoding key from model + dims + dtype + normalize + settings);
  per-batch durable checkpoints (`write_checkpoint` / `read_checkpoint`,
  magic `LBEM`, payload-hash verification → corrupt checkpoint re-encodes);
  `encode_batch_oom` (bounded halving loop, `oom_max_halvings`); BM25 side —
  `tokenize` / `term_id` / `bm25_weights` (client-side sparse vectors) and
  `SparseStats` + `compute_sparse_stats` / `load_sparse_stats` /
  `stats_sha` / `stats_row_upsert` (corpus statistics per statistics epoch,
  keyed by sha); error taxonomy `EmbeddingError` (non-transient),
  `ModelUnavailableError`, `EmbeddingOOMError`, `CheckpointCorruptError`.
- `indexing.py` — `QdrantOps` protocol + `RealQdrantOps` (thin qdrant-client
  wrapper; collection `library_chunks`) and `FakeQdrant` (in-memory: idempotent
  upsert by point id, `set_active` merges *only* the `active` flag, dense
  search ordered `(-score, point_id)`, sparse search skips zero-overlap,
  unfiltered delete rejected); `FieldCond` / `IndexFilter` (pure-Python
  `matches` evaluator that agrees with the `to_qdrant()` shape for eq/ne/in);
  `build_point` payload contract (ids, gen, pub, model revision, embedding
  sha, stats sha — deliberately no raw text, no `active`); `point_id` =
  `chunk_id:gen_id`; `publish_generation` (deterministic pub/gen ids,
  transaction boundaries B1 staged row → B2 upsert + flag points → B3 flag new
  active / clear superseded → B4 promote + supersede; full point-set
  verification *before* B1; returns `"published"` / `"noop"`);
  `reconcile_publications` (promote staged publication whose points are
  complete and whose job is open; delete orphaned staged; leave incomplete
  alone); publish lock with TTL (`acquire_publish_lock` /
  `release_publish_lock`); `visible_pub_ids` (active publications whose
  revision is active — the SQLite source of truth for search).
- `retrieval.py` — `search(db, cfg, qdrant, embedder, query, *, doc_id, rev_id,
  reranker)`: blank-query rejection before any IO; ping guard →
  `IndexUnavailableError` (never fabricates results); dense + per-epoch BM25
  sparse retrieval fused by RRF (`rrf_k`); **SQLite postvalidation**
  (`_validate`: chunk row must exist, rev_id must match, pub must be visible,
  revision must be active — the point payload is treated as a cache, identity
  is re-materialized from SQLite); degraded sparse-only mode with explicit
  `degraded_reason` when no embedder is configured or query inference raises
  `ModelUnavailableError`; per-book diversification (`min/max_passages`,
  top-up rounds, disabled by doc/rev filters); `Reranker` protocol +
  `PassthroughReranker`; `SearchResult` with the exact counts contract
  {dense, sparse, fused, validated, postvalidation_removed, reranked,
  selected, topup_rounds}.
- `worker.py` — new stages in the pipeline: `embed` (checkpoint resume,
  OOM-halving loop, handoff: version = `emb_sha:stats_sha` → enqueues publish,
  deduped by task key) and `publish` (ping → retryable `qdrant_unavailable`;
  `publish_generation`; no-op manifest for inactive revisions). Failure
  taxonomy: `embedding_oom` / `model_unavailable` retryable, `embedding_error`
  permanent, `embedding_not_configured` permanent (a full drain requires an
  embedder; no fabricated publish). Worker-start `_reconcile_index`: lost
  chunk jobs, publication gaps, and statistics-epoch convergence (a new book's
  stats epoch republishes all active publications against it).
- `identity.py` — `point_id` + publish/embed task keys; `migrations.py` — M4
  tables `embedding_batches`, `index_generations`, `sparse_corpus_stats`,
  `publications` (staged → active → superseded lifecycle, one active
  publication per revision); `config.py` — `EmbeddingSettings`
  (model_revision/fake, dimensions, dtype, normalize, batch_size,
  oom_max_halvings, into `settings_sha()`), `Bm25Settings` (k1, b),
  `RetrievalSettings` (dense_top, sparse_top, rrf_k, rerank_max,
  min/max_passages); `cli.py` — `search` command (clean EXIT_ERROR +
  diagnostic when the index is unreachable; human-readable passages).
- `tests/fixtures.py` — `ingest_and_publish` (full scan→worker drain) and
  `publish_handbuilt` (hand-built chunk rows + points for publication-state
  tests). New test files: `test_embeddings.py`, `test_indexing.py`,
  `test_publication_crash.py`, `test_worker_m4.py`, `test_retrieval.py`,
  `test_cli_search.py` (88 tests total). M3-era pipeline tests
  (`test_worker.py`, `test_routing.py`, `test_ocr.py`,
  `test_pipeline_m3.py`) adapted to the 4-stage pipeline: they inject
  `FakeEmbedder` + `FakeQdrant` and assert extract/chunk/embed/publish counts
  (no-OCR drain = 4 jobs; +2 OCR = 6; +3 = 7; +5 = 9).

### Test commands + results
| Command | Result |
| ------- | ------ |
| `uv run ruff check .` | All checks passed! |
| `uv run mypy` | Success: no issues found in 58 source files |
| `uv run pytest` | 236 passed, 2 warnings in 28.15s |
| `uv run pytest tests/test_embeddings.py tests/test_indexing.py tests/test_publication_crash.py tests/test_worker_m4.py tests/test_retrieval.py tests/test_cli_search.py` | 88 passed in 10.24s |

(The 2 warnings are a third-party starlette testclient/anyio
DeprecationWarning, not from this codebase. 88 = 236 − 148, i.e. every new
test is in the six M4 files.)

Gate cases (all passing, run this session):
- crash at each publication boundary never returns uncommitted or obsolete
  evidence → `test_publication_crash.py::test_crash_at_boundary_never_returns_
  bad_evidence[pre-b1-0 | b1-0 | b2-1 | b3-1 | b4-0]`: a two-revision book,
  crashed at every transaction boundary (trailing number = promotions
  reconcile performs: 0, 0, 1, 1, 0). Pre-reconcile search shows only the
  evidence the boundary allows; after `reconcile_publications` the surviving
  evidence is either fully v1 (pre-b1/b1: the staged v2 publication is deleted
  as an orphan) or fully v2 (b2–b4) — never a mix, never v2 before promotion.
  Plus `test_revision_replacement_supersedes_old_evidence` (an exact-term
  query cannot resurrect superseded points) and
  `test_publish_noop_for_inactive_revision`.
- re-upsert is idempotent → `test_indexing.py::test_fake_qdrant_upsert_is_
  idempotent` and `test_publish_generation_publishes_and_is_idempotent`
  (direct re-run of a generation returns `"noop"`; deleting the publication
  row and republishing recreates the *same* deterministic pub id with the same
  active point set).
- search works after restart → `test_publication_crash.py::
  test_search_works_after_restart`: fresh `Database.connect` on the same file,
  worker drain is a no-op, identical passages.
- sparse degraded mode works → `test_retrieval.py::
  test_degraded_without_embedder_is_sparse_only` (no embedder configured →
  `degraded=True`, reason "embedding model not configured; sparse-only
  search", sparse ranks only) and
  `test_degraded_when_query_inference_is_down` (embedder up but query
  inference raises → degraded, still returns passages).

### Notes / bugs found
- **`_load_run_vectors` id mismatch (source bug, fixed in `indexing.py`):**
  the post-upsert verification compared Qdrant *point* ids against *chunk*
  ids — point ids are `chunk_id:gen_id`, so any run with ≥1 chunk would have
  failed verification. The call site now passes the run's chunk ids and the
  parameter is `expected_chunk_ids`.
- **`retrieval.py::_validate` missing column (source bug, fixed):** the
  postvalidation SELECT took `doc_id` from the `chunks` table, which has no
  such column → `sqlite3.OperationalError` on every search. Fixed by
  `JOIN source_revisions r ON r.rev_id = c.rev_id`.
- **Evidence-gap semantics at pre-b1/b1 (by design, PRD §12):** after
  v2 registration the old revision is `is_active=0`, so when the staged v2
  publication is deleted as an orphan the document has *no* active
  publication — search returns an evidence gap rather than the obsolete v1.
  A gap is the only safe outcome; obsolete or uncommitted evidence is never
  served. Recovery happens when the publish job is retried (reconcile path).
- M3-test adaptations (product behavior correct, expectations pinned to
  3-stage M3 pipeline): every full-drain pipeline test now injects
  `FakeEmbedder` + `FakeQdrant`; a full no-OCR drain completes 4 jobs
  (extract/chunk/embed/publish). `test_worker_heartbeats_between_units`
  corrected to 4 beats for a 25-page book — publish heartbeats once
  (`worker.py` publish `on_progress`); chunk is pure in-memory work and never
  heartbeats. Beats = extract ×2 (pages 10/20) + embed ×1 (per batch) +
  publish ×1.
- The `qdrant-client` stubs type `Filter` fields as wide unions; the filter-
  shape test in `test_indexing.py` inspects the concrete shape via `Any`
  (commented in-test).

### Next unfinished task
- Begin M5 (PRD §12 "Cited answering and UI"): evidence manifests, local LLM
  adapter, bounded timeouts, citation validation, answer persistence, and the
  research UI; gate: unknown evidence IDs rejected; saved citations survive
  reindexing; unsupported/no-evidence cases abstain; search continues if the
  model server stops. The `evaluate` CLI command (PRD §9/§13) is also still
  unimplemented and belongs with the M5 evaluation work.

## M5 — Cited answering and UI

**Status: gate passed.** PRD §12 M5 gate, verbatim: "unknown evidence IDs
rejected; saved citations survive reindexing; unsupported/no-evidence cases
abstain; search continues if model server stops." All four clauses pass in
`tests/test_answers.py` (named below), plus 67 new test functions across seven
new files; `uv run pytest tests/` → 312 passed.

### Delivered
- `llm.py` — local answer-model adapter mirroring the embedding adapter:
  `AnswerModel` protocol; `LlamaCppAnswerModel` (OpenAI-compatible
  `/v1/chat/completions`, bounded timeouts on connect and read);
  `FakeAnswerModel` (scripted pops-then-fallback, records every call for
  assertions). Error taxonomy: server unavailable (dead port, HTTP error,
  timeout) → `ModelUnavailableError` (transient — search and other commands
  must keep working); malformed body → permanent `ModelError`.
  `make_answer_model` (unconfigured → `None`, `fake`, llama.cpp). The model is
  a pure text transducer — all trust rules live in `citations.py`/`answers.py`.
- `citations.py` — evidence manifests: the *only* vocabulary the model may
  cite is numbered `E1..En`, built from validated passages with location
  metadata taken from trusted sources (`source_revisions`/`chunks`/
  `source_units` rows + checksum-verified unit artifacts — the model's output
  is never trusted for location data). `parse_citations` (first-appearance
  order, dedup, out-of-contract ids rejected), `unknown_citations` (flags
  out-of-vocabulary ids), `parse_abstention` (`ABSTAIN` + reason). Frozen
  manifest snapshots (`schema 1`, `to_json`/`from_json`, rejects unknown
  schemas and bad evidence fields) are persisted with every answer.
- `answers.py` — cited-answering orchestration: fused search → evidence
  manifest → prompt (document text is untrusted data) → completion → citation
  validation. Rules: unknown/missing citation ids are rejected with exactly
  ONE bounded repair attempt, after which the answer persists as `failed`
  with the evidence still shown — never a fabricated fallback; abstention is a
  first-class persisted outcome; with no evidence at all the pipeline abstains
  *without calling the model*; every outcome (answered/abstained/failed)
  persists the frozen manifest. Citation resolution reads the snapshot, so
  re-chunking/re-publishing cannot move what a saved answer points at; the
  only live check is that the source revision is still registered and its
  archive readable (missing archive → explicit unavailable, never a substitute
  edition).
- `evaluate.py` — retrieval/answering harness (PRD §13): labeled JSON dataset
  (`expected_chunks`, `answerable`), `recall@k` over top-k candidates,
  abstention share for `answerable: false` questions, report formatting;
  nothing is invented — unlabeled metrics report `None`. Qdrant down →
  explicit unavailable error.
- `api.py` — FastAPI research app: `/health`, `/ready` (qdrant/embedding/
  answer booleans), `/library`, `/search` (503 IndexUnavailable / 400 blank /
  422 bad-shape split, degraded sparse-only surfaced in the payload),
  `/answer`, `/answers` (history), `/answers/{id}`, `/answers/{id}/citations/
  {evidence_id}` (snapshot resolution + reader location), ingestion controls
  (`/ingest/status`, `/scan`, `/ingest/pause`, `/ingest/resume`,
  `/ingest/retry`). Strict CSP on every response (`default-src 'self'`,
  `worker-src 'self'`, no `unsafe-eval`); bearer-token auth when configured
  with open liveness paths (`/health`, `/`, `/index.html`); static mount of
  `web/dist` at `/` via `find_web_dist()` (`$LIBRARY_RAG_WEB_DIST` override,
  repo `web/dist`, graceful API-only degradation when unbuilt).
- `reader.py` — book-manifest units now expose `unit_id` (additive change so
  the UI can fetch a unit directly; UUIDv5, not client-derivable).
- `migrations.py` — M5 `answers` table (CHECK `answered|abstained|failed`,
  frozen `evidence_manifest` JSON, `citations`, search counts,
  retrieval/model timings) + indexes; migration 0005.
- `cli.py` — `answer` (`--json`/`--doc`/`--rev`), `serve` (runs the FastAPI
  research app + reader), `evaluate` (`--dataset`/`--json`/`--k`) implemented;
  `backup`/`restore`/`verify` remain M7 stubs.
- `web/` — the research UI (PRD §10): TypeScript (strict) + Vite, no
  framework, pdfjs-dist bundled locally (no CDN; the worker ships as a
  same-origin asset, satisfying the server CSP). Two-pane research layout
  (query | reader) that collapses to Query/Reader tabs below 980px. Features:
  readiness dots (30s poll), book filter, Answer vs Search-only modes, answer
  cards with citation chips and per-evidence Locate/Excerpt, ranked passages
  with dense/sparse ranks and per-span Locate, persisted answer history,
  ingestion dashboard (status cards, job table, scan/pause/resume/retry, last
  scan report), 401 → token prompt (clears manifest caches), authed original
  download. Reader: PDF navigates by *physical* page — fitz bboxes (page
  points, bottom-left origin, unrotated space) converted to a top-left rect
  and through `convertToViewportRectangle` so zoom and page rotation are both
  accounted for; EPUB sections render only the server-sanitized HTML through a
  second client-side sanitize pass with deterministic paragraph anchors
  re-attached by mirroring `extraction.parse_blocks` (document order, block
  tags, non-empty text); a missing archived original (410) or any failure
  shows an explicit unavailable message — never a different edition.
  `web/dist` is built and committed (self-hosted gate); `web/node_modules`
  gitignored. `tests/test_web_dist.py` guards the bundle: present and
  discoverable, index.html references only local relative assets, no external
  resource-load patterns in the bundle, worker shipped and referenced.

### Test commands + results
| Command | Result |
| ------- | ------ |
| `uv run ruff check .` | All checks passed! |
| `uv run mypy` | Success: no issues found in 70 source files |
| `uv run pytest tests/` | 312 passed, 2 warnings in 42.60s |
| `cd web && npm run build` | tsc clean; vite 7.3.6 built (index 443.52 kB, pdf.worker 1,232.30 kB, css 8.43 kB) |

(The 2 warnings are a third-party starlette testclient/anyio
DeprecationWarning, not from this codebase. 67 new test functions:
`test_llm.py` 10, `test_citations.py` 15, `test_answers.py` 11,
`test_evaluate.py` 7, `test_api.py` 16, `test_cli_answer.py` 4,
`test_web_dist.py` 4.)

Gate cases (all passing, run this session):
- unknown evidence IDs rejected →
  `test_answers.py::test_unknown_citation_ids_fail_after_one_bounded_repair`
  (model cites `E9` out of a 3-entry manifest → one repair turn → `failed`,
  no fabricated citation) and `test_missing_citations_fail_after_one_repair`;
  `test_repair_success_is_answered` (repair turn cites valid ids → answered).
- saved citations survive reindexing →
  `test_answers.py::test_saved_citations_survive_reindexing` (answer saved,
  library re-chunked/re-published with new chunk ids, citation still resolves
  to the same frozen text via the persisted snapshot).
- unsupported/no-evidence cases abstain →
  `test_answers.py::test_no_evidence_abstains_without_model_call` (empty
  retrieval → abstained with zero model calls) and
  `test_scripted_abstention_is_persisted_with_reason`.
- search continues if the model server stops →
  `test_answers.py::test_down_model_server_fails_answer_but_search_keeps_
  working` (answer fails with an explicit model-unavailable reason while
  search still returns passages); `test_llm.py` covers dead-port/HTTP-error →
  `ModelUnavailableError` and malformed body → permanent error.

### Notes / bugs found
- pdfjs-dist 5.7.284 API drift from the widely-published examples:
  `PDFPageProxy.getLabel()` no longer exists — page labels come from the
  doc-level `getPageLabels()` (0-based array, empty string = unlabeled);
  `RenderParameters.canvas` is required (passing only `canvasContext` no
  longer type-checks). pdf.js does *not* resize a passed canvas
  (`beginDrawing` uses the existing canvas size and applies the viewport
  transform), so the client pre-sizes the canvas at `viewport × dpr` and sets
  the CSS size to `viewport`, then draws evidence boxes on the same canvas
  after the render promise resolves.
- `#canvasInUse` WeakSet in `InternalRenderTask` throws if the same canvas
  is handed to two concurrent renders; sequential re-renders are safe because
  `cancel()` releases the guard, which is what the reader does before each
  new render.
- pdf.js's internal Range requests cannot carry an `Authorization` header, so
  the authed client fetches the original bytes itself and hands the
  `ArrayBuffer` to `getDocument`.
- EPUB evidence carries no boxes (XHTML has no bboxes): "locate" resolves the
  section via `location.ref` → manifest unit (`kind=section`, matching `ref`)
  and scrolls the deterministic paragraph anchor instead.
- The client anchor re-attachment must mirror `extraction.parse_blocks`
  exactly (document-order walk, the same block-tag set, non-empty text) or
  anchors drift; a mismatch is logged (best-effort) but never fatal.
- `npm install` warns that esbuild's postinstall is not allow-listed; the
  platform binary resolves from the optional dependency and the build works —
  no system packages installed (PRD §1: no system installs).
- Unit tests still run without Docker/GPU: the web guard tests only inspect
  the committed `web/dist` files, and all M5 backend tests use
  `FakeEmbedder`/`FakeQdrant`/`FakeAnswerModel`.

### Next unfinished task
- Begin M6 (PRD §12 "Pilot and capacity report"): reproducible stratified
  sample manifest for 200–500 books with configurable page cap (all
  format/OCR/language groups); measure extracted tokens/pages, chunk count,
  extraction/OCR/embedding throughput, SSD bytes per chunk, peak RAM/VRAM,
  p50/p95 latency with ingestion on/off; prepare 100–200 manually labeled
  questions (exact terms, conceptual, OCR, EPUB, conflicts, unanswerable) with
  annotation/evaluation tools — do not invent human labels; deliver the pilot
  report (measured results, uncertainty, full-run ETA by bottleneck, storage
  estimate with rebuild headroom, recommended frozen config). **Operator
  approval is required before full ingestion.**

## M6 — Pilot and capacity report

**Status: COMPLETE** (run #2 deliberately stopped 2026-09-20 after all
300 books had passed extract→ocr→chunk→embed — the only remaining queue
was the O(N²) epoch-republish churn, which is finding F1 and is fixed as
Fix B; report, latency measurement, and question candidates are all
done; operator pre-approved the report 2026-09-20; the report is
`M6_PILOT_REPORT.md`, marked APPROVED).

### Delivered (code, gate green)
- `pilot.py` — `pilot survey` (serial MuPDF survey of the corpus → JSONL
  strata records: format, ocr_class, language, pages, size) and
  `pilot sample` (stratified sample manifest, seed-reproducible, page cap,
  schema `pilot-manifest/1`). Survey of the full corpus complete: 34,768
  records (25,279 pdf / 9,129 epub / 360 invalid), 51 strata.
- `pilot run` — ingests exactly the manifest's books into an isolated
  sandbox (derived config: own state/qdrant/archive/artifacts/scratch
  under the sandbox root, embedded local Qdrant, page cap from the
  manifest), runs the serial worker to drain, and writes measured metrics
  (per-stage seconds from the jobs table, units/chars, chunks/tokens,
  peak RAM via getrusage incl. children, VRAM deltas via nvidia-smi
  sampling, disk deltas per sub-root) to `scratch/pilot_run.json`.
- `latency.py` — `pilot latency`: p50/p95 for search/answer probes,
  idle vs with ingestion in flight (bounded ingest backlog from a
  source root), probes against the sandbox index.
- `questions.py` — annotation/evaluation tooling for 100–200
  human-labeled questions (exact_term / conceptual / ocr / epub /
  conflict / unanswerable). `pilot annotate suggest` proposes candidate
  questions from the sandbox index for a human to label; the bank
  accepts ONLY human-written labels — the tooling never invents labels.
- `pilot_report.py` — `pilot report`: measurements, uncertainty,
  full-run ETA by bottleneck, storage estimate (dense + two-generation
  rebuild headroom + archive copy vs free space), recommended frozen
  config, and explicit findings.

### Findings so far
- Embedding server is bge-m3 Q8_0 on 127.0.0.1:8081 (llama.cpp
  `--embedding`, build /dd2/andrei/bench/build/bin). Its `/v1/embeddings`
  endpoint returns the standard OpenAI shape (verified live).
- The running instance enforces a 512-token PHYSICAL BATCH ceiling: a
  single input above ~512 BPE tokens → HTTP 500 "input (N tokens) is too
  large to process". Before M6's run this was misclassified as a
  transient server error (infinite zombie retries). Now classified
  permanent (`embedding_error`, transient=False) in `embeddings.py` with
  a regression test, so affected books fail explicitly and are counted
  in the report instead of retrying forever.
- **Ceiling root cause (server startup log, confirmed 2026-09-20):**
  the 8081 server is launched with `--embedding -c 8192 --batch-size
  2048` (`/dd2/llama-server/embedder.sh`), but this llama.cpp build's
  embedding mode asserts `n_batch <= n_ubatch` and silently clamps at
  startup — `embedder-8081.log`: "embeddings enabled with n_batch (2048)
  > n_ubatch (512)" / "setting n_batch = n_ubatch = 512 to avoid
  assertion failure". The effective batch is the model default
  `n_ubatch=512`, not the 2048 requested — this was previously
  misreported as "an operator restart of the wrong server would fix
  it"; there is no wrong server. The fix is relaunching 8081 with
  `--ubatch-size 2048` (covers every observed failure: inputs 515–1293
  tokens).
- Consequence for the corpus: `WordTokenCounter` counts `\S+` runs, so a
  spaceless CJK paragraph is one "word" and a chunk can exceed the 512
  BPE ceiling. The manifest contains CJK strata; 132 of the 300 sample
  books failed explicitly with `embedding_error`. Before full
  ingestion: relaunch the embedder with `--ubatch-size 2048`; the report
  additionally recommends BPE-aware chunking (M7) for robustness beyond
  2048 tokens.
- Survey ran serial after a MuPDF thread race (concurrent
  `page_get_textpage` → segfault) made multithreaded surveying unsafe.

### Bugs the pilot found (its job is working)
- **First publish crashed the worker: `Collection library_chunks not
  found`.** `RealQdrantOps.ensure_collection` was only ever called in
  tests; on a fresh sandbox the first publish upserted into a
  non-existent collection, the qdrant exception escaped the job handler,
  and the run died at exit 1 after extract/chunk had finished (135 books
  embedded, 134 publishes left pending). Fixed in three parts, all
  gated:
  1. `worker.py` `_run_publish` now calls
     `q.ensure_collection(dimensions=…)` under the publish lock, before
     the switch — creation is serialized with publication.
  2. `indexing.py` `reconcile_publications` guards on
     `qdrant.collection_exists()` — a missing collection no longer
     looks like "vanished index points" and fabricates republish jobs.
  3. Regression test
     `tests/test_worker_m4.py::test_publish_creates_missing_collection_in_local_mode`
     runs the full pipeline in embedded (local) Qdrant mode from an
     empty collection. Gate after fixes: **373 passed, ruff clean,
     mypy clean (76 files)**.
- **tesseract was missing from the host** while ~99% of the sample
  needs OCR: all 1048 OCR page units in run #1 permanently failed with
  `ocr_unavailable` ("tesseract binary not found"). Operator approved
  the install; tesseract 5.3.4 (+ eng + osd, Ubuntu 24.04 apt) is now
  at /usr/bin/tesseract and was smoke-tested with the exact worker
  invocation (pdftoppm 150 dpi → `tesseract -l eng --psm 3 tsv`,
  OMP_THREAD_LIMIT=1 → exit 0). Full ingestion requires tesseract —
  requirement now satisfied.
- Run #1 forensics (per-stage counts, the 55 embed failures by token
  count — 513–531, all CJK —, 110 zero-chunk books) saved to
  `/mnt/models_sata_ssd/library-rag/scratch/pilot_run1_forensics.md`.
  Sandbox wiped; run #2 launched from clean state.
- **O(N²) statistics-epoch republishing (HEADLINE CAPACITY FINDING).**
  Every successful publish converges all earlier books onto the current
  corpus-stats epoch: `_enqueue_epoch_republishes` (called at
  worker.py:837, defined worker.py:846) enqueues one republish job per
  earlier active publication. Measured: 161 embed successes → exactly
  13,041 = 161×162/2 publish jobs, settling at 14.2/min (0.24/s) under
  the global publication lock (indexing.py:546). The handler
  "recomputes the epoch" (worker.py:846 docstring) — i.e. recomputes
  corpus statistics — even when the book is already on the current
  epoch, so no-op jobs cost ~4.2 s too. The first four stages finished
  in ~52 min; the publish tail alone is ~15 h (7.7 h measured for
  6,586/13,041 at stop). At full-corpus scale (34,768 books; ~19k at
  the pilot's 55% embed success) this is ~180M–600M republish jobs —
  infeasible. **The report must recommend a frozen corpus-stats epoch**
  (freeze the epoch at ingest start; republish only on a real epoch
  change) or batched end-of-run republishing. Not a pilot blocker — the
  explosion and its rate are fully measured at 6,586/13,041.
  **Mechanism confirmed in code (2026-09-20):** `stats_sha`
  (embeddings.py:513) hashes `{"N", "avgdl", "df", "k1", "b"}` — the
  document count is inside the identity, so **the epoch necessarily
  moves on every new book** (by design: adding a document changes every
  sparse vector, exactly like changing k1/b does). Every publish then
  pays a corpus-wide `compute_sparse_stats` (~4.2 s, O(corpus)) before
  its own current-epoch check can return noop. Sandbox DB evidence:
  113 generations share the initial epoch `dc0fbcb8…`, then one new
  epoch per book (~149 distinct shas over 273 generations); ~12.8k of
  the 13,041 publish jobs were stale-epoch no-ops that each paid the
  full recompute — that is the constant 14.2/min. (The superseded-row
  pattern — 112 books with one superseded pub row, 49 with none — is
  the same per-book epoch movement at the publication level.) Fix plan:
  **B1** — persist the corpus epoch in the meta table (under the
  publish lock) and skip the recompute in `_run_publish`/
  `publish_generation` when it is unchanged (kills the ~12.8k wasted
  recomputes); **B2** — epoch stability for full scale: thresholded
  stats hash (sub-~0.1%-df terms excluded from the identity sha only;
  `bm25_weights` keeps the full dfs) and/or an opt-in frozen-epoch
  config knob (default = current behavior) → two-phase ingestion:
  live ingest without per-publish fan-out, then ONE bounded full
  re-sparse at campaign end.
- **1-pixel raster-size check loses OCR pages (ocr.py) — FIXED
  2026-09-20.** 162/1048 OCR pages (15.5%) permanently failed with
  e.g. `unexpected raster size 1400x2058, expected 1400x2059` — a
  1-pixel mismatch between `build_transform`'s `ceil()` prediction and
  the pixmap PyMuPDF actually sizes (different float association in
  PyMuPDF's matrix→raster math), deterministic per page, so all 3
  retries fail identically → `ocr_error` → permanent_failed. 15.5% of
  OCR'd text was missing from the index. Fix: ±1 px per-dimension
  tolerance — citation boxes divide by the exact scale, never the
  render size, so a 1 px raster difference cannot skew a box
  (docstring + inline comment in ocr.py). Regression tests
  `test_ocr_page_tolerates_one_pixel_raster_drift` (±1 px predicted
  drift → OCR proceeds, page-point boxes unchanged, scratch cleaned)
  and `test_ocr_page_still_rejects_two_pixel_raster_mismatch` (+2 px →
  `ocr_error`, scratch cleaned); both monkeypatch
  `build_transform` because the drift is NOT reproducible
  synthetically on PyMuPDF 1.28.2 (0 drifts in a 0.001 pt geometry
  scan at the pilot's implied 336.0×493.92 pt @300 dpi — it is a
  property of PyMuPDF's internal float association, not of any
  particular page size). Gate after fix: **375 passed (360 + 15 in the
  dot tally), ruff clean, mypy clean (76 files)**. The 162 pages
  re-OCR cleanly on the next worker run (retry the permanent_failed
  jobs).
- **Final measured state of run #2** (stopped 2026-09-20; worker
  confirmed dead by pgrep at now≈1789898062): extract 300/300 ✓;
  ocr 886 ✓ + 162 ✗ (2.95 s/page, 43.5 min for 886 pages); chunk
  300/300 ✓ (34,908 chunks; only 7 empty books vs 110 in run #1 — with
  OCR working, books get text); embed 161 ✓ + 132 ✗ (embedding_error,
  the known 512-token ceiling; 46.7 min for all 293 jobs); publish
  6,586 ✓ + 6,454 pending + 1 running of 13,041 (14.2/min sustained,
  queue monotonically shrinking — embed fully settled so no further
  republish jobs are created). Publications: 161 active + 112
  superseded. All publish input_versions share one 64-char prefix
  (single embedding epoch; bge-m3 for all 161).

### Run #2 — stopped (2026-09-20)
- Same 300-book manifest (seed 42, page cap 32), clean sandbox at
  /mnt/models_sas_ssd/library-rag/pilot-sandbox (archive 1.6G / 300
  files). **All 300 books passed extract→ocr→chunk→embed**; the only
  remaining work was the O(N²) republish tail (6,454 of 13,041
  pending), already measured in kind at 14.2/min. Stopped deliberately
  (TaskStop `bkue5984m`; worker death verified by pgrep; monitor
  `bjnbqjfn2` had already expired) — waiting ~7.7 h more would only
  complete known quadratic churn and would block the latency probe,
  which needs a quiescent index. The pilot is "complete in kind": every
  per-stage rate the report needs is measured, and the republish
  explosion is the finding, not a gap.
- `scratch/pilot_run.json` was NOT auto-written (pilot.py writes it at
  run end). Metrics must be collected from the sandbox jobs table with
  pilot.py's own collector and written with an explicit
  "interrupted: republish tail stopped" flag — see Next.
- Post-run runbook at
  `/mnt/models_sata_ssd/library-rag/scratch/m6_post_run.sh`:
  report → latency (idle vs 8-book ingest backlog) → annotate suggest.

### Worker architecture note (capacity finding)
- The host has **32 CPUs, no cgroup quota** (cpu.max = max; affinity
  0-31 — verified via sched_getaffinity; `nproc` misreports as 1 on
  this host, which caused an earlier wrong "1 effective CPU" note).
- OCR/extract use ~1 CPU because the M4 worker is a **single serial
  process** (one job at a time; tesseract OMP_THREAD_LIMIT=1 per call).
  PRD lines 35/53 specify coordinator + bounded worker subprocesses
  with extraction at min(4, cores) and OCR at 2 workers — **not yet
  implemented**. Sandbox local-mode Qdrant's exclusive flock also bars
  a second worker process (production's Qdrant container does not).
- OCR phase finished in 43.5 min (2.95 s/page measured), as predicted;
  the long pole turned out to be the O(N²) republish tail, not OCR —
  run #2 was stopped there (see above). The report presents full-run
  ETA for both the shipped serial worker and the PRD-parallel config,
  and flags the worker-architecture gap as a before-full-ingestion
  item (M7 candidate).

### Next unfinished task
- [DONE 2026-09-20] 1-px raster fix (±1 px tolerance) + 2 regression
  tests; gate green: 375 passed, ruff clean, mypy 76 files clean.
- [DONE 2026-09-20] Epoch-republish mechanism confirmed in code (see
  the O(N²) bullet: doc_count inside `stats_sha` ⇒ the epoch moves on
  every new book; ~12.8k of the 13,041 jobs were stale-epoch no-ops
  that each paid the ~4.2 s corpus recompute).
- [DONE 2026-09-20] **FIX B: kill the O(N²) republish cost.** Shipped
  design — the thresholded-identity hash variant was analyzed and
  rejected as unsound (N enters the idf of *every* existing term).
  - **B1 (always-on): corpus-epoch record.** `meta` key
    `corpus_stats_epoch` (JSON `{"at","bm25","doc_count","stats_sha"}`)
    written **inside `publish_generation`'s B4 SQLite transaction**
    (the only place the active set changes, under the publish lock →
    crash-atomic with the committed active set).
    `indexing._corpus_stats_for_publish` reuses the record when the
    bm25 sha matches AND (the revision is already active — 99.9% of
    the measured tail — OR frozen mode); otherwise it computes.
    Missing row = self-healing fallback.
  - **B2 (opt-in knob): `retrieval.stats_epoch: live|frozen`**
    (default `live` = current behavior; deliberately NOT part of
    `bm25.settings_sha()`, so the embedding identity never changes).
    Frozen pins the epoch for the whole campaign: a new book's novel
    terms get zero sparse weight until re-freeze, and the fan-out
    finds no mismatches → zero fan-out jobs. Campaign-end re-freeze
    recipe (documented in config.example.yaml): flip to `live`,
    `DELETE FROM meta WHERE key='corpus_stats_epoch';`, then one
    publish recomputes over the full corpus and its fan-out converges
    everything in a single bounded round.
  - **O(1) version strings:** the embed-handoff publish job
    (`_run_embed`) and the reconcile publication-gap job
    (`_reconcile_index`) use `f"{emb_sha}:{record_sha|'init'}"` — no
    job-table queries; the handler resolves the real epoch under the
    publish lock. Accepted consequence: re-embedding an
    already-published book enqueues one extra publish job under the
    post-publish sha that early-no-ops in O(1).
  - **Files:** embeddings.py (record helpers), indexing.py (fast path
    + record store in B4), worker.py (O(1) versions, docstrings),
    config.py (`stats_epoch` field), config.example.yaml (documented
    `retrieval` block, recommends frozen for full ingestion).
  - **Tests (test_worker_m4.py):** 3 new —
    `test_republish_fast_path_skips_stats_recompute` (stale-epoch job
    for an active book: zero compute calls, noop manifest),
    `test_frozen_epoch_pins_and_skips_fanout` (second book: no
    fan-out job, record doc_count unchanged),
    `test_refreeze_recomputes_once_and_converges` (re-freeze recipe:
    exactly one full-corpus compute, fan-out converges both books).
    Plus 4 mechanical count updates (re-embed handoff now enqueues an
    O(1) no-op republish; the Qdrant-down retried publish coalesces
    with the gap job under emb:init).
  - **Gate (run 2026-09-20, real output):** ruff "All checks passed!",
    mypy "Success: no issues found in 76 source files", pytest
    **378 passed** (baseline 375 + 3 new).
- [DONE 2026-09-20] Run metrics: `m6_collect_run_metrics.py` (scratch)
  reconstructs `pilot-sandbox/scratch/pilot_run.json` READ-ONLY using
  run_pilot()'s verbatim collection SQL, `interrupted: true` (run_pilot
  itself cannot be re-run — it would drain the 6,454 pending legacy
  republish jobs and mutate the frozen artifact). Plus
  `m6_stage_attribution.py`: real per-stage worker time from the
  completion sequence — publish (old code) 27,288.9 s = 89.6% of the
  run; ocr 2,005.9 s; embed 511.9 s (12,385 tok/s); extract 346.4 s;
  chunk 294.9 s. The auto-report's per-stage ETAs (759,999-day) are
  queue-wait inflation in a single-consumer queue; its measured sections
  stay authoritative, its projections are replaced by report §4.
- [DONE 2026-09-20] Latency (post-runbook step 2, replaced after the
  stuck-run forensics): the full post-runbook latency run in the legacy
  sandbox hung in the idle phase — root-caused with py-spy: 49 distinct
  active epochs × 52,257 points × per-point payload-filter evaluation in
  pure Python (local-mode Qdrant is a pure-Python embedded backend;
  payload indexes have no effect locally — client warning) → ~4 min per
  probe. Replaced with a two-part measurement:
  (a) single read-only probe in the frozen legacy state:
  **230,275 ms per query** (49 epochs);
  (b) fresh 30-book sandbox under the recommended frozen config
  (manifest30.json seed 42 cap 32; config.m6lat.yaml = config.yaml +
  `retrieval.stats_epoch: frozen`): **idle p50 215.3 ms / p95 218.4 ms;
  ingesting p50 233.9 ms / p95 246.5 ms** (8 new books enqueued, n=30
  per phase) — concurrent ingest is +10%, not the problem; the epoch
  fan-out is. The 30-book run also live-verified both fixes: OCR
  205/205 OK (Fix A; 162/1048 failed in the legacy run) and publish
  18 jobs / 18 publications, 1:1, zero fan-out (Fix B frozen; legacy:
  13,041 jobs / 300 books). 12 embed failures = the known 512-BPE
  ceiling (8081 not relaunched — first action next session).
  Search-side consequence (report §5): a query runs one full-corpus
  sparse search per active epoch, so a full-scale pre-fix campaign
  approaches ~34k per query.
- [DONE 2026-09-20] Annotate suggest (post-runbook step 3): 80
  unlabeled candidates (40 chunks x exact_term+conceptual, seed 42) →
  /mnt/models_sata_ssd/library-rag/scratch/question_candidates.jsonl,
  generated READ-ONLY against the frozen pilot sandbox DB (mode=ro
  connection wrapped in Database — no artifact mutation). Labeling
  remains a human task (never invent labels).
- [DONE 2026-09-20] Final report: M6_PILOT_REPORT.md — measured run +
  corrected full-corpus ETA by bottleneck (~700 h ≈ 29 days serial
  single worker; ~1 week with the M7 PRD-parallel architecture),
  O(N²) finding + Fix B (frozen epoch recommended), latency (§5),
  storage 1,481 GB vs ~14.4 TB free (~4x headroom), 512-token ceiling
  (relaunch 8081 with `--ubatch-size 2048` = first next-session action),
  1-px raster fix, question-bank status, recommended frozen config.
  **Operator pre-approved the report (2026-09-20, "consider it already
  approved")** — marked APPROVED, not blocked on presenting it.
- [DONE 2026-09-20] M6 close: commit **ea36018** ("M6: pilot + capacity
  report (gate passed)") + push to origin/master + `shutdown now` per the
  terminal directive. Host rebooted 17:00 same day; the operator
  relaunched the embedder fleet (8081-8084, screen session `embed`) and
  vLLM 8091 before this session.
- [DONE 2026-09-20] **8081 relaunched with `--ubatch-size 2048`** (first
  next-session action). The build's default ubatch is 512, and the server
  validates **each input** of a `/v1/embeddings` batch against it: an
  8-chunk request with ~3,440 total BPE passes as long as every chunk is
  ≤ 2048, while a single 2277-token input still 500s. The pilot's 132
  failed embed jobs were single chunks of 515-723 BPE (error texts in the
  sandbox jobs table) — all now fit with 3x margin. Only the 8081
  instance was stopped (8082-8084 and 8091 untouched); `--ubatch-size
  2048` was added to /dd2/llama-server/embedder.sh (persists for future
  relaunches via that script); relaunched detached (log:
  /dd2/llama-server/embedder_8081_restart.log). Verified live: ~700-BPE
  single input → 1024-dim unit-norm vector; 8 x ~430-BPE batch → 8
  vectors, indexes 0-7.
- [DONE 2026-09-20] **Retrying the 294 permanent_failed jobs**
  (132 embed + 162 ocr) in the pilot sandbox — requeued
  (`library-rag retry --include-permanent`, "requeued 294 job(s)"), then
  drained. By the time of the second reboot the 162 OCR jobs and 131 of
  the 132 embed books had already succeeded (OCR 1048/1048 ✓; embed
  285/293 ✓) against the ubatch-2048 fleet; the first drain
  (`retry_run.log`) died on the local-Qdrant self-lock bug (below) part
  way through the publish tail.
- [DONE 2026-09-20] **Drain worker death: local-Qdrant self-lock —
  root-caused, fixed, regression-tested.** The host rebooted a second
  time; the operator relaunched the fleet as **8 instances 8081-8088**
  (all `--ubatch-size 2048 --batch-size 2048`, GPU 2, auto-restart on
  boot; operator: "and i can add 4 more") and vLLM at **port 8000**
  (pid 3453 — the old :8091 note is stale). The re-drain ran to the
  first non-noop publish and died:
  `RuntimeError: Storage folder ... is already accessed by another
  instance of Qdrant client`. Root cause: `RealQdrantOps` in local mode
  holds an exclusive flock on `<qdrant_path>/.lock` for the client's
  whole lifetime; the old worker built client #1 for
  `reconcile_publications` and discarded it, then `_run_publish` built
  client #2 in the same process → POSIX flock self-denial. Fix: ONE
  shared `RealQdrantOps` per `run_worker` call when no client is
  injected (opened in `run_worker`, shared into reconcile and every
  publish via `partial`, closed in `finally`; injected clients are
  never closed by the worker). `_run_publish`'s `qdrant` parameter is
  now required. Regression test
  `tests/test_worker.py::test_worker_single_local_qdrant_client_per_run`
  pins: exactly one client constructed per run (counting constructor),
  two publish jobs succeed through it, and the storage lock is free
  again after return.
- [DONE 2026-09-20] **Embedding endpoint pooling** for the 8-instance
  fleet: `services.embed_ports: list[int]` (default `[]` = single
  `embed_port` behavior); round-robin per `encode_documents`; failover
  only on transient `ModelUnavailableError`; ports excluded from
  `embedding_sha` (endpoint choice never changes the embedding key).
  Config validation: each port must be 1-65535. Tests in
  tests/test_config.py + tests/test_embeddings.py.
- [DONE 2026-09-20] **Full gate green** after both changes: ruff
  "All checks passed!", mypy "Success: no issues found in 76 source
  files", pytest **400 passed** (exit 0).
- [IN PROGRESS 2026-09-20] **Sandbox drain #2** (fixed worker):
  `uv run library-rag ingest --config pilot-sandbox/scratch/
  config.sandbox.yaml --once` detached, log
  `pilot-sandbox/scratch/drain2.log`, started 18:56. At start: chunk
  16 pending / 300 succeeded; embed 285 succeeded / 7 retryable (deferred
  on the unsettled chunk jobs — all prose books, sample chunks 108-818
  BPE, will pass) / 1 permanent; publish 6586 succeeded / 6578 pending /
  1 stale-running (crash remnant; lease reclaim verified working).
  Rate ~1 publish/~12 s → ETA ~22 h for the legacy republish tail
  (converges: frozen stats_epoch ⇒ fan-out re-enqueues only existing
  task keys, no new work).
- **NEW FINDING (2026-09-20, measured): the 2048-BPE per-input ceiling
  is still below real chunks for number/hex-dense content.** The one
  permanent embed job is `ibm.pdf` (it/networking/protocols, "Mega
  protocols" — packet dumps, hex tables). Its 10 chunks are only
  156-254 WORDS but measure **1616-3656 BPE tokens** each via the
  server's `usage` field (7 of 10 exceed 2048; worst 3656).
  Word→BPE ratio here is 8-15x vs the 1.4-1.8x the config assumes for
  prose — `WordTokenCounter`'s word budget cannot bound BPE length for
  this content class. Consequences: (a) sandbox final state will be
  292/293 embed succeeded, 1 permanent; that book's publish jobs fail
  `publication_error` transiently 3× then go permanent → the book is
  in the index minus 7/10 chunks; (b) full-library ingestion will
  reproduce this on technical books (stratified sample: 1/300 = 0.3%
  of the sample is this class; the report already defers BPE-aware
  chunking to M7). Operator options: raise `--batch-size`/
  `--ubatch-size` on the fleet (4096 covers every measured input;
  8192 = "never reject anything that fits the context" — safe: bge-m3
  KV at 8192 is ~1.2 GB), or accept explicit failures (designed
  behavior: counted, never zombie-retried), or M7 BPE-aware chunking.
- [DONE 2026-09-20] **Sandbox drain #2 finished, verified from the
  jobs table:** extract 300 ✓; ocr 1048/1048 ✓; chunk 316 ✓ (300 +
  16 re-keys); embed 306 ✓ + 1 permanent (the documented ibm.pdf,
  number/hex-dense class); publish 13,172 ✓ + 1 permanent (the
  ibm.pdf book — in the index minus the over-long chunks, designed
  behavior). 292 active publications; 300 documents. Sandbox
  quiescent — kept untouched as the pilot artifact.
- [DONE 2026-09-20] Committed + pushed the worker self-lock fix +
  regression test + embed-ports pooling as **22aaa70** (gate: ruff
  clean, mypy 76 files, 400 passed).
- [DONE 2026-09-20] config.yaml (gitignored, operator file):
  `retrieval.stats_epoch: frozen` + `services.embed_ports:
  [8081..8088]` (embed_port 8081 kept as the single-endpoint
  fallback).
- [DONE 2026-09-20] **Fleet bumped to 8192** (operator: "Bump to
  8192, then launch"): all 8 instances run `--batch-size 8192
  --ubatch-size 8192` (embedder.sh updated, health-verified
  8081-8088); per-input ceiling = 8192 BPE tokens — a 3656-token
  input verified accepted on all 8 endpoints, so the number/hex-dense
  rejection class (ibm.pdf) is class-fixed. Measured word→BPE ratios
  EN 1.39 / DE 1.69 / FR 1.65; ~7000 tok/s per instance on a batch of
  8.
- [DONE 2026-09-20] **Per-job verbose logging** (operator request,
  verbatim: "logging system which you will able to use if something
  goes wrong. we won't keep verbose logging forever, you run verbose
  logging next to short logging for current jobs, if a job fails,
  then keep verbose log for debugging, otherwise flush it"): each
  claimed job is captured at DEBUG to
  `<state_root>/job_logs/<job_id>.attempt<N>.log` — written under its
  final name from the first line, so a mid-job crash leaves the log
  (the file's existence during a run is NOT a failure signal);
  keep-on-failure only (permanent, or retryable with an error
  category), flush on success/deferral/lost-lease; newest-500 cap on
  keep. 6 new tests; gate: ruff clean, mypy 76 files, **394
  passed**. Committed + pushed as **ecd2582**.
- [IN PROGRESS 2026-09-20] **Batch 2 — second stratified sample**
  (operator: "sample again books in repository
  /mnt/models_sas_ssd/books/ for running second batch of tests, to
  see if other bugs pop up"): 300 books, seed 1337, 35 strata, no
  page cap, **zero overlap with the pilot's 300** (survey
  pre-filtered), 32,441 pages; isolated sandbox
  `/mnt/models_sas_ssd/library-rag/batch2-sandbox/` (own
  state/qdrant/archive/artifact/scratch; inherits frozen epoch +
  8-port embed pooling from config.yaml). Launched detached:
  `uv run library-rag pilot run --config config.yaml --manifest
  batch2-sandbox/scratch/manifest_batch2.json --sandbox-root
  batch2-sandbox` (short log: `batch2-sandbox/scratch/run.log`).
  First checks: registration 300/300, ~95 extracts ✓, **zero failed
  jobs** — a "kept log" sighting (job 96) was just the live capture
  of an 8-minute extract, flushed on success (the keep/flush
  mechanism verified both ways). ETA several hours (OCR-dominated,
  ~1.9 s/page). Watch: any file left in `state/job_logs/` after a
  job settles = a real failure to investigate; run.log for worker
  errors; embed-pool health.
- [IN PROGRESS 2026-09-20] **Abrupt-shutdown simulation** (operator:
  "may be, simulate machine abrupt shutdown and recovery from that").
  At 20:49:48 (drain 62% through extract: 200/300 ✓, 2,789 ocr
  pending, 25,016 units, **0 failed jobs**) `kill -9` on the worker
  (pid 64669) with job **279 in flight** (extract, attempt 1, lease
  to 20:56:41). Post-mortem, all PASS:
  - `PRAGMA integrity_check` = ok, `quick_check` = ok, 0 FK
    violations on the sandbox library.db.
  - Local-Qdrant flock acquired by a fresh probe → the kernel
    released the dead process's lock; a new client can open the
    store.
  - Job 279 stayed `running` in the DB (worker died before state
    update) — exactly the lease-expiry + reclaim path the design
    relies on.
  - **The crashed job's verbose log was kept**:
    `state/job_logs/279.attempt1.log`, last line
    `20:49:28 extract: start {job 279, rev b8cd000f…, doc
    5a568094…}` — the logging system doing its one job.
  Recovery: identical `pilot run` command, same `--sandbox-root`,
  appended to the same run.log (recovery run: wrapper 72172, worker
  72176). Pre-crash snapshot (3,289 jobs, 300 docs) recorded for
  before/after comparison. RECOVERY VERIFIED — drill PASS:
  re-registration added 0 docs/0 duplicate jobs (300 docs, growth
  3289→3671 = exactly 80 new small-book extracts → +80 chunk
  +302 ocr jobs); job 279 correctly NOT claimed before its lease
  lapsed; after expiry (20:56:41) it was reclaimed and **succeeded
  as attempt 2** with error_category None — `279.attempt1.log`
  stays as the crash artifact, the attempt-2 log flushed on
  success (keep/flush semantics verified through a real
  crash+retry); drain re-converging (289/300 extracts, 0 failed
  jobs) and continuing to completion (OCR-dominated).
- **Measured per-stage rates (mid-drain, 2026-09-20 21:34; saved as
  mycelium #55):** gap analysis of the completion timeline (single
  worker: inter-completion gap = that job's duration): extract
  6.2 s/book (300 books in 0.52 h wall); **OCR 4.59 s/job median**
  (p10 1.08 / p90 5.07 / max 6.1) — OCR is SELECTIVE per PRD §8C
  (only text-layer-less pages get routed): 6,176 OCR jobs for 32,441
  units = **19% of pages scanned**, 20.6 OCR jobs/book; chunk
  median 4.65 s/book (p10 0.45, max 43.7 s for a 1,622-page book);
  embed/publish not yet complete in batch2 (embed ≈ 12 h corpus-
  scale at the fleet's 7,000 tok/s over ~300M BPE tokens).
  NOTE: 4.59 s/OCR-page is 2.4× the pilot's 1.9 s — batch2 has no
  page cap and includes large scanned books; 4.59 is the planning
  rate.
- **Mount-sentinel gap found and closed (2026-09-20, post-crash
  sim):** the configured sentinel had silently never been enforced —
  config.yaml keyed `mount_sentinels` by a human label (`books:`)
  while `scan_root` looks it up by **source root path string**
  (proven by tests/test_scan.py:176; the stale "human label"
  docstring in config.py was the misleading root cause). Three-part
  fix: (1) created
  `/mnt/models_sas_ssd/books/.library-rag-sentinel` (21:39,
  zero-byte marker — NOTE: this file lives INSIDE the source root;
  flagged to operator per "do not delete or modify source books";
  not picked up by scans since it is neither PDF nor EPUB);
  (2) config.yaml re-keyed to the root path + config.py docstring
  corrected (no "empty value = no sentinel" claim: scan.py treats a
  set-but-missing/empty value as BLOCKING — only a missing KEY means
  no check); (3) `library-rag reconcile` hardcoded `sentinel=None`,
  so an empty-but-present mount could prune every `path_alias` under
  its root — it now passes the configured sentinel through (one-line
  fix; `reconcile_catalog`/`mount_present` already implement the
  semantics and module tests cover them). Verified: gate replication
  positive (real sentinel → scan proceeds) and negative
  (nonexistent sentinel → mount blocked); full gate green (ruff,
  mypy, 394 passed). Verified CLI signatures: `library-rag scan
  [--config] [--json]`, `library-rag ingest [--config] [--once]
  [--lease-ttl]` (ingest default runs until SIGTERM).
- **Full-run launch pre-verification (2026-09-20, read-only):** no
  production root dir exists yet (pilot + batch2 ran in sandboxes);
  that is fine — `_open_state` (cli.py:113) auto-creates and
  idempotently migrates the state DB (`Database.connect` mkdirs the
  parent, db.py:49), so the launch sequence is `scan` → `ingest`
  (an explicit `library-rag init` is optional/idempotent, not
  required). Capacity check against free space (df 2026-09-20
  ~21:58): sas_ssd 6.9 T free; sata_ssd 295 G free (86% used).
  Linear scale of the 300-book pilot footprint ×115.9 → 34,768
  books: archive ~185 G + artifacts ~87 G (sas_ssd, ample), state
  ~55 G + qdrant ~88 G ≈ 143 G (sata_ssd, ~2× headroom vs 295 G).
  Consistent with the pilot capacity gate that passed.
- **Next unfinished task:**
  1. **Let the batch-2 drain finish** (recovery run, worker 72176;
     monitors armed: failed-job poll "re-arm 4", 30-min expiry,
     exit watcher on 72176). State at 23:03: **extract 300/300
     COMPLETE**, OCR 2,488 done / 3,687 remaining (+1 running),
     chunk 187/300, embed 183 pending, 0 failed jobs. Pace:
     22:17–23:03 window 1,257 OCR jobs / 46 min ≈ 27.3 jobs/min
     (another short-book burst; re-measure at each re-arm).
     Remaining OCR ≈ 135 min at this pace → **ETA ~01:30–03:30**
     (the 05:00–06:15 band assumed the 13.1 jobs/min planning
     rate and is now stale-fast). Terminal expectation: ~300
     documents, all stages succeeded except legitimate per-book
     permanents — any permanent failure gets its kept verbose log
     read and classified before the full run.
  2. Once batch 2 is fully drained and the crash sim is clean:
     **full-library ingestion** (approved, report pre-approved
     2026-09-20): `uv run library-rag scan --config config.yaml`
     then `uv run library-rag ingest --config config.yaml` — detached
     + monitored; **recomputed ETA from batch-2 rates (replaces the
     report's ~700 h ≈ 29 days): extract ~17 h, chunk ~48 h, OCR
     ~914–990 h (34,768 books × 20.6 scanned pages × 4.59 s) ≈
     38–41 days, embed ~12 h, publish ~10–20 h (frozen epoch) →
     ~41–45 days single-worker, OCR ≈ 93% of it.** The PRD
     coordinator/worker-subprocess architecture (M7) is the
     prerequisite for a ~2-day parallel run (32 OCR processes). Do
     not claim completion without evidence.
  3. Operator labels the 80 question candidates (top up to 100-200)
     in /mnt/models_sata_ssd/library-rag/scratch/question_candidates.
     jsonl — human-only; never invent labels.

## M7 — Full-run operations and maintenance

Scope (PRD line 180): scheduled discovery, safe revision
replacement, explicit removal, generation migration, GC with
reference checks, backup/restore, coverage reports. Gate: "restore
into an isolated directory and verify source links/search; add a
book without recomputing unchanged sources; full-library launch
command and stop/resume runbook documented." Sliced so each piece
ships green during the full-run window.

### Slice 1 — backup / restore / verify (PRD §14)
- [DONE 2026-09-21] **Committed bff67a6, pushed to origin/master.**
  New `src/library_rag/backup.py` (+ tests/test_backup.py, 11 tests;
  CLI `backup`/`restore`/`verify` subcommands replacing the stubs;
  `Paths.backup_root: Path | None` in config.py — gitignored
  config.yaml points it at /mnt/models_sas_ssd/library-rag/backups):
  - `create_backup`: consistent state DB via SQLite's backup API
    (snapshot left in DELETE journal mode), Qdrant local storage
    copied only while the storage `.lock` is free (publication
    quiesced — refuses a locked client, remote mode, or a
    non-empty destination), archive/artifacts included, config
    last. Same-device files are HARDLINKED (nearly-free backups on
    the big disk); every file recorded in a manifest written LAST
    (sha256 per file; archive files are content-addressed so the
    path IS the checksum — `sha256_source: "path"`).
  - `restore_backup`: into a fresh isolated directory (force only
    for an existing EMPTY one), materializes every manifest file,
    and writes a rewritten self-contained `config.yaml` (5 roots
    re-pointed under the target, `qdrant_path` → target/qdrant,
    `backup_root` dropped, source_roots untouched). A live
    `api_token` is redacted to "REDACTED" in the backed-up copy.
  - `verify_system` (the M7 gate check): db.integrity (PRAGMA),
    archive.links + artifacts.links (size-checked), qdrant.index
    (active point count vs `SUM(expected_points)` of active
    publications), search.smoke (real embedder round-trip), and
    backup.checksums when a backup dir is given. Degrades
    explicitly without a Qdrant client.
  - **WAL-flip nuance:** `Database.connect` sets WAL, so the first
    open of a restored DELETE-mode snapshot rewrites header bytes
    18/19 — the state file's sha256 legitimately diverges from the
    backup after boot. `_is_wal_database()` (header check) makes
    `backup.checksums` size-only for such files; integrity is
    covered by the live db.integrity check.
- **Gate (2026-09-21, real output):** ruff "All checks passed!",
  mypy "Success: no issues found in 78 source files", pytest
  **405 passed in 46.53s** (2 anyio warnings, benign).
- Bugs found en route: `_same_device` crashed on the not-yet-created
  destination (compare against parent); `--force` into an existing
  empty dir hit FileExistsError (mkdir exist_ok); the WAL flip broke
  `backup.checksums` on the restored tree (see nuance above);
  leftover stub subparser registrations conflicted with the real
  M7 parsers (argparse ArgumentError) — stubs + dead helpers
  removed.

### Slice 2 — GC with reference checks (PRD §14)
- [DONE 2026-09-21] **Committed 33e6eb9, pushed to origin/master.**
  New `src/library_rag/gc.py` (+ tests/test_gc.py, 9 tests; CLI
  `gc` subcommand; archive.py docstring now points here):
  - `run_gc(db, cfg, *, execute=False, grace_seconds=600.0,
    now=None) -> GcReport`. Three kinds walked bottom-up:
    *archive* (`<archive_root>/<2hex>/<sha256>` — candidate iff no
    `source_revisions` row for that SHA; malformed layout →
    "malformed archive path"), *artifact* (extract units per
    `source_units`, embedding checkpoints per
    `embedding_batches`; a referenced `batch_NNNNN.bin`
    implicitly protects its `.json` sidecar via
    `checkpoint_manifest_path`), *job_log*
    (`<state_root>/job_logs/<job_id>.attempt<N>.log` — candidate
    iff the job row is gone; unrecognized filenames are never
    collected).
  - Safety: refuses (`GcError`) while any job is `running`;
    mtime grace window (default 600 s, `grace_seconds=0` allowed,
    negative rejected) covers the scan window in which archive
    bytes are copied BEFORE the revision row commits; dry-run is
    the default (`GcReport.candidates` with kind/relpath/size/
    reason), `--execute` unlinks and then prunes directories that
    became empty (bottom-up fixed-point — a `topdown=False`
    walk's dirnames list is stale mid-pass, so each pass
    re-checks live `iterdir()` until none changes). Per-file
    delete OSErrors land in `report.errors`, never raise. GC
    never touches source roots, the state DB, Qdrant storage, or
    scratch.
- **Gate (2026-09-21, real output):** ruff "All checks passed!",
  mypy "Success: no issues found in 80 source files", pytest
  **414 passed in 48.43s** (405 baseline + 9 new).
- Bugs found en route: `_prune_empty_dirs` left a parent dir
  behind after pruning its child (stale `os.walk` dirnames —
  fixed with the fixed-point loop); mypy no-any-return on a
  `row[...]` cast in the test; ruff UP035 (Callable →
  collections.abc) + F541 (stray f-prefix).
- Drain snapshot 2026-09-21 (while writing slice 2): OCR
  3349/6176, chunk 287/300, embed 282 pending, 0 failed, worker
  72176 alive.

### Slice 3 — explicit removal (PRD §14)
- [DONE 2026-09-21] **Committed 45e06ee, pushed to origin/master.**
  New `src/library_rag/removal.py` (+ tests/test_removal.py, 9
  tests; CLI `remove <path-or-doc-id> [--execute] [--json]`):
  - `resolve_target`: UUID-shaped target → `documents` lookup
    (case-insensitive), otherwise `normalize_path` →
    `path_aliases`; unknown → `RemovalError("no document …")`.
    Path and doc_id resolve to the same deterministic identity, so
    removal is repeatable (second run reports "not found").
  - `remove_document(db, cfg, qdrant, doc_id, *, execute=False)`:
    *Points first* (all generations, active and superseded, via
    `doc_id` filter) with a leftover-count check BEFORE the catalog
    is touched — a crash there leaves the catalog intact and the
    removal simply re-runs. Then one FK-safe catalog transaction
    (children before parents): `scan_state` (so a re-scan of a
    re-added file registers fresh), `path_aliases`, `publications`
    (FK to `gen_id`), `index_generations`, `chunks`,
    `source_units`, `embedding_batches`, `extraction_runs`,
    `source_revisions`, `documents`; open jobs
    (pending / retryable_failed) for the document's rev/run ids
    are cancelled in the same transaction. Last, the revision's
    archive object is unlinked only if a `source_revisions`
    recount for its SHA shows no reference beyond the document's
    own rows (dry-run `own_refs=1`, post-transaction `own_refs=0`).
  - Safety: refuses while any job is `running` (same guard as
    `gc`); refuses when the document has points but Qdrant cannot
    be opened (worker holds the local storage lock) — catalog-only
    removal allowed when the document has no points. Dry-run is
    the default (before-counts + the archive plan). `execute`
    then *verifies*: every catalog table filtered to the document
    must be empty and the point count zero, else
    `RemovalError` with the leftovers named.
  - Deliberately untouched: source files (re-scan re-registers
    deterministically), artifact files (unreferenced → `gc`
    reclaims with its grace window), `answers` (frozen evidence
    manifests are self-contained, PRD §12 — `answers.doc_id` /
    `rev_id` are plain columns with no FK), `sparse_corpus_stats`
    (next publish self-heals).
- **Gate (2026-09-21, real output):** ruff "All checks passed!",
  mypy "Success: no issues found in 82 source files", pytest
  **423 passed in 51.76s** (414 baseline + 9 new).
- Bugs found en route (both FK-ordering, both hit by the real
  embedded-Qdrant tests): `DELETE FROM extraction_runs` before
  `embedding_batches` (FK `embedding_batches.run_id` →
  `extraction_runs`), and `DELETE FROM index_generations` before
  `publications` (FK `publications.gen_id` →
  `index_generations` — migrations.py:269). Fixed by reordering
  to the topologically verified sequence above; the full
  15-`REFERENCES` schema audit confirmed nothing else (jobs,
  answers, sparse_corpus_stats, meta) blocks it. Also: ambiguous
  `doc_id` in a test JOIN (aliased), ruff RUF022 `__all__`
  sort.
- Drain snapshot 2026-09-21 (while writing slice 3): OCR
  4619/6176, chunk 291/300, embed 286 pending, 0 failed, worker
  72176 alive.

### Slice 4 — coverage report (PRD §2 / §14)
- [DONE 2026-09-21] **Committed 367ebc8, pushed to origin/master.**
  New `src/library_rag/coverage.py` (+ tests/test_coverage.py, 11
  tests; CLI `coverage [--json]`):
  - `coverage_report(db, cfg, *, qdrant=None)`: read-only, no DB
    writes, no file mutations, **no hashing** — staleness is the
    same stat-only size/mtime check the scan fast check uses.
    Qdrant is optional: when the worker holds the local storage
    lock the point counts degrade to `None` and everything else
    is still reported.
  - **Pipeline funnel** (PRD §14 vocabulary, `StageCounts`):
    documents; revisions (total + active); archived (revisions
    whose archive object is on disk — `archive_missing` lists the
    missing relpaths); extracted (revisions with a succeeded
    `extraction_runs`); OCR units by `ocr_state` (routed/done/
    pending/failed); chunked / embedded / indexed (distinct revs
    with chunks / embedding batches / ready generations, plus
    `SUM(point_count)` over ready generations); publications by
    state (active/staged/superseded) + distinct published docs;
    and index points: catalog expectation
    (`SUM(expected_points)` over active publications) vs live
    active/total counts (the backup-verify pattern: `active`
    payload flag — superseded points stay counted in total).
  - **Per-root coverage** (`RootCoverage`, scan semantics):
    `mount_unavailable` via the same sentinel + directory check
    as `scan_root` (nothing below a dead mount is meaningful);
    `discovered` (candidates via `iter_candidate_paths`);
    `registered` (alias rows under the root prefix);
    `unindexed` (valid-format candidates with no alias — on
    disk, never scanned); `orphaned` (alias rows whose file is
    gone — the scan's `missing` set, report-only); `stale`
    (`scan_state` rows whose size/mtime no longer match the
    file; a gone file is orphaned, not stale); `invalid`
    (magic bytes match no supported format — distinct from
    unindexed).
  - **Stalled documents** (`StalledDoc`): documents without an
    active publication, labeled with the furthest stage reached
    (indexed > embedded > chunked > extracted > registered),
    sorted furthest first.
  - **Failures** (`FailureCounts`): jobs by state, failed jobs
    by stage, failed `extraction_runs`, failed OCR units.
  - CLI: `library-rag coverage [--config X] [--json]`; text
    view = funnel line, OCR line, points line (degrades to
    "index: unavailable"), failures line, per-root lines with
    up to 10 examples each, stalled summary. Qdrant is opened
    defensively like `_verify` (open fails → None, no crash).
- **Gate (2026-09-21, real output):** ruff "All checks
  passed!", mypy "Success: no issues found in 84 source files",
  pytest **434 passed in 54.62s** (423 baseline + 11 new).
- Bugs found en route (both column-name, both caught by the
  real embedded-Qdrant tests): `embedding_batches` has no
  `rev_id` of its own (join through `extraction_runs`), and
  `index_generations` has no `doc_id` at all (carries `rev_id`
  → join through `source_revisions` for stalled-doc stage
  labeling).
- Live smoke (read-only, over the live batch2 sandbox, 2026-09-21):
  34768 discovered / 300 registered / 34108 unindexed / 360
  invalid / 0 orphaned / 0 stale; funnel 300/300/300 (extracted)
  → 286 chunked → 0 embedded/indexed/published (drain in
  progress — matches the drain snapshot); 300 stalled
  (286 chunked + 14 extracted); points active/total None
  (worker holds the Qdrant lock) as designed; 0 failed jobs;
  real wall time 6m41s. Invalid spot-check (5/5): zeroed PDFs,
  a REXX script, an AVI, and a `PK`-headed file that fails
  `zipfile.testzip()` (truncated) — all correct rejections, no
  false positives.
- Drain snapshot 2026-09-21 (while writing slice 4): OCR
  5073/6176 (1 running), chunk 291/300 (9 pending), embed 286
  pending, 0 failed, worker 72176 alive.

### Slice 5 — full-library launch + stop/resume runbook (PRD line 183)
- [DONE 2026-09-21] **New top-level `RUNBOOK.md`.** Docs-only slice
  (no code, no gate); every command, flag, and behavior in it was
  verified against the code before writing:
  - Layout table from the live (gitignored) `config.yaml`; services
    (8× llama-server bge-m3 on 8081–8088, vLLM :8091 — never
    reconfigure); the one-worker-at-a-time rule for local Qdrant
    mode (exclusive storage lock) and why read-only tools degrade
    while the worker runs.
  - **Launch**: doctor + 9-port health pre-flight, `scan`
    (fast check: size+mtime, no re-hash — adding books mid-run
    never recomputes unchanged sources), detached nohup `ingest`
    with the PID recorded, startup verification via `status`.
    Scale expectation: 34768 candidates (360 invalid) at the
    pilot rate ~11–13 jobs/min → the M6 ~41–45-day serial
    estimate; re-measure in the first hour.
  - **Monitoring**: `status` (jobs by state, OCR units, paused
    flag), `coverage` (funnel + roots + stalled + failures;
    point counts `null` while the worker holds the Qdrant lock),
    short `ingest_run.log` + kept-on-failure verbose job logs.
  - **Stopping** (three levels, gentlest first): `pause`
    (DB-persisted, survives restarts) → SIGTERM (finishes the
    in-flight job — `stop_event` is consulted between jobs —
    exits 0) → SIGKILL/power (lease TTL 300 s + fenced commits;
    WAL, no corruption).
  - **Resuming/crash recovery**: re-run the nohup command
    (automatic reconcile pass on start); `resume` after a
    pause; after a crash, expired-lease jobs reclaim to pending
    and handlers do not redo committed work (OCR is one job per
    page; chunking is chunk-fingerprint idempotent; embedding
    commits per batch, vectors-to-artifact-first). Each re-claim
    consumes one attempt (max 3) → crash-looped jobs land in
    `retryable_failed` → fix + `retry`.
  - **Failure triage table** (retryable/permanent, OCR unit
    failures, fleet outage, mount_unavailable, stalled docs,
    disk full) and the maintenance-safety split: read-only +
    `scan` + `retry` are safe with the worker running; `gc`,
    `remove`, `backup` require the worker stopped (gc/remove
    refuse while jobs run; backup refuses on the Qdrant lock).
- Drain snapshot 2026-09-21 (while writing the runbook): OCR
  5279/6176 (~85%), 0 failed, worker 72176 alive.

### Slice 6 — scheduled discovery (PRD lines 15/83/181)
- [DONE 2026-09-21] **Committed 119d796, pushed to origin/master.**
  New `src/library_rag/discover.py` (+ tests/test_discover.py, 6
  tests; CLI `discover` subcommand):
  - `run_discovery(db, cfg, jobs, *, interval_seconds=3600.0,
    stop_event=None, on_pass=None) -> int`: one scan pass
    immediately, then a fixed interval between passes until the
    stop signal. Each pass is the same streaming discovery as
    `scan` (PRD §8A: no tree in memory, ignore dirs, magic-byte
    validation, size+mtime fast check → unchanged books are not
    re-hashed; INSERT-OR-IGNORE enqueueing → no double jobs).
  - Stop semantics mirror `run_worker` (PRD §7): the flag is
    checked at the top of the loop and during an interruptible
    sleep (0.05 s ticks), so a pass never starts interrupted and
    an in-flight pass runs to completion — a read-only walk with
    idempotent per-file upserts, so even an interrupted pass is
    safe to repeat.
  - Adds-or-revises only: a file that disappeared since the last
    pass is *reported* in `ScanReport.missing`, never deleted
    (its alias rows stay; the coverage report flags it as
    orphaned). A scheduled pass therefore can only add or revise.
  - CLI `library-rag discover --config X [--interval SECONDS]`
    (default 3600 s — one full pass over the 34k-file library is
    ~5–7 min of I/O, so an hourly cadence keeps discovery
    overhead under ~10% while new books land within the hour).
    SIGTERM/SIGINT finish the in-flight pass, then exit 0 like
    `ingest` ("discover: N pass(es) completed"). A single pass
    is just `scan`; there is deliberately no `--once`.
  - Scans continue while paused: enqueueing is harmless — pause
    gates claiming, not enqueueing (jobs.py).
- **Gate (2026-09-21, real output):** ruff "All checks passed!",
  mypy "Success: no issues found in 86 source files", pytest
  **440 passed in 55.01s** (434 baseline + 6 new).
- What the tests pin: immediate stop (0 passes, nothing
  touched); one pass registers and enqueues exactly one extract
  job; a second pass is `unchanged` with no re-hash (boom on a
  monkeypatched `stream_hash`) and no double enqueue; a book
  dropped in mid-run is picked up on the next pass; the
  between-pass sleep is interruptible (stop at 0.15 s of a 10 s
  sleep → elapsed < 1 s, passes == 1); a deleted file is
  reported missing while its alias row stays.
- Drain snapshot 2026-09-21 (while writing slice 6): OCR
  5673/6176 (~92%), 503 pending, 0 failed; jobs 792 pending /
  1 running / 6269 succeeded; 81070 chunks, 0 embedding batches
  committed yet; 0 published docs; worker 72176 alive.

### Slice 7 — safe revision replacement + generation migration (PRD §14)
- [DONE 2026-09-21] **Committed 153ebcf, pushed to origin/master.**
  New `src/library_rag/migrate.py` (+ tests/test_migrate.py, 6 tests;
  CLI `migrate` subcommand); 5 new points-kind tests in test_gc.py;
  migration 0006 `publications_superseded_at`:
  - **Safe revision replacement closed**: B4's four supersede UPDATEs
    (publish switch ×2, reconcile B3+B4-completion ×2) now stamp
    `superseded_at` atomically with the state flip. `gc` learns a
    fourth kind, `points`: a superseded publication older than the
    grace window is a candidate whose object is its Qdrant point set
    — reclaimed by a filtered `pub_id` delete (no file I/O; the
    publications row stays as lineage). Without a Qdrant client (the
    worker holds the local lock) the kind is skipped with an explicit
    note rather than guessed at.
  - **Generation migration** `run_migration(db, cfg, *, execute,
    accept_maintenance_window) -> MigrationReport` — no Qdrant
    client at any point (state DB + storage-dir bytes only, so it
    works while the worker runs):
    * drift detection mirrors the worker's reconcile pass exactly:
      *rechunk* (stored `chunk_fingerprint` vs
      `chunk_fingerprint_for_run`; all succeeded runs, plus
      active-revision runs counted separately — only those
      re-enter the index) and *reembed* (embedding-batch checkpoint
      count under the current `embedding_sha` vs the expected
      `(n_chunks + batch - 1) // batch`); *supersede* = the active
      publications of the migrating revisions.
    * capacity (PRD line 190: two generations must fit, or a
      documented window): `second_gen = storage * migrating_points
      / active_points`, proportional to the *measured* store rather
      than chunks×dim×dtype, because the store still holds the
      not-yet-superseded generation. Unknown store size or free
      space (remote mode / stat failure) → `fits = None`, which
      counts as *not* fitting — an unverifiable window is an
      explicit window.
    * execution enqueues exactly what the canonical reconcile pass
      enqueues (idempotent; the job queue is the serialization
      boundary, the worker simply interleaves) and is refused
      without `--accept-maintenance-window` when the window is
      required, with the stop → `gc --execute` → re-check → retry
      recipe in the error.
  - CLI `library-rag migrate --config X [--execute]
    [--accept-maintenance-window] [--json]` — dry-run by default.
- **Gate (2026-09-21, real output):** ruff "All checks passed!",
  mypy "Success: no issues found in 88 source files", pytest
  **451 passed in 59.93s** (440 baseline + 6 migrate + 5 gc
  points-kind).
- What the tests pin: no-drift is a noop (0/0/0, 0 superseded,
  fits, nothing enqueued); a chunker change (target_tokens +1)
  re-chunks exactly one active run, supersedes its publication,
  and executing enqueues exactly the one chunk job whose handoff
  re-drives embed+publish from the new fingerprint; a dimensions
  change re-embeds; the fit estimate is proportional to the measured
  store (8 MiB store, all points migrate → second generation
  estimated at 8 MiB); a full disk (monkeypatched free=1024) refuses
  without acceptance and enqueues nothing; unknown capacity
  (nonexistent store dir) is treated as a required window that
  explicit acceptance still proceeds with.
- Bugs found en route: `publish_handbuilt` never stamped the run's
  `chunk_fingerprint` (NULL) — NULL is drift to the reconcile pass
  and therefore to migrate, so every hand-built library was
  "drifted"; the fixture now stamps `chunk_fingerprint_for_run`
  exactly as the chunk stage does. The supersede tests' second
  generation reused the existing doc_id, colliding with the
  fixture's plain `INSERT INTO documents` PK — now
  `ON CONFLICT(doc_id) DO NOTHING` (the doc row must stay the first
  generation's: B4 supersedes the doc's *other* active
  publications). mypy: a `Row` loop variable cannot be re-bound to
  `query_one`'s `Row | None` (renamed to `active_row`); the
  reembed count row is `assert`-guarded like the worker's.
- Drain snapshot 2026-09-21 (while writing slice 7): the machine
  rebooted ~02:17, killing worker 72176 mid-embedding (last log
  02:00:11, healthy 200 OKs to 8081–8088). Recovery: state DB
  integrity ok (WAL, PRAGMA), 300/300 runs succeeded, 0 failed
  jobs; both mounts rw, all 8 embedders 200. Worker relaunched
  02:43 (first new log line 02:43:13, embedding resumed); at
  relaunch: 294 jobs pending, 1 stale running (reclaimed at
  startup reconcile), 6799 succeeded.

### Slice 8 — logging configuration in the config file (operator ask)
- [DONE 2026-09-21] **Committed, pushed to origin/master.** The
  operator proposed a `logging:` section and it was missing from
  config.yaml; the CLI flags were the only knobs, so a detached
  nohup worker run had no persistent way to steer logging.
  - `LoggingSettings` in config.py (`level: "INFO"`,
    `format: "human"`, `job_log_retention: 500`) + a
    `Config.logging` field with defaults — existing configs keep
    working unchanged. Validators: level normalized to uppercase,
    only DEBUG/INFO/WARNING/ERROR; format only 'human'/'json';
    retention >= 1 — all raise `ConfigError` with a
    `logging.<key>` prefix, and `load_config`'s tail re-raises
    ConfigError unwrapped so the field names surface verbatim.
  - Precedence **explicit flag > config > built-in default**:
    `--log-level`/`--log-format` now default to None;
    `main()` calls `_apply_config_logging(args)` after
    parse_args and before dispatch — best-effort `load_config`
    (a missing/invalid config never blocks; the handlers'
    `_require_config` reports the real error) that fills only the
    None slots, defaulting to INFO/human. All ~30 existing
    `setup_logging(args.log_level, args.log_format)` call sites are
    untouched; resolution happens before any handler is set up,
    honoring setup_logging's fix-the-formatter-on-first-call
    constraint. The resolved level also feeds
    `uvicorn.run(..., log_level=...)`.
  - The failure-log retention cap is now config-driven: the
    worker's `prune_job_logs(..., limit=cfg.logging.job_log_retention)`
    (was a hardcoded 500).
  - `logging:` block documented in config.example.yaml (per-key
    comments) and added to the live config.yaml.
- **Gate (2026-09-21, real output):** ruff "All checks passed!",
  mypy "Success: no issues found in 46 source files", pytest
  **457 passed in 59.34s** (451 baseline + 6 new).
- What the tests pin: defaults (INFO/human/500); a config without
  a logging section gets the built-in defaults; lowercase levels
  normalize to uppercase; invalid level/format/retention raise
  ConfigError naming `logging.<key>`; a YAML file with
  `logging: debug/json/42` loads to DEBUG/json/42; a YAML file
  with `level: LOUD` raises ConfigError.

### Slice 9 — job version signature + cross-version execution gate (operator ask)
- [DONE 2026-09-21] **Committed (680b4c9), pushed to origin/master.**
  The operator asked: if the machine was rebooted and the software
  upgraded, the worker must not execute jobs from the previous
  version without explicit confirmation.
  - `software_version()` in versioning.py: SHA-256 content hash
    over the non-cache package files (`lru_cache`d per process),
    with a `software_version_for` test seam.
  - Migration 7: `jobs.created_by_version`, stamped at enqueue;
    `INSERT OR IGNORE` keeps the *original creator's* version on
    re-enqueue — the signature records who created the job, not
    who last tried it.
  - `Jobs.version_mismatch_counts(current)`: per-creator-version
    counts over not-yet-terminal jobs (pending/running/
    retryable_failed only — terminal states are never
    re-executed, so they are outside the gate). NULL =
    `legacy` (pre-signature jobs, necessarily foreign).
  - `check_version_gate` in `_ingest` (before reconcile/claim)
    and `run_latency`: raises `VersionGateError` with per-version
    counts unless the durable meta ack (key `version_gate_ack`,
    shape {to, from, at}) covers *all* foreign versions for the
    *current* version — a further upgrade re-triggers the gate
    even for previously-confirmed versions.
  - `--allow-version-mismatch` records the merged ack; `status`
    shows the software version plus foreign-job counts:
    `version gate: ok (software <content-hash>)` or
    `version gate: N job(s) from another software version
    (<version> xK ...) — run needs --allow-version-mismatch`.
- **Gate (2026-09-21, real output):** ruff clean, mypy clean,
  pytest **475 passed** (457 baseline + 18 new).

### Slice 10 — graceful stop + forced start (stale lock/pidfile/dead-lease cleanup) (operator ask)
- [DONE 2026-09-21] **Committed, pushed to origin/master.** The
  operator asked for an option for graceful shutdown and for
  forced startup (cleaning locks).
  - New `locks.py` module:
    - `pid_alive` — `os.kill(pid, 0)` probe plus a
      `/proc/<pid>/status` State check: a **zombie counts as
      dead** (it holds no locks; a long-lived unreaping parent
      would otherwise keep a crashed worker looking alive).
      PermissionError (another user's pid) counts as alive.
    - `is_ingest_worker_pid` — identity by argv, not substring:
      `library-rag` in the cmdline **and** `ingest` as its own
      argument (a config merely named `ingest-*.yaml`, or a
      `serve`/`discover` process, never matches).
    - `descendant_pids` — single /proc scan, BFS over PPid;
      lets `stop` look through a pre-slice-10 pidfile that
      names the `nohup`/`uv` wrapper.
    - pidfile helpers: the worker now writes its own
      `<scratch_root>/ingest_worker.pid` on start and removes it
      on clean exit (`remove_pidfile_if_ours` never clobbers a
      file naming someone else); `remove_stale_pidfile` removes
      **only** when the pid is provably dead — a live pid (even
      a reused one) keeps the file, so `stop`'s refusal for
      such a pid is not masked.
    - `find_flock_holders` — /proc/<pid>/fd readlink scan (the
      kernel keeps a lock's owner list private to itself);
      `kill_pids` — SIGTERM, escalate to SIGKILL after the
      timeout; `force_release_qdrant_lock` — non-blocking probe
      of `<qdrant_path>/.lock`; a foreign holder → RuntimeError
      **refusing** (not killing); a held lock with no discoverable
      holder → manual-investigation error. `QDRANT_LOCK_NAME`
      moved here (backup.py imports it from locks, avoiding a
      jobs→locks→backup→jobs cycle).
  - `library-rag stop [--timeout 300]`: no pidfile → error;
    dead pid → drop the stale file; live but not an ingest
    worker → refuse, showing its cmdline; otherwise SIGTERM the
    resolved worker(s) (pidfile pid + live ingest-worker
    descendants), SIGKILL survivors past `--timeout`, remove its
    own pidfile, print the job counts, and warn to `ingest
    --force` when jobs are still `running`.
  - `library-rag ingest --force`: runs **before** the version
    gate and normal startup work —
    `force_release_qdrant_lock` (kill a stuck ingest worker
    still holding the Qdrant local lock; a live hung holder is
    the only case, since the kernel releases flocks of dead
    processes automatically) → `remove_stale_pidfile` →
    `jobs.reclaim_dead_lease_owners()` (flip `running` jobs
    whose lease_owner pid — the trailing number of
    `worker_name()` — is provably dead back to `pending`
    immediately, instead of waiting out the 300 s lease TTL;
    live owners and unrecognized owner formats are left to the
    TTL). Startup then proceeds exactly as a normal start.
  - RUNBOOK updated: §2 launch note (the worker writes its own
    pidfile — do not create it by hand; the pre-slice-10
    `echo $!` line removed), §4 stop level 2 (the `stop`
    command, its refusal semantics, the raw-kill fallback for
    pre-slice-10 launches), §5 two new recovery bullets
    (`--force` and the post-upgrade version gate).
- **Gate (2026-09-21, real output):** ruff "All checks
  passed!", mypy "Success: no issues found in 48 source
  files", pytest **490 passed** (475 baseline + 15 new).
- What the tests pin (tests/test_ops_locks.py): a zombie
  counts as dead (waits for `State: Z`); the worker-identity
  check rejects this pytest process; pidfile roundtrip;
  stale removal is conservative (dead pid → removed, live
  pid → kept, the pid-reuse guard); `remove_pidfile_if_ours`
  leaves a foreign file alone; descendant BFS finds a
  grandchild; flock holders found from /proc; a **foreign**
  holder survives `force_release_qdrant_lock` (refusal,
  process alive); a stuck fake worker (a script literally
  named `library-rag`, argv `ingest <lock-dir>`) is killed
  and the lock released; a missing lock directory is a no-op;
  a dead-lease owner is reclaimed exactly once (second pass
  finds nothing); a live owner is left alone; an unrecognized
  owner format ("mystery-owner") is left to the TTL.

### Slice 11 — show-path-to-original config option (operator ask)
- [DONE 2026-09-21] The operator asked for a config option that
  shows the **full path to the original book file** in the reader
  pane (the right window, above the "locate in reader" content) —
  intended for local-machine runs only.
  - `Services.show_path_to_original: bool = False` (config.py) —
    off by default; a deployed app must never leak server
    filesystem layout.
  - `GET /books/{rev_id}` (reader.py `book_manifest`) adds
    `source_path` (from `source_revisions.first_path`, the
    registered absolute path) to the manifest **only when the
    flag is on** — when off the key is absent, never null.
  - Web UI: `BookManifest.source_path?` (api.ts); the reader pane
    renders a `#r-path` line (monospace, selectable,
    word-break) between the toolbar and the scroll area, driven
    from `this.resolvedManifests.get(s.revId)?.source_path` in
    `onReaderState` — every locate flow awaits the manifest
    before changing reader state, so it is always resolved when
    the state lands. Hidden when the server sent no path.
  - config.example.yaml documents the option (commented out).
  - Verified live: pilot-sandbox config flipped to
    `show_path_to_original: true`, serve app restarted (app pid
    90237, uv 90234, 2026-09-21); manifest for a pilot rev
    carries `source_path` equal to the DB `first_path`;
    no-token requests still 401; the rebuilt bundle
    (index-BGBKgYQW.js / index-CwpwoDEP.css) is what's served.
- **Gate (2026-09-21, real output):** pytest **492 passed**
  (490 baseline + 2 new reader tests; the config default
  assertion extends an existing test). New bundle built with
  `tsc --noEmit && vite build`, self-hosted, dist guard test
  green.
- What the tests pin (tests/test_reader.py,
  tests/test_config.py): default `show_path_to_original` is
  False; the manifest has **no** `source_path` key when the flag
  is off; with the flag on (cfg `model_copy` + fresh TestClient)
  the key equals the `source_revisions.first_path` row value and
  points at the ingested file.

### Slice 12 — Google button on citation lines (operator ask)
- [DONE 2026-09-21] The operator asked for a button on the very
  right of each reference line in the answer pane (the line with
  the `[n]` number, the reference title, and the location) that
  fires a Google search with the reference title as the query.
  - `evidenceItem` (web/src/app.ts): `evidence-head` gets a
    `.spacer` + a small "Google" button at the far right; the
    handler opens
    `https://www.google.com/search?q=<encodeURIComponent(ev.source_title)>`
    in a new tab (`noopener,noreferrer`).
  - No CSS changes (`.spacer` / `.btn.small` already existed).
  - Rebuilt bundle index-OKA8Gz2h.js; served live (static mount
    reads dist from disk — no serve restart; the app on
    0.0.0.0:8100 now serves the new JS; browser refresh picks it
    up). Gate: 492 passed, ruff clean, mypy clean, dist guard
    green.

### Slice 13 — configurable API-token requirement (operator ask)
- [DONE 2026-09-21] The operator asked to make the "API token"
  requirement **configurable, disabled by default**: off by default
  in the GUI, a fixed token value in config.yaml, and config.yaml
  enabling it in the GUI.
  - `Services.require_api_token: bool = False` (config.py) — the
    **master switch**, decoupled from `Services.api_token` (the
    value). Off (default): no request needs a token and the web UI
    never shows its token field. On: every non-static request needs
    `Authorization: Bearer <api_token>` and the UI shows the field.
  - The value is **inert while the switch is off** — a configured
    token no longer implicitly enables auth (old behavior).
  - Fail-closed at load: `load_config` raises ConfigError when
    `require_api_token: true` has no value (config or
    `$LIBRARY_RAG_API_TOKEN`, which still wins over the file value).
  - cli.py `_serve`: non-loopback bind with the switch off is now a
    **loud stderr warning** (deliberate operator choice) instead of
    a hard error; the loopback default is unchanged.
  - api.py middleware: enforcement requires
    `require_api_token and token` (previously "a value exists");
    `_OPEN_PATHS` + static assets stay open in both modes.
  - **No UI change needed**: the token field (app.ts) is already
    hidden until the first 401 — switch on ⇒ 401 ⇒ field appears;
    switch off ⇒ no 401 ⇒ field never shows. This is exactly
    "config.yaml enables it in gui".
  - config.example.yaml documents the two-part setup (fixed value +
    commented-out switch, with the warning behavior noted).
  - Verified live: pilot-sandbox config now carries fixed
    `api_token: "library-rag-local"` + `require_api_token: false`;
    serve relaunched **without** sourcing serve.env (config is the
    single source of the value; app pid 104264, uv 104260);
    /health, /library, /search all 200 with **no** Authorization
    header; the non-loopback warning fired in serve.log.
- **Gate (2026-09-21, real output):** pytest **495 passed** (492
  baseline + 2 new file-based config tests + 1 new API test; the
  existing auth test renamed and now sets the switch). ruff clean,
  mypy clean (48 files).
- What the tests pin: default `require_api_token` is False; switch
  on without a value → ConfigError at load; switch on with a value
  accepted; with the switch off a configured token value leaves
  /library and /search 200 (value inert); switch on + wrong token →
  401 (existing, renamed test).

### Slice 14 — token widget follows server state (operator ask)
- [DONE 2026-09-21] The operator: "if api token is not used, you
  should not show api token and its set field and button" — the
  header widget (label + password field + "Set" button) must not
  exist at all when the server does not enforce a token, not just
  start hidden.
  - Server: `GET /ready` (api.py) now also returns
    `token_required: bool(require_api_token and api_token)` — the
    same condition the middleware enforces.
  - Web (app.ts): new `tokenRequired: boolean | null` state fed by
    `loadReady()` (startup + 30 s poll). While `token_required` is
    false: the widget is force-hidden, `showTokenPrompt()` becomes
    a no-op (a stray 401 can never resurface it), and any stale
    stored token is cleared via `setToken("")` so it stops being
    sent. If the config flips back on, the next 30 s poll re-arms
    the field with no reload.
  - api.ts: `ready()` type gains `token_required`; `setToken("")`
    now removes the sessionStorage entry instead of storing "".
  - Rebuilt bundle index-Qqf7jg0J.js (CSS unchanged).
  - Verified live: serve restarted (app pid 109491, uv 109487);
    /ready carries `token_required: false`; /library 200 with no
    Authorization header; index.html serves the new bundle.
- **Gate (2026-09-21, real output):** pytest **496 passed** (495
  baseline + 1 new /ready test; 2 existing /ready expectations
  extended), ruff clean, mypy clean (48 files), dist guard green.
- What the tests pin: /ready reports `token_required: False` by
  default; with the switch on, /ready (a non-open path) is 401
  without the token and reports `token_required: True` with it.

### Next unfinished task
1. [DONE 2026-09-21] Slice 7 safe revision replacement +
   generation migration committed 153ebcf, pushed.
2. [DONE 2026-09-21] **All seven M7 slices shipped** — PRD line 181
   complete (discovery, safe revision replacement, explicit removal,
   generation migration, GC with reference checks, backup/restore,
   coverage reports) plus the line 183 runbook gate.
3. [DONE 2026-09-21] Slices 9 + 10 (operator asks) shipped:
   version signature + cross-version execution gate (680b4c9)
   and graceful stop + forced start (this commit).
4. Parallel track: batch-2 drain.
   - Timeline: worker 13396 relaunched 02:43 after the 02:17 reboot;
     published 32 revisions (03:13:10–03:53) before the ~13:00 reboot
     killed it (and the pid 15424 poller) and wiped /tmp — the batch2
     config died with it. Config reconstructed from the pilot-sandbox
     template with the five writable roots re-pointed under
     batch2-sandbox/ (`batch2-sandbox/scratch/config.sandbox.yaml`);
     drift-sensitive values (embedding revision/model/dimensions/dtype/
     normalize, chunking, normalization, `stats_epoch: frozen`)
     verified against the state DB's stored chunk_fingerprint /
     embedding_sha before relaunch. Worker relaunched 13:56:54
     (pid 24313).
   - **262→522 pending-publish anomaly — explained, benign.** The
     frozen corpus-stats epoch (stats_sha 8ce63e34…, 8 chunks from the
     first published book) was computed and committed at 03:13:10 —
     the sandbox's first publish. Every publish job enqueued before
     then carried the `"init"` hint (record absent; correct by
     construction): 55 created 01:54:34–02:55:14, of which 32 ran to
     success 03:13–03:53, and 239 created 02:52:02–03:13:09 left
     pending when the ~13:00 reboot hit. At the 13:56 relaunch,
     `_reconcile_index` found 262 gap revisions (no staged/active
     publication, full embed evidence) and enqueued 262 fresh publish
     jobs 13:57:00–02 with the frozen hint — different task key, so
     both families coexist per revision (the 262→522 jump). Under
     `stats_epoch: frozen` the drain is O(N) and convergent: all
     generations already share the one frozen epoch,
     `_corpus_stats_for_publish` reuses the pinned epoch,
     `_enqueue_epoch_republishes` finds no mismatches (no fan-out),
     and the redundant jobs no-op ("already current") when they run.
   - Snapshot 14:13 (atomic, single WAL read): 469 pending / 1
     running / 7,158 succeeded / 0 failed; 87 active publications
     (= 87 succeeded publish jobs, all init-hint; 32 pre-relaunch +
     55 post-relaunch, ~3.4/min); live publish lock
     `publish-b760one-24319`. Consistency: exactly 5 active revs have
     no publish job, all with 0 chunks / 0 embedding batches
     (legitimately nothing to publish). **Expected clean end state:
     295 active publications, 0/0/0 jobs.**
   - Drain-completion poller re-armed (Bash run_in_background, 60s
     poll, 30-min heartbeats, 6h cap, exits on 0/0/0).
   - **DONE 2026-09-21:** poller exited 0 on 0/0/0. Final state
     verified 16:13: 0 pending / 0 running / 0 failed, 7,628
     succeeded; 295 active publications — exactly the expected end
     state; zero failures across the whole drain. Idle sandbox
     worker stopped gracefully 16:13 (`stop`, SIGTERM, "jobs:
     succeeded=7628").
5. **Next: OPERATOR GATE — PASSED (user verdict 2026-09-21,
   post-reboot: "it was fine").** The interface test the
   full-library run was held behind is complete; what remains is
   only the user's explicit launch approval. Interface is live at
   http://192.168.0.30:8100/ (LAN, operator opt-in 2026-09-21 —
   `app_host: 0.0.0.0` in the pilot-sandbox config; serve app pid
   18534, uv 18530 — relaunched 2026-09-21 ~23:45 after the ~22:49
   machine reboot killed the previous app (pid 109491/uv 109487;
   clean "Shutting down" in serve.log), **without** sourcing
   serve.env; smoke: /ready all-true + token_required false,
   /library 300 books, search non-degraded, bundle
   index-Qqf7jg0J.js). The serve.log session from before the reboot
   shows the user's browser (192.168.0.100) loaded the new bundle and
   exercised the interface: 10× POST /answer 200, 17× /answers 200,
   reader unit reads + PDF source fetch 200 across several revisions
   (the earlier 401 block is from the pre-slice-13 run with the
   switch on);
   pilot-sandbox config → the 300-book stratified sample,
   seed 1337; pilot jobs 15,142 succeeded / 2 permanent_failed —
   known pilot issues). **Token requirement is OFF** (slice 13):
   sandbox config carries `require_api_token: false` + a fixed
   local `api_token` value, so the API is open — /library and
   /search 200 with no Authorization header — and the web UI
   shows **no** token field. Re-enabling is a one-word flip to
   `require_api_token: true` in the sandbox config (the API then
   enforces it and the UI shows the token field on the first 401,
   token stored in sessionStorage); the 0600 serve.env is no
   longer sourced by the serve command.
   The batch-2 sandbox (second 300-book stratified sample) is also
   fully drained and available if the user prefers testing against
   it. After the user's explicit go: full-library launch per
   RUNBOOK §2 (doctor → 9-port health 8081–8088+8091 → df -h →
   scan → `nohup uv run library-rag ingest --config config.yaml >>
   /mnt/models_sata_ssd/library-rag/scratch/ingest_run.log 2>&1 &`,
   worker self-writes its pidfile → status verify → re-measure
   first-hour rate, ETA ~41–45 days at ~11–13 jobs/min). Do not
   start it without the user's explicit approval.

## M8 — Extended per-book resumes + keyword search

**Status: gate passed** (505 passed, ruff + mypy clean, dist guard
green). Feature requested by the operator: an extended English
summary (~700–1000 words) for every processed book, plus a web-UI
tab that keyword-searches the stored resumes to find books related
to a subject.

### Delivered
- `migrations.py` — M8: `book_resumes` (rev_id PK, doc_id, run_id,
  title, text, word_count, model_revision, prompt_version,
  created/updated) + `idx_resumes_doc` + standalone FTS5 table
  `resumes_fts` (unicode61; bm25-ranked, lower score = better).
- `config.py` — `ResumeSettings`: `enabled` (true), `max_tokens`
  (2400), `temperature` (0.3), `timeout_seconds` (300),
  `prompt_version` ("resume-v1"), `input_char_budget` (16000);
  ConfigError on invalid values. Model identity always from
  `cfg.answer`; generation params from `cfg.resume` (longer/looser
  than cited answering). Documented in `config.example.yaml`.
- `resumes.py` (new) — `make_resume_model` (LlamaCppAnswerModel
  with the resume generation params; None when the answer model is
  unconfigured → per-job `answer_model_not_configured`),
  `sample_resume_input` (head + ≤16 evenly-spaced middle chunks +
  tail of the run, bounded to the char budget, reading order),
  `build_resume_messages` (prompt contract `resume-v1`),
  `store_resume` (one transaction: FTS delete+reinsert + upsert;
  returns word count), `enqueue_missing_resumes` (idempotent
  reconcile/backfill over active revs with active publications,
  keyed on the latest succeeded run; gated on
  `resume.enabled and answer.is_configured`), `search_resumes`
  (OR-of-quoted-terms FTS MATCH, bm25 rank, ~280-char excerpt),
  `get_resume`.
- `worker.py` + `scan.py` — `STAGE_RESUME = "resume"`;
  `_run_resume` handler (run must exist + succeeded, else defer;
  word count < `MIN_RESUME_WORDS` (100) → permanent
  `resume_too_short` — operator bumps `resume.max_tokens` and
  retries with `retry --include-permanent`; model errors →
  transient/non-transient per the existing LLM error classes);
  `_reconcile_resumes` reconcile pass; publish hook enqueues the
  resume job for newly published revisions; `run_worker` gains the
  `model` kwarg.
- `api.py` — `POST /resumes/search`
  (`ResumeSearchRequest(query 1..400, limit 1..50, default 20)`),
  `GET /resumes/{rev_id}` (404 when none stored);
  `/ingest/status` reports the `resumes` count.
- `cli.py` — `library-rag resumes` backfill subcommand (prints the
  enqueue count; `--json`).
- `web/` — third top-level view "Resumes" (nav button alongside
  Research/Ingestion): keyword search box (Enter or button) →
  ranked result cards (rank + title + bm25 score + excerpt);
  clicking a card fetches the full resume into the right pane
  (word count / model / prompt version line) with an "Open book"
  button that deep-links into the reader at page 1 / first
  section. **Also fixed a pre-existing latent bug**: the author
  rule `main > section { display:flex }` beats the UA stylesheet's
  `[hidden]{display:none}` in the cascade, so the `hidden`
  attribute was visually inert and both old views rendered
  stacked; the new `main > section[hidden] { display: none }`
  rule restores view toggling for all three views. Rebuilt bundle
  **index-CZW7vhZI.js** + index-BHUCIYHa.css.

### Test commands + results
| Command                          | Result                            |
| -------------------------------- | --------------------------------- |
| `uv run ruff check src tests`    | All checks passed!                |
| `uv run mypy src`                | Success: no issues found in 49 source files |
| `uv run pytest`                  | 505 passed (496 → 505: 9 new)     |

New tests (`tests/test_resumes.py`): M8 applies fresh + idempotent
re-run; end-to-end generation through the real worker loop
(durable job, scripted model, FTS mirror, title derivation,
manifest); enqueue idempotency + both gates; search API with real
bm25 ranking (dense-hit before sparse-hit, lower score better) +
full record + 404/422; sampling head/middle/tail in reading order;
ResumeSettings defaults + ConfigError on bad values.

### Live pilot verification (2026-09-22, pilot-sandbox)
- Pilot state DB migrated to schema v8 (online backup taken first,
  under the sandbox scratch dir). `library-rag resumes` enqueued
  **292** jobs (the pilot's 292 published revisions).
- Worker (new code) processed them against the real vLLM
  qwen3.8-27b: first landed resumes were 926 / 887 / 911 words —
  in the 700–1000 band; all six measured so far are 887–1028
  words and read as faithful, structured English summaries
  (title/author/contents/arguments/significance).
- `POST /resumes/search {"query": "drawing"}` returned ranked
  results (two drawing-instruction books above a political book
  that merely mentions "drawing"); `GET /resumes/{rev_id}` the
  full record; `/ingest/status` reports `resumes: 6` at
  snapshot time.
- **Local-Qdrant constraint re-confirmed**: the embedded Qdrant
  client is single-process — serve and the ingest worker cannot
  run at once (second process fails with "Storage folder … already
  accessed"). The pilot backfill is therefore paused while serve
  holds the lock for the operator's web-UI test; the remaining
  resume jobs are durable pending (286 at snapshot) and resume
  from where they stopped once the worker is relaunched
  (reconcile + idempotent enqueue).

### Cost note (full library)
~34,768 published revisions → ~34,768 resume LLM calls, serial on
the single vLLM (one resume ≈ 15–30 s of generation at ~900
tokens) → roughly **+3–5 days** on the full-run ETA. The stage is
durable and gated (`resume.enabled: false` skips it entirely), and
re-chunking / a prompt-version bump mints fresh jobs without
touching stored resumes until the new one succeeds.

### Post-verification addendum (2026-09-22 evening)
- Operator verdict on the web UI: passed ("it was fine"). A
  "Summary" button was then added to every evidence line in the
  answer pane (left of the Google button) opening that book's
  stored resume in the Resumes view, with a graceful
  "No summary stored for this book yet." on the 404 that is
  normal while the backfill runs (a87df16, bundle
  index-CukkH7fb.js).
- The pilot backfill (worker up, serve down for the local-Qdrant
  lock) crashed at 100/292 on a `ZeroDivisionError` in
  `sample_resume_input`: a 3-chunk book makes `_even_indices(3)`
  return `[]`, and the mid budget was divided by `len([])`.
  Fixed (guard: degrade to head+tail when there is no middle)
  with a 1/2/3-chunk regression test, gate 506 passed
  (ee1f497). The cross-version guard then held the 192
  remaining jobs (enqueued under the pre-fix hash) until the
  worker was relaunched with `--allow-version-mismatch` — the
  documented confirm-after-upgrade flow; job semantics were
  unchanged by the fix.
- One `permanent_failed` resume: a single-chunk book that
  generated 99 words (< the 100-word floor, `resume_too_short`)
  — by design non-transient. Superseded by the floor change
  below: at the new 20-word floor that 99-word generation is a
  valid résumé, so the book now needs
  `library-rag retry --include-permanent` (it will NOT auto-retry
  on worker restart: `Jobs.enqueue` is `INSERT OR IGNORE` on the
  task key, so the permanent_failed row stays put until an
  explicit retry resets it) once the worker is back on the new
  code.

### Floor addendum (2026-09-22, operator testing)
- Operator ask: "i can't accept a book without resumé just cuz
  it's under 100 words, the floor must be 20 words" —
  `MIN_RESUME_WORDS` lowered 100 → 20 (6d086c2). The 700–1000
  word prompt target in `build_resume_messages` is unchanged;
  only a genuinely truncated generation (< 20 words) now
  permanently fails with `resume_too_short`, and single-chunk
  micro-books get short summaries instead of no résumé. The
  worker's failure message already interpolated the constant (no
  second edit point). New two-sided regression test in
  `tests/test_resumes.py` (19 words → permanent_failed + nothing
  stored; 21 words → succeeded + stored); gate 507 passed,
  ruff + mypy clean.

### Summary-button test addendum (2026-09-22 evening)
- While serve was up for the operator's live test, two UI bugs
  were reported: "Open book" in the Resumes view did nothing,
  and the Summary button "showed nothing" even though a summary
  looked like ~1000 words (first thought to be epub-specific,
  then "the same with pdf — flacky, very often shows nothing").
  Root cause in `openResume`'s 404 path (normal mid-backfill):
  it hid the notice in the small left-pane status line, left the
  PREVIOUS book's meta line ("N words · model") over an EMPTY
  text area, and never updated `resumeState` — so "Open book"
  later targeted the previously loaded book or hit the null
  check. `openResumeBook` also gave no feedback during the slow
  whole-PDF source fetch, so the click looked dead. Fixed: the
  404 message is written into the right-pane text itself
  ("No summary stored for this book yet — the résumé backfill is
  still running. Use Open book to read the book itself in the
  meantime."), stale meta cleared, `resumeState` pointed at the
  clicked book, and an "Opening "<title>" …" reader status shown
  immediately on click (05eee57, bundle index-DjkRTiRs.js). A
  read-only DB check showed the misses are mostly DATA, not an
  epub bug: pdf 54/210 (25%) vs epub 67/82 (81%) have stored
  resumes because the paused backfill (121/292) had not reached
  most PDFs. The operator must hard-refresh the browser to pick
  up the new bundle (serve reads web/dist from disk, no restart).
  Gate: 507 passed, ruff + mypy clean.

### Backfill addendum (2026-09-22 late night)
- Operator said "done testing"; the swap ran as planned: serve
  stopped (8100 free); `library-rag retry --include-permanent`
  requeued 3 jobs — the 99-word book's resume plus ibm.pdf's
  permanently-failed embed + publish (a pilot book that had
  never published); worker relaunched in the background with
  `--allow-version-mismatch` (appends to worker-m8.log under the
  state dir); coarse backfill monitor armed (20-min ticks;
  re-arms at the 30-min expiry cap).
- One anomaly: sieglove.pdf (rev 5ccbe0fd, Wagner sheet music)
  hit `answer_model_error` "malformed chat completion response:
  content is not a string" — a one-off bad vLLM body (HTTP 200,
  non-string `content`) amid ~173 good generations. Requeued
  that single job via `retry --include-permanent`; the running
  worker retried it: succeeded, 215 words stored (short, valid
  for sheet music, above the 20-word floor). Not systematic for
  that book.
- ibm.pdf (rev c59f8aa7) published at 22:55 local; the publish
  hook minted its own resume job — the pilot total is 293, not
  292. A duplicate pending publish job for the same rev (a
  no-op re-publish once it runs) remains as the only non-resume
  open job.
- State at save: 187/293 stored, 105 pending + 1 running,
  0 failed at any stage; ~44 s/job → ETA ~1.5 h.

### Backfill complete (2026-09-23 late night)
- The last book was 362.PDF (rev 7f10c579): under the old code
  it had permanently failed on its single attempt with
  `answer_model_error` — a vLLM 200 whose `message.content` was
  `null` (the thinking model spent `max_tokens` inside the
  reasoning trace), misclassified as a permanent malformed
  response. The hardening fix (see M9) requeues it as transient;
  under the new code it succeeded on the first attempt in 30 s
  — 846 words stored.
- Final state (state DB, verified): **293/293 resume jobs
  succeeded, 293 resumes stored, 0 failed**; the ibm.pdf
  duplicate no-op publish job also completed — **15438/15438
  jobs of every stage succeeded, zero non-terminal, zero
  failed** anywhere in the pilot.
- Worker stopped gracefully (`stop`); serve relaunched on 8100;
  smoke green: `/health` ok, `/ready` all true (qdrant,
  embedding, answer), `POST /resumes/search` ranked results
  with excerpts, `GET /resumes/{rev}` full record (750 words,
  `qwen3.8-27b-fp8@vllm-dual-max`, `resume-v1`), unknown rev →
  404. The web UI Resumes tab works end-to-end on the live
  sandbox.
- Pilot résumé coverage is now 100%. The web-UI test the
  operator wanted is fully exercisable (all 293 books have
  resumés; search + "Open book" deep-link verified).

### Next unfinished task
- **Full-library launch remains HELD on the user's explicit
  approval** (M7 §5) — do not self-launch. M8 adds the
  `resume` stage to that run automatically via the publish
  hook; M9 (second LLM endpoint + worker concurrency 2 +
  thinking-model hardening) is deployed to the sandbox and in
  the tree, so the full run will use both endpoints and drain
  the LLM-bound tail at roughly twice the rate, with the
  null-content retry protecting against thinking-budget
  exhaustion.
- The full-library config must mirror the sandbox's M9
  settings: `answer.extra_endpoints` (host `ak`, port 8080,
  model renamed to match the primary) and
  `worker.max_concurrent_jobs: 2`.

## M9 — Second LLM endpoint + worker concurrency (2026-09-23)

**Status: deployed to pilot sandbox, gate passed** (521 passed,
ruff + mypy clean). Operator offered a second vLLM serving the
same model (`qwen3.8-27b`, FP8) on `http://ak:8080` ("you may
use a second vllm for processing, when it's available. right
now it IS available"). Design goal: drain LLM-bound backfills
(future: the full library's ~34,768 résumé calls) at roughly
twice the rate, without breaking the single-endpoint
deployments — an endpoint that is off must cost nothing but a
fast connect timeout.

### Delivered
- `llm.py` — `AnswerModelPool`: round-robin over N endpoints
  serving the same model; on any `AnswerModelError` (including
  `AnswerModelUnavailableError`) fails over to the next
  endpoint for the SAME call; raises the last error if every
  endpoint fails; thread-safe (lock-protected round-robin
  counter, safe to share across the worker's job threads);
  rejects an empty pool. `build_answer_pool(cfg, *,
  timeout_seconds, max_tokens, temperature)` — never raises:
  unconfigured → `None`, `answer.fake` → `FakeAnswerModel`, no
  extra endpoints → the bare primary (byte-identical to the
  pre-M9 path), otherwise the pool. `make_answer_model` is now
  a thin wrapper over it.
- `resumes.py` — `make_resume_model` likewise wraps
  `build_answer_pool` with `cfg.resume` generation params, so
  the `resume` stage load-balances across the pool too (it is
  the LLM-heavy stage M9 exists for).
- `config.py` — `AnswerEndpoint` model (`host`, `port`,
  optional `model_name` — inherits `answer.model_name` when
  omitted; validator rejects empty host / port out of range),
  `AnswerSettings.extra_endpoints` (default empty), and a new
  top-level `WorkerSettings` section: `max_concurrent_jobs`
  (default `1`, validated `>= 1`). `config.example.yaml`
  documents both.
- `worker.py` — `run_worker` accepts `worker.max_concurrent_jobs`:
  `1` keeps the original strictly-sequential loop (extracted
  unchanged into the shared per-job handler `_handle_claimed`,
  so the default path's behavior is byte-identical); `> 1`
  claims jobs into a `ThreadPoolExecutor(max_workers=N)` and
  runs their handlers concurrently, draining futures as they
  complete, still respecting `once` and `stop_event`. Safe by
  construction: `jobs.claim` is one transaction on the
  shared, lock-protected SQLite connection, and every handler's
  lease fencing is per-job.
- `indexing.py` — `RealQdrantOps` gains a `threading.RLock()`;
  every public client method is guarded with a small `@_locked`
  decorator. qdrant-client's embedded (local) mode has no
  internal locking, and concurrent publish jobs would otherwise
  interleave client calls; the remote backend is fine with the
  lock too (it holds for one request).
- Thinking-model hardening (the fix that unblocked 362.PDF,
  the pilot's last résumé): `LlamaCppAnswerModel.complete`
  now classifies a 200 response with `message.content: null`
  as `AnswerModelUnavailableError` ("model returned null
  content (thinking budget exhausted?)") — transient, so the
  job retries with backoff and the pool can fail over —
  instead of the permanent `AnswerModelError` it was before.
  A non-null non-string content is still permanent (a broken
  server, not a budget condition). This is a per-call budget
  mode of qwen3.8-27b under vLLM: it spends `max_tokens`
  inside the reasoning trace and emits `content: null`.
- Tests: `tests/test_llm.py` (pool round-robin, per-call
  failover on unavailable AND on malformed, last-error when
  all fail, len/revision, empty rejected; the null-content
  regression with the 362.PDF history in the comment; factory
  builds a pool only when `extra_endpoints` is configured),
  `tests/test_config.py` (endpoint/worker validation: port
  range, empty host, `max_concurrent_jobs < 1`),
  `tests/test_resumes.py` (the resume stage picks up
  pool-scripted failures via `FakeAnswerModel` script pops —
  a pool failure still ends in the right job state).
- Deployed to the pilot sandbox: `config.sandbox.yaml` (never
  committed) carries `answer.extra_endpoints: [{host: ak,
  port: 8080}]` and `worker.max_concurrent_jobs: 2`. The
  worker ran the new code for the final drain; 362.PDF
  succeeded on it in 30 s (primary `127.0.0.1:8091` answered
  that call — no failover was needed, but the pool was live).
  `ak:8080` was verified serving the same `model_name`.
  Sandbox recovery note: if `ak` is off at generation time,
  the pool degrades to the primary (one fast connect timeout
  per affected call) — no config change required.

### Known behavior
- The round-robin counter advances once per `complete` call
  even when it fails over, so a dead endpoint is retried on
  average once per N calls — the intended probe cadence.
- Concurrency `> 1` means the worker no longer bounds total
  Qdrant traffic to one job; the `RealQdrantOps` lock
  serializes the client, not the load. Embed-stage jobs are
  unchanged (they hit the 8 bge-m3 servers, not the pool).
- `max_tokens: 1024` (answer) and `2400` (resume) leave
  thinking models with a real chance of burning the budget in
  reasoning; the null-content retry absorbs that at the cost
  of one backoff per occurrence. If a book's résumé keeps
  coming back null, raise `resume.max_tokens` — the
  `resume_too_short` floor (20 words) still guards quality.

### Addendum — dual-LLM test batch of 20 ePUBs (2026-09-23, post-reboot)

Operator ordered a 20-ePUB test batch run through both LLMs
(local vLLM `127.0.0.1:8091` + `http://ak:8080`). The batch
surfaced and fixed two client-side bugs, then exposed one
server-side condition on `ak` that no client change can fix.

**Root cause 1 — per-job pool minting starved the second
endpoint (fixed, 3fd4587).** The CLI ingest path called
`run_worker` with no model, so each resume job minted a fresh
`AnswerModelPool`; a single-call job's round-robin always
started at endpoint 0 (the primary), so `ak` never received a
request ("no traffic on ak"). Fix: `run_worker` builds the
resume model once per run and shares it across the job
threads. Regression test:
`test_resume_model_built_once_and_shared`.

**Root cause 2 — empty-string content from thinking models
(fixed, 9ae74c0).** `ak`'s llama.cpp returns 200 with
`content: ""` (vLLM returns `content: null`) when the
thinking trace exhausts `max_tokens`. `""` passed the
`isinstance(str)` gate → 0 words → permanent
`resume_too_short` with no retry. Fix in
`LlamaCppAnswerModel.complete`: empty/whitespace content is
now `AnswerModelUnavailableError` (transient → in-call pool
failover + backoff retry). Round-1 arithmetic proof: 9 ak
calls burned exactly 2400 predicted tokens each
(7200/3=2400 per the earlier 3-job probe) — the whole
budget in reasoning.

**Round 1 (20 ePUBs, sandbox `resume.max_tokens` 2400 → 8192
in `config.sandbox.yaml`):** 9 succeeded (all primary), 9
permanent-failed `resume_too_short` (all routed to `ak` at
2400 — pre-9ae74c0 behavior replayed by the requeue tag),
2 permanent-failed `no_sample_text` (two ePUBs with no
extractable text — correct terminal behavior, not re-run).
`ak` received 9 real requests; its Prometheus metrics
moved 74 → 10860 tokens, confirming live traffic.

**Round 2 (the 9 re-queued under `dual-llm-regen2:{run_id}`
task keys, new tags because terminal jobs never re-execute):**
9/9 succeeded, word counts 826–1026 (band respected). The
worker log shows 12 model calls for 9 jobs: 3 requests to
`ak` still came back empty at 8192, and each was followed
~25 s later by a primary call (04:59:41→05:00:08,
05:04:41→05:05:06, 05:07:30→05:07:55) — the committed
in-call failover worked exactly as designed; the jobs
succeeded on vLLM. `ak` metrics across round 2: prompt
26746→30080, predicted 21861→52139 (+30278 predicted /
3 requests ≈ 10092/request — more than the 8192 requested).

**Root cause 3 (server-side, on `ak`; NOT fixable from the
client):** `ak`'s llama-server does not honor the request
`max_tokens` as a generation cap. Its metrics show a
`n_tokens_max 11634` context ceiling and ~10092 predicted
tokens per request (≈1111 prompt + thinking trace running to
the context edge, zero visible content). The Qwen3 `/no_think`
prompt toggle was probed with the exact round-2 request for
'Billy Crystal - 700 Sundays' but the probe was aborted per
operator instruction (stop burning tokens on `ak`) before a
response — no verdict. Recommendations for `ak` (any one):
raise `--ctx-size`, confirm `/no_think` is honored by the
served checkpoint, or set a server-side reasoning budget.
Until then the dual-endpoint design still delivers its value:
`ak` absorbs load, and any budget exhaustion fails over in-
call with one extra ~25 s of latency.

**Batch net:** 18/20 books have freshly regenerated resumes
(826–1026 words); 2 are unsummarizable (no extractable
text). `book_resumes` total 331 (round-2 rows are upserts on
existing revs, so the total is unchanged by design).

**SHA deduplication (operator request, verified existing
since M1):** no new code needed. `scan.py` computes a
`stream_hash` per file and `register_source` enforces
`UNIQUE(doc_id, sha256)` with `path_aliases` for renamed
copies; a renamed identical file re-registers to the existing
source instead of reprocessing. Pilot check: 40/40 files in
the test batch have unique SHAs; the two "Piano" books are
genuinely different files (bytes differ).

**Final state:** worker stopped cleanly after the drain
("ingest: 9 job(s) completed"); queue at 0 non-terminal;
serve relaunched on 8100 (`/ready` all true,
`POST /resumes/search` ranking verified); sandbox config
keeps `resume.max_tokens: 8192`. Gate at 9ae74c0: 524
passed, ruff + mypy clean (94 files). Full-library launch
still held on explicit operator approval; its config must
mirror the sandbox (`extra_endpoints` ak:8080,
`max_concurrent_jobs: 2`, `resume.max_tokens: 8192`) and,
for `ak` to contribute real output rather than only
failover traffic, the `ak` server-side condition above must
be addressed first.

### Addendum — root cause 3 resolved: `--reasoning-budget 4096` on `ak` (2026-09-23)

Operator pointed at `ak`'s launch script (`/cygdrive/d/llama.cpp/claude.sh`,
Cygwin shell on Windows) and asked for its arguments to be fixed so the
project can use the endpoint properly.

**Diagnosis corrected.** The earlier "context ceiling" reading was a
misread: `llamacpp:n_tokens_max` is a *largest observed sequence length*
counter (prompt + generation), not a context limit — on the freshly
restarted, idle server it read 0. The request `max_tokens` WAS honored
all along (probe: `finish_reason: length`, `completion_tokens` exactly
8192, content 0 chars, reasoning 30,828 chars). The real condition:
the Qwen3.8-27B thinking mode spends the entire `max_tokens` budget
inside the reasoning trace, leaving nothing for content.

**Fix (server-side, on `ak`):** added `--reasoning-budget 4096` to
`claude.sh` — a hard per-request cap on thinking tokens (this build
supports it: `-1` unrestricted, `0` immediate end, `N` cap). A 4096
thinking cap leaves ≥ 4096 of the 8192 resume budget for content
(~1.5k tokens needed for 700–1000 words). Also merged a duplicate
`--alias` line: the old script passed `--alias qwen3.8-27b` and a later
`--alias local-metrics`, and only the last one took effect (the model
was served as `local-metrics` only); now `--alias qwen3.8-27b,local-metrics`
so both names resolve. Pre-fix backup: `claude.sh.bak-20260923` on `ak`.
The operator restarted the server with the fixed script (project rule:
`ak`'s server is never restarted from the client side).

**Shared-server note:** `ak:8080` also serves the `claude-llama-proxy`
(127.0.0.1:8787) used for interactive Claude Code sessions, so the
4096 thinking cap applies there too. Revert path if that feels tight:
bump `--reasoning-budget` to 8192 (one line) and raise the sandbox
`resume.max_tokens` accordingly.

**Verified with exactly one production-shape request** (same book as
the probe, job 32472 "Billy Crystal - 700 Sundays", `max_tokens` 8192,
temperature 0.3, no `/no_think`): 200, `finish_reason: stop`, 144.9 s;
usage 3,322 prompt + 5,468 completion; reasoning 17,544 chars (≈3.5k
tokens — under the 4096 cap); **content 6,588 chars ≈ 1,000 words,
non-empty and well-formed** (proper book summary, head and tail
intact). The model now thinks within budget and finishes naturally
before `max_tokens`.

**Consequence:** `ak` now produces real content — the dual-endpoint
design delivers its intended ~2× throughput for LLM-bound stages, with
in-call failover as a safety net rather than the only reason jobs
succeed. The full-library launch config is unchanged, but `ak` will
now contribute output, not just failover traffic.

### Addendum — ak-batch3: first 10-job dual-endpoint batch after the fix (2026-09-23)

Operator-approved small end-to-end batch through the real `AnswerModelPool`
(primary vLLM `127.0.0.1:8091` + fixed `ak:8080`): 10 published revisions
(5 ePUB + 5 PDF, all already resumed → rows upserted, `book_resumes` total
stayed 331) re-enqueued under the new task-key tag `ak-batch3`, because
terminal jobs never re-execute. Exactly 10 LLM calls, no more.

**Result: 10/10 succeeded, zero failures, zero in-call failovers.** All
11... all 10 HTTP calls returned 200 on the first try. Per-endpoint split
from the worker log: **5 calls on vLLM (avg ~78 s) + 5 on ak (avg ~142 s)**
— ak's real content output, ~1.8× slower than vLLM. Batch wall ~10 min
with `max_concurrent_jobs: 2`; the RR dispatch self-pipelined (each freed
thread re-hits its own endpoint), so both endpoints ran idle-zero —
steady-state throughput is the sum of both capacities.

Word counts (band 700–1000; model is free-form, only a 100-word floor is
enforced): 794, 833, 868, 887, 904, 922, 964, 1019, 1172, 1267 — 7 in band,
3 slightly over (1019/1172/1267), none under. Both endpoints' output lands
in the same range (vLLM 887–1172, ak 794–1267), confirming the fixed ak is
quality-equivalent, not just reachable.

Sandbox serve was found down at batch start (graceful shutdown in the log;
satisfied the single-process Qdrant lock swap) and was relaunched after the
batch — `/health` ok, `/ready` all true, `/resumes/search` smoke hit returns
the batch's fresh rows. Worker stopped cleanly after the drain.

Open design question (operator's): endpoint weights for the pool. Analysis:
with concurrency == endpoint count, static RR is already near-optimal in
steady state; a weight would only shave finite-batch tails (~0.3% on the
331-job library). The knobs that would actually do something: a
per-endpoint `max_inflight` cap (protects interactive Claude Code traffic
sharing `ak:8080` during long backfills) or weighted least-inflight
(future-proofs `max_concurrent_jobs` > endpoint count). Decision pending.

### Addendum — least-inflight dispatch with per-endpoint weights (2026-09-23)

Operator decision on the open weights question: **option 2 — weighted
least-inflight** (replacing pure round-robin in `AnswerModelPool`), plus the
mid-turn instruction to configure `ak` as the 2×-slower endpoint right away.

**Mechanics:** the pool now tracks in-flight calls per endpoint and picks
`argmin(inflight_i / weight_i)` by cross-multiplication, with round-robin as
the idle tie-break (sequential calls still alternate `a,b,a,b` when nothing
is in flight — M9 behavior preserved; a down primary is still probed every
other call). At equilibrium (Little's law) `inflight_i / weight_i` is equal,
so dispatch settles at `weight_i × speed_i`: with all weights 1 the pool is
pure speed-proportional automatically (measured W_v/W_a ≈ 1.82 → ≈65/35);
weights are a deliberate distortion on top. **Knob direction:** raise the
PRIMARY's weight to shed batch load off a secondary that also serves
interactive traffic (the usual case).

**Config:** new `answer.weight` (primary) and
`answer.extra_endpoints[].weight` (per extra endpoint), both int, default 1,
`>= 1` enforced at load (`ConfigError`). Documented in `config.example.yaml`.

**Sandbox config (applied, per operator instruction):** primary vLLM
`weight: 2`, `ak` `weight: 1` — expected split ≈78/22 (vs ≈65/35 at 1:1
auto-matching), so `ak`'s share of batch load drops ~half; `ak:8080` also
carries the interactive `claude-llama-proxy`, which this protects. Takes
effect on the next worker run; the full-library launch config should mirror
the same `answer.weight: 2`.

**Tests (6 new, gate green — 530 passed + ruff + mypy + dist guard):**
least-inflight prefers a free endpoint under concurrency (deterministic
`_GatedModel` threading, no timing dependence); a weight-2 endpoint wins a
loaded tie (1/2 < 1/1); in-flight counters release after failover, after an
all-endpoints-fail call, and after an unexpected non-`AnswerModelError`;
bad weights rejected (length mismatch, `< 1`); config defaults/parse/
validation for both weight fields.

### Addendum — connect-phase timeout + failover/recovery test rounds (2026-09-23)

**Shipped (commit `d8abe4d`, pushed):** `services.connect_timeout_seconds`
(default 2.0, `> 0` enforced) wired as the connect-phase budget of
`httpx.Timeout` on both LLM pools (primary `AnswerModelPool` + resume pool);
read phase keeps `answer`/`resume` `timeout_seconds`. Config validation +
pool connect-timeout tests; gate green (pytest + ruff + mypy + dist guard).
Sandbox embed block also reverted to local `127.0.0.1:8081` after the ak
4-replica embedder test (203 calls / 51-51-51-50 / zero failures; backup at
`config.sandbox.yaml.pre-ak-embed`).

**failover-batch2** (10 resume jobs, tag `failover-batch2`, ak DOWN from start
— TCP-refused at 17:09:02): 10/10 succeeded, all attempts=1, 10 vLLM 200s,
**zero** `ak:8080` lines. ak came up ~17:15 (first successful TCP connect
17:14:59) but the last dispatch decision landed ~17:15:0x — the batch drained
within ~60 s of recovery. **Recovery side NOT captured** (timing).

**failover-batch3** (same 10 pairs, tag `failover-batch3`, ak down from
17:19:56): 9/10 succeeded (vLLM, attempts=1). Operator started ak mid-batch;
the 10th job (rev `01dfe315…`) dispatched to ak and **ak HUNG processing the
call** (operator: "ak is getting stuck while processing a job"). Job reached
attempts=2 (reclaim; a vLLM 200 landed 17:27:39 for the retry) when the
operator ordered a stop; worker SIGTERM was ignored (graceful-shutdown
handler blocked on the hung call) → SIGKILL 17:27:52. Job left in `running`
with an expired lease — the next worker start reclaims and completes it
(idempotent upsert).

**State at save:** no library-rag processes running; **serve 8100 is DOWN**
(stopped before batch2, not yet relaunched); ak llama-server was hung —
operator is restarting it (server-side issue, outside this repo).

**Open findings:** (1) the recovery test (secondary comes back mid-batch) is
STILL not captured — batch2/batch3 both drained before recovery delivered a
successful ak call; (2) new: the 2 s connect budget does NOT protect against
a HUNG read (endpoint accepts TCP, then stalls) — that is the 300 s resume
read timeout's job, and it will eventually reclaim; (3) per-call ~2 s bounce
cost of ak-first dispatches while down is inferred, not measured (failed
connects emit no log line).

**Next unfinished task:** after operator restarts ak — relaunch serve (no
`--allow-version-mismatch`) + smoke; run the recovery round properly:
`failover-batch4`, ak down at start, ~20 jobs (~14 min window), operator
starts ak ~2–3 min in; the worker run reclaims the leftover batch3 job
`01dfe315…` first.

### Addendum — failover-batch4: recovery CAPTURED (2026-09-24 ~01:18–01:31)

**Round ran as planned and drained clean.** Worker started 01:18:47
(pid 52209); ak confirmed down at start (TCP-refused). Enqueued 20 jobs
under tag `failover-batch4` (the 10 batch2/3 pairs + 10 fresh pairs,
chunk-count spread 11–438); the batch3 leftover `01dfe315…` was
reclaimed at worker start (attempts→3) and completed on vLLM — **the
batch3 hang is now resolved**. Operator started ak at 01:22:47 (~4 min
in). The first ak dispatch landed within seconds of the TCP-up; first
ak 200 at 01:23:51 — a **64 s recovery gap** (includes the generation
itself). Drain 01:30:47, ~12 min window: **21/21 succeeded, zero
job-level failures**, total attempts 23 (20×1 + the leftover's 3).
Per endpoint: **4 ak / 17 vLLM**; the post-recovery split is 4/14 ≈ 29%
of traffic to ak — right on the weight 1:2 prediction (~33%). Down
phase: 7 vLLM 200s, all ak dispatches bounced invisibly (refused
connects log nothing) — the ~2 s per-call bounce cost remains inferred,
not measured. Full 21-line httpx timeline (all 200) in
`pilot-sandbox/scratch/worker-failover-batch4.log`.

**Wrap-up hiccup (noted for the runbook):** the first serve relaunch
crashed with a Qdrant lock error — the drained worker was still
running and holding the embedded-Qdrant folder (serve and worker cannot
coexist; single-process local storage). Stopped the worker with its
explicit pids (graceful SIGTERM sufficed; queue was drained), then
serve relaunched cleanly. **State at save: serve is UP on
0.0.0.0:8100** (pid in `scratch/serve.pid`), no worker running, ak up.
Smoke green: `/ready` → qdrant true, browse true; `POST
/resumes/search` 200; `GET /browse/dir` lists the books tree.

**Open findings updated:** (1) CLOSED — the recovery test is now
captured (above); (2) kept — the 2 s connect budget does NOT protect
against a HUNG read (endpoint accepts TCP, then stalls); that is the
300 s resume read timeout's job; (3) kept — per-call ~2 s bounce cost
of ak-first dispatches while down is inferred, not measured (failed
connects emit no log line).

**M9 failover/recovery validation is complete** — no unfinished tasks
on this thread. The full-library launch remains gated on explicit
operator go; its config must mirror the sandbox (ak:8080
extra_endpoint, answer weight 2, max_concurrent_jobs 2,
resume.max_tokens 8192, browse block).

## M10 — Filesystem browse tab (2026-09-24)

**Status: gate passed (567 passed, ruff + mypy clean).** Operator
request: a web-UI tab that navigates the books tree by hand — root set
in configuration (`/mnt/models_sas_ssd/books` in our case), paths shown
relative to the root, drill down to pdf/epub (the file types to show
are also configured: `[.pdf, .epub, .PDF, .EPUB]`), left-click a file
opens it in the reader, right-click opens its summary. The whole
feature is enable/disable-able in configuration (off by default), and a
missing mount must degrade the feature, never the app.

### Design
- Browse is a lens on the existing catalog + filesystem, not a new
  reader. `GET /browse/dir?path=<rel>` lists one level of the
  configured root: directories are always listed (drill-down), files
  are filtered by the configured `file_types` (case-insensitive
  match), and each file row carries the *active* revision's
  `rev_id` + title + `size_bytes` when the path is indexed.
- Left-click file → the existing reader pane (`openReaderFor(rev)`);
  right-click → the existing Resumes pane via `GET /resumes/{rev}`
  (M8's endpoint, whose 404 "no summary stored yet" handling is
  reused). A file not in the index opens with a plain "not in the
  index yet" message — no raw-file endpoint (the reader/archive
  routes already serve bytes with auth). Breadcrumb navigation; the
  client only ever sends relative segments the server gave it.
- Containment: the request path must be relative (null bytes and
  absolute paths rejected up front); the joined path is
  containment-checked against the root's `realpath`, which kills
  `..` escapes and symlinks pointing out of the tree; symlinked
  entries are skipped in listings entirely — the same stance as
  `scan.follow_symlinks: False`. Reuses `_is_within` (commonpath
  based).
- Path → revision lookup: one batched `IN (...)` query per listed
  directory joining `path_aliases` to the doc's *active*
  `source_revisions` row — a stale duplicate alias must still show
  the document's current revision (the same rule `/library` applies).
- Soft availability: `create_app` computes `browse_available =
  enabled and root.is_dir()` (app-creation FS check, like
  `find_web_dist()`); `/ready` reports the flag, the SPA shows the
  Browse nav button only when it is true (the 30 s poll self-heals
  the button when a mount returns), `/browse/*` 404s when
  unavailable, and `_serve` prints a stderr warning when enabled but
  the root is missing.
- Auth: nothing new — the existing bearer middleware covers
  `/browse/*` automatically (it is not in the open/static carve-outs).

### Delivered
- `src/library_rag/browse.py` (NEW) — pure module:
  `resolve_browse_path` (containment), `lookup_active_revisions`
  (batched path→active-revision join), `list_browse_dir` (one-level
  listing: dirs first then files, case-insensitive type filter,
  symlink skip, on-disk size for unindexed files, catalog
  rev_id/title/size for indexed ones).
- `src/library_rag/config.py` — `BrowseSettings` (off by default;
  `enabled` requires `root`, `root` must be absolute, `file_types`
  validated dotted suffixes, normalized lowercase and deduped —
  `[.pdf, .epub, .PDF, .EPUB]` loads as `[".pdf", ".epub"]`) and
  `Config.browse`.
- `src/library_rag/api.py` — `GET /browse/dir` (400 malformed /
  escapes root, 404 not-a-directory or feature unavailable),
  `browse_available` computed at app creation, `browse` flag added
  to `/ready`.
- `src/library_rag/cli.py` — `_serve` stderr warning when browse is
  enabled but the configured root does not exist.
- `web/src/api.ts` / `app.ts` / `styles.css` — Browse view: nav
  button shown only when `ready.browse`, breadcrumb bar, directory
  drill-down, file rows (click → reader, right-click → resume pane,
  unindexed files rendered muted); rebuilt `web/dist` with vite
  (dist guard green).
- `config.example.yaml` — documented `browse:` section (commented,
  off).
- Live operator config: `browse:` enabled with the operator's root and
  the four-case file_types list. Initially appended to the repo-root
  `config.yaml` (gitignored) on the assumption that it was the live
  config; the addendum below corrects that — the live config is
  `pilot-sandbox/scratch/config.sandbox.yaml`, where the block was
  appended afterwards (backup `config.sandbox.yaml.pre-m10`).

### Tests (31 new: 22 in test_browse.py, 9 in test_config.py; gate
green — 567 passed, ruff + mypy clean)
- `tests/test_browse.py` (22): containment — `..` escapes (including
  after a descent), absolute path, symlink pointing out of the root,
  null byte; an in-tree symlink resolves fine. Listings —
  file_types filter case-insensitive, a directory named like a file
  stays a directory, drill-down relative paths, empty dir, symlinks
  skipped, not-a-directory/missing → `BrowseNotFound`. Stale alias
  row: a duplicate path whose own rev went inactive still shows the
  document's ACTIVE revision (rev_id + size + title). Route level —
  listing shape, drill-down, `../x` and `/etc` → 400, file/missing
  → 404, `/ready.browse` true, disabled-by-default → false + 404,
  missing root degrades → false + 404 (app otherwise healthy).
- `tests/test_config.py` (9, browse section): defaults (off, no
  root), enabled-without-root rejected, non-absolute root rejected,
  disabled-with-root fine, empty file_types rejected, non-dotted
  suffixes rejected, whitespace+case normalization
  (`" .PDF "` → `.pdf`), dedupe preserving order, load-from-file
  with the four-case list.
- `tests/test_api.py` — the two exact-shape `/ready` assertions gain
  the `browse` key.

### Open
- Live E2E (serve 8100 currently DOWN): rides on the
  failover-batch4 relaunch, gated on the operator restarting ak.
  Verify `/ready.browse` true, root listing with relative paths,
  drill-down, containment → 400, UI Browse tab visible (and hidden
  with `browse.enabled: false`), PDF click → reader, right-click →
  résumé pane, mount-pull degradation (tab disappears within one 30 s
  poll, `/browse/dir` 404s, app otherwise healthy).
- Default-off reverted-check (no `browse` section at all): app boots
  identically, no nav button, `/browse/*` 404s — covered by the
  disabled-by-default route tests.

### Addendum (2026-09-24 ~01:00) — config mix-up corrected, live E2E done

- **The live operator config is NOT the repo-root `config.yaml`.** It
  is `/mnt/models_sas_ssd/library-rag/pilot-sandbox/scratch/
  config.sandbox.yaml` (all roots under `pilot-sandbox/`, source_roots
  `[/mnt/models_sas_ssd/books]`, app on 0.0.0.0:8100, answer pool
  vLLM:8091 weight 2 + ak:8080 weight 1). The repo-root `config.yaml`'s
  state_root (`/mnt/models_sata_ssd/library-rag/state`) was empty —
  created fresh at 00:47 by a serve I had launched against it, which
  is why the first E2E drill-down showed zero indexed files. The real
  catalog lives in the pilot-sandbox state DB (344 docs / 54270 chunks
  / 335 resumes / one leftover batch3 running job).
- **Re-point steps taken:** appended the `browse:` block to
  `config.sandbox.yaml` (backup `config.sandbox.yaml.pre-m10`, secret
  token line untouched); stopped the wrong serve (pids 42071/42074,
  `--config config.yaml`); verified the two sata dirs I had created
  (`state/`, `qdrant/`) held only the empty-shell DB + qdrant meta and
  deleted them (the pre-existing `scratch/` left alone); relaunched
  `serve --config …/config.sandbox.yaml` from the repo root (log in
  `serve.log`, untracked); updated the stale `scratch/serve.pid` (was
  31136, now 44926 — `ingest_worker.pid` 79597 is also stale, no
  worker running).
- **Live E2E (serve on 0.0.0.0:8100, 344-doc catalog) — all green:**
  - Startup log: only the expected non-loopback token warning; no
    sentinel/mount errors, no browse warning.
  - `/ready` → `browse: true`, `token_required: false`, qdrant /
    embedding / answer all true.
  - Root listing: 26 dirs, relative paths; drill-down through segment
    names with spaces, brackets and commas works (URL-encoded).
  - Indexed row (a deep-nested epub): `rev_id`, title and size
    (359265) match the DB's active revision exactly;
    `GET /resumes/{rev}` returns the stored 928-word résumé — the
    right-click path verified end-to-end.
  - Unindexed dir (`incoming/`): 6 pdf rows, all `rev_id: null` with
    on-disk sizes — the "not in the index yet" display.
  - Containment: `../x`, `/etc`, `bd/../../etc` → 400; a file path and
    a missing path → 404. (A literal NUL byte is not practically
    sendable through an HTTP URL; the guard is unit-tested.)
  - `/library` → 344 books, matching the DB.
- **Operator UI check passed (2026-09-24, "ok for me"):** Browse tab
  visible, PDF click → reader, right-click → résumé pane all work in
  the browser. Remaining from "Open": only the mount-pull degradation
  watch — its API side (`/ready.browse` false, `/browse/*` 404s, app
  otherwise healthy) is the verified missing-root behavior, and the
  30 s poll self-healing is the SPA's existing mechanism.

### Addendum (2026-09-24 ~14:30) — post-M10 navigation & header fixes (operator pass)

Operator-driven UI iteration on the SPA (vanilla TS + Vite; each change
rebuilds `web/dist`, committed; serve reads dist from disk, no restart).

- `f0c6073` — **browser back trap**: a guard history entry is seeded on
  load; every popstate (back / forward / hardware back) re-arms a fresh
  guard, so back can never cross out of the site — only closing the tab
  or typing a URL does. Back = home semantics; forward is absorbed.
- `cf08e65` — **header/nav**: Ingestion tab removed; a "Rag" button sits
  at the far right of the header (utility position). Research /
  Summaries / Browse labels are bold; the Resumes tab is renamed
  **Summaries**. Back inside Browse first moves to the books **root**;
  from the root (or any other non-Research view) it lands on Research;
  on Research it is a no-op.
- `17f0d6a` — **API token widget** (field + Set) moved from the header
  to the bottom of the Rag page, visibility rules unchanged (hidden
  unless the server reports a token may be needed); a 401 now also
  switches to the Rag view so the prompt is visible where the field
  lives (guarded against a 401 → showView → refetch loop).
- **Operator screenshots** `screens/Capture1-5.JPG` (~14:23, at
  `http://tr4:8100`): new nav confirmed (bold Research / Summaries /
  Browse + Rag far right); a live answer "What do you know about
  optics?" with 12 citations and the reader on
  `photo/Digital Optics For Digital Photography.pdf` p.9/10; the Rag
  page with the API-token widget at the bottom (visible in the
  operator's session; my `/ready` smoke at ~13:00 said
  `token_required: false`, so its visibility there came from a 401
  prompt or a config flip on their side — unverified).
- **Live state**: serve pid 7659 (`pilot-sandbox/scratch/serve.pid`
  updated), bundle `index-sx6oNGYF.js` served on 0.0.0.0:8100.
- **Unverified observations** (operator session, worth a look if
  asked): answer history shows a `failed` query "find me some facts
  about photography" (0 citations); Rag jobs succeeded 16589 /
  cancelled 604 / permanent_failed 11 with INGESTION running; the
  status card inside the Rag view still reads "INGESTION" (rename to
  "Rag" offered, not requested).
