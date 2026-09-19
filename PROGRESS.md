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
