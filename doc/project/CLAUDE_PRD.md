# Personal Library RAG — Claude Code Implementation PRD

## 1. Mission and working agreement
Build a local-first research application for approximately 300 GB of PDF and EPUB books. Answers must cite evidence and open the correct source page or EPUB paragraph. Initial ingestion may take weeks and must survive restarts. New books must be added incrementally.

Target: Linux, 64 GB host RAM, two RTX 3090 GPUs (24 GB each), one RTX 3080 Ti (12 GB), 10 TB SAS HDD, 1 TB SAS SSD. CPU count, Linux distribution, GPU driver, and mount paths must be detected, not assumed.

Claude Code: implement the milestones below in order. Deliver running code and tests, not only a design. Complete each milestone's gate before proceeding. Record progress and commands in PROGRESS.md. If GPU hardware or models are unavailable in the development environment, use explicit test doubles for tests, mark GPU integration unverified, and leave executable validation commands. Never claim a test passed without running it.

Do not delete or modify source books. Do not run full-library ingestion before the pilot configuration is approved. Do not download large models, install system packages, alter drivers, or format/mount disks without operator approval. Implement ordinary code and local tests without repeatedly requesting permission.

Keep the implementation simple: no agents, knowledge graph, Kubernetes, Redis, Celery, model fine-tuning, or generated summaries for every chunk. Prefer straightforward Python functions and typed data records over framework-heavy abstractions.

## 2. Product requirements
- Recursively discover PDFs and EPUBs; preserve original bytes and edition identity.
- Support native PDF text, existing OCR, mixed scanned/text documents, and EPUB chapters.
- Search with both lexical and semantic retrieval; rerank and optionally generate cited answers.
- Provide search-only mode independent of answer-model availability.
- Click a citation to open the exact physical PDF page or stable EPUB paragraph; highlight when mapping permits.
- Pause, resume, retry, inspect failures, add books, and replace revisions without whole-library reprocessing.
- Run locally without external inference, telemetry, or document uploads by default.
- Report partial coverage and processing failures honestly.

Non-goals for v1: public hosting, multi-user authorization, image understanding, guaranteed mathematical/table extraction, automatic bibliographic deduplication across editions, and universal EPUB page numbers.

## 3. Technology decisions
Use Python 3.11+, FastAPI, Pydantic, sqlite3, Qdrant server, and modular LlamaIndex packages. Use uv with a lockfile. Select mutually compatible stable releases at implementation time and pin them. Never copy old framework API examples without checking the installed API.

Use PyMuPDF for native PDF extraction, Tesseract CLI for page-level OCR, EbookLib plus an HTML parser for EPUB extraction, and PDF.js for reading. Use plain TypeScript and Vite for the frontend; a frontend framework is unnecessary. Bundle PDF.js locally with pinned npm dependencies; no CDN assets at runtime. Use llama.cpp server for answer generation through its local compatible API.

LlamaIndex scope: local model adapters and Qdrant retrieval integration where useful. Application-owned source records, jobs, chunk IDs, and publication state remain authoritative. Do not use an in-memory LlamaIndex docstore as the corpus database. Direct Qdrant client calls are permitted when they make publication/filtering safer or simpler.

Pilot defaults: BAAI/bge-m3 dense embeddings (1024 dimensions), Qdrant/bm25 sparse retrieval, and BAAI/bge-reranker-v2-m3 reranking. Verify pinned model compatibility, licensing, query/document encoding, BM25 IDF configuration, and offline operation. Do not accidentally enable BGE multi-vector indexing. Answer-model GGUF path is operator-configured; use an 8B–14B quantized instruction model as the initial size class, not an unverified automatic download.

Deployment: application container with coordinator and bounded worker subprocesses; Qdrant container; optional llama.cpp service. Use Docker Compose and explicit NVIDIA device mapping. Unit tests must run without Docker/GPU. Provide a CPU-only small-corpus integration profile.

## 4. Repository deliverables
Create a maintainable repository with pyproject.toml, uv.lock, package src/library_rag, tests, frontend, compose.yaml, .env.example, config.example.yaml, README.md, PROGRESS.md, and numbered SQLite migrations.

Suggested Python modules: config, db, catalog, jobs, artifacts, extraction/pdf, extraction/ocr, extraction/epub, normalization, chunking, embeddings, indexing, retrieval, citations, api, and cli. Merge tiny modules if clearer; do not build unnecessary plugin systems.

Commands must include: doctor, init, scan, ingest, pause, resume, status, retry, search, reconcile, evaluate, backup, restore, and verify. All destructive commands require explicit confirmation or a clearly named noninteractive flag. CLI help must document each command.

## 5. Storage, budgets, and configuration
Configurable roots: source_roots (read-only), archive_root and artifact_root on HDD; state_root and qdrant_root on SSD; model_root configurable; scratch_root on SSD. Refuse overlapping unsafe paths. Paths shown in example configuration are placeholders to be supplied by the operator.

Keep authoritative originals in a content-addressed archive using streamed copy, checksum verification, fsync, and atomic rename. If the operator explicitly chooses external-source mode to avoid a second copy, warn that retained citations cannot be guaranteed after external deletion. Never hardlink originals into an allegedly immutable archive if external modification is possible.

Persist page/section artifacts as compressed JSON with schema version and checksums; persist dense batches in a non-pickle numeric format plus a chunk manifest. Use bounded shards, never millions of arbitrary tiny embedding files. Delete raster scratch files after committed OCR output. Keep reconstructible caches separate from required citation artifacts.

Default host reserve: at least 12 GB RAM for Linux/cache/headroom. Initial application worker memory budget: 28 GB; Qdrant container memory limit: 20 GB; remaining headroom is unallocated. These are starting ceilings, not guaranteed RAM consumption. Pause admissions on measured memory pressure; do not repeatedly OOM and retry indefinitely. Configure vector storage/HNSW on disk where supported and measure latency.

Start native extraction at min(4, available CPU cores); start OCR at 2 workers with OMP_THREAD_LIMIT=1. Start embedding batches at 8 chunks and tune upward only after profiling. Bound every queue. SSD admission stops below max(20% capacity, 100 GB) free, and HDD admission stops below a configurable reserve. Scratch hard cap: 50 GB. Expose all budgets in configuration.

Use filesystem UUID mounts operationally, but never alter mount configuration automatically. On startup verify expected mount sentinel files; a missing mount must not cause scans or writes to an empty fallback directory.

## 6. Database and identity contract
Use SQLite WAL, foreign_keys=ON, busy_timeout, short transactions, and synchronous=FULL for durable state. Keep the database on local SSD, not network storage. Workers report through coordinator-controlled writes or use carefully bounded short transactions. Include migration tests.

Tables must cover: documents, source_revisions, path_aliases, extraction_runs, source_units, chunks, embedding_batches, index_generations, publications, jobs, and schema_migrations. Persist answer evidence snapshots or equivalent citation manifests when retaining chat history.

Document UUID identifies a catalog entry. Source revision records SHA-256, size, format, archive location, and original path. Exact duplicates become aliases. A rename does not re-embed. A content change at a known path creates a new revision. Do not automatically collapse distinct editions based on similar titles.

Extraction key = hash(source bytes hash, parser version, OCR settings, normalization settings). Chunk key = hash(extraction key, chunker configuration, ordered source span IDs, text hash). Embedding key = hash(chunk key, model revision, encoding configuration). Use deterministic UUID-format Qdrant point IDs derived from the indexing identity; store the full content hashes separately for audit.

Source unit stores physical PDF page index or EPUB spine href, text, geometry/anchors, page rotation, dimensions, quality flags, and artifact reference. Chunk source spans store unit ID, source offsets, and bounding boxes where known. Never discard source mappings after normalization.

Every stage configuration has a canonical serialized hash. Changing answer-model settings does not invalidate embeddings. Changing embeddings does not invalidate extraction. Changing OCR invalidates only affected extraction outputs and downstream consumers.

## 7. Durable execution contract
Jobs use pending, running, succeeded, retryable_failed, permanent_failed, or cancelled. Store unique task key, stage, input/version, range, attempts, lease owner, lease expiry, heartbeat, error category, and output manifest.

Claim atomically in a short SQLite transaction. A lease token fences stale workers: only the current token can commit. Running workers heartbeat; restarted coordinators reclaim expired leases. Default maximum 3 attempts for transient failures, backoff with jitter. Corrupt/encrypted unsupported files fail permanently until explicit retry.

Use at-least-once execution with idempotent effects. Never promise exactly-once transactions across SQLite/files/Qdrant. Write temporary artifacts on the target filesystem, flush and fsync, rename atomically, and fsync the containing directory before marking success. Reconcile durable outputs left behind by crashes.

Checkpoints: hash/archive per file; extraction/OCR per page or small bounded range; EPUB per spine item; chunking per bounded section; embedding and index writes per bounded batch; publication per document revision. Expensive OCR must resume without repeating completed pages. A killed hash operation may restart that file.

Pause persists in SQLite, prevents new claims, and allows active bounded jobs to finish. SIGTERM requests graceful stop; SIGKILL recovery must work. Limit shutdown wait and document remaining replay scope. Index upserts use deterministic IDs and acknowledgement before job success.

## 8. Processing pipeline tasks
### A. Inventory and archive
- Implement streaming discovery without loading the tree into memory; ignore configured temporary/system paths and constrain symlink traversal.
- Register size/mtime fast checks, wait for file stability, hash content, and detect files changing during hashing/copying.
- Validate PDF/EPUB format rather than trusting extension. Record page counts, readability, and metadata when available.
- Report missing files but never automatically delete indexed content; detect an unavailable source mount separately.

### B. PDF extraction and quality
- Extract page-by-page blocks/spans and page labels; internal page indices are zero-based, display/API reader page numbers are one-based.
- Preserve original extracted text and coordinates; assess density, malformed characters, repeated overlays, and obvious reading-order issues with language-sensitive heuristics.
- Distinguish legitimately sparse illustration/cover pages from pages needing OCR. Existing text layers are reused unless quality fails.
- Save quality decisions and configurable thresholds. Flag uncertain multi-column/table extraction. Provide a future fallback interface, but do not make Docling mandatory in v1.

### C. OCR
- Rasterize only selected pages, initially 300 DPI with bounded dimensions. Invoke Tesseract with timeout and explicit installed language packs; capture text and TSV/hOCR word boxes.
- Map raster boxes back to original page coordinates using recorded transform, crop, and rotation. Unit-test rotated and cropped pages.
- Avoid geometry-changing deskew in v1 unless inverse transforms are implemented. If geometry cannot be trusted, use page-only citation or an explicitly labeled derivative; never draw misleading highlights.
- Commit per-page artifacts and remove scratch. Keep OCRmyPDF whole-book derivatives optional; they must not be the checkpoint mechanism.

### D. EPUB extraction
- Respect spine order, preserve chapter titles, sanitize HTML, remove scripts and external resource loads, and create deterministic paragraph anchors.
- Bound ZIP uncompressed size, entry count, and compression ratio; reject path traversal and malformed archives. Store original EPUB and versioned rendered sections.
- Return chapter/paragraph citations, never fabricated page numbers.

### E. Normalization and chunking
- Keep source text separate from searchable text. Conservatively normalize whitespace/dehyphenation and remove repeated headers only when supported across pages.
- Maintain normalized-to-source span mappings; test removed headers and joined words. Quotations come from source text, not reconstructed cleaned strings.
- Initial target 600 model tokens, overlap 100; prefer paragraphs/sections and permit cross-page spans. Store previous/next chunk relationships.
- Count tokens using the pinned embedding tokenizer, including title/heading prefixes. Split before the model limit; forbid silent truncation. Disable implicit LlamaIndex rechunking that would lose IDs or provenance.

### F. Embedding and publication
- Stream chunks in bounded batches; explicit query/document encoding; persist model revision, dtype, dimensions, and normalization settings.
- On GPU OOM halve the batch with a bounded retry policy; if one chunk cannot fit, record a failure and stop that worker rather than loop forever.
- Use independent per-GPU embedding workers in bulk mode. Checkpoint vectors before Qdrant writes so index rebuilds do not repeat embeddings.
- Create versioned dense+sparse Qdrant collections with suitable metadata indexes. Test BM25 new-document behavior and required IDF handling against pinned versions.
- Use payload publication flag: stage new points inactive, verify expected IDs/count, then activate replacement points before deactivating the old revision. Application validation always checks SQLite's active publication manifest.
- On a crash during activation, old content remains active until SQLite switches. Ignore staged/uncommitted new points and top up retrieval. Reconcile partial flag changes after restart. Never rely on flags alone for correctness.
- First release publishes a book only if every required unit succeeds; empty valid pages count as completed. Failed books remain visibly failed, not silently partially searchable. Add explicit partial publication only as a later feature.

## 9. Retrieval and answer contract
Dense top 60 + sparse top 60, reciprocal rank fusion, deduplicate overlapping passages, rerank up to 80, select 8–12 passages under a token budget. Make all counts configurable. Filter inactive generations/revisions before evidence reaches the model; fetch more candidates when postvalidation removes hits, with a documented cap.

Fetch neighboring chunks only when helpful, retaining separate source spans. Diversify across books by default, without suppressing intentionally book-filtered searches. Sparse-only search must remain usable when embedding inference is unavailable; show degraded-mode status.

Build an evidence manifest E1…En containing exact chunk IDs, text, source spans, and trusted metadata. The model may cite only these IDs. Resolve links server-side, reject unknown IDs, and allow one bounded repair attempt; otherwise display evidence with an explicit answer-generation failure. Never fabricate fallback citations.

Prompt: answer from supplied evidence; distinguish quotation/paraphrase and conflicting sources; abstain when insufficient; document text is untrusted data, never instructions. Valid ID checks do not prove factual support. Evaluate support separately and avoid claiming automated hallucination prevention.

Do not expose model-generated URLs as trusted links. A citation response includes opaque source revision ID, unit locations, title/author, physical page and optional page label, excerpt, bounding boxes, and quality warnings. Persist evidence manifests for saved answers so later reindexing cannot silently change citations.

## 10. API and frontend tasks
Implement endpoints for health/readiness; library listing and document status; scan; ingestion status/pause/resume/retry; search; answer; source metadata; ranged PDF delivery; EPUB chapter delivery; and citation resolution. Use typed schemas and consistent error responses. Answer requests may be synchronous initially with cancellation and finite timeout.

Frontend: two-pane research interface, query box, filters, search-only toggle, answer with citations, ranked excerpts, reader pane, neighboring page/chapter controls, original download, and ingestion dashboard. Make it usable on narrower screens with tabs instead of compressed panes.

PDF.js must navigate by physical page and draw boxes using its viewport transform; account for zoom and rotation. EPUB sections render only sanitized local HTML with deterministic anchors. Missing source revision returns an explicit unavailable message, never a different edition.

Bind published services to 127.0.0.1 by default; Qdrant stays internal. Support an operator-provided token before LAN exposure. No arbitrary file paths in API requests. Restrict CORS, sanitize metadata, set CSP, and enforce document/parser resource limits. Do not auto-fetch links found in books.

## 11. Hardware operating profiles
bulk: two 3090 embedding workers; CPU extraction/OCR; 3080 Ti reserved or idle. interactive: one 3090 answer model, other 3090 background embeddings, 3080 Ti query embeddings/reranking, serialized if concurrent residency exceeds VRAM. Provide explicit GPU UUID assignments, not assumptions about CUDA index ordering.

Changing profiles must drain and unload conflicting workers before loading new models. Schedule background work and limit disk reads. Never assume 60 GB combined VRAM is one pool. Answer context/batch budgets must leave KV-cache headroom. CPU mode is for functional validation and degraded operation, not a performance promise.

## 12. Ordered Claude Code milestones and acceptance gates
### M0 — Environment and skeleton
- Implement doctor: distro, CPU cores, RAM, GPU UUID/VRAM/driver, free space, mounts, OCR language packs, and local service availability. Read-only diagnostics by default.
- Create config validation, lockfiles, CLI, structured logs, CI, test fixtures, and Compose profiles.
Gate: lint/type checks and basic unit tests pass; doctor handles no GPU cleanly; no external inference configured implicitly.

### M1 — Catalog and recovery foundation
- Implement migrations, identities, source archive, job leases/fencing, artifact commit protocol, pause/resume, and reconciliation primitives.
- Add crash injection hooks disabled outside tests: before/after artifact rename and before/after database commit.
Gate: duplicate/rename/change tests pass; killed worker replay produces one logical output; stale worker cannot commit; missing mount cannot delete content.

### M2 — Extraction and source reader
- Implement PDF/EPUB extraction, source mappings, ranged delivery, PDF.js page opening, and EPUB anchors before embedding work.
- Add native, rotated, cropped, mixed, bad-OCR, encrypted, corrupt, and EPUB fixtures generated for tests or permissively licensed.
Gate: citations open correct physical pages/sections on fixtures; no model required; path traversal/XSS/ZIP-bomb limits tested.

### M3 — Selective OCR and chunking
- Implement page quality routing, durable OCR, cleanup, normalization mapping, token-aware chunking, and cross-page spans.
Gate: kill after page N resumes from unfinished units; rotated OCR highlight mapping tested; no tokenizer truncation; originals unchanged.

### M4 — Index and search
- Implement checkpointed embeddings, sparse indexing, generation management, publication/reconciliation, filters, fusion, and reranking.
- Provide deterministic fake embeddings for tests, clearly labeled and forbidden in production config by default.
Gate: crash at each publication boundary never returns uncommitted or obsolete evidence; re-upsert is idempotent; search works after restart; sparse degraded mode works.

### M5 — Cited answering and UI
- Implement evidence manifests, local LLM adapter, bounded timeouts, citation validation, answer persistence, and research UI.
Gate: unknown evidence IDs rejected; saved citations survive reindexing; unsupported/no-evidence cases abstain; search continues if model server stops.

### M6 — Pilot and capacity report
- Provide reproducible stratified sample manifest for 200–500 books with configurable page cap. Include all identified format/OCR/language groups.
- Measure extracted tokens/pages, chunk count, extraction/OCR/embedding throughput, SSD bytes per chunk, peak RAM/VRAM, and p50/p95 latency with ingestion on/off.
- Create 100–200 manually labeled questions, including exact terms, conceptual queries, OCR, EPUB, conflicts, and unanswerable questions. Supply annotation/evaluation tools; do not invent human labels.
Gate: produce pilot report with measured results, uncertainty, full-run ETA by bottleneck, storage estimate including rebuild headroom, and recommended frozen config. Operator approval is required before full ingestion.

### M7 — Full-run operations and maintenance
- Implement scheduled discovery, safe revision replacement, explicit removal, generation migration, garbage collection with reference checks, backup/restore, and coverage reports.
- Keep pinned saved-citation artifacts during garbage collection or require explicit acknowledgement that those citations will become unavailable.
Gate: restore into an isolated directory and verify source links/search; add a book without recomputing unchanged sources; full-library launch command and stop/resume runbook documented.

## 13. Evaluation targets
Correct source navigation: 100% on 200 sampled citations. Native-PDF highlight correctness: at least 95%; page navigation always available. Retrieval recall@20: at least 90% on labeled answerable pilot questions. Supported cited claims: at least 95% by human review. Unanswerable abstention: at least 90% on curated negatives.

Latency targets, not guarantees: search plus rerank p95 under 5 seconds; typical answer completion under 30 seconds. Record corpus scale, model/context, concurrency, warm/cold state, and ingestion profile for each result. Recheck at full scale.

Capacity formula: raw dense bytes = chunks × 1024 × 4 for float32 BGE-M3 output. Add measured sparse/text/HNSW/database overhead and peak compaction/rebuild space. Source GB alone cannot predict token count or duration. A migration that cannot fit two generations must require a documented maintenance window.

## 14. Test matrix and operational finish criteria
Required unit tests: deterministic IDs/config hashes, normalization spans, page conventions, chunk boundaries, aliases/revisions, retry classes, lease fencing, evidence validation, and path sanitization. Required integration tests: real Qdrant publication/recovery, SQLite restart, local reader links, PDF range requests, OCR execution, and offline model adapters where hardware exists.

Inject failure during OCR, embedding checkpoint, Qdrant upsert, activation, and SQLite publication. Inject disk-full, absent mount, changed input, corrupt artifact, unavailable Qdrant, unavailable model, and GPU OOM. Verify bounded retry, explicit status, and safe resume. Never simulate these destructively against the operator's actual library.

Backups: use SQLite's consistent backup API; snapshot Qdrant while publication is paused; copy required manifests/configs/source artifacts; record checksums. Internal HDD backups help recover accidental loss but are not independent disaster backups. Document optional external backup targets without pretending one exists.

Definition of done: code and lockfiles committed; tests actually run and results recorded; source navigation demonstrated; restore verified; operator runbook complete; no silent external calls; known extraction/model limitations documented; incomplete GPU or scale validation explicitly listed. Full ingestion progress must distinguish discovered, archived, extracted, OCR-complete, indexed, published, and failed counts.

## 15. Prompt to start Claude Code
Read this PRD completely. Inspect the existing repository before modifying it. Implement M0, then continue sequentially through the milestones whose gates can be satisfied locally. Maintain PROGRESS.md with completed tasks, test commands/results, and remaining blockers. Prefer minimal working vertical slices. Do not start full-library ingestion or download large models without approval. At the end of each work session report the code delivered, tests actually run, and the next unfinished task. Ask only for decisions that cannot safely be inferred from this PRD.
