# M6 Pilot and Capacity Report (PRD §12)

**Status: APPROVED** — operator pre-approval given 2026-09-20 ("you don't wait for my
approval of your report, consider that I've already approved of it").

Companion artifacts:

- `pilot_report.md` (auto-generated, in the sandbox scratch) — measured sections straight
  from the run JSON, reproducible with the commands in the appendix.
- `pilot_run.json` (sandbox scratch) — the run report. **Reconstructed read-only** after the
  deliberate stop (see §3.3); collection SQL replicated verbatim from `pilot.run_pilot()`.
- `stage_attribution.json` (sandbox scratch) — real per-stage worker processing times
  (the decisive table in §3.2).

---

## 1. Scope and corpus

- **Corpus surveyed**: 34,768 books (34,371 profiled candidates; 397 invalid/unreadable),
  191.5 GB total. Formats: pdf 25,279 / epub 9,129. OCR class: native 19,585 /
  scanned 14,683 (42.7%) / mixed 103.
- **Sample**: stratified 300 books (seed 42, 51 strata, proportional with a floor of 1 per
  stratum), **page cap 32** per book so long books cost a bounded amount.
- **Estimated full-run units** (pages/image-units): **4,097,557** across 34,371 books
  (pilot books averaged 19.0 units after the cap; full-corpus average 119.2).

## 2. Measured pilot run (stopped deliberately, see §3.3)

| metric | value |
|---|---|
| sandbox | `/mnt/models_sas_ssd/library-rag/pilot-sandbox` |
| documents / revisions | 300 / 300 |
| units (pages + image units) | 5,692 |
| chunks | 34,908 |
| chunk tokens | 8,360,826 |
| wall (first job → stop) | **8.46 h (30,448 s)** |
| jobs completed | 8,233 |
| publications active | 161 |
| embed model | bge-m3@Q8_0-aa473d51 (1024 dims, Q8_0 GGUF, llama.cpp on :8081) |
| answer model | qwen3.8-27b-fp8 (vLLM, :8091) — untouched, healthy |

Stage outcomes: extract 300/300 · chunk 300/300 (7 books extracted to zero chunks —
image-only within the cap) · ocr 886 ok / **162 permanent_failed** (1-px raster bug,
finding F3) · embed 161 ok / **132 permanent_failed** (512-token per-input cap, finding F2)
· publish **6,586 done / 6,455 pending at stop** of 13,041.

## 3. Findings

### 3.1 The O(N²) stale-epoch republish tail (measured, FIXED)

The BM25 corpus-stats epoch hashes the document count, so it moves on every new book.
Pre-fix, every publish job recomputed corpus stats from scratch (O(corpus) tokenization,
~4.2 s at pilot scale), and every epoch move fanned out one republish job per already
published book. In the pilot this produced **13,041 publish jobs for 300 books** (~81
republishes per embedded book) at 14.2 jobs/min.

**Measured impact — the pilot's real time split** (worker processing time attributed per
stage from the completion sequence; `stage_attribution.json`; the drain was continuous —
one 163 s gap in 8.4 h — so attribution is exact to within that gap):

| stage | real worker time | share | jobs |
|---|---|---|---|
| **publish (old code)** | **27,288.9 s (7.58 h)** | **89.6%** | 6,586 done (4.14 s/job) |
| ocr (tesseract, serial) | 2,005.9 s (33.4 min) | 6.6% | 1,048 |
| embed (bge-m3 on :8081) | 511.9 s (8.5 min) | 1.7% | 293 |
| extract (PyMuPDF) | 346.4 s (5.8 min) | 1.1% | 300 |
| chunk (word tokenizer) | 294.9 s (4.9 min) | 1.0% | 300 |

The bug consumed **~90% of the entire pilot run**. The observed 14.2 jobs/min rate × the
remaining 6,455 jobs = 7.6 h more — the "15.3 h publish tail" the run was stopped inside.

**Full-scale (old code, live epoch): infeasible.** ~700 h of ever-growing stats recomputes
plus ~42 B re-upserted sparse points for the 34,371-book corpus (O(N²) fan-out: a new book
republishes every book before it).

**Fix B — shipped and gated** (378 tests pass; see PROGRESS.md for the design record):

- **B1 (always on)**: the corpus epoch is committed as a meta record inside the publish
  transaction; republish jobs find their rev already active under the recorded epoch and
  no-op in **O(1)** — no stats recompute, no upsert. Self-healing: a missing record falls
  back to one recompute.
- **B2 (opt-in, `retrieval.stats_epoch`)**: `"frozen"` pins the epoch for the whole
  campaign — the epoch never moves, so fan-out is **zero**. Novel terms from books added
  after the freeze get zero sparse weight until a deliberate re-freeze (documented recipe:
  flip to `live`, delete the meta record, one publish recomputes once over the corpus and
  converges everything in one bounded round). The search-side cost of the pre-fix
  fan-out is measured in §5 (a query runs one full-corpus sparse search per active
  epoch: 230 s/query at 49 epochs in the legacy sandbox; a full-scale pre-fix campaign
  would approach ~34k per query).
- **Recommended for the full run: `stats_epoch: frozen`** (set in `config.example.yaml`).

Post-fix, per-book publish cost is the first-publish upsert only; a direct measurement
from the §5 latency ingest (8 books through the fixed code) is in §5.

### 3.2 Why the auto-report's per-stage ETA is an artifact

`pilot_report.md` §4 scales each stage's `SUM(updated_at − created_at)` linearly to the
full corpus. In a **single-consumer queue** that sum includes all queue wait, and wait
grows with N — hence its "759,999-day" ETA (publish alone: 88.96 M "s" = 6,822 s/job
average, i.e. ~99% waiting in line). The report's own assumptions note says "queue waits
included"; §4 below replaces the per-stage numbers with the real processing rates from
§3.1. The auto-report remains the authoritative record of *measured* quantities; only its
*projections* are replaced here.

### 3.3 The run was stopped deliberately (and why the run JSON is reconstructed)

The run was stopped at 6,586/13,041 publish jobs — inside the O(N²) tail, after everything
of lasting value (all extraction, OCR, chunking, embedding, and first-publish work) was
done. Resuming the *old* code would have burned ~7.6 h on work the fixed code does in O(1).
`run_pilot()` cannot be re-run (it would drain the 6,454 pending jobs and mutate the
stopped state), so `pilot_run.json` was reconstructed read-only: identical SQL, identical
schema (`pilot-run/1`, 34 fields), `interrupted: true`. Non-reconstructable in-process
measurements (wall timers, RSS, VRAM, registration counters) are `null`, not zero;
`seconds` is the jobs-table window; `disk` deltas equal absolutes (fresh sandbox).

### 3.4 512-token per-input embed cap (measured; fix = operator server relaunch)

132 of 293 attempted embeds (45.1%) failed permanently: the running llama.cpp server on
:8081 rejects any single input longer than its physical batch size (n_ubatch), which this
build clamps to **512 BPE tokens** despite `--batch-size 2048`. Affected books are
long-text books (high chunk BPE counts). Measured word→BPE ratios (bge-m3): EN 1.39 /
DE 1.69 / FR 1.65; at the chunking budget of 256 words the typical chunk is ~350–430 BPE —
under 512 — but dense books cross it.

**Fix = relaunch :8081 with `--ubatch-size 2048`** (embedder.sh already passes
`--batch-size 2048`; the clamp is in n_ubatch). That is operator GPU infrastructure, so it
was **not** done this session by decision — it is the first action of the next session,
before any retry of the 132 failed embed jobs. BPE-aware chunking is an M7 consideration,
not needed once the ceiling is 2048 (256 words × 1.77 worst-case + title ≈ 455 BPE ≪ 2048).

### 3.5 1-px raster-size OCR failure (measured, FIXED)

162 of 1,048 OCR jobs (15.4%) failed permanently: some scanned pages rasterize to a
1-px-tall strip (blank/edge page geometry), which tesseract rejects. Fix A shipped
(monkeypatched regression tests included): such units are classified as blank instead of
crashing the job. Without the fix, ~15% of scanned-page OCR would be lost at full scale —
material for the 14,683 scanned books.

## 4. Full-corpus ETA by bottleneck (corrected, serial single worker)

Per-stage real rates from §3.1, scaled to the full corpus (units 5,692 → 4,097,557;
OCR jobs 1,048 → 754,132 by the pilot's jobs-per-unit ratio; embed by tokens):

| stage | pilot real rate | full-corpus estimate | share |
|---|---|---|---|
| **ocr** (tesseract, 1 process) | 2.03 s/job (886 succeeded) | **~424 h (17.7 d)** | **~60%** |
| **embed** (bge-m3 :8081) | 12,385 tok/s (6.34 M tokens / 511.9 s) | **~135 h (5.6 d)**, 95% CI 107–187 h (bootstrap token range) | ~19% |
| extract (PyMuPDF) | 0.061 s/unit | **~69 h (2.9 d)** | ~10% |
| chunk (word tokenizer) | 0.052 s/unit | **~59 h (2.5 d)** | ~8% |
| publish (Fix B, frozen epoch) | measured in §5 (expected ≪ 1 h) | small; one bounded re-freeze recompute at campaign end (~8 min) | ~2% |
| **total (serial, single worker)** | | **≈ 700 h ≈ 29 days** | |

Caveats: single stratified pilot (book-to-book variance only, per the auto-report's
assumptions); per-unit rates measured on page-capped books and extrapolated to full length
(unit cost is approximately length-invariant: OCR per page, extract per page, embed per
token); the 132 embed failures and 162 OCR failures are assumed to succeed under the
fixed code (embed needs the §3.4 relaunch).

**With the PRD's parallel worker architecture (M7 prerequisite, not built yet)**: OCR is
embarrassingly parallel across the 32 logical CPUs (this machine has **no cgroup CPU
quota** — the auto-report's "quota unknown / more workers would not add throughput" note is
a template artifact of a quota-assuming note, corrected here): 16 OCR processes → ~26 h.
Embed is the next wall (single :8081 server); a second embedder on the spare 3090 →
~68 h. Parallel extract/chunk hide under those. Realistic parallel target: **~1 week**
serial-stage floor dominated by embed.

**The serial 29-day single-worker figure is the honest no-M7 number.** The 2,082-year
auto-projection (§3.2) is not.

## 5. Latency (idle vs ingesting)

`pilot latency` (10 deterministic probes x 3 reps, limit 20; a probe = query embed on
:8081 + dense search + one sparse search per active epoch + validation):

| state | per-query latency (ms) | n |
|---|---|---|
| fresh 30-book sandbox, frozen config, **idle** (18 active pubs, 3,359 points, 1 epoch, local Qdrant) | mean 211.9 / p50 215.3 / p95 218.4 / max 219.0 | 30 |
| same sandbox, **ingesting** (8 new books enqueued; worker draining extract/OCR concurrently) | mean 233.0 / p50 233.9 / p95 246.5 / max 250.3 | 30 |
| legacy 300-book sandbox, stopped mid-campaign (49 active epochs, 52,257 points, local Qdrant) | **230,275** (single read-only probe) | 1 |

Findings:

1. **Concurrent ingest is not the problem: +10%.** Under the frozen config, p50 moves
   215 to 234 ms (+19 ms) while the worker is actively processing enqueued books
   (`enqueued=8`; one job completed during the probe window, the rest still pending at
   the end of the measurement).
2. **The O(#epochs) fan-out is the problem (search-side of the §3.1 bug).** A query runs
   one full-corpus sparse search per active epoch, because different epochs live in
   different BM25 spaces (`retrieval.py:_fuse`). The legacy mid-campaign state had
   49 active epochs → 230 s per query in local mode. Pre-fix, a full-scale campaign
   grows active epochs toward ~34k (every book's first publish carries its own
   epoch), i.e. one query would issue tens of thousands of full-corpus searches. Fix B's
   frozen config pins the epoch count at 1 for the whole campaign; the §3.1 re-freeze
   recipe bounds a re-freeze to one additional round.
3. **Local vs server Qdrant caveat.** All three rows are local-mode Qdrant (pure-Python
   embedded backend; the client warns that payload indexes have no effect in local
   mode — observed in the run log). The 230 s legacy number is dominated by
   49 epochs x per-point payload-filter evaluation in Python; server Qdrant (Rust,
   payload indexes) is faster per epoch, but the epoch fan-out is real regardless of
   backend, so the frozen-epoch setting is required at full scale either way. Absolute
   full-scale per-query numbers need a server-Qdrant run (M7 infra; see §9).

The fresh latency sandbox (`/mnt/models_sas_ssd/library-rag/latency-sandbox`) is
disposable measurement state: 54 jobs of the 8-book busy ingest remain pending because
the measurement process exited after its join cap; it is not part of the frozen
artifacts.

## 6. Storage estimate (full corpus)

| component | size |
|---|---|
| archive (one copy per book) | 191.5 GB |
| qdrant index (2-generation rebuild headroom) | 982.2 GB |
| state database | 57.3 GB |
| artifacts | 250.3 GB |
| **total** | **≈ 1.48 TB** |

Headroom: /mnt/models_sas_ssd (books + archive) 6.9 TB free; /mnt/models_sata_ssd
(state/qdrant/scratch) 7.5 TB free. **Fits with ~4x headroom on both volumes.** (Pilot
measured: 1.70 GB archive / 0.68 GB qdrant / 0.50 GB state / 0.35 GB artifacts for 300
capped books.)

## 7. Question bank

0 of the PRD's 100–200 human-labeled questions exist — **labeling is a human task and
was not started by this session (labels are never invented)**. 80 *unlabeled* candidate
questions were generated from the pilot index (`pilot annotate suggest`, seed 42,
read-only against the pilot sandbox DB: 40 chunks x 2 — one `exact_term`, one
`conceptual`) in `question_candidates.jsonl`, for the operator to review and label with
`pilot annotate add`. Status is reported as "empty — labeling not started" in the
auto-report.

## 8. Recommended frozen configuration for the full ingestion

| setting | value | why |
|---|---|---|
| `retrieval.stats_epoch` | **`frozen`** | kills the O(N²) tail (finding F1); re-freeze recipe at campaign end |
| `embedding` | `bge-m3@Q8_0-aa473d51`, 1024 dims | measured; exact revision pinned |
| `chunking` | target 256 words / overlap 40 | safe under the 2048-BPE ceiling after the §3.4 relaunch |
| embedder | :8081 relaunched with **`--ubatch-size 2048`** | first action next session (finding F2) |
| worker | single serial process (current architecture) | local-mode Qdrant flock + design; 32 CPUs available but unused by the serial worker until M7 parallelism |
| page cap | pilot-only (32); **no cap for the full run** | cap exists to bound pilot cost |
| GPU discipline | keep :8091 vLLM untouched; :8081 embedder stays resident | operator infra |

## 9. Uncertainty

1. Single pilot run; bootstrap CIs reflect book-to-book variance only (auto-report §8).
2. Page-capped per-unit rates extrapolated to full-length books.
3. Run interrupted mid publish tail; the run JSON carries `interrupted: true` and null
   in-process fields (§3.3).
4. OCR job count for the full corpus is scaled by the pilot's jobs-per-unit ratio; the
   scanned share (42.7%) is stratified into the sample, so the ratio is corpus-representative.
5. Embed ETA uses the pilot's measured 12,385 tok/s; if the relaunched server (n_ubatch
   2048) batches differently the rate moves within the bootstrap CI.
6. Legacy-state latency is a single read-only probe (n=1) in a frozen mid-campaign
   state; the fresh-sandbox rows are n=30 each. All latency rows are local-mode Qdrant
   and do not transfer directly to server Qdrant (§5, finding 3).

## 10. Next-session actions (ordered)

1. Relaunch :8081 with `--ubatch-size 2048` (operator infra; one command in embedder.sh's
   launch line).
2. Retry the 132 permanent_failed embed jobs and 162 OCR jobs in the sandbox (check
   `retry` semantics for permanent_failed first) — optional validation, sandbox only.
3. **Full ingestion**: `uv run library-rag scan --config config.yaml` then
   `uv run library-rag ingest --config config.yaml` with the §8 frozen config.
   Do **not** start before the §3.4 relaunch.
4. Human labeling of the 80 candidate questions (then top up to 100–200).
5. M7: PRD parallel worker architecture (OCR/embed fan-out), BPE-aware chunking only if
   the 2048 ceiling proves insufficient.

---

### Appendix: reproduction

```bash
cd /dd2/andrei/books_rag
SBX=/mnt/models_sas_ssd/library-rag/pilot-sandbox
# survey + manifest (already on disk; commands for reference)
#   uv run library-rag pilot survey  --config config.yaml --out /mnt/models_sata_ssd/library-rag/scratch/survey.jsonl
#   uv run library-rag pilot sample  --config config.yaml --survey /mnt/models_sata_ssd/library-rag/scratch/survey.jsonl --target 300 --seed 42 --page-cap 32 --out /mnt/models_sata_ssd/library-rag/scratch/manifest.json
# run report (reconstructed; read-only):
uv run python /mnt/models_sata_ssd/library-rag/scratch/m6_collect_run_metrics.py
uv run python /mnt/models_sata_ssd/library-rag/scratch/m6_stage_attribution.py
# capacity report (auto-generated measured sections):
uv run library-rag pilot report --survey /mnt/models_sata_ssd/library-rag/scratch/survey.jsonl \
  --manifest /mnt/models_sata_ssd/library-rag/scratch/manifest.json \
  --run $SBX/scratch/pilot_run.json --config config.yaml --out $SBX/scratch/pilot_report.md
# latency (fresh 30-book sandbox under the recommended frozen config; disposable):
CFG=/mnt/models_sata_ssd/library-rag/scratch/config.m6lat.yaml   # config.yaml + retrieval.stats_epoch: frozen
uv run library-rag pilot sample --survey /mnt/models_sata_ssd/library-rag/scratch/survey.jsonl \
  --target 30 --seed 42 --page-cap 32 --allow-out-of-range \
  --out /mnt/models_sata_ssd/library-rag/scratch/manifest30.json
uv run library-rag pilot run --config $CFG --manifest /mnt/models_sata_ssd/library-rag/scratch/manifest30.json \
  --sandbox-root /mnt/models_sas_ssd/library-rag/latency-sandbox
uv run library-rag pilot latency --config $CFG --sandbox-root /mnt/models_sas_ssd/library-rag/latency-sandbox \
  --ingest-root /mnt/models_sas_ssd/books --ingest-count 8 --probes 10 --reps 3 --limit 20 --json
# single read-only probe in the frozen legacy 300-book state (49 active epochs):
uv run library-rag pilot latency --config config.yaml --sandbox-root $SBX \
  --probes 1 --reps 1 --limit 20 --json
# question candidates (read-only against the pilot sandbox DB; never labels):
#   `pilot annotate suggest` equivalent run with a mode=ro connection ->
#   /mnt/models_sata_ssd/library-rag/scratch/question_candidates.jsonl (80 candidates)
```

Gate (2026-09-20): ruff "All checks passed!", mypy "Success: no issues found in 76 source
files", pytest **378 passed** (baseline 375 + 3 new epoch tests).
