# Operator Runbook — full-library ingestion

M7 (PRD line 183 gate: "full-library launch command and stop/resume
runbook documented"). Everything below is verified against the code, not
aspirational: signal handling (`cli.py::_ingest`, `worker.py::run_worker`),
job leases/fencing/pause (`jobs.py`), mount sentinels (`scan.py`), backup
refusal semantics (`backup.py`).

## 1. Layout

All paths from the live `config.yaml` (gitignored):

| Purpose          | Path                                        |
|------------------|---------------------------------------------|
| Source books (read-only) | `/mnt/models_sas_ssd/books`         |
| Source mount sentinel    | `/mnt/models_sas_ssd/books/.library-rag-sentinel` |
| Source archive (SHA-256 content store) | `/mnt/models_sas_ssd/library-rag/archive` |
| Artifacts (vectors, OCR images, logs)  | `/mnt/models_sas_ssd/library-rag/artifacts` |
| State (SQLite `library.db`, job logs)  | `/mnt/models_sata_ssd/library-rag/state` |
| Qdrant (local embedded)  | `/mnt/models_sata_ssd/library-rag/qdrant` |
| Models                   | `/mnt/models_sas_ssd/library-rag/models`    |
| Scratch (hard-capped 50 GB) | `/mnt/models_sata_ssd/library-rag/scratch` |
| Backups (M7)             | `/mnt/models_sas_ssd/library-rag/backups`   |

Services: embedding fleet = 8 llama-server instances (bge-m3 Q8_0,
`--batch-size 8192 --ubatch-size 8192`) on 127.0.0.1 ports 8081–8088
(round-robin, failover only on transient errors); answer model = vLLM
qwen3.8-27b FP8 on 127.0.0.1:8091 (separate work — never reconfigure).

**One worker at a time.** Qdrant runs in local (file-based) mode, and a
local Qdrant client holds an exclusive lock on the storage folder for its
lifetime. The ingestion worker therefore cannot run in parallel with
another worker, and read-only tools that open Qdrant (coverage point
counts, gc, backup, verify) degrade or refuse while the worker runs.
That is expected behavior, not a fault.

## 2. Launch (approved sequence)

Pre-flight, every time:

```bash
cd /dd2/andrei/books_rag
uv run library-rag doctor --config config.yaml          # mounts, sentinel, services
# embedder fleet + answer server health (expect 200 on all nine):
for p in 8081 8082 8083 8084 8085 8086 8087 8088 8091; do
  printf "%s %s\n" "$p" \
    "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$p/health)"
done
df -h /mnt/models_sas_ssd /mnt/models_sata_ssd          # capacity
```

Step 1 — discover, register, enqueue (fast check makes a re-scan of
unchanged books nearly free: size+mtime, no re-hash):

```bash
uv run library-rag scan --config config.yaml
```

Step 2 — start the worker detached:

```bash
mkdir -p /mnt/models_sata_ssd/library-rag/scratch
nohup uv run library-rag ingest --config config.yaml \
  >> /mnt/models_sata_ssd/library-rag/scratch/ingest_run.log 2>&1 &
echo $! > /mnt/models_sata_ssd/library-rag/scratch/ingest_worker.pid
```

Step 3 — verify it took:

```bash
uv run library-rag status --config config.yaml          # jobs: pending > 0, then running appears
tail -20 /mnt/models_sata_ssd/library-rag/scratch/ingest_run.log
```

The worker runs an automatic reconcile pass on start (chunks, index,
best-effort publication repair) before claiming, so a fresh worker is
safe after any prior interruption.

**Expected scale** (measured, 2026-09-21): 34768 candidate files under
the source root (360 invalid magic bytes; ~34.4k registrable). Serial
worker at the pilot/batch-2 rate of ~11–13 jobs/min gives the M6 estimate
of ~41–45 days for the full library; re-measure the rate during the first
hour and update the ETA. Parallel workers are not possible in local
Qdrant mode (flock); scaling requires a Qdrant server (container), which
needs operator approval to install.

## 3. Monitoring

```bash
uv run library-rag status --config config.yaml [--json]
```

→ schema version, `paused` flag, jobs by state, documents/revisions,
OCR units by state, chunk count.

```bash
uv run library-rag coverage --config config.yaml [--json]
```

→ the full funnel (discovered → archived → extracted → OCR → chunked →
embedded → indexed → published), per-root coverage (unindexed /
orphaned / stale / invalid), stalled documents, and failures by
state/stage. Read-only. While the worker holds the local Qdrant lock,
`points_active`/`points_total` come back `null` — everything else is
still reported; point counts are live after the worker stops.

Logs: the short per-job line goes to `ingest_run.log`. In addition, while
a job runs, a verbose log is captured to
`/mnt/models_sata_ssd/library-rag/state/job_logs/<job_id>.attempt<N>.log`
and **kept only if the job failed** (flushed on success, deferral, or a
lost lease) — so a failure always has its debug trail on disk.

A periodic `coverage` (hourly) plus `status` on change is sufficient;
the operator's own monitors poll the job table directly.

## 4. Stopping

Three levels, pick the gentlest that fits:

1. **Pause (keep the process running).**
   `uv run library-rag pause --config config.yaml --reason "..."`
   Persists in the DB (`meta.paused`). The worker finishes its current
   job, then idles without claiming. Survives restarts: a (re)started
   worker also refuses to claim while paused. Resume with
   `uv run library-rag resume --config config.yaml`.
2. **Graceful stop (SIGTERM/SIGINT — the normal stop).**
   `kill $(cat /mnt/models_sata_ssd/library-rag/scratch/ingest_worker.pid)`
   The worker finishes the job in flight (its units commit individually
   and durably as they go), then stops claiming and exits 0 with
   `ingest: N job(s) completed`. Nothing is lost or half-committed.
3. **Hard stop (SIGKILL / power loss).** The in-flight job's lease
   expires (TTL 300 s, `--lease-ttl` to change) and is reclaimed; see
   §5. WAL + fenced commits mean a hard stop cannot corrupt the state DB
   or leave a torn publication.

`gc`, `remove`, and `backup` all require the worker to be stopped first:
`gc`/`remove` refuse while any job is running, and `backup` refuses while
the Qdrant local lock is held.

## 5. Resuming and crash recovery

- **After SIGTERM:** just re-run the step-2 nohup command. The startup
  reconcile pass runs automatically; the worker claims from where the
  queue left off. Committed units are never reprocessed.
- **After a pause:** `uv run library-rag resume --config config.yaml`;
  the running (or restarted) worker claims again.
- **After a crash / power loss:** no manual repair is normally needed.
  Jobs left `running` with an expired lease are flipped back to `pending`
  by the next claim, and the stage handlers do not redo committed work:
  OCR is one job per page (at most the in-flight page is re-OCR'd),
  chunking is idempotent via the run's chunk fingerprint (an already
  chunked run is a no-op), and embedding commits per bounded batch
  (vectors are written to the artifact before the batch row), so only
  uncommitted work is repeated. Note that each re-claim counts as an
  attempt (default max 3): a job that crash-loops three times lands in
  `retryable_failed` — fix the cause, then
  `uv run library-rag retry --config config.yaml`.
- **After any recovery:** confirm with `status` and `coverage` — expect
  `paused=false`, no `permanent_failed`, and a consistent funnel.

## 6. Failure triage

| Signal | Meaning | Action |
|---|---|---|
| `retryable_failed` jobs in `status` | transient error, attempts exhausted | `uv run library-rag retry` (requeues) |
| `permanent_failed` job | classified non-transient error | read the kept verbose log in `state/job_logs/`, fix the root cause, `retry --include-permanent` |
| OCR units `failed` (in `coverage`) | tesseract failed on those units | inspect the kept log; usually per-file (corrupt scan) — file stays excluded, rest proceeds |
| Embed failures across the fleet | all 8 endpoints down | check `llama-server` processes; embed jobs retry transient failures, so a brief outage self-heals |
| `mount_unavailable` in `coverage` | sentinel file missing or root not a directory | restore the mount; nothing under the root is pruned while it is unavailable (scan and reconcile both treat it as "don't touch") |
| Stalled documents in `coverage` | docs without an active publication, labeled by furthest stage | re-run `scan` (missing files re-register), or check the failed stage for those docs |
| Disk full | jobs start failing with I/O errors | free space; `gc --execute` (after stopping the worker) reclaims unreferenced objects with a grace window |

## 7. Maintenance while a run is in progress

Safe with the worker running (WAL; read-only or append-only):
`status`, `coverage`, `search`, `doctor`, `scan` (drop new books into
the source root, then scan — the fast check skips everything unchanged,
so existing books are not recomputed), `retry`, and `discover`
(the same scan repeated on an interval as a *second* detached
process, so new books land without re-running `scan` by hand —
`nohup uv run library-rag discover --config config.yaml
>> /mnt/models_sata_ssd/library-rag/scratch/discover.log 2>&1 &` —
SIGTERM stops it after the in-flight pass, like the worker; it
never deletes catalog content, a vanished file is only reported),
and `migrate` (generation-migration planner: which runs need
re-chunking / re-embedding under the current chunker or embedding
configuration, which publications the new generation will supersede,
and whether the new generation fits beside the current one — planning
is read-only, and `--execute` only enqueues the reconcile jobs the
worker's own startup pass would enqueue, so both are safe mid-run).

Requires the worker stopped first (SIGTERM, then run, then resume):
`backup` (→ `restore`/`verify` into an isolated directory), `gc`
(dry-run by default; `--execute` prunes), `remove <path-or-doc-id>
[--execute]` (explicit removal; points-first with verification),
`reconcile` (also runs automatically on worker start, so manual runs are
rarely needed).

If `migrate` reports a maintenance window (the new generation will not
fit beside the current one): SIGTERM the worker, `gc --execute` to
reclaim superseded point sets, re-run `migrate` to re-check capacity,
then `migrate --execute --accept-maintenance-window`.

## 8. Known constraints

- Serial worker by design in local Qdrant mode; ETA above assumes the
  pilot rate. OCR dominates runtime (tesseract, single-threaded per
  call by configuration).
- `stats_epoch: frozen` (pilot report §8): corpus stats are stamped once
  per campaign, so publishes do not fan out O(N²) re-publishes.
- Backups on `/mnt/models_sas_ssd/library-rag/backups` are on the same
  hardware as the data they protect — they recover accidental loss, not
  disk failure.
- The zero-byte sentinel lives *inside* the source root (the only file
  in the library that is not a book); scans ignore it.
