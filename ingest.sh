#!/bin/bash
# ingest.sh — batch launcher for the library-rag ingest pipeline.
#
#   ./ingest.sh              scan all configured source roots (discover,
#                            register, enqueue extract jobs for every new
#                            book), then start the ingest worker detached,
#                            unbounded (log: worker.log); stop with
#                            kill -TERM <pid> (graceful) or library-rag stop
#   ./ingest.sh --limit N    scan enqueues at most N new books' extract jobs
#                            across all roots; new books past the limit are
#                            deferred (left unregistered, so the next scan
#                            picks them up). The worker is bounded to N books
#                            end-to-end and exits on its own when done;
#                            re-run to continue with the next batch
#
# The scan runs in the foreground — its per-root report (discovered /
# unchanged / new / enqueued / deferred) is the scheduling result. The worker
# starts exactly like `./serve.sh ingest`: same preconditions (no worker
# already running, serve down — embedded Qdrant is single-process), same pid
# discovery, same worker.log. A missing source root does not abort the run:
# the scan reports it ("mount unavailable ... skipped") and the worker still
# drains whatever is already queued.
#
# Config: $LIBRARY_RAG_CONFIG (default: the live pilot-sandbox config).

set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${LIBRARY_RAG_CONFIG:-/mnt/models_sas_ssd/library-rag/pilot-sandbox/scratch/config.sandbox.yaml}"
APP_PORT=8100            # keep in sync with services.app_port in $CONFIG
BASE="http://127.0.0.1:$APP_PORT"
PIDFILE="$PWD/serve.pid"
WORKER_LOG="$PWD/worker.log"

# [c] character classes keep the script's own shell (whose command line
# contains the pattern text) from matching itself under pgrep -f. Anchored at
# .venv/bin/ so `uv run library-rag ...` wrappers do not match — only the
# real python process does (the one that handles SIGTERM).
WORKER_PROC='\.venv/bin/[l]ibrary-rag ingest'

die() { echo "ingest.sh: $*" >&2; exit 1; }

usage() {
  cat <<EOF
usage: $0 [--limit N]

scan the configured source roots (discover, register, enqueue extract jobs),
then start the ingest worker:

  --limit N   the scan registers + enqueues at most N new books' extract jobs
              across all roots; new books past the limit are deferred — left
              unregistered, so the next scan picks them up. The worker is
              bounded to N books end-to-end and exits on its own when done
              (re-run to continue with the next batch)
  (no flag)   the scan enqueues every new book; the worker drains the queue
              until stopped with kill -TERM <pid> (graceful) or library-rag stop

config: $CONFIG  (override with LIBRARY_RAG_CONFIG)
EOF
}

discover_worker_pid() { pgrep -f "$WORKER_PROC" | head -1 || true; }
health_code() { curl -s -m 2 -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null || echo 000; }
serve_pid() { cat "$PIDFILE" 2>/dev/null || true; }
alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

# --- args ----------------------------------------------------------------------
limit=""
while [ $# -gt 0 ]; do
  case "$1" in
    --limit)
      [ $# -ge 2 ] || { echo "ingest.sh: --limit needs a value" >&2; usage >&2; exit 2; }
      limit="$2"
      shift 2
      ;;
    --limit=*) limit="${1#--limit=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ingest.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [ -n "$limit" ]; then
  case "$limit" in (*[!0-9]*) die "--limit: N must be a positive integer, got '$limit'" ;; esac
  [ "$limit" -ge 1 ] || die "--limit: N must be >= 1"
fi

# --- preconditions (refuse early, before a long scan) ---------------------------
[ -f "$CONFIG" ] || die "config not found: $CONFIG"
command -v uv >/dev/null 2>&1 || die "uv not on PATH"
wpid=$(discover_worker_pid)
if [ -n "${wpid:-}" ]; then
  die "ingest worker already running (pid $wpid)"
fi
if alive "$(serve_pid)"; then
  die "serve is up (pid $(serve_pid)) — stop it first (embedded Qdrant is single-process): ./serve.sh stop"
fi
if [ "$(health_code)" = "200" ]; then
  die "serve is up on :$APP_PORT — stop it first (embedded Qdrant is single-process): ./serve.sh stop"
fi

# --- scan: discover, register, enqueue (foreground — this is the report) --------
echo "scanning source roots ..."
scan_extra=()
if [ -n "$limit" ]; then
  scan_extra=(--limit "$limit")
fi
uv run library-rag scan --config "$CONFIG" "${scan_extra[@]}"
echo

# --- worker: detached, exactly like `./serve.sh ingest` --------------------------
extra=()
if [ -n "$limit" ]; then
  # Bound the worker to the same N so the run exits on its own when its
  # books are done (an unbounded worker would idle forever on the empty
  # queue and block the next batch).
  extra=(--max-books "$limit")
fi
nohup uv run library-rag ingest --config "$CONFIG" "${extra[@]}" >> "$WORKER_LOG" 2>&1 &
pid=""
for i in $(seq 1 15); do
  pid=$(discover_worker_pid)
  if [ -n "${pid:-}" ]; then break; fi
  sleep 1
done
[ -n "${pid:-}" ] || die "worker did not start within 15 s — see tail of $WORKER_LOG"
echo "ingest worker started (pid $pid, log: $WORKER_LOG)"
if [ -n "$limit" ]; then
  echo "bounded run: at most $limit new book(s) enqueued; the worker is bounded to $limit book(s) and exits on its own when done"
  echo "(new books past the limit are deferred — re-run './ingest.sh --limit $limit' to continue)"
else
  echo "unbounded run: stops on 'kill -TERM $pid' (graceful); the queue keeps draining meanwhile"
fi
