#!/bin/bash
# serve.sh — operator control script for the library-rag serve app.
#
#   ./serve.sh start     launch serve detached (nohup; log -> repo-root serve.log)
#   ./serve.sh stop      graceful shutdown: POST /system/shutdown, SIGTERM/SIGKILL fallback
#   ./serve.sh status    config paths + serve + ingest worker + model fleet status
#   ./serve.sh ingest    start the ingest worker detached (log -> repo-root worker.log);
#                        runs until stopped with kill -TERM <pid> (graceful)
#   ./serve.sh ingest N  bounded run: at most N new books end-to-end, then the
#                        worker exits on its own (out-of-budget books are parked
#                        for a later run; re-run to continue)
#
# This script owns serve.pid (repo root): start writes the python pid into it,
# stop removes it. The model fleet (bge-m3 8081-8088, vLLM :8091, ak:8080) is
# NOT managed here — it auto-restarts at boot and is operator-managed — status
# only reports on it. The ingest worker can be started from here (ingest [N])
# but can never run alongside serve (embedded Qdrant is single-process): start
# refuses while a worker is up, and so does ingest. Stop the worker with
# kill -TERM <pid> (it handles SIGTERM gracefully).
#
# status reports every configured root against the filesystem: an unmounted
# drive shows as EMPTY/MISSING with a hint instead of failing later inside the
# app with an obscure permission error. start pre-checks the same paths and
# dies on missing source roots (writable roots are auto-created, so only warned).
#
# Config: $LIBRARY_RAG_CONFIG (default: the live pilot-sandbox config).
# <config dir>/serve.env is sourced if present (token env; values never echoed).

set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${LIBRARY_RAG_CONFIG:-/mnt/models_sas_ssd/library-rag/pilot-sandbox/scratch/config.sandbox.yaml}"
APP_PORT=8100            # keep in sync with services.app_port in $CONFIG
BASE="http://127.0.0.1:$APP_PORT"
PIDFILE="$PWD/serve.pid"
LOG="$PWD/serve.log"
WORKER_LOG="$PWD/worker.log"
ENV_FILE="$(dirname "$CONFIG")/serve.env"

# [c] character classes keep the script's own shell (whose command line
# contains the pattern text) from matching itself under pgrep -f. Both are
# anchored at .venv/bin/ so `uv run library-rag ...` wrappers do not match —
# only the real python process does (the one that handles SIGTERM).
SERVE_PROC='\.venv/bin/[l]ibrary-rag serve'
WORKER_PROC='\.venv/bin/[l]ibrary-rag ingest'

# kind<TAB>path for every root in $CONFIG, via the app's own config loader.
config_roots() {
  [ -x "$PWD/.venv/bin/python" ] || return 1
  "$PWD/.venv/bin/python" - "$CONFIG" <<'PY'
import sys

from library_rag.config import load_config

p = load_config(sys.argv[1]).paths
for kind, path in (
    [("source", r) for r in p.source_roots]
    + [
        ("archive", p.archive_root),
        ("artifact", p.artifact_root),
        ("state", p.state_root),
        ("qdrant", p.qdrant_root),
        ("model", p.model_root),
        ("scratch", p.scratch_root),
    ]
):
    print(f"{kind}\t{path}")
if p.backup_root:
    print(f"backup\t{p.backup_root}")
PY
}

# Prints " (on <dir> — drive unmounted?)" when <dir> looks like an unmounted
# mountpoint: an existing, EMPTY directory sitting on the root filesystem.
unmount_hint() {
  local p="$1" dev root_dev
  [ -d "$p" ] || return 0
  [ -z "$(ls -A "$p" 2>/dev/null | head -1)" ] || return 0
  command -v findmnt >/dev/null 2>&1 || return 0
  dev=$(findmnt -n -o SOURCE --target "$(realpath -m "$p")" 2>/dev/null || true)
  root_dev=$(findmnt -n -o SOURCE / 2>/dev/null || true)
  [ -n "${dev:-}" ] && [ "$dev" = "$root_dev" ] && printf ' (on %s — drive unmounted?)' "$p"
}

# status section: every configured root vs the filesystem.
report_paths() {
  local roots kind path dev hint ancestor
  echo "paths:"
  if ! roots=$(config_roots 2>/dev/null); then
    echo "  (no .venv or unreadable config — run 'uv sync' in the repo)"
    return 0
  fi
  while IFS=$'\t' read -r kind path; do
    if [ -d "$path" ]; then
      hint=$(unmount_hint "$path")
      dev=$(findmnt -n -o SOURCE --target "$(realpath -m "$path")" 2>/dev/null || true)
      if [ -n "${hint:-}" ]; then
        printf '  %-8s EMPTY    %s%s\n' "$kind" "$path" "$hint"
      else
        printf '  %-8s OK       %s%s\n' "$kind" "$path" "${dev:+  ($dev)}"
      fi
    else
      # Walk up to the nearest existing ancestor; if it is an unmounted
      # mountpoint, say so — that is how a missing drive usually presents.
      ancestor="$path"
      while [ ! -e "$ancestor" ] && [ "$ancestor" != "/" ]; do
        ancestor=$(dirname "$ancestor")
      done
      hint=$(unmount_hint "$ancestor")
      printf '  %-8s MISSING  %s%s\n' "$kind" "$path" "${hint:-}"
    fi
  done <<< "$roots"
}

die() { echo "serve.sh: $*" >&2; exit 1; }

load_env() {   # token env for API calls; values are never echoed
  if [ -f "$ENV_FILE" ]; then . "$ENV_FILE"; fi
}

serve_pid() { cat "$PIDFILE" 2>/dev/null || true; }

alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

discover_serve_pid() { pgrep -f "$SERVE_PROC" | head -1 || true; }
discover_worker_pid() { pgrep -f "$WORKER_PROC" | head -1 || true; }

health_code() { curl -s -m 2 -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null || echo 000; }

wait_for_exit() {   # $1 pid, $2 seconds
  local i
  for i in $(seq 1 "$2"); do
    kill -0 "$1" 2>/dev/null || return 0
    sleep 1
  done
  return 1
}

probe() {   # $1 label, $2 url
  local c
  c=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$2" 2>/dev/null || echo 000)
  if [ "$c" = "200" ]; then echo "$1: up"; else echo "$1: DOWN (http $c)"; fi
}

# --- start -------------------------------------------------------------------

cmd_start() {
  command -v uv >/dev/null 2>&1 || die "uv not on PATH"
  local pid wpid
  pid=$(serve_pid)
  if alive "${pid:-}"; then
    die "serve already running (pid $pid, :$APP_PORT)"
  fi
  if [ "$(health_code)" = "200" ]; then
    pid=$(discover_serve_pid)
    if [ -n "${pid:-}" ]; then echo "$pid" > "$PIDFILE"; fi
    die "something is already serving :$APP_PORT (pid ${pid:-unknown}) — use stop first"
  fi
  wpid=$(discover_worker_pid)
  if [ -n "${wpid:-}" ]; then
    die "ingest worker running (pid $wpid) — stop it first (embedded Qdrant is single-process)"
  fi
  [ -f "$CONFIG" ] || die "config not found: $CONFIG"
  # Pre-check the configured roots (status shows them all): a missing source
  # root means an unmounted drive and would fail later with an obscure error;
  # writable roots are auto-created, so only warn there.
  local roots kind path hint ancestor missing_src=0
  if roots=$(config_roots 2>/dev/null); then
    while IFS=$'\t' read -r kind path; do
      [ -d "$path" ] && continue
      ancestor="$path"
      while [ ! -e "$ancestor" ] && [ "$ancestor" != "/" ]; do
        ancestor=$(dirname "$ancestor")
      done
      hint=$(unmount_hint "$ancestor")
      if [ "$kind" = "source" ]; then
        echo "missing source root: $path${hint}" >&2
        missing_src=1
      else
        echo "note: writable root not present yet (created on first use): $path${hint}" >&2
      fi
    done <<< "$roots"
  fi
  [ "$missing_src" = "0" ] || die "source roots missing — mount the drive and retry ('status' shows all paths)"
  load_env
  nohup uv run library-rag serve --config "$CONFIG" >> "$LOG" 2>&1 &
  local i code=000
  for i in $(seq 1 30); do
    code=$(health_code)
    [ "$code" = "200" ] && break
    sleep 1
  done
  [ "$code" = "200" ] || die "serve did not come up within 30 s — see tail of $LOG"
  pid=$(discover_serve_pid)
  echo "${pid:-unknown}" > "$PIDFILE"
  echo "serve started (pid ${pid:-unknown}, :$APP_PORT, log: $LOG)"
  echo "ready: $(curl -s -m 30 "$BASE/ready" || echo unreachable)"
}

# --- stop --------------------------------------------------------------------

cmd_stop() {
  local pid resp
  pid=$(serve_pid)
  if ! alive "${pid:-}"; then
    if [ "$(health_code)" = "200" ]; then
      pid=$(discover_serve_pid)
      if [ -n "${pid:-}" ]; then
        echo "$pid" > "$PIDFILE"
        echo "pidfile stale — recovered pid $pid"
      fi
    else
      rm -f "$PIDFILE"
      echo "serve not running"
      return 0
    fi
  fi
  [ -n "${pid:-}" ] || die "serve appears up on :$APP_PORT but no pid could be found"
  load_env
  if [ -n "${LIBRARY_RAG_API_TOKEN:-}" ]; then
    resp=$(curl -s -m 5 -X POST -H "Authorization: Bearer $LIBRARY_RAG_API_TOKEN" \
      "$BASE/system/shutdown" 2>/dev/null || true)
  else
    resp=$(curl -s -m 5 -X POST "$BASE/system/shutdown" 2>/dev/null || true)
  fi
  if echo "${resp:-}" | grep -q 'shutting_down'; then
    echo "graceful shutdown requested via /system/shutdown (pid $pid)"
  else
    echo "API shutdown unavailable — sending SIGTERM to $pid"
    kill -TERM "$pid" 2>/dev/null || true
  fi
  if wait_for_exit "$pid" 30; then
    echo "serve stopped (pid $pid)"
  else
    echo "still alive after 30 s — sending SIGKILL to $pid" >&2
    kill -KILL "$pid" 2>/dev/null || true
    wait_for_exit "$pid" 10 || die "could not kill pid $pid"
    echo "serve killed (pid $pid)"
  fi
  rm -f "$PIDFILE"
}

# --- ingest ------------------------------------------------------------------

cmd_ingest() {   # $1 = optional N (max books)
  local n="${1:-}" extra=() pid i
  [ -f "$CONFIG" ] || die "config not found: $CONFIG"
  if [ -n "${n:-}" ]; then
    case "$n" in (*[!0-9]*) die "ingest N: N must be a positive integer, got '$n'" ;; esac
    [ "$n" -ge 1 ] || die "ingest N: N must be >= 1"
    extra=(--max-books "$n")
  fi
  pid=$(discover_worker_pid)
  if [ -n "${pid:-}" ]; then
    die "ingest worker already running (pid $pid)"
  fi
  if alive "$(serve_pid)" || [ "$(health_code)" = "200" ]; then
    die "serve is up — stop it first (embedded Qdrant is single-process): ./serve.sh stop"
  fi
  command -v uv >/dev/null 2>&1 || die "uv not on PATH"
  nohup uv run library-rag ingest --config "$CONFIG" "${extra[@]}" >> "$WORKER_LOG" 2>&1 &
  pid=""
  for i in $(seq 1 15); do
    pid=$(discover_worker_pid)
    [ -n "${pid:-}" ] && break
    sleep 1
  done
  [ -n "${pid:-}" ] || die "worker did not start within 15 s — see tail of $WORKER_LOG"
  echo "ingest worker started (pid $pid, log: $WORKER_LOG)"
  if [ -n "${n:-}" ]; then
    echo "bounded run: at most $n new book(s) end-to-end; exits on its own when done"
    echo "(out-of-budget books are parked — re-run './serve.sh ingest' to continue)"
  else
    echo "unbounded run: stops on 'kill -TERM $pid' (graceful); the queue keeps draining meanwhile"
  fi
}

# --- status ------------------------------------------------------------------

cmd_status() {
  local pid code st fleet_via_api=0
  echo "config: $CONFIG"
  report_paths
  pid=$(serve_pid)
  if alive "${pid:-}"; then
    local etime
    etime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ' || echo '?')
    code=$(health_code)
    echo "serve:  RUNNING  pid $pid  up $etime  /health $code  log: $LOG"
    echo "ready:  $(curl -s -m 30 "$BASE/ready" 2>/dev/null || echo unreachable)"
    st=$(curl -s -m 20 "$BASE/system/status?timeout=2" 2>/dev/null || true)
    if [ -n "${st:-}" ] && command -v jq >/dev/null 2>&1; then
      echo "endpoints:"
      echo "$st" | jq -r '.services.llm[]? | "  llm  \(.label) \(.host):\(.port): \(if .up then "up" else "DOWN" end) (\(.detail))"'
      echo "$st" | jq -r '.services.embedders[]? | "  embed \(.label): \(if .up then "up" else "DOWN" end) (\(.detail))"'
      if [ "$(echo "$st" | jq -r '.gpu.available')" = "true" ]; then
        echo "gpu:"
        echo "$st" | jq -r '.gpu.gpus[] | "  [\(.index)] \(.name): mem \(.mem_used_mib)/\(.mem_total_mib) MiB, util \(.util_pct)%, \(.temp_c) C"'
      fi
      if [ "$(echo "$st" | jq -r '.cpu | type' 2>/dev/null)" = "object" ]; then
        echo "cpu:  $(echo "$st" | jq -c '.cpu')"
      fi
      fleet_via_api=1
    fi
  else
    if [ "$(health_code)" = "200" ]; then
      echo "serve:  RUNNING but pidfile missing/stale — pid $(discover_serve_pid)"
    else
      echo "serve:  STOPPED  (:$APP_PORT free)"
    fi
  fi
  local wpid
  wpid=$(discover_worker_pid)
  if [ -n "${wpid:-}" ]; then
    echo "worker: RUNNING  pid $wpid  log: $WORKER_LOG"
  else
    echo "worker: not running — start with './serve.sh ingest [N]' (serve must be stopped)"
  fi
  if [ "$fleet_via_api" = "0" ]; then
    # serve down (or /system/status unusable): probe the fleet directly.
    echo "models:"
    probe "  answer primary 127.0.0.1:8091" "http://127.0.0.1:8091/health"
    probe "  embed 127.0.0.1:8081"          "http://127.0.0.1:8081/health"
    probe "  answer extra ak:8080"          "http://ak:8080/health"
  fi
}

# --- dispatch ----------------------------------------------------------------

case "${1:-}" in
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  ingest) cmd_ingest "${2:-}" ;;
  *) echo "usage: $0 {start|stop|status|ingest [N]}" >&2; exit 2 ;;
esac
