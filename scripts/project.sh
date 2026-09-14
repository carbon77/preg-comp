#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
RUN_DIR="$PROJECT_ROOT/.run"
PID_FILE="$RUN_DIR/streamlit.pid"
PORT="${PORT:-8501}"

usage() {
  cat <<'EOF'
Usage: bash scripts/project.sh <command> [args]

Commands:
  init                 Create runtime directories.
  run                  Start Streamlit in the foreground.
  start                Start Streamlit in background and save PID/log.
  stop                 Stop the background Streamlit process.
  status               Show saved PID and process state.
  find [pattern]       Find project files by name (default: *.py).
  grep <pattern>       Search text in source/config files.
  analyze              Summarize logs and Python source files.
  system               Print Linux/system and disk information.
  clean                Remove generated logs, PID and temporary results.
EOF
}

init_dirs() {
  mkdir -p "$LOG_DIR" "$RUN_DIR" "$PROJECT_ROOT/results" "$PROJECT_ROOT/tmp"
}

run_foreground() {
  init_dirs
  exec uv run streamlit run main.py --server.port "$PORT"
}

start_background() {
  init_dirs
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Streamlit is already running with PID $(cat "$PID_FILE")"
    exit 0
  fi
  nohup uv run streamlit run main.py --server.port "$PORT" > "$LOG_DIR/streamlit.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "Started Streamlit with PID $pid"
  echo "Log: $LOG_DIR/streamlit.log"
}

stop_background() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "No saved PID file: $PID_FILE"
    exit 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "Stopped PID $pid"
  else
    echo "Process $pid is not running"
  fi
  rm -f "$PID_FILE"
}

status_process() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      ps -o pid,ppid,stat,etime,cmd -p "$pid"
      return
    fi
    echo "Saved PID $pid is not running"
  else
    echo "No saved Streamlit PID"
  fi
  pgrep -af 'streamlit run main.py' || true
}

find_files() {
  local pattern="${1:-*.py}"
  find "$PROJECT_ROOT" -type f -name "$pattern" \
    -not -path '*/.git/*' \
    -not -path '*/.venv/*' \
    -not -path '*/__pycache__/*' | sort
}

grep_project() {
  if [[ $# -lt 1 ]]; then
    echo "grep requires a pattern" >&2
    exit 2
  fi
  grep -RInE --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ \
    --exclude='*.ipynb' --exclude='uv.lock' "$1" "$PROJECT_ROOT" || true
}

analyze_project() {
  echo '--- Python files ---'
  find_files '*.py'
  echo
  echo '--- Python file statistics ---'
  find "$PROJECT_ROOT/app" -type f -name '*.py' -print0 | xargs -0 wc -l | sort -n
  echo
  echo '--- Log statistics ---'
  if [[ -d "$LOG_DIR" ]]; then
    find "$LOG_DIR" -type f -maxdepth 1 -print0 | xargs -0 -r wc -l | sort -n
    if [[ -f "$LOG_DIR/streamlit.log" ]]; then
      echo 'Last 20 log lines:'
      tail -n 20 "$LOG_DIR/streamlit.log"
    fi
  fi
}

system_info() {
  echo '--- Identity and location ---'
  whoami
  pwd
  echo
  echo '--- OS/kernel ---'
  uname -a
  echo
  echo '--- Tools ---'
  command -v bash || true
  command -v python || true
  command -v uv || true
  command -v make || true
  command -v tmux || true
  echo
  echo '--- Disk usage ---'
  du -sh "$PROJECT_ROOT"
  du -sh "$PROJECT_ROOT"/* 2>/dev/null | sort -h || true
  df -h "$PROJECT_ROOT"
}

clean_generated() {
  rm -rf "$LOG_DIR" "$RUN_DIR" "$PROJECT_ROOT/results" "$PROJECT_ROOT/tmp"
  echo 'Generated runtime data removed.'
}

command_name="${1:-}"
shift || true
case "$command_name" in
  init) init_dirs ;;
  run) run_foreground ;;
  start) start_background ;;
  stop) stop_background ;;
  status) status_process ;;
  find) find_files "$@" ;;
  grep) grep_project "$@" ;;
  analyze) analyze_project ;;
  system) system_info ;;
  clean) clean_generated ;;
  -h|--help|help|'') usage ;;
  *) echo "Unknown command: $command_name" >&2; usage; exit 2 ;;
esac
