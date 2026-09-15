#!/usr/bin/env bash

set -e

SESSION="preg-comp-monitor"
PROJECT_DIR="${1:-$HOME/preg-comp}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists."
    tmux attach-session -t "$SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" -n "monitor" -c "$PROJECT_DIR"

tmux send-keys -t "$SESSION:monitor" "htop" C-m

tmux split-window -h -t "$SESSION:monitor" -c "$PROJECT_DIR"

tmux send-keys -t "$SESSION:monitor.1" "watch -n 2 'ps aux --sort=-%cpu | head -15'" C-m

tmux split-window -v -t "$SESSION:monitor.1" -c "$PROJECT_DIR"

tmux send-keys \
    -t "$SESSION:monitor.2" \
    "watch -n 2 'echo === MEMORY ===; free -h; echo; echo === DISK ===; df -h .; echo; echo === PROJECT ===; du -sh .'" \
    C-m

tmux select-pane -t "$SESSION:monitor.0"

tmux split-window -v -t "$SESSION:monitor.0" -c "$PROJECT_DIR"

tmux send-keys \
    -t "$SESSION:monitor.3" \
    "watch -n 2 'echo === STREAMLIT ===; pgrep -af \"streamlit run main.py\" || true; echo; echo === PROJECT ===; pwd; echo; echo === GIT ===; git status --short'" \
    C-m

tmux select-pane -t "$SESSION:monitor.0"
tmux attach-session -t "$SESSION"
