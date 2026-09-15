#!/usr/bin/env bash

set -e

SESSION="preg-comp-dev"
PROJECT_DIR="${1:-$HOME/preg-comp}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists."
    tmux attach-session -t "$SESSION"
    exit 0
fi

tmux new-session -d \
    -s "$SESSION" \
    -n "nvim" \
    -c "$PROJECT_DIR" \
    "nvim ."

tmux new-window \
    -t "$SESSION" \
    -n "codex" \
    -c "$PROJECT_DIR" \
    "codex ."

tmux select-window -t "$SESSION:nvim"

tmux attach-session -t "$SESSION"

