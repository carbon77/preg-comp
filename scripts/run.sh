#!/usr/bin/env bash

set -e

SESSION="preg-comp"
PORT="${1:-8501}"

echo "Creating tmux session: $SESSION"
echo "Starting application on port $PORT"

tmux new-session -d \
    -s "$SESSION" \
    "uv run streamlit run main.py --server.port $PORT"

echo "Application started."
echo "Attach with:"
echo "tmux attach -t $SESSION"
