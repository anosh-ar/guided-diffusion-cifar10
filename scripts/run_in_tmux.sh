#!/usr/bin/env bash
# Run a command detached in a tmux session (with low CPU priority), so training survives a closed SSH connection.
#
# Usage:  scripts/run_in_tmux.sh <session_name> <command...>
# Example:
#   scripts/run_in_tmux.sh ddpm bash scripts/reproduce.sh
#   tmux attach -t ddpm      # watch progress; detach with Ctrl-b d
set -euo pipefail

show_help() {
    echo "Usage: $0 <session_name> <command...>"
    echo "Set CONDA_ENV=<name> to activate a conda environment first."
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi
if [[ "$#" -lt 2 ]]; then
    echo "Error: expected a session name and a command."
    show_help
    exit 1
fi

SESSION=$1
shift
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO_DIR/runs/${SESSION}.log"
mkdir -p "$REPO_DIR/runs"

nice -n 10 tmux new-session -d -s "$SESSION" -c "$REPO_DIR"
if [[ -n "${CONDA_ENV:-}" ]]; then
    tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" C-m
fi
tmux send-keys -t "$SESSION" "$* 2>&1 | tee -a '$LOG'" C-m
echo "Started '$*' in tmux session '$SESSION' (log: $LOG). Attach with: tmux attach -t $SESSION"
