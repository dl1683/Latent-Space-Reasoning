#!/bin/bash
# Run Codex blind reviews for all completed legal v2 tasks.
# Usage: bash experiments/run_codex_eval.sh

set -e
cd "$(dirname "$0")/.."

# First, extract blind review files for all complete tasks
python experiments/eval_legal_batch.py experiments/legal_v2_full.json

# Find all blind review files and run Codex on each
for f in experiments/blind_review_v2_*.json; do
    task_id=$(basename "$f" .json | sed 's/blind_review_//')
    output="experiments/codex_review_${task_id}.txt"

    if [ -f "$output" ]; then
        echo "SKIP: $task_id (review already exists)"
        continue
    fi

    echo "REVIEWING: $task_id"
    prompt=$(sed "s/TASKID/${task_id}/g" experiments/codex_review_template.txt)
    codex exec -s read-only --skip-git-repo-check -C "$(pwd)" -o "$output" "$prompt" 2>&1
    echo "DONE: $task_id -> $output"
done

echo ""
echo "All reviews complete. Files in experiments/codex_review_v2_*.txt"
