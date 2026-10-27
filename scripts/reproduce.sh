#!/usr/bin/env bash
# Reproduce all results of the README:
#   1. train one model per beta schedule
#   2. FID / IS evaluation over guidance weights for every model
#   3. qualitative figures for the best model
#   4. collect figures and tables in assets/
#
# Configure with environment variables, e.g.
#   EPOCHS=200 SCHEDULES="cosine" NUM_SAMPLES=10000 bash scripts/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-100}
SCHEDULES=${SCHEDULES:-"linear cosine sigmoid"}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
GUIDANCE=${GUIDANCE:-"0 0.5 1 2 4"}
DATA_DIR=${DATA_DIR:-./data}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}

mkdir -p assets
python plot_schedules.py --out assets/beta_schedules.png

RUNS=()
for s in $SCHEDULES; do
    run="runs/cifar10_${s}"
    RUNS+=("$run")
    if [[ -f "$run/summary.json" ]]; then
        echo ">> $run already trained, skipping"
    else
        resume=()
        [[ -f "$run/checkpoints/last.pt" ]] && resume=(--resume "$run/checkpoints/last.pt")
        python train.py --schedule "$s" --epochs "$EPOCHS" --run_name "cifar10_${s}" --data_dir "$DATA_DIR" \
            --amp ${resume[@]+"${resume[@]}"} $EXTRA_TRAIN_ARGS
    fi
    if [[ ! -f "$run/eval/results.md" ]]; then
        python evaluate.py --checkpoint "$run/checkpoints/best.pt" --data_dir "$DATA_DIR" \
            --num_samples "$NUM_SAMPLES" --guidance_scales $GUIDANCE --amp
    fi
done

python compare_runs.py "${RUNS[@]}" --out assets/schedule_comparison

# pick the run with the lowest FID for the qualitative figures
best_run=$(python - "${RUNS[@]}" <<'EOF'
import json, sys
best = min(sys.argv[1:], key=lambda r: min(x["fid"] for x in json.load(open(f"{r}/eval/results.json"))["results"]))
print(best)
EOF
)
best_w=$(python -c "import json; r=json.load(open('$best_run/eval/results.json'))['results']; print(min(r, key=lambda x: x['fid'])['guidance_scale'])")
echo ">> best run: $best_run (w = $best_w)"

python sample.py --checkpoint "$best_run/checkpoints/best.pt" --guidance_scale "$best_w" \
    --guidance_scales $GUIDANCE --out_dir "$best_run/figures"

cp "$best_run/figures/class_grid.png" "$best_run/figures/trajectory.png" "$best_run/figures/guidance_sweep.png" assets/
cp "$best_run/eval/fid_vs_guidance.png" "$best_run/eval/results.md" assets/
cp "$best_run/loss_curve.png" assets/
echo ">> done. Figures and tables are in assets/ (results.md, schedule_comparison.md)."
