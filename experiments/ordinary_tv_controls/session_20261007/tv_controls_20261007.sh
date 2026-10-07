#!/usr/bin/env bash
set -euo pipefail
cd /autofs/unitytravail/travail/mdzislam/trdp2/r2_gaussian
exec > >(tee -a output/limited_angle_generalization_queue/tv_controls_20261007.log) 2>&1
trap 'code=$?; echo "Queue exit status: $code at $(date -Is)"' EXIT
export MPLBACKEND=Agg
wait_for_gpu() {
    local deadline=$((SECONDS + 28800))
    local processes utilization
    echo "Waiting for an idle GPU at $(date -Is)"
    while true; do
        processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        utilization=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        if [[ -z "$processes" && "$utilization" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] && (( utilization < 10 )); then
            echo "GPU available at $(date -Is)"
            return
        fi
        if (( SECONDS >= deadline )); then
            echo "GPU wait timed out; no further stages launched"
            return 1
        fi
        sleep 30
    done
}
for weight in 0.025 0.1; do
    for case in chest_start0_span120 chest_start90_span120; do
        wait_for_gpu
        echo "Starting $case TV=$weight at $(date -Is)"
        /net/cremi/mdzislam/espaces/travail/trdp2/tools/miniforge3/envs/trdp2-r2/bin/python -u experiments/ordinary_tv_controls/run.py --case "$case" --weight "$weight" --execute
    done
done
echo "All four TV controls completed at $(date -Is)"
