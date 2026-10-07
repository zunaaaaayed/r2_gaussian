#!/usr/bin/env bash
set -euo pipefail
cd /autofs/unitytravail/travail/mdzislam/trdp2/r2_gaussian
exec > >(tee -a output/limited_angle_generalization_queue/start90_20261007.log) 2>&1
trap 'code=$?; echo "Queue exit status: $code at $(date -Is)"' EXIT
export MPLBACKEND=Agg
TRDP2_PY=/net/cremi/mdzislam/espaces/travail/trdp2/tools/miniforge3/envs/trdp2-r2/bin/python
TRDP2_RUNNER=experiments/limited_angle_generalization/run_cases.py
TRDP2_MANIFEST=experiments/limited_angle_generalization/manifests/development_v4.json
TRDP2_CASE=chest_start90_span120
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
for stage in generate verify initialize train evaluate; do
    if [[ "$stage" == generate || "$stage" == initialize || "$stage" == train ]]; then
        wait_for_gpu
    fi
    echo "Starting $stage at $(date -Is)"
    if [[ "$stage" == verify ]]; then
        "$TRDP2_PY" -u "$TRDP2_RUNNER" --manifest "$TRDP2_MANIFEST" --case "$TRDP2_CASE" --stage "$stage"
    else
        "$TRDP2_PY" -u "$TRDP2_RUNNER" --manifest "$TRDP2_MANIFEST" --case "$TRDP2_CASE" --stage "$stage" --execute
    fi
done
echo "All stages completed at $(date -Is)"
