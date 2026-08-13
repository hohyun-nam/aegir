#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
aegir_root=$(realpath "$script_dir/..")
workspace=$(realpath "$aegir_root/..")
base=$(realpath "$workspace/..")
dns_env="${DN_ENV:-$base/.conda/envs/dnsplatter}"
python_bin="$dns_env/bin/python"
log_dir="$workspace/AEGIR/logs"
log_file="$log_dir/hypersim_remaining.log"
status_file="$log_dir/hypersim_remaining.status"
pid_file="$log_dir/hypersim_remaining.pid"
scenes=(ai_010_006 ai_015_003 ai_032_001 ai_034_002)

mkdir -p "$log_dir"
printf '%s\n' "$$" > "$pid_file"
printf 'running pid=%s started=%s\n' "$$" "$(date -Is)" > "$status_file"

finish() {
    code=$?
    if (( code == 0 )); then
        printf 'completed finished=%s\n' "$(date -Is)" > "$status_file"
    else
        printf 'failed exit=%s finished=%s\n' "$code" "$(date -Is)" > "$status_file"
    fi
}
trap finish EXIT

exec >> "$log_file" 2>&1
printf '\n[%s] tmux restart\n' "$(date -Is)"

for scene in "${scenes[@]}"; do
    output_dir="$workspace/datasets/aegir_nerfstudio/hypersim_$scene"
    if [[ -f "$output_dir/transforms.json" ]]; then
        printf '[%s] SKIP completed %s\n' "$(date -Is)" "$scene"
        continue
    fi

    printf '[%s] START %s\n' "$(date -Is)" "$scene"
    PYTHONUNBUFFERED=1 "$python_bin" "$workspace/AEGIR/scripts/export_hypersim_nerfstudio.py" \
        --scene-dir "$workspace/datasets/raw/hypersim/$scene" \
        --output-dir "$output_dir"
    printf '[%s] DONE %s\n' "$(date -Is)" "$scene"
done
