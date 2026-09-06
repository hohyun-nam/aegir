#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace/hohyun_ws/AEGIR

DNS_ENV=/workspace/.conda/envs/dnsplatter
PYTHON="$DNS_ENV/bin/python"
SCENE=fipt_real_conference
RUN_TAG=fipt_real_conference_dnsplit_v8_20k_20260818

CHECKPOINT=/workspace/hohyun_ws/AEGIR/outputs/$RUN_TAG/$SCENE/aegir_20k.pt
DATA=/workspace/hohyun_ws/datasets/aegir_nerfstudio/$SCENE
PRIORS=/workspace/hohyun_ws/datasets/diffusionrenderer_priors/$SCENE
SUBSET_DATA=/workspace/hohyun_ws/AEGIR/outputs/$RUN_TAG/$SCENE/secondary20_uniform_nerfstudio
OUT=/workspace/hohyun_ws/AEGIR/outputs/$RUN_TAG/$SCENE/inspect_secondary_bounce_20_uniform
LOG=/workspace/hohyun_ws/AEGIR/logs/$RUN_TAG/${SCENE}_inspect_secondary20.log

mkdir -p "$SUBSET_DATA" "$(dirname "$LOG")"

"$PYTHON" - "$DATA" "$SUBSET_DATA" <<'PY'
import json
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

meta = json.loads((src / "transforms.json").read_text(encoding="utf-8"))
frames = meta["frames"]
n = len(frames)

if n <= 20:
    indices = list(range(n))
else:
    indices = sorted({round(i * (n - 1) / 19) for i in range(20)})

selected = []
for i in indices:
    frame = dict(frames[i])

    for key, value in list(frame.items()):
        if not key.endswith("_path") and key != "file_path":
            continue
        if isinstance(value, str) and value:
            frame_path = Path(value)
            if not frame_path.is_absolute():
                frame[key] = str((src / frame_path).resolve())

    selected.append(frame)

meta["frames"] = selected
meta.setdefault("aegir_subset", {})
meta["aegir_subset"].update({
    "source": str(src),
    "selection": "uniform_20",
    "source_frame_count": n,
    "indices": indices,
})

(dst / "transforms.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

for name in ("color_space.json", "splits.json"):
    src_file = src / name
    if src_file.exists():
        shutil.copy2(src_file, dst / name)

print("selected uniform frame indices:", indices)
PY

export CUDA_HOME="$DNS_ENV/lib/python3.10/site-packages/nvidia/cu13"
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR=/workspace/hohyun_ws/AEGIR/outputs/torch_extensions_cache_workspace
export PATH="$DNS_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH=/workspace/hohyun_ws/AEGIR/outputs/cuda_link_stubs:$CUDA_HOME/lib:$CUDA_HOME/lib64:$DNS_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LIBRARY_PATH=/workspace/hohyun_ws/AEGIR/outputs/cuda_link_stubs:$CUDA_HOME/lib:$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_HOME=/workspace/hohyun_ws/.cache/torch
export XDG_CACHE_HOME=/workspace/hohyun_ws/.cache
export HF_HOME=/workspace/hohyun_ws/.cache/huggingface
export PYTHONPATH=/workspace/hohyun_ws/AEGIR:/workspace/hohyun_ws/external/dn-splatter${PYTHONPATH:+:$PYTHONPATH}

"$PYTHON" -c '
import runpy
import torch

total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 24000 * 1024**2 / total), 0)
print(f"PyTorch CUDA memory cap: 24000 MiB / {total / 1024**2:.0f} MiB", flush=True)
runpy.run_module("aegir.cli", run_name="__main__")
' inspect \
  --checkpoint "$CHECKPOINT" \
  --nerfstudio-dir "$SUBSET_DATA" \
  --prior-dir "$PRIORS" \
  --output-dir "$OUT" \
  --device cuda \
  --split all \
  --render-samples 128 \
  --evaluation-seed 2026 \
  2>&1 | tee "$LOG"
