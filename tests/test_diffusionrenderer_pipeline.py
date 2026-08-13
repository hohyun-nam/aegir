from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def run_script(name: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        check=True,
        cwd=ROOT,
    )


def test_prepare_and_import_material_priors_preserve_frame_identity(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "masks").mkdir()
    frames = []
    for index in range(5):
        rgb = np.full((4, 8, 3), index * 20 + 10, dtype=np.uint8)
        mask = np.full((4, 8), 255, dtype=np.uint8)
        mask[0, 0] = 0
        Image.fromarray(rgb).save(data / "images" / f"frame_{index:04d}.png")
        Image.fromarray(mask).save(data / "masks" / f"frame_{index:04d}.png")
        frames.append(
            {
                "file_path": f"images/frame_{index:04d}.png",
                "mask_path": f"masks/frame_{index:04d}.png",
            }
        )
    (data / "transforms.json").write_text(
        json.dumps(
            {
                "frames": frames,
                "train_filenames": [frame["file_path"] for frame in frames[:4]],
                "val_filenames": [frames[4]["file_path"]],
                "test_filenames": [frames[4]["file_path"]],
            }
        ),
        encoding="utf-8",
    )

    prepared = tmp_path / "prepared"
    run_script(
        "prepare_diffusionrenderer_input.py",
        "--nerfstudio-dir",
        str(data),
        "--output-dir",
        str(prepared),
        "--chunk-size",
        "3",
    )
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    assert [frame["frame_index"] for frame in manifest["frames"]] == list(range(5))
    assert manifest["uses_gt_depth"] is False
    assert manifest["uses_gt_normal"] is False
    assert np.asarray(Image.open(prepared / "chunk_0000" / "frame_0000.png"))[0, 0].sum() == 0

    result = tmp_path / "result"
    for frame in manifest["frames"]:
        folder = result / frame["chunk_name"]
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f"0000.{frame['local_index']:04d}"
        Image.fromarray(np.full((6, 6, 3), 128, dtype=np.uint8)).save(folder / f"{prefix}.basecolor.png")
        for name, value in (("roughness", 64), ("metallic", 32)):
            Image.fromarray(np.full((6, 6, 3), value, dtype=np.uint8)).save(folder / f"{prefix}.{name}.png")

    priors = tmp_path / "priors"
    run_script(
        "import_diffusionrenderer_priors_to_dir.py",
        "--input-dir",
        str(prepared),
        "--result-dir",
        str(result),
        "--output-dir",
        str(priors),
    )
    for index in range(5):
        payload = torch.load(priors / f"frame_{index:04d}.pt", weights_only=False)
        assert payload["frame_index"] == index
        assert payload["prior_albedo"].shape == (6, 6, 3)
        assert payload["prior_roughness"].shape == (6, 6, 1)
        assert payload["prior_metallic"].shape == (6, 6, 1)
        assert "prior_depth" not in payload
        assert "prior_normal" not in payload
