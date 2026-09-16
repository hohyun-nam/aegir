"""Input provenance and content-checked cache manifests for practical training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEPTH_FORMAT = "aegir-predicted-depth-v1"
ARTIFACT_FORMAT = "aegir-artifact-provenance-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_predicted_depth(root: Path, metadata: dict | None = None) -> dict:
    root = Path(root)
    if metadata is None:
        metadata = json.loads((root / "transforms.json").read_text())
    provenance = metadata.get("aegir_depth_provenance", {})
    if provenance.get("origin") != "rgb_model" or provenance.get("gt_alignment") is not False:
        raise ValueError("Practical training requires RGB-model depth without GT alignment")
    if (
        provenance.get("unit") != "metre"
        or provenance.get("mode") != "z"
        or provenance.get("convention") != "opencv"
    ):
        raise ValueError("Predicted depth must be metre-scale OpenCV axial Z-depth")
    if float(metadata.get("depth_unit_scale_factor", 1.0)) != 1.0:
        raise ValueError("Normalized predicted depth must have unit scale 1")
    manifest_path = root / provenance.get("manifest", "predicted_depth/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("format") != DEPTH_FORMAT
        or manifest.get("origin") != "rgb_model"
        or manifest.get("gt_alignment") is not False
        or any(
            manifest.get(k) != v for k, v in {"unit": "metre", "mode": "z", "convention": "opencv"}.items()
        )
    ):
        raise ValueError("Unverified depth provenance; regenerate from the RGB depth model")
    if manifest.get("metadata_signature") != fingerprint(metadata):
        raise ValueError("Predicted-depth metadata content mismatch; regenerate its training view")
    records = {entry["file_path"]: entry for entry in manifest["frames"]}
    for frame in metadata["frames"]:
        record = records.get(frame["file_path"])
        if record is None or record["depth_file_path"] != frame.get("depth_file_path"):
            raise ValueError("Depth manifest does not match this frame/depth path")
        for key, hash_key in (("file_path", "rgb_sha256"), ("depth_file_path", "depth_sha256")):
            path = root / frame[key]
            if sha256_file(path) != record[hash_key]:
                raise ValueError(f"Depth provenance content mismatch: {path}")
    return manifest


def validate_depth_directory(directory: Path) -> dict:
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {
        "format": DEPTH_FORMAT,
        "origin": "rgb_model",
        "unit": "metre",
        "mode": "z",
        "convention": "opencv",
        "gt_alignment": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Uncertified model depth; regenerate metre Z-depth from RGB")
    for entry in manifest["frames"]:
        depth = directory / Path(entry["depth_file_path"]).name
        rgb = directory.parent / entry["file_path"]
        if sha256_file(depth) != entry["depth_sha256"] or sha256_file(rgb) != entry["rgb_sha256"]:
            raise ValueError(f"Model-depth cache content mismatch: {depth}")
    return manifest


def dataset_signature(root: Path) -> dict:
    root = Path(root)
    metadata = json.loads((root / "transforms.json").read_text())
    files = {}
    for frame in metadata["frames"]:
        for key in (
            "file_path",
            "depth_file_path",
            "mask_path",
            "emitter_mask_path",
            "radiance_hdr_file_path",
        ):
            if frame.get(key):
                path = root / frame[key]
                files[frame[key]] = sha256_file(path)
        # Native-2DGS views expose the certified metre depth through mono_depth.
        if not frame.get("depth_file_path") and (root / "mono_depth").exists():
            path = root / "mono_depth" / (Path(frame["file_path"]).stem + ".npy")
            files[str(path.relative_to(root))] = sha256_file(path)
    if metadata.get("ply_file_path"):
        files[metadata["ply_file_path"]] = sha256_file(root / metadata["ply_file_path"])
    return {"root": str(root.resolve()), "metadata": fingerprint(metadata), "files": files}


def artifact_request(kind: str, settings: dict, inputs: dict) -> dict:
    return {"format": ARTIFACT_FORMAT, "kind": kind, "settings": settings, "inputs": inputs}


def verify_cache(manifest_path: Path, request: dict, output: Path) -> bool:
    if not output.exists():
        return False
    if not manifest_path.exists():
        raise ValueError(f"Unverified cache {output}; choose a fresh output path")
    stored = json.loads(manifest_path.read_text())
    if stored.get("request") != request or stored.get("output_sha256") != sha256_file(output):
        raise ValueError(f"Cache provenance/configuration mismatch: {output}; choose a fresh output path")
    return True


def publish_artifact(manifest_path: Path, request: dict, output: Path) -> None:
    write_json(manifest_path, {"request": request, "output_sha256": sha256_file(output)})


def training_inputs(root: Path, geometry: Path, priors: Path | None = None) -> dict:
    result = {"data": dataset_signature(root), "geometry_sha256": sha256_file(geometry)}
    if priors is not None:
        result["priors"] = {
            str(path.relative_to(priors)): sha256_file(path)
            for path in sorted(priors.rglob("*"))
            if path.is_file()
        }
    return result


def validate_practical_geometry(geometry: Path) -> dict:
    geometry = Path(geometry)
    manifest = geometry.with_suffix(geometry.suffix + ".provenance.json")
    if not manifest.exists():
        raise ValueError("Geometry has no practical training provenance; retrain 2DGS")
    stored = json.loads(manifest.read_text())
    request = stored.get("request", {})
    if (
        request.get("kind") != "2dgs"
        or request.get("settings", {}).get("ALLOW_UNVERIFIED_DEPTH") != "0"
        or stored.get("output_sha256") != sha256_file(geometry)
    ):
        raise ValueError("Geometry content or practical predicted-depth provenance mismatch")
    view = request.get("inputs", {}).get("view_provenance", {}).get("request", {})
    depth = view.get("inputs", {}).get("depth", {})
    seed = view.get("inputs", {}).get("seed_provenance", {}).get("request", {})
    if (
        view.get("kind") != "native_2dgs_view"
        or view.get("settings", {}).get("allow_unverified_depth") is not False
        or depth.get("origin") != "rgb_model"
        or depth.get("gt_alignment") is not False
        or seed.get("kind") != "depth_seed"
        or seed.get("settings", {}).get("allow_gt_depth") is not False
    ):
        raise ValueError("Geometry does not prove a model-depth seed and native training view")
    return stored


def validate_model_depth_seed(seed: Path) -> dict:
    seed = Path(seed)
    manifest = seed.with_suffix(seed.suffix + ".provenance.json")
    stored = json.loads(manifest.read_text())
    request = stored.get("request", {})
    if (
        request.get("kind") != "depth_seed"
        or request.get("settings", {}).get("allow_gt_depth") is not False
        or stored.get("output_sha256") != sha256_file(seed)
    ):
        raise ValueError("Seed must be generated from certified RGB-model depth; regenerate the point cloud")
    return stored


def validate_native_view(root: Path) -> dict:
    root = Path(root)
    validate_depth_directory(root / "mono_depth")
    stored = json.loads((root / "geometry_view_provenance.json").read_text())
    request = stored.get("request", {})
    if (
        request.get("kind") != "native_2dgs_view"
        or request.get("settings", {}).get("allow_unverified_depth") is not False
        or stored.get("output_sha256") != sha256_file(root / "transforms.json")
        or stored.get("derived_signature") != dataset_signature(root)
    ):
        raise ValueError("Unverified/changed native 2DGS view; rebuild from model depth")
    validate_model_depth_seed(root / "point_cloud.ply")
    return stored
