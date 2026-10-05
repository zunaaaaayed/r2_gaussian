"""Generate matched noise-free 360-degree and 120-degree cone-beam cases."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import tigre

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from r2_gaussian.utils.ct_utils import get_geometry_tigre


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project(volume, scanner, angles):
    """Match the original generator's xyz->zyx and detector-row conventions."""
    geometry = get_geometry_tigre(copy.deepcopy(scanner))
    volume_zyx = np.ascontiguousarray(volume.transpose(2, 1, 0))
    projections = tigre.Ax(volume_zyx, geometry, angles)[:, ::-1, :]
    projections = np.ascontiguousarray(projections, dtype=np.float32)

    expected = (len(angles), *scanner["nDetector"])
    assert projections.shape == expected, (projections.shape, expected)
    assert np.isfinite(projections).all()
    return projections


def save_split(case, name, projections, angles):
    folder = case / name
    folder.mkdir()
    frames = []

    for index, (projection, angle) in enumerate(zip(projections, angles)):
        relative = Path(name) / f"{name}_{index:04d}.npy"
        np.save(case / relative, projection)
        frames.append({
            "file_path": relative.as_posix(),
            "angle": float(angle),
        })

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with (args.source / "meta_data.json").open() as stream:
        source_meta = json.load(stream)

    scanner = copy.deepcopy(source_meta["scanner"])
    assert scanner["mode"] == "cone"
    scanner.update(noise=False, startAngle=0.0)

    volume_path = args.source / source_meta["vol"]
    volume = np.ascontiguousarray(np.load(volume_path), dtype=np.float32)
    assert volume.shape == tuple(scanner["nVoxel"])
    assert np.isfinite(volume).all()

    cases = {
        arc: args.output / f"arc_{arc}" / "0_chest_cone"
        for arc in (360, 120)
    }
    for case in cases.values():
        if case.exists():
            raise FileExistsError(
                f"{case} already exists. Choose a fresh output directory."
            )

    # Fixed full-circle evaluation angles at bin midpoints.
    # These do not overlap either training-angle grid.
    test_angles = np.deg2rad((np.arange(100) + 0.5) * 3.6)
    print("Generating shared evaluation projections...", flush=True)
    test_projections = project(volume, scanner, test_angles)

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()

    for arc, case in cases.items():
        cfg = copy.deepcopy(scanner)
        cfg["totalAngle"] = float(arc)
        train_angles = np.deg2rad(np.arange(50) * arc / 50.0)

        distances = np.abs(train_angles[:, None] - test_angles[None, :])
        assert not np.any(distances < 1e-10)

        print(f"Generating 50 training projections over {arc} degrees...",
              flush=True)
        train_projections = project(volume, cfg, train_angles)

        case.mkdir(parents=True)
        np.save(case / "vol_gt.npy", volume)

        train_frames = save_split(
            case, "proj_train", train_projections, train_angles
        )
        test_frames = save_split(
            case, "proj_test", test_projections, test_angles
        )

        metadata = {
            "scanner": cfg,
            "vol": "vol_gt.npy",
            "bbox": source_meta.get("bbox", [[-1, -1, -1], [1, 1, 1]]),
            "proj_train": train_frames,
            "proj_test": test_frames,
        }
        (case / "meta_data.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        provenance = {
            "git_commit": git_commit,
            "generator_sha256": sha256(Path(__file__)),
            "source_volume_sha256": sha256(volume_path),
            "source_metadata_sha256": sha256(args.source / "meta_data.json"),
            "noise": False,
            "n_train": 50,
            "n_test": 100,
            "start_degrees": 0.0,
            "arc_degrees": arc,
            "endpoint_included": False,
            "last_training_angle_degrees": float(
                np.rad2deg(train_angles[-1])
            ),
            "evaluation": "Shared full-circle midpoint grid; evaluation only",
            "initialization": "Not copied; must be generated per case",
        }
        (case / "generation.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )

        # Generated-data checksums, before adding initialization.
        files = sorted(p for p in case.rglob("*") if p.is_file())
        with (case / "dataset.sha256").open("w") as stream:
            for path in files:
                stream.write(
                    f"{sha256(path)}  {path.relative_to(case).as_posix()}\n"
                )

        print(f"Saved: {case}", flush=True)

    print("Matched dataset generation: PASSED")


if __name__ == "__main__":
    main()
