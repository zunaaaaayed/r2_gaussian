"""Generate the pre-specified 90-degree rotation of the 120-degree case."""

import copy
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from generate import project, save_split, sha256


def main():
    source = Path(
        "data/trdp2/chest_50views_noisefree/arc_120/0_chest_cone"
    )
    output = Path(
        "data/trdp2/chest_50views_noisefree/"
        "arc_120_start_90/0_chest_cone"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    meta_path = source / "meta_data.json"
    meta = json.loads(meta_path.read_text())
    scanner = copy.deepcopy(meta["scanner"])
    assert scanner["mode"] == "cone"
    assert scanner["noise"] == False
    scanner.update(startAngle=90.0, totalAngle=120.0)

    source_angles = np.array([
        frame["angle"] for frame in meta["proj_train"]
    ])
    assert len(source_angles) == 50
    assert np.allclose(
        source_angles, np.deg2rad(np.arange(50) * 2.4)
    )
    angles = np.deg2rad(90.0 + np.arange(50) * 2.4)
    test_angles = np.array([
        frame["angle"] for frame in meta["proj_test"]
    ])
    assert len(test_angles) == 100
    assert np.allclose(
        test_angles, np.deg2rad((np.arange(100) + 0.5) * 3.6)
    )
    separation = np.abs(np.angle(
        np.exp(1j * (angles[:, None] - test_angles[None, :]))
    ))
    assert separation.min() > 1e-8, "Training/test angle overlap"

    volume_path = source / meta["vol"]
    volume = np.ascontiguousarray(
        np.load(volume_path, allow_pickle=False), dtype=np.float32
    )
    assert volume.shape == tuple(scanner["nVoxel"])
    assert np.isfinite(volume).all()

    print("Generating 50 views from 90 to 207.6 degrees...", flush=True)
    projections = project(volume, scanner, angles)

    output.mkdir(parents=True)
    shutil.copy2(volume_path, output / "vol_gt.npy")
    train_frames = save_split(output, "proj_train", projections, angles)

    for frame in meta["proj_test"]:
        src = source / frame["file_path"]
        dst = output / frame["file_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert sha256(src) == sha256(dst)

    rotated_meta = copy.deepcopy(meta)
    rotated_meta.update(
        scanner=scanner,
        vol="vol_gt.npy",
        proj_train=train_frames,
    )
    (output / "meta_data.json").write_text(
        json.dumps(rotated_meta, indent=2) + "\n"
    )

    provenance = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "generator_sha256": sha256(Path(__file__)),
        "projector_helper_sha256": sha256(
            Path(__file__).with_name("generate.py")
        ),
        "source_metadata_sha256": sha256(meta_path),
        "source_volume_sha256": sha256(volume_path),
        "start_degrees": 90.0,
        "arc_degrees": 120.0,
        "n_train": 50,
        "spacing_degrees": 2.4,
        "last_training_angle_degrees": 207.6,
        "endpoint_included": False,
        "noise": False,
        "evaluation": "Exact copy of original shared 100 held-out views",
        "initialization": "Must be generated from rotated training views",
    }
    (output / "generation.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    files = sorted(p for p in output.rglob("*") if p.is_file())
    with (output / "dataset.sha256").open("w") as stream:
        for path in files:
            stream.write(
                f"{sha256(path)}  {path.relative_to(output).as_posix()}\n"
            )

    print(f"Saved: {output}")
    print("Shared evaluation files verified; no train/test angle overlap.")


if __name__ == "__main__":
    main()
