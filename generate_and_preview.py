"""
Generate a dataset (same layout as per-task scripts) then preview random clips.

Run from anywhere:
  python generate_and_preview.py elasticity
  python generate_and_preview.py friction --num-videos 500
  python generate_and_preview.py mass
  python generate_and_preview.py drag

Outputs: <this_dir>/generated6.0/<task>/dataset.npz and manifest.csv
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

TASKS = {
    "elasticity": {
        "module": "generate_elasticity",
        "out": ROOT / "generated6.0" / "elasticity",
        "property_key": "elasticity_values",
        "label": "elasticity",
        "property_range": (0.1, 0.95),
        "default_num_videos": 750,
    },
    "friction": {
        "module": "generate_friction",
        "out": ROOT / "generated6.0" / "friction",
        "property_key": "friction_values",
        "label": "friction",
        "property_range": (0.15, 1.2),
        "default_num_videos": 750,
    },
    "mass": {
        "module": "generate_mass",
        "out": ROOT / "generated6.0" / "mass",
        "property_key": "mass_ratio_m",
        "label": "mass_ratio_m",
        "property_range": (0.25, 4.0),
        "default_num_videos": 750,
    },
    "drag": {
        "module": "generate_drag",
        "out": ROOT / "generated6.0" / "drag",
        "property_key": "drag_coefficients",
        "label": "drag_coeff",
        "property_range": (0.3, 4.0),
        "default_num_videos": 750,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset + random preview.")
    parser.add_argument("task", choices=list(TASKS), help="Which dataset to generate.")
    parser.add_argument("--num-videos", type=int, default=None, help="Override video count.")
    parser.add_argument("--preview", type=int, default=10, help="Number of videos to play.")
    parser.add_argument("--dataset-seed", type=int, default=1337, help="Dataset RNG seed.")
    parser.add_argument("--skip-generate", action="store_true", help="Preview existing dataset.")
    parser.add_argument("--frame-ms", type=int, default=30, help="imshow frame delay (ms).")
    args = parser.parse_args()

    cfg = TASKS[args.task]
    out_dir = cfg["out"]
    num_videos = args.num_videos if args.num_videos is not None else cfg["default_num_videos"]

    if not args.skip_generate:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        mod = importlib.import_module(cfg["module"])
        sig = inspect.signature(mod.generate_dataset)
        kwargs = {
            "property_range": cfg["property_range"],
            "output_dir": str(out_dir),
            "dataset_seed": args.dataset_seed,
        }
        if "num_videos" in sig.parameters:
            kwargs["num_videos"] = num_videos
        elif "num_scenarios" in sig.parameters and "values_per_scenario" in sig.parameters:
            kwargs["num_scenarios"] = max(1, num_videos // 5)
            kwargs["values_per_scenario"] = 5
        mod.generate_dataset(**kwargs)

    npz_path = out_dir / "dataset.npz"
    if not npz_path.is_file():
        print(f"No dataset at {npz_path}", file=sys.stderr)
        sys.exit(1)

    d = np.load(npz_path)
    n = d["frames"].shape[0]
    k = min(args.preview, n)
    rng = np.random.default_rng()
    indices = rng.choice(n, size=k, replace=False)

    prop = d[cfg["property_key"]]
    label = cfg["label"]
    scenario_ids = d["scenario_id"] if "scenario_id" in d else None

    for i in indices:
        if scenario_ids is not None:
            print(f"video {int(i)}: {label}={float(prop[i]):.6f}, scenario={int(scenario_ids[i])}")
        else:
            print(f"video {int(i)}: {label}={float(prop[i]):.6f}")
        for f in d["frames"][i]:
            cv2.imshow(f"preview ({args.task})", f)
            if cv2.waitKey(args.frame_ms) == ord("q"):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
