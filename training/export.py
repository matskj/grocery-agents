from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import SCHEMA_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-mode models to Rust policy artifact.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out", default="models/policy_artifacts.json")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    modes = {}
    for path in sorted(models_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mode = payload.get("mode")
        if not mode:
            continue
        modes[str(mode)] = {
            "weights": payload.get("weights", {}),
            "ordering_weights": payload.get("ordering_weights", {}),
            "feature_columns": payload.get("feature_columns", []),
            "normalization": payload.get("normalization", {}),
            "heads": payload.get("heads", {}),
            "calibration": payload.get("calibration", {}),
        }

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "modes": modes,
    }
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"wrote artifact with modes={sorted(modes.keys())} to {out_path}")


if __name__ == "__main__":
    main()
