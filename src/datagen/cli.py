"""Command line interface: datagen generate --out DIR ..."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .check import SelfCheckError, self_check
from .detections import generate_detections, generate_plants
from .params import draw_dataset_params, load_ranges, resolve_profile
from .scenario import build_scenario
from .writer import write_detections, write_scenario


def generate(out: Path, profile: str, ranges_file: str | None, seed: int | None, overwrite: bool) -> dict:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise SystemExit(f"error: --out {out} is not empty (use --overwrite)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.Generator(np.random.PCG64(seed))

    ranges, ranges_sha256 = load_ranges(ranges_file)
    drawn = draw_dataset_params(ranges, profile, rng)
    spec = resolve_profile(ranges, profile)
    scenario = build_scenario(drawn, rng, spec["dataset"])
    plants = generate_plants(scenario, drawn, spec["object"], rng)
    tables, false_positives = generate_detections(scenario, drawn, spec["image"], spec["object"], plants, rng)
    write_scenario(scenario, out)
    write_detections(tables, out)

    manifest = {
        "generator_version": __version__,
        "seed": seed,
        "profile": profile,
        "ranges_sha256": ranges_sha256,
        "drawn": {**drawn, **scenario.drawn},
        "derived": scenario.derived,
        "counts": {
            "frames": int(scenario.images["frame_idx"].nunique()),
            "cameras": drawn["cameras"],
            "passes": drawn["passes"],
            "images": len(scenario.images),
            "plants": len(plants),
            "detections": sum(len(t) for t in tables.values()),
            "false_positives": false_positives,
        },
    }
    with open(out / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    try:
        self_check(out)
    except SelfCheckError as e:
        raise SystemExit(f"error: {out}: {e}") from None
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="datagen")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="generate a synthetic dataset")
    gen.add_argument("--out", required=True, type=Path, help="output folder (must be empty or not exist)")
    gen.add_argument("--profile", default="standard", choices=["small", "standard", "noisy"],
                     help="parameter profile (default: standard)")
    gen.add_argument("--ranges", default=None, help="ranges YAML (default: packaged ranges.yaml)")
    gen.add_argument("--seed", type=int, default=None,
                     help="random seed (default: drawn from the OS and stored in manifest.json)")
    gen.add_argument("--overwrite", action="store_true", help="delete the contents of a non-empty --out first")
    args = parser.parse_args(argv)

    if args.command == "generate":
        manifest = generate(args.out, args.profile, args.ranges, args.seed, args.overwrite)
        print(f"wrote {args.out / 'manifest.json'} (seed {manifest['seed']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
