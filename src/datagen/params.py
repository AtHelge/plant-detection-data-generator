"""Loading parameter ranges and drawing dataset-level values."""

from __future__ import annotations

import hashlib
import math
from importlib import resources
from pathlib import Path

import numpy as np
import yaml

from .geometry import DISTORTION_KEYS, camera_from_dict, distortion_is_valid

_DISTORTION_ENTRIES = (*DISTORTION_KEYS, "cam_distortion_dev_frac")   # redrawn together
_MAX_LENS_DRAWS = 1000


def load_ranges(path: str | Path | None = None) -> tuple[dict, str]:
    """Return (parsed ranges, sha256 of the file bytes with CRLF normalized to LF). None = packaged default."""
    if path is None:
        data = resources.files("datagen").joinpath("ranges.yaml").read_bytes()
    else:
        data = Path(path).read_bytes()
    data = data.replace(b"\r\n", b"\n")
    return yaml.safe_load(data), hashlib.sha256(data).hexdigest()


def resolve_profile(ranges: dict, profile: str) -> dict:
    """Return the profile's sections with "base" inheritance applied."""
    profiles = ranges["profiles"]
    if profile not in profiles:
        raise ValueError(f"unknown profile {profile!r}, available: {sorted(profiles)}")
    spec = profiles[profile]
    resolved = resolve_profile(ranges, spec["base"]) if "base" in spec else {}
    for section in ("dataset", "image", "object"):
        resolved[section] = {**resolved.get(section, {}), **spec.get(section, {})}
    return resolved


def _count(per: str, drawn: dict) -> int:
    if per == "camera":
        return drawn["cameras"]
    if per == "pair":
        return drawn["cameras"] - 1
    if per == "corner":
        return 4
    if per == "pass":
        return drawn["passes"]
    raise ValueError(f"unknown per {per!r}")


def _draw_one(spec, rng: np.random.Generator):
    if not isinstance(spec, dict):
        return spec
    if "choice" in spec:
        return spec["choice"][int(rng.integers(0, len(spec["choice"])))]
    if spec.get("type") == "int":
        return int(rng.integers(spec["min"], spec["max"] + 1))
    return float(rng.uniform(spec["min"], spec["max"]))


def _draw_entry(spec, rng: np.random.Generator, drawn: dict):
    per = spec.get("per") if isinstance(spec, dict) else None
    if per is None:
        return _draw_one(spec, rng)
    return [_draw_one(spec, rng) for _ in range(_count(per, drawn))]


def _check(name: str, value, spec) -> None:
    if isinstance(spec, dict) and "choice" in spec:
        assert value in spec["choice"], f"{name}={value} not in {spec['choice']}"
    elif isinstance(spec, dict):
        assert spec["min"] <= value <= spec["max"], f"{name}={value} outside [{spec['min']}, {spec['max']}]"
    else:
        assert value == spec, f"{name}={value} != fixed {spec}"


def _shared_camera(drawn: dict) -> tuple[int, int, float, float, float, float]:
    """(W, H, fx, fy, cx, cy) of the camera model shared by the rig."""
    w, h = (int(v) for v in drawn["resolution_px"].split("x"))
    fx = (w / 2) / math.tan(math.radians(drawn["hfov_deg"]) / 2)
    return w, h, fx, fx * drawn["fy_ratio"], w / 2 + drawn["cx_offset_frac"] * w, h / 2 + drawn["cy_offset_frac"] * h


def _rig_cameras(drawn: dict) -> list[dict]:
    """Intrinsics of every rig camera: shared model plus individual deviations."""
    w, h, fx, fy, cx, cy = _shared_camera(drawn)
    cams = []
    for i in range(drawn["cameras"]):
        scale = 1.0 + drawn["cam_fx_dev_frac"][i]
        dist = 1.0 + drawn["cam_distortion_dev_frac"][i]
        cams.append({
            "width": w, "height": h, "fx": fx * scale, "fy": fy * scale,
            "cx": cx + drawn["cam_cx_dev_frac"][i] * w, "cy": cy + drawn["cam_cy_dev_frac"][i] * h,
            "distortion": {k: drawn[k] * dist for k in DISTORTION_KEYS},
        })
    return cams


def draw_dataset_params(ranges: dict, profile: str, rng: np.random.Generator) -> dict:
    """Draw all dataset-level values once; each is asserted to lie within its range.

    The distortion entries are redrawn (in file order) until every rig camera's lens maps its whole
    image one-to-one.
    """
    specs = resolve_profile(ranges, profile)["dataset"]
    drawn: dict = {}
    for name, spec in specs.items():
        drawn[name] = _draw_entry(spec, rng, drawn)

    for _ in range(_MAX_LENS_DRAWS):
        if all(distortion_is_valid(camera_from_dict(c)) for c in _rig_cameras(drawn)):
            break
        for name in _DISTORTION_ENTRIES:
            drawn[name] = _draw_entry(specs[name], rng, drawn)
    else:
        raise ValueError(f"no valid lens distortion found in {_MAX_LENS_DRAWS} draws; check the k/p ranges")

    for name, spec in specs.items():
        for v in drawn[name] if isinstance(drawn[name], list) else [drawn[name]]:
            _check(name, v, spec)

    cams = _rig_cameras(drawn)
    w, h, fx, fy, cx, cy = _shared_camera(drawn)
    drawn.update({
        "image_width_px": w, "image_height_px": h, "fx_px": fx, "fy_px": fy, "cx_px": cx, "cy_px": cy,
        "alt_agl_m": drawn["gsd_mm_per_px"] / 1000.0 * float(np.mean([c["fx"] for c in cams])),
        "distortion": {k: drawn[k] for k in DISTORTION_KEYS},
        "camera_intrinsics": cams,
    })
    return drawn
