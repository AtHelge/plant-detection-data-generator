"""Scenario: field, rig, passes and one pose per image. Everything in UTM, no file output.

Local field frame: s = along the heading, c = across (to the right of the heading),
origin = drawn field centre.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon

from .geometry import (Camera, Pose, camera_from_dict, full_footprint, heading_axes, image_center_ground, utm_epsg,
                       wgs84_to_utm)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_START_MIN = datetime(2020, 1, 1, tzinfo=timezone.utc)
_FIELD_LENGTH_MIN_M = 1.0


@dataclass
class Scenario:
    epsg: int
    cameras: list[Camera]   # one per rig position (cam_index)
    camera_ids: list[str]
    field: Polygon          # UTM
    images: pd.DataFrame    # one row per image: true pose + reported pose (*_meas), UTM
    drawn: dict             # values drawn while building the scenario
    derived: dict           # values computed from the drawn ones


def cameras_from_params(p: dict) -> list[Camera]:
    return [camera_from_dict(c) for c in p["camera_intrinsics"]]


def ms_to_datetime(ms: int) -> datetime:
    return _EPOCH + timedelta(milliseconds=int(ms))


def _interval_on_line(poly: Polygon, y: float) -> tuple[float, float]:
    """x-extent of poly on the horizontal line at y."""
    minx, _, maxx, _ = poly.bounds
    cut = poly.intersection(LineString([(minx - 1.0, y), (maxx + 1.0, y)]))
    if cut.is_empty:
        raise ValueError(f"line y={y} misses footprint")
    return cut.bounds[0], cut.bounds[2]


def _length_on_vline(poly: Polygon, x: float) -> float:
    _, miny, _, maxy = poly.bounds
    cut = poly.intersection(LineString([(x, miny - 1.0), (x, maxy + 1.0)]))
    return cut.bounds[3] - cut.bounds[1]


def solve_rig(p: dict, cams: list[Camera], dataset_spec: dict) -> dict:
    """Mounting offsets so that the measured ground overlaps equal the drawn ones.

    Computed with the vehicle at the origin heading north: across = east (x), along = north (y).
    Each camera uses its own real (distorted) footprint.
    Cross overlap of pair (i, i+1) is measured on the across line through the mean along position
    of both image centres, as a fraction of the mean ground width of both footprints on that line.
    """
    n, alt = p["cameras"], p["alt_agl_m"]
    polys, centers = [], []
    for i in range(n):
        pose = Pose(0.0, 0.0, alt, p["yaw_dev_deg"][i], p["pitch_deg"][i], p["roll_deg"][i])
        polys.append(full_footprint(pose, cams[i]))
        centers.append(image_center_ground(pose, cams[i]))

    along_target = [f * cams[i].height * alt / cams[i].fy for i, f in enumerate(p["along_offset_frac"])]
    along = np.array([along_target[i] - centers[i][1] for i in range(n)])
    across = np.zeros(n)
    across[0] = -centers[0][0]
    for j in range(1, n):
        y_ref = (along_target[j - 1] + along_target[j]) / 2
        a_i, b_i = _interval_on_line(polys[j - 1], y_ref - along[j - 1])
        a_i, b_i = a_i + across[j - 1], b_i + across[j - 1]
        a_j, b_j = _interval_on_line(polys[j], y_ref - along[j])
        width = ((b_i - a_i) + (b_j - a_j)) / 2
        across[j] = b_i - p["cross_overlap_frac"][j - 1] * width - a_j

    left = _interval_on_line(polys[0], -along[0])[0] + across[0]
    right = _interval_on_line(polys[-1], -along[-1])[1] + across[-1]
    across -= (left + right) / 2

    # along-track step: harmonic mean of the footprint lengths makes the rig-mean overlap exact;
    # it is clamped so that every camera's own overlap stays in its range
    lengths = np.array([_length_on_vline(polys[i], centers[i][0]) for i in range(n)])
    l_h = n / np.sum(1.0 / lengths)

    def step(key: str) -> float:
        d = (1 - p[key]) * l_h
        spec = dataset_spec[key]
        if isinstance(spec, dict):
            lo, hi = (1 - spec["max"]) * lengths.max(), (1 - spec["min"]) * lengths.min()
            if lo <= hi:
                d = min(max(d, lo), hi)
        return float(d)

    return {
        "mount_along_m": along.tolist(),
        "mount_across_m": across.tolist(),
        "rig_width_m": right - left,
        "step_normal_m": step("along_overlap_frac"),
        "step_standstill_m": step("standstill_overlap_frac"),
    }


def _field_corners(length: float, width: float, p: dict) -> list[tuple[float, float]]:
    """(s, c) corners in order bottom-left, bottom-right, top-right, top-left."""
    nominal = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    corners = []
    for i, (ss, cs) in enumerate(nominal):
        s = ss * length / 2 + p["corner_jitter_along_frac"][i] * length
        c = cs * (width / 2 - p["corner_jitter_across_frac"][i] * width)
        corners.append((s, c))
    return corners


def _edge_s(c: float, p0, p1) -> float:
    t = (c - p0[1]) / (p1[1] - p0[1])
    return p0[0] + t * (p1[0] - p0[0])


def _smallest(f, lo: float, hi: float, target: int) -> float:
    """Smallest x in [lo, hi] with f(x) >= target, f monotone non-decreasing, f(hi) >= target."""
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def build_scenario(p: dict, rng: np.random.Generator, dataset_spec: dict) -> Scenario:
    """dataset_spec = the profile's dataset ranges (keep per-camera overlaps and headlands in range)."""
    cams = cameras_from_params(p)
    n_cam, n_pass, n_frames = p["cameras"], p["passes"], p["frames"]

    # ids, time, standstill placement
    alnum = np.array(list(string.ascii_letters + string.digits))
    flight_id = "".join(rng.choice(alnum, size=12))
    camera_ids: list[str] = []
    while len(camera_ids) < n_cam:
        cid = f"{int(rng.integers(0, 2**48)):012X}"
        if cid not in camera_ids:
            camera_ids.append(cid)
    start_frame_idx = int(rng.integers(0, 100_000))
    start_ms = int((_START_MIN - _EPOCH).total_seconds() * 1000) \
        + int(rng.integers(0, 6 * 365)) * 86_400_000 + int(rng.integers(6 * 3_600_000, 18 * 3_600_000))
    n_still = int(round(p["standstill_share"] * n_frames))
    still_pass = int(rng.integers(0, n_pass))
    still_frac = float(rng.uniform(0, 1))

    rig = solve_rig(p, cams, dataset_spec)
    spacing = rig["rig_width_m"] * p["field_width_factor"]
    width = n_pass * spacing
    c_pass = [(k - (n_pass - 1) / 2) * spacing for k in range(n_pass)]

    # progress along each pass (frame 0 at 0), standstill block in one pass
    positions, kinds = [], []
    for k in range(n_pass):
        steps = np.full(n_frames + 2, rig["step_normal_m"])
        steps[0] = 0.0
        kind = np.array(["normal"] * (n_frames + 2), dtype=object)
        kind[0] = "start"
        if k == still_pass and n_still > 0:
            i0 = 1 + int(still_frac * max(n_frames // n_pass - n_still - 1, 0))
            steps[i0:i0 + n_still] = rig["step_standstill_m"]
            kind[i0:i0 + n_still] = "standstill"
        positions.append(np.cumsum(steps))
        kinds.append(kind)

    def pass_layout(length: float, scale: float):
        corners = _field_corners(length, width, p)
        out = []
        for k in range(n_pass):
            bottom = _edge_s(c_pass[k], corners[0], corners[1])
            top = _edge_s(c_pass[k], corners[3], corners[2])
            hs, he = p["headland_start_m"][k] * scale, p["headland_end_m"][k] * scale
            extent = top - bottom
            pos = positions[k]
            # last frame: last one <= end headland past the field, but always outside the field
            n = max(int(np.searchsorted(pos, hs + extent + he, "right")),
                    int(np.searchsorted(pos, hs + extent, "right")) + 1)
            out.append({"bottom": bottom, "top": top, "hs": hs, "extent": extent, "n": n})
        return corners, out

    def total(length: float, scale: float) -> int:
        return sum(x["n"] for x in pass_layout(length, scale)[1])

    scale, length = 1.0, _FIELD_LENGTH_MIN_M
    if total(length, 1.0) >= n_frames:
        scale = _smallest(lambda s: total(length, s), 0.0, 1.0, n_frames) if total(length, 1.0) > n_frames else 1.0
    else:
        hi = length
        while total(hi, 1.0) < n_frames:
            hi *= 2
        length = _smallest(lambda L: total(L, 1.0), length, hi, n_frames)

    # A long along-track step (large footprint) can put the first frame past the field more than the
    # headland maximum out. Lengthen the field, keeping every pass's frame count, until all end headlands fit.
    he_spec = dataset_spec["headland_end_m"]
    if scale == 1.0 and isinstance(he_spec, dict):
        lay = pass_layout(length, 1.0)[1]
        needed = max((positions[k][x["n"] - 1] - x["hs"] - he_spec["max"]) * length / x["extent"]
                     for k, x in enumerate(lay))
        if needed > length:
            cand = needed * (1 + 1e-9)
            lay2 = pass_layout(cand, 1.0)[1]
            if [x["n"] for x in lay2] == [x["n"] for x in lay] and \
                    all(positions[k][x["n"] - 1] > x["hs"] + x["extent"] for k, x in enumerate(lay2)):
                length = cand
    corners, layout = pass_layout(length, scale)

    # world frame
    epsg = utm_epsg(p["center_lon_deg"], p["center_lat_deg"])
    e0, n0 = wgs84_to_utm(p["center_lon_deg"], p["center_lat_deg"], epsg)
    a_vec, r_vec = heading_axes(p["heading_deg"])
    origin = np.array([e0, n0])
    field = Polygon([origin + s * a_vec + c * r_vec for s, c in corners])
    assert field.is_valid and len(field.interiors) == 0

    interval_ms = int(round(p["frame_interval_s"] * 1000))
    speed = rig["step_normal_m"] / p["frame_interval_s"]
    sig_pos, sig_alt, sig_ang = p["pose_noise_pos_sigma_m"], p["pose_noise_alt_sigma_m"], p["pose_noise_angle_sigma_deg"]

    rows, pass_info = [], []
    frame_idx, t_ms = start_frame_idx, start_ms
    for k, lay in enumerate(layout):
        forward = k % 2 == 0
        heading = p["heading_deg"] if forward else (p["heading_deg"] + 180.0) % 360.0
        a_k, r_k = heading_axes(heading)
        pos = positions[k][:lay["n"]]
        end_headland = pos[-1] - lay["hs"] - lay["extent"]
        if isinstance(he_spec, dict) and end_headland > he_spec["max"]:
            # still too far out (field could not be lengthened for every pass): start this pass later
            shift = min(end_headland - he_spec["max"], lay["hs"])
            lay = {**lay, "hs": lay["hs"] - shift}
            end_headland -= shift
        pass_info.append({"frames": lay["n"], "start_headland_m": lay["hs"], "end_headland_m": float(end_headland)})
        if k > 0:
            prev = pass_info[k - 1]
            turn_m = math.pi * spacing / 2 + prev["end_headland_m"] + lay["hs"]
            t_ms += max(interval_ms, int(round(1000 * turn_m / speed)))
        for f, prog in enumerate(pos):
            if f > 0:
                t_ms += interval_ms
            s = lay["bottom"] - lay["hs"] + prog if forward else lay["top"] + lay["hs"] - prog
            vehicle = origin + s * a_vec + c_pass[k] * r_vec
            for i in range(n_cam):
                e, n = vehicle + rig["mount_along_m"][i] * a_k + rig["mount_across_m"][i] * r_k
                yaw = (heading + p["yaw_dev_deg"][i]) % 360.0
                noise = rng.normal(0.0, 1.0, size=6) * [sig_pos, sig_pos, sig_alt, sig_ang, sig_ang, sig_ang]
                rows.append({
                    "frame_idx": frame_idx, "time_ms": t_ms, "camera_id": camera_ids[i], "cam_index": i,
                    "pass_idx": k, "kind": kinds[k][f], "heading": heading,
                    "east": e, "north": n, "alt_agl": p["alt_agl_m"],
                    "yaw": yaw, "pitch": p["pitch_deg"][i], "roll": p["roll_deg"][i],
                    "east_meas": e + noise[0], "north_meas": n + noise[1], "alt_agl_meas": p["alt_agl_m"] + noise[2],
                    "yaw_meas": (yaw + noise[3]) % 360.0, "pitch_meas": p["pitch_deg"][i] + noise[4],
                    "roll_meas": p["roll_deg"][i] + noise[5],
                })
            frame_idx += 1

    images = pd.DataFrame(rows).sort_values(["frame_idx", "camera_id"], ignore_index=True)
    stamps = images["time_ms"].map(ms_to_datetime)
    images["file_name"] = [
        f"{fi}_c_{dt.strftime('%Y%m%d%H%M%S')}{ms % 1000:03d}_t_{cid}_cam"
        for fi, dt, ms, cid in zip(images["frame_idx"], stamps, images["time_ms"], images["camera_id"])
    ]
    images["timestamp"] = [f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{ms % 1000:03d}Z"
                           for dt, ms in zip(stamps, images["time_ms"])]
    images["flight_id"] = flight_id

    drawn = {
        "flight_id": flight_id, "camera_ids": camera_ids, "start_frame_idx": start_frame_idx,
        "start_time": images["timestamp"].iloc[0], "standstill_pass": still_pass, "standstill_start_frac": still_frac,
    }
    derived = {
        "epsg": epsg, "field_length_m": length, "field_width_m": width, "pass_spacing_m": spacing,
        "headland_scale": scale, "standstill_frames": int((images["kind"] == "standstill").sum() // n_cam),
        "passes": pass_info, **rig,
    }
    return Scenario(epsg, cams, camera_ids, field, images, drawn, derived)
