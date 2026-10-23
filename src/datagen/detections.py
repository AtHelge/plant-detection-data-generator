"""Plants inside the field and per-image detections (from the TRUE poses)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from .geometry import Pose, full_footprint, ground_to_pixel, heading_axes
from .scenario import Scenario

DETECTION_COLUMNS = ["x_center", "y_center", "width", "height", "conf", "class", "pixel_count"]
CLASS_NAME = "Canola"
PLANT_JITTER_ALONG_FRAC = 0.15   # sigma, fraction of the in-row plant spacing
PLANT_JITTER_ACROSS_M = 0.02     # sigma


def generate_plants(sc: Scenario, p: dict, obj: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Plants in rows parallel to the heading, only inside the field. Sorted by along coordinate s."""
    a, r = heading_axes(p["heading_deg"])
    origin = np.array([sc.field.centroid.x, sc.field.centroid.y])
    xy = np.asarray(sc.field.exterior.coords) - origin
    s_min, s_max = (xy @ a).min(), (xy @ a).max()
    c_min, c_max = (xy @ r).min(), (xy @ r).max()

    row_spacing = p["row_spacing_m"]
    in_row = 1.0 / (p["plant_density_per_m2"] * row_spacing)
    c_rows = np.arange(c_min + rng.uniform(0, row_spacing), c_max, row_spacing)
    s_grid = np.arange(s_min + rng.uniform(0, in_row), s_max, in_row)
    s, c = (m.ravel() for m in np.meshgrid(s_grid, c_rows))
    s = s + rng.normal(0.0, PLANT_JITTER_ALONG_FRAC * in_row, s.size)
    c = c + rng.normal(0.0, PLANT_JITTER_ACROSS_M, c.size)

    east = origin[0] + s * a[0] + c * r[0]
    north = origin[1] + s * a[1] + c * r[1]
    inside = shapely.contains_xy(sc.field, east, north)
    order = np.argsort(s[inside], kind="stable")
    plants = pd.DataFrame({"s": s[inside][order], "east": east[inside][order], "north": north[inside][order]})
    d = obj["plant_diameter_m"]
    plants["diameter_m"] = rng.uniform(d["min"], d["max"], len(plants))
    return plants


def _boxes(diameter_m, gsd_m, obj, rng):
    """Box sides in px: diameter / GSD shaped by the aspect ratio, clipped to the ground-equivalent range."""
    asp = obj["box_aspect"]
    lo, hi = obj["box_size_m"]["min"] / gsd_m, obj["box_size_m"]["max"] / gsd_m
    aspect = rng.uniform(asp["min"], asp["max"], len(diameter_m))
    size = np.asarray(diameter_m) / gsd_m
    w = np.rint(np.clip(size * np.sqrt(aspect), lo, hi)).astype(int)
    h = np.rint(np.clip(size / np.sqrt(aspect), lo, hi)).astype(int)
    return w, h


def generate_detections(sc: Scenario, p: dict, image_spec: dict, obj: dict, plants: pd.DataFrame,
                        rng: np.random.Generator):
    """Returns ({file_name: detection table}, number of false positives).

    Each image uses its own camera (sc.cameras[cam_index]); GSD of a camera = alt_agl / fx.
    """
    sigma = p["center_noise_sigma_px"]
    a, _ = heading_axes(p["heading_deg"])
    origin = np.array([sc.field.centroid.x, sc.field.centroid.y])
    plant_s = plants["s"].to_numpy()
    rec, fpf = image_spec["recall"], image_spec["false_positive_frac"]
    ct, cf = obj["conf_true"], obj["conf_false"]
    dmin, dmax = obj["plant_diameter_m"]["min"], obj["plant_diameter_m"]["max"]

    tables, false_positives = {}, 0
    for img in sc.images.itertuples():
        cam = sc.cameras[img.cam_index]
        W, H = cam.width, cam.height
        gsd_m = img.alt_agl / cam.fx
        pose = Pose(img.east, img.north, img.alt_agl, img.yaw, img.pitch, img.roll)
        fp = full_footprint(pose, cam)
        fs = (np.asarray(fp.exterior.coords) - origin) @ a
        i0, i1 = np.searchsorted(plant_s, [fs.min(), fs.max()])
        cand = plants.iloc[i0:i1]
        cand = cand[shapely.contains_xy(fp, cand["east"].to_numpy(), cand["north"].to_numpy())]

        recall = rng.uniform(rec["min"], rec["max"])
        fp_frac = rng.uniform(fpf["min"], fpf["max"])
        if len(cand):
            u_true, v_true = ground_to_pixel(cand["east"].to_numpy(), cand["north"].to_numpy(), pose, cam)
        else:
            u_true = v_true = np.empty(0)
        u = u_true + rng.normal(0.0, 1.0, len(cand)) * sigma
        v = v_true + rng.normal(0.0, 1.0, len(cand)) * sigma
        detected = rng.random(len(cand)) < recall
        u, v = np.rint(u), np.rint(v)
        keep = detected & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        n_tp = int(keep.sum())
        n_fp = int(round(fp_frac / (1.0 - fp_frac) * n_tp))

        tw, th = _boxes(cand["diameter_m"].to_numpy()[keep], gsd_m, obj, rng)
        t_conf = ct["min"] + (ct["max"] - ct["min"]) * rng.beta(ct["a"], ct["b"], n_tp)
        f_u, f_v = rng.integers(0, W, n_fp), rng.integers(0, H, n_fp)
        fw, fh = _boxes(rng.uniform(dmin, dmax, n_fp), gsd_m, obj, rng)
        f_conf = rng.uniform(cf["min"], cf["max"], n_fp)

        x = np.concatenate([u[keep], f_u]).astype(int)
        y = np.concatenate([v[keep], f_v]).astype(int)
        w, h = np.concatenate([tw, fw]), np.concatenate([th, fh])
        det = pd.DataFrame({
            "x_center": x, "y_center": y, "width": w, "height": h,
            "conf": np.round(np.concatenate([t_conf, f_conf]), 3),
            "class": CLASS_NAME, "pixel_count": w * h,
        })
        tables[img.file_name] = det.sort_values(["conf", "x_center", "y_center"], ascending=[False, True, True],
                                                kind="mergesort", ignore_index=True)
        false_positives += n_fp

    return tables, false_positives
