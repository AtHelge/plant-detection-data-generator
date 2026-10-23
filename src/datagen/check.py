"""Self-check of a generated dataset folder, using only the files in it."""

from __future__ import annotations

import io
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from .detections import CLASS_NAME, DETECTION_COLUMNS
from .geometry import DISTORTION_KEYS
from .writer import POSE_COLUMNS

FILE_NAME_RE = r"\d+_c_\d{17}_t_[0-9A-F]{12}_cam"
REQUIRED = [
    "crop_canola", "positions.csv", "camsettings.yaml", "manifest.json",
    *(f"field_boundary/field_boundary.{ext}" for ext in ("shp", "shx", "dbf", "prj", "cpg")),
]
CAMERA_KEYS = {"camera_id", "model", "width", "height", "fx", "fy", "cx", "cy", "distortion"}


class SelfCheckError(Exception):
    pass


def _fail(problems: list[str]) -> SelfCheckError:
    return SelfCheckError("self-check failed:\n  " + "\n  ".join(problems))


def _read_text_csv(path: Path, columns: list[str], problems: list[str]) -> pd.DataFrame | None:
    raw = path.read_bytes()
    if b"\r" in raw:
        problems.append(f"{path.name}: line endings must be LF")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        problems.append(f"{path.name}: not UTF-8")
        return None
    if text.split("\n", 1)[0] != ",".join(columns):
        problems.append(f"{path.name}: header must be {','.join(columns)!r}")
        return None
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def _bad(mask: pd.Series, df: pd.DataFrame, what: str, problems: list[str]) -> None:
    """Record a problem for the rows in mask, with up to three example file names."""
    if mask.any():
        problems.append(f"{what}: {int(mask.sum())} row(s), e.g. {df.loc[mask, 'file_name'].head(3).tolist()}")


def _check_poses(poses: pd.DataFrame, problems: list[str]) -> None:
    n0 = len(problems)

    def fmt(col, pattern):
        _bad(~poses[col].str.fullmatch(pattern), poses, f"positions.csv {col} format", problems)

    fmt("file_name", FILE_NAME_RE)
    fmt("flight_id", r"[A-Za-z0-9_]{1,32}")
    fmt("frame_idx", r"0|[1-9]\d*")
    fmt("timestamp", r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z")
    fmt("camera_id", r"[0-9A-F]{12}")
    fmt("lat", r"-?\d+\.\d{9}")
    fmt("lon", r"-?\d+\.\d{9}")
    fmt("alt_agl", r"-?\d+\.\d{3}")
    for col in ("yaw", "pitch", "roll"):
        fmt(col, r"-?\d+\.\d{4}")
    if len(problems) > n0:
        return

    num = poses[["frame_idx", "lat", "lon", "alt_agl", "yaw", "pitch", "roll"]].astype(float)
    _bad(~num["lat"].between(-90, 90) | ~num["lon"].between(-180, 180), poses, "positions.csv lat/lon range", problems)
    _bad(num["alt_agl"] <= 0, poses, "positions.csv alt_agl must be > 0", problems)
    _bad(~((num["yaw"] >= 0) & (num["yaw"] < 360)), poses, "positions.csv yaw must be in [0, 360)", problems)
    _bad(~num["pitch"].between(-180, 180) | ~num["roll"].between(-180, 180), poses,
         "positions.csv pitch/roll range", problems)

    expected = poses["frame_idx"] + "_c_" + poses["timestamp"].str.replace(r"\D", "", regex=True) \
        + "_t_" + poses["camera_id"] + "_cam"
    _bad(poses["file_name"] != expected, poses, "positions.csv file_name does not match frame_idx/timestamp/camera_id",
         problems)
    if poses["file_name"].duplicated().any():
        problems.append("positions.csv: duplicate file_name")
    if poses["flight_id"].nunique() != 1:
        problems.append("positions.csv: flight_id must be the same for the whole dataset")
    if poses.groupby("frame_idx")["timestamp"].nunique().gt(1).any():
        problems.append("positions.csv: cameras of one frame have different timestamps")
    keys = list(zip(num["frame_idx"].astype(int), poses["camera_id"]))
    if keys != sorted(keys):
        problems.append("positions.csv: not sorted by frame_idx, camera_id")


def _check_detections(out: Path, poses: pd.DataFrame, cams: dict, problems: list[str]) -> None:
    det_dir = out / "crop_canola"
    files = {f.name for f in det_dir.iterdir()}
    expected = {f"{n}.csv" for n in poses["file_name"]}
    if files - expected:
        problems.append(f"crop_canola: files without pose: {sorted(files - expected)[:3]}")
    if expected - files:
        problems.append(f"crop_canola: poses without CSV: {sorted(expected - files)[:3]}")

    n0 = len(problems)
    parts = []
    for name in sorted(files & expected):
        df = _read_text_csv(det_dir / name, DETECTION_COLUMNS, problems)
        if df is not None and len(df):
            parts.append(df.assign(file_name=name[:-4]))
    if len(problems) > n0 or not parts:
        return
    det = pd.concat(parts, ignore_index=True)

    for col in ("x_center", "y_center", "width", "height", "pixel_count"):
        _bad(~det[col].str.fullmatch(r"0|[1-9]\d*"), det, f"detections {col} must be a non-negative int", problems)
    _bad(~det["conf"].str.fullmatch(r"\d\.\d{3}"), det, "detections conf must have 3 decimals", problems)
    _bad(det["class"] != CLASS_NAME, det, f"detections class must be {CLASS_NAME!r}", problems)
    if len(problems) > n0:
        return

    n = det[["x_center", "y_center", "width", "height", "pixel_count"]].astype(int)
    conf = det["conf"].astype(float)
    cam_of = poses.set_index("file_name")["camera_id"]
    wh = det["file_name"].map(cam_of).map(lambda c: (cams[c]["width"], cams[c]["height"]))
    w_img, h_img = wh.str[0], wh.str[1]
    _bad(~((n["x_center"] < w_img) & (n["y_center"] < h_img)), det, "detections centre outside image", problems)
    _bad(n["pixel_count"] != n["width"] * n["height"], det, "detections pixel_count != width*height", problems)
    _bad((n["width"] <= 0) | (n["height"] <= 0), det, "detections width/height must be > 0", problems)
    _bad(~conf.between(0, 1), det, "detections conf must be in [0, 1]", problems)

    same = det["file_name"].eq(det["file_name"].shift(-1))
    c0, c1 = -conf, -conf.shift(-1)
    x0, x1, y0, y1 = n["x_center"], n["x_center"].shift(-1), n["y_center"], n["y_center"].shift(-1)
    ordered = (c0 < c1) | ((c0 == c1) & ((x0 < x1) | ((x0 == x1) & (y0 <= y1))))
    _bad(same & ~ordered, det, "detections not sorted by conf desc, x_center, y_center", problems)


def _check_field(out: Path, problems: list[str]) -> None:
    base = out / "field_boundary" / "field_boundary"
    if base.with_suffix(".cpg").read_text(encoding="utf-8").strip().upper() not in ("UTF-8", "UTF8"):
        problems.append("field_boundary.cpg must be UTF-8")
    gdf = gpd.read_file(base.with_suffix(".shp"))
    if len(gdf) != 1:
        problems.append(f"field_boundary.shp must have exactly one feature, has {len(gdf)}")
        return
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        problems.append(f"field_boundary.shp CRS must be EPSG:4326, is {gdf.crs}")
    geom = gdf.geometry.iloc[0]
    if geom.geom_type != "Polygon" or not geom.is_valid or len(geom.interiors) or geom.is_empty:
        problems.append("field_boundary.shp geometry must be one valid polygon without holes")
    if "FID" not in gdf.columns or int(gdf["FID"].iloc[0]) != 0:
        problems.append("field_boundary.shp attribute FID must be 0")


def self_check(out: Path) -> None:
    """Raise SelfCheckError listing every problem found."""
    out = Path(out)
    problems = [f"missing {p}" for p in REQUIRED if not (out / p).exists()]
    if problems:
        raise _fail(problems)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for key in ("generator_version", "seed", "profile", "ranges_sha256", "drawn", "counts"):
        if key not in manifest:
            problems.append(f"manifest.json: missing {key}")

    cam_doc = yaml.safe_load((out / "camsettings.yaml").read_text(encoding="utf-8"))
    if cam_doc.get("angle_convention") != "ENU_YPR_V1":
        problems.append("camsettings.yaml: angle_convention must be ENU_YPR_V1")
    cams = {}
    for c in cam_doc.get("cameras", []):
        if set(c) != CAMERA_KEYS or set(c.get("distortion", {})) != set(DISTORTION_KEYS):
            problems.append(f"camsettings.yaml: camera {c.get('camera_id')} has wrong keys")
            continue
        if not (c["width"] > 0 and c["height"] > 0 and c["fx"] > 0 and c["fy"] > 0):
            problems.append(f"camsettings.yaml: camera {c['camera_id']} has invalid intrinsics")
        cams[c["camera_id"]] = c

    poses = _read_text_csv(out / "positions.csv", POSE_COLUMNS, problems)
    if poses is not None:
        n0 = len(problems)
        _check_poses(poses, problems)
        missing = set(poses["camera_id"]) - set(cams)
        if missing:
            problems.append(f"camsettings.yaml: camera_ids from positions.csv missing: {sorted(missing)}")
        if len(problems) == n0:
            _check_detections(out, poses, cams, problems)
            if manifest.get("counts", {}).get("images") != len(poses):
                problems.append("manifest.json counts.images != rows in positions.csv")
    _check_field(out, problems)
    if problems:
        raise _fail(problems)
