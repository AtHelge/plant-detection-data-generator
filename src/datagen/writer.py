"""File output: positions.csv, camsettings.yaml, field boundary shapefile, detection CSVs."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Polygon

from .geometry import DISTORTION_KEYS, utm_to_wgs84
from .scenario import Scenario

POSE_COLUMNS = ["file_name", "flight_id", "frame_idx", "timestamp", "camera_id",
                "lat", "lon", "alt_agl", "yaw", "pitch", "roll"]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """UTF-8, LF, no index; float columns (detection conf) with 3 decimals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.3f")


def poses_table(sc: Scenario) -> pd.DataFrame:
    """positions.csv table with the reported (noisy) poses."""
    img = sc.images
    lon, lat = utm_to_wgs84(img["east_meas"].to_numpy(), img["north_meas"].to_numpy(), sc.epsg)
    yaw = np.round(img["yaw_meas"].to_numpy() % 360.0, 4) % 360.0
    return pd.DataFrame({
        "file_name": img["file_name"],
        "flight_id": img["flight_id"],
        "frame_idx": img["frame_idx"],
        "timestamp": img["timestamp"],
        "camera_id": img["camera_id"],
        "lat": [f"{v:.9f}" for v in lat],
        "lon": [f"{v:.9f}" for v in lon],
        "alt_agl": [f"{v:.3f}" for v in img["alt_agl_meas"]],
        "yaw": [f"{v:.4f}" for v in yaw],
        "pitch": [f"{v:.4f}" for v in img["pitch_meas"]],
        "roll": [f"{v:.4f}" for v in img["roll_meas"]],
    })[POSE_COLUMNS]


def write_camsettings_yaml(sc: Scenario, path: Path) -> None:
    doc = {
        "angle_convention": "ENU_YPR_V1",
        "cameras": [
            {"camera_id": cid, "model": "synthetic", "width": c.width, "height": c.height,
             "fx": float(c.fx), "fy": float(c.fy), "cx": float(c.cx), "cy": float(c.cy),
             "distortion": {k: float(getattr(c, k)) for k in DISTORTION_KEYS}}
            for cid, c in zip(sc.camera_ids, sc.cameras)
        ],
    }
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def write_field(sc: Scenario, path: Path) -> None:
    """Field polygon in WGS84, one feature with FID=0."""
    lon, lat = utm_to_wgs84(*np.asarray(sc.field.exterior.coords).T, sc.epsg)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame({"FID": [0]}, geometry=[Polygon(np.column_stack([lon, lat]))], crs="EPSG:4326")
    gdf.to_file(path, driver="ESRI Shapefile", engine="pyogrio", encoding="UTF-8")


def write_detections(tables: dict, out: Path) -> None:
    for file_name, det in tables.items():
        _write_csv(det, out / "crop_canola" / f"{file_name}.csv")


def write_scenario(sc: Scenario, out: Path) -> None:
    _write_csv(poses_table(sc), out / "positions.csv")
    write_camsettings_yaml(sc, out / "camsettings.yaml")
    write_field(sc, out / "field_boundary" / "field_boundary.shp")
