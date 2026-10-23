"""Camera geometry: UTM helpers, rotation, pixel <-> ground projection on the plane z=0.

World frame: UTM easting/northing (ENU), up = z, ground at z=0.
Camera frame: x right, y down, z viewing direction. Pixels: (0,0) top-left, x right, y down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon

DISTORTION_KEYS = ("k1", "k2", "k3", "p1", "p2")
FOOTPRINT_SEGMENTS = 16          # points per image edge in full_footprint
_UNDISTORT_MAX_ITER = 200
_UNDISTORT_TOL = 1e-15
_LENS_TOL_PX = 1e-6


# --- UTM -------------------------------------------------------------------

def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat), WGS84."""
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=None)
def _transformer(src: int, dst: int) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


def wgs84_to_utm(lon, lat, epsg: int):
    """(lon, lat) degrees -> (E, N) metres."""
    return _transformer(4326, epsg).transform(lon, lat)


def utm_to_wgs84(east, north, epsg: int):
    """(E, N) metres -> (lon, lat) degrees."""
    return _transformer(epsg, 4326).transform(east, north)


# --- camera and pose ---------------------------------------------------------

@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


def camera_from_dict(c: dict) -> Camera:
    """Camera from a dict with width, height, fx, fy, cx, cy and distortion {k1, k2, k3, p1, p2}."""
    return Camera(c["width"], c["height"], c["fx"], c["fy"], c["cx"], c["cy"], **c["distortion"])


@dataclass(frozen=True)
class Pose:
    """Pose of one camera: position in UTM, altitude above ground, angles in degrees."""
    east: float
    north: float
    alt_agl: float
    yaw: float
    pitch: float
    roll: float


R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_wc(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """R_wc = Rz(-yaw) · Rx(pitch) · Ry(roll) · R0, angles in degrees."""
    y, p, r = np.radians([yaw, pitch, roll])
    return _rz(-y) @ _rx(p) @ _ry(r) @ R0


def heading_axes(heading_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (E, N) along the heading and to its right."""
    h = math.radians(heading_deg)
    return np.array([math.sin(h), math.cos(h)]), np.array([math.cos(h), -math.sin(h)])


# --- distortion (OpenCV k1, k2, k3, p1, p2) ---------------------------------

def distort(x, y, cam: Camera):
    """Normalized undistorted -> normalized distorted coordinates."""
    r2 = x * x + y * y
    radial = 1.0 + cam.k1 * r2 + cam.k2 * r2**2 + cam.k3 * r2**3
    xd = x * radial + 2.0 * cam.p1 * x * y + cam.p2 * (r2 + 2.0 * x * x)
    yd = y * radial + cam.p1 * (r2 + 2.0 * y * y) + 2.0 * cam.p2 * x * y
    return xd, yd


def undistort(xd, yd, cam: Camera):
    """Normalized distorted -> undistorted, by fixed-point iteration."""
    xd = np.asarray(xd, dtype=float)
    yd = np.asarray(yd, dtype=float)
    if not any((cam.k1, cam.k2, cam.k3, cam.p1, cam.p2)):
        return xd, yd
    x, y = xd.copy(), yd.copy()
    for _ in range(_UNDISTORT_MAX_ITER):
        r2 = x * x + y * y
        radial = 1.0 + cam.k1 * r2 + cam.k2 * r2**2 + cam.k3 * r2**3
        dx = 2.0 * cam.p1 * x * y + cam.p2 * (r2 + 2.0 * x * x)
        dy = cam.p1 * (r2 + 2.0 * y * y) + 2.0 * cam.p2 * x * y
        x_new = (xd - dx) / radial
        y_new = (yd - dy) / radial
        step = max(np.max(np.abs(x_new - x), initial=0.0), np.max(np.abs(y_new - y), initial=0.0))
        x, y = x_new, y_new
        if step < _UNDISTORT_TOL:
            break
    return x, y


def distortion_is_valid(cam: Camera) -> bool:
    """True if the lens maps the whole image one-to-one: the radial mapping does not fold before the
    farthest image corner (with 5 % margin) and pixel -> undistort -> distort returns to the pixel."""
    xs = (np.array([0.0, cam.width]) - cam.cx) / cam.fx
    ys = (np.array([0.0, cam.height]) - cam.cy) / cam.fy
    r_corner = math.sqrt(np.max(np.abs(xs)) ** 2 + np.max(np.abs(ys)) ** 2)
    r = np.linspace(0.0, 4.0 * r_corner, 4001)
    r2 = r * r
    f = r * (1.0 + cam.k1 * r2 + cam.k2 * r2**2 + cam.k3 * r2**3)
    df = 1.0 + 3.0 * cam.k1 * r2 + 5.0 * cam.k2 * r2**2 + 7.0 * cam.k3 * r2**3
    fold = np.flatnonzero(df <= 0.0)
    reach = f[: fold[0]] if len(fold) else f
    if reach.max() < 1.05 * r_corner:
        return False

    s = np.linspace(0.0, 1.0, 33)
    u = np.concatenate([s * cam.width, np.full_like(s, cam.width), s * cam.width, np.zeros_like(s),
                        np.repeat(s, 9) * cam.width])
    v = np.concatenate([np.zeros_like(s), s * cam.height, np.full_like(s, cam.height), s * cam.height,
                        np.tile(np.linspace(0.0, 1.0, 9), len(s)) * cam.height])
    x, y = undistort((u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, cam)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return False
    xd, yd = distort(x, y, cam)
    err = np.hypot(cam.fx * xd + cam.cx - u, cam.fy * yd + cam.cy - v)
    return bool(np.max(err) < _LENS_TOL_PX)


# --- projection ------------------------------------------------------------------

def pixel_to_ground(u, v, pose: Pose, cam: Camera):
    """Pixel(s) -> ground point(s) (E, N) on z=0. Raises ValueError if a ray misses the ground."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    x, y = undistort((u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, cam)
    d_c = np.stack([x, y, np.ones_like(x)], axis=-1)
    d_w = d_c @ rotation_wc(pose.yaw, pose.pitch, pose.roll).T
    dz = d_w[..., 2]
    if np.any(dz >= 0):
        n = int(np.count_nonzero(dz >= 0))
        raise ValueError(f"{n} pixel ray(s) do not hit the ground (d_w.z >= 0); pose {pose}")
    t = -pose.alt_agl / dz
    return pose.east + t * d_w[..., 0], pose.north + t * d_w[..., 1]


def ground_to_pixel(east, north, pose: Pose, cam: Camera):
    """Ground point(s) (E, N, z=0) -> pixel(s) (u, v). Raises ValueError for points behind the camera."""
    east = np.asarray(east, dtype=float)
    north = np.asarray(north, dtype=float)
    p_w = np.stack([east - pose.east, north - pose.north, np.full_like(east, -pose.alt_agl)], axis=-1)
    p_c = p_w @ rotation_wc(pose.yaw, pose.pitch, pose.roll)  # = (R_wc^T · p_w) per point
    z = p_c[..., 2]
    if np.any(z <= 0):
        n = int(np.count_nonzero(z <= 0))
        raise ValueError(f"{n} ground point(s) lie behind the camera; pose {pose}")
    xd, yd = distort(p_c[..., 0] / z, p_c[..., 1] / z, cam)
    return cam.fx * xd + cam.cx, cam.fy * yd + cam.cy


def full_footprint(pose: Pose, cam: Camera) -> Polygon:
    """Image border projected to the ground (FOOTPRINT_SEGMENTS points per edge), UTM polygon."""
    n = FOOTPRINT_SEGMENTS
    s = np.arange(n) / n
    w, h = cam.width, cam.height
    u = np.concatenate([s * w, np.full(n, w), (1 - s) * w, np.zeros(n)])
    v = np.concatenate([np.zeros(n), s * h, np.full(n, h), (1 - s) * h])
    e, n = pixel_to_ground(u, v, pose, cam)
    return Polygon(np.column_stack([e, n]))


def image_center_ground(pose: Pose, cam: Camera) -> tuple[float, float]:
    """Ground point of the principal point (cx, cy)."""
    e, n = pixel_to_ground(cam.cx, cam.cy, pose, cam)
    return float(e), float(n)
