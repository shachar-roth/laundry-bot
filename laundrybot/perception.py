"""Everything the overhead camera tells the director.

1. CameraModel   - where the camera is, found from ArUco markers taped to the
                   floor at known spots (no manual measuring of the camera).
2. RobotTracker  - each robot's position + heading from the marker on its top.
3. ClothDetector - the towel's outline and corners (colour segmentation).

All of this is classic OpenCV, no machine learning yet. It works because the
sim towel is a solid colour on a contrasting floor. Real laundry on a real bed
will need a learned segmenter here; the rest of the pipeline stays the same.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import RobotGeometry

ARUCO_DICT = cv2.aruco.DICT_4X4_50


def marker_corners_local(size: float) -> np.ndarray:
    """ArUco corner order (TL, TR, BR, BL) in the marker's own frame.

    The marker's +x points from its left edge to its right edge, +y from its
    bottom edge to its top edge (as printed).
    """
    h = size / 2
    return np.array([[-h, h], [h, h], [h, -h], [-h, -h]], dtype=float)


def marker_corners_world(cx: float, cy: float, size: float, yaw: float = 0.0, z: float = 0.0) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    xy = marker_corners_local(size) @ R.T + [cx, cy]
    return np.column_stack([xy, np.full(4, z)])


_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
    (lambda p: (setattr(p, "cornerRefinementMethod", cv2.aruco.CORNER_REFINE_SUBPIX), p)[1])(
        cv2.aruco.DetectorParameters()
    ),
)


def detect_markers(rgb: np.ndarray) -> dict[int, np.ndarray]:
    """{marker id: 4x2 pixel corners (TL, TR, BR, BL)}."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


class CameraModel:
    """Pinhole camera + its pose relative to the floor."""

    def __init__(self, K: np.ndarray):
        self.K = np.asarray(K, dtype=float)
        self.Kinv = np.linalg.inv(self.K)
        self.R: np.ndarray | None = None  # world -> camera rotation
        self.t: np.ndarray | None = None
        self.reprojection_error_px: float | None = None

    @property
    def calibrated(self) -> bool:
        return self.R is not None

    def calibrate(self, detections: dict[int, np.ndarray], floor_markers: dict[int, tuple[float, float, float]]) -> bool:
        """Solve camera pose from floor markers {id: (x, y, size)} seen in the image."""
        obj, img = [], []
        for mid, (x, y, size) in floor_markers.items():
            if mid in detections:
                obj.append(marker_corners_world(x, y, size))
                img.append(detections[mid])
        if len(obj) < 2:
            return False
        obj = np.concatenate(obj).astype(np.float64)
        img = np.concatenate(img).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return False
        self.R, _ = cv2.Rodrigues(rvec)
        self.t = tvec.reshape(3)
        proj = self.world_to_pixel(obj)
        self.reprojection_error_px = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))
        return True

    def pixel_to_world(self, uv: np.ndarray, z: float = 0.0) -> np.ndarray:
        """Intersect the camera rays through pixels uv (Nx2) with the plane at height z."""
        uv = np.atleast_2d(uv).astype(float)
        rays_cam = np.column_stack([uv, np.ones(len(uv))]) @ self.Kinv.T
        rays = rays_cam @ self.R  # == (R^T @ ray)^T, camera -> world
        origin = -self.R.T @ self.t
        s = (z - origin[2]) / rays[:, 2]
        return origin + rays * s[:, None]

    def world_to_pixel(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts).astype(float)
        if pts.shape[1] == 2:
            pts = np.column_stack([pts, np.zeros(len(pts))])
        cam = pts @ self.R.T + self.t
        uv = cam @ self.K.T
        return uv[:, :2] / uv[:, 2:3]


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def heading(self) -> np.ndarray:
        return np.array([math.cos(self.yaw), math.sin(self.yaw)])


class RobotTracker:
    """Robot poses from the markers on their tops."""

    def __init__(self, camera: CameraModel, geometry: RobotGeometry, robot_ids: list[int]):
        self.cam = camera
        self.g = geometry
        self.poses: dict[int, Pose2D] = {}
        self.last_seen: dict[int, float] = {i: -math.inf for i in robot_ids}

    def update(self, detections: dict[int, np.ndarray], t: float) -> None:
        for rid in self.last_seen:
            if rid not in detections:
                continue
            c = self.cam.pixel_to_world(detections[rid], z=self.g.marker_z)[:, :2]
            tl, tr, br, bl = c
            right = (tr - tl) + (br - bl)  # marker +x == robot forward
            yaw = math.atan2(right[1], right[0])
            centre = c.mean(axis=0)
            base = centre - self.g.marker_offset_x * np.array([math.cos(yaw), math.sin(yaw)])
            self.poses[rid] = Pose2D(base[0], base[1], yaw)
            self.last_seen[rid] = t

    def age(self, rid: int, t: float) -> float:
        return t - self.last_seen[rid]


@dataclass
class ClothObservation:
    corners: np.ndarray  # 4x2 world, metres (actual cloth corner points)
    outline: np.ndarray  # Nx2 world
    area: float  # m^2
    centre: np.ndarray  # 2
    u: np.ndarray  # unit vector along the long side
    v: np.ndarray  # unit vector along the short side
    half_long: float
    half_short: float
    pixel_corners: np.ndarray  # 4x2 px (for drawing)
    pixel_outline: np.ndarray


class ClothDetector:
    """Finds a solid-coloured towel by hue."""

    def __init__(self, hue_range=(95, 130), min_sat=80, min_val=40, min_area_px=1500):
        self.hue_range = hue_range
        self.min_sat = min_sat
        self.min_val = min_val
        self.min_area_px = min_area_px

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lo = np.array([self.hue_range[0], self.min_sat, self.min_val])
        hi = np.array([self.hue_range[1], 255, 255])
        m = cv2.inRange(hsv, lo, hi)
        k = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

    def detect(self, rgb: np.ndarray, cam: CameraModel, cloth_z: float = 0.006) -> ClothObservation | None:
        m = self.mask(rgb)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < self.min_area_px:
            return None
        pix_outline = cnt.reshape(-1, 2).astype(float)
        # Approximate rectangle, then snap each rectangle corner to the nearest
        # real outline point so we target actual cloth, not empty floor.
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
        pix_corners = np.array([pix_outline[np.argmin(np.sum((pix_outline - b) ** 2, axis=1))] for b in box])

        outline = cam.pixel_to_world(pix_outline, z=cloth_z)[:, :2]
        corners = cam.pixel_to_world(pix_corners, z=cloth_z)[:, :2]
        area = 0.5 * abs(np.dot(outline[:, 0], np.roll(outline[:, 1], 1)) - np.dot(outline[:, 1], np.roll(outline[:, 0], 1)))

        # Rectangle axes in world coordinates, from the minimum-area rectangle
        box_w = cam.pixel_to_world(box, z=cloth_z)[:, :2]
        centre = box_w.mean(axis=0)
        e1, e2 = box_w[1] - box_w[0], box_w[2] - box_w[1]
        long_e, short_e = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
        u = long_e / np.linalg.norm(long_e)
        v = np.array([-u[1], u[0]])
        pu, pv = (outline - centre) @ u, (outline - centre) @ v
        return ClothObservation(
            corners=corners,
            outline=outline,
            area=float(area),
            centre=centre,
            u=u,
            v=v,
            half_long=float(np.max(np.abs(pu))),
            half_short=float(np.max(np.abs(pv))),
            pixel_corners=pix_corners,
            pixel_outline=pix_outline,
        )


# ---------------------------------------------------------- shirts, pants
@dataclass
class GarmentFrame:
    """A garment's own axes on the floor: e_v points bottom -> top (hem to
    collar, or leg hems to waistband), e_h to the garment's right."""

    centre: np.ndarray
    e_h: np.ndarray
    e_v: np.ndarray

    def to_local(self, pts: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(pts) - self.centre
        return np.column_stack([d @ self.e_h, d @ self.e_v])

    def to_world(self, hv: np.ndarray) -> np.ndarray:
        hv = np.atleast_2d(hv)
        return self.centre + hv[:, :1] * self.e_h + hv[:, 1:2] * self.e_v


@dataclass
class ShirtKeypoints:
    sleeve_left: np.ndarray
    sleeve_right: np.ndarray
    hem_left: np.ndarray
    hem_right: np.ndarray
    body_centre_h: float  # body centre line, shirt frame
    body_half_width: float


def _poly(outline: np.ndarray) -> np.ndarray:
    return outline.astype(np.float32).reshape(-1, 1, 2)


ShirtFrame = GarmentFrame  # backwards-compatible name


def garment_frame(obs: ClothObservation, top: str = "wider") -> GarmentFrame:
    """Find a flat garment's orientation from its silhouette alone.

    1. The two principal axes of the shape (image moments).
    2. The symmetry axis is the one the outline is mirror-symmetric about.
    3. Which end is the top:
       - top="wider":  the wider end (shirts: that's where the sleeves are)
       - top="closed": the end where the centre line runs through fabric
                       (pants: the other end has the gap between the legs)
    A learned keypoint detector would replace this for real, messy clothes.
    """
    poly = _poly(obs.outline)
    m = cv2.moments(poly)
    c = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    theta = 0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])
    axes = [np.array([math.cos(theta), math.sin(theta)]), np.array([-math.sin(theta), math.cos(theta)])]

    def asymmetry(a):
        b = np.array([-a[1], a[0]])
        d = obs.outline - c
        mirrored = c + np.outer(d @ a, a) - np.outer(d @ b, b)
        return float(np.mean([abs(cv2.pointPolygonTest(poly, (float(x), float(y)), True)) for x, y in mirrored]))

    e_v = min(axes, key=asymmetry)
    e_h = np.array([e_v[1], -e_v[0]])
    v = (obs.outline - c) @ e_v
    h = (obs.outline - c) @ e_h
    span = v.max() - v.min()
    if top == "wider":
        flip = np.ptp(h[v < v.min() + 0.3 * span]) > np.ptp(h[v > v.max() - 0.3 * span])
    elif top == "closed":
        inside = lambda t: cv2.pointPolygonTest(poly, tuple(map(float, c + t * e_v)), False) >= 0
        frac = lambda lo, hi: np.mean([inside(t) for t in np.linspace(lo, hi, 15)])
        flip = frac(v.min() + 0.02, v.min() + 0.2 * span) > frac(v.max() - 0.2 * span, v.max() - 0.02)
    else:
        raise ValueError(top)
    if flip:
        e_v, e_h = -e_v, -e_h
    return GarmentFrame(c, e_h, e_v)


def shirt_frame(obs: ClothObservation) -> GarmentFrame:
    return garment_frame(obs, "wider")


def shirt_keypoints(outline: np.ndarray, frame: ShirtFrame) -> ShirtKeypoints:
    hv = frame.to_local(outline)
    h, v = hv[:, 0], hv[:, 1]
    pick = lambda score: frame.to_world(hv[np.argmin(score)])[0]
    sleeve_left = pick(h)  # outermost point of each sleeve
    sleeve_right = pick(-h)
    hem_left = pick(h + v)  # most "bottom-left" point
    hem_right = pick(-h + v)
    band = (v > v.min() + 0.08) & (v < v.min() + 0.2)
    lo, hi = h[band].min(), h[band].max()
    return ShirtKeypoints(sleeve_left, sleeve_right, hem_left, hem_right, float((lo + hi) / 2), float((hi - lo) / 2))


@dataclass
class PantsKeypoints:
    waist_left: np.ndarray
    waist_right: np.ndarray
    hem_left: np.ndarray  # outer corner of each leg's hem
    hem_right: np.ndarray
    centre_h: float  # centre line (between the waist corners), garment frame


def pants_keypoints(outline: np.ndarray, frame: GarmentFrame) -> PantsKeypoints:
    hv = frame.to_local(outline)
    h, v = hv[:, 0], hv[:, 1]
    pick = lambda score: frame.to_world(hv[np.argmin(score)])[0]
    wl, wr = pick(h - v), pick(-h - v)  # most top-left / top-right
    hl, hr = pick(h + v), pick(-h + v)  # most bottom-left / bottom-right
    centre_h = float(frame.to_local(np.array([wl, wr]))[:, 0].mean())
    return PantsKeypoints(wl, wr, hl, hr, centre_h)
