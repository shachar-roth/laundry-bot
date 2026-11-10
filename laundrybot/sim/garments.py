"""Cloth items for the simulator: a rectangular towel and a T-shirt.

Each garment knows how to write itself as a MuJoCo <flexcomp> (named "cloth",
so vertex bodies are cloth_0, cloth_1, ...).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


def _contact_xml(self_collision: bool, cloth_contacts: bool = True) -> str:
    # contype 1 hits the floor; conaffinity bit 2 lets robot chassis push the
    # cloth; bit 1 lets other garments collide with it (expensive with many).
    aff = 3 if cloth_contacts else 2
    return (
        f'<contact condim="3" solref="0.01 1" selfcollide="{"auto" if self_collision else "none"}" '
        f'contype="1" conaffinity="{aff}" friction="0.8"/>\n      <edge equality="true" damping="0.002"/>'
    )


@dataclass
class TowelSpec:
    count_x: int = 11  # vertices along x (towel 0.5 m wide)
    count_y: int = 15  # vertices along y (towel 0.7 m long)
    spacing: float = 0.05
    centre: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    mass: float = 0.15
    rgba: tuple = (0.15, 0.35, 0.85, 1.0)
    # Layers rest on each other instead of passing through. ~5x slower physics.
    self_collision: bool = True
    radius: float = 0.004  # cloth half-thickness for collisions
    cloth_contacts: bool = True  # collide with other garments

    @property
    def n_vertices(self) -> int:
        return self.count_x * self.count_y

    def flexcomp_xml(self, name: str = "cloth", material: str = "cloth") -> str:
        return f"""<flexcomp name="{name}" type="grid" count="{self.count_x} {self.count_y} 1"
              spacing="{self.spacing} {self.spacing} {self.spacing}" pos="{self.centre[0]:.4f} {self.centre[1]:.4f} 0.006"
              euler="0 0 {self.yaw:.5f}" radius="{self.radius}" dim="2" mass="{self.mass}" material="{material}">
      {_contact_xml(self.self_collision, self.cloth_contacts)}
    </flexcomp>"""


def tshirt_outline(
    body_width=0.50, length=0.70, sleeve_length=0.20, sleeve_width=0.17, sleeve_angle_deg=-25.0, neck_width=0.18, neck_depth=0.065
) -> np.ndarray:
    """T-shirt silhouette, laid flat, collar toward +y. Counter-clockwise polygon (N x 2)."""
    b, top = body_width / 2, length / 2
    a = math.radians(sleeve_angle_deg)
    d = np.array([math.cos(a), math.sin(a)])  # along the sleeve
    nrm = np.array([-d[1], d[0]])  # across the sleeve opening (points up)
    armpit = np.array([b, top - 0.21])
    tip_low = armpit + sleeve_length * d
    tip_high = tip_low + sleeve_width * nrm
    shoulder_y = top - 0.02
    shoulder = tip_high - ((shoulder_y - tip_high[1]) / -d[1]) * d  # sleeve top seam meets the body
    nw = neck_width / 2
    neck = [(nw * math.cos(t), top + 0.015 - neck_depth * math.sin(t)) for t in np.linspace(0, math.pi / 2, 5)]
    right = [(0.0, -top), (b, -top), tuple(armpit), tuple(tip_low), tuple(tip_high), tuple(shoulder), (nw, top + 0.015)] + neck[1:]
    # right half goes bottom-centre -> up the right side -> neck centre; mirror for the left half
    left = [(-x, y) for x, y in reversed(right[1:-1])]
    return np.array(right + left, dtype=float)


def mesh_polygon(poly: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Triangle mesh of a (possibly concave) polygon with edge length ~h."""
    from scipy.spatial import Delaunay

    cnt = poly.astype(np.float32).reshape(-1, 1, 2)
    inside = lambda p, tol=0.0: cv2.pointPolygonTest(cnt, (float(p[0]), float(p[1])), True) >= tol
    boundary = []
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(1, int(math.ceil(np.linalg.norm(q - p) / h)))
        boundary += [p + (q - p) * k / n for k in range(n)]
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    xs = np.arange(lo[0] + h / 2, hi[0], h)
    ys = np.arange(lo[1] + h / 2, hi[1], h * math.sqrt(3) / 2)
    interior = [
        np.array([x + (h / 2 if i % 2 else 0), y]) for i, y in enumerate(ys) for x in xs if inside((x + (h / 2 if i % 2 else 0), y), 0.45 * h)
    ]
    pts = np.array(boundary + interior)
    tri = Delaunay(pts).simplices
    keep = []
    for t in tri:
        a, b, c = pts[t]
        if not all(inside(m, -1e-6) for m in ((a + b + c) / 3, (a + b) / 2, (b + c) / 2, (a + c) / 2)):
            continue
        area = 0.5 * ((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0])
        if abs(area) < 0.05 * h * h:
            continue
        keep.append(t if area > 0 else t[[0, 2, 1]])  # counter-clockwise = normal up
    return pts, np.array(keep)


def pants_outline(waist=0.40, hip=0.46, length=1.0, rise=0.28, hem=0.20, hem_splay=0.04) -> np.ndarray:
    """Trousers laid flat, waistband toward +y, legs slightly apart. Counter-clockwise."""
    top = length / 2
    crotch = (0.0, top - rise)
    outer_hem_x = hip / 2 + hem_splay
    right = [crotch, (outer_hem_x - hem, -top), (outer_hem_x, -top), (hip / 2, top - rise + 0.06), (waist / 2, top)]
    left = [(-x, y) for x, y in reversed(right[1:])]
    return np.array(right + left, dtype=float)


@dataclass
class _FlatGarment:
    """A garment made from one flat polygon (laid out with its top toward +y at yaw 0)."""

    centre: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    mesh_size: float = 0.045
    mass: float = 0.18
    rgba: tuple = (0.12, 0.3, 0.8, 1.0)
    self_collision: bool = True
    radius: float = 0.004
    cloth_contacts: bool = True

    def outline(self) -> np.ndarray:
        raise NotImplementedError

    def __post_init__(self):
        local, self.elements = mesh_polygon(self.outline(), self.mesh_size)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.points = local @ np.array([[c, s], [-s, c]]) + np.asarray(self.centre)

    @property
    def n_vertices(self) -> int:
        return len(self.points)

    def flexcomp_xml(self, name: str = "cloth", material: str = "cloth") -> str:
        pts = " ".join(f"{x:.4f} {y:.4f} 0.006" for x, y in self.points)
        els = " ".join(str(i) for i in self.elements.ravel())
        return f"""<flexcomp name="{name}" type="direct" point="{pts}" element="{els}"
              radius="{self.radius}" dim="2" mass="{self.mass}" material="{material}">
      {_contact_xml(self.self_collision, self.cloth_contacts)}
    </flexcomp>"""


@dataclass
class ShirtSpec(_FlatGarment):
    """T-shirt; yaw 0 = collar toward +y."""

    def outline(self) -> np.ndarray:
        return tshirt_outline()


@dataclass
class PantsSpec(_FlatGarment):
    """Trousers (denim-ish blue, heavier); yaw 0 = waistband toward +y."""

    mass: float = 0.35
    rgba: tuple = (0.13, 0.25, 0.62, 1.0)

    def outline(self) -> np.ndarray:
        return pants_outline()
