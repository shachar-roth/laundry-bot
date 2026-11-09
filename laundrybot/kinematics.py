"""Inverse / forward kinematics for the planar 3-joint arm.

Joint convention: every joint pitches about the robot's -y axis, so a positive
angle lifts the link. q3 (wrist) is always chosen so the finger points straight
down, which is what you want for pinching cloth off a flat surface.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import RobotGeometry


class Unreachable(ValueError):
    pass


def arm_ik(g: RobotGeometry, r: float, z: float) -> np.ndarray:
    """Joint angles (q1, q2, q3) that put the fingertip at reach r, height z.

    Uses the elbow-up solution so the arm never digs into the floor.
    """
    wx = r - g.mount_x
    wz = z + g.finger - g.mount_z  # wrist sits `finger` above the tip
    d = math.hypot(wx, wz)
    if d > g.l1 + g.l2 - 1e-3 or d < abs(g.l1 - g.l2) + 0.02:
        raise Unreachable(f"tip (r={r:.3f}, z={z:.3f}) is out of reach")
    c = (d * d - g.l1**2 - g.l2**2) / (2 * g.l1 * g.l2)
    q2 = -math.acos(max(-1.0, min(1.0, c)))
    q1 = math.atan2(wz, wx) - math.atan2(g.l2 * math.sin(q2), g.l1 + g.l2 * math.cos(q2))
    q3 = -(q1 + q2)
    return np.array([q1, q2, q3])


def arm_fk(g: RobotGeometry, q) -> tuple[float, float]:
    """Fingertip (r, z) for joint angles q."""
    q1, q2, q3 = q
    wx = g.l1 * math.cos(q1) + g.l2 * math.cos(q1 + q2)
    wz = g.l1 * math.sin(q1) + g.l2 * math.sin(q1 + q2)
    phi = q1 + q2 + q3
    r = g.mount_x + wx + g.finger * math.sin(phi)
    z = g.mount_z + wz - g.finger * math.cos(phi)
    return r, z
