"""Draws what the director believes on top of the camera image (for videos/debugging)."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .director import Director

GREEN, RED, CYAN, YELLOW, WHITE, BLACK, MAGENTA = (40, 200, 60), (230, 40, 40), (0, 170, 230), (250, 200, 0), (255, 255, 255), (0, 0, 0), (230, 0, 200)


def _text(img, s, org, scale=0.9, color=WHITE):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 5, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_overlay(director: Director, image: np.ndarray) -> np.ndarray:
    img = image.copy()
    cam = director.cam
    if not cam.calibrated:
        _text(img, "calibrating...", (20, 40))
        return img
    if director.cloth is not None:
        c = director.cloth
        pts = cam.world_to_pixel(c.outline).astype(np.int32)
        cv2.polylines(img, [pts], True, GREEN, 2, cv2.LINE_AA)
        for i, p in enumerate(cam.world_to_pixel(c.corners).astype(int)):
            cv2.circle(img, tuple(p), 9, RED, -1, cv2.LINE_AA)
    if director.fold_line is not None:
        a, b = cam.world_to_pixel(np.array(director.fold_line))
        n = 14
        for i in range(0, n, 2):  # dashed
            p, q = a + (b - a) * i / n, a + (b - a) * (i + 1) / n
            cv2.line(img, tuple(p.astype(int)), tuple(q.astype(int)), MAGENTA, 3, cv2.LINE_AA)
    for name, pt in director.keypoints.items():
        p = cam.world_to_pixel(np.atleast_2d(pt))[0].astype(int)
        cv2.circle(img, tuple(p), 8, MAGENTA, -1, cv2.LINE_AA)
        _text(img, name, (p[0] + 10, p[1] - 10), 0.6, MAGENTA)
    for rs in director.robots:
        if rs.id not in director.tracker.poses:
            continue
        p = director.tracker.poses[rs.id]
        z = director.g.marker_z
        base = cam.world_to_pixel(np.array([[p.x, p.y, z]]))[0].astype(int)
        nose = cam.world_to_pixel(np.array([[p.x + 0.25 * math.cos(p.yaw), p.y + 0.25 * math.sin(p.yaw), z]]))[0].astype(int)
        cv2.arrowedLine(img, tuple(base), tuple(nose), YELLOW, 4, cv2.LINE_AA, tipLength=0.3)
        if rs.path:
            pts = cam.world_to_pixel(np.array(rs.path)).astype(np.int32)
            cv2.polylines(img, [pts], False, CYAN, 2, cv2.LINE_AA)
            for q in pts[1:]:
                cv2.circle(img, tuple(q), 6, CYAN, -1)
        _text(img, f"{rs.driver.name}: {rs.status}", (base[0] - 70, base[1] + 85), 0.75, YELLOW)
    _text(img, f"t={director.t:5.1f}s  {director.phase}", (20, 45), 1.1)
    return img
