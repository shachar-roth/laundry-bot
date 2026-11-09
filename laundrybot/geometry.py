"""Physical dimensions of one robot ("arm on wheels").

These numbers are shared by the simulator (to build the model) and by the
director (to plan motions), exactly as a real robot's spec sheet would be.

Kinematic layout of the arm
---------------------------
The arm is planar: three pitch joints (shoulder, elbow, wrist) acting in the
robot's forward/up plane, plus a gripper. This matches the pitch joints of an
SO-100/SO-101 arm (shoulder_lift, elbow_flex, wrist_flex) with its pan and roll
joints held at zero. Sideways reach comes from turning the holonomic base
instead, so the arm stays simple.

Frames: robot frame x = forward, y = left, z = up, origin on the floor under the
base centre. "Reach" r is the forward distance of the fingertip from the base
centre; z is fingertip height above the floor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RobotGeometry:
    # Base (rectangular, holonomic like LeKiwi's three omni-wheels)
    base_half_length: float = 0.16
    base_half_width: float = 0.13
    base_height: float = 0.10

    # ArUco marker on the base's top plate, at the rear (away from the arm)
    marker_size: float = 0.11  # outer black square, metres
    marker_offset_x: float = -0.075  # marker centre, in robot frame

    # Arm
    mount_x: float = 0.14  # shoulder position, robot frame
    mount_z: float = 0.16
    l1: float = 0.24  # upper arm
    l2: float = 0.24  # forearm
    finger: float = 0.08  # wrist joint -> fingertip (kept vertical)

    @property
    def marker_z(self) -> float:
        return self.base_height + 0.002

    @property
    def footprint_radius(self) -> float:
        """Radius of the circle that contains the base at any heading."""
        return math.hypot(self.base_half_length, self.base_half_width)


DEFAULT_GEOMETRY = RobotGeometry()
