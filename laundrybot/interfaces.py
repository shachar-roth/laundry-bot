"""The boundary between the "brain" (director) and the hardware.

The director only ever talks to these two interfaces. The simulator implements
them in `laundrybot.sim.world`; a real robot (e.g. LeKiwi base + SO-101 arm) and
a real ceiling webcam would implement the same methods, and the director code
would not change.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Camera(Protocol):
    #: 3x3 pinhole intrinsics. For a real camera, measure once with a
    #: checkerboard (cv2.calibrateCamera). Position/orientation are NOT needed:
    #: the director works them out from markers on the floor.
    K: np.ndarray

    def capture(self) -> np.ndarray:
        """Latest RGB frame, HxWx3 uint8."""
        ...


class RobotDriver(Protocol):
    name: str
    marker_id: int  # ArUco id printed on top of the robot

    def set_base_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Body-frame velocity command: forward m/s, left m/s, turn rad/s.

        Like real wheels, the robot will NOT move exactly as commanded (slip,
        uneven floor). The director corrects using the overhead camera.
        """
        ...

    def set_arm_joints(self, q: np.ndarray) -> None:
        """Target angles for the 3 pitch joints; servos move there at their own speed."""
        ...

    def set_gripper(self, closed: bool) -> None: ...

    def holding(self) -> bool:
        """True if the gripper closed on something (real grippers sense this
        from the jaw opening / motor current)."""
        ...
