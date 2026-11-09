"""Builds the MuJoCo world as an MJCF XML string.

Scene: a floor, a towel (MuJoCo 'flex' cloth), N robots, ArUco markers on the
floor corners and on each robot, a ceiling camera looking straight down and a
second 'view' camera for pretty videos.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry import RobotGeometry
from ..perception import ARUCO_DICT


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

    @property
    def n_vertices(self) -> int:
        return self.count_x * self.count_y


@dataclass
class RobotSpec:
    name: str
    marker_id: int
    x: float
    y: float
    yaw: float
    colour: tuple = (0.95, 0.5, 0.1, 1.0)


@dataclass
class SceneSpec:
    towel: TowelSpec = field(default_factory=TowelSpec)
    robots: list[RobotSpec] = field(default_factory=list)
    geometry: RobotGeometry = field(default_factory=RobotGeometry)
    # Floor markers {id: (x, y, size)} - their positions are "measured" once at install time
    floor_markers: dict[int, tuple[float, float, float]] = field(
        default_factory=lambda: {0: (-1.45, 1.1, 0.14), 1: (1.45, 1.1, 0.14), 2: (1.45, -1.1, 0.14), 3: (-1.45, -1.1, 0.14)}
    )
    camera_height: float = 2.6
    camera_fovy_deg: float = 55.0
    camera_res: tuple[int, int] = (1280, 960)  # (W, H)


def _marker_geoms(marker_id: int, size: float, cx: float, cy: float, z: float, prefix: str) -> str:
    """An ArUco marker as a white plate with black boxes (no textures needed).

    Image row 0 (top of the printed marker) is placed toward +y, column 0
    toward -x, so seen from above (camera up = +y) the marker reads correctly.
    """
    bits = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), marker_id, 6)
    n = bits.shape[0]  # 6 cells incl. border
    cell = size / n
    plate = (n + 2) * cell  # one white cell of quiet zone each side
    out = [
        f'<geom name="{prefix}_plate" type="box" size="{plate/2:.5f} {plate/2:.5f} 0.0005" '
        f'pos="{cx:.5f} {cy:.5f} {z - 0.0006:.5f}" rgba="1 1 1 1" contype="0" conaffinity="0" mass="0"/>'
    ]
    for i in range(n):  # merge horizontal runs of black cells into one box
        j = 0
        while j < n:
            if bits[i, j] == 0:
                k = j
                while k < n and bits[i, k] == 0:
                    k += 1
                x0 = cx + (j - n / 2) * cell
                x1 = cx + (k - n / 2) * cell
                y = cy + (n / 2 - i - 0.5) * cell
                out.append(
                    f'<geom type="box" size="{(x1-x0)/2:.5f} {cell/2:.5f} 0.0005" pos="{(x0+x1)/2:.5f} {y:.5f} {z:.5f}" '
                    f'rgba="0 0 0 1" contype="0" conaffinity="0" mass="0"/>'
                )
                j = k
            else:
                j += 1
    return "\n".join(out)


def _robot_xml(r: RobotSpec, g: RobotGeometry) -> str:
    c = " ".join(f"{v:.3f}" for v in r.colour)
    hz = g.base_height / 2
    marker = _marker_geoms(r.marker_id, g.marker_size, g.marker_offset_x, 0.0, g.marker_z, f"{r.name}_marker")
    wheels = "\n".join(
        f'<geom type="cylinder" size="0.04 0.012" pos="{0.1*math.cos(a):.3f} {0.1*math.sin(a):.3f} 0.04" '
        f'euler="{math.pi/2:.4f} 0 {a + math.pi/2:.4f}" rgba=".1 .1 .1 1" contype="0" conaffinity="0" mass="0"/>'
        for a in (0, 2 * math.pi / 3, 4 * math.pi / 3)
    )
    arm_geom = 'contype="0" conaffinity="0"'
    return f"""
    <body name="{r.name}" pos="{r.x:.4f} {r.y:.4f} 0">  <!-- initial yaw is set through the yaw joint -->
      <joint name="{r.name}_x" type="slide" axis="1 0 0"/>
      <joint name="{r.name}_y" type="slide" axis="0 1 0"/>
      <joint name="{r.name}_yaw" type="hinge" axis="0 0 1"/>
      <!-- chassis: collides with the cloth only (so driving over a towel drags it, like real wheels would) -->
      <geom name="{r.name}_chassis" type="box" size="{g.base_half_length} {g.base_half_width} {hz}" pos="0 0 {hz}"
            rgba=".22 .23 .26 1" contype="2" conaffinity="0" mass="6"/>
      {wheels}
      {marker}
      <geom type="box" size="0.035 0.035 {(g.mount_z - g.base_height)/2:.4f}" pos="{g.mount_x} 0 {(g.mount_z + g.base_height)/2:.4f}"
            rgba="{c}" {arm_geom} mass="0.2"/>
      <body name="{r.name}_upper" pos="{g.mount_x} 0 {g.mount_z}" gravcomp="1">
        <joint name="{r.name}_q1" type="hinge" axis="0 -1 0" range="-1.4 2.4" damping="1" armature="0.01"/>
        <geom type="capsule" fromto="0 0 0 {g.l1} 0 0" size="0.018" rgba="{c}" {arm_geom} mass="0.3"/>
        <body name="{r.name}_fore" pos="{g.l1} 0 0" gravcomp="1">
          <joint name="{r.name}_q2" type="hinge" axis="0 -1 0" range="-3.0 0.6" damping="1" armature="0.01"/>
          <geom type="capsule" fromto="0 0 0 {g.l2} 0 0" size="0.015" rgba="{c}" {arm_geom} mass="0.2"/>
          <body name="{r.name}_wrist" pos="{g.l2} 0 0" gravcomp="1">
            <joint name="{r.name}_q3" type="hinge" axis="0 -1 0" range="-3.2 3.2" damping="0.5" armature="0.01"/>
            <geom type="capsule" fromto="0 0 0 0 0 {-(g.finger - 0.015):.4f}" size="0.011" rgba=".2 .2 .2 1" {arm_geom} mass="0.05"/>
            <body name="{r.name}_tip" pos="0 0 {-g.finger}">
              <geom name="{r.name}_tipgeom" type="sphere" size="0.012" rgba=".2 .9 .2 1" {arm_geom} mass="0.01"/>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def build_scene_xml(spec: SceneSpec) -> str:
    t, g = spec.towel, spec.geometry
    W, H = spec.camera_res
    floor_markers = "\n".join(
        _marker_geoms(mid, size, x, y, 0.0012, f"floor_marker{mid}") for mid, (x, y, size) in spec.floor_markers.items()
    )
    robots = "\n".join(_robot_xml(r, g) for r in spec.robots)

    # "Grasp" model: one inactive connect-constraint per (robot, cloth vertex).
    # Closing the gripper switches on the constraints for vertices within the
    # gripper's pinch zone. See SimRobot.set_gripper.
    eqs = "\n".join(
        f'<connect name="{r.name}_grip{i}" body1="towel_{i}" body2="{r.name}_tip" anchor="0 0 0" active="false" solref="0.01 1"/>'
        for r in spec.robots
        for i in range(t.n_vertices)
    )
    acts = "\n".join(
        f"""<velocity name="{r.name}_vx" joint="{r.name}_x" kv="400"/>
        <velocity name="{r.name}_vy" joint="{r.name}_y" kv="400"/>
        <velocity name="{r.name}_wz" joint="{r.name}_yaw" kv="40"/>
        <position name="{r.name}_a1" joint="{r.name}_q1" kp="150" kv="6"/>
        <position name="{r.name}_a2" joint="{r.name}_q2" kp="120" kv="4"/>
        <position name="{r.name}_a3" joint="{r.name}_q3" kp="60" kv="2"/>"""
        for r in spec.robots
    )
    return f"""
<mujoco model="laundry_bot">
  <compiler angle="radian"/>
  <option timestep="0.004" integrator="implicitfast"/>
  <visual>
    <global offwidth="{W}" offheight="{H}"/>
    <headlight ambient="0.38 0.38 0.38" diffuse="0.3 0.3 0.3" specular="0 0 0"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".80 .77 .72" rgb2=".78 .75 .70" width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0"/>
    <material name="cloth" rgba="{' '.join(map(str, t.rgba))}" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0.5 -0.5 3" dir="-0.15 0.15 -1" diffuse="0.5 0.5 0.5" castshadow="true"/>
    <geom name="floor" type="plane" size="2.5 2 0.1" material="floor" friction="0.8" contype="1" conaffinity="1"/>
    {floor_markers}
    <camera name="overhead" pos="0 0 {spec.camera_height}" fovy="{spec.camera_fovy_deg}"/>
    <camera name="view" pos="0 -2.9 2.1" xyaxes="1 0 0 0 0.6 0.8" fovy="50"/>
    <flexcomp name="towel" type="grid" count="{t.count_x} {t.count_y} 1" spacing="{t.spacing} {t.spacing} {t.spacing}"
              pos="{t.centre[0]:.4f} {t.centre[1]:.4f} 0.006" euler="0 0 {t.yaw:.5f}" radius="0.004" dim="2"
              mass="{t.mass}" material="cloth">
      <contact condim="3" solref="0.01 1" selfcollide="{'auto' if t.self_collision else 'none'}" contype="1" conaffinity="3" friction="0.8"/>
      <edge equality="true" damping="0.002"/>
    </flexcomp>
    {robots}
  </worldbody>
  <equality>
    {eqs}
  </equality>
  <actuator>
    {acts}
  </actuator>
</mujoco>"""
