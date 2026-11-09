"""Simulated hardware: implements the Camera and RobotDriver interfaces on MuJoCo."""
from __future__ import annotations

import math
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np

from ..geometry import RobotGeometry
from ..perception import Pose2D
from .scene import SceneSpec, build_scene_xml


class SimWorld:
    def __init__(self, spec: SceneSpec, seed: int = 0):
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_string(build_scene_xml(spec))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        nv = spec.towel.n_vertices
        self.vertex_body_ids = np.array([self.model.body(f"towel_{i}").id for i in range(nv)])
        self.robots = [SimRobot(self, i, r.name, r.marker_id, spec.geometry, self.rng) for i, r in enumerate(spec.robots)]
        for r in self.robots:
            r.reset_arm()
        mujoco.mj_forward(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def step(self, duration: float) -> None:
        n = max(1, int(round(duration / self.model.opt.timestep)))
        for _ in range(n):
            for r in self.robots:
                r._apply()
            mujoco.mj_step(self.model, self.data)

    def vertex_positions(self) -> np.ndarray:
        return self.data.xpos[self.vertex_body_ids].copy()


class SimCamera:
    """The overhead camera. Implements interfaces.Camera."""

    def __init__(self, world: SimWorld, name: str = "overhead"):
        W, H = world.spec.camera_res
        self.world = world
        self.cam_name = name
        self.renderer = mujoco.Renderer(world.model, H, W)
        fy = (H / 2) / math.tan(math.radians(world.model.camera(name).fovy[0]) / 2)
        self.K = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]])

    def capture(self) -> np.ndarray:
        self.renderer.update_scene(self.world.data, camera=self.cam_name)
        return self.renderer.render()


class SimRobot:
    """One arm-on-wheels. Implements interfaces.RobotDriver.

    Realism knobs:
      * wheel slip: every robot has its own speed-scale error and drift, so
        driving "blind" goes wrong - the director must use the camera.
      * servo speed limit: arm joints move toward their targets at <= 2.5 rad/s.
      * gripper: pinches every cloth vertex within ~2.5 cm of the fingertip.
        This is a simplification ("magic pinch"); real fabric grasping physics
        is the hardest part to simulate and is better learned on hardware.
    """

    SERVO_SPEED = 2.5  # rad/s
    PINCH_RADIUS = 0.025
    PINCH_HEIGHT = 0.02

    def __init__(self, world: SimWorld, index: int, name: str, marker_id: int, g: RobotGeometry, rng):
        self.world, self.index, self.name, self.marker_id, self.g = world, index, name, marker_id, g
        m = world.model
        self._jx, self._jy, self._jyaw = (m.joint(f"{name}_{s}").qposadr[0] for s in ("x", "y", "yaw"))
        world.data.qpos[self._jyaw] = world.spec.robots[index].yaw
        self._x0, self._y0 = m.body(name).pos[:2]
        self._act = {k: m.actuator(f"{name}_{k}").id for k in ("vx", "vy", "wz", "a1", "a2", "a3")}
        self._arm_qadr = [m.joint(f"{name}_q{i}").qposadr[0] for i in (1, 2, 3)]
        self._tip_body = m.body(f"{name}_tip").id
        self._tip_geom = m.geom(f"{name}_tipgeom").id
        nv = world.spec.towel.n_vertices
        self._eq_ids = np.array([m.equality(f"{name}_grip{i}").id for i in range(nv)])
        # wheel slip personality
        self._scale = 1 + rng.normal(0, 0.08, size=3)
        self._cross = rng.normal(0, 0.05)
        self._cmd = np.zeros(3)
        self._q_target = np.zeros(3)
        self._q_cmd = np.zeros(3)
        self._closed = False

    # ----- RobotDriver interface -----
    def set_base_velocity(self, vx: float, vy: float, wz: float) -> None:
        self._cmd = np.array([vx, vy, wz], dtype=float)

    def set_arm_joints(self, q) -> None:
        self._q_target = np.asarray(q, dtype=float).copy()

    def set_gripper(self, closed: bool) -> None:
        d, m = self.world.data, self.world.model
        if closed and not self._closed:
            tip = d.xpos[self._tip_body]
            Rtip = d.xmat[self._tip_body].reshape(3, 3)
            verts = self.world.vertex_positions()
            delta = verts - tip
            near = (np.hypot(delta[:, 0], delta[:, 1]) < self.PINCH_RADIUS) & (np.abs(delta[:, 2]) < self.PINCH_HEIGHT)
            for vi in np.flatnonzero(near):
                eq = self._eq_ids[vi]
                m.eq_data[eq, 0:3] = 0.0
                m.eq_data[eq, 3:6] = Rtip.T @ delta[vi]  # keep the current offset (no snapping)
                d.eq_active[eq] = 1
        elif not closed:
            d.eq_active[self._eq_ids] = 0
        self._closed = closed
        m.geom_rgba[self._tip_geom] = (0.9, 0.15, 0.15, 1) if closed else (0.2, 0.9, 0.2, 1)

    def holding(self) -> bool:
        return bool(self.world.data.eq_active[self._eq_ids].any())

    # ----- simulation internals -----
    def reset_arm(self, q=(1.24, -1.48, 0.24)) -> None:
        q = np.asarray(q, float)
        self._q_target = q.copy()
        self._q_cmd = q.copy()
        for adr, v in zip(self._arm_qadr, q):
            self.world.data.qpos[adr] = v

    def _apply(self) -> None:
        d = self.world.data
        dt = self.world.model.opt.timestep
        yaw = self.true_pose().yaw
        vx, vy, wz = self._cmd * self._scale
        vy += self._cross * vx  # a bit of sideways drift when driving forward
        c, s = math.cos(yaw), math.sin(yaw)
        d.ctrl[self._act["vx"]] = c * vx - s * vy
        d.ctrl[self._act["vy"]] = s * vx + c * vy
        d.ctrl[self._act["wz"]] = wz
        step = np.clip(self._q_target - self._q_cmd, -self.SERVO_SPEED * dt, self.SERVO_SPEED * dt)
        self._q_cmd += step
        for k, v in zip(("a1", "a2", "a3"), self._q_cmd):
            d.ctrl[self._act[k]] = v

    def true_pose(self) -> Pose2D:
        """Ground truth, for evaluation only - the director never calls this."""
        q = self.world.data.qpos
        return Pose2D(self._x0 + q[self._jx], self._y0 + q[self._jy], q[self._jyaw])
