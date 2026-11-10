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
        m = self.model
        # All cloth vertices of all garments, in one global numbering
        self.vertex_body_ids = m.flex_vertbodyid.copy()
        self.garment_slices = [slice(int(m.flex_vertadr[k]), int(m.flex_vertadr[k] + m.flex_vertnum[k])) for k in range(m.nflex)]
        tris = []
        for k in range(m.nflex):  # triangles, with global vertex indices
            a = m.flex_elemdataadr[k]
            tris.append(m.flex_elem[a : a + 3 * m.flex_elemnum[k]].reshape(-1, 3) + m.flex_vertadr[k])
        self.elements = np.concatenate(tris)
        self.element_garment = np.concatenate([np.full(len(t), k) for k, t in enumerate(tris)])
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
      * gripper: pinches the fabric under the fingertip - every cloth triangle
        (in every stacked layer) the fingertip is over, plus vertices within
        2 cm. This is a simplification ("magic pinch"); real fabric grasping
        physics is the hardest part to simulate and is better learned on hardware.
    """

    SERVO_SPEED = 2.5  # rad/s
    PINCH_RADIUS = 0.02
    PINCH_HEIGHT = 0.035  # grabs a stack of up to ~4 layers

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
        nv = len(world.vertex_body_ids)
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

    TOP_DEPTH_SLACK = 0.005

    def _under_tip(self):
        """Cloth surface heights directly under the fingertip: (vertex ids of
        triangles under the tip, their surface z at the tip), plus nearby vertices."""
        d = self.world.data
        tip = d.xpos[self._tip_body]
        verts = self.world.vertex_positions()
        tri = verts[self.world.elements]  # T x 3 x 3
        a, b, c = tri[:, 0, :2], tri[:, 1, :2], tri[:, 2, :2]
        v0, v1, v2 = b - a, c - a, tip[:2] - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        bary = np.column_stack([1 - l1 - l2, l1, l2])
        under = bary.min(axis=1) > -0.1
        z = np.sum(bary * tri[:, :, 2], axis=1)
        near = np.hypot(*(verts[:, :2] - tip[:2]).T) < self.PINCH_RADIUS
        return tip, verts, under, z, near

    def touching(self) -> bool:
        tip, verts, under, z, near = self._under_tip()
        top = max(z[under].max(initial=-1.0), verts[near, 2].max(initial=-1.0))
        return bool(top >= tip[2] - 0.003)

    def set_gripper(self, closed: bool, depth: float | None = None) -> None:
        d, m = self.world.data, self.world.model
        if closed and not self._closed:
            tip, verts, under, z, near = self._under_tip()
            Rtip = d.xmat[self._tip_body].reshape(3, 3)
            delta = verts - tip
            if depth is None:  # everything between the fingers at this height
                lo, hi = tip[2] - self.PINCH_HEIGHT, tip[2] + self.PINCH_HEIGHT
            else:  # only the top layer(s) under the fingertip
                hi = max(z[under].max(initial=-1.0), verts[near, 2].max(initial=-1.0)) + self.TOP_DEPTH_SLACK
                lo = hi - self.TOP_DEPTH_SLACK - depth
            take = near & (verts[:, 2] >= lo) & (verts[:, 2] <= hi)
            tris = under & (z >= lo) & (z <= hi)
            take[np.unique(self.world.elements[tris])] = True
            for vi in np.flatnonzero(take):
                eq = self._eq_ids[vi]
                m.eq_data[eq, 0:3] = 0.0
                m.eq_data[eq, 3:6] = Rtip.T @ delta[vi]  # keep the current offset (no snapping)
                d.eq_active[eq] = 1
        elif not closed:
            d.eq_active[self._eq_ids] = 0
        self._closed = closed
        m.geom_rgba[self._tip_geom] = (0.9, 0.15, 0.15, 1) if closed else (0.2, 0.9, 0.2, 1)

    def gripper_force(self) -> np.ndarray:
        d = self.world.data
        active = set(int(e) for e in self._eq_ids[d.eq_active[self._eq_ids] == 1])
        if not active:
            return np.zeros(3)
        rows = np.flatnonzero((d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & np.isin(d.efc_id[: d.nefc], list(active)))
        f = d.efc_force[rows]
        return np.array([f[0::3].sum(), f[1::3].sum(), f[2::3].sum()]) if len(f) else np.zeros(3)

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
