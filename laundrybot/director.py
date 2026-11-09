"""The director: watches the overhead camera and tells every robot what to do.

Control style
-------------
Everything runs in one loop at CTRL_DT (10 Hz):  camera frame -> perceive ->
advance the current task -> send commands. Tasks are written as Python
generators: each `yield` means "wait for the next camera frame". That makes
multi-step behaviour read top-to-bottom like a recipe, and `parallel(...)`
runs several robots' recipes at once.

The robots are never trusted to drive accurately on their own: every base
motion is closed-loop on the camera's estimate of where the robot really is.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Generator, Iterable

import numpy as np

from .geometry import RobotGeometry
from .interfaces import RobotDriver
from .kinematics import arm_ik
from .perception import CameraModel, ClothDetector, ClothObservation, Pose2D, RobotTracker, detect_markers

Task = Generator[None, None, object]

CTRL_DT = 0.1


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def min_jerk(tau: float) -> tuple[float, float]:
    """Smooth 0->1 profile and its derivative (w.r.t. tau)."""
    tau = min(max(tau, 0.0), 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    return s, ds


def parallel(*tasks: Task) -> Task:
    """Run several tasks side by side; returns their results in order."""
    tasks = list(tasks)
    results: list[object] = [None] * len(tasks)
    live = set(range(len(tasks)))
    while live:
        for i in sorted(live):
            try:
                next(tasks[i])
            except StopIteration as e:
                results[i] = e.value
                live.discard(i)
        if live:
            yield
    return results


@dataclass
class RobotState:
    driver: RobotDriver
    tip: np.ndarray  # commanded fingertip (reach r, height z), robot frame
    status: str = "idle"
    path: list[np.ndarray] | None = None

    @property
    def id(self) -> int:
        return self.driver.marker_id


@dataclass
class FoldAssignment:
    robot: RobotState
    grab: np.ndarray  # world xy of the corner to pick up
    place: np.ndarray  # world xy where it goes
    facing: float  # robot yaw while folding (faces into the towel)
    stage: Pose2D  # where to park before grasping


class FoldFailed(RuntimeError):
    pass


class Director:
    # Arm heights (fingertip above floor)
    GRASP_Z = 0.006
    HOVER_Z = 0.10
    HOLD_Z = 0.08
    PLACE_Z = 0.03
    HOME_TIP = (0.45, 0.25)
    REACH = 0.52  # base centre -> grasped corner; keeps the arm stretched out
    # Base motion
    V_MAX = 0.25
    W_MAX = 1.0
    KP_POS = 1.5
    KP_YAW = 2.0
    STALE_LIMIT = 0.5  # s without seeing a robot's marker -> stop it

    def __init__(
        self,
        K: np.ndarray,
        floor_markers: dict[int, tuple[float, float, float]],
        robots: Iterable[RobotDriver],
        geometry: RobotGeometry,
        log: Callable[[str], None] = print,
    ):
        self.cam = CameraModel(K)
        self.floor_markers = floor_markers
        self.g = geometry
        self.robots = [RobotState(r, np.array(self.HOME_TIP, float)) for r in robots]
        self.tracker = RobotTracker(self.cam, geometry, [r.id for r in self.robots])
        self.cloth_detector = ClothDetector()
        self.cloth: ClothObservation | None = None
        self.first_cloth: ClothObservation | None = None
        self.image: np.ndarray | None = None
        self.detections: dict[int, np.ndarray] = {}
        self.t = 0.0
        self.phase = "starting"
        self._log = log
        self.events: list[tuple[float, str]] = []

    # ------------------------------------------------------------------ utils
    def log(self, msg: str) -> None:
        self.events.append((self.t, msg))
        self._log(f"[t={self.t:6.1f}s] {msg}")

    def pose(self, rs: RobotState) -> Pose2D:
        return self.tracker.poses[rs.id]

    def perceive(self, image: np.ndarray, t: float) -> None:
        self.t, self.image = t, image
        self.detections = detect_markers(image)
        if not self.cam.calibrated:
            if self.cam.calibrate(self.detections, self.floor_markers):
                self.log(f"camera calibrated from floor markers (reprojection error {self.cam.reprojection_error_px:.2f} px)")
            else:
                return
        self.tracker.update(self.detections, t)

    def wait(self, seconds: float) -> Task:
        for _ in range(max(1, int(round(seconds / CTRL_DT)))):
            yield

    def stop(self, rs: RobotState) -> None:
        rs.driver.set_base_velocity(0.0, 0.0, 0.0)

    def _send_world_velocity(self, rs: RobotState, v_world: np.ndarray, wz: float) -> None:
        yaw = self.pose(rs).yaw
        c, s = math.cos(yaw), math.sin(yaw)
        rs.driver.set_base_velocity(c * v_world[0] + s * v_world[1], -s * v_world[0] + c * v_world[1], wz)

    # ------------------------------------------------------------------ arm
    def set_tip(self, rs: RobotState, r: float, z: float) -> None:
        rs.driver.set_arm_joints(arm_ik(self.g, r, z))
        rs.tip = np.array([r, z])

    def move_arm(self, rs: RobotState, r: float, z: float, speed: float = 0.3) -> Task:
        start, goal = rs.tip.copy(), np.array([r, z], float)
        n = max(1, math.ceil(np.linalg.norm(goal - start) / (speed * CTRL_DT)))
        for i in range(1, n + 1):
            p = start + (goal - start) * i / n
            self.set_tip(rs, *p)
            yield
        yield from self.wait(0.3)  # let the servos catch up

    # ------------------------------------------------------------------ base
    def wait_until_tracked(self, rs: RobotState) -> Task:
        while self.tracker.age(rs.id, self.t) > self.STALE_LIMIT:
            self.stop(rs)
            rs.status = "lost - waiting for camera"
            yield

    def drive_to(self, rs: RobotState, goal: Pose2D, avoid_cloth: bool = True, pos_tol=0.008, yaw_tol=0.02, timeout=40.0) -> Task:
        """Closed-loop drive to a pose, going around the towel if needed."""
        yield from self.wait_until_tracked(rs)
        start = self.pose(rs).xy
        path = self.plan_path(start, goal.xy) if avoid_cloth else [goal.xy]
        rs.path = [start] + path
        t0 = self.t
        settled = 0
        for k, wp in enumerate(path):
            last = k == len(path) - 1
            while True:
                if self.t - t0 > timeout:
                    self.stop(rs)
                    raise FoldFailed(f"{rs.driver.name}: drive timed out")
                yield from self.wait_until_tracked(rs)
                rs.status = f"driving ({k+1}/{len(path)})"
                p = self.pose(rs)
                e = wp - p.xy
                dist = float(np.linalg.norm(e))
                yaw_err = wrap(goal.yaw - p.yaw)
                if last:
                    if dist < pos_tol and abs(yaw_err) < yaw_tol:
                        settled += 1
                        if settled >= 2:
                            break
                    else:
                        settled = 0
                    v = e * self.KP_POS
                elif dist < 0.06:
                    break
                else:
                    v = e / dist * self.V_MAX
                n = np.linalg.norm(v)
                if n > self.V_MAX:
                    v *= self.V_MAX / n
                self._send_world_velocity(rs, v, float(np.clip(self.KP_YAW * yaw_err, -self.W_MAX, self.W_MAX)))
                yield
        self.stop(rs)
        rs.path = None
        rs.status = "arrived"

    def face_point(self, rs: RobotState, target: np.ndarray, tol=0.008) -> Task:
        """Turn in place until `target` is straight ahead (the arm only reaches forward)."""
        hold = self.pose(rs).xy
        settled = 0
        for _ in range(100):
            yield from self.wait_until_tracked(rs)
            p = self.pose(rs)
            yaw_err = wrap(math.atan2(*(target - p.xy)[::-1]) - p.yaw)
            if abs(yaw_err) < tol:
                settled += 1
                if settled >= 2:
                    break
            else:
                settled = 0
            self._send_world_velocity(rs, (hold - p.xy) * self.KP_POS, float(np.clip(self.KP_YAW * yaw_err, -self.W_MAX, self.W_MAX)))
            yield
        self.stop(rs)

    # ---------------------------------------------------------- path planning
    def _cloth_box(self, margin: float):
        c = self.cloth
        return c.centre, c.u, c.v, c.half_long + margin, c.half_short + margin

    def _inside(self, p: np.ndarray, box) -> bool:
        centre, u, v, hu, hv = box
        d = p - centre
        return abs(d @ u) < hu and abs(d @ v) < hv

    def _segment_clear(self, a: np.ndarray, b: np.ndarray, box) -> bool:
        n = max(2, int(np.linalg.norm(b - a) / 0.01))
        return not any(self._inside(a + (b - a) * s, box) for s in np.linspace(0, 1, n)[1:-1])

    def plan_path(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        """Shortest way around the towel (its rectangle grown by the robot's size).

        Visibility graph over the grown rectangle's corners + Dijkstra. Good
        enough for one obstacle; a home with furniture needs a real planner.
        """
        if self.cloth is None:
            return [goal]
        margin = self.g.footprint_radius + 0.05
        box = self._cloth_box(margin)
        prefix = []
        if self._inside(start, box):  # already too close: back straight out first
            centre, u, v, hu, hv = box
            d = start - centre
            du, dv = hu - abs(d @ u), hv - abs(d @ v)
            out = start + (u * np.sign(d @ u) * (du + 0.02) if du < dv else v * np.sign(d @ v) * (dv + 0.02))
            prefix, start = [out], out
        centre, u, v, hu, hv = self._cloth_box(margin + 0.03)
        nodes = [start, goal] + [centre + su * hu * u + sv * hv * v for su in (-1, 1) for sv in (-1, 1)]
        dist = {0: 0.0}
        prev: dict[int, int] = {}
        heap = [(0.0, 0)]
        while heap:
            d, i = heapq.heappop(heap)
            if i == 1:
                break
            if d > dist.get(i, math.inf):
                continue
            for j in range(len(nodes)):
                if j != i and self._segment_clear(nodes[i], nodes[j], box):
                    nd = d + float(np.linalg.norm(nodes[j] - nodes[i]))
                    if nd < dist.get(j, math.inf):
                        dist[j], prev[j] = nd, i
                        heapq.heappush(heap, (nd, j))
        if 1 not in prev:
            return prefix + [goal]
        path, i = [], 1
        while i != 0:
            path.append(nodes[i])
            i = prev[i]
        return prefix + path[::-1]

    # -------------------------------------------------------------- cloth
    def observe_cloth(self) -> Task:
        """Look at the towel (robots must be parked clear of it)."""
        for _ in range(5):
            yield
            obs = self.cloth_detector.detect(self.image, self.cam)
            if obs is not None:
                self.cloth = obs
                self.first_cloth = self.first_cloth or obs
                self.log(
                    f"towel seen: {2*obs.half_long:.2f} x {2*obs.half_short:.2f} m, area {obs.area:.3f} m^2"
                )
                return obs
        raise FoldFailed("no towel in view")

    def plan_half_fold(self, cloth: ClothObservation) -> list[FoldAssignment]:
        """Fold the towel in half across its long side.

        Two robots stand on either long side, each grabs one corner of a short
        edge, and both carry that edge over to the opposite short edge. We fold
        toward the middle of the workspace so the result stays in camera view.
        """
        c, u, v = cloth.centre, cloth.u, cloth.v
        proj_u = (cloth.corners - c) @ u
        ends = {+1: cloth.corners[proj_u > 0], -1: cloth.corners[proj_u <= 0]}
        if any(len(e) != 2 for e in ends.values()):
            raise FoldFailed("could not find the towel's two short edges")
        # place-end = the end closer to the workspace centre (0, 0)
        place_sign = min((+1, -1), key=lambda s: np.linalg.norm(ends[s].mean(axis=0)))
        grab_pts, place_pts = ends[-place_sign], ends[place_sign]

        legs = []
        for gp in grab_pts:
            side = 1.0 if (gp - c) @ v > 0 else -1.0
            pp = place_pts[np.argmin([abs(((p - c) @ v) - ((gp - c) @ v)) for p in place_pts])]
            inward = -side * v  # robot looks across the towel
            facing = math.atan2(inward[1], inward[0])
            stage_xy = gp - inward * self.REACH
            legs.append((gp, pp, facing, Pose2D(stage_xy[0], stage_xy[1], facing)))

        # choose which robot takes which side (least total driving)
        r0, r1 = self.robots[:2]
        cost = lambda a, b: np.linalg.norm(self.pose(a).xy - legs[0][3].xy) + np.linalg.norm(self.pose(b).xy - legs[1][3].xy)
        if cost(r1, r0) < cost(r0, r1):
            r0, r1 = r1, r0
        return [FoldAssignment(r, *leg) for r, leg in zip((r0, r1), legs)]

    # ---------------------------------------------------------- manipulation
    def pick(self, rs: RobotState, corner: np.ndarray) -> Task:
        rs.status = "aiming"
        yield from self.face_point(rs, corner)
        r = float(np.linalg.norm(corner - self.pose(rs).xy))
        rs.status = "reaching"
        yield from self.move_arm(rs, r, self.HOVER_Z)
        yield from self.move_arm(rs, r, self.GRASP_Z, speed=0.1)
        rs.driver.set_gripper(True)
        yield from self.wait(0.3)
        yield from self.move_arm(rs, r, self.HOLD_Z, speed=0.15)
        ok = rs.driver.holding()
        rs.status = "holding" if ok else "missed"
        return ok

    def release_and_back_off(self, rs: RobotState, back: float = 0.15) -> Task:
        rs.driver.set_gripper(False)
        yield from self.wait(0.4)
        yield from self.move_arm(rs, rs.tip[0], self.HOVER_Z)
        p = self.pose(rs)
        goal = Pose2D(*(p.xy - p.heading * back), p.yaw)
        yield from self.drive_to(rs, goal, avoid_cloth=False, pos_tol=0.02, yaw_tol=0.05)
        yield from self.move_arm(rs, *self.HOME_TIP)
        rs.status = "idle"

    def carry(self, plan: list[FoldAssignment]) -> Task:
        """Both robots drive in lock-step, lifting the edge in an arc over the towel."""
        starts = [self.pose(a.robot).xy.copy() for a in plan]
        deltas = [a.place - a.grab for a in plan]
        length = float(np.mean([np.linalg.norm(d) for d in deltas]))
        T = max(4.0, length / 0.08)
        peak = min(0.30, 0.4 * length + self.HOLD_Z)
        reach = [a.robot.tip[0] for a in plan]
        for a in plan:
            a.robot.status = "folding"
        n = int(T / CTRL_DT)
        for k in range(1, n + 1):
            s, ds = min_jerk(k / n)
            for a, p0, d, r in zip(plan, starts, deltas, reach):
                rs = a.robot
                if self.tracker.age(rs.id, self.t) > self.STALE_LIMIT:
                    self.stop(rs)
                    continue
                p = self.pose(rs)
                v = d * ds / T + (p0 + s * d - p.xy) * self.KP_POS
                self._send_world_velocity(rs, v, float(np.clip(self.KP_YAW * wrap(a.facing - p.yaw), -self.W_MAX, self.W_MAX)))
                self.set_tip(rs, r, self.HOLD_Z + (peak - self.HOLD_Z) * math.sin(math.pi * s))
            yield
        for a in plan:
            self.stop(a.robot)
        # fine-position each robot so the corner lands on its target
        yield from parallel(
            *(self.drive_to(a.robot, Pose2D(*(p0 + d), a.facing), avoid_cloth=False, pos_tol=0.01) for a, p0, d in zip(plan, starts, deltas))
        )

    # ------------------------------------------------------------- top level
    def fold_in_half(self, label: str, attempts: int = 2) -> Task:
        for attempt in range(1, attempts + 1):
            self.phase = f"{label}: looking"
            cloth = yield from self.observe_cloth()
            plan = self.plan_half_fold(cloth)
            for a in plan:
                self.log(f"{a.robot.driver.name}: grab corner {np.round(a.grab, 3)} -> place at {np.round(a.place, 3)}")
            self.phase = f"{label}: robots to start positions"
            yield from parallel(*(self.drive_to(a.robot, a.stage) for a in plan))
            self.phase = f"{label}: grasping"
            ok = yield from parallel(*(self.pick(a.robot, a.grab) for a in plan))
            if all(ok):
                self.phase = f"{label}: folding"
                yield from self.carry(plan)
                self.phase = f"{label}: placing"
                yield from parallel(*(self.move_arm(a.robot, a.robot.tip[0], self.PLACE_Z, speed=0.1) for a in plan))
                yield from parallel(*(self.release_and_back_off(a.robot) for a in plan))
                self.log(f"{label} done")
                return True
            self.log(f"{label}: grasp failed {ok} (attempt {attempt}) - letting go and re-planning")
            yield from parallel(*(self.release_and_back_off(a.robot) for a in plan))
        raise FoldFailed(f"{label} failed after {attempts} attempts")

    def fold_towel(self, folds: int = 2) -> Task:
        while not self.cam.calibrated or len(self.tracker.poses) < len(self.robots):
            yield
        self.log("all robots located: " + ", ".join(f"{r.driver.name} at {np.round(self.pose(r).xy, 2)}" for r in self.robots))
        for i in range(folds):
            yield from self.fold_in_half(f"fold {i+1}/{folds}")
        self.phase = "checking result"
        final = yield from self.observe_cloth()
        self.phase = "done"
        return final
