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
import itertools
import math
from dataclasses import dataclass
from typing import Callable, Generator, Iterable

import cv2
import numpy as np

from .geometry import RobotGeometry
from .interfaces import RobotDriver
from .kinematics import arm_ik
from .perception import (
    CameraModel,
    ClothDetector,
    ClothObservation,
    Pose2D,
    RobotTracker,
    detect_markers,
    garment_frame,
    pants_keypoints,
    shirt_frame,
    shirt_keypoints,
)

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
        # Robots stay inside the rectangle spanned by the floor markers: that
        # area is known to be in camera view (the markers themselves are).
        xy = np.array([(x, y) for x, y, _ in floor_markers.values()])
        self.bounds = (xy.min(axis=0), xy.max(axis=0))
        self.work_centre = xy.mean(axis=0)
        self.g = geometry
        self.robots = [RobotState(r, np.array(self.HOME_TIP, float)) for r in robots]
        self.tracker = RobotTracker(self.cam, geometry, [r.id for r in self.robots])
        self.cloth_detector = ClothDetector()
        self.cloth: ClothObservation | None = None
        self.first_cloth: ClothObservation | None = None
        self.frame = None  # GarmentFrame of a shirt / pants, found at first sight
        self.shirt = None  # (centre-line h, fold offset L) once a shirt is seen
        self.fold_line: tuple[np.ndarray, np.ndarray] | None = None  # for drawing
        self.keypoints: dict[str, np.ndarray] = {}
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
    def in_bounds(self, xy: np.ndarray, margin: float = 0.0) -> bool:
        """Is xy inside the work area marked out by the floor markers?"""
        (x0, y0), (x1, y1) = self.bounds
        return x0 + margin <= xy[0] <= x1 - margin and y0 + margin <= xy[1] <= y1 - margin

    def wait_until_tracked(self, rs: RobotState, recover_after: float = 1.0, give_up_after: float = 8.0) -> Task:
        """Pause a robot the camera can't see. If it stays lost, creep it back
        toward the middle of the work area (steering by its last known
        heading) until the camera finds it again."""
        while self.tracker.age(rs.id, self.t) > self.STALE_LIMIT:
            lost_for = self.tracker.age(rs.id, self.t)
            if lost_for > give_up_after:
                self.stop(rs)
                raise FoldFailed(f"{rs.driver.name}: lost by the camera for {lost_for:.0f} s")
            if lost_for > recover_after and rs.id in self.tracker.poses:
                rs.status = "lost - creeping back into view"
                last = self.tracker.poses[rs.id]
                to_centre = self.work_centre - last.xy
                v = to_centre / max(np.linalg.norm(to_centre), 1e-6) * 0.08
                c, s = math.cos(last.yaw), math.sin(last.yaw)
                rs.driver.set_base_velocity(c * v[0] + s * v[1], -s * v[0] + c * v[1], 0.0)
            else:
                self.stop(rs)
                rs.status = "lost - waiting for camera"
            yield
        if rs.status.startswith("lost"):
            self.stop(rs)
            self.log(f"{rs.driver.name}: back in view")
            rs.status = "found"

    def drive_to(
        self, rs: RobotState, goal: Pose2D, avoid_cloth: bool = True, avoid_robots=(), pos_tol=0.008, yaw_tol=0.02, timeout=40.0
    ) -> Task:
        """Closed-loop drive to a pose, going around the cloth (and any robots
        in avoid_robots) if needed."""
        yield from self.wait_until_tracked(rs)
        start = self.pose(rs).xy
        path = self.plan_path(start, goal.xy, avoid_robots) if avoid_cloth else [goal.xy]
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

    def _robot_box(self, other: RobotState, margin: float):
        """Keep-out square around another robot (its size + ours + margin)."""
        hw = 2 * self.g.footprint_radius + margin
        return self.pose(other).xy, np.array([1.0, 0.0]), np.array([0.0, 1.0]), hw, hw

    def _inside(self, p: np.ndarray, box) -> bool:
        centre, u, v, hu, hv = box
        d = p - centre
        return abs(d @ u) < hu and abs(d @ v) < hv

    def _segment_clear(self, a: np.ndarray, b: np.ndarray, boxes) -> bool:
        n = max(2, int(np.linalg.norm(b - a) / 0.01))
        return not any(self._inside(a + (b - a) * s, box) for s in np.linspace(0, 1, n)[1:-1] for box in boxes)

    def plan_path(self, start: np.ndarray, goal: np.ndarray, avoid: list[RobotState] = ()) -> list[np.ndarray]:
        """Shortest way around the cloth (its rectangle grown by the robot's
        size) and around other robots (squares around them).

        Visibility graph over the obstacles' corners + Dijkstra. Good enough
        for a few obstacles; a home with furniture needs a real planner.
        """
        margin = self.g.footprint_radius + 0.05
        boxes = ([self._cloth_box(margin)] if self.cloth is not None else []) + [self._robot_box(r, 0.04) for r in avoid]
        boxes = [b for b in boxes if not self._inside(goal, b)]  # never let an obstacle hide the goal
        if not boxes:
            return [goal]
        prefix = []
        for box in boxes:
            if self._inside(start, box):  # already too close: back straight out first
                centre, u, v, hu, hv = box
                d = start - centre
                du, dv = hu - abs(d @ u), hv - abs(d @ v)
                out = start + (u * np.sign(d @ u) * (du + 0.02) if du < dv else v * np.sign(d @ v) * (dv + 0.02))
                prefix.append(out)
                start = out
        nodes = [start, goal]
        for centre, u, v, hu, hv in boxes:
            nodes += [centre + su * (hu + 0.03) * u + sv * (hv + 0.03) * v for su in (-1, 1) for sv in (-1, 1)]
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
                if j != i and self._segment_clear(nodes[i], nodes[j], boxes):
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

    def drive_all(self, moves: list[tuple[RobotState, Pose2D]]) -> Task:
        """Send robots to their poses without running into each other: in
        parallel when their routes stay apart, otherwise one at a time, each
        steering around the others."""
        gap = 2 * self.g.footprint_radius + 0.1
        def seg_dist(a0, a1, b0, b1):
            pa = [a0 + (a1 - a0) * t for t in np.linspace(0, 1, 20)]
            pb = [b0 + (b1 - b0) * t for t in np.linspace(0, 1, 20)]
            return min(np.linalg.norm(p - q) for p in pa for q in pb)
        close = any(
            seg_dist(self.pose(r1).xy, g1.xy, self.pose(r2).xy, g2.xy) < gap
            for (r1, g1), (r2, g2) in itertools.combinations(moves, 2)
        )
        if not close:
            yield from parallel(*(self.drive_to(r, g) for r, g in moves))
            return
        # one at a time: whoever's route passes nearest another robot goes last
        def worst(rg):
            r, g = rg
            return min(seg_dist(self.pose(r).xy, g.xy, self.pose(o).xy, self.pose(o).xy) for o, _ in moves if o is not r)
        for r, g in sorted(moves, key=worst, reverse=True):
            yield from self.drive_to(r, g, avoid_robots=[o for o, _ in moves if o is not r])

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
                    f"cloth seen: {2*obs.half_long:.2f} x {2*obs.half_short:.2f} m, area {obs.area:.3f} m^2"
                )
                return obs
        raise FoldFailed("no towel in view")

    # ------------------------------------------------------------ fold plans
    # A planner looks at the cloth and returns (grab points, place points).
    # Every fold in this project is "pick these points up, put them there".

    def plan_half_fold(self, cloth: ClothObservation) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Fold a roughly rectangular item in half across its long side.

        Grab the two corners of one short edge and lay them on the opposite
        short edge. Fold toward the middle of the workspace so the result
        stays in camera view.
        """
        c, u, v = cloth.centre, cloth.u, cloth.v
        proj_u = (cloth.corners - c) @ u
        ends = {+1: cloth.corners[proj_u > 0], -1: cloth.corners[proj_u <= 0]}
        if any(len(e) != 2 for e in ends.values()):
            raise FoldFailed("could not find the item's two short edges")
        place_sign = min((+1, -1), key=lambda s: np.linalg.norm(ends[s].mean(axis=0)))
        grabs = list(ends[-place_sign])
        places = [ends[place_sign][np.argmin([abs((p - gp) @ v) for p in ends[place_sign]])] for gp in grabs]
        mid = (ends[-place_sign].mean(axis=0) + ends[place_sign].mean(axis=0)) / 2
        self.fold_line = (mid - v * (cloth.half_short + 0.08), mid + v * (cloth.half_short + 0.08))
        return grabs, places

    def plan_shirt_side_fold(self, cloth: ClothObservation, side: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Fold one side of a T-shirt (body panel + sleeve) in toward the middle.

        The fold line runs hem-to-collar at distance L from the centre line. L
        is chosen so the mirrored sleeve tip still lands inside the other fold
        line - then the second side folds cleanly over the first.
        """
        if self.shirt is None:
            frame = shirt_frame(cloth)
            k = shirt_keypoints(cloth.outline, frame)
            tips = frame.to_local(np.array([k.sleeve_left, k.sleeve_right]))[:, 0] - k.body_centre_h
            L = max(0.5 * k.body_half_width, float(np.max(np.abs(tips))) / 3 + 0.01)
            L = min(L, k.body_half_width - 0.05)
            self.frame, self.shirt = frame, (k.body_centre_h, L)
            hv = frame.to_local(cloth.outline)
            self.log(
                f"shirt found: collar points {np.round(frame.e_v, 2)}, body {2*k.body_half_width:.2f} m wide, "
                f"sleeve span {np.ptp(hv[:, 0]):.2f} m -> fold lines {L:.2f} m either side of centre"
            )
        frame, (hc, L) = self.frame, self.shirt
        k = shirt_keypoints(cloth.outline, frame)
        self.keypoints = {"sleeve L": k.sleeve_left, "sleeve R": k.sleeve_right, "hem L": k.hem_left, "hem R": k.hem_right}
        h0 = hc + side * L
        mirror = lambda p: p + 2 * (h0 - frame.to_local(p)[0, 0]) * frame.e_h
        grabs = [k.sleeve_left, k.hem_left] if side < 0 else [k.sleeve_right, k.hem_right]
        v = frame.to_local(cloth.outline)[:, 1]
        self.fold_line = tuple(frame.to_world(np.array([[h0, v.min() - 0.08], [h0, v.max() + 0.08]])))
        return grabs, [mirror(p) for p in grabs]

    def plan_frame_half_fold(self, cloth: ClothObservation) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Bottom corners up to the top, across the half-length line, in the
        garment's own frame (shirt: hem to collar; pants: hems to waist)."""
        frame = self.frame
        hv = frame.to_local(cloth.outline)
        h, v = hv[:, 0], hv[:, 1]
        v0 = (v.min() + v.max()) / 2
        # outline points nearest the bottom corners of the garment's bounding box
        corner = lambda hc: frame.to_world(hv[np.argmin(np.hypot(h - hc, v - v.min()))])[0]
        bl, br = corner(h.min()), corner(h.max())
        grabs = [bl, br]
        self.keypoints = {"bottom L": bl, "bottom R": br}
        self.fold_line = tuple(frame.to_world(np.array([[h.min() - 0.08, v0], [h.max() + 0.08, v0]])))
        return grabs, [p + 2 * (v0 - frame.to_local(p)[0, 1]) * frame.e_v for p in grabs]

    plan_shirt_half_fold = plan_frame_half_fold

    def plan_pants_leg_fold(self, cloth: ClothObservation) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Pants fold 1: lay the left leg on the right one, folding along the
        centre line. Grab the left waist corner and the left leg's outer hem."""
        if self.frame is None:
            self.frame = garment_frame(cloth, top="closed")
            hv = self.frame.to_local(cloth.outline)
            self.log(
                f"pants found: waist points {np.round(self.frame.e_v, 2)}, "
                f"{np.ptp(hv[:, 1]):.2f} m long, {np.ptp(hv[:, 0]):.2f} m wide"
            )
        frame = self.frame
        k = pants_keypoints(cloth.outline, frame)
        self.keypoints = {"waist L": k.waist_left, "waist R": k.waist_right, "hem L": k.hem_left, "hem R": k.hem_right}
        h0 = k.centre_h
        grabs = [k.waist_left, k.hem_left]
        v = frame.to_local(cloth.outline)[:, 1]
        self.fold_line = tuple(frame.to_world(np.array([[h0, v.min() - 0.08], [h0, v.max() + 0.08]])))
        return grabs, [p + 2 * (h0 - frame.to_local(p)[0, 0]) * frame.e_h for p in grabs]

    def _clearance(self, pts: np.ndarray) -> float:
        """Smallest distance from any of pts to the cloth (negative = on top of it)."""
        poly = self.cloth.outline.astype(np.float32).reshape(-1, 1, 2)
        return min(-cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in pts)

    def plan_fold(self, grabs: list[np.ndarray], places: list[np.ndarray]) -> list[FoldAssignment]:
        """Decide where each robot stands and which robot takes which point.

        A robot carries its point by driving sideways relative to its arm, so
        it faces perpendicular to the carry direction - on whichever side keeps
        its base off the cloth for the whole carry.
        """
        options = []  # per grab point: both possible sides to stand on
        for gp, pp in zip(grabs, places):
            d = pp - gp
            perp = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            opts = []
            for n in (perp, -perp):
                path = np.array([gp - n * self.REACH + d * s for s in np.linspace(0, 1, 8)])
                opts.append((self._clearance(path), n, path))
            options.append(opts)
        # Choose sides for all robots together: keep every base off the cloth
        # AND keep robots at least two bodies apart for the whole carry.
        gap = 2 * self.g.footprint_radius + 0.05
        def score(combo):
            clear = min(o[0] for o in combo)
            sep = min(
                (np.min(np.linalg.norm(a[2] - b[2], axis=1)) for a, b in itertools.combinations(combo, 2)), default=math.inf
            )
            return (sep >= gap, clear if sep >= gap else sep)
        combo = max(itertools.product(*options), key=score)
        if not score(combo)[0]:
            raise FoldFailed("no way to place the robots without them colliding")
        legs = []
        for gp, pp, (clearance, n, _) in zip(grabs, places, combo):
            if clearance < self.g.footprint_radius:
                self.log(f"warning: robot will pass within {clearance:.2f} m of the cloth")
            facing = math.atan2(n[1], n[0])
            stage = gp - n * self.REACH
            if not self.in_bounds(stage):
                self.log(f"warning: start position {np.round(stage, 2)} is outside the marked work area")
            legs.append((gp, pp, facing, Pose2D(stage[0], stage[1], facing)))
        robots = self.robots[: len(legs)]
        best = min(
            itertools.permutations(robots),
            key=lambda perm: sum(np.linalg.norm(self.pose(r).xy - leg[3].xy) for r, leg in zip(perm, legs)),
        )
        return [FoldAssignment(r, *leg) for r, leg in zip(best, legs)]

    # ---------------------------------------------------------- manipulation
    def pick(self, rs: RobotState, corner: np.ndarray, inset: float = 0.015) -> Task:
        """Grasp near a corner. We aim `inset` metres inside the cloth: aiming at
        the exact edge risks closing on bare floor."""
        if self.cloth is not None and inset > 0:
            inward = self.cloth.centre - corner
            corner = corner + inward / np.linalg.norm(inward) * inset
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
        while back > 0.02 and not self.in_bounds(p.xy - p.heading * back):
            back /= 2  # don't back out of the camera's view
        if back > 0.02:
            goal = Pose2D(*(p.xy - p.heading * back), p.yaw)
            yield from self.drive_to(rs, goal, avoid_cloth=False, pos_tol=0.02, yaw_tol=0.05)
        yield from self.move_arm(rs, *self.HOME_TIP)
        rs.status = "idle"

    def carry(self, plan: list[FoldAssignment]) -> Task:
        """Both robots drive in lock-step, lifting the edge in an arc over the towel."""
        starts = [self.pose(a.robot).xy.copy() for a in plan]
        deltas = [a.place - a.grab for a in plan]
        lengths = [float(np.linalg.norm(d)) for d in deltas]
        T = max(4.0, max(lengths) / 0.08)
        peaks = [min(0.30, 0.4 * ln + self.HOLD_Z) for ln in lengths]
        reach = [a.robot.tip[0] for a in plan]
        for a in plan:
            a.robot.status = "folding"
        n = int(T / CTRL_DT)
        for k in range(1, n + 1):
            s, ds = min_jerk(k / n)
            for a, p0, d, r, peak in zip(plan, starts, deltas, reach, peaks):
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
    def execute_fold(self, label: str, planner, attempts: int = 2) -> Task:
        for attempt in range(1, attempts + 1):
            self.phase = f"{label}: looking"
            cloth = yield from self.observe_cloth()
            plan = self.plan_fold(*planner(cloth))
            for a in plan:
                self.log(f"{a.robot.driver.name}: grab {np.round(a.grab, 3)} -> place at {np.round(a.place, 3)}")
            self.phase = f"{label}: robots to start positions"
            yield from self.drive_all([(a.robot, a.stage) for a in plan])
            self.phase = f"{label}: grasping"
            ok = yield from parallel(*(self.pick(a.robot, a.grab, inset=0.015 * attempt) for a in plan))
            if all(ok):
                self.phase = f"{label}: folding"
                yield from self.carry(plan)
                self.phase = f"{label}: placing"
                yield from parallel(*(self.move_arm(a.robot, a.robot.tip[0], self.PLACE_Z, speed=0.1) for a in plan))
                yield from parallel(*(self.release_and_back_off(a.robot) for a in plan))
                self.fold_line = None
                self.keypoints = {}
                self.log(f"{label} done")
                return True
            self.log(f"{label}: grasp failed {ok} (attempt {attempt}) - letting go and re-planning")
            yield from parallel(*(self.release_and_back_off(a.robot) for a in plan))
        raise FoldFailed(f"{label} failed after {attempts} attempts")

    def _start(self) -> Task:
        while not self.cam.calibrated or len(self.tracker.poses) < len(self.robots):
            yield
        self.log("all robots located: " + ", ".join(f"{r.driver.name} at {np.round(self.pose(r).xy, 2)}" for r in self.robots))

    def _finish(self) -> Task:
        self.phase = "checking result"
        self.keypoints = {}
        final = yield from self.observe_cloth()
        self.phase = "done"
        return final

    def fold_towel(self, folds: int = 2) -> Task:
        yield from self._start()
        for i in range(folds):
            yield from self.execute_fold(f"fold {i+1}/{folds}", self.plan_half_fold)
        return (yield from self._finish())

    def fold_shirt(self) -> Task:
        """Standard retail fold: left side in, right side in, then in half."""
        yield from self._start()
        yield from self.execute_fold("fold 1/3 left side + sleeve", lambda c: self.plan_shirt_side_fold(c, -1))
        yield from self.execute_fold("fold 2/3 right side + sleeve", lambda c: self.plan_shirt_side_fold(c, +1))
        yield from self.execute_fold("fold 3/3 hem to collar", self.plan_frame_half_fold)
        return (yield from self._finish())

    def fold_pants(self) -> Task:
        """Leg over leg, then hems to waist, then in half again."""
        yield from self._start()
        yield from self.execute_fold("fold 1/3 leg over leg", self.plan_pants_leg_fold)
        yield from self.execute_fold("fold 2/3 hems to waist", self.plan_frame_half_fold)
        yield from self.execute_fold("fold 3/3 in half again", self.plan_frame_half_fold)
        return (yield from self._finish())
