"""Unpiling: separate a pile of garments and spread each one flat.

    for each item, top-most first:
        drag it out of the pile to a free spot     (one robot)
        repeat while its flat area keeps growing:  (two robots)
            grab its two farthest-apart points
            lift, back away until the fabric is taut
            sweep sideways while lowering -> it lies down flat
            let go, look again

"Top-most" is guessed as the item showing the most fabric. Items are told
apart by colour here; a real product needs an instance-segmentation model in
the same place (`Unpiler.observe_items`).
"""
from __future__ import annotations

import math

import numpy as np

from .director import CTRL_DT, Director, FoldFailed, Pose2D, RobotState, Task, min_jerk, parallel, wrap
from .perception import ClothDetector, ClothObservation


class Unpiler(Director):
    DRAG_LIFT = 0.22  # fingertip height while dragging an item out (it mostly slides)
    DRAG_SPEED = 0.15
    DRAG_PAST = 0.30  # pull the grasped point this far past the spot centre
    SPREAD_LIFT = 0.32
    TENSION_N = 1.5  # stop backing away when the sideways pull rises this much above its start
    PULL_MAX = 0.45  # ... or after this far
    PULL_SPEED = 0.05
    SWEEP = 0.35  # sideways sweep while laying the item down
    MAX_SPREADS = 4

    def __init__(self, K, floor_markers, robots, geometry, items: dict[str, tuple[int, int]], spots, log=print):
        super().__init__(K, floor_markers, robots, geometry, log)
        self.detectors = {n: ClothDetector(hue_range=h, min_area_px=800) for n, h in items.items()}
        self.spots = [np.asarray(s, float) for s in spots]
        self.views: dict[str, ClothObservation] = {}
        self.spot_of: dict[str, int] = {}
        self.history: dict[str, list[float]] = {}  # visible area after each step

    # ------------------------------------------------------------ perception
    def observe_items(self) -> Task:
        yield
        views = {n: d.detect(self.image, self.cam, merge=True) for n, d in self.detectors.items()}
        self.views = {n: v for n, v in views.items() if v is not None}
        return self.views

    def focus(self, name: str) -> ClothObservation:
        self.cloth = self.views[name]
        self.others = [v for n, v in self.views.items() if n != name]
        return self.cloth

    def isolated(self, name: str) -> bool:
        v = self.views.get(name)
        if v is None or v.pieces > 1:
            return False
        others = [o for n, o in self.views.items() if n != name]
        return not others or self._clearance(v.outline[::3], others) > 0.03

    # --------------------------------------------------------------- helpers
    def clamp_in_bounds(self, start: np.ndarray, end: np.ndarray, margin: float = 0.1) -> np.ndarray:
        """Shorten the move start->end so it stays inside the work area."""
        for s in np.linspace(1, 0, 21):
            p = start + (end - start) * s
            if self.in_bounds(p, margin):
                return p
        return start

    def park_away(self, rs: RobotState, keep_clear: list[np.ndarray]) -> Task:
        """Move an idle robot out of the way of points another robot will pass."""
        (x0, y0), (x1, y1) = self.bounds
        parks = [np.array(p) for p in ((x0 + 0.5, y0 + 0.4), (x1 - 0.5, y0 + 0.4), (x0 + 0.5, y1 - 0.4), (x1 - 0.5, y1 - 0.4))]
        far = lambda p: min(np.linalg.norm(p - q) for q in keep_clear)
        here = self.pose(rs).xy
        if far(here) > 0.8:
            return
        best = max(parks, key=far)
        yield from self.drive_to(rs, Pose2D(best[0], best[1], self.pose(rs).yaw), pos_tol=0.05, yaw_tol=0.2)

    # ------------------------------------------------------------- drag out
    def drag_to_spot(self, name: str, spot: np.ndarray) -> Task:
        v = self.focus(name)
        pts = v.inner_points
        if pts is None or not len(pts):
            raise FoldFailed(f"{name}: no visible fabric to grab")
        grab = pts[np.argmin(np.linalg.norm(pts - spot, axis=1))]  # its edge nearest the spot
        d = spot - grab
        dirn = d / max(np.linalg.norm(d), 1e-6)
        facing = math.atan2(-dirn[1], -dirn[0])  # look back at the pile, pull by reversing
        stage = grab + dirn * self.REACH
        end = self.clamp_in_bounds(stage, spot + dirn * (self.DRAG_PAST + self.REACH))
        rs = min(self.robots, key=lambda r: np.linalg.norm(self.pose(r).xy - stage))
        other = [r for r in self.robots if r is not rs]
        self.keypoints = {f"grab {name}": grab}
        self.fold_line = (grab, end - dirn * self.REACH)
        self.log(f"{name}: {rs.driver.name} drags it from {np.round(grab, 2)} toward spot {np.round(spot, 2)}")
        yield from parallel(
            *(self.park_away(o, [stage, end, (stage + end) / 2]) for o in other),
            self.drive_to(rs, Pose2D(stage[0], stage[1], facing), avoid_robots=other),
        )
        ok = yield from self.pick(rs, grab, inset=0.0, touch=True)
        if not ok:
            self.log(f"{name}: missed the grasp")
            yield from self.release_and_back_off(rs)
            return False
        yield from self.move_arm(rs, rs.tip[0], self.DRAG_LIFT, speed=0.2)
        rs.status = "dragging"
        yield from self.drive_to(rs, Pose2D(end[0], end[1], facing), avoid_cloth=False, pos_tol=0.03, yaw_tol=0.05, v_max=self.DRAG_SPEED)
        yield from self.move_arm(rs, rs.tip[0], self.PLACE_Z, speed=0.15)
        yield from self.release_and_back_off(rs, back=0.1)
        self.fold_line, self.keypoints = None, {}
        return True

    # --------------------------------------------------------------- spread
    def plan_spread(self, v: ClothObservation):
        """Two far-apart points on the item, robots standing beyond them on the
        line through them, and the side to sweep toward. None if nothing fits."""
        hull, inner = v.outline, v.inner_points
        if inner is None or len(inner) < 2:
            return None
        dist = np.linalg.norm(hull[:, None] - hull[None], axis=2)
        pairs = np.dstack(np.unravel_index(np.argsort(-dist, axis=None), dist.shape))[0]
        snap = lambda p: inner[np.argmin(np.linalg.norm(inner - p, axis=1))]
        longest = dist.max()
        for i, j in pairs:
            if i >= j:
                continue
            if dist[i, j] < 0.6 * longest:  # only fairly far-apart pairs are worth it
                break
            p1, p2 = snap(hull[i]), snap(hull[j])
            if np.linalg.norm(p2 - p1) < 0.15:
                continue
            u = (p2 - p1) / np.linalg.norm(p2 - p1)
            stages = [p1 - u * self.REACH, p2 + u * self.REACH]
            # room to stand plus a little to back away; the pull itself stops at the work-area edge
            room = [p1 - u * (self.REACH + 0.1), p2 + u * (self.REACH + 0.1)]
            if not all(self.in_bounds(p, 0.0) for p in room):
                continue
            if self.others and self._clearance(np.array(stages), self.others) < self.g.footprint_radius:
                continue
            w = np.array([-u[1], u[0]])
            best = None
            for sgn in (1, -1):
                ends = [p + sgn * w * self.SWEEP for p in room]
                if not all(self.in_bounds(p, 0.05) for p in ends):
                    continue
                space = self._clearance(np.array(ends), self.others) if self.others else 9.0
                if best is None or space > best[0]:
                    best = (space, sgn * w)
            if best is None:
                continue
            facings = [math.atan2(u[1], u[0]), math.atan2(-u[1], -u[0])]  # each faces the item
            return [p1, p2], [Pose2D(s[0], s[1], f) for s, f in zip(stages, facings)], best[1]
        return None

    def spread_once(self, name: str) -> Task:
        v = self.focus(name)
        plan = self.plan_spread(v)
        if plan is None:
            self.log(f"{name}: no room to spread it")
            return False
        grabs, stages, sweep = plan
        robots = self.robots[:2]
        if sum(np.linalg.norm(self.pose(r).xy - s.xy) for r, s in zip(robots[::-1], stages)) < sum(
            np.linalg.norm(self.pose(r).xy - s.xy) for r, s in zip(robots, stages)
        ):
            robots = robots[::-1]
        self.keypoints = {"grab 1": grabs[0], "grab 2": grabs[1]}
        self.fold_line = (grabs[0], grabs[1])
        yield from self.drive_all(list(zip(robots, stages)))
        ok = yield from parallel(*(self.pick(r, g, inset=0.0, touch=True) for r, g in zip(robots, grabs)))
        if not all(ok):
            self.log(f"{name}: spread grasp failed {ok}")
            yield from parallel(*(self.release_and_back_off(r, back=0.1) for r in robots))
            return False
        yield from parallel(*(self.move_arm(r, r.tip[0], self.SPREAD_LIFT, speed=0.2) for r in robots))

        # back away until the fabric between the grippers is taut
        for r in robots:
            r.status = "pulling taut"
        starts = [self.pose(r).xy.copy() for r in robots]
        outward = [-self.pose(r).heading for r in robots]
        done, peak = [False, False], [0.0, 0.0]
        yield from self.wait(0.5)  # let the lifted cloth stop swinging
        base = [np.linalg.norm(r.driver.gripper_force()[:2]) for r in robots]  # its own weight
        for _ in range(int(self.PULL_MAX / self.PULL_SPEED / CTRL_DT) + 20):
            for i, r in enumerate(robots):
                if done[i]:
                    self.stop(r)
                    continue
                f = np.linalg.norm(r.driver.gripper_force()[:2]) - base[i]
                peak[i] = max(peak[i], f)
                moved = np.linalg.norm(self.pose(r).xy - starts[i])
                if f > self.TENSION_N or moved > self.PULL_MAX or not self.in_bounds(self.pose(r).xy + outward[i] * 0.05, 0.05):
                    done[i] = True
                    self.stop(r)
                    continue
                self._send_world_velocity(r, outward[i] * self.PULL_SPEED, 0.0)
            if all(done):
                break
            yield
        moved = [np.linalg.norm(self.pose(r).xy - p) for r, p in zip(robots, starts)]
        self.log(
            f"{name}: pulled taut - backed {moved[0]:.2f} / {moved[1]:.2f} m, "
            f"extra pull {peak[0]:.1f} / {peak[1]:.1f} N (own weight {base[0]:.1f} / {base[1]:.1f} N)"
        )

        # sweep sideways while lowering: the hanging part lies down flat
        for r in robots:
            r.status = "laying down"
        p0 = [self.pose(r).xy.copy() for r in robots]
        yaws = [self.pose(r).yaw for r in robots]
        T = 4.0
        n = int(T / CTRL_DT)
        for k in range(1, n + 1):
            s, ds = min_jerk(k / n)
            for r, p, yaw in zip(robots, p0, yaws):
                target = p + sweep * self.SWEEP * s
                vel = sweep * self.SWEEP * ds / T + (target - self.pose(r).xy) * self.KP_POS
                self._send_world_velocity(r, vel, float(np.clip(self.KP_YAW * wrap(yaw - self.pose(r).yaw), -1, 1)))
                self.set_tip(r, r.tip[0], self.SPREAD_LIFT + (self.PLACE_Z - self.SPREAD_LIFT) * s)
            yield
        for r in robots:
            self.stop(r)
        yield from parallel(*(self.release_and_back_off(r, back=0.12) for r in robots))
        self.fold_line, self.keypoints = None, {}
        return True

    def flatten(self, name: str) -> Task:
        views = yield from self.observe_items()
        best = views[name].area
        self.history[name].append(best)
        for i in range(self.MAX_SPREADS):
            self.phase = f"{name}: spreading it flat ({i + 1})"
            ok = yield from self.spread_once(name)
            views = yield from self.observe_items()
            area = views[name].area if name in views else 0.0
            self.history[name].append(area)
            self.log(f"{name}: visible area {best:.3f} -> {area:.3f} m^2")
            if not ok and i > 0:
                break
            if area < best * 1.05:
                break
            best = area

    # -------------------------------------------------------------- top level
    def unpile(self) -> Task:
        yield from self._start()
        views = yield from self.observe_items()
        self.log("pile: visible fabric " + ", ".join(f"{n} {v.area:.3f} m^2 ({v.pieces} piece(s))" for n, v in views.items()))
        remaining = [n for n in self.detectors]
        free = list(range(len(self.spots)))
        while remaining:
            views = yield from self.observe_items()
            seen = [n for n in remaining if n in views]
            if not seen:
                raise FoldFailed(f"can't see {remaining}")
            name = max(seen, key=lambda n: views[n].area)  # most visible = probably on top
            self.history[name] = []
            spot_i = min(free, key=lambda i: np.linalg.norm(self.spots[i] - views[name].centre))
            free.remove(spot_i)
            self.spot_of[name] = spot_i
            spot = self.spots[spot_i]
            for attempt in range(3):
                self.phase = f"{name}: pulling it out of the pile"
                yield from self.drag_to_spot(name, spot)
                views = yield from self.observe_items()
                if self.isolated(name):
                    break
                self.log(f"{name}: still touching another item (attempt {attempt + 1})")
            yield from self.flatten(name)
            remaining.remove(name)
        self.phase = "done"
        self.cloth, self.others = None, []
        return (yield from self.observe_items())
