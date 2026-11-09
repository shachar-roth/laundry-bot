#!/usr/bin/env python
"""Two arm-on-wheels robots fold laundry, guided only by a ceiling camera.

    python scripts/run_fold_demo.py --seed 3                  # towel, folded in half twice
    python scripts/run_fold_demo.py --garment shirt --seed 3  # T-shirt, retail fold
    python scripts/run_fold_demo.py --garment pants --seed 3  # trousers

Writes to output/: fold_demo.mp4 (camera view with the director's overlay +
a side view), before/after images, and summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from laundrybot.director import CTRL_DT, Director, FoldFailed  # noqa: E402
from laundrybot.sim.scene import PantsSpec, RobotSpec, SceneSpec, ShirtSpec, TowelSpec  # noqa: E402
from laundrybot.sim.world import SimCamera, SimWorld  # noqa: E402
from laundrybot.viz import draw_overlay  # noqa: E402


def make_scene(seed: int, garment: str = "towel", fast: bool = False) -> SceneSpec:
    rng = np.random.default_rng(seed)
    centre = tuple(rng.uniform(-0.1, 0.1, 2))
    if garment == "shirt":  # any orientation: the director has to work out where the collar is
        item = ShirtSpec(centre=centre, yaw=float(rng.uniform(-math.pi, math.pi)), self_collision=not fast)
    elif garment == "pants":  # 1 m long: keep it near the middle so robots at both ends stay in view
        item = PantsSpec(centre=tuple(np.asarray(centre) / 2), yaw=float(rng.uniform(-math.pi, math.pi)), self_collision=not fast)
    else:
        item = TowelSpec(centre=centre, yaw=float(rng.uniform(-0.45, 0.45)), self_collision=not fast)
    robots = [
        RobotSpec("r0", 10, -0.95 + rng.uniform(-0.1, 0.1), -0.8, math.pi / 2 + rng.uniform(-0.5, 0.5)),
        RobotSpec("r1", 11, 0.95 + rng.uniform(-0.1, 0.1), -0.8, math.pi / 2 + rng.uniform(-0.5, 0.5), colour=(0.2, 0.75, 0.3, 1)),
    ]
    return SceneSpec(garment=item, robots=robots)


def footprint_area(xy: np.ndarray) -> float:
    hull = cv2.convexHull(xy.astype(np.float32))
    return float(cv2.contourArea(hull))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--garment", choices=["towel", "shirt", "pants"], default="towel")
    ap.add_argument("--folds", type=int, default=2, help="towel only")
    ap.add_argument("--max-time", type=float, default=240.0, help="simulated seconds")
    ap.add_argument("--out", type=Path, default=ROOT / "output")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--fast", action="store_true", help="no cloth self-collision (~3x faster, layers pass through each other)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    spec = make_scene(args.seed, args.garment, args.fast)
    world = SimWorld(spec, seed=args.seed)
    camera = SimCamera(world)
    side_view = None if args.no_video else SimCamera(world, "view")
    director = Director(camera.K, spec.floor_markers, world.robots, spec.geometry)
    task = {"shirt": director.fold_shirt, "pants": director.fold_pants}.get(args.garment, lambda: director.fold_towel(args.folds))()

    writer = None
    if not args.no_video:
        import imageio.v2 as imageio

        writer = imageio.get_writer(args.out / f"fold_demo_{args.garment}.mp4", fps=int(1 / CTRL_DT) * 2, macro_block_size=8, quality=7)
    side_renderer = None
    if side_view is not None:
        import mujoco

        side_renderer = mujoco.Renderer(world.model, 480, 640)

    true_area0 = footprint_area(world.vertex_positions()[:, :2])
    track_err, track_yaw_err = [], []
    wall0 = time.time()
    result, error, final = None, None, None
    first_frame = None
    while world.time < args.max_time:
        world.step(CTRL_DT)
        frame = camera.capture()
        director.perceive(frame, world.time)
        for r in world.robots:
            est = director.tracker.poses.get(r.marker_id)
            if est is not None and director.tracker.age(r.marker_id, world.time) < 1e-6:
                tp = r.true_pose()
                track_err.append(float(np.hypot(est.x - tp.x, est.y - tp.y)))
                track_yaw_err.append(abs(math.remainder(est.yaw - tp.yaw, 2 * math.pi)))
        try:
            next(task)
        except StopIteration as e:
            result = final = e.value
        except FoldFailed as e:
            error = str(e)
            director.log(f"FAILED: {e}")
        overlay = draw_overlay(director, frame)
        if first_frame is None and director.cloth is not None:
            first_frame = overlay
            cv2.imwrite(str(args.out / f"before_{args.garment}.png"), overlay[:, :, ::-1])
        if writer is not None:
            side_renderer.update_scene(world.data, camera="view")
            side = side_renderer.render()
            small = cv2.resize(overlay, (640, 480), interpolation=cv2.INTER_AREA)
            writer.append_data(np.hstack([small, side]))
        if result is not None or error is not None:
            break
    else:
        error = f"ran out of time ({args.max_time}s simulated)"
        director.log(error)

    cv2.imwrite(str(args.out / f"after_{args.garment}.png"), draw_overlay(director, camera.capture())[:, :, ::-1])
    if writer is not None:
        writer.close()

    verts = world.vertex_positions()
    true_area1 = footprint_area(verts[:, :2])
    summary = {
        "garment": args.garment,
        "seed": args.seed,
        "success": error is None,
        "error": error,
        "sim_seconds": round(world.time, 1),
        "wall_seconds": round(time.time() - wall0, 1),
        "footprint_m2": {"before": round(true_area0, 4), "after": round(true_area1, 4)},
        "footprint_ratio": round(true_area1 / true_area0, 3),
        "ideal_ratio": round(0.5**args.folds, 3) if args.garment == "towel" else None,
        "final_size_m": [round(2 * final.half_long, 3), round(2 * final.half_short, 3)] if final is not None else None,
        "max_cloth_height_m": round(float(verts[:, 2].max()), 3),
        "camera_reprojection_error_px": director.cam.reprojection_error_px,
        "tracking_error_mm": {"mean": round(1000 * float(np.mean(track_err)), 2), "max": round(1000 * float(np.max(track_err)), 2)},
        "heading_error_deg": {"mean": round(math.degrees(float(np.mean(track_yaw_err))), 2), "max": round(math.degrees(float(np.max(track_yaw_err))), 2)},
        "events": [f"{t:6.1f}s {m}" for t, m in director.events],
    }
    (args.out / f"summary_{args.garment}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=2))
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
