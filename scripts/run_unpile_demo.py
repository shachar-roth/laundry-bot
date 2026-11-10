#!/usr/bin/env python
"""Two arm-on-wheels robots separate a pile of 3 garments and spread each one flat.

    python scripts/run_unpile_demo.py --seed 0

Writes to output/: unpile_demo.mp4, unpile_before.png, unpile_after.png,
summary_unpile.json. The pile itself is cached in output/piles/.
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

from laundrybot.director import CTRL_DT, FoldFailed  # noqa: E402
from laundrybot.sim.pile import HUE_RANGES, ITEMS, SPOTS, load_or_build_pile, pile_scene  # noqa: E402
from laundrybot.sim.world import SimCamera, SimWorld  # noqa: E402
from laundrybot.unpile import Unpiler  # noqa: E402
from laundrybot.viz import draw_overlay  # noqa: E402


def covered_area(world, k: int, res: float = 0.005) -> float:
    """Floor area the garment covers seen from above (ground truth)."""
    v = world.vertex_positions()[:, :2]
    tris = world.elements[world.element_garment == k]
    pts = v[tris]  # T x 3 x 2
    lo = pts.reshape(-1, 2).min(axis=0) - 0.05
    hi = pts.reshape(-1, 2).max(axis=0) + 0.05
    W, H = np.ceil((hi - lo) / res).astype(int)
    canvas = np.zeros((H, W), np.uint8)
    cv2.fillPoly(canvas, [((t - lo) / res).astype(np.int32) for t in pts], 1)
    return float(canvas.sum() * res * res)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-time", type=float, default=600.0, help="simulated seconds")
    ap.add_argument("--out", type=Path, default=ROOT / "output")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    spec = pile_scene(args.seed)
    world = SimWorld(spec, seed=args.seed)
    flat = [covered_area(world, k) for k in range(len(ITEMS))]  # garments start laid flat
    load_or_build_pile(world, args.seed, ROOT / "output" / "piles")
    piled = [covered_area(world, k) for k in range(len(ITEMS))]

    camera = SimCamera(world)
    director = Unpiler(camera.K, spec.floor_markers, world.robots, spec.geometry, HUE_RANGES, SPOTS)
    task = director.unpile()

    writer = side = None
    if not args.no_video:
        import imageio.v2 as imageio
        import mujoco

        writer = imageio.get_writer(args.out / "unpile_demo.mp4", fps=int(1 / CTRL_DT) * 2, macro_block_size=8, quality=7)
        side = mujoco.Renderer(world.model, 480, 640)

    wall0, error, saved_before = time.time(), None, False
    while world.time < args.max_time:
        world.step(CTRL_DT)
        frame = camera.capture()
        director.perceive(frame, world.time)
        try:
            next(task)
        except StopIteration:
            break
        except FoldFailed as e:
            error = str(e)
            director.log(f"FAILED: {e}")
            break
        overlay = draw_overlay(director, frame)
        if not saved_before and director.views:
            cv2.imwrite(str(args.out / "unpile_before.png"), cv2.resize(overlay, (960, 720))[:, :, ::-1])
            saved_before = True
        if writer is not None:
            side.update_scene(world.data, camera="view")
            writer.append_data(np.hstack([cv2.resize(overlay, (640, 480), interpolation=cv2.INTER_AREA), side.render()]))
    else:
        error = f"ran out of time ({args.max_time}s simulated)"
        director.log(error)
    cv2.imwrite(str(args.out / "unpile_after.png"), cv2.resize(draw_overlay(director, camera.capture()), (960, 720))[:, :, ::-1])
    if writer is not None:
        writer.close()

    final = [covered_area(world, k) for k in range(len(ITEMS))]
    summary = {
        "seed": args.seed,
        "success": error is None,
        "error": error,
        "sim_seconds": round(world.time, 1),
        "wall_seconds": round(time.time() - wall0, 1),
        "flat_coverage": {  # covered floor area / area when laid perfectly flat
            n: {"in_pile": round(p / f, 2), "after": round(a / f, 2)} for n, f, p, a in zip(ITEMS, flat, piled, final)
        },
        "visible_area_history_m2": {n: [round(x, 3) for x in h] for n, h in director.history.items()},
        "events": [f"{t:6.1f}s {m}" for t, m in director.events],
    }
    (args.out / "summary_unpile.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=2))
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
