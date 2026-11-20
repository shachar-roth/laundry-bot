#!/usr/bin/env python
"""Can this machine render camera images on the GPU, and how fast? Prints JSON."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from laundrybot.sim.pile import pile_scene  # noqa: E402
from laundrybot.sim.world import SimCamera, SimWorld  # noqa: E402

world = SimWorld(pile_scene(0))
cam = SimCamera(world)
cam.capture()  # warm-up
t = time.time()
for _ in range(20):
    img = cam.capture()
dt = (time.time() - t) / 20
print(json.dumps({"ok": True, "backend": os.environ["MUJOCO_GL"], "resolution": list(img.shape[:2]),
                  "ms_per_frame": round(dt * 1000, 1), "mean_pixel": round(float(img.mean()), 1)}))
