"""A crumpled pile of three garments (towel, shirt, pants).

Each garment starts laid flat in its own "spot". An invisible mocap hand (the
"dropper") then picks each one up by a random point, carries it over the
middle of the room and lets go, so the garments land tangled and crumpled
on top of each other. Building the pile takes a while to simulate, so the
result is cached on disk per seed.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .garments import PantsSpec, ShirtSpec, TowelSpec
from ..geometry import RobotGeometry
from .scene import RobotSpec, SceneSpec

# Each item has its own colour. This stands in for a real "instance
# segmentation" model that tells overlapping garments apart.
ITEMS = ["towel", "shirt", "pants"]
COLOURS = {"towel": (0.1, 0.6, 0.55, 1.0), "shirt": (0.12, 0.3, 0.8, 1.0), "pants": (0.5, 0.15, 0.7, 1.0)}
HUE_RANGES = {"towel": (78, 97), "shirt": (100, 125), "pants": (128, 155)}  # OpenCV hue, 0-180

# Spots where items start (before the pile is made) - and where the robots
# will spread them out again. The pile is made at PILE_CENTRE.
SPOTS = [(-1.65, 0.45), (1.65, 0.45), (0.0, -1.05)]
PILE_CENTRE = (0.0, 0.4)

# A bigger room needs a wider view: a 90 degree lens on a 2.6 m ceiling sees
# about 6.9 x 5.2 m of floor.
FLOOR_MARKERS = {0: (-3.1, 2.3, 0.22), 1: (3.1, 2.3, 0.22), 2: (3.1, -2.3, 0.22), 3: (-3.1, -2.3, 0.22)}


def drop_order(seed: int) -> list[int]:
    return [int(k) for k in np.random.default_rng(seed + 1000).permutation(len(ITEMS))]


def pile_scene(seed: int) -> SceneSpec:
    """Simplification: garments do NOT collide with each other (that costs
    ~500 s of CPU per simulated second here). Instead each garment rests at its
    own height - the later it was dropped, the higher - so the camera still sees
    a consistent "who is on top" order. Garments don't collide with themselves
    either: a crumpled garment's own layers stacking up would poke through the
    garments above it. They do collide with the floor and the robots."""
    rng = np.random.default_rng(seed)
    yaw = lambda: float(rng.uniform(-math.pi, math.pi))
    rank = {k: i for i, k in enumerate(drop_order(seed))}
    kw = lambda k: dict(self_collision=False, cloth_contacts=False, radius=0.004 + 0.005 * rank[k])
    garments = [
        TowelSpec(centre=SPOTS[0], yaw=yaw(), rgba=COLOURS["towel"], **kw(0)),
        ShirtSpec(centre=SPOTS[1], yaw=yaw(), rgba=COLOURS["shirt"], **kw(1)),
        PantsSpec(centre=SPOTS[2], yaw=yaw(), rgba=COLOURS["pants"], **kw(2)),
    ]
    robots = [
        RobotSpec("r0", 10, -2.3, -1.5, math.pi / 2),
        RobotSpec("r1", 11, 2.3, -1.5, math.pi / 2, colour=(0.2, 0.75, 0.3, 1)),
    ]
    return SceneSpec(
        garments=garments,
        dropper_vertices={k: int(rng.integers(g.n_vertices)) for k, g in enumerate(garments)},
        robots=robots,
        # wide lens: robots near the edge are seen from the side, so their marker
        # sits on a 40 cm mast, above the arm and any lifted cloth
        geometry=RobotGeometry(marker_mast=0.40),
        floor_markers=FLOOR_MARKERS,
        camera_fovy_deg=90.0,
        camera_res=(1920, 1440),
        view_camera='pos="0 -4.8 3.6" xyaxes="1 0 0 0 0.6 0.8" fovy="55"',
        # ~2000 cloth DOFs: the default Newton solver factorises a big matrix
        # every step (7 s per simulated second); conjugate gradient is ~20x faster.
        solver_options='solver="CG" iterations="30"',
    )


def build_pile(world, seed: int, log=print) -> None:
    """Drop the garments onto a pile, in a random order."""
    m, d = world.model, world.data
    rng = np.random.default_rng(seed + 2000)
    mocap = m.body("dropper").mocapid[0]
    order = drop_order(seed)

    def move(target, seconds):
        start = d.mocap_pos[mocap].copy()
        n = max(1, int(seconds / 0.02))
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            d.mocap_pos[mocap] = start + (np.asarray(target) - start) * s
            world.step(0.02)

    for k in order:
        eq = m.equality(f"drop{k}").id
        body = m.eq_obj1id[eq]
        d.mocap_pos[mocap] = d.xpos[body]
        m.eq_data[eq, 0:6] = 0.0
        d.eq_active[eq] = 1
        name = ITEMS[k] if k < len(ITEMS) else f"item{k}"
        log(f"pile: lifting the {name}")
        p = d.xpos[body].copy()
        move([p[0], p[1], 1.1], 1.2)
        off = rng.uniform(-0.15, 0.15, 2)
        move([PILE_CENTRE[0] + off[0], PILE_CENTRE[1] + off[1], 1.1], 1.6)
        move([PILE_CENTRE[0] + off[0], PILE_CENTRE[1] + off[1], 0.3 + rng.uniform(0, 0.2)], 0.8)
        d.eq_active[eq] = 0
        world.step(1.0)
    world.step(1.5)  # settle


def load_or_build_pile(world, seed: int, cache_dir: Path, log=print) -> None:
    path = cache_dir / f"pile_seed{seed}.npz"
    if path.exists():
        s = np.load(path)
        if s["qpos"].shape == world.data.qpos.shape:
            world.data.qpos[:] = s["qpos"]
            world.data.qvel[:] = 0
            world.data.time = 0.0
            mujoco.mj_forward(world.model, world.data)
            log(f"pile: loaded from {path.name}")
            return
    build_pile(world, seed, log)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, qpos=world.data.qpos)
    world.data.time = 0.0
    log(f"pile: built and saved to {path.name}")
