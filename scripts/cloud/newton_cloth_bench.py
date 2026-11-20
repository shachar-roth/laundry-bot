#!/usr/bin/env python
"""Benchmark: piles of 3 towel-sized cloths falling on each other, with real
cloth-cloth contact, in Newton (GPU physics on NVIDIA Warp).

    python scripts/cloud/newton_cloth_bench.py --device cuda --piles 16 --seconds 3

Many independent piles are simulated side by side in one model - that is how
a GPU gets its speed: lots of work in parallel. Prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import warp as wp

import newton


def build(piles: int, device: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    cols = max(1, int(math.ceil(math.sqrt(piles))))
    nx, ny, cell = 10, 14, 0.05  # 0.5 x 0.7 m towel, 11 x 15 particles like the MuJoCo version
    per_particle_mass = 0.15 / ((nx + 1) * (ny + 1))
    for p in range(piles):
        cx, cy = (p % cols) * 1.6, (p // cols) * 1.6  # piles 1.6 m apart: no interaction
        for k in range(3):
            yaw = float(rng.uniform(-math.pi, math.pi))
            builder.add_cloth_grid(
                pos=wp.vec3(cx + float(rng.uniform(-0.1, 0.1)), cy + float(rng.uniform(-0.1, 0.1)), 0.05 + 0.04 * k),
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=nx,
                dim_y=ny,
                cell_x=cell,
                cell_y=cell,
                mass=per_particle_mass,
                tri_ke=1.0e3,
                tri_ka=1.0e3,
                tri_kd=1.0e1,
                edge_ke=1.0e-2,
                particle_radius=0.004,
            )
    builder.color(include_bending=True)
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 1.0e1
    model.soft_contact_mu = 0.8
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if wp.is_cuda_available() else "cpu")
    ap.add_argument("--piles", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=1.0, help="simulated seconds")
    ap.add_argument("--substeps", type=int, default=10)
    args = ap.parse_args()

    wp.init()
    t0 = time.time()
    model = build(args.piles, args.device)
    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=10,
        particle_enable_self_contact=True,  # cloth touches cloth (itself and the others)
        particle_self_contact_margin=0.01,
        particle_self_contact_gap=0.005,
    )
    s0, s1 = model.state(), model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    frame_dt = 1.0 / 60.0
    dt = frame_dt / args.substeps

    def frame():
        nonlocal s0, s1
        for _ in range(args.substeps):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0

    graph = None
    if wp.get_device(args.device).is_cuda:  # record the frame once, replay it fast
        with wp.ScopedCapture(device=args.device) as cap:
            frame()
        graph = cap.graph
    else:
        frame()  # warm-up / kernel compilation
    wp.synchronize()
    setup = time.time() - t0

    n_frames = max(1, int(args.seconds / frame_dt))
    t1 = time.time()
    for _ in range(n_frames):
        wp.capture_launch(graph) if graph is not None else frame()
    wp.synchronize()
    run = time.time() - t1

    q = s0.particle_q.numpy()
    print(json.dumps({
        "device": str(wp.get_device(args.device)),
        "piles": args.piles,
        "particles": int(model.particle_count),
        "sim_seconds": round(n_frames * frame_dt, 3),
        "wall_seconds": round(run, 3),
        "setup_seconds": round(setup, 1),
        "wall_per_sim_second_per_pile": round(run / (n_frames * frame_dt) / args.piles, 4),
        "finite": bool(np.isfinite(q).all()),
        "min_z": round(float(q[:, 2].min()), 4),
        "max_z": round(float(q[:, 2].max()), 4),
    }))


if __name__ == "__main__":
    main()
