# laundry-bot

Simulated "arms on wheels" that fold laundry, guided by one ceiling camera.

![overview](output/after.png)

Two mobile robots and a towel on the floor. The ceiling camera locates
everything; a director program plans the fold and steers every robot in a
closed loop. Today it folds a towel in half twice.

## Run it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/run_fold_demo.py              # ~5 min, writes output/fold_demo.mp4
python scripts/run_fold_demo.py --fast       # no cloth self-collision, ~3x faster
python scripts/run_fold_demo.py --seed 4     # different towel pose / wheel slip
python -m pytest tests
```

On a headless Linux machine you need an OpenGL backend, e.g.
`sudo apt install libosmesa6` (MuJoCo then renders with `MUJOCO_GL=osmesa`, the default here).

## How it is organised

```
laundrybot/
  interfaces.py   Camera + RobotDriver: the only things the brain talks to
  director.py     the brain: perception loop, fold planning, path planning, motion
  perception.py   camera self-calibration, robot tracking, towel detection (OpenCV)
  kinematics.py   arm inverse kinematics
  geometry.py     robot dimensions (shared by sim and brain)
  viz.py          draws the brain's beliefs over the camera image
  sim/scene.py    builds the MuJoCo world (cloth, robots, markers, cameras)
  sim/world.py    simulated Camera + RobotDriver (with wheel slip)
scripts/run_fold_demo.py
tests/
```

**The key design rule:** `director.py` never reads simulator state. It only gets
camera images and sends velocity / joint / gripper commands through
`interfaces.py`. To run on hardware you write a `RobotDriver` for a real robot
(e.g. LeKiwi base + SO-101 arm via Hugging Face LeRobot) and a `Camera` for a
real webcam; the director stays the same.

### What happens in a fold

1. **Calibrate:** solve the camera's pose from 4 ArUco markers on the floor
   (0.15 px reprojection error). No measuring where the camera hangs.
2. **Track:** each robot has a marker on its back; position ~1 mm, heading ~0.4°.
3. **See the towel:** colour segmentation, then its outline and four corners.
4. **Plan:** fold across the long side, toward the middle of the room (keeps
   everything in camera view). One robot per side, each takes one corner.
5. **Drive:** shortest path around the towel, closed-loop on the camera. The
   simulated wheels slip ±8% and drift sideways, so driving blind would miss.
6. **Grasp:** turn to face the corner, lower, pinch, lift; check the gripper
   is actually holding.
7. **Fold:** both robots drive in lock-step while lifting the edge in an arc,
   fine-position, lower, release, back off. On a failed grasp, let go and re-plan.
8. Repeat.

## Honest limitations (what's simplified)

| Simplification | Reality | Plan |
|---|---|---|
| "Magic pinch" gripper: closing grabs any cloth within 2.5 cm | Pinching thin fabric off a surface is the hardest physical bit | Prototype gripper on real hardware early; learn grasping from teleoperated demos |
| Solid blue towel, plain floor; colour threshold finds it | Patterned clothes, on a bed, in a pile | Swap `ClothDetector` for a learned segmenter (train on synthetic renders from this sim + real photos) |
| Towel starts laid flat | Laundry arrives crumpled | Flattening / unfolding is its own (harder) task, next big milestone |
| Robots don't avoid each other; one obstacle (the towel) | Furniture, pets, each other | Proper multi-robot planner |
| Arm reaches forward only (planar, 3 pitch joints) | SO-101 also has pan + wrist roll | Add when needed; the base turning covers it for now |
| Cloth simulated as stretch-free mesh, no bending stiffness | Real fabrics vary a lot | Fine for planning; real behaviour comes from real data |
