import math
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import pytest

from laundrybot.geometry import DEFAULT_GEOMETRY as G
from laundrybot.kinematics import Unreachable, arm_fk, arm_ik
from laundrybot.perception import CameraModel, ClothDetector, RobotTracker, detect_markers
from laundrybot.sim.scene import RobotSpec, SceneSpec, TowelSpec
from laundrybot.sim.world import SimCamera, SimWorld


@pytest.mark.parametrize("r,z", [(0.52, 0.006), (0.45, 0.25), (0.6, 0.1), (0.4, 0.3)])
def test_ik_roundtrip(r, z):
    q = arm_ik(G, r, z)
    assert np.allclose(arm_fk(G, q), (r, z), atol=1e-9)
    assert abs(q.sum()) < 1e-9  # finger points straight down


def test_ik_unreachable():
    with pytest.raises(Unreachable):
        arm_ik(G, 1.5, 0.0)


@pytest.fixture(scope="module")
def sim():
    spec = SceneSpec(
        towel=TowelSpec(centre=(0.05, -0.03), yaw=0.3, self_collision=False),
        robots=[RobotSpec("r0", 10, -0.9, -0.75, math.pi / 2), RobotSpec("r1", 11, 0.9, -0.6, 2.0)],
    )
    world = SimWorld(spec)
    world.step(0.3)
    cam = SimCamera(world)
    return spec, world, cam, cam.capture()


def test_camera_calibration_and_tracking(sim):
    spec, world, cam, img = sim
    det = detect_markers(img)
    model = CameraModel(cam.K)
    assert model.calibrate(det, spec.floor_markers)
    assert model.reprojection_error_px < 0.5
    tracker = RobotTracker(model, spec.geometry, [10, 11])
    tracker.update(det, 0.0)
    for r in world.robots:
        est, true = tracker.poses[r.marker_id], r.true_pose()
        assert math.hypot(est.x - true.x, est.y - true.y) < 0.005
        assert abs(math.remainder(est.yaw - true.yaw, 2 * math.pi)) < math.radians(1.5)


def test_cloth_corners(sim):
    spec, world, cam, img = sim
    model = CameraModel(cam.K)
    model.calibrate(detect_markers(img), spec.floor_markers)
    obs = ClothDetector().detect(img, model)
    v = world.vertex_positions()[:, :2]
    true_corners = v[[0, 14, 150, 164]]
    for c in obs.corners:
        assert np.min(np.linalg.norm(true_corners - c, axis=1)) < 0.01
    assert 2 * obs.half_long == pytest.approx(0.7, abs=0.02)
    assert 2 * obs.half_short == pytest.approx(0.5, abs=0.02)


def test_gripper_pinches_only_nearby_cloth(sim):
    _, world, _, _ = sim
    r = world.robots[0]
    r.set_gripper(True)  # tip is far from the towel
    assert not r.holding()
    r.set_gripper(False)
