"""Geometry test suite: the project's accuracy contract.

@file    test_geometry.py
@brief   Verifies the ground-plane geometry for both pole-mounting configurations.
@details These are not smoke tests.  Every claim the speed pipeline makes about accuracy
         reduces to a property asserted here, so failures here must be treated as
         correctness bugs rather than numerical noise.

         The suite is organised around a small number of invariants that are true for any
         physically valid camera, and that between them pin down every sign convention in
         ``cardetect.geometry``.  That style was chosen deliberately: during development,
         several incorrect sign combinations produced matrices that were perfectly
         orthonormal and projections that agreed with ``cv2.projectPoints`` to 1e-12 px --
         because OpenCV was simply fed the same wrong matrix.  Only physical invariants
         expose that class of error, so the invariants are what is tested.

         The three primary oracles for a pitched (roadside) camera:

         1. the point where the optical axis meets the road projects to the image centre;
         2. ground points nearer the camera appear lower in the frame;
         3. a point offset toward world +X appears to the right.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cardetect.geometry import (
    AffineGroundPlane,
    CameraModel,
    Extrinsics,
    GeometryError,
    Intrinsics,
    RoadsideProjector,
    TopDownProjector,
    apply_homography,
    build_projector,
    find_homography,
    rotation_matrix,
    solve_height_from_ground_scale,
    solve_pitch_from_ground_scale,
    solve_pitch_from_horizon,
)

# @brief A realistic 720p phone-class camera: 70 degree horizontal field of view.
WIDTH, HEIGHT = 1280, 720
HFOV = 70.0


@pytest.fixture()
def intrinsics() -> Intrinsics:
    """@brief Standard 720p intrinsics used across the suite."""
    return Intrinsics.from_fov(WIDTH, HEIGHT, hfov_deg=HFOV)


# ---------------------------------------------------------------------------
# Rotation and pose invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pitch", [0.0, 10.0, 35.0, 45.0, 70.0, 89.9])
def test_rotation_is_orthonormal_and_proper(pitch: float) -> None:
    """@brief Pitched poses must be proper rotations (``det = +1``)."""
    rot = rotation_matrix(pitch)
    assert np.allclose(rot @ rot.T, np.eye(3))
    assert np.isclose(np.linalg.det(rot), 1.0)


def test_overhead_pose_is_a_documented_reflection() -> None:
    """@brief The exactly-vertical pose is a reflection, on purpose.

    @details An overhead camera must be mounted at one particular end of the track, which
             fixes the handedness of a frame in which image "down" means increasing road
             distance.  The determinant is therefore ``-1``.  This is asserted so that a
             future "fix" cannot silently mirror every overhead measurement instead.
    """
    rot = rotation_matrix(90.0)
    assert np.allclose(rot @ rot.T, np.eye(3))
    assert np.isclose(np.linalg.det(rot), -1.0)


@pytest.mark.parametrize("pitch", [10.0, 20.0, 35.0, 45.0, 70.0])
def test_optical_axis_meets_road_at_image_centre(pitch: float, intrinsics: Intrinsics) -> None:
    """@brief Oracle 1: the optical-axis/ground intersection is the principal point.

    @details The optical axis meets the road at ``h / tan(pitch)`` ahead of the camera.  That
             point must project to ``(cx, cy)``; a camera whose frame is mirrored or whose
             pitch sign is inverted fails this while remaining internally self-consistent.
    """
    model = CameraModel(intrinsics=intrinsics, extrinsics=Extrinsics(1.5, pitch))
    depth = 1.5 / math.tan(math.radians(pitch))
    point = model.ground_to_image(np.array([[0.0, depth]]))[0]
    assert point[0] == pytest.approx(intrinsics.cx, abs=1e-6)
    assert point[1] == pytest.approx(intrinsics.cy, abs=1e-6)


@pytest.mark.parametrize("pitch", [10.0, 20.0, 35.0, 45.0, 70.0])
def test_nearer_ground_appears_lower_and_plus_x_is_right(pitch: float, intrinsics: Intrinsics) -> None:
    """@brief Oracles 2 and 3: depth ordering and horizontal orientation."""
    model = CameraModel(intrinsics=intrinsics, extrinsics=Extrinsics(1.5, pitch))
    rows = [
        float(model.ground_to_image(np.array([[0.0, depth]]))[0, 1])
        for depth in (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 1e4)
    ]
    # Monotonically decreasing row means nearer ground is lower in the frame, as a real
    # camera looking down the road must show.
    assert all(rows[idx] > rows[idx + 1] for idx in range(len(rows) - 1))
    left = float(model.ground_to_image(np.array([[-1.0, 3.0]]))[0, 0])
    right = float(model.ground_to_image(np.array([[1.0, 3.0]]))[0, 0])
    assert right > left


# ---------------------------------------------------------------------------
# Homography algebra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pitch", [20.0, 35.0, 55.0, 89.9])
@pytest.mark.parametrize(("yaw", "roll"), [(0.0, 0.0), (7.0, -3.0)])
def test_closed_form_homography_matches_project_points(
    pitch: float, yaw: float, roll: float, intrinsics: Intrinsics
) -> None:
    """@brief The closed-form ground homography matches OpenCV's projection.

    @details This validates the *algebra* (that the homography really is ``K [r1 r3 t]`` for
             the stored rotation), not the sign convention, since OpenCV is handed the same
             rotation matrix.  Convention is covered by the oracles above.
    """
    cv2 = pytest.importorskip("cv2")
    extrinsics = Extrinsics(height_m=1.5, pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll)
    model = CameraModel(intrinsics=intrinsics, extrinsics=extrinsics)
    ground = np.array([[-1.0, 0.5], [0.0, 2.0], [0.8, 3.5], [0.0, 0.0], [0.5, 10.0]])
    world = np.column_stack([ground[:, 0], np.zeros(len(ground)), ground[:, 1]])
    rot_vec, _ = cv2.Rodrigues(extrinsics.rotation)
    trans_vec = -extrinsics.rotation @ extrinsics.position
    reference, _ = cv2.projectPoints(world, rot_vec, trans_vec, intrinsics.matrix, None)
    assert np.abs(model.ground_to_image(ground) - reference.reshape(-1, 2)).max() < 1e-6


@pytest.mark.parametrize("pitch", [20.0, 35.0, 55.0])
@pytest.mark.parametrize(("yaw", "roll"), [(0.0, 0.0), (7.0, -3.0), (0.0, -3.0)])
def test_horizon_row_is_the_along_road_vanishing_point(
    pitch: float, yaw: float, roll: float, intrinsics: Intrinsics
) -> None:
    """@brief ``horizon_row`` must agree with the homography's vanishing point.

    @details Deriving the horizon from the rotation and from the homography are two
             independent routes to the same quantity, so disagreement means the usable-range
             gating is computed from a different geometry than the projections it guards.
    """
    model = CameraModel(
        intrinsics=intrinsics,
        extrinsics=Extrinsics(1.5, pitch, yaw_deg=yaw, roll_deg=roll),
    )
    vanishing = model.ground_to_image_matrix()[:, 1]
    assert vanishing[2] != 0.0
    assert model.horizon_row == pytest.approx(vanishing[1] / vanishing[2], abs=1e-9)


@pytest.mark.parametrize("pitch", [10.0, 20.0, 28.0, 45.0, 70.0])
def test_horizon_identity_and_pitch_inversion(pitch: float, intrinsics: Intrinsics) -> None:
    """@brief ``horizon = cy - fy*tan(pitch)`` for an aligned pose, exactly invertible."""
    model = CameraModel(intrinsics=intrinsics, extrinsics=Extrinsics(1.4, pitch))
    expected = intrinsics.cy - intrinsics.fy * math.tan(math.radians(pitch))
    assert model.horizon_row == pytest.approx(expected, abs=1e-6)
    assert solve_pitch_from_horizon(model.horizon_row, intrinsics) == pytest.approx(pitch, abs=1e-9)


def test_solve_pitch_from_horizon_rejects_impossible_horizon(intrinsics: Intrinsics) -> None:
    """@brief A horizon at or below the principal point cannot be looking at the road."""
    with pytest.raises(GeometryError):
        solve_pitch_from_horizon(intrinsics.cy + 10.0, intrinsics)


def test_ground_image_round_trip_is_exact(intrinsics: Intrinsics) -> None:
    """@brief ``image_to_ground(ground_to_image(p)) == p`` to machine precision."""
    model = CameraModel(
        intrinsics=intrinsics,
        extrinsics=Extrinsics(1.5, 35.0, yaw_deg=7.0, roll_deg=-3.0),
    )
    ground = np.array([[-1.0, 0.5], [0.0, 2.0], [0.8, 3.5], [0.5, 10.0], [0.2, 40.0]])
    assert np.abs(model.image_to_ground(model.ground_to_image(ground)) - ground).max() < 1e-9


# ---------------------------------------------------------------------------
# Local scale: the accuracy budget
# ---------------------------------------------------------------------------


def test_analytic_jacobian_matches_numerical_derivative(intrinsics: Intrinsics) -> None:
    """@brief The analytic scale matches a tight numerical derivative.

    @details The local metres-per-pixel feeds the calibration solvers, so a finite-difference
             approximation with meaningful truncation error would bias every reported speed.
             A tight step and a central difference are used as the reference.
    """
    model = CameraModel(intrinsics=intrinsics, extrinsics=Extrinsics(1.5, 35.0))
    ground = model.ground_point_at_row(600.0)
    analytic = float(model.ground_scale(ground[None])["m_per_px_long"][0])
    eps = 1e-7
    ahead = float(model.ground_to_image(np.array([[0.0, ground[1] + eps]]))[0, 1])
    behind = float(model.ground_to_image(np.array([[0.0, ground[1] - eps]]))[0, 1])
    numerical = 1.0 / abs((ahead - behind) / (2.0 * eps))
    assert analytic == pytest.approx(numerical, rel=1e-6)


def test_roadside_error_grows_with_distance_and_overhead_does_not(intrinsics: Intrinsics) -> None:
    """@brief The central accuracy difference between the two rigs.

    @details A roadside camera's metres-per-pixel degrades monotonically with distance, which
             is why far-field detections are gated.  An overhead camera's is constant, which is
             why it is the recommended rig for a table-top track.  This test documents that
             difference as a property rather than a comment.
    """
    roadside = RoadsideProjector.from_geometry(intrinsics, height_m=1.5, pitch_deg=35.0)
    scales = [
        float(roadside.scale(np.array([[0.0, depth]]))["m_per_px_max"][0])
        for depth in (1.0, 2.0, 5.0, 10.0, 20.0)
    ]
    assert all(scales[idx] < scales[idx + 1] for idx in range(len(scales) - 1))
    assert float(roadside.scale(np.array([[0.0, 20.0]]))["anisotropy"][0]) > 1.0

    overhead = TopDownProjector.from_geometry(intrinsics, height_m=2.0)
    flat = overhead.scale(np.array([[0.0, 0.0], [5.0, 5.0]]))["m_per_px_max"]
    assert flat[0] == pytest.approx(flat[1], rel=1e-12)
    assert overhead.scale(np.array([[0.0, 0.0]]))["anisotropy"][0] == pytest.approx(1.0)


def test_min_speed_sigma_scales_with_time_and_scale(intrinsics: Intrinsics) -> None:
    """@brief The speed-noise floor reveals why frame rate and ground scale dominate.

    @details Halving the inter-frame time must double the noise floor, which is the
             quantitative statement of "a slow camera makes speed estimation hard".
    """
    model = RoadsideProjector.from_geometry(intrinsics, height_m=1.5, pitch_deg=35.0)
    ground = np.array([[0.0, 3.0]])
    slow = float(model.min_speed_sigma(ground, 0.1, 2.0)[0])
    fast = float(model.min_speed_sigma(ground, 0.05, 2.0)[0])
    assert fast == pytest.approx(2.0 * slow, rel=1e-9)
    assert slow > 0.0


# ---------------------------------------------------------------------------
# Calibration solvers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("height", "pitch", "row"),
    [(1.35, 28.0, 640.0), (1.2, 22.0, 500.0), (2.5, 40.0, 400.0), (1.5, 60.0, 500.0)],
)
def test_scale_solvers_are_exact_inverses(
    height: float, pitch: float, row: float, intrinsics: Intrinsics
) -> None:
    """@brief Height and pitch recovered from a measured scale must be exact.

    @details Both solvers invert the same forward model, which is the same code path used at
             run time.  Sharing it is what prevents a calibration from being consistent with a
             re-derived formula but inconsistent with actual measurements.
    """
    model = CameraModel(intrinsics=intrinsics, extrinsics=Extrinsics(height, pitch))
    ground = model.ground_point_at_row(row)
    scale = float(model.ground_scale(ground[None])["m_per_px_long"][0])
    assert solve_height_from_ground_scale(intrinsics, pitch, row, scale) == pytest.approx(height, abs=1e-6)
    assert solve_pitch_from_ground_scale(intrinsics, height, row, scale) == pytest.approx(pitch, abs=1e-6)


def test_affine_plane_round_trip_and_tape_scale() -> None:
    """@brief The overhead affine model round-trips and honours a tape measurement exactly."""
    plane = AffineGroundPlane.from_scale(0.004, rotation_deg=6.0, offset_uv=(639.5, 359.5))
    pixels = np.array([[100.0, 100.0], [500.0, 120.0], [900.0, 600.0]])
    assert np.abs(plane.ground_to_image(plane.image_to_ground(pixels)) - pixels).max() < 1e-9

    tape = AffineGroundPlane.from_measured_distance([400.0, 300.0], [800.0, 300.0], 0.3048)
    assert tape.m_per_px == pytest.approx(0.3048 / 400.0, rel=1e-12)
    assert tape.residuals_m[0] == pytest.approx(0.0, abs=1e-12)


def test_homography_fit_recovers_ground_control_points(intrinsics: Intrinsics) -> None:
    """@brief The ArUco / four-point calibration path reproduces its own control points."""
    model = CameraModel(intrinsics=intrinsics, extrinsics=Extrinsics(1.5, 35.0))
    ground = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0], [0.5, 1.0]])
    image = model.ground_to_image(ground)
    fitted = find_homography(image, ground)
    assert np.abs(apply_homography(fitted, image) - ground).max() < 1e-6
    projector = RoadsideProjector.from_homography(intrinsics, image, ground)
    assert np.abs(projector.to_ground(image) - ground).max() < 1e-6


def test_roadside_tape_calibration_recovers_pose(intrinsics: Intrinsics) -> None:
    """@brief The tape-measure workflow on a roadside rig recovers the true mount.

    @details The reference marks are placed 30.48 cm apart in the world, projected into the
             image, and then fed back as if they had been clicked.  The recovered height must
             be close to the truth and the measured distance must be reproduced essentially
             exactly, since the pose is refined against it by construction.
    """
    truth = CameraModel(intrinsics=intrinsics, extrinsics=Extrinsics(1.5, 35.0))
    near = truth.ground_to_image(np.array([[0.0, 1.2]]))[0]
    far = truth.ground_to_image(np.array([[0.0, 1.5048]]))[0]

    projector = RoadsideProjector.from_measured_distance(
        intrinsics, near, far, 0.3048, height_prior_m=1.5, max_error_m=0.05
    )
    recovered = projector.to_ground(np.vstack([near, far]))
    assert np.linalg.norm(recovered[1] - recovered[0]) == pytest.approx(0.3048, abs=1e-6)
    assert projector.model.extrinsics.height_m == pytest.approx(1.5, rel=0.02)
    assert projector.usable_range_m() > 0.0


def test_roadside_tape_calibration_requires_enough_information(intrinsics: Intrinsics) -> None:
    """@brief Height and pitch are not separately observable from one distance alone."""
    with pytest.raises(GeometryError):
        RoadsideProjector.from_measured_distance(
            intrinsics, [600.0, 400.0], [640.0, 380.0], 0.3048
        )


# ---------------------------------------------------------------------------
# Validity gating
# ---------------------------------------------------------------------------


def test_validity_rejects_off_frame_and_behind_camera(intrinsics: Intrinsics) -> None:
    """@brief Points that cannot be trusted metrically must be rejected.

    @details The important case is a point **above the horizon**.  A ground homography is
             defined for every pixel, so such a point maps to a finite but *negative* road
             distance -- a position behind the camera.  Nothing about that number looks wrong
             until it is differenced into a speed, which is why it is gated here.

             Note that for a real roadside rig the horizon sits above the top of the frame, so
             the horizon test alone rejects nothing inside the image and the depth test is what
             does the work.  This test therefore covers both regimes explicitly.
    """
    # A pitch shallow enough that the horizon falls *inside* the frame, so the horizon test
    # itself is exercised.
    shallow = RoadsideProjector.from_geometry(intrinsics, height_m=1.5, pitch_deg=12.0)
    assert 0.0 < shallow.horizon_row < HEIGHT, "test needs the horizon inside the frame"
    above_horizon = np.array([[intrinsics.cx, shallow.horizon_row - 5.0]])
    assert not shallow.is_valid(above_horizon).any()

    right_way_up = np.array([[intrinsics.cx, shallow.horizon_row + 50.0]])
    assert shallow.is_valid(right_way_up).any()

    # A steeper, realistic rig: the horizon is off-frame and off-frame points are rejected.
    steep = RoadsideProjector.from_geometry(intrinsics, height_m=1.5, pitch_deg=35.0)
    assert steep.horizon_row < 0.0
    points = np.array(
        [
            [639.0, 500.0],   # on the road: valid
            [639.0, -500.0],  # off frame: invalid
            [5000.0, 300.0],  # off frame: invalid
        ]
    )
    mask = steep.is_valid(points)
    assert mask[0]
    assert not mask[1:].any()


def test_error_budget_gates_the_far_field(intrinsics: Intrinsics) -> None:
    """@brief A strict error budget must drop distant detections.

    @details This is the mechanism that keeps one noisy far-field sample from dominating a
             speed estimate.  With a tight budget, a point far down the road is rejected while
             a near one is kept; loosening the budget admits the far point.
    """
    strict = RoadsideProjector.from_geometry(intrinsics, height_m=1.5, pitch_deg=35.0, max_error_m=0.01)
    loose = RoadsideProjector.from_geometry(intrinsics, height_m=1.5, pitch_deg=35.0, max_error_m=0.5)
    # @note The detector-jitter assumption is a property of the detector, not the rig, so it is
    #       set on the instance rather than passed through the geometric constructor.
    strict.pixel_sigma_px = 2.0
    loose.pixel_sigma_px = 2.0
    # @note 4 m is chosen because it sits between the two budgets' usable ranges (about 2.1 m
    #       and 6.1 m at this pitch).  Beyond roughly 8 m the ground leaves the top of the frame
    #       and the on-frame test rejects everything regardless of budget, so a further point
    #       would test the wrong mechanism.
    far_point = np.array([[intrinsics.cx, steep_row_for(strict, 4.0)]])
    assert not strict.is_valid(far_point).any()
    assert loose.is_valid(far_point).any()
    assert strict.usable_range_m() < loose.usable_range_m()


def steep_row_for(projector: RoadsideProjector, depth_m: float) -> float:
    """@brief Image row at which a given road depth projects, for use in gating tests."""
    return float(projector.to_image(np.array([[0.0, depth_m]]))[0, 1])


def test_overhead_and_roadside_factory_round_trip(intrinsics: Intrinsics) -> None:
    """@brief Both rigs survive a config save/load cycle without changing behaviour."""
    intrinsics_cfg = {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "width": WIDTH,
        "height": HEIGHT,
    }
    topdown = build_projector({"kind": "topdown", "intrinsics": intrinsics_cfg, "height_m": 2.0})
    roadside = build_projector(
        {
            "kind": "roadside",
            "intrinsics": intrinsics_cfg,
            "extrinsics": {"height_m": 1.5, "pitch_deg": 35.0},
        }
    )
    assert topdown.kind == "topdown"
    assert roadside.kind == "roadside"

    reloaded = build_projector(
        {"kind": "roadside", "intrinsics": intrinsics_cfg, "calibration": roadside.to_config()}
    )
    probe = np.array([[0.0, 0.0], [1.0, 2.0], [0.5, 5.0]])
    assert np.abs(reloaded.to_ground(probe) - roadside.to_ground(probe)).max() < 1e-9


def test_both_rigs_agree_when_the_camera_is_vertical(intrinsics: Intrinsics) -> None:
    """@brief The overhead affine and projective models describe the same projection.

    @details This is the guard against a mirrored overhead plane.  A scale-only check cannot
             catch the mirror, because ``m_per_px`` is derived from a determinant and stays
             positive either way; only comparing positions does.
    """
    affine = TopDownProjector.from_geometry(intrinsics, height_m=2.0)
    projective = CameraModel(intrinsics=intrinsics, extrinsics=Extrinsics.overhead(2.0))
    ground = np.array([[0.0, 0.0], [1.0, 1.0], [-1.5, 2.0], [3.0, -4.0]])
    recovered = affine.to_ground(projective.ground_to_image(ground))
    assert np.abs(recovered - ground).max() < 1e-9


def test_unknown_projector_kind_is_rejected() -> None:
    """@brief An unknown mounting configuration must fail loudly, not silently default."""
    with pytest.raises(GeometryError):
        build_projector({"kind": "sideways"})
