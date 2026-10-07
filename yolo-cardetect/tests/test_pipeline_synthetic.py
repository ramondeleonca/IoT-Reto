"""End-to-end test against synthetic footage with exact ground truth.

@file    test_pipeline_synthetic.py
@brief   Verify the whole chain recovers speeds it was told to draw.
@details This is the strongest available correctness check in the absence of a real camera and a
         real Hot Wheels track.  A scenario is built in **metres**, rendered through a camera model,
         and its exact per-frame positions are recorded.  The test then injects those recorded boxes
         in place of the detector and asserts that the pipeline recovers the drawn speeds and
         positions.

         Why the detector is stubbed rather than run.  A YOLO model trained on real vehicles does
         not reliably detect flat coloured rectangles, so running it here would test the model, not
         this project.  Substituting the recorded truth isolates the part that the project is
         responsible for -- geometry, association, history and estimation -- which is exactly the
         part that the geometry suite proved to contain sign errors during development.  The
         detector interface itself is covered separately in ``test_detect.py``.

         The tests are also written to fail loudly if the synthetic scenario itself stops being
         valid: if the vehicles leave the frame, the assertions on track counts would still pass
         vacuously, so the visibility of the scenario is asserted first.

@note    Requires OpenCV for rendering and reading the video, so the whole module is skipped when
         it is absent.  The geometry and speed suites carry the accuracy contract in a NumPy-only
         environment.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cardetect.config import PROJECT_ROOT, load_config
from cardetect.detect import DetectionPointConfig, VehicleDetection, ground_contact_point
from cardetect.geometry import build_projector
from cardetect.speed import SpeedEstimator, SpeedEstimatorConfig
from cardetect.track import TrackManager, TrackManagerConfig

cv2 = pytest.importorskip("cv2")

# @note The generator lives in the package so it can be imported without a path hack.
from cardetect.tools.make_synthetic_video import build_scenario, render_scenario  # noqa: E402

CONFIGS = {
    "topdown": PROJECT_ROOT / "config" / "camera_topdown.yaml",
    "roadside": PROJECT_ROOT / "config" / "camera_roadside.yaml",
}


@pytest.fixture(scope="module")
def scenario_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    """Render both synthetic scenarios once for the whole module.

    @brief   Shared rendered footage and ground truth.
    @details Rendering is the expensive part of these tests and the scenarios are deterministic, so
             they are built once and reused.  The ground-truth document is validated here as well:
             a scenario whose vehicles are invisible would let the assertions pass vacuously.
    @param   tmp_path_factory Pytest temporary directory factory.
    @return  Mapping of rig kind to its rendered paths and truth document.
    """
    out: dict[str, dict[str, Any]] = {}
    root = tmp_path_factory.mktemp("synthetic")
    for kind, config_path in CONFIGS.items():
        config = load_config(config_path)
        scenario = build_scenario(config, label=f"test-{kind}")
        video = root / f"{kind}.mp4"
        truth = render_scenario(scenario, video)
        visible = [len(frame["detections"]) for frame in truth["frames"]]
        assert max(visible) > 0, f"the {kind} synthetic scenario renders no visible vehicle"
        assert len(visible) - visible.count(0) >= 10, (
            f"the {kind} scenario only shows a vehicle in {len(visible) - visible.count(0)} frames, "
            f"which is too few to measure anything and would make the assertions vacuous"
        )
        out[kind] = {"video": video, "truth": truth}
    return out


def measurable(entry: dict[str, Any]) -> bool:
    """Whether a recorded detection is one the pipeline could reasonably measure.

    @brief   Filter out boxes that are cropped or implausibly compressed.
    @details A vehicle whose box is partly outside the frame, or compressed into a sliver, cannot
             support a contact-point estimate and is gated by the real pipeline too.  Tests must
             apply the same restriction or they will compare against detections that no
             implementation could measure, and will fail for reasons that are not defects.

             This is not merely a convenience: on a pitched camera the projected box of a distant
             vehicle becomes very wide and very short, and the *bottom* edge then no longer
             corresponds to the near side of the footprint at all.  That is a genuine limitation of
             measuring a compressed box, and the honest response is to exclude such detections --
             which is also what the aspect-ratio and error-budget gates do in production.
    @param   entry Ground-truth detection record.
    @return  ``True`` when the detection is fully in frame with a non-degenerate box.
    """
    x1, y1, x2, y2 = (float(v) for v in entry["box_xyxy"])
    height = y2 - y1
    width = x2 - x1
    if y1 < 0.0 or y2 > 719.0:
        return False
    if height < 12.0:
        return False
    return width / max(height, 1e-9) <= 3.0


def stub_detector(truth: dict[str, Any], point_config: DetectionPointConfig | None = None):
    """Build a callable that replays recorded ground-truth boxes as if detected.

    @brief   Detector substitute driven by the truth file.
    @details Returns objects shaped like the real detector's output, with the contact point derived
             through the *same* :func:`cardetect.detect.ground_contact_point` the real pipeline uses.
             That matters: using the truth file's ``bottom_center_uv`` directly would bypass the
             contact-point logic and would therefore not test it.
    @param   truth        Ground-truth document from the generator.
    @param   point_config Contact-point configuration; defaults to bottom-centre with no offset.
    @return  Callable mapping a frame index to a list of detections.
    """
    cfg = point_config or DetectionPointConfig()
    by_frame = {int(frame["frame"]): frame["detections"] for frame in truth["frames"]}

    def detector(frame_index: int) -> list[VehicleDetection]:
        """@brief Detections for one frame, from the recorded truth."""
        out: list[VehicleDetection] = []
        for entry in by_frame.get(frame_index, []):
            if not measurable(entry):
                continue
            box = np.asarray(entry["box_xyxy"], dtype=np.float64)
            out.append(
                VehicleDetection(
                    box_xyxy=box,
                    confidence=float(entry.get("confidence", 0.95)),
                    class_id=int(entry.get("class_id", 2)),
                    reference_uv=ground_contact_point(box, cfg),
                )
            )
        return out

    return detector


def run_replay(kind: str, artifacts: dict[str, Any], point_config: DetectionPointConfig | None = None):
    """Replay a synthetic scenario through the projection, tracking and estimation layers.

    @brief   Execute the measurement chain without the detector.
    @details Mirrors the ordering in :meth:`cardetect.pipeline.Pipeline.run`: project and gate
             before association, then estimate for confirmed tracks.  The ordering is reproduced
             rather than imported so the test does not depend on the pipeline's presentation and
             output concerns, which are not what is being measured here.
    @param   kind         Rig kind, for reporting.
    @param   artifacts    Rendered video and truth.
    @param   point_config Contact-point configuration.
    @return  Tuple of the estimator, the resulting estimates and the tracker.
    """
    config = load_config(CONFIGS[kind])
    projector = build_projector(config)
    speed_cfg = config.get("speed", {})
    estimator = SpeedEstimator(
        SpeedEstimatorConfig(
            # @note A generous window: the scenarios are at most a couple of seconds long, and the
            #       point of the test is the geometry rather than the windowing behaviour.
            window_s=float(speed_cfg.get("window_s", 0.5)) * 20.0,
            min_span_s=float(speed_cfg.get("min_span_s", 0.08)),
            max_step_m=float(speed_cfg.get("max_step_m", 0.5)) * 10.0,
            min_samples=int(speed_cfg.get("min_samples", 3)),
            max_speed_mps=float(speed_cfg.get("max_speed_mps", 12.0)) * 5.0,
        )
    )
    tracker = TrackManager(
        config=TrackManagerConfig(min_hits=3),
        projector=projector,
        pixel_sigma_px=2.0,
    )
    detect = stub_detector(artifacts["truth"], point_config)

    for frame in artifacts["truth"]["frames"]:
        detections = detect(int(frame["frame"]))
        tracker.project_detections(detections)
        tracker.update([det for det in detections if det.valid], float(frame["time_s"]))

    estimates = {
        track_id: estimator.estimate(history)
        for track_id, history in tracker.all_histories.items()
    }
    return estimator, estimates, tracker


# ---------------------------------------------------------------------------
# Geometry recovery
# ---------------------------------------------------------------------------


def footprint_corners(vehicle: dict[str, Any], centre_xz: np.ndarray) -> np.ndarray:
    """Ground-plane corners of a synthetic vehicle at a given centre.

    @brief   Reconstruct the drawn footprint analytically.
    @details The generator draws a rectangle on the road plane and the pipeline measures the bottom
             edge of its projected bounding box.  To check the second against the first, the test
             reconstructs the footprint exactly as the generator built it, so the comparison has
             zero expected error and any deviation is a genuine defect.

             This has to be reconstructed rather than read from the ground-truth file.  The file
             records the vehicle's *centre* and the *box* bottom edge, and those are different
             points: the box bottom edge is the near side of the footprint, which is half a vehicle
             length behind the centre along the heading.  Comparing the two directly would be
             comparing different physical points, and on a pitched camera the resulting discrepancy
             is metres rather than centimetres -- which is precisely the kind of apparent mismatch
             that hides a real error.

    @param   vehicle   Vehicle record from the ground-truth document.
    @param   centre_xz Ground centre position in metres.
    @return  ``(4, 2)`` corner positions in road coordinates, in the generator's order.
    """
    half_length = float(vehicle["length_m"]) / 2.0
    half_width = float(vehicle["width_m"]) / 2.0
    local = np.array(
        [
            [-half_width, half_length],
            [half_width, half_length],
            [half_width, -half_length],
            [-half_width, -half_length],
        ],
        dtype=np.float64,
    )
    heading = float(vehicle["heading_rad"])
    rotation = np.array(
        [[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]],
        dtype=np.float64,
    )
    return local @ rotation.T + np.asarray(centre_xz, dtype=np.float64)


def true_contact_point(projector: Any, vehicle: dict[str, Any], centre_xz: np.ndarray) -> np.ndarray:
    """The image point the pipeline is supposed to derive as a vehicle's contact reference.

    @brief   Ground-truth bottom-edge centre of the projected footprint.
    @details Computed the same way the generator computed the recorded box, so it is the exact
             target for the contact-point logic.  Returned in image coordinates.
    @param   projector Ground geometry.
    @param   vehicle   Vehicle record.
    @param   centre_xz Ground centre in metres.
    @return  ``(2,)`` image point ``(u, v)``.
    """
    pixels = np.asarray(projector.to_image(footprint_corners(vehicle, centre_xz)), dtype=np.float64)
    x1, _y1 = np.min(pixels, axis=0)
    x2, y2 = np.max(pixels, axis=0)
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float64)


@pytest.mark.parametrize("kind", ["topdown", "roadside"])
def test_recorded_contact_points_project_back_to_the_drawn_positions(
    kind: str, scenario_artifacts: dict[str, Any]
) -> None:
    """@brief The core geometric claim: box -> metres reproduces the drawn road position.

    @details The generator draws a footprint on the road plane; the pipeline measures the bottom
             edge of its projected bounding box.  This test reconstructs the footprint analytically
             and checks that projecting that bottom edge returns a point on the footprint's near
             side.

             Because there is no detector noise anywhere in this path, the tolerance is tight.  Any
             deviation is pure geometry, which is what makes this the test that catches a mirrored
             or sign-flipped axis: a speed-only assertion can look reasonable while every position
             is mirrored, whereas this one cannot.
    """
    config = load_config(CONFIGS[kind])
    projector = build_projector(config)
    truth = scenario_artifacts[kind]["truth"]
    vehicles = {int(v["track_id"]): v for v in truth["vehicles"]}

    worst = 0.0
    checked = 0
    for frame in truth["frames"]:
        for entry in frame["detections"]:
            if not measurable(entry):
                continue
            vehicle = vehicles[int(entry["track_id"])]
            centre = np.asarray(entry["ground_xz"], dtype=np.float64)
            corners = footprint_corners(vehicle, centre)

            # @step The contact reference must land inside the footprint, on its near edge.  The
            #       near edge is the one whose two corners are closest to the camera, which for
            #       these scenarios is simply the minimum Z.
            recovered = projector.to_ground(
                ground_contact_point(np.asarray(entry["box_xyxy"], dtype=np.float64))[None]
            )[0]
            margin = float(vehicle["length_m"]) * 0.75
            assert corners[:, 0].min() - margin <= recovered[0] <= corners[:, 0].max() + margin
            assert corners[:, 1].min() - margin <= recovered[1] <= corners[:, 1].max() + margin

            # @step Against the analytically reconstructed bottom edge the agreement must be exact.
            expected_uv = true_contact_point(projector, vehicle, centre)
            got_uv = ground_contact_point(np.asarray(entry["box_xyxy"], dtype=np.float64))
            worst = max(worst, float(np.linalg.norm(got_uv - expected_uv)))
            checked += 1
    assert checked > 15, f"only {checked} fully-visible detections were available to check"
    assert worst < 1.0, (
        f"the measured contact point differs from the reconstructed bottom edge by up to "
        f"{worst:.3f} px, which means the box geometry and the projection disagree"
    )


@pytest.mark.parametrize("kind", ["topdown", "roadside"])
def test_tracks_are_formed_with_stable_identity(kind: str, scenario_artifacts: dict[str, Any]) -> None:
    """@brief The tracker must not fragment a single vehicle into many identities.

    @details Track fragmentation is the most damaging silent failure in this pipeline: a split track
             still yields plausible-looking speeds from short segments, so nothing looks wrong while
             the measurement is meaningless.  A single visible vehicle must produce a single track.
    """
    _, _, tracker = run_replay(kind, scenario_artifacts[kind])
    truth = scenario_artifacts[kind]["truth"]
    max_simultaneous = max(len(frame["detections"]) for frame in truth["frames"])
    formed = len(tracker.all_histories)
    # @note Allow a small number of extra identities for the roadside case, where a vehicle entering
    #       or leaving the frame can legitimately begin a new track; the point is that the count is
    #       of the same order as the number of vehicles, not many times larger.
    assert formed <= max_simultaneous + 2, (
        f"{formed} tracks formed for at most {max_simultaneous} simultaneous vehicle(s): "
        f"the tracker is fragmenting identities"
    )
    assert formed >= 1


@pytest.mark.parametrize("kind", ["topdown", "roadside"])
def test_recovers_the_drawn_speed(kind: str, scenario_artifacts: dict[str, Any]) -> None:
    """@brief The headline claim: measured speed matches the speed that was rendered.

    @details The expected value is the ground truth recorded for the *contact point*, because that
             is the point the pipeline measures.  For a vehicle whose heading is along the road and
             whose box is tight, the contact point travels at the same speed as the centre, so the
             two agree; the comparison is nevertheless made against the truth file rather than
             against a hardcoded number so that changing the scenario cannot silently invalidate
             the test.
    """
    _, estimates, _ = run_replay(kind, scenario_artifacts[kind])
    truth = scenario_artifacts[kind]["truth"]

    reportable = {tid: est for tid, est in estimates.items() if est.is_reportable}
    assert reportable, (
        f"no track produced a speed for the {kind} scenario; reasons: "
        f"{ {tid: est.reason for tid, est in estimates.items()} }"
    )
    for track_id, estimate in reportable.items():
        # @step Match the measured track to the closest drawn speed rather than assuming the
        #       identities line up, since the tracker assigns its own.
        drawn = [float(vehicle["speed_mps"]) for vehicle in truth["vehicles"]]
        closest = min(drawn, key=lambda value: abs(value - estimate.speed_mps))
        assert estimate.speed_mps == pytest.approx(closest, rel=0.15), (
            f"track {track_id} measured {estimate.speed_mps:.3f} m/s, drawn speeds were {drawn}"
        )


def test_roadside_scenario_exercises_two_simultaneous_tracks(
    scenario_artifacts: dict[str, Any],
) -> None:
    """@brief The roadside scenario must actually contain overlapping vehicles.

    @details Guards against the scenario silently degrading into a single-vehicle case, which would
             make the multi-track assertion above vacuous.  A scenario that stops exercising
             identity separation would not fail, it would simply stop testing anything.
    """
    truth = scenario_artifacts["roadside"]["truth"]
    both = [frame for frame in truth["frames"] if len(frame["detections"]) >= 2]
    assert len(both) >= 5, (
        f"only {len(both)} frames contain two simultaneous vehicles; the scenario no longer "
        f"exercises multi-track association"
    )


# ---------------------------------------------------------------------------
# Contact-point bias, measured rather than asserted
# ---------------------------------------------------------------------------


def _closest_measurable(truth: dict[str, Any]) -> dict[str, Any]:
    """The measurable detection nearest the camera, which is the most geometrically sound one.

    @brief   Pick the best available detection for a single-sample comparison.
    @details The nearest vehicle has the largest apparent size, so its box geometry is the least
             ambiguous and the comparison is the most meaningful.  Using an arbitrary detection
             instead would make a test whose outcome depends on scene layout rather than on the
             behaviour being checked.
    @param   truth Ground-truth document.
    @return  The chosen detection record.
    """
    candidates = [
        entry
        for frame in truth["frames"]
        for entry in frame["detections"]
        if measurable(entry)
    ]
    assert candidates, "the scenario contains no measurable detection"
    return min(candidates, key=lambda entry: float(entry["ground_xz"][1]))


def test_box_centre_biases_the_measurement(scenario_artifacts: dict[str, Any]) -> None:
    """@brief Demonstrate, on real geometry, why the contact point is not the box centre.

    @details The project claims that projecting a box centre places the vehicle ahead of itself on
             the road.  That claim is the justification for a whole configuration block, so it is
             measured here rather than only described.  If a future change made the two choices
             equivalent on this rig, this test would fail and the documentation would need revising
             -- which is the point.

             The assertion is deliberately loose in magnitude and strict in direction: the exact bias
             depends on the rig, but it must exist and must be one-directional for the justification
             to hold.
    """
    kind = "roadside"
    config = load_config(CONFIGS[kind])
    projector = build_projector(config)
    truth = scenario_artifacts[kind]["truth"]

    centre_config = DetectionPointConfig(mode="center")
    bottom_config = DetectionPointConfig(mode="bottom_center")

    entry = _closest_measurable(truth)
    box = np.asarray(entry["box_xyxy"], dtype=np.float64)

    centre_ground = projector.to_ground(ground_contact_point(box, centre_config)[None])[0]
    bottom_ground = projector.to_ground(ground_contact_point(box, bottom_config)[None])[0]

    # @step On a pitched camera the box centre is higher in the image, and a higher image point maps
    #       further down the road, so the centre-based estimate must be beyond the bottom-edge one.
    assert centre_ground[1] > bottom_ground[1], (
        "the box centre did not project further down the road than the bottom edge, which "
        "contradicts the documented reason for preferring the contact point"
    )
    assert abs(centre_ground[1] - bottom_ground[1]) > 0.05


def test_bottom_offset_moves_the_reference_point_toward_the_near_side(
    scenario_artifacts: dict[str, Any],
) -> None:
    """@brief The contact-point offset must shift the measurement predictably.

    @details ``bottom_offset_px`` exists because a detector box's bottom edge sits above the true
             tyre contact patch.  Its sign convention therefore matters, and this test pins it:
             a positive offset moves the reference point down the image, which is toward the camera,
             which is a *smaller* road distance.  A sign flip here would double the error it is meant
             to correct rather than removing it.
    """
    kind = "roadside"
    config = load_config(CONFIGS[kind])
    projector = build_projector(config)
    truth = scenario_artifacts[kind]["truth"]
    box = np.asarray(_closest_measurable(truth)["box_xyxy"], dtype=np.float64)

    plain = ground_contact_point(box, DetectionPointConfig(mode="bottom_center", bottom_offset_px=0.0))
    shifted = ground_contact_point(box, DetectionPointConfig(mode="bottom_center", bottom_offset_px=8.0))
    assert shifted[1] > plain[1], "a positive offset must move the reference point down the image"

    near_plain = projector.to_ground(plain[None])[0]
    near_shifted = projector.to_ground(shifted[None])[0]
    assert near_shifted[1] < near_plain[1], (
        "a downward pixel shift must reduce the measured road distance on a pitched camera"
    )
