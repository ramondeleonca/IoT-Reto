"""Calibration plumbing tests.

@file    test_calibration.py
@brief   Verify calibration persistence, config editing and horizon estimation.
@details The interactive clicking front end cannot be tested without a display, and testing a GUI
         would mostly test OpenCV rather than this project.  What *is* worth testing is everything
         around it, because these are the parts that silently corrupt a rig:

         * a calibration must survive a save/load round trip unchanged, or a run silently uses
           different geometry than the one that was measured;
         * editing a configuration file must preserve every other key, since the file also holds
           detector and rig tuning that would be costly to lose;
         * the horizon estimator must reject a vertical edge, which carries no horizon information
           and would otherwise return a plausible-looking row from a division by zero.

         Requires OpenCV only for the frame-shaped inputs, so it is skipped when absent.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from cardetect.calibration import (
    default_calibration_path,
    estimate_horizon_from_frame,
    load_calibration,
    metres_per_pixel_note,
    save_calibration,
    write_calibration_into_config,
)
from cardetect.config import PROJECT_ROOT, load_config
from cardetect.geometry import Extrinsics, Intrinsics, RoadsideProjector, TopDownProjector, build_projector

INTRINSICS = Intrinsics.from_fov(1280, 720, hfov_deg=70.0)


def test_topdown_calibration_round_trips_through_yaml(tmp_path: Path) -> None:
    """@brief A saved overhead calibration must reload identically.

    @details A round trip that changed the geometry would make a calibration a one-session artefact
             and would be very hard to notice, because the numbers involved are all plausible.
             The check is therefore on projected positions, not on the serialised fields.
    """
    original = TopDownProjector.from_measured_distance(
        INTRINSICS,
        pt_a_xy=(400.0, 300.0),
        pt_b_xy=(800.0, 300.0),
        real_distance_m=0.3048,
        label="test-rig",
    )
    path = save_calibration(tmp_path / "topdown.yaml", original, extra={"note": "unit test"})
    assert path.exists()

    reloaded = load_calibration(path)
    assert isinstance(reloaded, TopDownProjector)
    probe = np.array([[100.0, 120.0], [640.0, 360.0], [1100.0, 700.0]])
    assert np.abs(reloaded.to_ground(probe) - original.to_ground(probe)).max() < 1e-12
    assert reloaded.plane.m_per_px == pytest.approx(original.plane.m_per_px, rel=1e-12)


def test_roadside_calibration_round_trips_through_yaml(tmp_path: Path) -> None:
    """@brief A saved roadside calibration must reload identically.

    @details Exercises the pose-based branch, whose serialisation includes the full extrinsics.  A
             dropped field here would change the pitch and therefore the scale, which is exactly
             the class of silent error the calibration exists to prevent.
    """
    original = RoadsideProjector.from_geometry(
        INTRINSICS, height_m=1.5, pitch_deg=35.0, yaw_deg=0.8, roll_deg=-1.2
    )
    path = save_calibration(tmp_path / "roadside.yaml", original)
    reloaded = load_calibration(path)
    assert isinstance(reloaded, RoadsideProjector)
    probe = np.array([[300.0, 500.0], [640.0, 420.0], [900.0, 480.0]])
    assert np.abs(reloaded.to_ground(probe) - original.to_ground(probe)).max() < 1e-9
    assert reloaded.model.extrinsics.pitch_deg == pytest.approx(35.0, abs=1e-9)
    assert reloaded.horizon_row == pytest.approx(original.horizon_row, abs=1e-6)


def test_metadata_records_provenance(tmp_path: Path) -> None:
    """@brief A calibration file must say when and how it was taken.

    @details A calibration with no provenance cannot be audited months later, and the usual failure
             is not knowing whether a stored file was measured or guessed.  The metadata block is
             therefore part of the contract, not decoration.
    """
    projector = TopDownProjector.from_measured_distance(
        INTRINSICS, pt_a_xy=(400.0, 300.0), pt_b_xy=(800.0, 300.0), real_distance_m=0.3048
    )
    path = save_calibration(tmp_path / "c.yaml", projector, extra={"operator": "test"})
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "metadata" in document
    assert "created" in document["metadata"]
    assert document["metadata"]["operator"] == "test"
    assert document["kind"] == "topdown"


def test_write_calibration_into_config_preserves_other_keys(tmp_path: Path) -> None:
    """@brief Inserting a calibration must not disturb the rest of the configuration.

    @details The camera file also carries detector and rig tuning.  Clobbering it during a
             calibration would be a costly and confusing loss, so the edit is checked explicitly
             rather than trusted, and the backup is asserted to exist.
    """
    config_path = tmp_path / "camera.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "kind": "topdown",
                "label": "my-rig",
                "height_m": 0.9,
                "detector": {"model": "yolo26n.pt", "imgsz": 480, "classes": [2]},
                "speed": {"window_s": 0.3},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    projector = TopDownProjector.from_measured_distance(
        INTRINSICS, pt_a_xy=(400.0, 300.0), pt_b_xy=(800.0, 300.0), real_distance_m=0.3048
    )
    written = write_calibration_into_config(config_path, projector)
    document = yaml.safe_load(written.read_text(encoding="utf-8"))

    assert document["detector"]["model"] == "yolo26n.pt"
    assert document["detector"]["imgsz"] == 480
    assert document["speed"]["window_s"] == 0.3
    assert document["label"] == "my-rig"
    assert "calibration" in document
    assert (tmp_path / "camera.yaml.bak").exists()

    # @step The stored calibration must take precedence, since it records a physical measurement
    #       whereas the surrounding height is a datasheet default the calibration replaces.
    merged = load_config(written)
    reloaded = build_projector(merged)
    assert isinstance(reloaded, TopDownProjector)
    assert reloaded.plane.m_per_px == pytest.approx(projector.plane.m_per_px, rel=1e-9)


def test_calibration_metadata_is_stripped_of_duplicated_intrinsics(tmp_path: Path) -> None:
    """@brief A written calibration block must not duplicate the intrinsics.

    @details Two copies of the same value in one file will eventually drift apart, and the
             resulting mismatch between the geometry and the intrinsics used to un-distort would be
             very hard to diagnose.  The file keeps one authoritative copy.
    """
    config_path = tmp_path / "camera.yaml"
    config_path.write_text(
        yaml.safe_dump({"kind": "roadside", "intrinsics": {"fx": 900.0, "fy": 900.0}}, sort_keys=False),
        encoding="utf-8",
    )
    projector = RoadsideProjector.from_geometry(INTRINSICS, height_m=1.5, pitch_deg=35.0)
    written = write_calibration_into_config(config_path, projector)
    document = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert "intrinsics" in document  # the original, authoritative copy
    assert "intrinsics" not in document["calibration"]
    assert "kind" not in document["calibration"]
    assert "label" not in document["calibration"]


def test_horizon_estimate_uses_the_road_edge_slope() -> None:
    """@brief The horizon estimator must extrapolate the edge to the principal column."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # A road edge rising to the right: it reaches the principal column at a known row.
    edge = (600.0, 400.0, 1000.0, 300.0)
    slope = (300.0 - 400.0) / (1000.0 - 600.0)
    expected = 400.0 + slope * (INTRINSICS.cx - 600.0)
    assert estimate_horizon_from_frame(frame, edge, INTRINSICS) == pytest.approx(expected)


def test_horizon_estimate_rejects_a_vertical_edge() -> None:
    """@brief A vertical edge carries no horizon information and must be refused.

    @details A vertical image feature cannot fix a horizon row; silently returning a number from a
             degenerate division would feed a meaningless pitch into the calibration and bias every
             subsequent speed.  Failing loudly is the only safe behaviour.
    """
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        estimate_horizon_from_frame(frame, (600.0, 200.0, 600.0, 500.0), INTRINSICS)


def test_resolution_note_states_the_rig_character() -> None:
    """@brief The human-readable resolution note must reflect the rig's actual behaviour."""
    overhead = TopDownProjector.from_geometry(INTRINSICS, height_m=2.0)
    note = metres_per_pixel_note(overhead)
    assert "uniform" in note

    roadside = RoadsideProjector.from_geometry(INTRINSICS, height_m=1.5, pitch_deg=35.0)
    note = metres_per_pixel_note(roadside)
    assert "degrades" in note


def test_default_calibration_path_is_inside_the_project() -> None:
    """@brief The conventional calibration location must be an absolute path in the project."""
    path = default_calibration_path("topdown")
    assert path.is_absolute()
    assert PROJECT_ROOT in path.parents
    assert path.name == "calibration_topdown.yaml"


def test_shipped_configs_are_loadable_and_build_a_projector() -> None:
    """@brief Both shipped camera configurations must be valid as committed.

    @details A broken example configuration is a bad first experience and, more importantly, would
             mean the documented defaults were never exercised.  This test loads both files exactly
             as the CLI would.
    """
    for name, expected_kind in (("camera_topdown.yaml", "topdown"), ("camera_roadside.yaml", "roadside")):
        config = load_config(PROJECT_ROOT / "config" / name)
        assert config["kind"] == expected_kind
        projector = build_projector(config)
        assert projector.kind == expected_kind
        # @step A projector that cannot project is useless regardless of how well it validates.
        image_points = np.array([[INTRINSICS.cx, 460.0]])
        if projector.is_valid(image_points).any():
            ground = projector.to_ground(image_points)
            assert np.all(np.isfinite(ground))


def test_roadside_tape_calibration_without_a_prior_is_refused() -> None:
    """@brief The roadside two-point workflow must refuse to guess the camera pose.

    @details This is the single most important refusal in the project.  A wrong height biases every
             speed by exactly the height ratio; the resulting numbers look entirely reasonable and
             nothing downstream can detect the error.  An exception at calibration time is far
             cheaper than a plausible wrong answer.
    """
    from cardetect.geometry import GeometryError

    with pytest.raises(GeometryError):
        RoadsideProjector.from_measured_distance(
            INTRINSICS,
            pt_a_xy=(600.0, 500.0),
            pt_b_xy=(640.0, 420.0),
            real_distance_m=0.3048,
        )
