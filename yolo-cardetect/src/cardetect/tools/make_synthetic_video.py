"""Synthetic test-video generator with exact ground truth.

@file    make_synthetic_video.py
@brief   Render vehicles onto a virtual road with known positions and speeds.
@details The value of this tool is that it makes the whole geometric chain falsifiable without a
         camera, a model or a track.  It builds a scenario in **metres**, projects it through a
         camera model, draws it, and records the exact ground-truth position of every vehicle in
         every frame.  A test can then inject those boxes past the detector and check that the
         pipeline recovers the speeds it was told to draw.

         That ordering matters.  The alternative -- recording a real video and asserting whatever
         the pipeline produces -- cannot distinguish a correct implementation from a consistently
         wrong one, and would have missed every sign error found during development of the geometry
         module.

         Two deliberate simplifications, both stated because they bound what this tool can prove:

         * Vehicles are drawn as flat rectangles on the road plane, not as 3D shapes.  The pipeline
           recovers the *bottom edge* of a detection as the contact point, so a rectangle exercises
           exactly the geometry that matters and nothing else.  It cannot validate behaviour on the
           silhouettes of real cars seen at an angle, where the visible bottom edge is not the
           contact patch.
         * The rendering has no lens distortion even when the camera model does.  Distortion is
           tested separately and directly in the geometry suite, where the expected answer is
           analytic; mixing it into a rendered scene would make a failure harder to localise.

@note    Run as a module:

             uv run python -m cardetect.tools.make_synthetic_video --help

         or as a script:

             uv run python tools/make_synthetic_video.py --config config/camera_topdown.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cardetect.config import PROJECT_ROOT, load_config
from cardetect.geometry import (
    Extrinsics,
    Intrinsics,
    RoadsideProjector,
    TopDownProjector,
    build_projector,
)

__all__ = ["VehicleTrack", "Scenario", "render_scenario", "build_scenario", "main"]


@dataclass
class VehicleTrack:
    """One synthetic vehicle and its motion.

    @brief   Ground-truth trajectory of a rendered vehicle.
    @details Motion is specified on the road plane in metres, never in pixels, so the ground truth
             is independent of the camera.  The same scenario can therefore be rendered through
             both mounting configurations and compared, which is how the two rigs' accuracy claims
             are checked against each other.

    @param  track_id    Identifier, also used to pick the drawn colour.
    @param  start_xz    Starting road position ``(X, Z)`` in metres.
    @param  speed_mps   Speed along the road.
    @param  heading_rad Direction of travel in the ground plane.
    @param  length_m    Vehicle length along its heading.
    @param  width_m     Vehicle width across its heading.
    @param  class_id    COCO class index written into the ground-truth file.
    @param  start_s     Time at which the vehicle enters the scene.
    @param  end_s       Optional time at which it leaves; ``None`` runs to the end.
    @param  acceleration_mps2 Constant acceleration, so a scenario need not be constant-velocity.
    """

    track_id: int
    start_xz: tuple[float, float]
    speed_mps: float
    heading_rad: float = 0.0
    length_m: float = 0.075
    width_m: float = 0.032
    class_id: int = 2
    start_s: float = 0.0
    end_s: float | None = None
    acceleration_mps2: float = 0.0

    def position_at(self, time_s: float) -> np.ndarray | None:
        """Road position at a given time, or ``None`` when the vehicle is off-scene.

        @brief   Evaluate the trajectory.
        @details Constant acceleration is integrated analytically rather than stepped, so the
                 ground truth is exact at every sampled time and does not depend on the chosen
                 frame rate.  A numerically integrated path would drift with the sampling and
                 would silently loosen every tolerance built on it.
        @param   time_s Scene time in seconds.
        @return  ``(2,)`` road position in metres, or ``None`` when out of scene.
        """
        if time_s < self.start_s:
            return None
        if self.end_s is not None and time_s > self.end_s:
            return None
        elapsed = time_s - self.start_s
        travelled = self.speed_mps * elapsed + 0.5 * self.acceleration_mps2 * elapsed * elapsed
        direction = np.array([math.sin(self.heading_rad), math.cos(self.heading_rad)])
        return np.asarray(self.start_xz, dtype=np.float64) + direction * travelled


@dataclass
class Scenario:
    """A complete synthetic scene: camera, vehicles and timing.

    @brief   Everything needed to render ground-truth footage.
    @param  projector  Camera geometry the scene will be rendered through.
    @param  vehicles   Vehicles to render.
    @param  fps        Frame rate.
    @param  duration_s Scene length in seconds.
    @param  size       ``(width, height)`` output frame size in pixels.
    @param  label      Human-readable scenario name, recorded in the metadata.
    """

    projector: Any
    vehicles: list[VehicleTrack] = field(default_factory=list)
    fps: float = 30.0
    duration_s: float = 3.0
    size: tuple[int, int] = (1280, 720)
    label: str = "synthetic"

    @property
    def frame_count(self) -> int:
        """@brief Number of frames this scenario will render."""
        return int(round(self.duration_s * self.fps))

    def frame_times(self) -> np.ndarray:
        """@brief Timestamps of every rendered frame, in seconds."""
        return np.arange(self.frame_count, dtype=np.float64) / float(self.fps)


def _corner_points(vehicle: VehicleTrack, centre_xz: np.ndarray) -> np.ndarray:
    """Compute the four ground-plane corners of a vehicle.

    @brief   Rectangle footprint on the road, in metres.
    @details The footprint is built in the vehicle's own frame and rotated by its heading, so a
             vehicle crossing the road at an angle -- which a real one does -- is represented
             correctly rather than being axis-aligned.
    @param   vehicle   Vehicle definition.
    @param   centre_xz Ground centre position in metres.
    @return  ``(4, 2)`` corner positions in road coordinates.
    """
    half_length = vehicle.length_m / 2.0
    half_width = vehicle.width_m / 2.0
    local = np.array(
        [
            [-half_width, half_length],
            [half_width, half_length],
            [half_width, -half_length],
            [-half_width, -half_length],
        ],
        dtype=np.float64,
    )
    cos_h, sin_h = math.cos(vehicle.heading_rad), math.sin(vehicle.heading_rad)
    # @note Ground-plane rotation about the vertical: X across the road, Z along it.
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
    return local @ rotation.T + centre_xz


def render_scenario(
    scenario: Scenario,
    output_video: str | Path,
    ground_truth_path: str | Path | None = None,
    background: tuple[int, int, int] = (48, 48, 48),
    road_color: tuple[int, int, int] = (70, 70, 72),
) -> dict[str, Any]:
    """Render a scenario to a video file plus a ground-truth sidecar.

    @brief   Produce reproducible test footage with exact truth.
    @details The ground-truth file is the point of the exercise.  It records, for every frame, the
             exact image box and exact road position of every visible vehicle, which lets a test
             inject the boxes past the detector and assert that the pipeline's recovered positions
             and speeds match what was drawn.  The video itself is useful for a human to watch.

             Boxes are computed by projecting the ground-plane footprint and taking the axis-aligned
             bounds of the projected corners.  That is the correct model for this pipeline, because
             the pipeline takes the bottom edge of a box as the road contact reference: rendering
             an axis-aligned rectangle on the ground plane means the projected bottom edge really is
             the near edge of the footprint, so the geometry being exercised is the geometry that
             matters.

    @param   scenario         Scene to render.
    @param   output_video     Destination video path.
    @param   ground_truth_path Destination JSON path, or ``None`` to derive it from the video name.
    @param   background       BGR colour for the area outside the road.
    @param   road_color       BGR colour for the road surface.
    @return  The ground-truth document that was written.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("rendering requires OpenCV: uv sync --extra vision") from exc

    video_path = Path(output_video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path = Path(ground_truth_path) if ground_truth_path else video_path.with_suffix(".json")

    width, height = scenario.size
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), scenario.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open a video writer for {video_path}")

    truth: dict[str, Any] = {
        "label": scenario.label,
        "fps": scenario.fps,
        "duration_s": scenario.duration_s,
        "size": [width, height],
        "kind": scenario.projector.kind,
        "geometry": scenario.projector.to_config(),
        "vehicles": [
            {
                "track_id": vehicle.track_id,
                "start_xz": list(vehicle.start_xz),
                "speed_mps": vehicle.speed_mps,
                "heading_rad": vehicle.heading_rad,
                "length_m": vehicle.length_m,
                "width_m": vehicle.width_m,
                "class_id": vehicle.class_id,
                "start_s": vehicle.start_s,
                "end_s": vehicle.end_s,
                "acceleration_mps2": vehicle.acceleration_mps2,
            }
            for vehicle in scenario.vehicles
        ],
        "frames": [],
    }

    # @step Precompute the visible road extent so the drawn surface matches the geometry the
    #       pipeline will use.  Rendering a road that does not correspond to the calibration would
    #       make the footage misleading rather than merely decorative.
    road_polygon = _road_polygon(scenario.projector, extent_m=12.0, half_width_m=4.0)

    for frame_index, time_s in enumerate(scenario.frame_times()):
        image = np.full((height, width, 3), background, dtype=np.uint8)
        if road_polygon is not None:
            cv2.fillPoly(image, [road_polygon], road_color)
            _draw_lane_markings(image, scenario.projector)

        frame_truth: list[dict[str, Any]] = []
        for vehicle in scenario.vehicles:
            centre = vehicle.position_at(float(time_s))
            if centre is None:
                continue
            corners = _corner_points(vehicle, centre)
            try:
                pixels = np.asarray(scenario.projector.to_image(corners), dtype=np.float64)
            except Exception:
                continue
            if not np.all(np.isfinite(pixels)):
                continue
            x1, y1 = np.min(pixels, axis=0)
            x2, y2 = np.max(pixels, axis=0)
            # @step Skip anything wholly off-frame so the truth file only lists what a detector
            #       could plausibly have found.
            if x2 < 0 or x1 > width - 1 or y2 < 0 or y1 > height - 1:
                continue
            colour = _vehicle_colour(vehicle.track_id)
            polygon = np.round(pixels).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(image, [polygon], colour)
            cv2.polylines(image, [polygon], True, (245, 245, 245), 1, cv2.LINE_AA)
            frame_truth.append(
                {
                    "track_id": vehicle.track_id,
                    "class_id": vehicle.class_id,
                    "ground_xz": [float(centre[0]), float(centre[1])],
                    # @note The contact reference the pipeline will derive from the box, recorded
                    #       explicitly so a test can compare like with like rather than guessing.
                    "bottom_center_uv": [float((x1 + x2) / 2.0), float(y2)],
                    "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "speed_mps": float(
                        vehicle.speed_mps + vehicle.acceleration_mps2 * (time_s - vehicle.start_s)
                    ),
                    "confidence": 0.95,
                }
            )

        truth["frames"].append({"frame": frame_index, "time_s": float(time_s), "detections": frame_truth})
        cv2.putText(
            image,
            f"{scenario.label}  frame {frame_index}  t={time_s:.3f}s",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{scenario.label}  frame {frame_index}  t={time_s:.3f}s",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(image)

    writer.release()
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def _vehicle_colour(track_id: int) -> tuple[int, int, int]:
    """@brief Distinct BGR fill per synthetic vehicle."""
    palette = (
        (60, 200, 255),
        (255, 160, 60),
        (120, 255, 120),
        (255, 120, 220),
        (120, 220, 255),
    )
    return palette[int(track_id) % len(palette)]


def _road_polygon(projector: Any, extent_m: float, half_width_m: float):
    """Project the road surface outline into image space.

    @brief   Image polygon approximating the visible road.
    @details Sampled along the road rather than projected as a quadrilateral, because a roadside
             homography maps the far edge of a rectangle to a curve once perspective is strong; a
             four-corner fill would misrepresent the surface the calibration describes.
    @param   projector  Ground geometry.
    @param   extent_m   How far down the road to draw.
    @param   half_width_m How far across.
    @return  ``(N, 1, 2)`` integer polygon, or ``None`` when nothing is visible.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:  # pragma: no cover - optional dependency
        return None

    depths = np.linspace(0.05, extent_m, 60)
    left = np.column_stack([np.full_like(depths, -half_width_m), depths])
    right = np.column_stack([np.full_like(depths, half_width_m), depths])
    try:
        left_px = np.asarray(projector.to_image(left), dtype=np.float64)
        right_px = np.asarray(projector.to_image(right), dtype=np.float64)
    except Exception:
        return None
    outline = np.vstack([left_px, right_px[::-1]])
    if not np.all(np.isfinite(outline)):
        return None
    return np.round(outline).astype(np.int32).reshape(-1, 1, 2)


def _draw_lane_markings(image: np.ndarray, projector: Any, spacing_m: float = 0.5) -> None:
    """Draw dashed centre-line markings so the rendered road has visible scale.

    @brief   Add metric reference marks to the drawn road.
    @details A featureless surface gives a human reviewer no way to judge whether the rendered
             geometry is plausible, and gives no reference for spotting a mirrored or scaled
             projection.  Dashes at a known spacing make a wrong calibration visible by eye.
    @param   image      BGR frame, modified in place.
    @param   projector  Ground geometry.
    @param   spacing_m  Dash period in metres.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        return
    for start_m in np.arange(0.25, 12.0, spacing_m):
        samples = np.array([[0.0, start_m], [0.0, start_m + spacing_m * 0.5]])
        try:
            pixels = np.asarray(projector.to_image(samples), dtype=np.float64)
        except Exception:
            continue
        if not np.all(np.isfinite(pixels)):
            continue
        if np.any(pixels[:, 0] < -image.shape[1]) or np.any(pixels[:, 0] > 2 * image.shape[1]):
            continue
        points = np.round(pixels).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, (200, 200, 200), 2, cv2.LINE_AA)


def _visible_depth_span(projector: Any, image_height: int) -> float:
    """Depth of road visible down the centre column of the frame, in metres.

    @brief   How much road the camera can actually see.
    @details Used to build scenarios that stay inside the frame.  For an overhead rig this is
             simply the frame height times the metres-per-pixel, which is the cleanest expression
             of why a table-top camera sees so little road: the visible depth is set by the mount
             height, not by the sensor.
    @param   projector    Ground geometry.
    @param   image_height Frame height in pixels.
    @return  Visible depth in metres along the road centre line.
    """
    cx = getattr(projector.intrinsics, "cx", image_height / 2.0)
    top = projector.to_ground(np.array([[cx, 0.0]]))[0]
    bottom = projector.to_ground(np.array([[cx, float(image_height - 1)]]))[0]
    span = abs(float(bottom[1]) - float(top[1]))
    if not math.isfinite(span) or span <= 0.0:
        # @note A roadside rig can have the horizon inside the frame, where the top row does not
        #       correspond to any finite depth; fall back to a depth that is certainly visible.
        return 8.0
    return span


def build_scenario(config: dict[str, Any], label: str | None = None, seed: int = 0) -> Scenario:
    """Construct a default scenario for a configuration.

    @brief   A representative single-vehicle pass for the configured rig.
    @details The vehicle speed and geometry are chosen to be realistic for the two very different
             scales this project spans: a 1:64 toy car on a table, or a real car on a street.  The
             distinction is drawn from the rig height, because that is what actually separates the
             two situations.

             A second, deliberately slower vehicle is added on the roadside scenario so that
             multiple simultaneous tracks are exercised -- the single-vehicle case is the easy one
             and would not catch an identity mix-up.
    @param   config Merged configuration.
    @param   label  Scenario name override.
    @param   seed   Random seed, reserved for future stochastic scenarios.
    @return  A :class:`Scenario` ready to render.
    """
    projector = build_projector(config)
    intrinsics_cfg = config.get("intrinsics", {})
    size = (
        int(intrinsics_cfg.get("width", 1280)),
        int(intrinsics_cfg.get("height", 720)),
    )
    fps = float(config.get("source", {}).get("fps") or 30.0)
    mount_height = float(config.get("height_m", (config.get("extrinsics") or {}).get("height_m", 2.0)))
    is_tabletop = mount_height < 1.0

    if is_tabletop:
        # @note The motion here is derived from the geometry rather than hardcoded, and the reason
        #       is a real constraint on this configuration that is worth stating: an overhead
        #       camera close enough to fill the frame with a table-top track only sees a few tens
        #       of centimetres of road. At 0.85 m and a 70 degree lens the whole visible depth is
        #       about 0.67 m, so a 1 m/s toy car crosses the entire frame in under a second and
        #       produces almost no usable measurement window. The scenario is therefore built
        #       inside the window the camera actually has.
        span_z = _visible_depth_span(projector, size[1])
        usable = max(span_z * 0.7, 0.05)
        # @step Start a little behind centre and travel forward through most of the window.
        start_z = -span_z * 0.45
        speed = 0.25
        duration = usable / speed
        vehicles = [
            VehicleTrack(
                track_id=1,
                start_xz=(0.0, start_z),
                speed_mps=speed,
                heading_rad=0.0,
                length_m=0.075,
                width_m=0.032,
            ),
        ]
    else:
        # @note A real street car. The depth is chosen inside the rig's error budget so the
        #       measurement is actually usable, which is the whole point of reporting that budget.
        budget = config.get("max_error_m")
        if budget is not None and hasattr(projector, "usable_range_m"):
            far_m = max(2.0, min(projector.usable_range_m() * 0.6, 25.0))
        else:
            far_m = 4.0
        near_m = max(0.8, far_m * 0.25)
        # @note The vehicle travels slowly enough to stay inside the *measurable* band rather than
        #       merely inside the frame, and the distinction is a real property of this rig rather
        #       than a convenience for the test.  A pitched camera compresses a distant vehicle's
        #       box until it is very wide and very short, at which point its bottom edge no longer
        #       corresponds to the near side of the footprint and the box cannot support a
        #       contact-point estimate at all. The band in which the box is still usable is
        #       narrower than the frame, so a scenario that crosses the frame quickly spends most
        #       of its length in a region the pipeline correctly refuses to measure.
        speed = 1.2
        vehicles = [
            VehicleTrack(
                track_id=1,
                start_xz=(0.2, near_m),
                speed_mps=speed,
                heading_rad=0.0,
                length_m=4.2,
                width_m=1.8,
            ),
            VehicleTrack(
                track_id=2,
                start_xz=(-0.6, near_m * 1.4),
                speed_mps=1.2,
                heading_rad=0.0,
                length_m=4.4,
                width_m=1.9,
                start_s=0.8,
            ),
        ]
        # @step Long enough for the vehicles to cross most of the measurable band.
        duration = min(6.0, max(2.5, (far_m - near_m) / speed))

    return Scenario(
        projector=projector,
        vehicles=vehicles,
        fps=fps,
        duration_s=duration,
        size=size,
        label=label or f"synthetic-{projector.kind}",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the generator.

    @brief   Render synthetic footage for a configuration.
    @param   argv Argument list.
    @return  Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="make_synthetic_video",
        description=(
            "Render synthetic ground-truth footage for a cardetect configuration. The vehicles are "
            "drawn from ground-plane footprints projected through the configured camera, and their "
            "exact positions and speeds are written to a JSON sidecar so the pipeline's output can "
            "be checked against the truth."
        ),
    )
    parser.add_argument("--config", default=None, help="camera configuration file")
    parser.add_argument("--output", default=None, help="destination video path")
    parser.add_argument("--duration", type=float, default=None, help="scene length in seconds")
    parser.add_argument("--fps", type=float, default=None, help="frame rate")
    parser.add_argument("--speed-mps", type=float, default=None, help="override the primary vehicle speed")
    parser.add_argument("--label", default=None, help="scenario name recorded in the metadata")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    scenario = build_scenario(config, label=args.label, seed=args.seed)
    if args.duration:
        scenario.duration_s = float(args.duration)
    if args.fps:
        scenario.fps = float(args.fps)
    if args.speed_mps is not None and scenario.vehicles:
        scenario.vehicles[0].speed_mps = float(args.speed_mps)

    output = Path(args.output) if args.output else PROJECT_ROOT / "samples" / f"synthetic_{scenario.projector.kind}.mp4"
    truth = render_scenario(scenario, output)
    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.json')}")
    print(f"  rig        : {truth['kind']}")
    print(f"  frames     : {len(truth['frames'])} at {truth['fps']} fps, {truth['size'][0]}x{truth['size'][1]}")
    print(f"  vehicles   : {len(truth['vehicles'])}")
    for vehicle in truth["vehicles"]:
        print(f"    #{vehicle['track_id']}: {vehicle['speed_mps']} m/s from {vehicle['start_xz']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
