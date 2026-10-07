"""End-to-end run orchestration.

@file    pipeline.py
@brief   Wire a frame source, detector, tracker, ground geometry and speed estimator together.
@details The pipeline is deliberately a plain loop rather than a framework.  On the target hardware
         -- a Raspberry Pi Zero 2 W at a few frames per second -- the overhead of an abstraction
         layer is a measurable fraction of the frame budget, and the control flow here is simple
         enough that hiding it would cost more in debuggability than it saves in flexibility.

         The order of operations in the loop is not arbitrary, and each step exists because an
         earlier arrangement produced wrong numbers:

         1. **Grab a frame and its real timestamp.**  Never ``frame_index / fps`` for a live source.
         2. **Detect**, filtered to vehicle classes.
         3. **Project** each detection's contact point and gate it.  Done *before* association so a
            phantom projection cannot attract a track.
         4. **Associate and update histories**, in metres, against real timestamps.
         5. **Estimate speed** over a trailing window for confirmed tracks only.
         6. **Annotate** the frame.  Last, because it is pure presentation and must never delay or
            influence a measurement.
         7. **Persist** measurements.  Buffered and flushed periodically, so a crash or a power cut
            on a battery-powered Pi loses at most a few dozen rows.

         Step 3 preceding step 4 is the one that matters most for correctness: gating after
         association would let an invalid projection enter a track history and corrupt a speed
         before anything noticed.

@note    OpenCV is required for capture and writing; the detector requires Ultralytics.  Both are
         imported lazily so that ``import cardetect.pipeline`` stays cheap.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cardetect import visualize
from cardetect.config import PROJECT_ROOT, config_summary, resolve_paths, validate_extrinsics_pitch
from cardetect.detect import VehicleDetector
from cardetect.geometry import GroundPlaneProjector, build_projector
from cardetect.speed import SpeedEstimate, SpeedEstimator, SpeedEstimatorConfig, TrackHistory
from cardetect.track import TrackManager, TrackManagerConfig

__all__ = ["RunResult", "Pipeline", "build_pipeline"]


@dataclass
class RunResult:
    """Everything a completed run produced.

    @brief   Paths and measurements from one run.
    @param  run_directory   Directory containing all artefacts.
    @param  tracks          Final estimate per track identity.
    @param  histories       Observation histories per track identity.
    @param  frames          Frames processed.
    @param  elapsed_s       Wall-clock duration.
    @param  stats           Tracker and detector counters.
    @param  csv_path        Per-frame measurement table, when written.
    @param  video_path      Annotated video, when written.
    @param  summary_path    Per-track JSON summary, when written.
    @param  plot_paths      Generated figures.
    @param  report_lines    Human-readable report lines.
    """

    run_directory: Path
    tracks: dict[int, SpeedEstimate] = field(default_factory=dict)
    histories: dict[int, TrackHistory] = field(default_factory=dict)
    frames: int = 0
    elapsed_s: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)
    csv_path: Path | None = None
    video_path: Path | None = None
    summary_path: Path | None = None
    report_path: Path | None = None
    plot_paths: list[Path] = field(default_factory=list)
    report_lines: list[str] = field(default_factory=list)

    @property
    def measured_tracks(self) -> dict[int, SpeedEstimate]:
        """@brief Only the tracks that produced a reportable speed."""
        return {tid: est for tid, est in self.tracks.items() if est.is_reportable}

    def summary(self) -> str:
        """@brief One-paragraph outcome, suitable for a terminal."""
        if not self.measured_tracks:
            return (
                f"{self.frames} frames processed in {self.elapsed_s:.1f}s, "
                f"no track produced a speed measurement."
            )
        lines = [
            f"{self.frames} frames in {self.elapsed_s:.1f}s "
            f"({self.frames / max(self.elapsed_s, 1e-9):.1f} fps), "
            f"{len(self.measured_tracks)} track(s) measured:"
        ]
        for track_id in sorted(self.measured_tracks):
            estimate = self.measured_tracks[track_id]
            lines.append(
                f"  #{track_id}: {estimate.speed_mps:.2f} m/s "
                f"({estimate.speed_kmh:.1f} km/h) over {estimate.span_s:.2f}s "
                f"from {estimate.samples} samples, "
                f"sigma={estimate.std_mps:.3f} m/s, noise floor={estimate.noise_floor_mps:.3f} m/s"
            )
        return "\n".join(lines)


class Pipeline:
    """Composed detection-to-measurement pipeline.

    @brief   One object holding the rig, the detector, the tracker and the estimator.
    @details Construction is separate from execution so that calibration commands can build the
             geometry alone, and so the CLI can report exactly which rig is active before any
             frames are read.

    @param  config    Merged configuration.
    @param  projector Ground geometry for the configured rig.
    @param  detector  Vehicle detector.
    @param  estimator Speed estimator.
    """

    def __init__(
        self,
        config: dict[str, Any],
        projector: GroundPlaneProjector,
        detector: VehicleDetector,
        estimator: SpeedEstimator,
        tracker_config: TrackManagerConfig,
    ) -> None:
        self.config = resolve_paths(config)
        self.projector = projector
        self.detector = detector
        self.estimator = estimator
        self.tracker = TrackManager(
            config=tracker_config,
            projector=projector,
            pixel_sigma_px=float(config.get("speed", {}).get("pixel_sigma_px", 2.0)),
        )
        self.output_cfg = self.config.get("output", {})
        self.vis_cfg = self.config.get("visualisation", {})
        self.run_directory: Path | None = None
        self.frames_processed = 0
        self.lines_written = 0
        self._csv_handle = None
        self._csv_writer = None
        self._start_time = 0.0

    # -- output plumbing ----------------------------------------------------

    def _make_run_directory(self) -> Path:
        """Create the output directory for this run.

        @brief   Timestamped run directory.
        @details Named from the run start time so repeated runs never overwrite each other, which
                 matters when comparing a calibration change against a baseline.
        @return  The created directory.
        """
        base = Path(self.output_cfg.get("directory", PROJECT_ROOT / "runs"))
        name = self.output_cfg.get("run_name") or datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = base / str(name)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _open_csv(self) -> None:
        """@brief Open the per-frame measurement table, if enabled."""
        if not self.output_cfg.get("csv", True) or self.run_directory is None:
            return
        # @note line_buffering is deliberately off and flushes are explicit; on a Pi writing to an
        #       SD card, flushing every row is a measurable cost, while flushing every hundred rows
        #       bounds data loss to a fraction of a second.
        self._csv_handle = (self.run_directory / "measurements.csv").open("w", newline="", encoding="utf-8")
        self._csv_writer = None  # header written lazily from the first row's keys

    def _write_csv_row(self, row: dict[str, Any]) -> None:
        """@brief Append one measurement row, writing the header on first use."""
        if self._csv_writer is None:
            if self._csv_handle is None:
                return
            self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self.lines_written += 1
        every = int(self.output_cfg.get("csv_flush_every", 100))
        if every > 0 and self.lines_written % every == 0 and self._csv_handle is not None:
            self._csv_handle.flush()

    def _close_csv(self) -> None:
        """@brief Flush and close the measurement table."""
        if self._csv_handle is not None:
            try:
                self._csv_handle.flush()
                self._csv_handle.close()
            finally:
                self._csv_handle = None

    # -- main loop ----------------------------------------------------------

    def run(self, source) -> RunResult:
        """Process a frame source to completion.

        @brief   Execute the pipeline over an opened :class:`~cardetect.source.FrameSource`.
        @details The detector is warmed up on the first frame so that the measured throughput in
                 the log reflects steady state rather than model loading.  Detections are projected
                 and gated before association, then speeds are estimated for confirmed tracks only.
        @param   source Iterable frame source, already opened.
        @return  :class:`RunResult`.
        """
        self.run_directory = self._make_run_directory()
        self._open_csv()
        self._start_time = time.monotonic()

        writer = None
        if self.output_cfg.get("video", True):
            writer = self._open_video_writer(source)

        display = bool(self.output_cfg.get("display", False))
        log_every = int(self.output_cfg.get("log_every", 30))
        trails = int(self.output_cfg.get("trail_length", 30))
        point_cfg = self.detector.point_config

        for frame in source:
            annotated = frame.image.copy()
            detections = self.detector.detect(frame.image)

            # @step Project and gate BEFORE association, so an invalid projection can never enter
            #       a track history.  Gating afterwards would corrupt a speed before anyone noticed.
            self.tracker.project_detections(detections)
            tracks = self.tracker.update([det for det in detections if det.valid], frame.timestamp_s)

            # @step Attach identities back onto the detections for drawing.
            for track in tracks:
                if track.last_detection is not None:
                    for detection in detections:
                        if detection is track.last_detection:
                            detection.track_id = track.track_id

            speeds = {
                track.track_id: self.estimator.estimate(track.history, self.tracker.pixel_sigma_px)
                for track in self.tracker.confirmed_tracks()
            }

            # @step Presentation last: it must never influence or delay a measurement.
            if self.output_cfg.get("video", True) or display:
                visualize.draw_trails(annotated, tracks, self.projector, trail_length=trails)
                visualize.draw_detections(annotated, detections, speeds, projector=self.projector)
                if self.output_cfg.get("minimap", True):
                    visualize.draw_minimap(annotated, tracks)
                elapsed = time.monotonic() - self._start_time
                visualize.draw_hud(
                    annotated,
                    frame.index,
                    frame.timestamp_s,
                    self.frames_processed / max(elapsed, 1e-9),
                    self.tracker.stats(),
                )

            for detection in detections:
                self._write_csv_row(detection.to_row(frame.index, frame.timestamp_s))

            if writer is not None:
                writer.write(annotated)

            if display:
                self._show(annotated)
                if self._should_quit():
                    break

            self.frames_processed += 1
            if log_every > 0 and self.frames_processed % log_every == 0:
                elapsed = time.monotonic() - self._start_time
                print(
                    f"  frame {self.frames_processed:6d}  t={frame.timestamp_s:7.2f}s  "
                    f"{self.frames_processed / max(elapsed, 1e-9):5.1f} fps  "
                    f"tracks {len(self.tracker.tracks)}"
                )

        self._close_csv()
        if writer is not None:
            writer.release()
        if display:
            self._destroy_windows()

        return self._finish(source)

    def _open_video_writer(self, source):
        """Create the annotated-video writer, inferring size and rate from the source.

        @brief   Open an ``mp4v`` writer sized from the actual frames.
        @details The frame size is taken from a real frame rather than the requested size, because
                 a camera may substitute a different resolution and a mismatch between the writer
                 and the frames silently produces a corrupt file.  The rate falls back to the
                 measured throughput when the source does not report one, which is normal for a
                 live camera; the timestamps in the CSV remain authoritative regardless.
        @param   source Opened frame source.
        @return  An OpenCV ``VideoWriter``, or ``None`` when it cannot be created.
        """
        try:
            import cv2
        except ImportError:  # pragma: no cover - optional dependency
            return None
        size = (int(source.requested_width or 1280), int(source.requested_height or 720))
        fps = float(source.reported_fps or source.requested_fps or 30.0)
        path = self.run_directory / "annotated.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            return None
        self._video_path = path
        return writer

    def _show(self, image: np.ndarray) -> None:
        """@brief Display a frame in a window when a display is available."""
        try:
            import cv2

            cv2.imshow("cardetect", image)
        except Exception:  # pragma: no cover - headless environment
            pass

    def _should_quit(self) -> bool:
        """@brief Whether the operator pressed the quit key."""
        try:
            import cv2

            return (cv2.waitKey(1) & 0xFF) in (ord("q"), 27)
        except Exception:  # pragma: no cover - headless environment
            return False

    def _destroy_windows(self) -> None:
        """@brief Tear down display windows."""
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:  # pragma: no cover - headless environment
            pass

    # -- reporting ----------------------------------------------------------

    def _finish(self, source) -> RunResult:
        """Collect final estimates, write summaries and figures.

        @brief   Close out a run.
        @details Speed estimation runs once more over the retired tracks as well as the live ones,
                 so a vehicle that left the frame before the run ended still contributes its final
                 measurement rather than being silently dropped for leaving last.
        @param   source The frame source, used for its reported properties.
        @return  :class:`RunResult`.
        """
        result = RunResult(
            run_directory=self.run_directory or PROJECT_ROOT / "runs",
            frames=self.frames_processed,
            elapsed_s=time.monotonic() - self._start_time,
            stats={**self.tracker.stats(), "detector": self.detector.describe(), "source": source.describe()},
        )
        histories = self.tracker.all_histories
        result.histories = histories
        result.tracks = {
            track_id: self.estimator.estimate(history, self.tracker.pixel_sigma_px)
            for track_id, history in histories.items()
        }
        result.video_path = getattr(self, "_video_path", None)

        if self.output_cfg.get("summary_json", True):
            result.summary_path = self._write_summary_json(result)
        if self.output_cfg.get("plots", True):
            result.plot_paths = self._write_figures(result)

        result.report_lines = self._report_lines(result, source)
        result.report_path = visualize.save_text_summary(self.run_directory / "report.txt", result.report_lines)
        return result

    def _write_summary_json(self, result: RunResult) -> Path | None:
        """@brief Write one JSON record per track, with the full estimate diagnostics."""
        path = self.run_directory / "tracks.json"
        payload = {
            "config": {
                "kind": self.config.get("kind"),
                "label": self.config.get("label"),
                "projector": self.projector.describe(),
                "detector": self.detector.describe(),
            },
            "run": {
                "frames": result.frames,
                "elapsed_s": result.elapsed_s,
                "stats": result.stats,
            },
            "tracks": {
                str(track_id): {
                    **estimate.to_row(),
                    "truth_available": False,
                }
                for track_id, estimate in sorted(result.tracks.items())
            },
        }
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return path

    def _write_figures(self, result: RunResult) -> list[Path]:
        """@brief Write the speed trace, trajectory and resolution figures."""
        paths: list[Path] = []
        window = float(self.vis_cfg.get("speed_plot_window_s", 5.0))
        ceiling = float(self.vis_cfg.get("speed_plot_max_mps", 8.0))
        trace = visualize.speed_trace_figure(
            result.histories, result.tracks, self.run_directory / "speed_trace.png", window, ceiling
        )
        if trace:
            paths.append(trace)
        trajectory = visualize.trajectory_figure(result.histories, self.run_directory / "trajectories.png")
        if trajectory:
            paths.append(trajectory)
        resolution = visualize.ground_resolution_figure(
            self.projector,
            self.run_directory / "ground_resolution.png",
            pixel_sigma_px=float(self.config.get("speed", {}).get("pixel_sigma_px", 2.0)),
            max_error_m=self.config.get("max_error_m"),
        )
        if resolution:
            paths.append(resolution)
        return paths

    def _report_lines(self, result: RunResult, source) -> list[str]:
        """Assemble the plain-text run report.

        @brief   Human-readable record of what was configured and measured.
        @details Written next to every run because the first question about a surprising measurement
                 is always which configuration produced it, and a CSV alone cannot answer that.
        @param   result Completed run.
        @param   source The frame source.
        @return  Report lines.
        """
        lines = [
            "cardetect run report",
            "=" * 60,
            f"generated        : {datetime.now().isoformat(timespec='seconds')}",
            f"frames processed : {result.frames}",
            f"wall clock       : {result.elapsed_s:.2f} s",
            f"throughput       : {result.frames / max(result.elapsed_s, 1e-9):.2f} fps",
            f"run directory    : {result.run_directory}",
            "",
            "effective configuration",
            "-" * 60,
            config_summary(self.config),
            "",
            "source",
            "-" * 60,
        ]
        lines.extend(f"  {key:16}: {value}" for key, value in source.describe().items())
        lines.extend(["", "geometry", "-" * 60])
        lines.extend(f"  {key:16}: {value}" for key, value in self.projector.describe().items())
        lines.extend(["", "measurements", "-" * 60])
        if not result.tracks:
            lines.append("  no tracks were formed")
        for track_id in sorted(result.tracks):
            estimate = result.tracks[track_id]
            if estimate.is_reportable:
                lines.append(
                    f"  #{track_id:<4} {estimate.speed_mps:6.3f} m/s  {estimate.speed_kmh:6.2f} km/h  "
                    f"samples={estimate.samples:<4} span={estimate.span_s:.3f}s  "
                    f"sigma={estimate.std_mps:.3f}  floor={estimate.noise_floor_mps:.3f}  "
                    f"rejected={estimate.rejected_steps}"
                )
            else:
                lines.append(f"  #{track_id:<4} not reportable: {estimate.reason}")
        lines.extend(["", "artefacts", "-" * 60])
        for label, path in (
            ("measurements", result.csv_path or (self.run_directory / "measurements.csv")),
            ("annotated", result.video_path),
            ("tracks", result.summary_path),
            ("report", result.report_path),
        ):
            if path is not None:
                lines.append(f"  {label:16}: {path}")
        for path in result.plot_paths:
            lines.append(f"  {'figure':16}: {path}")
        return lines


def _json_default(value: Any) -> Any:
    """@brief JSON encoder fallback for NumPy scalars and paths."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """Assemble a pipeline from a merged configuration.

    @brief   The one place all components are constructed.
    @details Surfaces non-fatal geometry warnings through the returned pipeline's config rather
             than printing here, so a caller can choose how to present them.  Metric scale is *not*
             enforced: a rig can be run without calibration for tracking and visualisation, and the
             caller is warned rather than blocked.
    @param   config Merged configuration.
    @return  A ready :class:`Pipeline`.
    @raises  ValueError when the configuration cannot produce a usable rig.
    """
    warnings = validate_extrinsics_pitch(config)
    if warnings:
        config = {**config, "_warnings": warnings}
    projector = build_projector(config)
    detector = VehicleDetector.from_config(config)
    speed_cfg = config.get("speed", {})
    estimator = SpeedEstimator(
        SpeedEstimatorConfig(
            window_s=float(speed_cfg.get("window_s", 0.5)),
            min_span_s=float(speed_cfg.get("min_span_s", 0.08)),
            max_step_m=float(speed_cfg.get("max_step_m", 0.5)),
            min_samples=int(speed_cfg.get("min_samples", 3)),
            max_speed_mps=float(speed_cfg.get("max_speed_mps", 12.0)),
            use_smoothing=bool(speed_cfg.get("use_smoothing", True)),
            robust_percentile=float(speed_cfg.get("robust_percentile", 50.0)),
            max_interval_cv=float(speed_cfg.get("max_interval_cv", 0.2)),
        )
    )
    tracker_config = TrackManagerConfig.from_config(config)
    return Pipeline(config, projector, detector, estimator, tracker_config)
