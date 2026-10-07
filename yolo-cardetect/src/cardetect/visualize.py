"""Overlays, plots and calibration helpers for displaying and verifying measurements.

@file    visualize.py
@brief   Draw annotated frames and produce the plots a measurement should be judged against.
@details Two distinct jobs live here, and keeping them separate is deliberate:

         **Annotation** is drawn onto the video as it is processed.  It is bound by the frame
         budget, so it does the minimum work that makes a run interpretable: a box, an identity,
         a speed, and a short trajectory.  Anything historical or expensive belongs in a plot, not
         in the frame loop -- on a Raspberry Pi Zero 2 W, drawing every track's entire history
         costs more than the detector does.

         **Figures** are produced after a run and are where the *verification* happens.  A speed
         number in a CSV is easy to believe; a speed-versus-time trace with the measured noise
         floor drawn on it, next to the bird's-eye trajectory that produced it, is not.  The plots
         here are therefore treated as evidence rather than decoration, and each one is chosen to
         expose a specific failure mode:

         * the **speed trace** shows whether a measurement is stable, and the noise-floor band
           makes resolution-limited jitter distinguishable from a detector problem;
         * the **bird's-eye trajectory** shows whether the ground projection is sane -- a track
           that curves, doubles back or teleports is a geometry or tracking fault that a speed
           number alone would hide;
         * the **ground-resolution map** shows where on the road the rig can actually measure,
           which is the information needed to choose a mounting position.

@note    ``matplotlib`` and ``cv2`` are imported lazily so the core stays importable on a
         constrained device.  Every function degrades to a no-op rather than raising when its
         optional dependency is missing, because failing to draw a plot must never lose a
         measurement.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "COLORS",
    "draw_detections",
    "draw_hud",
    "draw_minimap",
    "draw_ground_grid",
    "speed_trace_figure",
    "ground_resolution_figure",
    "trajectory_figure",
    "save_text_summary",
]

# @brief A small, high-contrast palette.  Chosen so that consecutive track identities are
#        distinguishable on a phone screen in daylight, which is where most of these videos are
#        actually reviewed.  Deliberately avoids red/green adjacency for the most common pair.
COLORS: tuple[tuple[int, int, int], ...] = (
    (60, 200, 255),    # amber
    (255, 160, 60),    # blue
    (120, 255, 120),   # green
    (255, 120, 220),   # magenta
    (120, 220, 255),   # cyan
    (200, 200, 90),    # teal
    (255, 200, 120),   # light blue
    (170, 130, 255),   # violet
)


def color_for(track_id: int) -> tuple[int, int, int]:
    """@brief Stable BGR colour for a track identity."""
    return COLORS[int(track_id) % len(COLORS)]


def _label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """Draw a text label with a filled background for legibility.

    @brief   Outlined text that survives a busy background.
    @details A road surface is a high-contrast, high-detail background, and unbacked text is
             frequently unreadable against it in exactly the frames worth examining.  The box is
             filled first, then the text drawn in black for maximum contrast against the fill.
    @param   image      BGR image, modified in place.
    @param   text       Label text.
    @param   origin     Bottom-left corner of the text, in pixels.
    @param   color      BGR fill colour.
    @param   font_scale OpenCV font scale.
    @param   thickness  Stroke thickness.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        return
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = origin
    top = max(0, y - text_h - baseline)
    cv2.rectangle(image, (x, top), (x + text_w + 4, y + baseline), color, -1)
    cv2.putText(
        image,
        text,
        (x + 2, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(
    image: np.ndarray,
    detections: Iterable[Any],
    speeds: dict[int, Any] | None = None,
    font_scale: float = 0.5,
    thickness: int = 2,
    trail_length: int = 30,
    projector: Any | None = None,
) -> np.ndarray:
    """Draw boxes, identities, speeds and short trajectories onto a frame.

    @brief   Annotate a frame in place.
    @details Trajectories are projected back to the image through the geometry when a projector is
             supplied, so the drawn path is exactly what the measurement used rather than a
             visually plausible approximation.  That distinction matters when diagnosing a bad
             speed: a trajectory that visibly disagrees with the road is a geometry fault, and a
             trajectory that agrees while the speed is wrong is an estimator fault.
    @param   image        BGR frame, modified in place.
    @param   detections   Objects exposing ``box_xyxy``, ``track_id``, ``name``, ``confidence`` and
                          optionally ``ground_xz``.
    @param   speeds       Mapping of track identity to an object exposing ``speed_kmh`` and
                          ``speed_mps``.
    @param   font_scale   OpenCV font scale.
    @param   thickness    Line thickness.
    @param   trail_length Number of trajectory points to draw.
    @param   projector    Ground geometry, used to project trajectories back to the image.
    @return  The same image, for chaining.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        return image

    speeds = speeds or {}
    for detection in detections:
        track_id = detection.track_id if detection.track_id is not None else -1
        color = color_for(track_id)
        box = np.asarray(detection.box_xyxy, dtype=np.float64)
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # @step Mark the reference point actually used, not the box centre.  Seeing it move with
        #       the geometry is the quickest way to spot a wrong contact-point configuration.
        reference = getattr(detection, "reference_uv", None)
        if reference is not None:
            cv2.circle(image, (int(round(reference[0])), int(round(reference[1]))), 3, color, -1)
            cv2.line(
                image,
                (int(round((box[0] + box[2]) / 2.0)), y2),
                (int(round(reference[0])), int(round(reference[1]))),
                color,
                1,
                cv2.LINE_AA,
            )

        estimate = speeds.get(track_id)
        if estimate is not None and getattr(estimate, "speed_mps", None) is not None:
            label = f"#{track_id} {estimate.speed_kmh:.1f}km/h"
        else:
            label = f"#{track_id} {getattr(detection, 'name', 'vehicle')} {getattr(detection, 'confidence', 0.0):.2f}"
        _label(image, label, (x1, max(y1 - 2, 12)), color, font_scale, 1)

    return image


def draw_trails(
    image: np.ndarray,
    tracks: Iterable[Any],
    projector: Any,
    trail_length: int = 30,
    thickness: int = 2,
) -> np.ndarray:
    """Draw each track's recent ground trajectory, projected back into the image.

    @brief   Overlay trajectories as they appear on the road.
    @details Kept separate from :func:`draw_detections` because it needs the projector and is the
             expensive part; callers on a constrained device can omit it entirely.
    @param   image        BGR frame, modified in place.
    @param   tracks       Objects exposing ``track_id`` and ``trail(length)``.
    @param   projector    Ground geometry used to project road positions back to pixels.
    @param   trail_length Number of points per trajectory.
    @param   thickness    Line thickness.
    @return  The same image, for chaining.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        return image
    if projector is None:
        return image

    for track in tracks:
        ground = np.asarray(track.trail(trail_length), dtype=np.float64)
        if len(ground) < 2:
            continue
        try:
            pixels = projector.to_image(ground)
        except Exception:
            continue
        pixels = np.asarray(pixels, dtype=np.float64)
        if not np.all(np.isfinite(pixels)):
            continue
        color = color_for(track.track_id)
        points = np.round(pixels).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)
    return image


def draw_hud(
    image: np.ndarray,
    frame_index: int,
    timestamp_s: float,
    fps_measured: float,
    stats: dict[str, Any] | None = None,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw the heads-up diagnostics panel.

    @brief   Show frame rate, elapsed time and track counters.
    @details The *measured* frame rate is shown rather than the requested one.  On the hardware
             this project targets those differ by a factor of several, and that difference is
             precisely what limits speed accuracy, so it should be visible while the run happens
             rather than discovered afterwards.
    @param   image        BGR frame, modified in place.
    @param   frame_index  Current frame number.
    @param   timestamp_s  Elapsed time in seconds.
    @param   fps_measured Measured processing rate in frames per second.
    @param   stats        Optional additional counters to display.
    @param   font_scale   OpenCV font scale.
    @return  The same image, for chaining.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        return image

    lines = [
        f"frame {frame_index}  t={timestamp_s:6.2f}s",
        f"throughput {fps_measured:5.1f} fps",
    ]
    if stats:
        lines.append(
            f"tracks live {stats.get('live_tracks', 0)}  done {stats.get('retired_tracks', 0)}  "
            f"gated {stats.get('rejected_projections', 0)}"
        )
    y = 16
    for line in lines:
        cv2.putText(
            image,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += int(18 * (font_scale / 0.5))
    return image


def draw_ground_grid(
    image: np.ndarray,
    projector: Any,
    spacing_m: float = 0.25,
    extent_m: float = 6.0,
    cross_spacing_m: float = 0.25,
    cross_extent_m: float = 1.5,
    font_scale: float = 0.4,
) -> np.ndarray:
    """Draw a metric grid on the road surface, projected through the ground geometry.

    @brief   Visual calibration check: a ruler drawn on the road.
    @details This is the fastest way to confirm a calibration is right.  A grid whose lines are
             evenly spaced on the real road is a correct calibration; a grid that crowds or fans
             out is wrong, and the direction of the error usually identifies which parameter is
             off.  Distances are labelled so the check is quantitative rather than visual.

             Deliberately kept out of the live loop: it projects tens of points per line and is a
             commissioning tool, not a runtime overlay.
    @param   image            BGR frame, modified in place.
    @param   projector        Ground geometry.
    @param   spacing_m        Longitudinal spacing of grid lines, metres.
    @param   extent_m         How far down the road to draw, metres.
    @param   cross_spacing_m  Lateral spacing of grid lines, metres.
    @param   cross_extent_m   How far across the road to draw, metres.
    @param   font_scale       OpenCV font scale.
    @return  The same image, for chaining.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        return image
    if projector is None:
        return image

    color = (80, 220, 80)
    # @step Longitudinal lines: constant X, varying Z, so the spacing reveals the depth scale.
    lateral = np.arange(-cross_extent_m, cross_extent_m + 1e-9, cross_spacing_m)
    depths = np.arange(0.0, extent_m + 1e-9, spacing_m)
    for x_value in lateral:
        samples = np.column_stack([np.full_like(depths, x_value), depths])
        _draw_polyline(image, projector, samples, color, 1)
    # @step Lateral lines: constant Z, varying X, so the spacing reveals the cross-road scale.
    for z_value in depths[:: max(1, int(round(0.5 / max(spacing_m, 1e-6))))]:
        samples = np.column_stack([np.linspace(-cross_extent_m, cross_extent_m, 32), np.full(32, z_value)])
        _draw_polyline(image, projector, samples, color, 1)
        anchor = projector.to_image(np.array([[-cross_extent_m, z_value]]))[0]
        if np.all(np.isfinite(anchor)) and 0 <= anchor[0] < image.shape[1] and 0 <= anchor[1] < image.shape[0]:
            cv2.putText(
                image,
                f"{z_value:.2f}m",
                (int(anchor[0]) + 2, int(anchor[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
    return image


def _draw_polyline(
    image: np.ndarray,
    projector: Any,
    ground_samples: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Project and draw one ground-space polyline, skipping anything off-frame.

    @brief   Shared helper for the ground grid.
    @details Projections that leave the frame or land on the horizon produce values that are
             finite but meaningless; they are dropped rather than drawn, which is why the grid
             terminates at the frame edge instead of drawing a spurious line across the sky.
    @param   image         BGR frame, modified in place.
    @param   projector     Ground geometry.
    @param   ground_samples ``(N, 2)`` road positions in metres.
    @param   color         BGR colour.
    @param   thickness     Line thickness.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        return
    try:
        pixels = np.asarray(projector.to_image(ground_samples), dtype=np.float64)
    except Exception:
        return
    inside = (
        np.all(np.isfinite(pixels), axis=1)
        & (pixels[:, 0] >= -image.shape[1])
        & (pixels[:, 0] <= 2 * image.shape[1])
        & (pixels[:, 1] >= -image.shape[0])
        & (pixels[:, 1] <= 2 * image.shape[0])
    )
    if np.count_nonzero(inside) < 2:
        return
    points = np.round(pixels[inside]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)


def draw_minimap(
    image: np.ndarray,
    tracks: Iterable[Any],
    extent_m: tuple[float, float] = (1.6, 6.0),
    size_px: tuple[int, int] = (200, 160),
    margin: int = 10,
) -> np.ndarray:
    """Draw a bird's-eye inset showing track trajectories in the road frame.

    @brief   Top-down trajectory view in metres.
    @details This is the honest view of the measurement: it is drawn in the same coordinates the
             speed was computed from, so a wrong calibration is visible as a trajectory that does
             not lie along the road rather than hidden behind a plausible number.  The extent
             defaults to a few metres, which suits a table-top track; a roadside rig needs a larger
             ``extent_m``.
    @param   image    BGR frame, modified in place.
    @param   tracks   Objects exposing ``track_id`` and ``trail(length)``.
    @param   extent_m ``(half_width, max_depth)`` in metres.
    @param   size_px  Inset size in pixels.
    @param   margin   Margin from the frame edge.
    @return  The same image, for chaining.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        return image

    half_width, max_depth = extent_m
    inset_w, inset_h = size_px
    x0 = image.shape[1] - inset_w - margin
    y0 = margin
    # @step Semi-transparent background so the video stays visible underneath.
    panel = image[y0 : y0 + inset_h, x0 : x0 + inset_w].copy()
    cv2.rectangle(panel, (0, 0), (inset_w - 1, inset_h - 1), (20, 20, 20), -1)
    image[y0 : y0 + inset_h, x0 : x0 + inset_w] = cv2.addWeighted(
        panel, 0.35, image[y0 : y0 + inset_h, x0 : x0 + inset_w], 0.65, 0
    )

    def to_inset(x_m: float, z_m: float) -> tuple[int, int]:
        """@brief Map road metres into inset pixel coordinates."""
        u = int(np.clip((x_m + half_width) / (2 * half_width) * inset_w, 0, inset_w - 1))
        # @note Z increases with distance from the camera, drawn upward in the inset.
        v = int(np.clip(inset_h - 1 - (z_m / max_depth) * inset_h, 0, inset_h - 1))
        return x0 + u, y0 + v

    # @step A centre line and metric ticks, so the inset is readable as a scale rather than a
    #       decorative squiggle.
    cv2.line(image, to_inset(0.0, 0.0), to_inset(0.0, max_depth), (90, 90, 90), 1)
    for depth_m in np.arange(1.0, max_depth + 1e-9, 1.0):
        left = to_inset(-half_width, depth_m)
        right = to_inset(half_width, depth_m)
        cv2.line(image, left, right, (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{depth_m:.0f}m",
            (left[0] + 2, left[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

    for track in tracks:
        ground = np.asarray(track.trail(60), dtype=np.float64)
        if len(ground) < 2:
            continue
        color = color_for(track.track_id)
        points = np.asarray([to_inset(x_m, z_m) for x_m, z_m in ground], dtype=np.int32)
        cv2.polylines(image, [points.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
        cv2.circle(image, tuple(points[-1]), 2, color, -1)
    return image


def speed_trace_figure(
    histories: dict[int, Any],
    speeds: dict[int, Any],
    output_path: str | Path,
    window_s: float = 5.0,
    max_speed_mps: float = 8.0,
) -> Path | None:
    """Plot per-frame speed against time for every track, with the noise floor shown.

    @brief   The primary evidence figure for a run.
    @details Two things make this usable as evidence rather than decoration.  The measured
             noise floor is drawn as a band, so jitter that is explained by ground resolution is
             immediately distinguishable from jitter that is not -- without it, every noisy trace
             looks equally like a detector failure.  And the instantaneous trace is drawn alongside
             the robust windowed estimate, so it is clear which number the pipeline actually
             reports and how the robust statistic behaves on real data.
    @param   histories    Mapping of track identity to :class:`~cardetect.speed.TrackHistory`.
    @param   speeds       Mapping of track identity to :class:`~cardetect.speed.SpeedEstimate`.
    @param   output_path  Destination image path.
    @param   window_s     Length of the reported window, drawn as a shaded region at the end.
    @param   max_speed_mps Y-axis ceiling in metres per second.
    @return  The written path, or ``None`` when matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return None

    figure, axes = plt.subplots(figsize=(10, 5), dpi=140)
    plotted = 0
    for track_id in sorted(histories):
        history = histories[track_id]
        times, positions, _scales = history.as_arrays()
        if len(times) < 3:
            continue
        # @step Instantaneous speed from the ground-truth-free series, drawn faintly.
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        dt = np.diff(times)
        valid = dt > 0.0
        if not np.any(valid):
            continue
        instantaneous = steps[valid] / dt[valid]
        color = tuple(channel / 255.0 for channel in reversed(color_for(track_id)))
        axes.plot(times[1:][valid], instantaneous, color=color, alpha=0.35, linewidth=1)
        axes.plot(times[-1], float(np.median(instantaneous)), "o", color=color, markersize=5)

        estimate = speeds.get(track_id)
        if estimate is not None and getattr(estimate, "speed_mps", None) is not None:
            axes.axhline(
                estimate.speed_mps,
                color=color,
                linestyle="--",
                linewidth=1,
                alpha=0.8,
            )
            if estimate.noise_floor_mps > 0.0:
                # @step The resolution-limited band: any variation inside this is explained by
                #       ground scale and frame interval alone.
                axes.fill_between(
                    [times[0], times[-1]],
                    max(0.0, estimate.speed_mps - estimate.noise_floor_mps),
                    estimate.speed_mps + estimate.noise_floor_mps,
                    color=color,
                    alpha=0.12,
                )
            axes.annotate(
                f"#{track_id} {estimate.speed_mps:.2f} m/s",
                xy=(times[-1], estimate.speed_mps),
                xytext=(4, 4),
                textcoords="offset points",
                color=color,
                fontsize=8,
            )
        plotted += 1

    axes.set_xlabel("time (s)")
    axes.set_ylabel("speed (m/s)")
    axes.set_ylim(0.0, max_speed_mps)
    axes.grid(alpha=0.25)
    axes.set_title("Measured speed per track (dashed = reported robust estimate, band = noise floor)")
    if plotted == 0:
        axes.text(0.5, 0.5, "no track had enough samples for a measurement", ha="center", va="center")
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def trajectory_figure(
    histories: dict[int, Any], output_path: str | Path, extent_m: tuple[float, float] = (2.0, 8.0)
) -> Path | None:
    """Plot every track's trajectory in the road plane, in metres.

    @brief   Bird's-eye view of what was measured.
    @details A trajectory that is not a smooth line along the road indicates a geometry or
             tracking fault, which is exactly the class of problem that leaves the speed number
             looking plausible.  Equal axis scaling is enforced so the shape is not misleading.
    @param   histories    Mapping of track identity to track history.
    @param   output_path  Destination image path.
    @param   extent_m     ``(half_width, max_depth)`` limits in metres.
    @param   output_path  Destination path.
    @return  The written path, or ``None`` when matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return None

    half_width, max_depth = extent_m
    figure, axes = plt.subplots(figsize=(5, 7), dpi=140)
    for track_id in sorted(histories):
        positions = np.asarray(histories[track_id].positions, dtype=np.float64).reshape(-1, 2)
        if len(positions) < 2:
            continue
        color = tuple(channel / 255.0 for channel in reversed(color_for(track_id)))
        axes.plot(positions[:, 0], positions[:, 1], color=color, linewidth=1.4, label=f"#{track_id}")
        axes.plot(positions[0, 0], positions[0, 1], "o", color=color, markersize=4)
    axes.set_xlim(-half_width, half_width)
    axes.set_ylim(0.0, max_depth)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("X across road (m)")
    axes.set_ylabel("Z along road (m)")
    axes.grid(alpha=0.25)
    axes.set_title("Ground trajectories (start marked with a dot)")
    if axes.get_legend_handles_labels()[0]:
        axes.legend(fontsize=7, loc="upper right")
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def ground_resolution_figure(
    projector: Any,
    output_path: str | Path,
    extent_m: tuple[float, float] = (2.0, 12.0),
    pixel_sigma_px: float = 2.0,
    max_error_m: float | None = None,
) -> Path | None:
    """Plot the rig's positional error across the road surface.

    @brief   Where the camera can actually measure.
    @details This is the figure to consult when choosing a mounting position.  It shows the
             metric positional error implied by detector jitter at every point on the road, so the
             usable area is a visible region rather than a number discovered by trial.  On a
             roadside rig the error grows steeply with distance; on an overhead rig it is flat,
             which is the quantitative statement of why the overhead configuration is recommended
             for a table-top track.
    @param   projector    Ground geometry.
    @param   output_path  Destination image path.
    @param   extent_m     ``(half_width, max_depth)`` sampled area in metres.
    @param   pixel_sigma_px Assumed detector jitter in pixels.
    @param   max_error_m  Error budget; the region inside it is outlined.
    @return  The written path, or ``None`` when matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return None

    half_width, max_depth = extent_m
    xs = np.linspace(-half_width, half_width, 90)
    zs = np.linspace(0.05, max_depth, 120)
    grid_x, grid_z = np.meshgrid(xs, zs)
    points = np.column_stack([grid_x.ravel(), grid_z.ravel()])
    error = projector.position_sigma_m(points, pixel_sigma_px).reshape(grid_z.shape) * 100.0  # cm

    figure, axes = plt.subplots(figsize=(6, 7), dpi=140)
    mesh = axes.pcolormesh(grid_x, grid_z, error, shading="auto", cmap="viridis")
    figure.colorbar(mesh, ax=axes, label="positional error (cm, 1 sigma)")
    if max_error_m is not None:
        budget_cm = max_error_m * 100.0
        axes.contour(grid_x, grid_z, error, levels=[budget_cm], colors="red", linewidths=1.6)
        axes.plot([], [], color="red", linewidth=1.6, label=f"{budget_cm:.1f} cm budget")
        axes.legend(fontsize=8, loc="upper right")
    axes.set_xlabel("X across road (m)")
    axes.set_ylabel("Z along road (m)")
    axes.set_aspect("equal", adjustable="box")
    axes.set_title(f"Positional error from {pixel_sigma_px:.1f} px detector jitter")
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def save_text_summary(path: str | Path, lines: Sequence[str]) -> Path:
    """Write a plain-text summary file.

    @brief   Persist a human-readable run report next to the CSV.
    @details Provided because a CSV is awkward to read when something looks wrong, and the first
             question after a strange measurement is always which configuration produced it.
    @param   path  Destination file.
    @param   lines Lines to write.
    @return  The written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
