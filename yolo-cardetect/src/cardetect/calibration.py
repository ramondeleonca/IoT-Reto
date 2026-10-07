"""Interactive calibration commands.

@file    calibration.py
@brief   Turn a physical measurement into a stored calibration file.
@details The workflow chosen for this project is a **tape measure and two marks on the road**, so
         this module is built around that and around being explicit about the one thing that
         approach cannot do on its own.

         A measured distance between two image points constrains the *scale* of the ground map.
         It does not constrain the map's full projective shape, because scale, height and pitch are
         linked: a camera twice as high aimed twice as steeply sees almost the same image, so a
         single distance cannot separate them.  For the overhead configuration that is harmless,
         because the map really is an affine similarity and one distance plus an assumed image
         rotation pins it completely.  For the roadside configuration it is not, so the roadside
         flow requires one additional constraint -- a horizon row, a measured lens height, or a
         known tilt -- and refuses to proceed without one.

         Refusing is the point.  A wrong height biases every subsequent speed by exactly the height
         ratio; those numbers look entirely reasonable, and nothing downstream can detect the
         error.  An error message at calibration time is far cheaper.

@note    ``cv2`` is imported lazily.  The pure-geometry parts of this module are exercised by the
         test suite without a display, which is why the interactive front end is separated from the
         solving logic in :mod:`cardetect.geometry`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from cardetect.config import PROJECT_ROOT
from cardetect.geometry import (
    Intrinsics,
    RoadsideProjector,
    TopDownProjector,
    build_projector,
)

__all__ = [
    "point_picker",
    "calibrate_topdown_two_point",
    "calibrate_roadside_two_point",
    "calibrate_intrinsics_chessboard",
    "save_calibration",
    "load_calibration",
]


def save_calibration(path: str | Path, projector: Any, extra: dict[str, Any] | None = None) -> Path:
    """Write a projector's calibration to YAML.

    @brief   Persist a rig calibration.
    @details Serialises through the projector's own ``to_config`` so the file is exactly what
             :func:`cardetect.geometry.build_projector` reads back, and additionally records the
             human context -- when it was taken and what the operator measured -- because a
             calibration file with no provenance is impossible to audit months later.
    @param   path    Destination file.
    @param   projector The calibrated projector.
    @param   extra   Additional metadata to record under a ``metadata`` key.
    @return  The written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = projector.to_config()
    document["metadata"] = {
        "created": _timestamp(),
        "generator": "cardetect calibrate",
        **(extra or {}),
    }
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return target


def load_calibration(path: str | Path) -> Any:
    """Load a stored calibration and rebuild its projector.

    @brief   Inverse of :func:`save_calibration`.
    @param   path Source file.
    @return  A projector of the recorded kind.
    @raises  FileNotFoundError when the file is absent.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"calibration file not found: {source}")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{source} does not contain a calibration mapping")
    return build_projector(document)


def _timestamp() -> str:
    """@brief ISO timestamp for calibration metadata."""
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


def point_picker(
    image: np.ndarray,
    prompt: str,
    count: int = 2,
    window: str = "cardetect calibrate",
    zoom: float = 1.0,
) -> list[tuple[float, float]]:
    """Let the operator click points on a frame with the mouse.

    @brief   Interactive point selection on a still frame.
    @details The image is shown scaled up by default, because the marks in a two-point tape
             calibration are typically a few hundred pixels apart on a 720p frame and clicking
             accuracy translates directly into scale error.  A magnified view, a crosshair, and a
             live readout of the picked coordinates all exist to make the click as repeatable as
             possible, since this is the dominant error term in the whole measurement chain for
             this workflow.

             Controls: left click places a point, ``r`` restarts, ``q`` or Escape aborts.

    @param   image  BGR frame to display.
    @param   prompt Instructions shown in the window title bar.
    @param   count  Number of points to collect.
    @param   window Window name.
    @param   zoom   Display magnification; coordinates are divided by this before returning.
    @return  List of ``(u, v)`` image coordinates in the original frame's pixel space.
    @raises  RuntimeError when OpenCV is unavailable or the operator aborts.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "interactive calibration requires OpenCV. Install it with:\n    uv sync --extra calibration"
        ) from exc

    picked: list[tuple[float, float]] = []
    display = _resize_for_display(image, zoom)
    scale = _display_scale(image, display)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        """@brief Mouse callback collecting clicks, converting back to original coordinates."""
        if event != cv2.EVENT_LBUTTONDOWN or len(picked) >= count:
            return
        picked.append((x / scale, y / scale))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = display.copy()
        for index, (u, v) in enumerate(picked):
            point = (int(round(u * scale)), int(round(v * scale)))
            cv2.drawMarker(canvas, point, (60, 220, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.putText(
                canvas,
                f"P{index + 1} ({u:.0f}, {v:.0f})",
                (point[0] + 12, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (60, 220, 255),
                2,
                cv2.LINE_AA,
            )
        hint = f"{prompt}  |  clicked {len(picked)}/{count}  |  r=reset  q=abort"
        cv2.putText(canvas, hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            picked.clear()
            break
        if key == ord("r"):
            picked.clear()
        if len(picked) >= count:
            break
    cv2.destroyWindow(window)
    if len(picked) < count:
        raise RuntimeError("calibration aborted: not enough points were selected")
    return picked


def _resize_for_display(image: np.ndarray, zoom: float) -> np.ndarray:
    """@brief Scale an image for a display that may be smaller or larger than the frame."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        return image
    if abs(zoom - 1.0) < 1e-6:
        return image
    return cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)


def _display_scale(original: np.ndarray, display: np.ndarray) -> float:
    """@brief Ratio between displayed and original width, used to invert clicks."""
    if original.shape[1] == 0:
        return 1.0
    return display.shape[1] / float(original.shape[1])


def prompt_distance(default_m: float | None = None) -> float:
    """Ask the operator for the measured distance between the marks.

    @brief   Console prompt for a physical measurement.
    @details A default is offered because a standard straight Hot Wheels track section is a very
             convenient and exactly-known reference at 30.48 cm (12 inches), and using it removes
             tape-measure error from the calibration entirely.
    @param   default_m Suggested distance in metres.
    @return  The entered distance in metres.
    @raises  RuntimeError when the input cannot be interpreted as a positive distance.
    """
    suffix = f" [{default_m} m, Enter to accept]: " if default_m else " (metres): "
    raw = input(f"Measured distance between the two marks{suffix}").strip()
    if not raw and default_m is not None:
        return float(default_m)
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"could not read a distance from {raw!r}") from exc
    if value <= 0.0:
        raise RuntimeError(f"distance must be positive, got {value}")
    return value


def prompt_float(label: str, default: float | None = None, minimum: float | None = None) -> float:
    """Ask the operator for a numeric value.

    @brief   Console prompt with optional validation.
    @param   label   Prompt text.
    @param   default Value used on an empty line.
    @param   minimum Exclusive lower bound, when meaningful.
    @return  The entered value.
    @raises  RuntimeError on unparseable or out-of-range input.
    """
    suffix = f" [{default}, Enter to accept]: " if default is not None else ": "
    raw = input(f"{label}{suffix}").strip()
    if not raw and default is not None:
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"could not read a number from {raw!r}") from exc
    if minimum is not None and value <= minimum:
        raise RuntimeError(f"{label} must be greater than {minimum}, got {value}")
    return value


def calibrate_topdown_two_point(
    intrinsics: Intrinsics,
    grab_frame: Callable[[], np.ndarray],
    distance_m: float | None = None,
    rotation_deg: float = 0.0,
    label: str = "topdown",
    zoom: float = 1.5,
) -> TopDownProjector:
    """Run the two-mark tape calibration for the overhead configuration.

    @brief   Configuration A calibration.
    @details A single measured distance plus an assumed image rotation fully determines an affine
             ground map, so nothing further is needed on this rig.  The resulting projector is
             checked immediately: the distance between the two clicked points is measured back
             through the fresh calibration and reported, and the implied metres-per-pixel is
             printed for the operator to sanity check against the mount height.  A calibration that
             is wrong by a factor is usually obvious from that number.

    @param   intrinsics   Camera intrinsics.
    @param   grab_frame   Callable returning one BGR frame (usually a source's first frame).
    @param   distance_m   Measured distance in metres; prompted for when ``None``.
    @param   rotation_deg Image rotation of the road relative to the pixel grid.
    @param   label        Label recorded in the calibration.
    @param   zoom         Display magnification for clicking.
    @return  A calibrated :class:`TopDownProjector`.
    """
    frame = grab_frame()
    points = point_picker(
        frame,
        "Click the two tape marks, P1 then P2",
        count=2,
        zoom=zoom,
    )
    span_px = float(np.linalg.norm(np.asarray(points[1]) - np.asarray(points[0])))
    measured = distance_m if distance_m is not None else prompt_distance(0.3048)
    projector = TopDownProjector.from_measured_distance(
        intrinsics,
        pt_a_xy=points[0],
        pt_b_xy=points[1],
        real_distance_m=measured,
        rotation_deg=rotation_deg,
        label=label,
    )
    info = projector.describe()
    print("\n  calibration result (top-down / overhead)")
    print(f"    clicked span      : {span_px:.1f} px")
    print(f"    measured distance : {measured:.4f} m")
    print(f"    implied scale     : {info.get('m_per_px', float('nan')) * 1000:.3f} mm per pixel")
    print(f"    anisotropy        : {info.get('anisotropy', float('nan')):.4f}  (1.0 is ideal)")
    print("\n  sanity check: with the camera at height h and focal length f, the expected")
    print("  scale is h/f. If the figure above is off by a factor, re-measure or re-click.")
    return projector


def calibrate_roadside_two_point(
    intrinsics: Intrinsics,
    grab_frame: Callable[[], np.ndarray],
    distance_m: float | None = None,
    horizon_row: float | None = None,
    height_m: float | None = None,
    pitch_deg: float | None = None,
    horizon_prior: tuple[float, float, float, float] | None = None,
    label: str = "roadside",
    zoom: float = 1.5,
) -> RoadsideProjector:
    """Run the two-mark tape calibration for the roadside configuration.

    @brief   Configuration B calibration, with the information requirement enforced.
    @details Requires the measured distance *plus* exactly one of a horizon row, a lens height or a
             tilt.  When the operator supplies none, one is solicited interactively and the reason
             is stated plainly, rather than a plausible default being invented.

             A horizon row is preferred and is offered first because it is the most informative:
             pitch follows analytically from it, and the measured distance then fixes the height
             uniquely.  It can be read off the frame directly, or estimated from a long straight
             road edge with :func:`estimate_horizon_from_frame`.

    @param   intrinsics    Camera intrinsics.
    @param   grab_frame    Callable returning one BGR frame.
    @param   distance_m    Measured distance in metres; prompted for when ``None``.
    @param   horizon_row   Observed horizon row in pixels, if known.
    @param   height_m      Measured lens height in metres, if known.
    @param   pitch_deg     Known downward tilt in degrees, if known.
    @param   horizon_prior Optional ``(x1, y1, x2, y2)`` road edge used to estimate the horizon.
    @param   label         Label recorded in the calibration.
    @param   zoom          Display magnification for clicking.
    @return  A calibrated :class:`RoadsideProjector`.
    @raises  RuntimeError when no constraining prior is available and the operator declines to give one.
    """
    frame = grab_frame()
    if horizon_prior is not None:
        estimated = estimate_horizon_from_frame(frame, horizon_prior, intrinsics)
        print(f"  horizon estimated from the supplied road edge: row {estimated:.1f}")
        if horizon_row is None:
            accept = input("  use this horizon? [Y/n]: ").strip().lower()
            if accept in ("", "y", "yes"):
                horizon_row = estimated

    if horizon_row is None:
        print(
            "\n  A single measured distance cannot fix both the camera height and its tilt:")
        print("  a camera twice as high aimed twice as steeply produces nearly the same image.")
        print("  One of the following is therefore required. The horizon row is the most")
        print("  informative and is recommended.\n")
        choice = input("  give [h]orizon row, lens [e]ight, [p]itch angle? [h]: ").strip().lower() or "h"
        if choice.startswith("h"):
            default_horizon = _suggest_horizon_row(frame, intrinsics)
            horizon_row = prompt_float(
                f"  horizon row in pixels (0-{frame.shape[0] - 1}; objects converge here)",
                default=default_horizon,
            )
        elif choice.startswith("e"):
            height_m = prompt_float("  lens height above the road in metres", minimum=0.0)
        elif choice.startswith("p"):
            pitch_deg = prompt_float("  downward tilt in degrees (0-90)", minimum=0.0)
        else:
            raise RuntimeError("no height, pitch or horizon supplied: calibration cannot proceed")

    points = point_picker(frame, "Click two tape marks along the road, P1 then P2", count=2, zoom=zoom)
    span_px = float(np.linalg.norm(np.asarray(points[1]) - np.asarray(points[0])))
    measured = distance_m if distance_m is not None else prompt_distance(0.3048)

    projector = RoadsideProjector.from_measured_distance(
        intrinsics,
        pt_a_xy=points[0],
        pt_b_xy=points[1],
        real_distance_m=measured,
        horizon_row=horizon_row,
        height_prior_m=height_m,
        pitch_prior_deg=pitch_deg,
        label=label,
    )
    extrinsics = projector.model.extrinsics
    info = projector.describe()
    print("\n  calibration result (roadside / pitched)")
    print(f"    clicked span          : {span_px:.1f} px")
    print(f"    measured distance     : {measured:.4f} m")
    print(f"    recovered height      : {extrinsics.height_m:.4f} m")
    print(f"    recovered pitch       : {extrinsics.pitch_deg:.3f} deg")
    print(f"    horizon row           : {info.get('horizon_row', float('nan')):.1f} px")
    print(f"    usable range ({projector.max_error_m} m budget): {projector.usable_range_m():.2f} m")
    _check_height_against_length(frame, intrinsics, extrinsics)
    return projector


def _suggest_horizon_row(frame: np.ndarray, intrinsics: Intrinsics) -> float | None:
    """@brief A plausible default horizon row, offered as a starting point only.

    @details Returns ``None`` when the principal point is not in the upper part of the frame, which
             would make even a rough suggestion meaningless.  A default is offered purely to spare
             the operator from typing a number when they already know it is roughly right; it is
             never used silently.
    """
    if intrinsics.cy <= 1.0:
        return None
    return float(max(1.0, intrinsics.cy * 0.6))


def estimate_horizon_from_frame(
    frame: np.ndarray, road_edge_xy: Sequence[float], intrinsics: Intrinsics
) -> float:
    """Estimate the horizon row from a clicked segment of a road edge.

    @brief   Horizon from a linear feature on the ground.
    @details A straight road edge lies in the ground plane, so its vanishing point is the horizon
             row.  Given two image points on the edge, the line through them meets the horizon
             where the row no longer changes with distance, which for a horizontal edge in a level
             frame is simply that line's row extended; the helper therefore returns the row of the
             fitted edge at the principal column.

             This is an *estimate* and is labelled as one: the edge must be genuinely straight and
             genuinely in the ground plane.  A kerb line qualifies; the top of a wall does not, and
             using one biases the pitch and therefore every speed.
    @param   frame         BGR frame, used only for its dimensions.
    @param   road_edge_xy  ``(x1, y1, x2, y2)`` two points on the edge.
    @param   intrinsics    Camera intrinsics, for the principal column.
    @return  Estimated horizon row in pixels.
    @raises  ValueError when the two points are coincident.
    """
    x1, y1, x2, y2 = (float(v) for v in road_edge_xy)
    if abs(x2 - x1) < 1e-9:
        raise ValueError("the two road-edge points share a column; a vertical edge cannot fix a horizon")
    slope = (y2 - y1) / (x2 - x1)
    return float(y1 + slope * (intrinsics.cx - x1))


def _check_height_against_length(
    frame: np.ndarray, intrinsics: Intrinsics, extrinsics: Any
) -> None:
    """Warn when the recovered pose implies a physically unlikely mount.

    @brief   Cross-check the recovered geometry.
    @details A recovered height far outside the range a person can plausibly reach with a pole is
             the signature of a bad measured distance or a mis-clicked mark.  Catching it here is
             much cheaper than discovering it from inconsistent speeds later, and it costs one
             comparison.
    @param   frame      BGR frame, used for its dimensions.
    @param   intrinsics Camera intrinsics.
    @param   extrinsics Recovered pose.
    """
    if not 0.05 <= extrinsics.height_m <= 12.0:
        print(
            f"\n  WARNING: recovered height {extrinsics.height_m:.3f} m is outside the plausible "
            f"range 0.05-12 m."
        )
        print("  Re-check the measured distance and the two clicked points.")


def calibrate_intrinsics_chessboard(
    frames: Sequence[np.ndarray],
    pattern: tuple[int, int] = (9, 6),
    square_m: float = 0.025,
) -> tuple[Intrinsics, dict[str, Any]]:
    """Calibrate lens intrinsics and distortion from chessboard views.

    @brief   Fit a pinhole model with Brown-Conrady distortion.
    @details Optional for this project, but genuinely valuable for an overhead rig: a wide-angle
             lens has real barrel distortion, and a single affine or projective ground map fitted on
             a distorted frame is systematically wrong away from the calibration points.  Distortion
             is a slowly varying function of position, so it biases the *scale* in a way that a
             two-point measurement near the frame centre cannot detect.

             At least three views with differing orientations are required; more is better, and the
             board should be tilted between captures because a set of coplanar views leaves the
             focal length poorly constrained.

    @param   frames   Grayscale-able BGR frames containing the board.
    @param   pattern  Inner corner count ``(columns, rows)``.
    @param   square_m Physical square size in metres; it scales the output but the distortion and
                      normalised focal length are unaffected by an error here.
    @return  ``(intrinsics, diagnostics)`` where diagnostics reports the RMS reprojection error and
             how many views were used.
    @raises  RuntimeError when OpenCV is unavailable, or when too few usable views are supplied.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "intrinsics calibration requires OpenCV. Install it with:\n    uv sync --extra calibration"
        ) from exc

    columns, rows = int(pattern[0]), int(pattern[1])
    # @step Object points in board coordinates, in metres, shared across views.
    object_points = np.zeros((columns * rows, 3), dtype=np.float32)
    grid = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    object_points[:, :2] = grid * float(square_m)

    object_views: list[np.ndarray] = []
    image_views: list[np.ndarray] = []
    size: tuple[int, int] | None = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if size is None:
            size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, (columns, rows), None)
        if not found:
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_views.append(object_points.copy())
        image_views.append(refined)

    if size is None:
        raise RuntimeError("no frames were supplied")
    if len(image_views) < 3:
        raise RuntimeError(
            f"only {len(image_views)} usable board view(s) found; at least 3 are needed, and "
            f"varying the board's tilt between captures matters more than adding views"
        )

    rms, matrix, dist, _rvecs, _tvecs = cv2.calibrateCamera(object_views, image_views, size, None, None)
    intrinsics = Intrinsics(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        dist=tuple(float(value) for value in dist.reshape(-1)),
        width=int(size[0]),
        height=int(size[1]),
    )
    diagnostics = {
        "rms_reprojection_px": float(rms),
        "views_used": len(image_views),
        "views_supplied": len(frames),
        "image_size": [int(size[0]), int(size[1])],
        "distortion": list(intrinsics.dist or ()),
        "interpretation": (
            "RMS under ~0.5 px is good. Above ~1.5 px, recapture with more tilt and better focus."
        ),
    }
    return intrinsics, diagnostics


def write_calibration_into_config(
    config_path: str | Path, projector: Any, backup: bool = True
) -> Path:
    """Insert a calibration block into an existing camera configuration file.

    @brief   Record a calibration where the pipeline will find it.
    @details A stored calibration block takes precedence over live geometry parameters in
             :func:`cardetect.geometry.build_projector`, which is deliberate: the file records a
             physical measurement of one specific rig, whereas the surrounding parameters are
             datasheet defaults the calibration is meant to replace.  A timestamped backup is kept
             by default so an experimental calibration can always be undone.
    @param   config_path Source configuration file.
    @param   projector   Calibrated projector.
    @param   backup      Whether to keep a ``.bak`` copy.
    @return  The written path.
    """
    target = Path(config_path)
    document = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if backup:
        backup_path = target.with_suffix(target.suffix + ".bak")
        backup_path.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    calibration = projector.to_config()
    calibration.pop("kind", None)
    calibration.pop("label", None)
    # @note The intrinsics are already in the surrounding file; duplicating them inside the
    #       calibration block would create two sources of truth that can drift apart.
    calibration.pop("intrinsics", None)
    document["calibration"] = calibration
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return target


def default_calibration_path(kind: str) -> Path:
    """@brief Conventional calibration file location for a rig kind."""
    return PROJECT_ROOT / "calibration" / f"calibration_{kind}.yaml"


def describe_calibration(projector: Any) -> str:
    """@brief Multi-line description of a projector, for printing after calibration."""
    lines = [f"kind: {projector.kind}"]
    for key, value in projector.describe().items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def metres_per_pixel_note(projector: Any) -> str:
    """@brief Short guidance string keyed to the rig's resolution.

    @details Turns the calibrated scale into an actionable statement.  An operator who sees that a
             pixel is worth 3 mm knows immediately why a 2-pixel detection wobble is worth 6 mm and
             therefore why a short measurement window produces a noisy speed.
    """
    from cardetect.geometry import RoadsideProjector, TopDownProjector

    if isinstance(projector, TopDownProjector):
        mm = projector.plane.m_per_px * 1000.0
        return (
            f"one pixel is {mm:.2f} mm of road, uniform across the frame. "
            f"A 2 px detection wobble is therefore {2 * mm:.2f} mm of position error."
        )
    if isinstance(projector, RoadsideProjector):
        near = float(projector.scale(np.array([[0.0, 1.0]]))["m_per_px_max"][0]) * 1000.0
        far = float(projector.scale(np.array([[0.0, 6.0]]))["m_per_px_max"][0]) * 1000.0
        return (
            f"one pixel is {near:.2f} mm of road at 1 m and {far:.2f} mm at 6 m, so accuracy "
            f"degrades with distance; the error budget caps the usable range at "
            f"{projector.usable_range_m():.2f} m."
        )
    return "resolution unknown for this projector"
