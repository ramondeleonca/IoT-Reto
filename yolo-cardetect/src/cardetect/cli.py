"""Command-line entry point.

@file    cli.py
@brief   The ``cardetect`` command: run, calibrate, inspect and self-test.
@details Subcommands are organised by what the operator is trying to establish:

             ``doctor``    is this machine able to run the project at all?
             ``check``     is this configuration internally consistent, and what will it do?
             ``calibrate`` turn a physical measurement into a stored calibration.
             ``preview``   see the rig geometry and resolution without a detector or a model.
             ``run``       process a camera or a file and measure speeds.

         ``preview`` and ``check`` are the two commands worth having on a device that cannot run
         the detector at all: they answer "is my geometry right?" and "what will happen when I
         point this at the track?" using only NumPy and the configuration file, which is exactly
         the situation on a Raspberry Pi Zero 2 W before its model is exported.

         Every subcommand prints the effective configuration before doing anything, because the
         most common confusing result during development is having loaded the wrong camera file.

@note    Heavy dependencies are imported inside the subcommand handlers rather than at module
         scope, so ``cardetect --help`` and ``cardetect check`` work on a machine with neither
         torch nor OpenCV installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cardetect.config import (
    PROJECT_ROOT,
    ConfigError,
    config_summary,
    has_metric_scale,
    load_config,
    resolve_paths,
    validate_extrinsics_pitch,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    @brief   Define the CLI surface.
    @details Each subcommand takes ``--config`` for consistency, and the camera geometry flags
             exist on every command that needs geometry so that a quick experiment does not require
             editing a YAML file first.  Flags always win over the file, which is stated in the help
             text because the precedence is observable.
    @return  Configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="cardetect",
        description=(
            "Detect 1:64-scale vehicles and estimate their speed from one pole-mounted camera. "
            "Both an overhead and a roadside mounting are supported; choose with --config."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  cardetect doctor\n"
            "  cardetect check --config config/camera_topdown.yaml\n"
            "  cardetect preview --config config/camera_topdown.yaml --save-geometry\n"
            "  cardetect calibrate topdown --config config/camera_topdown.yaml\n"
            "  cardetect run --config config/camera_topdown.yaml --source 0\n"
        ),
    )
    parser.add_argument("--version", action="version", version="cardetect 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- doctor -------------------------------------------------------------------------
    doctor = subparsers.add_parser(
        "doctor",
        help="report which optional dependencies and accelerators are available",
        description=(
            "Report the state of every optional dependency and accelerator this project can use. "
            "Run this first on a new device: it distinguishes 'the geometry works' from 'the "
            "detector can run', which on a Raspberry Pi Zero 2 W are different questions."
        ),
    )
    doctor.set_defaults(handler=_cmd_doctor)

    # --- check --------------------------------------------------------------------------
    check = subparsers.add_parser(
        "check",
        help="validate a configuration and report what it will do",
        description=(
            "Load, merge and validate a configuration, then report the geometry it describes. "
            "Needs no detector, no model and no camera, so it is the fastest way to confirm a rig "
            "is set up correctly before a run."
        ),
    )
    _add_config_args(check)
    check.set_defaults(handler=_cmd_check)

    # --- preview ------------------------------------------------------------------------
    preview = subparsers.add_parser(
        "preview",
        help="summarise the rig geometry and resolution, optionally saving figures",
        description=(
            "Report the effective metres-per-pixel, the usable road range and the positional error "
            "implied by detector jitter. This is the command to use when choosing where to mount "
            "the camera, because it answers how accurately each part of the road can be measured "
            "rather than how the picture looks."
        ),
    )
    _add_config_args(preview)
    preview.add_argument(
        "--save-geometry",
        action="store_true",
        help="write the resolution and trajectory figures into the run directory",
    )
    preview.add_argument(
        "--extent-m",
        type=float,
        default=None,
        help="how far down the road to sample, in metres (default: rig dependent)",
    )
    preview.set_defaults(handler=_cmd_preview)

    # --- calibrate ----------------------------------------------------------------------
    calibrate = subparsers.add_parser(
        "calibrate",
        help="measure a rig and store its calibration",
        description=(
            "Turn a physical measurement into a stored calibration file. The two-mark tape workflow "
            "needs no printing. On a roadside rig a single distance is not sufficient on its own and "
            "the command will ask for a horizon row, a lens height or a tilt."
        ),
    )
    _add_config_args(calibrate)
    calibrate.add_argument(
        "target",
        choices=["topdown", "roadside", "intrinsics"],
        help=(
            "topdown: two marks plus a measured distance. "
            "roadside: two marks plus a distance plus one pose constraint. "
            "intrinsics: chessboard lens calibration from saved frames."
        ),
    )
    calibrate.add_argument("--source", default=None, help="camera index or image file for the still frame")
    calibrate.add_argument("--distance-m", type=float, default=None, help="measured distance between the marks")
    # @note --height-m and --pitch-deg arrive via _add_config_args and are deliberately not
    #       redefined here: argparse rejects a duplicate option string outright, and the shared
    #       flags already serve double duty as both calibration priors and configuration overrides.
    calibrate.add_argument("--horizon-row", type=float, default=None, help="horizon row in pixels")
    calibrate.add_argument(
        "--road-edge",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="two points on a straight kerb line, used to estimate the horizon",
    )
    calibrate.add_argument("--rotation-deg", type=float, default=None, help="image rotation of the road")
    calibrate.add_argument("--zoom", type=float, default=1.5, help="magnification while clicking marks")
    calibrate.add_argument(
        "--board-frames",
        default=None,
        help="directory of chessboard frames, for the intrinsics target",
    )
    calibrate.add_argument(
        "--board-pattern",
        default="9x6",
        help="inner corner count of the chessboard, for example 9x6",
    )
    calibrate.add_argument("--board-square-m", type=float, default=0.025, help="chessboard square size, metres")
    calibrate.add_argument("--output", default=None, help="where to write the calibration file")
    calibrate.add_argument(
        "--write-config",
        action="store_true",
        help="also insert the calibration into the configuration file (keeps a .bak copy)",
    )
    calibrate.set_defaults(handler=_cmd_calibrate)

    # --- run ----------------------------------------------------------------------------
    run = subparsers.add_parser(
        "run",
        help="process a camera or a video file and measure speeds",
        description=(
            "Run the full pipeline. The annotated video, a per-frame measurement table, per-track "
            "summaries and figures are written into a timestamped directory under output.directory."
        ),
    )
    _add_config_args(run)
    run.add_argument("--source", default=None, help="camera index, video file or stream URL (default: from config)")
    run.add_argument("--mode", choices=["stream", "burst"], default=None, help="capture mode override")
    run.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    run.add_argument("--display", action="store_true", help="show a live window (press q to stop)")
    run.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    run.add_argument("--model", default=None, help="detector weights override, for example yolo11n.pt")
    run.add_argument("--imgsz", type=int, default=None, help="inference size override, for example 320")
    run.add_argument("--device", default=None, help="device override: auto, cpu, mps, cuda")
    run.add_argument("--conf", type=float, default=None, help="confidence threshold override")
    run.add_argument(
        "--no-metric",
        action="store_true",
        help="run without metric scale: tracks and pixels only, no speeds",
    )
    run.set_defaults(handler=_cmd_run)

    return parser


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """@brief Attach the shared configuration and geometry arguments."""
    parser.add_argument(
        "--config",
        default=None,
        help="camera configuration file, for example config/camera_topdown.yaml",
    )
    parser.add_argument("--height-m", type=float, default=None, help="lens height above the road, metres")
    parser.add_argument("--pitch-deg", type=float, default=None, help="downward tilt in degrees")
    parser.add_argument("--measured-distance-m", type=float, default=None, help="tape distance, metres")


def _collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Turn command-line geometry flags into a nested configuration override.

    @brief   Map flat CLI arguments onto the nested configuration structure.
    @details Flags win over the file, which is documented in the help text because the precedence is
             observable.  Only explicitly supplied flags appear in the result, so a flag that is
             absent cannot silently blank out a value from the file.
    @param   args Parsed arguments.
    @return  Possibly empty override mapping.
    """
    overrides: dict[str, Any] = {}
    if getattr(args, "height_m", None) is not None:
        overrides["height_m"] = float(args.height_m)
    if getattr(args, "measured_distance_m", None) is not None:
        overrides["measured_distance_m"] = float(args.measured_distance_m)
    if getattr(args, "pitch_deg", None) is not None:
        overrides.setdefault("extrinsics", {})["pitch_deg"] = float(args.pitch_deg)
    return overrides


def _load(args: argparse.Namespace) -> dict[str, Any]:
    """@brief Load the configuration with CLI overrides applied."""
    return load_config(args.config, overrides=_collect_overrides(args))


def _print_banner(cfg: dict[str, Any]) -> None:
    """@brief Print the effective configuration before doing any work."""
    print("cardetect")
    print("-" * 66)
    print(config_summary(cfg))
    if not has_metric_scale(cfg):
        print(
            "  NOTE         : no metric scale is configured, so speeds will not be reported. "
            "Calibrate with 'cardetect calibrate', or supply --height-m."
        )
    print("-" * 66)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report dependency and accelerator availability.

    @brief   Environment diagnosis.
    @details Checks are ordered from most fundamental to most optional, and each reports what it
             enables.  On a Raspberry Pi Zero 2 W the expected outcome is NumPy plus OpenCV, no
             torch, and CPU only -- which is a fully supported configuration for geometry work and
             for a burst-capture deployment once a model has been exported elsewhere.
    @param   args Parsed arguments (unused).
    @return  Exit code.
    """
    print("cardetect environment")
    print("-" * 66)

    def report(name: str, ok: bool, detail: str, enables: str) -> None:
        """@brief Print one dependency line."""
        mark = "ok  " if ok else "MISS"
        print(f"  [{mark}] {name:16} {detail}")
        if not ok and enables:
            print(f"         enables: {enables}")

    try:
        import numpy as np

        report("numpy", True, f"{np.__version__}", "")
    except ImportError:
        report("numpy", False, "not installed", "everything: install with 'uv sync'")

    try:
        import yaml

        report("pyyaml", True, f"{yaml.__version__}", "")
    except ImportError:
        report("pyyaml", False, "not installed", "configuration loading")

    try:
        import cv2

        report("opencv", True, f"{cv2.__version__}", "")
    except ImportError:
        report("opencv", False, "not installed", "capture, overlays, interactive calibration ('--extra vision')")

    torch_device = "unavailable"
    try:
        import torch

        if torch.cuda.is_available():
            torch_device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch_device = "mps (Apple)"
        else:
            torch_device = "cpu"
        report("torch", True, f"{torch.__version__}, device {torch_device}", "")
    except ImportError:
        report("torch", False, "not installed", "detector inference ('--extra vision')")

    try:
        import ultralytics

        report("ultralytics", True, f"{ultralytics.__version__}", "")
    except ImportError:
        report("ultralytics", False, "not installed", "detection and tracking ('--extra vision')")

    try:
        import lap  # noqa: F401

        report("lap", True, "installed", "")
    except ImportError:
        report(
            "lap",
            False,
            "not installed",
            "ByteTrack and BoT-SORT association; NOT a dependency of ultralytics itself, so it "
            "must be installed explicitly or tracking fails only once it starts",
        )

    try:
        import matplotlib

        report("matplotlib", True, f"{matplotlib.__version__}", "")
    except ImportError:
        report("matplotlib", False, "not installed", "speed and resolution figures")

    print("-" * 66)
    print("  a NumPy-only environment can still do geometry, calibration solving and 'preview'.")
    print("  detection needs torch and ultralytics; tracking additionally needs lap.")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Validate a configuration and report the geometry it describes.

    @brief   Configuration check.
    @details Builds the projector, which exercises every geometry code path the run will use, and
             reports the resolution figures.  A failure here is always cheaper than a failure
             three frames into a run.
    @param   args Parsed arguments.
    @return  Exit code.
    """
    cfg = _load(args)
    _print_banner(cfg)
    try:
        from cardetect.geometry import build_projector

        projector = build_projector(cfg)
    except Exception as exc:
        print(f"\nFAILED to build the ground geometry: {exc}", file=sys.stderr)
        return 2
    print("geometry resolved OK")
    for key, value in projector.describe().items():
        print(f"  {key:22}: {value}")
    from cardetect.calibration import metres_per_pixel_note

    print(f"\n  {metres_per_pixel_note(projector)}")
    warnings = validate_extrinsics_pitch(cfg)
    for warning in warnings:
        print(f"  WARNING: {warning}")
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    """Report the rig's achievable accuracy across the road.

    @brief   Resolution preview, with no detector required.
    @details This is the command for choosing a mounting position.  It reports where on the road
             the rig can measure to within the configured error budget, which is a geometric
             property independent of the detector and therefore knowable before any model runs.
    @param   args Parsed arguments.
    @return  Exit code.
    """
    cfg = _load(args)
    _print_banner(cfg)
    try:
        from cardetect.geometry import build_projector

        projector = build_projector(cfg)
    except Exception as exc:
        print(f"\nFAILED to build the ground geometry: {exc}", file=sys.stderr)
        return 2

    pixel_sigma = float(cfg.get("speed", {}).get("pixel_sigma_px", 2.0))
    budget = cfg.get("max_error_m")
    extent = args.extent_m
    if extent is None:
        kind = cfg.get("kind")
        if kind == "topdown":
            # @note The overhead frame is short in Z, so a small extent shows detail rather than
            #       a large empty region.
            extent = 3.0 if hasattr(projector, "usable_range_m") is False else 3.0
            half_width = 2.0
        else:
            extent = float(min(projector.usable_range_m() * 1.3, 25.0)) if hasattr(projector, "usable_range_m") else 10.0
            half_width = max(1.5, extent * 0.2)
    else:
        half_width = max(1.5, extent * 0.25)

    depths = np.linspace(0.25, extent, 8)
    points = np.column_stack([np.zeros_like(depths), depths])
    scales = projector.scale(points)
    errors = projector.position_sigma_m(points, pixel_sigma)

    print(f"  detector jitter assumed : {pixel_sigma:.2f} px")
    if budget:
        print(f"  positional error budget : {budget:.3f} m")
    print("\n  along the road centre line:")
    print(f"    {'depth (m)':>10}  {'mm/px (worst)':>14}  {'anisotropy':>11}  {'1-sigma error':>14}  status")
    for index, depth in enumerate(depths):
        within = "ok" if budget is None or errors[index] <= budget else "outside budget"
        print(
            f"    {depth:10.2f}  {scales['m_per_px_max'][index] * 1000:14.3f}  "
            f"{scales['anisotropy'][index]:11.3f}  {errors[index] * 1000:11.2f} mm  {within}"
        )

    if hasattr(projector, "usable_range_m"):
        print(f"\n  usable range at the configured budget: {projector.usable_range_m():.2f} m")
    from cardetect.calibration import metres_per_pixel_note

    print(f"  {metres_per_pixel_note(projector)}")

    if args.save_geometry:
        from cardetect import visualize

        out = resolve_paths(cfg)["output"]["directory"] / "preview"
        out.mkdir(parents=True, exist_ok=True)
        figure = visualize.ground_resolution_figure(
            projector,
            out / "ground_resolution.png",
            extent_m=(half_width, extent),
            pixel_sigma_px=pixel_sigma,
            max_error_m=budget,
        )
        if figure:
            print(f"\n  wrote {figure}")
        else:
            print("\n  matplotlib is not installed; no figure written")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Run an interactive calibration and store the result.

    @brief   Calibration entry point.
    @details Delegates the actual geometry to :mod:`cardetect.calibration`, which separates the
             interactive front end from the solving logic so the solvers can be tested headlessly.
    @param   args Parsed arguments.
    @return  Exit code.
    """
    from cardetect import calibration

    cfg = _load(args)

    if args.target == "intrinsics":
        return _calibrate_intrinsics(args, cfg)

    if args.output:
        output = Path(args.output)
    else:
        output = calibration.default_calibration_path(args.target)

    # @step Resolve the still frame the operator will click on.
    frame = _grab_still(args, cfg)
    if frame is None:
        return 2

    intrinsics_cfg = dict(cfg.get("intrinsics", {}))
    if frame is not None:
        height_px, width_px = frame.shape[:2]
        # @note Intrinsics must describe the frame actually captured.  A camera that silently
        #       substitutes a different resolution would otherwise invalidate every projection
        #       by a constant scale factor that looks like a calibration error.
        intrinsics_cfg["width"] = int(width_px)
        intrinsics_cfg["height"] = int(height_px)
    from cardetect.geometry import Intrinsics

    intrinsics = Intrinsics.from_config(intrinsics_cfg)
    print(f"  intrinsics: {width_px}x{height_px}, fx={intrinsics.fx:.2f}px fy={intrinsics.fy:.2f}px")
    if args.target == "topdown" and intrinsics_cfg.get("hfov_deg") is not None:
        print("  NOTE: intrinsics come from the datasheet field of view, so the scale is only as")
        print("        good as that figure. 'calibrate intrinsics' measurably improves on it.")

    try:
        if args.target == "topdown":
            projector = calibration.calibrate_topdown_two_point(
                intrinsics,
                grab_frame=lambda: frame,
                distance_m=args.distance_m,
                rotation_deg=args.rotation_deg or 0.0,
                zoom=args.zoom,
            )
        else:
            projector = calibration.calibrate_roadside_two_point(
                intrinsics,
                grab_frame=lambda: frame,
                distance_m=args.distance_m,
                horizon_row=args.horizon_row,
                height_m=args.height_m or cfg.get("height_m"),
                pitch_deg=args.pitch_deg,
                horizon_prior=tuple(args.road_edge) if args.road_edge else None,
                zoom=args.zoom,
            )
    except RuntimeError as exc:
        print(f"\ncalibration failed: {exc}", file=sys.stderr)
        return 1

    saved = calibration.save_calibration(
        output,
        projector,
        extra={
            "source": str(args.source),
            "measured_distance_m": args.distance_m,
            "note": "two-mark tape calibration; see README for what each geometry can determine",
        },
    )
    print(f"\n  calibration written to {saved}")

    if args.write_config and args.config:
        written = calibration.write_calibration_into_config(args.config, projector)
        print(f"  calibration inserted into {written} (backup kept as .bak)")
        print("  a stored calibration takes precedence over the geometry parameters in that file")
    else:
        print(f"  to use it, add 'calibration:' from {saved.name} into {args.config or 'your camera config'},")
        print("  or re-run with --write-config")

    print(f"\n  {calibration.metres_per_pixel_note(projector)}")
    return 0


def _calibrate_intrinsics(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    """Calibrate lens intrinsics from a directory of chessboard frames.

    @brief   Lens calibration entry point.
    @details Frames are read from a directory so the operator can capture a good set with whatever
             tool they already have.  This is optional for the overhead rig but valuable for a wide
             lens, where distortion biases the ground scale in a way a centre-frame two-point
             measurement cannot detect.
    @param   args Parsed arguments.
    @param   cfg  Effective configuration.
    @return  Exit code.
    """
    import cv2

    from cardetect import calibration

    if not args.board_frames:
        print(
            "the intrinsics target needs --board-frames pointing at a directory of chessboard "
            "images, and --board-pattern matching the board",
            file=sys.stderr,
        )
        return 2
    directory = Path(args.board_frames)
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 2
    files = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    )
    if not files:
        print(f"no image files found in {directory}", file=sys.stderr)
        return 2
    frames = []
    for path in files:
        image = cv2.imread(str(path))
        if image is not None:
            frames.append(image)
    print(f"  reading {len(frames)} frame(s) from {directory}")
    try:
        columns, rows = (int(part) for part in str(args.board_pattern).lower().split("x"))
    except ValueError:
        print(f"could not parse --board-pattern {args.board_pattern!r}; expected something like 9x6", file=sys.stderr)
        return 2

    try:
        intrinsics, diagnostics = calibration.calibrate_intrinsics_chessboard(
            frames, pattern=(columns, rows), square_m=args.board_square_m
        )
    except RuntimeError as exc:
        print(f"\nintrinsics calibration failed: {exc}", file=sys.stderr)
        return 1

    print("\n  intrinsics calibration result")
    print(f"    fx={intrinsics.fx:.3f}  fy={intrinsics.fy:.3f}  cx={intrinsics.cx:.3f}  cy={intrinsics.cy:.3f}")
    print(f"    distortion: {intrinsics.dist}")
    print(f"    RMS reprojection error: {diagnostics['rms_reprojection_px']:.4f} px")
    print(f"    views used: {diagnostics['views_used']} of {diagnostics['views_supplied']}")

    if args.output:
        output = Path(args.output)
    else:
        output = calibration.default_calibration_path("intrinsics")
    output.parent.mkdir(parents=True, exist_ok=True)
    import yaml as _yaml

    output.write_text(
        _yaml.safe_dump({"intrinsics": intrinsics.to_config() if hasattr(intrinsics, "to_config") else _intrinsics_dict(intrinsics), "diagnostics": diagnostics}, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\n  intrinsics written to {output}")
    print("  copy the 'intrinsics' block into your camera configuration file")
    if args.write_config and args.config:
        calibration.write_calibration_into_config(args.config, _IntrinsicsCarrier(intrinsics))
        print(f"  intrinsics inserted into {args.config} (backup kept as .bak)")
    return 0


class _IntrinsicsCarrier:
    """Adapter exposing a projector-like ``to_config`` for intrinsics-only output.

    @brief  Lets intrinsics reuse the configuration-writing helper.
    @details :func:`cardetect.calibration.write_calibration_into_config` expects a projector, but an
             intrinsics-only calibration has no geometry.  This adapter supplies the same interface
             so the two paths share one implementation rather than duplicating the YAML editing.
    """

    def __init__(self, intrinsics: Any) -> None:
        self.intrinsics = intrinsics

    def to_config(self) -> dict[str, Any]:
        """@brief Serialise as a calibration block carrying only intrinsics."""
        return {"intrinsics": _intrinsics_dict(self.intrinsics)}


def _intrinsics_dict(intrinsics: Any) -> dict[str, Any]:
    """@brief Serialise intrinsics to a plain mapping."""
    out: dict[str, Any] = {
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
    }
    if intrinsics.dist is not None:
        out["dist"] = [float(value) for value in intrinsics.dist]
    if intrinsics.width is not None:
        out["width"] = int(intrinsics.width)
    if intrinsics.height is not None:
        out["height"] = int(intrinsics.height)
    return out


def _grab_still(args: argparse.Namespace, cfg: dict[str, Any]) -> np.ndarray | None:
    """Obtain a single frame for interactive calibration.

    @brief   Grab one still, from a file or a camera.
    @details A file is preferred when supplied because it makes calibration reproducible: the same
             marks can be re-clicked later against the same image.  With a live camera the operator
             is given a moment to settle before the frame is captured, and the captured frame is
             written next to the calibration so the click positions remain auditable.
    @param   args Parsed arguments.
    @param   cfg  Effective configuration.
    @return  The frame, or ``None`` when it could not be obtained.
    """
    import cv2

    source_spec = args.source or cfg.get("source", {}).get("uri", 0)
    if isinstance(source_spec, str) and not str(source_spec).isdigit():
        path = Path(str(source_spec)).expanduser()
        if path.exists() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            image = cv2.imread(str(path))
            if image is None:
                print(f"could not read image {path}", file=sys.stderr)
                return None
            print(f"  using still image {path}")
            return image
        if path.exists():
            print(f"  grabbing the first frame of {path}")
            capture = cv2.VideoCapture(str(path))
            ok, image = capture.read()
            capture.release()
            if not ok:
                print(f"could not read a frame from {path}", file=sys.stderr)
                return None
            return image

    from cardetect.source import FrameSource, SourceError

    try:
        with FrameSource(source_spec, mode="stream", warmup_frames=10) as source:
            for frame in source:
                return frame.image
    except SourceError as exc:
        print(f"could not open {source_spec!r}: {exc}", file=sys.stderr)
        return None
    print(f"no frame available from {source_spec!r}", file=sys.stderr)
    return None


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the full pipeline.

    @brief   Run entry point.
    @details Reports a final summary and returns a non-zero status only for genuine failures, not
             for a run that measured nothing -- an empty result is a legitimate outcome when no
             vehicle passed, and conflating the two would make scripting awkward.
    @param   args Parsed arguments.
    @return  Exit code.
    """
    cfg = _load(args)
    if args.mode:
        cfg = {**cfg, "source": {**cfg.get("source", {}), "mode": args.mode}}
    if args.model:
        cfg = {**cfg, "detector": {**cfg.get("detector", {}), "model": args.model}}
    if args.imgsz:
        cfg = {**cfg, "detector": {**cfg.get("detector", {}), "imgsz": int(args.imgsz)}}
    if args.device:
        cfg = {**cfg, "detector": {**cfg.get("detector", {}), "device": args.device}}
    if args.conf is not None:
        cfg = {**cfg, "detector": {**cfg.get("detector", {}), "conf": float(args.conf)}}
    if args.no_video:
        cfg = {**cfg, "output": {**cfg.get("output", {}), "video": False}}
    if args.display:
        cfg = {**cfg, "output": {**cfg.get("output", {}), "display": True}}

    _print_banner(cfg)

    if args.no_metric and not has_metric_scale(cfg):
        print("running without metric scale: tracks and pixels only, no speeds will be reported")

    # @step Build the pipeline, which constructs the detector and therefore loads the model.
    try:
        from cardetect.pipeline import build_pipeline

        pipeline = build_pipeline(cfg)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"could not build the pipeline: {exc}", file=sys.stderr)
        return 2

    if not has_metric_scale(cfg):
        print(
            "\n  WARNING: this rig has no metric scale, so no speed can be reported. Tracks and "
            "the annotated video will still be produced. Calibrate with 'cardetect calibrate'."
        )

    print(f"  detector: {pipeline.detector.describe()}")

    from cardetect.source import FrameSource, SourceError

    source_spec = args.source or cfg.get("source", {}).get("uri", 0)
    source_cfg = cfg.get("source", {})
    try:
        source = FrameSource(
            source_spec,
            width=source_cfg.get("width"),
            height=source_cfg.get("height"),
            fps=source_cfg.get("fps"),
            mode=source_cfg.get("mode", "stream"),
            max_frames=args.max_frames,
            burst_seconds=float(source_cfg.get("burst_seconds", 2.0)),
            burst_preroll_s=float(source_cfg.get("burst_preroll_s", 0.3)),
            motion_threshold=float(source_cfg.get("motion_threshold", 0.02)),
        )
    except SourceError as exc:
        print(f"source error: {exc}", file=sys.stderr)
        return 2

    print(f"  source: {source.describe()}")
    print()
    try:
        with source:
            result = pipeline.run(source)
    except KeyboardInterrupt:
        print("\ninterrupted by the operator; partial output has been flushed")
        return 130
    except SourceError as exc:
        print(f"source error during capture: {exc}", file=sys.stderr)
        return 2

    print()
    print(result.summary())
    print()
    print(f"artefacts in {result.run_directory}")
    if result.csv_path:
        print(f"  measurements : {result.csv_path}")
    if result.video_path:
        print(f"  annotated    : {result.video_path}")
    if result.summary_path:
        print(f"  tracks       : {result.summary_path}")
    if result.report_path:
        print(f"  report       : {result.report_path}")
    for path in result.plot_paths:
        print(f"  figure       : {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``cardetect`` console script.

    @brief   Parse arguments and dispatch.
    @details User-facing failures are reported as messages rather than tracebacks, because the
             errors this tool produces are almost always configuration mistakes with an obvious
             fix, and a traceback buries that fix.
    @param   argv Argument list, defaulting to ``sys.argv[1:]``.
    @return  Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(f"file not found: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
