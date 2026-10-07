"""Configuration loading and merging.

@file    config.py
@brief   Load, merge and validate the YAML configuration tree.
@details Configuration is split by *lifetime* rather than by topic, which is the distinction
         that matters operationally:

             ``config/default.yaml``     describes the software -- detector, tracker, speed
                                         estimator, capture, output.  Shared by every rig.
             ``config/camera_*.yaml``    describes ONE physical camera and its mounting.  Swapping
                                         files must not be able to change detector behaviour, so
                                         only geometry and rig-specific tuning live here.

         A camera file is therefore loaded *over* the defaults, and the merge is recursive so a
         camera file can override a single nested key without restating its siblings.

@note    Only PyYAML and NumPy are required; importing this module is cheap enough for a
         Raspberry Pi Zero, which keeps the CLI able to print a config summary without loading a
         detector or a model.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ConfigError",
    "load_config",
    "deep_merge",
    "resolve_paths",
    "config_summary",
    "PROJECT_ROOT",
]

# @brief Repository root, derived from this file's location rather than the working directory, so
#        the package behaves identically no matter where it is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised for a missing, malformed or internally inconsistent configuration."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``.

    @brief   Merge nested mappings without restating siblings.
    @details Mappings are merged key by key and recursion continues into nested mappings.  Any
             non-mapping value in ``override`` replaces the corresponding value in ``base``
             outright, including lists -- a list is treated as a single value rather than being
             concatenated, because concatenating the detector's class filter with a rig-specific
             one would silently widen which objects are tracked.
    @param   base     Base mapping; not mutated.
    @param   override Mapping whose values take precedence; not mutated.
    @return  A new merged mapping.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    """@brief Read one YAML file and require a mapping at the top level."""
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # pragma: no cover - depends on malformed user input
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level, got {type(data).__name__}")
    return data


def load_config(
    camera_config: str | Path | None = None,
    defaults: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the defaults, then a camera file, then explicit overrides.

    @brief   Build the effective configuration in one deterministic order.
    @details The precedence order is fixed and documented because it is observable: CLI overrides
             beat the camera file, which beats the defaults.  A stored calibration block inside
             the camera file is *not* merged away -- the geometry factory prefers it over live
             parameters, which is deliberate since a stored calibration records a physical
             measurement of one rig.
    @param   camera_config Path to a camera configuration, or ``None`` for defaults only.
    @param   defaults      Path to the defaults file; ``config/default.yaml`` by default.
    @param   overrides     Highest-precedence mapping, typically from command-line arguments.
    @return  The merged configuration mapping.
    @raises  ConfigError if a required file is missing or malformed.
    """
    defaults_path = Path(defaults) if defaults else PROJECT_ROOT / "config" / "default.yaml"
    merged = _read_yaml(defaults_path)
    if camera_config is not None:
        merged = deep_merge(merged, _read_yaml(Path(camera_config)))
    if overrides:
        merged = deep_merge(merged, overrides)
    _validate(merged)
    return merged


def _validate(cfg: dict[str, Any]) -> None:
    """Check the merged configuration for problems that would otherwise fail confusingly later.

    @brief   Fail fast on inconsistent configuration.
    @details These checks exist because each of them corresponds to a failure that would
             otherwise surface deep inside the pipeline as an unrelated-looking error, or worse,
             as plausible but wrong output:

             * an unknown ``kind`` would be caught by the geometry factory, but catching it here
               with the file name in hand gives a better message;
             * a class filter that excludes every vehicle class means nothing is ever tracked;
             * confidence outside ``(0, 1)`` silently produces either no detections or all of them;
             * ``source.mode`` typos would otherwise fall through to a default.

    @param   cfg Merged configuration.
    @raises  ConfigError when the configuration is inconsistent.
    """
    kind = cfg.get("kind")
    if kind is None:
        raise ConfigError(
            "no 'kind' set: expected 'topdown' or 'roadside'. Pass a camera configuration with "
            "--config config/camera_topdown.yaml or config/camera_roadside.yaml"
        )
    if kind not in ("topdown", "roadside"):
        raise ConfigError(f"unknown 'kind' {kind!r}: expected 'topdown' or 'roadside'")

    detector = cfg.get("detector", {})
    classes = detector.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ConfigError("detector.classes must be a non-empty list of COCO class indices")
    if any(int(cls) < 0 for cls in classes):
        raise ConfigError(f"detector.classes must be non-negative indices, got {classes}")
    conf = float(detector.get("conf", 0.25))
    if not 0.0 < conf < 1.0:
        raise ConfigError(f"detector.conf must be in (0, 1), got {conf}")
    if int(detector.get("imgsz", 640)) <= 0:
        raise ConfigError("detector.imgsz must be positive")

    mode = cfg.get("source", {}).get("mode", "stream")
    if mode not in ("stream", "burst"):
        raise ConfigError(f"source.mode must be 'stream' or 'burst', got {mode!r}")

    point_mode = cfg.get("detection_point", {}).get("mode", "bottom_center")
    if point_mode not in ("bottom_center", "bottom_third", "center"):
        raise ConfigError(
            f"detection_point.mode must be 'bottom_center', 'bottom_third' or 'center', got {point_mode!r}"
        )

    tracker_type = str(cfg.get("tracker", {}).get("type", "bytetrack")).lower()
    if tracker_type not in ("bytetrack", "bot-sort", "botsort"):
        raise ConfigError(f"tracker.type must be 'bytetrack' or 'bot-sort', got {tracker_type!r}")

    # @step Geometry sanity: an obviously impossible mount should not reach the projection code,
    #       which is designed to accept any physically valid pose and would happily accept a
    #       nonsensical one too.
    for key in ("height_m",):
        if key in cfg and (not isinstance(cfg[key], (int, float)) or cfg[key] <= 0.0):
            raise ConfigError(f"{key} must be a positive number of metres, got {cfg[key]!r}")
    extrinsics = cfg.get("extrinsics") or {}
    if "pitch_deg" in extrinsics:
        pitch = float(extrinsics["pitch_deg"])
        if not 0.0 < pitch <= 90.0:
            raise ConfigError(
                f"extrinsics.pitch_deg must be in (0, 90] for a camera aimed at the road, got {pitch}"
            )
    max_error = cfg.get("max_error_m")
    if max_error is not None and (not isinstance(max_error, (int, float)) or max_error <= 0.0):
        raise ConfigError(f"max_error_m must be a positive number of metres, got {max_error!r}")


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with relative output paths anchored to the repository root.

    @brief   Make output paths independent of the working directory.
    @details Without this, running the CLI from a different directory would scatter ``runs/``
             output in unexpected places, and a stored calibration reference would break.  The
             original mapping is not modified.
    @param   cfg Merged configuration.
    @return  A copy with ``output.directory`` resolved to an absolute path.
    """
    out = copy.deepcopy(cfg)
    output = out.setdefault("output", {})
    directory = Path(str(output.get("directory", "runs")))
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    output["directory"] = directory
    return out


def config_summary(cfg: dict[str, Any]) -> str:
    """Format a short human-readable summary of the effective configuration.

    @brief   One-paragraph summary for logs and the CLI banner.
    @details Deliberately printed before any run so an operator can see which rig and which
             calibration route is active.  Almost every confusing measurement traced back to a
             session during development started with the wrong config being loaded, and this
             makes that visible in the first three lines of output.
    @param   cfg Merged configuration.
    @return  Multi-line summary string.
    """
    detector = cfg.get("detector", {})
    source = cfg.get("source", {})
    speed = cfg.get("speed", {})
    lines = [
        f"  rig          : {cfg.get('kind')} ({cfg.get('label', 'unlabelled')})",
        f"  camera       : {_camera_geometry_summary(cfg)}",
    ]
    calibration = cfg.get("calibration")
    if calibration:
        source_note = calibration.get("source") or calibration.get("plane", {}).get("source", "stored")
        lines.append(f"  calibration  : stored ({source_note})")
    elif cfg.get("kind") == "topdown" and "height_m" in cfg:
        lines.append(f"  calibration  : analytic height/focal-length route (h={cfg['height_m']} m)")
    elif cfg.get("kind") == "topdown" and "measured_distance_m" in cfg:
        lines.append(f"  calibration  : two-point tape ({cfg['measured_distance_m']} m)")
    elif "extrinsics" in cfg:
        extr = cfg["extrinsics"]
        lines.append(
            f"  calibration  : measured geometry (h={extr.get('height_m')} m, "
            f"pitch={extr.get('pitch_deg')} deg)"
        )
    elif "measured_distance_m" in cfg:
        lines.append(f"  calibration  : two-point tape ({cfg['measured_distance_m']} m) with prior")
    else:
        lines.append("  calibration  : NONE -- metric speed unavailable until calibrated")

    lines.extend(
        [
            f"  detector     : {detector.get('model')} @ {detector.get('imgsz')}px, "
            f"conf {detector.get('conf')}, classes {detector.get('classes')}, device {detector.get('device')}",
            f"  tracker      : {cfg.get('tracker', {}).get('type')}",
            f"  source       : {source.get('uri')} ({source.get('mode')} mode)",
            f"  speed window : {speed.get('window_s')} s, "
            f"max {speed.get('max_speed_mps')} m/s, jitter {speed.get('pixel_sigma_px')} px",
            f"  error budget : {cfg.get('max_error_m')} m",
        ]
    )
    if cfg.get("_warnings"):
        for warning in cfg["_warnings"]:
            lines.append(f"  WARNING      : {warning}")
    return "\n".join(lines)


def _camera_geometry_summary(cfg: dict[str, Any]) -> str:
    """@brief One-line description of the active camera pose."""
    intrinsics = cfg.get("intrinsics", {})
    size = f"{intrinsics.get('width', '?')}x{intrinsics.get('height', '?')}"
    if "hfov_deg" in intrinsics:
        lens = f"hfov {intrinsics['hfov_deg']} deg"
    elif "fx" in intrinsics:
        lens = f"fx {float(intrinsics['fx']):.1f} px"
    else:
        lens = "intrinsics unknown"
    extrinsics = cfg.get("extrinsics") or {}
    pose = ""
    if "height_m" in cfg:
        pose = f", h={cfg['height_m']} m"
    elif extrinsics:
        pose = f", h={extrinsics.get('height_m')} m, pitch={extrinsics.get('pitch_deg')} deg"
    return f"{size}, {lens}{pose}"


def validate_extrinsics_pitch(cfg: dict[str, Any]) -> list[str]:
    """Return non-fatal warnings about questionable geometry.

    @brief   Advisory checks that should not stop a run.
    @details These flag configurations that are legal but likely to produce disappointing
             results, and are surfaced as warnings rather than errors so an operator can still
             run an experiment:

             * a top-down rig whose mount is not within a few degrees of vertical -- the affine
               model is still the better estimator, but the accuracy advantage over the roadside
               rig is largely gone;
             * a roadside rig pitched so steeply that it degenerates toward overhead;
             * a roadside rig pitched so shallowly that the usable stretch of road is very short.

    @param   cfg Merged configuration.
    @return  List of warning strings, possibly empty.
    """
    warnings: list[str] = []
    kind = cfg.get("kind")
    extrinsics = cfg.get("extrinsics") or {}
    pitch = extrinsics.get("pitch_deg")
    if kind == "topdown" and pitch is not None and abs(float(pitch) - 90.0) > 10.0:
        warnings.append(
            f"topdown rig declares pitch {pitch} deg, more than 10 deg from vertical; "
            "consider the roadside configuration instead"
        )
    if kind == "roadside" and pitch is not None:
        pitch = float(pitch)
        if pitch > 70.0:
            warnings.append(
                f"roadside rig declares pitch {pitch} deg, which is close to overhead; "
                "the affine configuration would be better conditioned"
            )
        if pitch < 12.0:
            warnings.append(
                f"roadside rig declares pitch {pitch} deg, which is very shallow; "
                "expect the usable stretch of road to be short"
            )
    if kind == "roadside" and not (cfg.get("calibration") or extrinsics or "measured_distance_m" in cfg):
        warnings.append("roadside rig has no calibration; speed will not be reported")
    return warnings


def has_metric_scale(cfg: dict[str, Any]) -> bool:
    """@brief Whether the configuration supplies enough information for metric output.

    @details Used to decide whether to warn that speeds will be absent.  A rig can still be run
             and demoed without metric scale -- tracks and pixel motion are useful -- so this is
             reported rather than enforced.
    """
    if cfg.get("calibration"):
        return True
    if cfg.get("kind") == "topdown":
        return "height_m" in cfg or "measured_distance_m" in cfg
    return bool(cfg.get("extrinsics") or "measured_distance_m" in cfg)


def describe_intrinsics(cfg: dict[str, Any]) -> str:
    """@brief Readable description of the camera intrinsics, for error messages."""
    intrinsics = cfg.get("intrinsics", {})
    if "fx" in intrinsics:
        return f"fx={float(intrinsics['fx']):.2f}px fy={float(intrinsics.get('fy', intrinsics['fx'])):.2f}px"
    if "hfov_deg" in intrinsics:
        return f"{intrinsics.get('width')}x{intrinsics.get('height')} @ hfov {intrinsics['hfov_deg']}deg"
    return "unknown"


def geometric_mean_focal(intrinsics_cfg: dict[str, Any]) -> float:
    """@brief Geometric mean focal length in pixels, from either explicit or field-of-view form.

    @details Used by the analytic overhead calibration, which needs a single representative focal
             length.  The geometric mean is chosen because it keeps the resulting pixel-to-metre
             map area-preserving, which is the least-surprising choice when one scale must
             represent a possibly anamorphic sensor.
    """
    if "fx" in intrinsics_cfg:
        fx = float(intrinsics_cfg["fx"])
        fy = float(intrinsics_cfg.get("fy", fx))
    elif "hfov_deg" in intrinsics_cfg:
        width = intrinsics_cfg.get("width")
        height = intrinsics_cfg.get("height")
        if width is None or height is None:
            raise ConfigError("field-of-view intrinsics require width and height")
        fx = (float(width) / 2.0) / math.tan(math.radians(float(intrinsics_cfg["hfov_deg"])) / 2.0)
        vfov = intrinsics_cfg.get("vfov_deg")
        fy = fx if vfov is None else (float(height) / 2.0) / math.tan(math.radians(float(vfov)) / 2.0)
    else:
        raise ConfigError("intrinsics need either 'fx' or 'hfov_deg'")
    return math.sqrt(fx * fy)
