"""cardetect: metric velocity estimation for 1:64-scale vehicles from one camera.

@file    __init__.py
@brief   Package entry point.
@details The public surface is intentionally narrow.  Heavy dependencies (OpenCV,
         ultralytics, torch) are imported lazily inside the modules that need them, so
         ``import cardetect`` and ``import cardetect.geometry`` stay cheap enough to run on
         a Raspberry Pi Zero 2 W.  See ``pyproject.toml`` for the dependency layering rules.

@note    The package is split so that the numerically delicate part -- the part that turns
         pixels into metres -- has no dependency beyond NumPy and is therefore fully
         unit-testable headless:

             geometry   camera models, projections, calibration solvers
             detect     YOLO inference wrapper                (needs ultralytics)
             track      multi-object tracking                 (needs ultralytics)
             speed      robust velocity estimation            (numpy only)
             pipeline   orchestration                         (needs opencv)
             cli        command line entry point
"""

from __future__ import annotations

from cardetect.geometry import (
    AffineGroundPlane,
    CameraModel,
    Extrinsics,
    GroundPlaneProjector,
    GeometryError,
    Intrinsics,
    RoadsideProjector,
    TopDownProjector,
    build_projector,
)

__version__ = "0.1.0"

__all__ = [
    "AffineGroundPlane",
    "CameraModel",
    "Extrinsics",
    "GeometryError",
    "GroundPlaneProjector",
    "Intrinsics",
    "RoadsideProjector",
    "TopDownProjector",
    "__version__",
    "build_projector",
]
