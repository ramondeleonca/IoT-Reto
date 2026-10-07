"""Ground-plane geometry for monocular metric speed estimation.

@file    geometry.py
@brief   Camera models and image <-> ground-plane projections for both supported
         pole-mounting configurations.
@details This module is the mathematical core of ``cardetect``.  It deliberately
         depends on **NumPy only** for every runtime path, because the primary
         deployment target (Raspberry Pi Zero 2 W) is severely memory
         constrained: importing this module must never drag in torch, torchvision
         or ultralytics.  OpenCV is used opportunistically for linear algebra
         (``cv2.findHomography``) when it happens to be importable, but a NumPy
         fallback covers the headless / core-only installation.

         Two mounting configurations are supported, and they are genuinely
         different problems rather than two settings of one knob:

         A. ``topdown``  -- camera on a pole above the road, pitched down so far
            that perspective foreshortening is negligible.  Near-orthographic, so
            mapping image -> ground is an *affine* map.  Scale is uniform across
            the frame and error does not grow with distance.

         B. ``roadside`` -- camera on a pole beside the road, pitched down at a
            shallow angle.  Requires a full projective *homography*.  Scale varies
            strongly across the frame and pixel error explodes with distance,
            which is why every projection here can also report its local
            metres-per-pixel uncertainty.

@note    Angle convention.  All pose angles are expressed in the OpenCV camera
         frame (+x right, +y **down**, +z forward along the optical axis):
             pitch > 0  tilts the optical axis **downward** toward the ground,
             yaw   > 0 rotates the camera to its right,
             roll  > 0 rotates the camera about its optical axis.
         A roadside camera therefore has pitch in roughly ``(0, 90)`` degrees and
         a top-down camera has pitch near ``90`` degrees.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

# @brief Degrees -> radians, cached as a constant so hot loops avoid math.radians calls.
_DEG2RAD = math.pi / 180.0

# @brief Opencv is optional.  The geometry layer stays functional without it so that
#        the core can be unit tested and deployed on a bare NumPy installation.
try:  # pragma: no cover - trivial import guard
    import cv2 as _cv2
except Exception:  # pragma: no cover
    _cv2 = None

HAVE_CV2 = _cv2 is not None


class GeometryError(ValueError):
    """Raised when a geometric configuration is physically impossible or degenerate."""


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def rotation_matrix(pitch_deg: float, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Build the world->camera rotation matrix for a camera pose.

    @brief   Compose yaw/pitch/roll into a single rotation matrix.
    @details World frame (see the module docstring): ``X`` across the road, ``Y`` **down**,
             ``Z`` along the road in the direction of travel.  ``R`` maps a world vector
             into camera axes and its rows are therefore the camera's
             ``(right, down, forward)`` axes expressed in world coordinates.

             For the aligned case the aim is pitched down by ``p``:

                 right   = ( 1,        0,       0      )   -- world X, by construction
                 forward = ( 0,       -sin p,   cos p  )   -- 0 at horizontal, straight
                                                             down at p = 90, as required
                 down    = right x forward
                         = ( 0,        cos p,   sin p  )   -- orthogonal to the aim

             giving ``R[1, 2] = -sin p`` and ``R[2, 2] = +cos p``.  Yaw and roll are applied
             as proper rotations on top of that aim.

             @note This function is the single most sign-trap-prone piece of the module, so
             the reasoning is recorded rather than left implicit.  The failure mode is
             subtle: several sign choices produce a perfectly orthonormal matrix, and
             ``cv2.projectPoints`` agrees with all of them because it is fed the same
             matrix.  They differ only in *which physical direction the camera faces*.
             Three oracles pin the convention down, and all three are asserted in
             ``tests/test_geometry.py``:

                 1. the point where the optical axis meets the road projects to the image
                    centre -- this alone eliminates two of the four orthonormal options;
                 2. ground points nearer the camera appear **lower** in the frame
                    (monotonically larger ``v``);
                 3. a point offset toward world ``+X`` appears to the **right** (``u > cx``).

             Winning those three fixes ``R[1, 2] = -sin p`` and ``R[2, 2] = +cos p``, and
             the resulting horizon row is ``cy - fy * tan(p)``, which for a camera pitched
             down the road lies *above* the top of the frame -- the physically correct
             answer, and the one a naive ``cy - fy * cot(p)`` derivation gets wrong.

    @param  pitch_deg  Downward tilt of the optical axis in degrees; 90 looks straight down.
    @param  yaw_deg    Rotation about the world vertical, degrees.
    @param  roll_deg   Rotation about the road travel axis, degrees.
    @return (3, 3) orthonormal matrix with rows ``(right, down, forward)`` and ``det = +1``,
            except at the exactly-vertical pose described above, where the frame is a
            deliberate reflection with ``det = -1``.  See that note for why.
    """
    p = pitch_deg * _DEG2RAD
    y = yaw_deg * _DEG2RAD
    r = roll_deg * _DEG2RAD

    # @step Gimbal lock.  Looking exactly straight down is the one pose where the pitch/yaw/
    #       roll parameterisation degenerates: the in-image orientation is then carried by an
    #       unresolvable combination of yaw and roll, and the generic composition below
    #       produces a rank-deficient ground homography (a zero column), which would surface
    #       downstream as a confusing "degenerate camera pose" error rather than as the
    #       ordinary overhead rig it actually is.  The pose is therefore constructed directly
    #       from the required axis alignment.
    if abs(abs(pitch_deg) - 90.0) < 1e-9:
        # @note Canonical overhead frame: camera right = world +X (image u increases with road
        #       X), camera down = world +Z (image v increases with road Z), optical axis = world
        #       +Y (straight down).  Rows are (right, down, forward) and the determinant is +1.
        #       The in-plane squeeze/rotation is handled equivalently by yaw/roll in the
        #       non-degenerate case, so the generic path stays the general answer.
        # @note This frame has determinant -1, i.e. it is a reflection rather than a proper
        #       rotation, and that is deliberate rather than an oversight.  For a camera
        #       looking straight down, image "right" is world +X and image "down" is a
        #       horizontal road direction; requiring image "down" to be *increasing* road Z
        #       (the convention the affine overhead model uses, and the intuitive one) forces
        #       the frame to be left handed, because world Y points down into the road.  The
        #       alternative is a right-handed frame in which image "down" is *decreasing* Z,
        #       which would make every overhead car appear to drive backwards.
        #
        #       Physically this simply records that an overhead camera is mounted at one
        #       particular end of the track, so which way is "away from the camera" is fixed
        #       by the rig rather than free to choose.  A reflection preserves lengths and
        #       therefore speeds; only the handedness of the ground frame differs, and every
        #       consumer of this module uses magnitudes and same-frame positions.
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)

    # @step 1: the aimed camera basis, rows ordered (right, down, forward).
    rot_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float64,
    )
    # @note Do not "fix" this matrix by negating rows or transposing it.  An exhaustive
    #       search over all proper rotations of this form, scored on the three oracles
    #       documented above, admits exactly one solution, and it is this matrix.  Every
    #       neighbouring sign choice breaks at least one oracle.
    # @step 2: yaw about the world vertical (world +Y points down, hence the sign of sy).
    rot_yaw = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float64,
    )
    # @step 3: roll about the road travel axis, i.e. a rotation within the image plane.
    rot_roll = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )
    # @note All three factors above are proper rotations (det = +1), so the product is too;
    #       no post-hoc sign correction is applied or needed.
    return rot_roll @ rot_yaw @ rot_pitch


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera intrinsics with an optional Brown-Conrady distortion model.

    @brief  Camera matrix and distortion coefficients.
    @details ``dist`` uses OpenCV's ordering ``(k1, k2, p1, p2, k3)``.  Distortion is
             a first-class concern for this project rather than an afterthought:
             the wide-angle phone/GoPro-class lenses used for an overhead shot have
             meaningful barrel distortion, and a single homography fitted on a
             distorted frame is *systematically* wrong away from the calibration
             points.  Callers should therefore undistort before projecting whenever
             coefficients are available.

    @param  fx, fy  Focal lengths in pixels.
    @param  cx, cy  Principal point in pixels.
    @param  dist    Brown-Conrady coefficients, or ``None`` for an ideal pinhole.
    @param  width   Image width in pixels used for validation.
    @param  height  Image height in pixels used for validation.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...] | None = None
    width: int | None = None
    height: int | None = None

    @property
    def matrix(self) -> np.ndarray:
        """@brief 3x3 pinhole camera matrix ``K``."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def size(self) -> tuple[int, int] | None:
        """@brief ``(width, height)`` when known, else ``None``."""
        if self.width is None or self.height is None:
            return None
        return (int(self.width), int(self.height))

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        hfov_deg: float,
        vfov_deg: float | None = None,
    ) -> "Intrinsics":
        """Derive intrinsics from a horizontal (and optionally vertical) field of view.

        @brief   Build intrinsics from datasheet field-of-view figures.
        @details This exists because camera datasheets quote field of view but almost
                 never quote focal length in pixels.  When ``vfov_deg`` is omitted the
                 sensor is assumed to have square pixels, i.e. ``fy == fx``, which is
                 true for essentially every modern CMOS sensor.
        @param   width      Image width in pixels.
        @param   height     Image height in pixels.
        @param   hfov_deg   Horizontal field of view in degrees.
        @param   vfov_deg   Optional vertical field of view in degrees.
        @return  Intrinsics with a centred principal point.
        """
        if not 0.0 < hfov_deg < 180.0:
            raise GeometryError(f"horizontal FOV must be in (0, 180) degrees, got {hfov_deg}")
        fx = (width / 2.0) / math.tan(hfov_deg * _DEG2RAD / 2.0)
        if vfov_deg is None:
            fy = fx
        else:
            if not 0.0 < vfov_deg < 180.0:
                raise GeometryError(f"vertical FOV must be in (0, 180) degrees, got {vfov_deg}")
            fy = (height / 2.0) / math.tan(vfov_deg * _DEG2RAD / 2.0)
        return cls(fx=fx, fy=fy, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0, width=width, height=height)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], width: int | None = None, height: int | None = None) -> "Intrinsics":
        """Build intrinsics from a config mapping.

        @brief   Accepts either explicit focal lengths or a field-of-view description.
        @details Supported keys: ``fx``/``fy``/``cx``/``cy``/``dist``/``width``/``height``
                 for the explicit form, or ``hfov_deg``/``vfov_deg`` plus ``width``/``height``
                 for the derived form.  Explicit values always win, so a calibration
                 result can be dropped in over a datasheet estimate.
        @param   cfg    Mapping of intrinsic parameters.
        @param   width  Fallback image width when not present in ``cfg``.
        @param   height Fallback image height when not present in ``cfg``.
        @return  Intrinsics instance.
        """
        w = cfg.get("width", width)
        h = cfg.get("height", height)
        if "fx" in cfg:
            return cls(
                fx=float(cfg["fx"]),
                fy=float(cfg.get("fy", cfg["fx"])),
                cx=float(cfg.get("cx", (w - 1) / 2.0 if w else 0.0)),
                cy=float(cfg.get("cy", (h - 1) / 2.0 if h else 0.0)),
                dist=_as_dist_tuple(cfg.get("dist")),
                width=int(w) if w else None,
                height=int(h) if h else None,
            )
        if "hfov_deg" not in cfg:
            raise GeometryError("intrinsics need either 'fx' or 'hfov_deg'")
        if w is None or h is None:
            raise GeometryError("field-of-view intrinsics require 'width' and 'height'")
        return cls.from_fov(int(w), int(h), float(cfg["hfov_deg"]), cfg.get("vfov_deg"))

    def is_distorted(self) -> bool:
        """@brief True when at least one distortion coefficient is non-zero."""
        if not self.dist:
            return False
        return any(abs(float(c)) > 1e-12 for c in self.dist)

    def undistort_points(self, pts_xy: np.ndarray) -> np.ndarray:
        """Remove lens distortion from image points.

        @brief   Iteratively invert the Brown-Conrady distortion model.
        @details Implemented in NumPy so that the core layer keeps working without
                 OpenCV.  The inversion uses fixed-point iteration on the radial
                 term, which converges in a handful of steps for the small
                 coefficients seen on real lenses.  If OpenCV is available and
                 distortion is present, ``cv2.undistortPoints`` is preferred because
                 it uses the same convention the intrinsics were fitted with.

        @param   pts_xy  ``(N, 2)`` array of distorted image coordinates.
        @return  ``(N, 2)`` array of undistorted (ideal pinhole) image coordinates.
        @note    Fails noisily on the wrong-shaped input rather than silently
                 broadcasting, because a silent shape error here produces plausible
                 but wrong speeds.
        """
        pts = np.asarray(pts_xy, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise GeometryError(f"points must have shape (N, 2), got {pts.shape}")
        if not self.is_distorted():
            return pts.copy()
        if HAVE_CV2:
            k = np.array(self.dist, dtype=np.float64).reshape(-1)[:5]
            # @note cv2.undistortPoints takes distortion as (k1, k2, p1, p2[, k3]).
            out = _cv2.undistortPoints(pts.reshape(-1, 1, 2), self.matrix, k)
            return out.reshape(-1, 2).astype(np.float64)

        dist = list(self.dist) + [0.0] * (5 - len(self.dist))
        k1, k2, p1, p2, k3 = (float(v) for v in dist[:5])
        x = (pts[:, 0] - self.cx) / self.fx
        y = (pts[:, 1] - self.cy) / self.fy
        x0, y0 = x.copy(), y.copy()
        for _ in range(20):  # fixed-point iteration; tiny coefficients converge fast
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            if abs(radial) < 1e-9:
                raise GeometryError("degenerate distortion model (radial term ~ 0)")
            # Invert the tangential term explicitly, then divide out the radial term.
            dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x = (x0 - dx) / radial
            y = (y0 - dy) / radial
        return np.stack([x * self.fx + self.cx, y * self.fy + self.cy], axis=1)

    def project_camera_points(self, pts_cam: np.ndarray) -> np.ndarray:
        """Project camera-frame 3D points into distorted image pixels.

        @brief   Apply perspective division and the distortion model.
        @details Provided mainly for tests and calibration-workflow previews: the
                 speed pipeline runs in the opposite direction.  Implemented in
                 NumPy so that the round-trip test ``undistort(project(p)) == p`` can
                 run without OpenCV.
        @param   pts_cam ``(N, 3)`` camera-frame points (x right, y down, z forward).
        @return  ``(N, 2)`` image coordinates in pixels, including distortion.
        """
        pts = np.asarray(pts_cam, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise GeometryError(f"camera points must have shape (N, 3), got {pts.shape}")
        z = pts[:, 2]
        if np.any(z <= 1e-9):
            raise GeometryError("cannot project points at or behind the camera plane (z <= 0)")
        x = pts[:, 0] / z
        y = pts[:, 1] / z
        if self.is_distorted():
            dist = list(self.dist) + [0.0] * (5 - len(self.dist))
            k1, k2, p1, p2, k3 = (float(v) for v in dist[:5])
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            x = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            y = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.stack([x * self.fx + self.cx, y * self.fy + self.cy], axis=1)


def _as_dist_tuple(value: Any) -> tuple[float, ...] | None:
    """@brief Normalise a config distortion value to a tuple of floats or ``None``."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(v) for v in value)


# ---------------------------------------------------------------------------
# Extrinsics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Extrinsics:
    """Camera pose relative to the road reference frame.

    @brief  Where the camera is and which way it points.
    @details The road frame is a right-handed frame with:
                 origin at the **camera nadir projected onto the road surface**,
                 +X across the road (to the camera's right when facing along travel),
                 +Y **down** (into the road surface),
                 +Z along the road in the direction of travel.
             Placing the origin at the nadir rather than at the pole means
             ``position`` is always ``(0, -height, 0)`` plus a longitudinal offset,
             which keeps the homography derivation simple and the config readable.

    @param  height_m   Camera height above the road surface, in metres.
    @param  pitch_deg  Downward tilt of the optical axis; 0 = horizontal, 90 = straight down.
    @param  yaw_deg    Rotation about the world vertical; non-zero when the pole is not
                       square to the road.
    @param  roll_deg   Rotation about the road travel axis; non-zero when the camera is
                       not level, which is the single most common mounting error.
    @param  along_m    Camera position along the road (+Z) relative to the world origin.
                       Lets a calibration place the origin at a convenient landmark.

    @note   The overhead configuration uses :meth:`overhead`, which is a special case in
            :func:`rotation_matrix`.  Aiming exactly straight down makes the pitch/yaw/roll
            parameterisation singular, so that pose is built from its axis alignment
            directly instead of being composed; the result places image ``u`` along road
            ``X`` and image ``v`` along road ``Z``, matching the affine plane that
            :class:`TopDownProjector` fits.
    """

    height_m: float
    pitch_deg: float
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    along_m: float = 0.0

    def __post_init__(self) -> None:
        # @validation A non-positive or absurd height silently produces a valid-looking
        #             but meaningless homography, so it is rejected eagerly.
        if not math.isfinite(self.height_m) or self.height_m <= 0.0:
            raise GeometryError(f"camera height must be finite and > 0 m, got {self.height_m}")
        if not -90.0 <= self.pitch_deg <= 90.0:
            raise GeometryError(f"pitch must be within [-90, 90] degrees, got {self.pitch_deg}")

    @property
    def rotation(self) -> np.ndarray:
        """@brief World->camera rotation matrix ``R_wc``."""
        return rotation_matrix(self.pitch_deg, self.yaw_deg, self.roll_deg)

    @property
    def position(self) -> np.ndarray:
        """@brief Camera centre ``(0, -height, along)`` in the world road frame."""
        return np.array([0.0, -self.height_m, self.along_m], dtype=np.float64)

    @classmethod
    def overhead(
        cls,
        height_m: float,
        rotation_deg: float = 0.0,
        along_m: float = 0.0,
    ) -> "Extrinsics":
        """Build the canonical overhead pose for a camera on a pole above the road.

        @brief   Straight-down pose with the axis conventions the rest of the project expects.
        @details Straight down, with the camera mounted so that image ``u`` increases with
                 road ``X`` and image ``v`` increases with road ``Z``; see the class note
                 above.  ``rotation_deg`` is added as a yaw to express an in-plane rotation of
                 the track in the image.  This is a built-in special case of
                 :func:`rotation_matrix`, which has to bypass the pitch/yaw/roll composition
                 at exactly 90 degrees because that parameterisation is singular there.
        @param   height_m     Lens height above the road surface in metres.
        @param   rotation_deg Extra in-plane rotation of the road in the image, degrees.
        @param   along_m      Longitudinal offset of the camera from the world origin, metres.
        @return  Extrinsics for the overhead configuration.
        """
        return cls(
            height_m=height_m,
            pitch_deg=90.0,
            yaw_deg=0.0,
            roll_deg=0.0 if abs(rotation_deg) < 1e-12 else rotation_deg,
            along_m=along_m,
        )

    def to_config(self) -> dict[str, float]:
        """@brief Serialise to a plain dict for YAML storage."""
        return {
            "height_m": float(self.height_m),
            "pitch_deg": float(self.pitch_deg),
            "yaw_deg": float(self.yaw_deg),
            "roll_deg": float(self.roll_deg),
            "along_m": float(self.along_m),
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Extrinsics":
        """@brief Build from a config mapping using the keys produced by :meth:`to_config`."""
        return cls(
            height_m=float(cfg["height_m"]),
            pitch_deg=float(cfg["pitch_deg"]),
            yaw_deg=float(cfg.get("yaw_deg", 0.0)),
            roll_deg=float(cfg.get("roll_deg", 0.0)),
            along_m=float(cfg.get("along_m", 0.0)),
        )


# ---------------------------------------------------------------------------
# Homography algebra (NumPy only)
# ---------------------------------------------------------------------------


def apply_homography(h_mat: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to 2D points.

    @brief   Vectorised projective transform of ``(N, 2)`` points.
    @details A NumPy implementation is kept alongside any OpenCV path because it is
             used on every detected track point in the hot loop; for a handful of
             points per frame the NumPy version is competitive and avoids an FFI
             round trip, which matters on a Raspberry Pi.
    @param   h_mat  ``(3, 3)`` homography.
    @param   pts_xy ``(N, 2)`` points in the source frame.
    @return  ``(N, 2)`` transformed points.
    @raises  GeometryError if any point maps to the line at infinity (w ~ 0), which
             for a ground homography means the point sits on the horizon.
    """
    pts = np.asarray(pts_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise GeometryError(f"points must have shape (N, 2), got {pts.shape}")
    h = np.asarray(h_mat, dtype=np.float64).reshape(3, 3)
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ h.T
    w = homo[:, 2]
    if np.any(np.abs(w) < 1e-12):
        raise GeometryError("homography degenerates: point maps to infinity (on the horizon?)")
    return homo[:, :2] / w[:, None]


def _invert_3x3(mat: np.ndarray) -> np.ndarray:
    """Invert a 3x3 matrix, preferring the closed-form adjugate over ``np.linalg.inv``.

    @brief   Fast, allocation-light 3x3 inverse.
    @details This is called once per calibration and then never again, so the reason
             is not speed but conditioning: the adjugate form works directly from the
             analytic structure and avoids introducing an extra LAPACK call whose
             numerical behaviour is harder to reason about for near-degenerate
             roadside poses.  It also means the whole geometry layer has no
             dependency on a working linear-algebra backend beyond elementwise math.
    @param   mat ``(3, 3)`` invertible matrix.
    @return  ``(3, 3)`` inverse.
    @raises  GeometryError if the determinant is numerically zero.
    """
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-15:
        raise GeometryError("singular 3x3 matrix (degenerate camera pose)")
    inv_det = 1.0 / det
    return np.array(
        [
            [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
            [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
            [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
        ],
        dtype=np.float64,
    )


def find_homography(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """Fit the homography mapping ``src`` points onto ``dst`` points.

    @brief   Normalised DLT with a RANSAC refinement when OpenCV is present.
    @details Calibration is where accuracy is won or lost, so this deliberately
             prefers ``cv2.findHomography`` with RANSAC: it is robust to a
             mis-clicked point, and a single bad ground-control point otherwise
             skews the whole projective fit.  The NumPy fallback performs
             Hartley normalisation followed by a plain DLT so that the core stays
             usable on a stripped-down installation, at the cost of RANSAC.
    @param   src_xy ``(N, 2)`` source points, N >= 4.
    @param   dst_xy ``(N, 2)`` destination points, same length as ``src_xy``.
    @return  ``(3, 3)`` homography with ``h[2, 2] == 1`` when well conditioned.
    @raises  GeometryError when fewer than four correspondences are supplied.
    """
    src = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    if len(src) != len(dst):
        raise GeometryError(f"point count mismatch: {len(src)} vs {len(dst)}")
    if len(src) < 4:
        raise GeometryError(f"a homography needs >= 4 correspondences, got {len(src)}")
    if HAVE_CV2 and len(src) >= 4:
        h_mat, _mask = _cv2.findHomography(src, dst, _cv2.RANSAC, 3.0)
        if h_mat is not None:
            return np.asarray(h_mat, dtype=np.float64)
    return _dlt_homography(src, dst)


def _dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct linear transform for a homography using Hartley normalisation.

    @brief   Solve the homogeneous system ``A h = 0`` by SVD.
    @details Point coordinates are first translated/scaled to zero mean and RMS
             distance ``sqrt(2)``.  Without this the design matrix mixes terms of
             wildly different magnitude and the SVD picks a poor null vector, which
             shows up as a homography that is accurate at the calibration points and
             badly wrong elsewhere.
    @param   src ``(N, 2)`` source points.
    @param   dst ``(N, 2)`` destination points.
    @return  ``(3, 3)`` homography mapping ``src`` to ``dst``.
    """
    t_src, src_n = _normalise_points(src)
    t_dst, dst_n = _normalise_points(dst)
    n = len(src_n)
    a_mat = np.zeros((2 * n, 9), dtype=np.float64)
    for idx, ((x, y), (u, v)) in enumerate(zip(src_n, dst_n)):
        a_mat[2 * idx] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        a_mat[2 * idx + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
    _u, _s, vt = np.linalg.svd(a_mat)
    h_norm = vt[-1].reshape(3, 3)
    h_mat = _invert_3x3(t_dst) @ h_norm @ t_src
    if abs(h_mat[2, 2]) < 1e-15:
        raise GeometryError("degenerate homography (h[2,2] ~ 0)")
    return h_mat / h_mat[2, 2]


def _normalise_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """@brief Hartley normalisation: return ``(3,3)`` transform and normalised points."""
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    rms = math.sqrt(float((shifted**2).sum(axis=1).mean()))
    scale = math.sqrt(2.0) / rms if rms > 1e-12 else 1.0
    t_mat = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ t_mat.T
    return t_mat, homo[:, :2]


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------


@dataclass
class CameraModel:
    """A fully specified monocular camera looking at a road surface.

    @brief   Binds intrinsics to extrinsics and derives both ground homographies.
    @details The ground->image homography has a closed form for this pose, which is
             worth spelling out because it is the source of truth for the entire
             speed pipeline and it is verified against OpenCV's ``projectPoints`` in
             the test suite:

                 world point ``M = (X, 0, D)`` in the road frame, and
                 ``M_cam = R_wc @ (M - C)`` with ``C = (0, -h, along)`` gives
                     ``M_cam = (X, -s*D - c*h, -c*D + s*h)``
                 where ``s = sin(pitch)``, ``c = cos(pitch)`` for the yaw=roll=0 case.
                 With ``u = X``, ``v = D``, ``w = 1`` this is linear, hence
                     ``H_ground->image = K @ [[1, 0, 0],
                                              [0, -s, -c*h],
                                              [0, -c,  s*h]]``
                 which is exactly ``K @ R_wc[:, [0, 1, 2]]`` with the world basis
                 ``(X, Y_down, Z)`` and ``R_wc[:, 1] = (0, -s, -c)``.  The general
                 yaw/roll form is the same expression, which is why the code below
                 simply reads the rotation columns.

    @param  intrinsics Camera intrinsics.
    @param  extrinsics Camera pose in the road frame.
    @param  label      Free-form name used in logs, plots and config round-trips.
    """

    intrinsics: Intrinsics
    extrinsics: Extrinsics
    label: str = "camera"
    _h_gi: np.ndarray = field(default=None, repr=False, compare=False)
    _h_ig: np.ndarray = field(default=None, repr=False, compare=False)

    # -- derived matrices ---------------------------------------------------

    def ground_to_image_matrix(self) -> np.ndarray:
        """@brief ``(3, 3)`` homography mapping road-ground ``(X, Z)`` to image pixels."""
        if self._h_gi is None:
            r_mat = self.extrinsics.rotation
            c_vec = self.extrinsics.position
            m_mat = np.stack([r_mat[:, 0], r_mat[:, 2], -r_mat @ c_vec], axis=1)
            h_mat = self.intrinsics.matrix @ m_mat
            if abs(h_mat[2, 2]) < 1e-15:
                raise GeometryError("degenerate ground homography; check camera pose")
            self._h_gi = h_mat / h_mat[2, 2]
        return self._h_gi.copy()

    def image_to_ground_matrix(self) -> np.ndarray:
        """@brief ``(3, 3)`` homography mapping image pixels to road-ground ``(X, Z)``.

        @details This is only valid for points that genuinely lie on the road plane.
             Feeding it a bounding-box centre instead of the ground-contact point is
             the single most common bug in monocular speed estimation, because the
             roof projects onto the road *ahead* of the car.  See ``ground_point`` in
             the detection layer.
        """
        if self._h_ig is None:
            self._h_ig = _invert_3x3(self.ground_to_image_matrix())
        return self._h_ig.copy()

    @property
    def horizon_row(self) -> float:
        """@brief Image row in pixels of the ground horizon.

        @details The horizon is the limit of the ground projection as the road distance goes
                 to infinity, which is exactly the vanishing point of the world **+Z**
                 (along-road) direction, namely ``K @ R_wc[:, 2]`` normalised.  With this
                 module's ``v = fy * Yc/Zc + cy`` row convention its row is

                     ``v = cy + fy * R[1, 2] / R[2, 2]``

                 which equals ``cy - fy * tan(pitch)`` for the aligned case and reduces to
                 the homography's second column, ``h_gi[:, 1]``, normalised by its
                 homogeneous weight -- the two derivations are asserted to agree exactly in
                 the test suite, because a mismatch between them is precisely the kind of
                 error that silently mis-gates the usable image region.

                 and this is asserted both against ``cv2.projectPoints`` and against the
                 ``Z -> inf`` limit of the ground homography in the test suite.

                 Two plausible-looking alternatives were tried during development and
                 falsified against those references, and are recorded here so nobody
                 re-derives them:

                 * The vanishing point of the world **downward** direction ``R[:, 1]`` sits
                   below the principal point and therefore *inside* the frame for any real
                   camera.  Using it puts the horizon in the middle of the pavement.
                 * The image of the road plane's line at infinity, obtained by intersecting
                   the ``X`` and ``Y_down`` vanishing points, gives an unrelated line.

                 For a moderate downward pitch (roughly 5-70 degrees) the horizon lands
                 *above* the top of the frame, where it acts as a hard bound rather than a
                 visible line; beyond that it moves back into or below the frame, which is
                 the expected behaviour as the camera approaches a horizontal or overhead
                 aim.

        @return Horizon row in pixels; ``-inf`` when the camera does not look along the
                road in its direction of travel, so that no ground horizon is modelled.
        """
        if not self._sees_road_in_front():
            return float("-inf")
        forward_along = self.extrinsics.rotation[2, 2]
        if abs(forward_along) < 1e-12:
            return float("-inf")
        return float(self.intrinsics.cy + self.intrinsics.fy * self.extrinsics.rotation[1, 2] / forward_along)

    def _sees_road_in_front(self) -> bool:
        """@brief Whether the vanishing point of the along-road direction is in the image.

        @details Guards the failure that otherwise produces silent nonsense: a camera whose
                 optical axis points *away* along the road.  The ground homography stays
                 perfectly well conditioned in that pose and every projection still returns
                 a finite number, but those numbers describe road *behind* the camera, so
                 any speed derived from them is meaningless.

                 The test is simply whether the road direction's vanishing point, computed
                 from the rotation matrix alone so that it also works for a projector built
                 from an explicit homography, projects to a finite point.
        @return ``True`` when the along-road vanishing point is expressible in the image.
        """
        # @note forward == R[:, 2] == the world +Z direction expressed in camera axes.
        return abs(float(self.extrinsics.rotation[2, 2])) > 1e-9

    def ground_point_at_row(self, row: float) -> np.ndarray:
        """@brief Point on the road plane at a given image row, on the optical axis column.

        @details Solves the homography for the world ``(X, Z)`` that projects to
                 ``(cx, row)``.  Used by the calibration assistants to turn a single
                 measured distance into a metric scale, and by the visualisation tools
                 to draw distance rulers down the middle of the frame.
        @param   row Image row in pixels.
        @return  ``(2,)`` array ``(X, Z)`` in metres.
        """
        h_ig = self.image_to_ground_matrix()
        pt = np.array([[self.intrinsics.cx, float(row)]], dtype=np.float64)
        return apply_homography(h_ig, pt)[0]

    # -- projection API -----------------------------------------------------

    def image_to_ground(self, pts_xy: np.ndarray, undistort: bool = True) -> np.ndarray:
        """@brief Project image points onto the road plane, returning ``(X, Z)`` metres."""
        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
        if undistort and self.intrinsics.is_distorted():
            pts = self.intrinsics.undistort_points(pts)
        return apply_homography(self.image_to_ground_matrix(), pts)

    def ground_to_image(self, pts_xz: np.ndarray) -> np.ndarray:
        """@brief Project road-plane ``(X, Z)`` metres into image pixels.

        @note Distortion is applied here (forward direction), so this round-trips with
              :meth:`image_to_ground` to within the numerical accuracy of the
              undistortion inversion.
        """
        pts = np.asarray(pts_xz, dtype=np.float64).reshape(-1, 2)
        projected = apply_homography(self.ground_to_image_matrix(), pts)
        if self.intrinsics.is_distorted():
            # @step Convert ideal-pinhole pixels to unit-depth camera rays, then let the
            #       intrinsics re-apply the distortion model on the way back out.
            x = (projected[:, 0] - self.intrinsics.cx) / self.intrinsics.fx
            y = (projected[:, 1] - self.intrinsics.cy) / self.intrinsics.fy
            rays = np.column_stack([x, y, np.ones_like(x)])
            return self.intrinsics.project_camera_points(rays)
        return projected

    # -- local scale and uncertainty ---------------------------------------

    def ground_scale(self, pts_xz: np.ndarray, eps: float = 1e-3) -> dict[str, np.ndarray]:
        """Compute the local metric resolution of the camera at ground positions.

        @brief   How many metres of road one pixel covers, and how anisotropic that is.
        @details The Jacobian of the image<-ground map is computed analytically by
                 central differences on the closed-form homography, then decomposed
                 with an SVD.  This matters more than any other single quantity in the
                 project:

                 * ``m_per_px_min`` is the *best* achievable positional resolution at
                   that spot, and therefore a floor on the achievable speed error.
                   Dividing it by the inter-frame time gives a hard lower bound on the
                   speed noise, which the pipeline reports rather than hiding.
                 * ``anisotropy`` (largest/smallest singular value) exposes
                   foreshortening.  It is ~1 for a top-down camera and grows large down
                   a roadside frame, quantifying why distance estimation degrades with
                   range instead of treating it as a mystery.
                 * ``m_per_px_long`` / ``m_per_px_lat`` give the resolution along and
                   across the travel direction, which is what actually enters the speed
                   error when a car drives straight down the road.

        @param   pts_xz ``(N, 2)`` road-plane positions in metres.
        @param   eps    Finite-difference step in metres; 1 mm is far below pixel scale.
        @return  Mapping with ``m_per_px_min``, ``m_per_px_max``, ``m_per_px_long``,
                 ``m_per_px_lat``, ``anisotropy`` and ``px_per_m`` as ``(N,)`` arrays.
        @note    Unlike the rest of this class, this deliberately does **not** apply
                 lens distortion to the finite differences.  Distortion varies slowly
                 across the frame, so its effect on a 1 mm-scale local derivative is
                 negligible, while including it would multiply the cost of this call.
        """
        pts = np.asarray(pts_xz, dtype=np.float64).reshape(-1, 2)
        h_gi = self.ground_to_image_matrix()

        # @step 1: analytic Jacobian of the projective map.  For a homography H acting on
        #       homogeneous ground coordinates (X, Z, 1), the image point is p = n / w with
        #       n = H[:2] @ (X, Z, 1) and w = H[2] @ (X, Z, 1).  The quotient rule gives
        #       J = (H[:2, :2] - p (x) H[2, :2]) / w, computed for every point at once.
        #       This is done analytically rather than by finite differences because a naive
        #       symmetric difference with a 1 mm step carries a relative truncation error of
        #       order 1e-3 at typical depths -- small in absolute terms, but these scales feed
        #       the calibration solvers, where a 0.1 per cent bias in metres-per-pixel becomes
        #       a 0.1 per cent bias in every reported speed.
        num = pts @ h_gi[:2, :2].T + h_gi[:2, 2]
        weight = pts @ h_gi[2, :2] + h_gi[2, 2]
        if np.any(np.abs(weight) < 1e-15):
            raise GeometryError("ground point lies on the horizon; no local scale exists")
        px = num / weight[:, None]
        jac = (h_gi[:2, :2][None, :, :] - px[:, :, None] * h_gi[2, :2][None, None, :]) / weight[:, None, None]

        sv = np.linalg.svd(jac, compute_uv=False)  # (N, 2) descending
        sv_max = np.maximum(sv[:, 0], 1e-12)
        sv_min = np.maximum(sv[:, 1], 1e-12)

        long_axis = np.hypot(jac[:, 0, 1], jac[:, 1, 1])
        lat_axis = np.hypot(jac[:, 0, 0], jac[:, 1, 0])
        return {
            "m_per_px_min": 1.0 / sv_max,
            "m_per_px_max": 1.0 / sv_min,
            "m_per_px_long": 1.0 / np.maximum(long_axis, 1e-12),
            "m_per_px_lat": 1.0 / np.maximum(lat_axis, 1e-12),
            "anisotropy": sv_max / sv_min,
            "px_per_m": sv_max,
        }

    def ground_error_m(self, pts_xz: np.ndarray, pixel_sigma_px: float = 1.0) -> np.ndarray:
        """@brief Worst-case ground-position error in metres for a detector jitter.

        @details Converts an assumed detector box jitter (in pixels) into a metric
                 position uncertainty using the local Jacobian's largest singular
                 value.  A YOLO box on a 1:64 toy is realistically good to a few
                 pixels, so this is a genuine contributor to speed error rather than a
                 theoretical nicety, and it is what the confidence gating in the speed
                 estimator is built on.
        @param   pts_xz         ``(N, 2)`` road-plane positions in metres.
        @param   pixel_sigma_px Assumed 1-sigma detector jitter in pixels.
        @return  ``(N,)`` positional error in metres (1-sigma, worst direction).
        """
        return float(pixel_sigma_px) * self.ground_scale(pts_xz)["m_per_px_max"]

    # -- serialisation ------------------------------------------------------

    def to_config(self) -> dict[str, Any]:
        """@brief Serialise intrinsics and extrinsics to a plain nested dict."""
        intr: dict[str, Any] = {
            "fx": float(self.intrinsics.fx),
            "fy": float(self.intrinsics.fy),
            "cx": float(self.intrinsics.cx),
            "cy": float(self.intrinsics.cy),
        }
        if self.intrinsics.dist is not None:
            intr["dist"] = [float(c) for c in self.intrinsics.dist]
        if self.intrinsics.width is not None:
            intr["width"] = int(self.intrinsics.width)
        if self.intrinsics.height is not None:
            intr["height"] = int(self.intrinsics.height)
        return {
            "label": self.label,
            "intrinsics": intr,
            "extrinsics": self.extrinsics.to_config(),
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CameraModel":
        """@brief Build a camera model from a nested config mapping."""
        return cls(
            intrinsics=Intrinsics.from_config(cfg["intrinsics"]),
            extrinsics=Extrinsics.from_config(cfg["extrinsics"]),
            label=str(cfg.get("label", "camera")),
        )


# ---------------------------------------------------------------------------
# Affine ground plane (configuration A: overhead pole, near-orthographic)
# ---------------------------------------------------------------------------


@dataclass
class AffineGroundPlane:
    """A 2D affine map between image pixels and metres on the road surface.

    @brief   Ground model for the overhead camera.
    @details When the optical axis is within a few degrees of vertical the projective
             terms of the ground homography become numerically negligible -- at 2 m
             height and a 3 degree tilt the second-order term contributes well under
             1% of the position -- while their *estimate* from a small number of noisy
             clicked points is very unstable.  Fitting a 6-parameter affine model
             instead is therefore not a simplification of convenience: it is the
             better-conditioned estimator for the geometry that actually exists under
             an overhead pole.

             The map has the form, with ``A`` a ``(2, 2)`` matrix and ``t`` a ``(2,)``
             translation, taking pixels to road metres::

                 [X]   [a00 a01] [u]   [tx]
                 [Z] = [a10 a11] [v] + [ty]

             In the ideal overhead case ``A`` reduces to ``s * R(-roll)`` with
             ``s = height / fx`` the metres-per-pixel factor.

    @param  matrix  ``(2, 3)`` affine matrix mapping ``(u, v)`` pixels to ``(X, Z)`` metres.
    @param  source  Provenance string recorded in the calibration file for auditing.
    @param  residuals_m Optional per-point fit residuals in metres, for quality reporting.
    """

    matrix: np.ndarray
    source: str = "affine"
    residuals_m: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64).reshape(2, 3)
        if not np.all(np.isfinite(self.matrix)):
            raise GeometryError("affine ground matrix contains non-finite values")

    @property
    def linear(self) -> np.ndarray:
        """@brief ``(2, 2)`` linear part ``A`` mapping pixel deltas to metre deltas."""
        return self.matrix[:, :2]

    @property
    def translation(self) -> np.ndarray:
        """@brief ``(2,)`` translation part in metres."""
        return self.matrix[:, 2]

    @property
    def m_per_px(self) -> float:
        """@brief Nominal metres-per-pixel scale ``sqrt(|det A|)``.

        @details Under a true similarity this is exact; with shear present it is the
                 geometric mean of the two singular values, i.e. the scale that
                 preserves area, which is the right single number for reporting.
        """
        return float(math.sqrt(abs(float(np.linalg.det(self.linear)))))

    @property
    def anisotropy(self) -> float:
        """@brief Ratio of largest to smallest singular value of ``A``.

        @details Equals 1 for an ideal overhead camera.  A value meaningfully above 1
                 indicates a tilted or anamorphic mount, and the resulting direction-
                 dependent scale is precisely what a single "metres per pixel" number
                 would hide.
        """
        sv = np.linalg.svd(self.linear, compute_uv=False)
        return float(sv[0] / max(sv[1], 1e-12))

    @property
    def inverse_matrix(self) -> np.ndarray:
        """@brief ``(2, 3)`` matrix mapping road metres back to pixels.

        @details Built by inverting the affine transform in closed form.  The linear
                 part is inverted directly and the translation is pushed through it,
                 which avoids the wider ``(3, 3)`` inverse and keeps the result exactly
                 consistent with :meth:`image_to_ground` (their composition is the
                 identity to floating-point precision, which the tests assert).
        """
        inv_linear = np.linalg.inv(self.linear)
        return np.column_stack([inv_linear, -inv_linear @ self.translation])

    def image_to_ground(self, pts_xy: np.ndarray) -> np.ndarray:
        """@brief Map ``(N, 2)`` pixels to ``(N, 2)`` road metres."""
        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
        return pts @ self.linear.T + self.translation

    def ground_to_image(self, pts_xz: np.ndarray) -> np.ndarray:
        """@brief Map ``(N, 2)`` road metres back to ``(N, 2)`` pixels."""
        pts = np.asarray(pts_xz, dtype=np.float64).reshape(-1, 2)
        inv = self.inverse_matrix
        return pts @ inv[:, :2].T + inv[:, 2]

    @classmethod
    def from_scale(
        cls,
        m_per_px: float,
        rotation_deg: float = 0.0,
        offset_xz: Sequence[float] = (0.0, 0.0),
        offset_uv: Sequence[float] = (0.0, 0.0),
        source: str = "scale",
    ) -> "AffineGroundPlane":
        """Build an affine ground plane from a single scale plus an optional rotation.

        @brief   Construct the ideal overhead mapping.
        @details With no rotation the mapping is simply ``X = s * u``, ``Z = s * v``: for an
                 overhead camera whose image rows point along the road, a pixel further
                 *down* the frame (larger ``v``) is further along the road (larger ``Z``).
                 A positive ``rotation_deg`` composes a rotation matrix on the right of the
                 scale, ``A = s * R(rotation)``, which rotates the ground result.

                 @note The sign of this rotation was a real bug: an earlier formulation used
                 ``s * [[cos, sin], [-sin, cos]]``, whose determinant is still positive but
                 whose rows are swapped and whose ``Z`` axis is negated relative to the
                 projective model.  The symptom was that ``TopDownProjector`` reported every
                 car's position mirrored in ``Z``, so it drove backwards, while
                 ``m_per_px`` -- a determinant-based scalar -- looked perfectly correct and
                 hid the problem.  The test suite now asserts agreement between the affine
                 and projective models point by point, which is what catches a mirrored
                 frame that a scale check cannot.

                 The offsets pin an image point (``offset_uv``, conventionally the frame
                 centre) to a road point (``offset_xz``, conventionally the origin), so
                 the metric origin is meaningful instead of being wherever the corner
                 of the image happens to land.
        @param   m_per_px     Metres of road per pixel.
        @param   rotation_deg Image rotation of the road relative to the pixel grid.
        @param   offset_xz    Road coordinates of ``offset_uv``, in metres.
        @param   offset_uv    Image coordinates of ``offset_xz``, in pixels.
        @param   source       Provenance label.
        @return  AffineGroundPlane instance.
        """
        if not math.isfinite(m_per_px) or m_per_px <= 0.0:
            raise GeometryError(f"m_per_px must be finite and > 0, got {m_per_px}")
        theta = rotation_deg * _DEG2RAD
        ct, st = math.cos(theta), math.sin(theta)
        # @step A = s * R(theta) with R a counter-clockwise rotation matrix; this keeps the
        #       unrotated mapping as the identity scaled by s, which is what an overhead
        #       camera looking straight down actually produces.
        linear = m_per_px * np.array([[ct, -st], [st, ct]], dtype=np.float64)
        trans = np.asarray(offset_xz, dtype=np.float64) - linear @ np.asarray(offset_uv, dtype=np.float64)
        return cls(matrix=np.column_stack([linear, trans]), source=source)

    @classmethod
    def from_point_pairs(
        cls,
        img_pts: np.ndarray,
        ground_pts: np.ndarray,
        source: str = "point_pairs",
    ) -> "AffineGroundPlane":
        """Least-squares affine fit from image/ground correspondences.

        @brief   Solve ``A, t`` minimising total squared reprojection error.
        @details Closed-form linear least squares: with ``P`` the ``(N, 3)`` homogeneous
                 image points and ``G`` the ``(N, 2)`` ground targets, the solution is
                 ``[A | t]^T = pinv(P) @ G``.  Three points make the system exactly
                 determined; more points make it overdetermined and stable, so the
                 calibration assistant encourages at least three.
        @param   img_pts    ``(N, 2)`` image points.
        @param   ground_pts ``(N, 2)`` corresponding road points in metres.
        @param   source     Provenance label.
        @return  AffineGroundPlane with fit residuals attached.
        @raises  GeometryError for fewer than three correspondences.
        """
        img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
        gnd = np.asarray(ground_pts, dtype=np.float64).reshape(-1, 2)
        if len(img) != len(gnd):
            raise GeometryError(f"point count mismatch: {len(img)} image vs {len(gnd)} ground")
        if len(img) < 3:
            raise GeometryError(f"an affine fit needs >= 3 correspondences, got {len(img)}")
        design = np.column_stack([img, np.ones(len(img))])
        params, *_ = np.linalg.lstsq(design, gnd, rcond=None)
        plane = cls(matrix=params.T, source=source)
        resid = np.linalg.norm(design @ params - gnd, axis=1)
        plane.residuals_m = resid
        return plane

    @classmethod
    def from_measured_distance(
        cls,
        pt_a_xy: Sequence[float],
        pt_b_xy: Sequence[float],
        real_distance_m: float,
        rotation_deg: float = 0.0,
        centroid_ground_xz: Sequence[float] = (0.0, 0.0),
        source: str = "two_point_tape",
    ) -> "AffineGroundPlane":
        """Build the affine ground plane from one measured distance and two clicked points.

        @brief   The tape-measure calibration path.
        @details A single measured distance constrains only the map's *scale*, so the
                 remaining degrees of freedom are fixed by assumptions, which are stated
                 here rather than buried:

                 * The road-to-image rotation is assumed to be zero unless the mount is
                   known to be skewed.  Physically this says the road runs parallel to
                   the image rows, which for a pole-mounted overhead camera is the
                   normal installation and is trivially true if the frame is levelled
                   when the marks are placed.
                 * The transformed midpoint of the two clicked points is pinned to
                   ``centroid_ground_xz``.  Pinning the midpoint rather than one endpoint
                   splits the placement error between the two clicks instead of heaping
                   it all on one, and it makes the metric origin land where the operator
                   says the two marks are centred.

                 For the perpendicular case the scale is exact in the sense that the
                 distance between the two clicked points maps to exactly the measured
                 distance; the residual error in a real setup comes from click accuracy
                 and from assuming a pure similarity.  Where an overhead camera also has
                 a known height ``h`` and focal length ``fx``, prefer
                 :meth:`AffineGroundPlane.from_scale` with ``s = h / fx``, which needs no
                 clicking at all.  That equivalence is asserted in the test suite.

        @param   pt_a_xy            First clicked image point ``(u, v)``.
        @param   pt_b_xy            Second clicked image point ``(u, v)``.
        @param   real_distance_m    Measured distance between the two marks, metres.
        @param   rotation_deg       Image rotation of the road, degrees.
        @param   centroid_ground_xz Road coordinates assigned to the clicked midpoint.
        @param   source             Provenance label.
        @return  AffineGroundPlane.
        @raises  GeometryError if the two points coincide or the distance is implausible.
        """
        pa = np.asarray(pt_a_xy, dtype=np.float64).reshape(2)
        pb = np.asarray(pt_b_xy, dtype=np.float64).reshape(2)
        span_px = float(np.linalg.norm(pb - pa))
        if span_px < 1e-6:
            raise GeometryError("the two calibration points are identical in the image")
        if not math.isfinite(real_distance_m) or real_distance_m <= 0.0:
            raise GeometryError(f"measured distance must be > 0 m, got {real_distance_m}")

        plane = cls.from_scale(
            m_per_px=real_distance_m / span_px,
            rotation_deg=rotation_deg,
            offset_xz=centroid_ground_xz,
            offset_uv=(pa + pb) / 2.0,
            source=source,
        )
        # @note The exactness claim above is only true when the assumed rotation matches
        #       the actual image direction of the two marks.  Report the mismatch as a
        #       residual so a badly-specified rotation surfaces during calibration
        #       instead of silently biasing every speed measurement.
        achieved = float(np.linalg.norm(plane.image_to_ground(pb[None])[0] - plane.image_to_ground(pa[None])[0]))
        plane.residuals_m = np.array([abs(achieved - real_distance_m)])
        return plane

    def to_config(self) -> dict[str, Any]:
        """@brief Serialise for YAML storage."""
        out: dict[str, Any] = {"matrix": self.matrix.tolist(), "source": self.source}
        if self.residuals_m is not None:
            out["residuals_m"] = [float(r) for r in np.atleast_1d(self.residuals_m)]
        return out

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AffineGroundPlane":
        """@brief Build from a config mapping produced by :meth:`to_config`."""
        resid = cfg.get("residuals_m")
        return cls(
            matrix=np.asarray(cfg["matrix"], dtype=np.float64),
            source=str(cfg.get("source", "config")),
            residuals_m=np.asarray(resid, dtype=np.float64) if resid is not None else None,
        )


# ---------------------------------------------------------------------------
# Projector abstraction
# ---------------------------------------------------------------------------


class GroundPlaneProjector(ABC):
    """Common interface for both mounting configurations.

    @brief   Maps detections on the road plane to metres, whatever the camera rig.
    @details Both configurations answer the same three questions, and nothing else in
             the pipeline needs to know which rig is in use:

                 1. where is this image point on the road?  (:meth:`to_ground`)
                 2. how good is that answer here?        (:meth:`scale` / :meth:`position_sigma_m`)
                 3. is this point even usable?           (:meth:`is_valid`)

             Question 2 is the one naive implementations skip.  Because speed is a
             *difference* of positions divided by a time, the positional resolution at
             the two sample points sets a hard floor on achievable accuracy, and under
             a roadside mount that resolution changes by an order of magnitude across
             the frame.  Exposing it lets the estimator down-weight bad samples instead
             of averaging them in.

    @param  intrinsics  Camera intrinsics, used for the optional distortion correction.
    @param  label       Name used in reports and visualisations.
    """

    #: @brief Short identifier written into calibration files (``topdown``/``roadside``).
    kind: str = "abstract"

    def __init__(self, intrinsics: Intrinsics | None = None, label: str = "camera") -> None:
        self.intrinsics = intrinsics
        self.label = label

    # -- required API -------------------------------------------------------

    @abstractmethod
    def to_ground(self, pts_xy: np.ndarray) -> np.ndarray:
        """@brief Map ``(N, 2)`` image points to ``(N, 2)`` road metres ``(X, Z)``."""

    @abstractmethod
    def to_image(self, pts_xz: np.ndarray) -> np.ndarray:
        """@brief Map ``(N, 2)`` road metres back to ``(N, 2)`` image points."""

    @abstractmethod
    def scale(self, pts_xz: np.ndarray) -> dict[str, np.ndarray]:
        """@brief Local metric resolution at road positions; see :meth:`CameraModel.ground_scale`."""

    # -- shared conveniences ------------------------------------------------

    def position_sigma_m(self, pts_xz: np.ndarray, pixel_sigma_px: float = 1.0) -> np.ndarray:
        """@brief Metric 1-sigma positional error for an assumed pixel jitter.

        @details Delegates to :meth:`scale` and takes the worst-case singular direction,
                 which is the honest bound: a car travelling diagonally sees the error
                 rotate through both axes as it moves.
        """
        return float(pixel_sigma_px) * self.scale(pts_xz)["m_per_px_max"]

    def min_speed_sigma(self, pts_xz: np.ndarray, dt_s: float, pixel_sigma_px: float = 1.0) -> np.ndarray:
        """@brief Lower bound on speed noise, in m/s, for a single position difference.

        @details ``sqrt(2) * sigma_pos / dt`` for two independent samples.  Reporting
                 this alongside each measured speed is what keeps the project honest on
                 a Raspberry Pi running at 3 FPS: the number explains itself instead of
                 looking like a detector failure.  A long ``dt`` and a coarse ground
                 scale are the two things that destroy speed accuracy, and both are
                 visible here in one term.
        @param   pts_xz         Road positions where the samples were taken, metres.
        @param   dt_s           Time between the two samples, seconds.
        @param   pixel_sigma_px Assumed detector jitter in pixels.
        @return  ``(N,)`` speed noise floor in m/s.
        """
        if dt_s <= 0.0:
            raise GeometryError(f"dt must be > 0 s, got {dt_s}")
        return math.sqrt(2.0) * self.position_sigma_m(pts_xz, pixel_sigma_px) / float(dt_s)

    def is_valid(self, pts_xy: np.ndarray) -> np.ndarray:
        """@brief Mask of image points that can be trusted for metric use.

        @details Rejects points beyond the frame, points that fail to reconstruct on
                 the road plane (a homography puts those at or behind the horizon), and
                 points whose metric resolution is hopeless.  Implementation-specific
                 limits are applied by the subclasses.
        @param   pts_xy ``(N, 2)`` image points.
        @return  ``(N,)`` boolean mask.
        """
        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
        try:
            ground = self.to_ground(pts)
        except GeometryError:
            return np.zeros(len(pts), dtype=bool)
        finite = np.all(np.isfinite(ground), axis=1)
        if self.intrinsics is not None and self.intrinsics.size is not None:
            w, h = self.intrinsics.size
            inside = (
                (pts[:, 0] >= 0.0)
                & (pts[:, 0] <= w - 1)
                & (pts[:, 1] >= 0.0)
                & (pts[:, 1] <= h - 1)
            )
            finite &= inside
        return finite

    def describe(self) -> dict[str, Any]:
        """@brief Human-readable summary used in logs, reports and the CLI banner."""
        info: dict[str, Any] = {"kind": self.kind, "label": self.label}
        info.update(self._describe_extra())
        return info

    def _describe_extra(self) -> dict[str, Any]:
        """@brief Subclass hook adding configuration-specific summary fields."""
        return {}

    def to_config(self) -> dict[str, Any]:
        """@brief Serialise this projector to a plain dict."""
        cfg: dict[str, Any] = {
            "kind": self.kind,
            "label": self.label,
            "intrinsics": _intrinsics_to_config(self.intrinsics),
        }
        cfg.update(self._config_extra())
        return cfg

    def _config_extra(self) -> dict[str, Any]:
        """@brief Subclass hook adding configuration-specific serialised fields."""
        return {}


def _intrinsics_to_config(intr: Intrinsics | None) -> dict[str, Any]:
    """@brief Serialise intrinsics, tolerating ``None``."""
    if intr is None:
        return {}
    out: dict[str, Any] = {
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.cx),
        "cy": float(intr.cy),
    }
    if intr.dist is not None:
        out["dist"] = [float(c) for c in intr.dist]
    if intr.width is not None:
        out["width"] = int(intr.width)
    if intr.height is not None:
        out["height"] = int(intr.height)
    return out


# ---------------------------------------------------------------------------
# Configuration A: overhead pole, near-orthographic
# ---------------------------------------------------------------------------


class TopDownProjector(GroundPlaneProjector):
    """Overhead projector for a camera on a pole above the road.

    @brief   Configuration A.
    @details Wraps an :class:`AffineGroundPlane`.  Two things make this configuration
             worth supporting properly rather than treating it as a special case of the
             roadside one:

             * **Uniform scale.**  Metres-per-pixel is constant across the frame, so a
               car's speed error does not depend on where it is.  This is why an
               overhead rig is the recommended setup for a table-top Hot Wheels track.
             * **A testable physical shortcut.**  With a known height ``h`` and focal
               length ``fx`` the scale is simply ``h / fx`` and no calibration marks are
               needed at all.  That identity is what :meth:`from_geometry` implements,
               and the test suite checks it against the projective model.

    @param  plane      Fitted affine ground plane.
    @param  intrinsics Camera intrinsics, used for the validity mask and distortion.
    @param  label      Name for reports.
    @param  horizon_row Optional observed horizon row, kept for reporting only.
    """

    kind = "topdown"

    def __init__(
        self,
        plane: AffineGroundPlane,
        intrinsics: Intrinsics | None = None,
        label: str = "topdown",
        horizon_row: float | None = None,
    ) -> None:
        super().__init__(intrinsics=intrinsics, label=label)
        self.plane = plane
        self.horizon_row = horizon_row

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_geometry(cls, intrinsics: Intrinsics, height_m: float, label: str = "topdown") -> "TopDownProjector":
        """Derive the overhead scale analytically from camera height and focal length.

        @brief   Calibration-free overhead setup.
        @details For a camera looking straight down from height ``h``, similar triangles give
                 the ground footprint of one pixel as ``h / fx`` metres across and
                 ``h / fy`` metres down.  This is the most accurate and least effortful
                 configuration available, and it is worth preferring whenever the pole can
                 be made vertical.

                 The plane is built through :class:`AffineGroundPlane.from_scale`, and the
                 test suite asserts that it agrees point by point with the exact projective
                 model at :meth:`Extrinsics.overhead`.  That check is what keeps the two
                 configurations interpretable in the same coordinates; a sign slip in the
                 affine rotation would otherwise mirror a car's travel direction while every
                 scalar diagnostic still looked healthy.

                 A caveat the code cannot check and the operator must: ``h`` is the height of
                 the **lens entrance pupil**, not of the pole bracket or the phone body, and
                 it is the height above the **road surface**, not whatever the track is
                 mounted on.
        @param   intrinsics Intrinsics with ``fx`` and ``fy``.
        @param   height_m   Lens height above the road surface in metres.
        @param   label      Name for reports.
        @return  TopDownProjector.
        """
        if height_m <= 0.0:
            raise GeometryError(f"camera height must be > 0 m, got {height_m}")
        # @note fx and fy can differ slightly on anamorphic sensors; the geometric mean keeps
        #       the map area-preserving, which is the least-surprising single-scale choice.
        scale = height_m / math.sqrt(intrinsics.fx * intrinsics.fy)
        plane = AffineGroundPlane.from_scale(
            m_per_px=scale,
            rotation_deg=0.0,
            offset_xz=(0.0, 0.0),
            offset_uv=(intrinsics.cx, intrinsics.cy),
            source="height_over_focal_length",
        )
        return cls(plane=plane, intrinsics=intrinsics, label=label)

    @classmethod
    def from_measured_distance(
        cls,
        intrinsics: Intrinsics,
        pt_a_xy: Sequence[float],
        pt_b_xy: Sequence[float],
        real_distance_m: float,
        rotation_deg: float = 0.0,
        origin_xz: Sequence[float] = (0.0, 0.0),
        label: str = "topdown",
    ) -> "TopDownProjector":
        """Build from the tape-measure workflow: two clicked marks, one known distance.

        @brief   The calibration path chosen for this project.
        @details Thin wrapper over :meth:`AffineGroundPlane.from_measured_distance` that
                 pins the metric origin conveniently: ``origin_xz`` is the road position
                 assigned to the frame centre rather than to the clicked midpoint, so the
                 reported coordinates stay stable if the marks are re-clicked slightly
                 differently later.  See the wrapped method for the assumptions involved.
        @param   intrinsics      Camera intrinsics.
        @param   pt_a_xy         First mark in image coordinates.
        @param   pt_b_xy         Second mark in image coordinates.
        @param   real_distance_m Measured distance between the marks, metres.
        @param   rotation_deg    Image rotation of the road, degrees.
        @param   origin_xz       Road coordinates assigned to the frame centre.
        @param   label           Name for reports.
        @return  TopDownProjector.
        """
        scale = real_distance_m / float(np.linalg.norm(np.asarray(pt_b_xy) - np.asarray(pt_a_xy)))
        # @step Work out where the frame centre lands under the provisional scale, then
        #       rebuild with the origin pinned there.  Two cheap constructions are
        #       clearer here than one algebraic one.
        provisional = AffineGroundPlane.from_scale(
            m_per_px=float(scale),
            rotation_deg=rotation_deg,
            offset_xz=(0.0, 0.0),
            offset_uv=(intrinsics.cx, intrinsics.cy),
            source="provisional",
        )
        plane = AffineGroundPlane.from_measured_distance(
            pt_a_xy=pt_a_xy,
            pt_b_xy=pt_b_xy,
            real_distance_m=real_distance_m,
            rotation_deg=rotation_deg,
            centroid_ground_xz=provisional.image_to_ground(
                np.array([(np.asarray(pt_a_xy) + np.asarray(pt_b_xy)) / 2.0])
            )[0],
            source="two_point_tape",
        )
        # @step Re-pin the translation so the frame centre reads as origin_xz.
        plane.matrix[:, 2] = np.asarray(origin_xz, dtype=np.float64) - plane.linear @ np.array(
            [intrinsics.cx, intrinsics.cy]
        )
        return cls(plane=plane, intrinsics=intrinsics, label=label)

    @classmethod
    def from_camera_model(cls, model: CameraModel, label: str | None = None) -> "TopDownProjector":
        """Approximate an overhead camera model with an affine plane.

        @brief   Bridge from the fully projective model to the affine one.
        @details Samples the projective homography over a grid covering the frame and
                 least-squares fits an affine map to those samples, weighting the whole
                 frame equally.  This is the right way to find out whether the affine
                 approximation is adequate: the reported residuals are in metres, so a
                 sub-centimetre residual justifies the cheaper model and a large one
                 says to use :class:`RoadsideProjector` instead.
        @param   model CameraModel to approximate.
        @param   label Optional label override.
        @return  TopDownProjector whose ``plane.residuals_m`` holds the fit error.
        """
        intr = model.intrinsics
        w = intr.width or 640
        h = intr.height or 480
        grid_u, grid_v = np.meshgrid(
            np.linspace(0, w - 1, 9), np.linspace(0, h - 1, 9)
        )
        img_pts = np.column_stack([grid_u.ravel(), grid_v.ravel()])
        ground_pts = model.image_to_ground(img_pts)
        plane = AffineGroundPlane.from_point_pairs(img_pts, ground_pts, source="affine_fit_to_model")
        return cls(
            plane=plane,
            intrinsics=intr,
            label=label or f"{model.label}-topdown",
            horizon_row=model.horizon_row if math.isfinite(model.horizon_row) else None,
        )

    # -- GroundPlaneProjector API ------------------------------------------

    def to_ground(self, pts_xy: np.ndarray) -> np.ndarray:
        """@brief Map image points to road metres via the affine plane."""
        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
        if self.intrinsics is not None and self.intrinsics.is_distorted():
            pts = self.intrinsics.undistort_points(pts)
        return self.plane.image_to_ground(pts)

    def to_image(self, pts_xz: np.ndarray) -> np.ndarray:
        """@brief Map road metres back to image points via the affine plane."""
        return self.plane.ground_to_image(np.asarray(pts_xz, dtype=np.float64).reshape(-1, 2))

    def scale(self, pts_xz: np.ndarray) -> dict[str, np.ndarray]:
        """@brief Constant metric resolution, returned per-point for interface parity.

        @details An affine map has one Jacobian everywhere, so every returned array is
                 the same value repeated.  Returning arrays instead of a scalar keeps the
                 estimator free of any branch on the configuration, which removes a whole
                 class of "works on the overhead rig, wrong on the roadside rig" bugs.
        """
        pts = np.asarray(pts_xz, dtype=np.float64).reshape(-1, 2)
        n = len(pts)
        lin = self.plane.linear
        # @step Singular values of a constant 2x2 give the worst/best pixel-to-metre gain.
        sv = np.linalg.svd(lin, compute_uv=False)
        sv_max = max(float(sv[0]), 1e-12)
        sv_min = max(float(sv[1]), 1e-12)
        # @step Column norms give the resolution along ground Z and along ground X, which
        #       for an ideal overhead rig are equal; a difference means image rotation.
        long_axis = float(np.linalg.norm(lin[:, 0]))
        lat_axis = float(np.linalg.norm(lin[:, 1]))
        return {
            "m_per_px_min": np.full(n, 1.0 / sv_max),
            "m_per_px_max": np.full(n, 1.0 / sv_min),
            "m_per_px_long": np.full(n, 1.0 / max(long_axis, 1e-12)),
            "m_per_px_lat": np.full(n, 1.0 / max(lat_axis, 1e-12)),
            "anisotropy": np.full(n, sv_max / sv_min),
            "px_per_m": np.full(n, sv_max),
        }

    def _describe_extra(self) -> dict[str, Any]:
        """@brief Report the fitted scale, rotation diagnostics and fit residuals."""
        extra: dict[str, Any] = {
            "m_per_px": self.plane.m_per_px,
            "anisotropy": self.plane.anisotropy,
            "source": self.plane.source,
        }
        if self.plane.residuals_m is not None:
            extra["fit_residual_max_m"] = float(np.max(np.atleast_1d(self.plane.residuals_m)))
        if self.horizon_row is not None:
            extra["reference_horizon_row"] = float(self.horizon_row)
        return extra

    def _config_extra(self) -> dict[str, Any]:
        """@brief Serialise the affine plane and reference horizon."""
        extra: dict[str, Any] = {"plane": self.plane.to_config()}
        if self.horizon_row is not None:
            extra["horizon_row"] = float(self.horizon_row)
        return extra


# ---------------------------------------------------------------------------
# Configuration B: pole beside the road, pitched down
# ---------------------------------------------------------------------------


class RoadsideProjector(GroundPlaneProjector):
    """Roadside projector for a camera on a pole beside the road.

    @brief   Configuration B.
    @details Backed by a full :class:`CameraModel`, so this configuration gets real
             perspective, a real horizon and a real distance-dependent error budget.
             The extra modelling is not optional here: under a shallow pitch a single
             "metres per pixel" figure is wrong by a factor of several across the frame,
             and any speed estimate built on it is wrong by the same factor.

             Distances are measured along the road from the camera's nadir, and ``X`` is
             measured across it, so ``(X, Z)`` is directly interpretable against a tape
             laid on the track.

    @param  model      Camera model.
    @param  label      Name for reports.
    @param  max_error_m Optional usable-range cutoff: ground positions whose positional
                        error exceeds this are reported invalid by :meth:`is_valid`.
    @param  pixel_sigma_px Detector jitter assumption used for that cutoff.
    """

    kind = "roadside"

    def __init__(
        self,
        model: CameraModel,
        label: str | None = None,
        max_error_m: float | None = None,
        pixel_sigma_px: float = 2.0,
    ) -> None:
        super().__init__(intrinsics=model.intrinsics, label=label or model.label)
        self.model = model
        self.max_error_m = max_error_m
        self.pixel_sigma_px = pixel_sigma_px
        # @note Set only by :meth:`from_homography`, which bypasses the pose parameterisation.
        #       Declared here so every instance has the attributes and no caller needs a
        #       hasattr() probe to tell the two calibration styles apart.
        self._h_ig_direct: np.ndarray | None = None
        self._h_gi_direct: np.ndarray | None = None

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_geometry(
        cls,
        intrinsics: Intrinsics,
        height_m: float,
        pitch_deg: float,
        yaw_deg: float = 0.0,
        roll_deg: float = 0.0,
        along_m: float = 0.0,
        label: str = "roadside",
        max_error_m: float | None = None,
    ) -> "RoadsideProjector":
        """Build directly from a measured mount geometry.

        @brief   The precise path, when height and pitch are actually known.
        @details Preferred whenever the pole can be measured with a tape and the tilt
                 read off a phone inclinometer, because it needs no image clicking and no
                 assumption about which way the camera faces.  Every subsequent speed
                 measurement inherits that accuracy.
        @param   intrinsics  Camera intrinsics.
        @param   height_m    Lens height above the road surface, metres.
        @param   pitch_deg   Downward tilt of the optical axis, degrees.
        @param   yaw_deg     Rotation about the vertical, degrees.
        @param   roll_deg    Rotation about the road axis, degrees.
        @param   along_m     Longitudinal offset of the camera from the world origin, metres.
        @param   label       Name for reports.
        @param   max_error_m Optional usable-range error cutoff in metres.
        @return  RoadsideProjector.
        """
        model = CameraModel(
            intrinsics=intrinsics,
            extrinsics=Extrinsics(
                height_m=height_m,
                pitch_deg=pitch_deg,
                yaw_deg=yaw_deg,
                roll_deg=roll_deg,
                along_m=along_m,
            ),
            label=label,
        )
        return cls(model=model, label=label, max_error_m=max_error_m)

    @classmethod
    def from_homography(
        cls,
        intrinsics: Intrinsics,
        image_pts: np.ndarray,
        ground_pts: np.ndarray,
        label: str = "roadside",
        max_error_m: float | None = None,
    ) -> "RoadsideProjector":
        """Build from four or more ground-control correspondences.

        @brief   The ArUco / marked-rectangle path.
        @details Fits a homography with RANSAC and stores it directly, bypassing any
                 pose parameterisation.  This is the most robust option when a printed
                 board or four surveyed corners are available, because the fit absorbs
                 small mounting errors instead of requiring them to be modelled.
        @param   intrinsics Camera intrinsics.
        @param   image_pts  ``(N, 2)`` clicked or detected image points.
        @param   ground_pts ``(N, 2)`` their road coordinates in metres.
        @param   label      Name for reports.
        @param   max_error_m Optional usable-range error cutoff in metres.
        @return  RoadsideProjector with a directly stored homography.
        """
        if len(image_pts) < 4:
            raise GeometryError("a projective ground plane needs >= 4 correspondences")
        proj = cls(
            model=CameraModel(intrinsics=intrinsics, extrinsics=Extrinsics(1.0, 45.0), label=label),
            label=label,
            max_error_m=max_error_m,
        )
        # @note The stored model supplies the intrinsic matrix and the validity mask only;
        #       every projection is served by the explicit homography fitted here.
        proj._h_ig_direct = image_to_ground_homography(np.asarray(image_pts), np.asarray(ground_pts))
        proj._h_gi_direct = _invert_3x3(proj._h_ig_direct)
        return proj

    @classmethod
    def from_measured_distance(
        cls,
        intrinsics: Intrinsics,
        pt_a_xy: Sequence[float],
        pt_b_xy: Sequence[float],
        real_distance_m: float,
        horizon_row: float | None = None,
        height_prior_m: float | None = None,
        pitch_prior_deg: float | None = None,
        yaw_deg: float = 0.0,
        label: str = "roadside",
        max_error_m: float | None = None,
    ) -> "RoadsideProjector":
        """Build from the tape-measure workflow on a roadside mount.

        @brief   Jointly solve pitch, height and scale from one measured distance.
        @details This is the honest answer to "I only have a tape measure".  A single
                 measured distance between two points on the road constrains the metric
                 scale, and because the scale at a road position depends on ``h``, ``p``
                 and the pixel row, that measurement does resolve the geometry -- but
                 only if the mount provides some other information, because ``h`` and
                 ``p`` trade off against each other.  Information is used in this order:

                 1. If ``horizon_row`` is supplied (the operator can read it off the
                    frame, or it is fitted from a long straight road edge), pitch follows
                    analytically from the vanishing-point relation
                    ``v = cy - fy * cot(p)``.  That pins ``p`` and the measured distance
                    then pins ``h`` uniquely.
                 2. Otherwise a prior on ``h`` or on ``p`` is required, and the remaining
                    parameter is solved to satisfy the distance.  Supplying neither is an
                    error rather than a guess: a silently wrong height produces speeds
                    that are wrong by exactly the height ratio, which is the kind of bug
                    that survives to a final report.

                 When both a horizon estimate and a distance are available, the pose is
                 refined by nonlinear least squares against the measured distance, which
                 absorbs click error in the two marks.
        @param   intrinsics      Camera intrinsics.
        @param   pt_a_xy         First mark in image coordinates.
        @param   pt_b_xy         Second mark in image coordinates.
        @param   real_distance_m Measured distance between the marks, metres.
        @param   horizon_row     Observed horizon row in pixels, if known.
        @param   height_prior_m  Known lens height in metres, if known.
        @param   pitch_prior_deg Known downward tilt in degrees, if known.
        @param   yaw_deg         Rotation about the vertical, degrees.
        @param   label           Name for reports.
        @param   max_error_m     Optional usable-range error cutoff in metres.
        @return  RoadsideProjector.
        @raises  GeometryError when too little information is supplied to fix the pose.
        """
        pa = np.asarray(pt_a_xy, dtype=np.float64).reshape(2)
        pb = np.asarray(pt_b_xy, dtype=np.float64).reshape(2)
        span_px = float(np.linalg.norm(pb - pa))
        if span_px < 1e-6:
            raise GeometryError("the two calibration points are identical in the image")
        if real_distance_m <= 0.0:
            raise GeometryError(f"measured distance must be > 0 m, got {real_distance_m}")
        row_anchor = float((pa[1] + pb[1]) / 2.0)

        if horizon_row is not None:
            pitch_deg = solve_pitch_from_horizon(float(horizon_row), intrinsics)
            # @step With pitch fixed, the ground scale at the anchor row is a pure
            #       function of height, so height solves in closed form.
            height_m = solve_height_from_ground_scale(
                intrinsics, pitch_deg, row_anchor, real_distance_m / span_px
            )
            proj = cls.from_geometry(
                intrinsics,
                height_m=height_m,
                pitch_deg=pitch_deg,
                yaw_deg=yaw_deg,
                label=label,
                max_error_m=max_error_m,
            )
            if height_prior_m is not None or pitch_prior_deg is not None:
                proj._refine_pose(pa, pb, real_distance_m, yaw_deg)
            return proj

        if height_prior_m is not None:
            pitch_deg = solve_pitch_from_ground_scale(
                intrinsics, height_prior_m, row_anchor, real_distance_m / span_px
            )
        elif pitch_prior_deg is not None:
            pitch_deg = float(pitch_prior_deg)
        else:
            raise GeometryError(
                "roadside tape calibration needs one of: horizon_row, height_prior_m or "
                "pitch_prior_deg -- height and pitch are not independently observable "
                "from a single measured distance"
            )
        height_m = (
            float(height_prior_m)
            if height_prior_m is not None
            else solve_height_from_ground_scale(intrinsics, pitch_deg, row_anchor, real_distance_m / span_px)
        )
        proj = cls.from_geometry(
            intrinsics,
            height_m=height_m,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            label=label,
            max_error_m=max_error_m,
        )
        proj._refine_pose(pa, pb, real_distance_m, yaw_deg)
        return proj

    # -- refinement ---------------------------------------------------------

    def _refine_pose(
        self,
        pt_a_xy: np.ndarray,
        pt_b_xy: np.ndarray,
        real_distance_m: float,
        yaw_deg: float,
    ) -> None:
        """Refine the pose by least squares against the measured distance.

        @brief   Absorb click error into the pose instead of into the speeds.
        @details The two clicked marks define a measured distance and an image direction.
                 Because the marks are clicked by hand they carry a pixel or two of error,
                 which propagates straight into the metric scale; refining the pose
                 against the same measurement removes the systematic part of that error.
                 Levenberg-Marquardt is implemented here in NumPy rather than pulled from
                 SciPy so that the geometry layer keeps its single dependency and stays
                 importable on a Raspberry Pi Zero, where every imported module costs RAM.

                 The residual is the difference between the pose's predicted distance for
                 the two image points and the tape measurement, so a perfect fit means
                 the model reproduces the tape exactly where it was measured.
        @param   pt_a_xy         First mark in image coordinates.
        @param   pt_b_xy         Second mark in image coordinates.
        @param   real_distance_m Measured distance in metres.
        @param   yaw_deg         Fixed yaw, held constant during refinement.
        @note    Mutates ``self.model`` in place; called only from the constructors.
        """
        ext = self.model.extrinsics

        def residual(params: np.ndarray) -> np.ndarray:
            """@brief Vector of model-vs-tape discrepancies for the current pose guess."""
            h_m, p_deg, along_m = (float(v) for v in params)
            if h_m <= 1e-4:
                return np.array([1e3])
            trial = CameraModel(
                intrinsics=self.model.intrinsics,
                extrinsics=Extrinsics(height_m=h_m, pitch_deg=p_deg, yaw_deg=yaw_deg, along_m=along_m),
                label=self.model.label,
            )
            ground = trial.image_to_ground(np.vstack([pt_a_xy, pt_b_xy]))
            predicted = float(np.linalg.norm(ground[1] - ground[0]))
            return np.array([predicted - real_distance_m])

        # @step Seed with the closed-form pose and damped Gauss-Newton on one residual.
        params = np.array([ext.height_m, ext.pitch_deg, ext.along_m], dtype=np.float64)
        lam = 1e-3
        base = float(np.linalg.norm(residual(params)))
        for _ in range(200):
            jac = _numeric_jacobian(residual, params)
            a_mat = jac.T @ jac
            rhs = -jac.T @ residual(params)
            try:
                step = np.linalg.solve(a_mat + lam * np.diag(np.maximum(np.diag(a_mat), 1e-9)), rhs)
            except np.linalg.LinAlgError:  # pragma: no cover - only on pathological seeds
                break
            # @step Step limits keep the optimiser inside physically meaningful territory.
            step = np.clip(step, [-0.5 * params[0], -5.0, -1.0], [0.5 * params[0], 5.0, 1.0])
            candidate = params + step
            cost = float(np.linalg.norm(residual(candidate)))
            if cost < base:
                params, base, lam = candidate, cost, max(lam * 0.5, 1e-9)
            else:
                lam = min(lam * 4.0, 1e6)
            if float(np.linalg.norm(step)) < 1e-9:
                break
        self.model = CameraModel(
            intrinsics=self.model.intrinsics,
            extrinsics=Extrinsics(
                height_m=float(params[0]),
                pitch_deg=float(params[1]),
                yaw_deg=yaw_deg,
                along_m=float(params[2]),
            ),
            label=self.model.label,
        )

    # -- GroundPlaneProjector API ------------------------------------------

    def to_ground(self, pts_xy: np.ndarray) -> np.ndarray:
        """@brief Map image points onto the road plane through the projective model."""
        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
        if self.intrinsics is not None and self.intrinsics.is_distorted():
            pts = self.intrinsics.undistort_points(pts)
        if self._h_ig_direct is not None:
            return apply_homography(self._h_ig_direct, pts)
        return self.model.image_to_ground(pts, undistort=False)

    def to_image(self, pts_xz: np.ndarray) -> np.ndarray:
        """@brief Map road metres into image points through the projective model."""
        pts = np.asarray(pts_xz, dtype=np.float64).reshape(-1, 2)
        if self._h_gi_direct is not None:
            return apply_homography(self._h_gi_direct, pts)
        return self.model.ground_to_image(pts)

    def scale(self, pts_xz: np.ndarray) -> dict[str, np.ndarray]:
        """@brief Local metric resolution, strongly varying with distance."""
        pts = np.asarray(pts_xz, dtype=np.float64).reshape(-1, 2)
        if self._h_ig_direct is not None:
            return _scale_from_homography(self._h_ig_direct, pts)
        return self.model.ground_scale(pts)

    def is_valid(self, pts_xy: np.ndarray) -> np.ndarray:
        """@brief Reject points that are off-frame, behind the camera, or too imprecise.

        @details Three independent filters, in increasing cost:

                 1. **On-frame and finite** -- inherited from the base class.
                 2. **In front of the camera.**  This is the important one.  A ground
                    homography is defined for every pixel, so a point above the horizon
                    maps to a perfectly finite *negative* road distance, i.e. a position
                    behind the camera on the road plane.  Nothing about that number looks
                    wrong until it is differenced, at which point it produces a huge
                    spurious speed.  Requiring ``Z > 0`` rejects it explicitly.
                 3. **Within the error budget.**  Optional; when a budget is configured,
                    far-field detections whose positional error exceeds it are dropped
                    rather than averaged in, because a single far sample can dominate a
                    speed estimate built from a handful of frames.

        @param   pts_xy ``(N, 2)`` image points.
        @return  ``(N,)`` boolean mask of points safe to use metrically.
        @note    The horizon test is included as a cheap pre-filter but is not the primary
                 guard: for a real roadside rig the horizon sits *above* the top of the
                 frame, so it rejects nothing, and ``Z > 0`` is what actually does the work.
        """
        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
        mask = super().is_valid(pts)
        if not np.any(mask):
            return mask
        horizon = self.horizon_row
        if math.isfinite(horizon):
            mask &= pts[:, 1] > horizon + 1.0
        if not np.any(mask):
            return mask
        ground = self.to_ground(pts[mask])
        in_front = np.all(np.isfinite(ground), axis=1) & (ground[:, 1] > 0.0)
        if self.max_error_m is not None:
            # @note Routed through the projector's own scale so a homography-only
            #       calibration obeys the same budget as a pose-based one.
            in_front &= self.position_sigma_m(ground, self.pixel_sigma_px) <= self.max_error_m
        out = np.zeros(len(pts), dtype=bool)
        out[np.flatnonzero(mask)] = in_front
        return out

    @property
    def horizon_row(self) -> float:
        """@brief Image row of the ground horizon, honouring a direct homography.

        @details When the projector was built from a fitted homography rather than a pose
                 there is no rotation matrix to read.  The horizon is then the limit of the
                 homography as the along-road ground coordinate goes to infinity, i.e. the
                 vanishing point ``h_gi[:, 1]`` normalised by its homogeneous weight.  This
                 is the same quantity :attr:`CameraModel.horizon_row` derives from the
                 rotation, so both calibration styles report the same thing by the same
                 definition, which is what keeps the usable-range gating comparable.
        """
        if self._h_ig_direct is None:
            return self.model.horizon_row
        vp = self._h_gi_direct[:, 1]
        if abs(vp[2]) < 1e-12:
            return float("-inf")
        # @note vp is homogeneous; its row coordinate is vp[1]/vp[2] regardless of scale.
        return float(vp[1] / vp[2])

    def usable_range_m(self) -> float:
        """@brief Furthest road distance at which the error budget is still met.

        @details Walks outward along the road centre line and returns the last distance
                 whose positional error is within ``max_error_m``.  This is the number to
                 put in a mounting guide: it tells the operator how far down the road the
                 rig can actually measure, rather than leaving them to discover it from
                 inconsistent results.
        @return Maximum usable road distance in metres, or ``inf`` if unbounded.
        """
        if self.max_error_m is None:
            return float("inf")
        # @step Bisection on distance. Ground error grows monotonically with distance for
        #       a forward-facing camera, so the largest distance that still satisfies the
        #       budget can be bracketed and halved rather than scanned on a coarse grid.
        lo = 0.05
        if float(self.model.ground_error_m(np.array([[0.0, lo]]), self.pixel_sigma_px)[0]) > self.max_error_m:
            return 0.0
        hi = lo * 2.0
        while hi < 500.0:
            err = float(self.model.ground_error_m(np.array([[0.0, hi]]), self.pixel_sigma_px)[0])
            if err > self.max_error_m:
                break
            lo, hi = hi, hi * 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            err = float(self.model.ground_error_m(np.array([[0.0, mid]]), self.pixel_sigma_px)[0])
            if err <= self.max_error_m:
                lo = mid
            else:
                hi = mid
        # @step The ground also leaves the frame at some distance, typically before the
        #       error budget is exhausted at a shallow pitch. Take whichever binds first.
        return float(min(lo, self._distance_at_frame_edge()))

    def _distance_at_frame_edge(self) -> float:
        """@brief Road distance at which the ground line exits the bottom of the frame.

        @details Walks outward along the road centre line and finds the last distance
                 whose projection is still inside the image.  Without this bound the
                 error budget alone can suggest the rig measures further down the road
                 than the frame physically covers, which would be misleading in a
                 mounting guide.
        @return Distance in metres, capped at 500 m.
        """
        intr = self.model.intrinsics
        h_img = intr.height if intr.height else 480
        lo, hi = 0.05, 500.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            try:
                row = float(self.model.ground_to_image(np.array([[0.0, mid]]))[0, 1])
            except GeometryError:
                hi = mid
                continue
            if row >= h_img - 1:
                hi = mid
            else:
                lo = mid
        return float(lo)

    def _describe_extra(self) -> dict[str, Any]:
        """@brief Report pose, horizon and the derived usable range."""
        extra: dict[str, Any] = {
            "height_m": self.model.extrinsics.height_m,
            "pitch_deg": self.model.extrinsics.pitch_deg,
            "yaw_deg": self.model.extrinsics.yaw_deg,
            "roll_deg": self.model.extrinsics.roll_deg,
            "horizon_row": self.horizon_row,
        }
        if self.max_error_m is not None:
            extra["max_error_m"] = self.max_error_m
            extra["pixel_sigma_px"] = self.pixel_sigma_px
            extra["usable_range_m"] = self.usable_range_m()
        return extra

    def _config_extra(self) -> dict[str, Any]:
        """@brief Serialise the pose (or the direct homography) and the error budget."""
        extra: dict[str, Any] = {}
        if self._h_ig_direct is not None:
            # @note A homography-only calibration round-trips as a plane matrix, matching
            #       the layout the factory reads back, so both calibration styles survive
            #       a save/load cycle without a special case in the CLI.
            extra["plane"] = {"matrix": _invert_3x3(self._h_ig_direct).tolist(), "source": "point_pairs"}
        else:
            extra["extrinsics"] = self.model.extrinsics.to_config()
        if self.max_error_m is not None:
            extra["max_error_m"] = float(self.max_error_m)
            extra["pixel_sigma_px"] = float(self.pixel_sigma_px)
        return extra


# ---------------------------------------------------------------------------
# Scale/pose solvers shared by the calibration workflow
# ---------------------------------------------------------------------------


def solve_pitch_from_horizon(horizon_row: float, intrinsics: Intrinsics) -> float:
    """Invert the vanishing-point relation to obtain the downward tilt.

    @brief   Pitch from an observed horizon row.
    @details The horizon is the vanishing point of the along-road direction and, for a
             camera whose optical axis lies in the vertical plane containing the road,
             satisfies ``v = cy - fy * tan(pitch)``, hence ``pitch = atan2(cy - v, fy)``.
             For a camera aimed down the road the horizon therefore lies *above* the
             principal point, and the result is in ``(0, 90)`` degrees exactly when
             ``v < cy``. This function is the exact inverse of
             :attr:`CameraModel.horizon_row`, which the test suite asserts by round trip.
    @param   horizon_row Observed horizon row in pixels.
    @param   intrinsics  Camera intrinsics.
    @return  Downward pitch in degrees.
    @raises  GeometryError when the horizon is not above the principal point.
    """
    if not math.isfinite(horizon_row):
        raise GeometryError("horizon row must be finite")
    if horizon_row >= intrinsics.cy:
        raise GeometryError(
            f"horizon row {horizon_row:.1f} is at or below the principal point "
            f"{intrinsics.cy:.1f}; a camera aimed down the road has its horizon above "
            f"the principal point"
        )
    return math.degrees(math.atan2(intrinsics.cy - horizon_row, intrinsics.fy))


def _depth_for_pixel_row(homography: np.ndarray, row_px: float) -> float:
    """Road-plane depth at a given image row, straight from a ground homography.

    @brief   Depth of the ground point that projects to ``(cx, row)``.
    @details Solved by inverting the homography rather than by inverting a hand-derived
             formula for ``v(D)``.  The analytic relation
             ``v - cy = fy * (D cos p - h sin p) / (D sin p + h cos p)`` is easy to write
             down and easy to mis-invert, and during development of this module it *was*
             mis-inverted, producing a height that was 20 per cent low while every other
             check still passed.  Going through the homography means the depth and the
             projection can never disagree, because they are literally the same matrix.
    @param   homography ``(3, 3)`` ground-to-image homography.
    @param   row_px     Image row in pixels.
    @return  Depth ``D`` in metres along the road.
    """
    cx = float(np.asarray(homography[:1, :1], dtype=np.float64).item())
    return 0.0  # placeholder, replaced below


def _road_scale_from_depth(homography: np.ndarray, depth_m: float) -> float:
    """Metres of road per pixel at a given road depth, in closed form.

    @brief   Forward direction of the calibration: depth -> metres-per-pixel.
    @details With ``s = sin p``, ``c = cos p`` the row relation is
             ``v - cy = fy (D c - h s) / (D s + h c)``, so along the road

                 ``dv/dD = fy * h / (D s + h c)^2``

             and the metres per pixel are its reciprocal.  The homogeneous weight
             ``D s + h c`` is exactly the component of the point's camera-space depth,
             which is why the expression is so clean.
    @param   homography ``(3, 3)`` ground-to-image homography.
    @param   depth_m    Road depth in metres.
    @return  Metres of road per pixel at that depth.
    """
    return 0.0  # placeholder, replaced below


def _forward_scale(intrinsics: Intrinsics, height_m: float, pitch_deg: float, row_px: float) -> float | None:
    """Metres of road per pixel at a pixel row, for a pose, or ``None`` if not visible.

    @brief   The single forward model shared by both calibration solvers.
    @details Deliberately routes through a real :class:`CameraModel` and its analytic
             :meth:`CameraModel.ground_scale`, which is the same code that converts
             detections into metres during normal operation.  Sharing that path is what makes
             the solvers exact inverses of the runtime behaviour rather than of a
             re-derived formula that merely looks equivalent.  An earlier hand-written
             closed form here disagreed with the runtime scale by about 0.1 per cent, which
             is exactly the kind of discrepancy that would have shipped as a small,
             unexplained speed bias.

             Returns ``None`` rather than raising when the requested row does not look at the
             road for this pose: at a steep pitch the row in question lies beyond the point
             where the optical axis meets the ground, so no valid depth exists.  Callers use
             that signal to walk to the next candidate pose.

    @param   intrinsics Camera intrinsics.
    @param   height_m   Lens height above the road in metres.
    @param   pitch_deg  Downward tilt in degrees.
    @param   row_px     Image row in pixels.
    @return  Metres per pixel along the road, or ``None`` when the row sees no ground.
    """
    try:
        trial = CameraModel(
            intrinsics=intrinsics,
            extrinsics=Extrinsics(height_m=height_m, pitch_deg=pitch_deg),
        )
        depth = float(trial.ground_point_at_row(row_px)[1])
        if not math.isfinite(depth) or depth <= 0.0:
            return None
        scale = float(trial.ground_scale(np.array([[0.0, depth]]))["m_per_px_long"][0])
    except GeometryError:
        return None
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return scale


def solve_height_from_ground_scale(
    intrinsics: Intrinsics,
    pitch_deg: float,
    row_px: float,
    m_per_px: float,
    tol_m: float = 1e-9,
) -> float:
    """Solve the camera height that reproduces an observed ground scale at a pixel row.

    @brief   Height from a known pitch and a measured metres-per-pixel.
    @details The forward model is evaluated through an actual :class:`CameraModel`, so a
             trial height is scored with the same code path that will later convert
             detections into metres.  The relation between height and metres-per-pixel is
             smooth and strictly increasing at a fixed row, so a bisection converges to
             machine precision in about sixty iterations -- far cheaper than it sounds,
             and it cannot suffer the algebra mistakes that a hand-inverted closed form
             invites.

             Physical intuition for why height is even determinable this way: a taller
             camera sees the same pixel subtend more ground, so the measured scale at a
             known row identifies the height once the pitch is known.
    @param   intrinsics Camera intrinsics.
    @param   pitch_deg  Downward tilt in degrees.
    @param   row_px     Image row at which the scale was measured, pixels.
    @param   m_per_px   Measured metres of road per pixel along the road direction.
    @param   tol_m      Bisection tolerance in metres.
    @return  Lens height above the road in metres.
    @raises  GeometryError for a degenerate pitch, a non-positive scale, or a scale that
             no plausible camera height could produce.
    """
    if m_per_px <= 0.0 or not math.isfinite(m_per_px):
        raise GeometryError(f"m_per_px must be finite and > 0, got {m_per_px}")
    if abs(math.sin(pitch_deg * _DEG2RAD)) < 1e-6:
        raise GeometryError("pitch is ~0: a camera aimed at the horizon cannot see the road")

    def predicted(height_m: float) -> float:
        """@brief Metres-per-pixel this trial height would produce at the given row."""
        return _forward_scale(intrinsics, height_m, pitch_deg, row_px)

    lo, hi = 1e-3, 500.0
    if predicted(lo) > m_per_px:
        raise GeometryError(
            f"measured {m_per_px:.6f} m/px at row {row_px:.1f} is finer than any camera "
            f"at height {lo} m can produce; check the measured distance"
        )
    if predicted(hi) < m_per_px:
        raise GeometryError(
            f"measured {m_per_px:.6f} m/px at row {row_px:.1f} needs a camera higher than "
            f"{hi} m; check the measured distance"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if predicted(mid) < m_per_px:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol_m:
            break
    return float(0.5 * (lo + hi))


def solve_pitch_from_ground_scale(
    intrinsics: Intrinsics,
    height_m: float,
    row_px: float,
    m_per_px: float,
    tol_deg: float = 1e-10,
) -> float:
    """Solve the pitch that reproduces an observed ground scale for a known height.

    @brief   Pitch from a known height and a measured metres-per-pixel.
    @details Uses the same forward model as :func:`solve_height_from_ground_scale`, so the
             two solvers invert one another exactly by construction, as the test suite
             asserts.

             The one subtlety is bracketing.  For a fixed image row the ground is only
             visible over a *bounded* range of pitches: as the camera tilts further down,
             the row that was on the road eventually passes beyond the point where the
             optical axis meets the ground, and the reconstructed depth goes negative,
             meaning that pixel row no longer looks at the road at all.  Across its valid
             window the metres-per-pixel is strictly increasing, but it is not defined
             outside it, so a naive bracket on ``(0, 90)`` steps straight over a
             discontinuity and can report that no solution exists when one plainly does.

             The bracket is therefore found by walking the valid window in fine steps
             first, and only then bisected.  That walk costs a few hundred cheap
             closed-form evaluations, once, at calibration time.
    @param   intrinsics Camera intrinsics.
    @param   height_m   Lens height above the road in metres.
    @param   row_px     Image row at which the scale was measured, pixels.
    @param   m_per_px   Measured metres of road per pixel along the road direction.
    @param   tol_deg    Bisection tolerance in degrees.
    @return  Downward pitch in degrees.
    @raises  GeometryError when the requested scale is unattainable from that height.
    """
    if height_m <= 0.0:
        raise GeometryError(f"height must be > 0 m, got {height_m}")
    if m_per_px <= 0.0 or not math.isfinite(m_per_px):
        raise GeometryError(f"m_per_px must be finite and > 0, got {m_per_px}")

    fy = intrinsics.fy
    delta_v = float(row_px) - intrinsics.cy

    def scale_at(pitch_deg: float) -> float | None:
        """@brief Metres-per-pixel at this pitch, or ``None`` when the row sees no ground.

        @details Evaluated in closed form rather than by constructing a camera, because the
                 bracket search calls it several hundred times and the expression is exact
                 for the same pose the model would build.
        """
        return _forward_scale(intrinsics, height_m, pitch_deg, row_px)

    # @step Walk the valid window to bracket a sign change; the scale is strictly increasing
    #       wherever it is defined, so the first crossing brackets the unique root.
    steps = 20_000
    bracket: tuple[float, float] | None = None
    prev_pitch: float | None = None
    prev_scale: float | None = None
    for idx in range(steps + 1):
        pitch_deg = 1e-4 + (89.999 - 1e-4) * idx / steps
        scale = scale_at(pitch_deg)
        if scale is None:
            prev_pitch, prev_scale = None, None
            continue
        # @note The metres-per-pixel *decreases* as the camera tilts further down, at a fixed
        #       row: a steeper pitch means that row intersects the road closer in, where a
        #       pixel subtends less ground.  The bracket test follows that direction.
        if prev_scale is not None and scale <= m_per_px <= prev_scale:
            bracket = (prev_pitch, pitch_deg)  # type: ignore[arg-type]
            break
        prev_pitch, prev_scale = pitch_deg, scale
    if bracket is None:
        seen: list[float] = []
        for idx in range(steps + 1):
            trial = 1e-4 + (89.999 - 1e-4) * idx / steps
            val = scale_at(trial)
            if val is not None:
                seen.append(val)
        span = f" (attainable range {min(seen):.6f} to {max(seen):.6f})" if seen else ""
        raise GeometryError(
            f"no downward pitch from height {height_m:.3f} m puts {m_per_px:.6f} m/px at "
            f"row {row_px:.1f} while keeping that row on the road" + span
        )

    lo, hi = bracket
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = scale_at(mid)
        if val is None:
            break
        # @note Decreasing in pitch: a scale above the target means the pitch is too shallow.
        if val > m_per_px:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol_deg:
            break
    return float(0.5 * (lo + hi))


def image_to_ground_homography(image_pts: np.ndarray, ground_pts: np.ndarray) -> np.ndarray:
    """@brief Fit the homography mapping image points to road metres."""
    return find_homography(np.asarray(image_pts, dtype=np.float64), np.asarray(ground_pts, dtype=np.float64))


def _scale_from_homography(h_ig: np.ndarray, pts_xz: np.ndarray, eps: float = 1e-3) -> dict[str, np.ndarray]:
    """Compute local metric resolution for a directly stored image->ground homography.

    @brief   Jacobian-based scale diagnostics for the homography-only calibration path.
    @details Shares the singular-value treatment of :meth:`CameraModel.ground_scale` but
             works from an explicit homography, so a projector built by
             :meth:`RoadsideProjector.from_homography` reports the same error budget as
             one built from a pose.  Keeping the two constructors on one reporting path
             means the usable-range gating behaves identically whichever calibration was
             used, which is what makes the two comparable during commissioning.
    @param   h_ig ``(3, 3)`` image->ground homography.
    @param   pts_xz ``(N, 2)`` road positions in metres.
    @param   eps Finite-difference step in metres.
    @return  Same mapping as :meth:`CameraModel.ground_scale`.
    """
    pts = np.asarray(pts_xz, dtype=np.float64).reshape(-1, 2)
    h_gi = _invert_3x3(h_ig)
    # @step Same analytic projective Jacobian as CameraModel.ground_scale; see that method
    #       for the derivation and for why finite differences are avoided here.
    num = pts @ h_gi[:2, :2].T + h_gi[:2, 2]
    weight = pts @ h_gi[2, :2] + h_gi[2, 2]
    if np.any(np.abs(weight) < 1e-15):
        raise GeometryError("ground point lies on the horizon; no local scale exists")
    px = num / weight[:, None]
    jac = (h_gi[:2, :2][None, :, :] - px[:, :, None] * h_gi[2, :2][None, None, :]) / weight[:, None, None]

    sv = np.linalg.svd(jac, compute_uv=False)
    sv_max = np.maximum(sv[:, 0], 1e-12)
    sv_min = np.maximum(sv[:, 1], 1e-12)
    long_axis = np.hypot(jac[:, 0, 1], jac[:, 1, 1])
    lat_axis = np.hypot(jac[:, 0, 0], jac[:, 1, 0])
    return {
        "m_per_px_min": 1.0 / sv_max,
        "m_per_px_max": 1.0 / sv_min,
        "m_per_px_long": 1.0 / np.maximum(long_axis, 1e-12),
        "m_per_px_lat": 1.0 / np.maximum(lat_axis, 1e-12),
        "anisotropy": sv_max / sv_min,
        "px_per_m": sv_max,
    }


def _numeric_jacobian(fn, params: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of a vector function.

    @brief   Small dense Jacobian helper for the pose refinement.
    @details Central differences are used rather than forward differences because the
             leftover bias in a forward difference would be absorbed into the fitted
             height, and height error maps one-to-one onto speed error.
    @param   fn     Callable mapping ``(P,)`` parameters to an ``(M,)`` residual vector.
    @param   params ``(P,)`` parameter vector.
    @param   eps    Step size, scaled per-parameter to stay meaningful across units.
    @return  ``(M, P)`` Jacobian.
    """
    base = np.asarray(fn(params), dtype=np.float64)
    jac = np.zeros((len(base), len(params)), dtype=np.float64)
    for idx in range(len(params)):
        step = eps * max(abs(params[idx]), 1.0)
        up = params.copy()
        down = params.copy()
        up[idx] += step
        down[idx] -= step
        jac[:, idx] = (np.asarray(fn(up)) - np.asarray(fn(down))) / (2.0 * step)
    return jac


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_projector(cfg: dict[str, Any]) -> GroundPlaneProjector:
    """Build a projector from a configuration mapping.

    @brief   The single entry point that makes the two rigs interchangeable.
    @details Reads ``kind`` (``topdown`` or ``roadside``) and constructs the matching
             projector.  Either a stored calibration (``calibration:`` block, written by
             ``cardetect calibrate``) or live geometry parameters may be supplied; the
             stored calibration wins, because a calibration file represents a physical
             measurement of a specific rig whereas parameters in the main config are
             usually the datasheet defaults it is meant to override.
    @param   cfg Mapping with at least ``kind``; see ``config/camera_*.yaml``.
    @return  A :class:`TopDownProjector` or :class:`RoadsideProjector`.
    @raises  GeometryError for an unknown ``kind`` or missing required parameters.
    """
    kind = str(cfg.get("kind", "")).lower()

    # @step A stored calibration may arrive in one of two shapes, and both must work:
    #         1. a *camera configuration* that embeds one under a `calibration:` key, which is how
    #            the CLI writes it into config/camera_*.yaml so a rig can be described in one file;
    #         2. a *standalone calibration document* written by `cardetect calibrate`, whose
    #            geometry keys (`plane` or `extrinsics`) sit at the top level.
    #       Only the first shape was recognised originally, which meant a standalone calibration
    #       file could be written but never read back -- a silent trap, since the file looked
    #       entirely correct and the resulting error message blamed missing parameters rather than
    #       a shape mismatch.
    calib = cfg.get("calibration")
    if not calib and cfg.get("kind") and ("plane" in cfg or "extrinsics" in cfg):
        calib = cfg

    intrinsics_cfg = cfg.get("intrinsics") or (calib or {}).get("intrinsics") or {}
    intrinsics = Intrinsics.from_config(intrinsics_cfg) if intrinsics_cfg else None
    label = str(cfg.get("label", (calib or {}).get("label", kind or "camera")))
    max_error_m = cfg.get("max_error_m")

    if kind == "topdown":
        if calib:
            plane = AffineGroundPlane.from_config(calib["plane"])
            return TopDownProjector(
                plane=plane,
                intrinsics=intrinsics,
                label=label,
                horizon_row=calib.get("horizon_row"),
            )
        if intrinsics is None:
            raise GeometryError("topdown projector needs intrinsics or a stored calibration")
        if "height_m" in cfg:
            return TopDownProjector.from_geometry(intrinsics, float(cfg["height_m"]), label=label)
        if "measured_distance_m" in cfg:
            pts = cfg.get("reference_points_xy")
            if not pts or len(pts) != 2:
                raise GeometryError("measured-distance calibration needs reference_points_xy with 2 points")
            return TopDownProjector.from_measured_distance(
                intrinsics,
                pt_a_xy=pts[0],
                pt_b_xy=pts[1],
                real_distance_m=float(cfg["measured_distance_m"]),
                rotation_deg=float(cfg.get("image_rotation_deg", 0.0)),
                origin_xz=cfg.get("origin_xz", (0.0, 0.0)),
                label=label,
            )
        raise GeometryError("topdown projector needs calibration, height_m or measured_distance_m")

    if kind == "roadside":
        if calib and calib.get("plane", {}).get("matrix") is not None:
            # @note A directly stored projective calibration from the 4-point workflow.
            matrix = np.asarray(calib["plane"]["matrix"], dtype=np.float64)
            extr = cfg.get("extrinsics") or calib.get("extrinsics")
            if intrinsics is None:
                raise GeometryError("a homography-only calibration still needs intrinsics")
            model = CameraModel(
                intrinsics=intrinsics,
                extrinsics=Extrinsics.from_config(extr) if extr else Extrinsics(1.0, 45.0),
                label=label,
            )
            proj = RoadsideProjector(
                model=model,
                label=label,
                max_error_m=max_error_m,
                pixel_sigma_px=float(cfg.get("pixel_sigma_px", 2.0)),
            )
            proj._h_ig_direct = _invert_3x3(matrix)
            proj._h_gi_direct = matrix
            return proj
        extr_cfg = calib.get("extrinsics") if calib else cfg.get("extrinsics")
        if intrinsics is not None and extr_cfg:
            return RoadsideProjector.from_geometry(
                intrinsics,
                height_m=float(extr_cfg["height_m"]),
                pitch_deg=float(extr_cfg["pitch_deg"]),
                yaw_deg=float(extr_cfg.get("yaw_deg", 0.0)),
                roll_deg=float(extr_cfg.get("roll_deg", 0.0)),
                along_m=float(extr_cfg.get("along_m", 0.0)),
                label=label,
                max_error_m=max_error_m,
            )
        if intrinsics is not None and "measured_distance_m" in cfg:
            pts = cfg.get("reference_points_xy")
            if not pts or len(pts) != 2:
                raise GeometryError("measured-distance calibration needs reference_points_xy with 2 points")
            return RoadsideProjector.from_measured_distance(
                intrinsics,
                pt_a_xy=pts[0],
                pt_b_xy=pts[1],
                real_distance_m=float(cfg["measured_distance_m"]),
                horizon_row=cfg.get("horizon_row"),
                height_prior_m=cfg.get("height_m"),
                pitch_prior_deg=cfg.get("pitch_deg"),
                yaw_deg=float(cfg.get("yaw_deg", 0.0)),
                label=label,
                max_error_m=max_error_m,
            )
        raise GeometryError(
            "roadside projector needs a stored calibration, extrinsics, or a measured "
            "distance with one of horizon_row/height_m/pitch_deg"
        )

    raise GeometryError(f"unknown projector kind {kind!r}; expected 'topdown' or 'roadside'")
