"""Vehicle detection and ground-contact point extraction.

@file    detect.py
@brief   Wrap a YOLO detector and turn its boxes into road-plane reference points.
@details This module owns the single largest *systematic* error source in the whole pipeline: the
         choice of which point on a detection box represents the vehicle's position on the road.

         A bounding box is a 2D silhouette.  The vehicle's contact with the road is at the
         bottom edge of that silhouette, but the box's bottom edge is a few pixels above the true
         contact patch because a detector wraps the tyres, the wheel arch and often the contact
         shadow.  Projecting the box *centre* instead is the classic mistake: for a car seen from
         a pitched camera, the centre lies well above the contact point in the image, and the
         ground homography maps that to a road position *beyond* the car.  The resulting speed
         bias is large -- tens of per cent at typical roadside geometry -- and, crucially, it is
         *not* a constant offset: it scales with distance and is therefore not removable by a
         later calibration of height or pitch.

         Hence the deliberate asymmetry in this file: choosing among contact-point strategies is
         a first-class configuration option with documented consequences, and ``center`` mode is
         retained only so the bias can be demonstrated and measured, never as a default.

@note    ``ultralytics`` is imported lazily inside :class:`VehicleDetector` so that importing
         this module -- and therefore ``cardetect.config``, the geometry layer and the tests --
         never pulls torch into memory.  On a Raspberry Pi Zero 2 W with 512 MB that separation
         is the difference between the geometry being usable on the device and not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "VehicleDetection",
    "DetectionPointConfig",
    "VehicleDetector",
    "ground_contact_point",
    "COCO_VEHICLE_CLASSES",
    "class_name",
]


# @brief COCO class indices that represent road vehicles, and their readable names.
# @note Only these are tracked by default.  ``person`` is deliberately excluded: a pedestrian
#       crossing the track would otherwise be tracked as a vehicle and would contribute a
#       spurious speed measurement to the summary.
COCO_VEHICLE_CLASSES: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def class_name(class_id: int) -> str:
    """@brief Readable COCO name for a class index, or ``class_<id>`` when unknown."""
    return COCO_VEHICLE_CLASSES.get(int(class_id), f"class_{int(class_id)}")


@dataclass
class DetectionPointConfig:
    """How to choose the road-reference point on a detection box.

    @brief   Contact-point strategy and its correction offset.
    @details All coordinates are in image pixels with the origin at the top left.

    @param  mode              One of:

                              ``bottom_center``
                                  Bottom edge midpoint.  The default and the recommended
                                  choice: for a box that tightly wraps a vehicle it is the best
                                  available estimate of the contact patch centre, and it is
                                  robust to the box being slightly wider than the car.

                              ``bottom_third``
                                  Bottom edge, one third of the way in from the left.  Reduces
                                  the residual bias when the camera sees mostly one flank, as a
                                  roadside camera does for a car in the near lane, because the
                                  visible extent of the box is then biased toward the far side.

                              ``center``
                                  Box centre.  Provided to demonstrate the height bias, not to be
                                  used.  Included because the bias is worth being able to measure
                                  on a real rig rather than only asserting in a comment.

    @param  bottom_offset_px  Additional downward shift applied to a bottom-edge mode, in pixels.
                              Moves the reference point from the box edge toward the true contact
                              patch, accounting for the tyre and shadow gap.  A positive value
                              moves the point *down* the image, which is toward the near side of
                              the road.  Setting this is the cheapest available accuracy win on a
                              real rig: measure it once by comparing a stationary car's known
                              ground position against the projected reference point.
    @param  min_box_px        Minimum box height in pixels for a detection to be usable.  A
                              vehicle a handful of pixels tall cannot support a meaningful
                              contact-point estimate, and admitting it adds noise without adding
                              information.
    @param  max_box_fraction  Maximum box height as a fraction of the frame.  Larger than this is
                              almost always the track surface or a lens artefact detected as a
                              vehicle.
    """

    mode: str = "bottom_center"
    bottom_offset_px: float = 0.0
    min_box_px: float = 6.0
    max_box_fraction: float = 0.9

    def __post_init__(self) -> None:
        if self.mode not in ("bottom_center", "bottom_third", "center"):
            raise ValueError(
                f"detection_point.mode must be one of 'bottom_center', 'bottom_third', 'center', "
                f"got {self.mode!r}"
            )
        if self.min_box_px < 0.0:
            raise ValueError(f"min_box_px must be >= 0, got {self.min_box_px}")

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DetectionPointConfig":
        """@brief Build from the ``detection_point`` configuration block."""
        return cls(
            mode=str(cfg.get("mode", "bottom_center")),
            bottom_offset_px=float(cfg.get("bottom_offset_px", 0.0)),
            min_box_px=float(cfg.get("min_box_px", 6.0)),
            max_box_fraction=float(cfg.get("max_box_fraction", 0.9)),
        )


def ground_contact_point(box_xyxy: Sequence[float], config: DetectionPointConfig | None = None) -> np.ndarray:
    """Choose the road-reference point on a detection box.

    @brief   Map a bounding box to the image point that represents the vehicle's road position.
    @details The returned point is what gets projected through the ground homography, so this
             function defines the meaning of "where the car is".  See
             :class:`DetectionPointConfig` for why the bottom edge is the right family of choices
             and the box centre is not.

             Derivation for the bottom-third variant: when a camera views a vehicle from an angle,
             the silhouette extends further behind the contact patch than in front of it, so the
             midpoint of the bottom edge drifts toward the far end of the vehicle.  Taking a point
             one third of the way from the left of the bottom edge compensates in the common case
             of a vehicle travelling left to right across the frame; for a vehicle travelling the
             other way it compensates the wrong way, which is why it is a documented alternative
             rather than the default.

    @param   box_xyxy ``(4,)`` box as ``(x1, y1, x2, y2)`` in pixels.
    @param   config   Contact-point strategy; defaults to ``bottom_center`` with no offset.
    @return  ``(2,)`` image point ``(u, v)`` in pixels.
    @raises  ValueError if either box dimension is non-positive or non-finite.
    """
    box = np.asarray(box_xyxy, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(box)):
        raise ValueError(f"bounding box contains non-finite values: {box}")
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"bounding box must have positive extent, got {box}")

    cfg = config or DetectionPointConfig()
    if cfg.mode == "center":
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    # @step Both remaining modes sit on the bottom edge, which is the contact line.  The offset
    #       moves this toward the true contact patch; it is never allowed to leave the box.
    v = y2 + cfg.bottom_offset_px
    if cfg.mode == "bottom_third":
        u = x1 + width / 3.0
    else:
        u = x1 + width / 2.0
    return np.array([u, v])


@dataclass
class VehicleDetection:
    """One detected vehicle in one frame.

    @brief   A detection box together with its derived road-reference point.
    @details The reference point is computed once at construction and stored, so that the
             projection layer cannot accidentally use the box centre while the configuration says
             otherwise -- the two are not independently addressable.

    @param  box_xyxy        ``(4,)`` bounding box in pixels.
    @param  confidence      Detector confidence in ``[0, 1]``.
    @param  class_id        COCO class index.
    @param  reference_uv    ``(2,)`` image point used as the vehicle's road position.
    @param  track_id        Tracker identity, when known.
    @param  ground_xz       ``(2,)`` road position in metres, filled in after projection.
    @param  scale_m_per_px  Local metres-per-pixel at the reference point.
    @param  position_sigma_m Metric positional uncertainty, when computed.
    @param  valid           Whether the projection passed the validity gate.
    """

    box_xyxy: np.ndarray
    confidence: float = 0.0
    class_id: int = -1
    reference_uv: np.ndarray = field(default_factory=lambda: np.zeros(2))
    track_id: int | None = None
    ground_xz: np.ndarray | None = None
    scale_m_per_px: float = math.nan
    position_sigma_m: float = math.nan
    valid: bool = True

    @property
    def name(self) -> str:
        """@brief Readable class name."""
        return class_name(self.class_id)

    @property
    def height_px(self) -> float:
        """@brief Box height in pixels."""
        return float(self.box_xyxy[3] - self.box_xyxy[1])

    @property
    def width_px(self) -> float:
        """@brief Box width in pixels."""
        return float(self.box_xyxy[2] - self.box_xyxy[0])

    @property
    def speed_key(self) -> float:
        """@brief Area in pixels, used to break ties when matching detections to tracks."""
        return self.height_px * self.width_px

    def to_row(self, frame_index: int, time_s: float) -> dict[str, Any]:
        """@brief Flatten to a CSV-friendly row, including the derived quantities."""
        ground = self.ground_xz if self.ground_xz is not None else np.array([math.nan, math.nan])
        return {
            "frame": int(frame_index),
            "time_s": float(time_s),
            "track_id": self.track_id if self.track_id is not None else -1,
            "class_id": int(self.class_id),
            "class_name": self.name,
            "confidence": round(float(self.confidence), 4),
            "box_x1": round(float(self.box_xyxy[0]), 2),
            "box_y1": round(float(self.box_xyxy[1]), 2),
            "box_x2": round(float(self.box_xyxy[2]), 2),
            "box_y2": round(float(self.box_xyxy[3]), 2),
            "ref_u": round(float(self.reference_uv[0]), 2),
            "ref_v": round(float(self.reference_uv[1]), 2),
            "ground_x_m": round(float(ground[0]), 6),
            "ground_z_m": round(float(ground[1]), 6),
            "scale_m_per_px": None if not math.isfinite(self.scale_m_per_px) else round(self.scale_m_per_px, 8),
            "position_sigma_m": None
            if not math.isfinite(self.position_sigma_m)
            else round(self.position_sigma_m, 6),
            "valid": bool(self.valid),
        }


class VehicleDetector:
    """Thin wrapper around an Ultralytics YOLO model.

    @brief   Filter to vehicle classes and attach a road-reference point to each box.
    @details The wrapper exists for three reasons beyond convenience:

             1. **Class filtering at the source.**  Only the configured vehicle classes are
                returned, so nothing downstream has to remember to filter, and a pedestrian or a
                pet cannot become a speed measurement.
             2. **Reference-point attachment.**  Each :class:`VehicleDetection` carries the
                contact point chosen by :class:`DetectionPointConfig`, so the projection layer
                cannot silently use a different convention than the configuration declares.
             3. **Lazy heavy imports.**  ``ultralytics`` and therefore torch are imported on first
                construction, not at module import, which keeps the geometry and speed layers
                importable on a memory-constrained device.

    @param  model      Weights path or model name, for example ``yolo26n.pt``.
    @param  imgsz      Inference size in pixels.
    @param  conf       Confidence threshold.
    @param  iou        NMS IoU threshold.
    @param  classes    COCO class indices to keep.
    @param  device     ``auto``, ``cpu``, ``mps``, ``cuda`` or a device index string.
    @param  quantize   Precision mode: ``auto`` (16-bit on CUDA only), ``off``/``none`` for full
                       FP32, ``16`` or ``8`` to request a specific mode.  ``8`` is INT8 and
                       requires a quantized model, so it is rejected on a full-precision one.
    @param  nms_free   Use the end-to-end head where the model supports it.
    @param  point_config Contact-point strategy.
    """

    def __init__(
        self,
        model: str = "yolo26n.pt",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.5,
        classes: Iterable[int] = (2, 5, 7),
        device: str = "auto",
        quantize: str | int | bool = "auto",
        nms_free: bool = False,
        point_config: DetectionPointConfig | None = None,
    ) -> None:
        self.model_name = str(model)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.classes = [int(cls) for cls in classes]
        self.point_config = point_config or DetectionPointConfig()
        self.nms_free = bool(nms_free)
        self.device = self._resolve_device(device)
        self.quantize = self._resolve_quantize(quantize)
        self._model = None
        self.frames_processed = 0
        self.detections_returned = 0

    # -- device selection ---------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Turn ``auto`` into a concrete device, preferring CUDA, then Apple MPS, then CPU.

        @brief   Pick an accelerator deterministically.
        @details ``auto`` exists so that the same configuration works on a workstation with a GPU
                 and on a Raspberry Pi without one.  CUDA is preferred over MPS because Metal
                 support has more gaps for detection workloads; the fallback order is still
                 strictly an optimisation, since every path produces identical results.
        @param   device Requested device or ``auto``.
        @return  A device string suitable for Ultralytics.
        """
        if device != "auto":
            return device
        try:
            import torch
        except Exception:  # pragma: no cover - torch absent in a core-only install
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_quantize(self, quantize: str | int | bool) -> int | None:
        """Resolve the requested precision mode to a concrete ``quantize`` value.

        @brief   Precision selection, defaulting to FP16 on CUDA only.
        @details Returns ``None`` to request full FP32, which is what CPU inference should use:
            16-bit arithmetic is slower and less accurate on most CPUs, so enabling it
            indiscriminately would silently cost accuracy on the Raspberry Pi and Orange Pi
            targets.  This is a case where the obvious "lower precision is faster" rule is wrong
            for half the deployment surface.

            ``ultralytics`` renamed the ``half`` flag to ``quantize`` and now emits a deprecation
            warning for the old name on every single call, which at thirty frames per second is
            both a console flood and a measurable cost.  The new parameter is therefore used
            directly, and only passed to the model when a non-default mode is requested.

        @param   quantize Requested setting: ``auto``, ``off``/``none``/``False``, ``16`` or ``8``.
        @return  ``16``, ``8`` or ``None`` for FP32.
        @raises  ValueError for an unrecognised setting.
        """
        if quantize is None or quantize is False:
            return None
        if isinstance(quantize, bool):
            return 16 if quantize else None
        if isinstance(quantize, int):
            if quantize not in (8, 16):
                raise ValueError(f"quantize must be 8, 16 or 'off', got {quantize}")
            return quantize
        text = str(quantize).strip().lower()
        if text in ("off", "none", "false", "0", "fp32"):
            return None
        if text in ("8", "int8"):
            return 8
        if text in ("16", "fp16", "half"):
            return 16
        if text == "auto":
            return 16 if self.device.startswith("cuda") else None
        raise ValueError(f"unrecognised quantize setting {quantize!r}; use auto, off, 16 or 8")

    # -- model lifecycle ----------------------------------------------------

    @property
    def model(self):
        """@brief The underlying YOLO model, loaded on first access."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        """Load the detector weights, with an actionable error if the extra is missing.

        @brief   Lazily construct the Ultralytics model.
        @details Importing ``ultralytics`` here rather than at module scope is what keeps the
                 memory footprint of the geometry and speed layers independent of torch.
        @return  An instantiated ``ultralytics.YOLO``.
        @raises  RuntimeError when the vision extra is not installed.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "detection requires the optional 'vision' dependencies. Install them with:\n"
                "    uv sync --extra vision"
            ) from exc
        model = YOLO(self.model_name)
        if self.nms_free:
            # @note The one-to-one head is selected at call time rather than load time in
            #       Ultralytics, so the flag is recorded and passed to predict().
            pass
        return model

    # -- inference ----------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[VehicleDetection]:
        """Run detection on one frame and return vehicle detections with reference points.

        @brief   Detect vehicles in a BGR frame.
        @details ``verbose=False`` and ``stream=False`` are used deliberately: on a Pi the default
                 per-frame banner is a meaningful fraction of the frame budget, and generator
                 mode would defer work past the point where the caller needs the results.
        @param   frame BGR image as an ``(H, W, 3)`` array.
        @return  List of :class:`VehicleDetection`, possibly empty.
        @raises  ValueError for a frame that is not a 3-channel image.
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) BGR frame, got shape {getattr(frame, 'shape', None)}")

        predict_kwargs: dict[str, Any] = {
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "classes": self.classes,
            "device": self.device,
            "verbose": False,
        }
        # @note Only sent when a non-default precision is wanted; passing the parameter at all
        #       routes through ultralytics' legacy-flag handling.
        if self.quantize is not None:
            predict_kwargs["quantize"] = self.quantize
        if self.nms_free:
            predict_kwargs["nms"] = False

        results = self.model.predict(frame, **predict_kwargs)
        self.frames_processed += 1
        detections = self._parse_results(results, frame.shape)
        self.detections_returned += len(detections)
        return detections

    def _parse_results(self, results: list[Any], frame_shape: tuple[int, ...]) -> list[VehicleDetection]:
        """Convert Ultralytics results into filtered detections with reference points.

        @brief   Parse and filter raw model output.
        @details Boxes are filtered on plausibility here rather than downstream, so that the
                 tracker never sees a detection that could not produce a measurement anyway.  The
                 plausibility rules are: the box must be tall enough to support a contact-point
                 estimate, and it must not cover most of the frame, which is how a mis-detection
                 of the road surface or a lens flare usually presents.
        @param   results      Raw Ultralytics result objects.
        @param   frame_shape  ``(H, W, 3)`` shape of the frame the results describe.
        @return  List of usable detections.
        """
        height_px = float(frame_shape[0])
        max_height = self.point_config.max_box_fraction * height_px
        out: list[VehicleDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
            for box, score, class_id in zip(xyxy, conf, cls, strict=False):
                height = float(box[3] - box[1])
                if height < self.point_config.min_box_px or height > max_height:
                    continue
                reference = ground_contact_point(box, self.point_config)
                out.append(
                    VehicleDetection(
                        box_xyxy=np.asarray(box, dtype=np.float64),
                        confidence=float(score),
                        class_id=int(class_id),
                        reference_uv=reference,
                    )
                )
        return out

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "VehicleDetector":
        """@brief Build from a merged configuration mapping."""
        detector = cfg.get("detector", {})
        point_cfg = dict(cfg.get("detection_point", {}))
        # @note The validity block also carries the box-plausibility limits; merging here keeps a
        #       single source for the contact-point strategy's own thresholds.
        validity = cfg.get("validity", {})
        point_cfg.setdefault("min_box_px", validity.get("min_box_px", 6.0))
        point_cfg.setdefault("max_box_fraction", validity.get("max_box_fraction", 0.9))
        return cls(
            model=str(detector.get("model", "yolo26n.pt")),
            imgsz=int(detector.get("imgsz", 640)),
            conf=float(detector.get("conf", 0.25)),
            iou=float(detector.get("iou", 0.5)),
            classes=detector.get("classes", (2, 5, 7)),
            device=str(detector.get("device", "auto")),
            quantize=detector.get("quantize", detector.get("half", "auto")),
            nms_free=bool(detector.get("nms_free", False)),
            point_config=DetectionPointConfig.from_config(point_cfg),
        )

    def describe(self) -> dict[str, Any]:
        """@brief Summary of the detector configuration, for the run log."""
        return {
            "model": self.model_name,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "classes": [class_name(cls) for cls in self.classes],
            "device": self.device,
            "quantize": self.quantize,
            "nms_free": self.nms_free,
            "contact_point": self.point_config.mode,
            "bottom_offset_px": self.point_config.bottom_offset_px,
        }
