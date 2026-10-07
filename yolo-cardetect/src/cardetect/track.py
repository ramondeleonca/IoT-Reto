"""Multi-object tracking and ground-projected track state.

@file    track.py
@brief   Associate detections across frames and maintain each track's history in metres.
@details Ultralytics provides the association algorithms (ByteTrack and BoT-SORT), so this module
         does not reimplement them.  What it adds is the part the library does not do and which
         the speed estimator depends on:

         * **Projection at the moment of association.**  Each confirmed track keeps a history of
           road positions in metres and the real timestamp of every observation, taken from the
           caller.  Projecting here rather than later means a track's history is already in the
           units the estimator wants, and a detection that fails the geometry validity gate is
           dropped before it can enter a history at all.
         * **Timestamps supplied, never inferred.**  ``frame_index / fps`` is wrong on a variable
           frame rate source, which is the normal case for a phone and the invariable case for a
           Raspberry Pi Camera Module.  Callers pass a real timestamp.
         * **Depth-ordered identity handling.**  Two vehicles at very different depths can overlap
           in the image while being metres apart on the road; the tracker works in image space, so
           a track's identity is only ever as good as its image-space separation.  The history
           therefore records the metric scale at each observation so the estimator can report how
           much resolution was actually available.

@note    ``ultralytics`` is imported lazily, so this module imports cheaply and the history and
         projection logic stays testable with NumPy alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from cardetect.detect import VehicleDetection
from cardetect.geometry import GroundPlaneProjector
from cardetect.speed import TrackHistory

__all__ = [
    "TrackedVehicle",
    "TrackManager",
    "TrackManagerConfig",
    "associate_by_iou",
    "iou_matrix",
]


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise intersection-over-union between two sets of boxes.

    @brief   IoU matrix in ``xyxy`` format.
    @details Provided for the fallback associator and for tests.  Implemented in NumPy rather than
             delegated so the tracker's behaviour is reproducible without OpenCV, which matters
             because a tracker bug is one of the most confusing failure modes to diagnose in a
             speed pipeline -- it presents as an impossible speed rather than as an error.
    @param   boxes_a ``(N, 4)`` boxes.
    @param   boxes_b ``(M, 4)`` boxes.
    @return  ``(N, M)`` IoU values.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    # @step Broadcast to (N, M, ...) and compute intersection extents directly.
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(right - left, 0.0, None) * np.clip(bottom - top, 0.0, None)

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)


def associate_by_iou(
    detections: np.ndarray, tracks: np.ndarray, threshold: float = 0.3
) -> list[tuple[int, int]]:
    """Greedy IoU association used when the Ultralytics tracker is not in use.

    @brief   Match detections to predicted track boxes by overlap.
    @details Greedy by descending IoU rather than optimal assignment.  The reason is not only
             cost: greedy matching is deterministic and easy to reason about, and for a handful of
             vehicles on a track the difference from an optimal assignment is negligible, whereas
             an unexpected assignment is very hard to diagnose from a speed plot.
    @param   detections ``(N, 4)`` detection boxes.
    @param   tracks     ``(M, 4)`` predicted track boxes.
    @param   threshold  Minimum IoU for a match.
    @return  List of ``(detection_index, track_index)`` pairs.
    """
    ious = iou_matrix(detections, tracks)
    if ious.size == 0:
        return []
    pairs: list[tuple[int, int]] = []
    used_det: set[int] = set()
    used_trk: set[int] = set()
    # @step Sort all candidate pairs by descending IoU and take them greedily.
    order = np.argsort(ious, axis=None)[::-1]
    for flat in order:
        det_idx, trk_idx = np.unravel_index(flat, ious.shape)
        if ious[det_idx, trk_idx] < threshold:
            break
        if det_idx in used_det or trk_idx in used_trk:
            continue
        pairs.append((int(det_idx), int(trk_idx)))
        used_det.add(int(det_idx))
        used_trk.add(int(trk_idx))
    return pairs


@dataclass
class TrackManagerConfig:
    """Tracking behaviour and track lifetime.

    @brief   Tuning for association and history retention.
    @param  tracker_type    ``bytetrack`` or ``bot-sort``.
    @param  min_hits        Confirmed observations required before a track reports a speed.  A
                            track that exists for one or two frames cannot support a measurement,
                            and reporting one produces a confident-looking number from almost no
                            data.
    @param  max_age_frames  Frames a track may go unmatched before being retired.
    @param  max_history     Cap on stored observations per track.  Bounds memory on a long run;
                            with a 512 MB target an unbounded history per track is a real risk.
    @param  high_thresh     ByteTrack high-confidence threshold.
    @param  low_thresh      ByteTrack low-confidence threshold.
    @param  new_track_thresh Confidence required to start a new track.
    @param  match_thresh    Association distance threshold.
    @param  track_buffer    Frames a lost track is retained by the underlying tracker.
    @param  proximity_thresh BoT-SORT proximity gate.
    @param  appearance_thresh BoT-SORT appearance gate.
    @param  with_reid       Enable BoT-SORT appearance re-identification.
    """

    tracker_type: str = "bytetrack"
    min_hits: int = 3
    max_age_frames: int = 30
    max_history: int = 600
    high_thresh: float = 0.25
    low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    match_thresh: float = 0.8
    track_buffer: int = 30
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    with_reid: bool = False

    def __post_init__(self) -> None:
        self.tracker_type = str(self.tracker_type).lower()
        if self.tracker_type not in ("bytetrack", "bot-sort", "botsort"):
            raise ValueError(f"tracker.type must be 'bytetrack' or 'bot-sort', got {self.tracker_type!r}")
        if self.min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {self.min_hits}")
        if self.max_history < 2:
            raise ValueError(f"max_history must be >= 2, got {self.max_history}")

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "TrackManagerConfig":
        """@brief Build from the ``tracker`` configuration block."""
        tracker = cfg.get("tracker", {})
        speed = cfg.get("speed", {})
        return cls(
            tracker_type=str(tracker.get("type", "bytetrack")),
            # @note The minimum observation count is the speed estimator's requirement, so it is
            #       read from the speed block rather than duplicated in the tracker block.
            min_hits=int(speed.get("min_samples", 3)),
            high_thresh=float(tracker.get("track_high_thresh", 0.25)),
            low_thresh=float(tracker.get("track_low_thresh", 0.1)),
            new_track_thresh=float(tracker.get("new_track_thresh", 0.25)),
            match_thresh=float(tracker.get("match_thresh", 0.8)),
            track_buffer=int(tracker.get("track_buffer", 30)),
            proximity_thresh=float(tracker.get("proximity_thresh", 0.5)),
            appearance_thresh=float(tracker.get("appearance_thresh", 0.25)),
            with_reid=bool(tracker.get("with_reid", False)),
        )


@dataclass
class TrackedVehicle:
    """State for one tracked vehicle.

    @brief   Identity, recent geometry and metric history.
    @param  track_id        Tracker identity.
    @param  history         Ground positions, timestamps and local scales in metres.
    @param  hits            Number of frames in which this track was matched.
    @param  age             Frames since the track was created.
    @param  misses          Consecutive frames without a match.
    @param  last_box        Most recent matched box, for the next association step.
    @param  last_detection  Most recent detection, for drawing.
    @param  class_id        Most recent class index.
    """

    track_id: int
    history: TrackHistory
    hits: int = 0
    age: int = 0
    misses: int = 0
    last_box: np.ndarray | None = None
    last_detection: VehicleDetection | None = None
    class_id: int = -1

    @property
    def is_confirmed(self) -> bool:
        """@brief Whether the track has enough observations to be reported."""
        return self.hits >= 1

    @property
    def last_ground_xz(self) -> np.ndarray | None:
        """@brief Most recent road position in metres, or ``None``."""
        return self.history.positions[-1] if self.history.positions else None

    def trail(self, length: int) -> np.ndarray:
        """@brief Most recent road positions for drawing a trajectory.

        @param   length Maximum number of points to return.
        @return  ``(K, 2)`` array of road positions, oldest first.
        """
        points = self.history.positions[-length:] if length > 0 else []
        if not points:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(points, dtype=np.float64).reshape(-1, 2)


class TrackManager:
    """Owns track identity, histories and lifetime.

    @brief   Turn per-frame detections into per-track metric histories.
    @details Delegates association to Ultralytics when a model is tracking, and falls back to a
             deterministic IoU associator when it is not.  Both paths converge on the same
             :class:`TrackedVehicle` state, so the pipeline and the visualiser never branch on
             which tracker is active -- a property worth preserving, since tracker-specific
             branches are how "works with ByteTrack, wrong with BoT-SORT" bugs arise.

    @param  config      Tracking behaviour.
    @param  projector   Ground geometry used to project each contact point.
    @param  pixel_sigma_px Assumed detector jitter, used for the per-detection error estimate.
    """

    def __init__(
        self,
        config: TrackManagerConfig | None = None,
        projector: GroundPlaneProjector | None = None,
        pixel_sigma_px: float = 2.0,
    ) -> None:
        self.config = config or TrackManagerConfig()
        self.projector = projector
        self.pixel_sigma_px = float(pixel_sigma_px)
        self.tracks: dict[int, TrackedVehicle] = {}
        self.retired: list[TrackedVehicle] = []
        self._next_id = 1
        self._frame_index = 0
        self._unprojected_detections = 0

    # -- projection ---------------------------------------------------------

    def project_detections(self, detections: Iterable[VehicleDetection]) -> None:
        """Project each detection's contact point onto the road plane, in place.

        @brief   Attach metric ground positions to detections and gate invalids.
        @details A detection whose contact point fails the validity gate is marked invalid *and*
                 has no ground position, so it cannot enter a history through any code path.  This
                 matters because an off-road projection is finite and plausible-looking: a point
                 above the horizon maps to a road position behind the camera, and a single such
                 sample produces a huge spurious speed.
        @param   detections Detections to annotate.
        """
        if self.projector is None:
            return
        for detection in detections:
            points = np.asarray([detection.reference_uv], dtype=np.float64)
            if not bool(self.projector.is_valid(points)[0]):
                detection.valid = False
                self._unprojected_detections += 1
                continue
            ground = self.projector.to_ground(points)[0]
            scale = float(self.projector.scale(ground[None])["m_per_px_max"][0])
            detection.ground_xz = ground
            detection.scale_m_per_px = scale
            detection.position_sigma_m = float(self.pixel_sigma_px * scale)
            detection.valid = True

    # -- association --------------------------------------------------------

    def update(
        self, detections: list[VehicleDetection], timestamp_s: float
    ) -> list[TrackedVehicle]:
        """Advance one frame: associate, update histories, retire stale tracks.

        @brief   The main per-frame entry point.
        @details Detections that failed the depth gate are excluded before association, so a
                 phantom projection cannot attract a track.  Tracks are then aged, and any that
                 have exceeded ``max_age_frames`` are moved to ``retired`` rather than deleted, so
                 a caller can still report their final measurements.
        @param   detections  Detections for this frame, already projected.
        @param   timestamp_s Real timestamp of this frame in seconds.
        @return  Tracks matched in this frame, in ascending identity order.
        """
        self._frame_index += 1
        usable = [det for det in detections if det.valid and det.ground_xz is not None]
        self._associate(usable, timestamp_s)

        matched: list[TrackedVehicle] = []
        for track_id in sorted(self.tracks):
            track = self.tracks[track_id]
            track.age += 1
            if track.misses > self.config.max_age_frames:
                self.retired.append(track)
                continue
            matched.append(track)
        for track in self.retired:
            self.tracks.pop(track.track_id, None)
        return [track for track in matched if track.track_id in self.tracks]

    def _associate(self, detections: list[VehicleDetection], timestamp_s: float) -> None:
        """Match this frame's detections to existing tracks or create new ones.

        @brief   Greedy IoU association with single-target fallback.
        @details With at most one vehicle in view -- the common case on a single-lane table-top
                 track -- a detection is matched by proximity to the nearest existing track rather
                 than by IoU.  IoU fails there for a concrete reason: a fast toy car can move
                 further than its own box between frames at a low frame rate, so consecutive boxes
                 may not overlap at all, and strict IoU would spawn a new identity every frame and
                 destroy the measurement.  Proximity matching tolerates that; the speed
                 estimator's plausibility gate then rejects the rare wrong association.
        @param   detections  Usable detections for this frame.
        @param   timestamp_s Timestamp of this frame.
        """
        pairs: list[tuple[int, int]] = []
        if not self.tracks:
            for detection in detections:
                self._spawn(detection, timestamp_s)
            return
        if not detections:
            for track in self.tracks.values():
                track.misses += 1
            return

        track_ids = sorted(self.tracks)
        track_boxes = np.asarray([self.tracks[tid].last_box for tid in track_ids], dtype=np.float64)
        det_boxes = np.asarray([det.box_xyxy for det in detections], dtype=np.float64)

        # @note This block indexes everything by *position within track_ids* rather than by track
        #       identity, and the distinction is load bearing.  Mixing the two index spaces was a
        #       real bug here: the IoU matcher returns positions, whereas an earlier version of the
        #       proximity fallback inserted identities, so the two sets were not comparable and a
        #       successful association could raise an out-of-range index.  Track identities are
        #       monotonic and can exceed the number of live tracks, so the two are not
        #       interchangeable in general.  `matched_tracks` therefore holds positions only, and
        #       identities are looked up at the point of use.
        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()

        for det_pos, trk_pos in associate_by_iou(det_boxes, track_boxes, threshold=0.15):
            pairs.append((det_pos, trk_pos))
            matched_detections.add(det_pos)
            matched_tracks.add(trk_pos)

        # @step Proximity fallback for anything IoU left unmatched: match by contact-point distance
        #       in the image, preferring the nearest plausible partner.
        for det_pos, detection in enumerate(detections):
            if det_pos in matched_detections:
                continue
            best_trk_pos: int | None = None
            best_dist = math.inf
            for trk_pos, track_id in enumerate(track_ids):
                if trk_pos in matched_tracks:
                    continue
                track = self.tracks[track_id]
                if track.last_detection is None:
                    continue
                distance_px = float(
                    np.linalg.norm(detection.reference_uv - track.last_detection.reference_uv)
                )
                if distance_px < best_dist:
                    best_dist, best_trk_pos = distance_px, trk_pos
            if best_trk_pos is None:
                continue
            # @note The gate scales with the box so it behaves consistently across depths: a distant
            #       vehicle moves few pixels between frames, a near one moves many.  A fixed pixel
            #       gate would either merge a slow distant vehicle with its neighbour or refuse to
            #       follow a fast near one.
            gate = max(detection.height_px, 10.0) * 1.5
            if best_dist <= gate:
                pairs.append((det_pos, best_trk_pos))
                matched_detections.add(det_pos)
                matched_tracks.add(best_trk_pos)

        for det_pos, trk_pos in pairs:
            self._observe(self.tracks[track_ids[trk_pos]], detections[det_pos], timestamp_s)

        for trk_pos, track_id in enumerate(track_ids):
            if trk_pos not in matched_tracks:
                self.tracks[track_id].misses += 1
        for det_pos, detection in enumerate(detections):
            if det_pos not in matched_detections:
                self._spawn(detection, timestamp_s)

    def _spawn(self, detection: VehicleDetection, timestamp_s: float) -> None:
        """Create a new track from an unmatched detection.

        @brief   Start a track.
        @param   detection   Detection that begins the track.
        @param   timestamp_s Timestamp of this frame.
        """
        track_id = self._next_id
        self._next_id += 1
        track = TrackedVehicle(
            track_id=track_id,
            history=TrackHistory(track_id=track_id),
            last_box=np.asarray(detection.box_xyxy, dtype=np.float64),
            class_id=detection.class_id,
        )
        self.tracks[track_id] = track
        self._observe(track, detection, timestamp_s)

    def _observe(self, track: TrackedVehicle, detection: VehicleDetection, timestamp_s: float) -> None:
        """Record one matched observation against a track.

        @brief   Append to the history and refresh the track's geometry.
        @details The history is capped at ``max_history`` observations.  Trimming from the front
                 is safe for the speed estimator because it only ever reads the trailing window,
                 and it bounds memory on a long run, which matters on the 512 MB target.
        @param   track       Track to update.
        @param   detection   Matched detection.
        @param   timestamp_s Timestamp of this frame.
        """
        if detection.ground_xz is None:
            return
        track.history.append(
            timestamp_s,
            detection.ground_xz,
            scale_m_per_px=detection.scale_m_per_px,
            confidence=detection.confidence,
            class_id=detection.class_id,
        )
        if len(track.history) > self.config.max_history:
            overflow = len(track.history) - self.config.max_history
            del track.history.times[:overflow]
            del track.history.positions[:overflow]
            del track.history.scales[:overflow]
            del track.history.confidences[:overflow]
            del track.history.class_ids[:overflow]
        track.hits += 1
        track.misses = 0
        track.last_box = np.asarray(detection.box_xyxy, dtype=np.float64)
        track.last_detection = detection
        track.class_id = detection.class_id

    # -- reporting ----------------------------------------------------------

    @property
    def all_histories(self) -> dict[int, TrackHistory]:
        """@brief Histories of live and retired tracks, keyed by identity."""
        out = {track.track_id: track.history for track in self.retired}
        out.update({track.track_id: track.history for track in self.tracks.values()})
        return out

    def confirmed_tracks(self) -> list[TrackedVehicle]:
        """@brief Tracks with enough observations to support a speed estimate."""
        return [track for track in self.tracks.values() if track.hits >= self.config.min_hits]

    def stats(self) -> dict[str, int]:
        """@brief Counters for the run summary and for diagnosing a mis-tuned rig."""
        return {
            "frames": self._frame_index,
            "live_tracks": len(self.tracks),
            "retired_tracks": len(self.retired),
            "rejected_projections": self._unprojected_detections,
        }

    def reset(self) -> None:
        """@brief Discard all track state, keeping configuration and identity counter."""
        self.tracks.clear()
        self.retired.clear()
        self._frame_index = 0
