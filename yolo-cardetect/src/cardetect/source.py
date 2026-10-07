"""Video capture with trustworthy timestamps.

@file    source.py
@brief   Frame sources for webcam, video file and Raspberry Pi camera, with honest timing.
@details Timestamps are the quiet foundation of every speed measurement in this project, and the
         obvious approach is wrong on the hardware this is aimed at.

         ``CAP_PROP_FPS`` is an *encoding* property.  On a video file it is a header field, so
         ``frame_index / fps`` is usually fine.  On a live capture device it is a request, not a
         promise: a Raspberry Pi Camera Module reports 30 while delivering anything between 4 and
         30 depending on load, and a USB webcam similarly delivers whatever it manages.  Using the
         nominal figure there produces speeds that are wrong by an unknown, time-varying factor --
         the worst kind of error, because it is invisible.

         The rule implemented here is therefore:

         * a **file** source uses ``frame_index / fps`` with the verified container rate, because
           playback is deterministic and this is exact;
         * a **live** source uses a monotonic clock sampled at the moment each frame is grabbed,
           because that is the only statement about elapsed time that is actually true.

         One further detail specific to cameras: the first frames of a live capture are often
         delivered before exposure converges, and ``read()`` can return stale frames from the
         driver's internal queue.  The queue is therefore drained, and the clock is sampled after
         the grab succeeds rather than before.

@note    ``cv2`` is imported lazily so the geometry, speed and config layers stay importable
         without OpenCV -- which is what allows the test suite to run headless with NumPy alone.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

__all__ = ["FrameSource", "Frame", "SourceError", "resolve_source_uri", "probe_source"]


class SourceError(RuntimeError):
    """Raised when a capture source cannot be opened or read."""


@dataclass
class Frame:
    """One captured frame with its timestamp.

    @brief   Image plus the time it was captured.
    @param  image       BGR ``(H, W, 3)`` array.
    @param  index       0-based frame counter within this run.
    @param  timestamp_s Monotonic timestamp in seconds, relative to the source's first frame.
    @param  source_time_s Optional timestamp reported by the source itself, when available.
    """

    image: np.ndarray
    index: int
    timestamp_s: float
    source_time_s: float | None = None


def resolve_source_uri(uri: str | int) -> str | int:
    """Normalise a user-supplied source specification into something OpenCV accepts.

    @brief   Accept a camera index, a file path, a stream URL or a glob.
    @details Supports the forms an operator actually types:

             ``0``, ``1``      a camera device index
             ``clip.mp4``      a file path, resolved relative to the working directory
             ``rtsp://...``    a network stream
             one path only     a glob is deliberately NOT expanded, because silently processing
                               twelve files when one was intended hides mistakes; the CLI takes a
                               directory for batch work instead

    @param   uri Source specification.
    @return  An ``int`` device index or a ``str`` OpenCV can open.
    @raises  SourceError when a file path does not exist and is not a URL.
    """
    if isinstance(uri, int):
        return uri
    text = str(uri).strip()
    if text.isdigit():
        return int(text)
    lowered = text.lower()
    if lowered.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")):
        return text
    path = Path(text).expanduser()
    if not path.exists():
        raise SourceError(f"source not found: {path}")
    return str(path)


class FrameSource:
    """Iterable, resumable frame source over OpenCV's capture API.

    @brief   Context-managed video input with honest timestamps.
    @details Provides two capture regimes:

             ``stream``  yields every frame as it arrives.
             ``burst``   watches cheaply for motion and then captures a short high-frame-rate clip
                         around each event.  This exists for the Raspberry Pi Zero 2 W, which
                         cannot run a detector continuously: since speed is a per-track quantity,
                         only the frames containing a vehicle carry information, and those are
                         exactly the frames a burst captures.  It is not a degradation, it is the
                         correct strategy for the hardware.

    @param  uri             Camera index, file path or stream URL.
    @param  width           Requested capture width, or ``None`` for the source default.
    @param  height          Requested capture height, or ``None``.
    @param  fps             Requested capture rate for live sources, or ``None``.
    @param  mode            ``stream`` or ``burst``.
    @param  max_frames      Stop after this many frames, for tests and bounded runs.
    @param  warmup_frames   Frames to discard at open time so exposure has settled.
    @param  burst_seconds   Length of a burst clip.
    @param  burst_preroll_s How much of the pre-event buffer to keep.
    @param  motion_threshold Fraction of changed pixels that arms a burst.
    """

    def __init__(
        self,
        uri: str | int = 0,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        mode: str = "stream",
        max_frames: int | None = None,
        warmup_frames: int = 5,
        burst_seconds: float = 2.0,
        burst_preroll_s: float = 0.3,
        motion_threshold: float = 0.02,
    ) -> None:
        if mode not in ("stream", "burst"):
            raise SourceError(f"source mode must be 'stream' or 'burst', got {mode!r}")
        self.resolved = resolve_source_uri(uri)
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.mode = mode
        self.max_frames = max_frames
        self.warmup_frames = max(0, int(warmup_frames))
        self.burst_seconds = float(burst_seconds)
        self.burst_preroll_s = float(burst_preroll_s)
        self.motion_threshold = float(motion_threshold)

        self._capture = None
        self._is_live = isinstance(self.resolved, int) or str(self.resolved).lower().startswith(
            ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")
        )
        self.reported_fps: float | None = None
        self.frame_count: int | None = None
        self._index = 0
        self._start_time: float | None = None
        self._time_offset = 0.0
        self._file_duration_limit: float | None = None

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> "FrameSource":
        """Open the capture device.

        @brief   Prepare the source and read its reported properties.
        @details The frame queue is deliberately left as small as the backend allows, because a
                 deep queue means ``read()`` returns a frame captured long ago while the detector
                 works on the previous one.  That inflates apparent latency and, worse, makes the
                 sampled clock disagree with the frame's true capture time.
        @return  ``self``, so the source can be used as a context manager or chained.
        @raises  SourceError when the device or file cannot be opened.
        """
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SourceError(
                "video capture requires OpenCV. Install it with:\n    uv sync --extra vision"
            ) from exc

        capture = cv2.VideoCapture(self.resolved)
        if not capture.isOpened():
            raise SourceError(
                f"could not open source {self.resolved!r}. For a camera index, check that the "
                f"device exists and is not already in use; on macOS grant camera permission to "
                f"the terminal."
            )

        if self.requested_width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.requested_width))
        if self.requested_height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.requested_height))
        if self.requested_fps and self._is_live:
            capture.set(cv2.CAP_PROP_FPS, float(self.requested_fps))
        if self._is_live:
            # @note A one-frame buffer is the smallest most backends accept; some ignore it
            #       entirely, which is exactly why the timestamp is sampled after the grab.
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:  # pragma: no cover - backend dependent
                pass

        self._capture = capture
        reported = capture.get(cv2.CAP_PROP_FPS)
        self.reported_fps = float(reported) if reported and reported > 0.0 and math.isfinite(reported) else None
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_count = int(count) if count and count > 0 else None
        if not self._is_live and self.reported_fps:
            # @note For a file, playback is deterministic, so index/fps is exact and is preferred
            #       over wall-clock, which would drift with decoder stalls.
            self._file_duration_limit = None
        for _ in range(self.warmup_frames if self._is_live else 0):
            capture.read()
        return self

    def close(self) -> None:
        """@brief Release the capture device, tolerating repeated calls."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:  # pragma: no cover - backend dependent
                pass
            self._capture = None

    def __enter__(self) -> "FrameSource":
        return self.open()

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    # -- iteration ----------------------------------------------------------

    def __iter__(self) -> Iterator[Frame]:
        """@brief Iterate frames, dispatching on the configured capture mode."""
        if self._capture is None:
            self.open()
        if self.mode == "burst":
            yield from self._iter_burst()
        else:
            yield from self._iter_stream()

    def _iter_stream(self) -> Iterator[Frame]:
        """Yield every frame with its timestamp.

        @brief   Straightforward sequential capture.
        @details Two timing regimes as described in the module docstring: files use the exact
                 container rate, live sources use a monotonic clock sampled after each successful
                 grab.  The distinction is the reason this module exists.
        """
        import cv2

        assert self._capture is not None
        if self._start_time is None:
            self._start_time = time.monotonic()
        while True:
            if self.max_frames is not None and self._index >= self.max_frames:
                break
            ok, image = self._capture.read()
            if not ok or image is None:
                break
            timestamp = self._timestamp_for(self._index)
            yield Frame(image=image, index=self._index, timestamp_s=timestamp)
            self._index += 1

    def _timestamp_for(self, index: int) -> float:
        """Compute the timestamp of a frame according to the source type.

        @brief   Exact playback time for files, sampled wall-clock for live sources.
        @details Kept as a single method so the two regimes are impossible to confuse, and so the
                 file path can be unit tested without a camera.
        @param   index Frame index within this run.
        @return  Timestamp in seconds.
        """
        if not self._is_live:
            fps = self.reported_fps or self.requested_fps or 30.0
            return float(index) / float(fps)
        # @step Live: sample the monotonic clock now, which is the moment the frame was grabbed.
        assert self._start_time is not None
        return time.monotonic() - self._start_time + self._time_offset

    def _iter_burst(self) -> Iterator[Frame]:
        """Capture short clips around motion events, for devices that cannot run continuously.

        @brief   Motion-triggered burst capture.
        @details The idle loop is intentionally as cheap as possible: frames are only scaled down
                 and differenced, with no detector involved, so an idle Raspberry Pi costs a few
                 per cent of one core.  When the fraction of changed pixels exceeds the threshold,
                 the pre-roll already held in the ring buffer plus the following
                 ``burst_seconds`` are emitted as a normal frame sequence.

                 Speed is a per-track quantity, so nothing is lost by ignoring the idle frames --
                 they contain no vehicle, and therefore no information.
        @return  Iterator over frames belonging to burst events.
        """
        import cv2

        assert self._capture is not None
        fps = self.reported_fps or self.requested_fps or 30.0
        preroll = max(1, int(self.burst_preroll_s * fps))
        burst_len = max(2, int(self.burst_seconds * fps))
        ring: list[tuple[int, np.ndarray, float]] = []
        previous: np.ndarray | None = None
        emitting = 0
        emitted = 0
        next_index = 0

        while True:
            if self.max_frames is not None and self._index >= self.max_frames:
                break
            ok, image = self._capture.read()
            if not ok or image is None:
                break
            timestamp = self._timestamp_for(self._index)
            index = next_index
            next_index += 1

            if emitting > 0:
                emitting -= 1
                yield Frame(image=image, index=index, timestamp_s=timestamp)
                self._index += 1
                emitted += 1
                continue

            # @step Idle: motion detection only.  Downscale hard so this costs almost nothing.
            small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
            if previous is not None:
                difference = cv2.absdiff(small, previous)
                changed = float(np.count_nonzero(cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY) > 25))
                if changed / float(small.shape[0] * small.shape[1]) >= self.motion_threshold:
                    # @step Emit the pre-roll, then continue for the rest of the burst.
                    for buffered_index, buffered_image, buffered_time in ring:
                        yield Frame(
                            image=buffered_image,
                            index=buffered_index,
                            timestamp_s=buffered_time,
                        )
                        self._index += 1
                        emitted += 1
                    ring.clear()
                    emitting = burst_len
                    previous = small
                    continue
            previous = small
            ring.append((index, image, timestamp))
            if len(ring) > preroll:
                ring.pop(0)

    # -- diagnostics --------------------------------------------------------

    def describe(self) -> dict[str, object]:
        """@brief Summary of the source and its timing regime, for the run log."""
        return {
            "uri": self.resolved,
            "live": self._is_live,
            "mode": self.mode,
            "reported_fps": self.reported_fps,
            "frame_count": self.frame_count,
            "timestamp_source": "monotonic clock (live)" if self._is_live else "frame_index / fps (file)",
            "size": [self.requested_width, self.requested_height],
        }


def probe_source(uri: str | int, width: int | None = None, height: int | None = None) -> dict[str, object]:
    """Open a source briefly to report its actual resolution and rate.

    @brief   One-shot source inspection.
    @details Used by the calibration command to size its window and to validate intrinsics against
             a real frame.  The measured resolution can differ from the request -- a camera may not
             support the requested size and will silently substitute another -- and a mismatch
             against the configured intrinsics would invalidate every projection, so it is worth
             checking rather than assuming.
    @param   uri    Source specification.
    @param   width  Requested width.
    @param   height Requested height.
    @return  Mapping with the observed frame size, reported rate and timestamp regime.
    @raises  SourceError when the source cannot be opened or yields no frame.
    """
    source = FrameSource(uri, width=width, height=height, mode="stream", warmup_frames=0)
    with source as src:
        for frame in src:
            height_px, width_px = frame.image.shape[:2]
            info = src.describe()
            info["observed_size"] = [int(width_px), int(height_px)]
            info["first_frame_timestamp_s"] = frame.timestamp_s
            return info
    raise SourceError(f"source {uri!r} opened but produced no frames")
