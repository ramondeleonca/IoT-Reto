"""Robust metric velocity estimation from tracked ground positions.

@file    speed.py
@brief   Turn a track's position history into a defensible speed in metres per second.
@details This module deliberately depends on **NumPy only** and knows nothing about YOLO,
         OpenCV or pixels.  It receives already-projected ground positions in metres
         together with the timestamp of each observation, so it can be tested exhaustively
         against synthetic ground truth on a memory-constrained device without a detector in
         the loop.

         Velocity looks like a one-line calculation -- distance over time -- and that single
         line is where most monocular speed projects go wrong.  The specific traps handled
         here, each of which is a real failure mode rather than a theoretical concern:

         * **Timestamps.**  ``frame_index / fps`` is wrong whenever the source is
           variable-frame-rate, which includes phone recordings and, critically, the Raspberry
           Pi Camera Module, whose ``CAP_PROP_FPS`` is unreliable and which carries no hardware
           timestamp.  Timestamps are therefore supplied by the caller rather than assumed.
         * **Scalar time.**  A car crossing the frame is measured over a finite interval.  If
           the interval is short the position noise dominates; if it is long the car's own
           acceleration contaminates the estimate.  Both matter, so both are reported.
         * **Outliers and identity switches.**  A single mis-associated detection can move a
           track's apparent position by metres for one frame.  Naive differencing turns that
           into a huge spurious speed that survives smoothing, so steps are gated and the
           estimator is median-based rather than mean-based.
         * **Quantisation.**  Detector boxes are integers in many pipelines, which puts a
           floor on achievable resolution.  The reported noise floor makes that visible
           instead of letting it masquerade as measured variation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SpeedEstimate",
    "TrackHistory",
    "SpeedEstimator",
    "SpeedEstimatorConfig",
    "smooth_positions",
    "SAVITZKY_GOLAY_5",
]


# @brief Savitzky-Golay quadratic smoothing coefficients for a 5-point window.
#       Written out rather than computed so the module needs no SciPy, and because this is
#       the only window length the pipeline uses.  A quadratic fit is the right choice here:
#       a moving average biases the estimate wherever a car is accelerating, whereas a local
#       quadratic preserves curvature, and the coefficients sum to 1 so a constant is exact.
SAVITZKY_GOLAY_5: tuple[float, ...] = (-3.0, 12.0, 17.0, 12.0, -3.0)


def smooth_positions(
    positions: np.ndarray,
    window_coeffs: tuple[float, ...] = SAVITZKY_GOLAY_5,
    normalize: bool = True,
) -> np.ndarray:
    """Smooth a position series with a Savitzky-Golay kernel.

    @brief   Reduce detector jitter without flattening genuine acceleration.
    @details Convolution along the time axis with edge replication.  Short series (shorter
             than the kernel) are returned unchanged, because hiding a track's only
             observations behind a filter would produce a confident-looking number from
             almost no data.

             @note Edge replication makes the filter exact for a constant signal and for a
             linear ramp in the *interior*, but not at the two boundary samples: replicating
             the last value implies zero slope there, so the fitted quadratic bends toward it.
             For a linear ramp the last two samples are pulled low by roughly a third of one
             step.  That is a deliberate trade.  Linear extrapolation would preserve a ramp
             exactly, but it would also extrapolate a car's position into free space at the
             frame edge, which is precisely where tracks begin and end and where a fabricated
             sample is most misleading.  A small boundary bias on a car that is entering or
             leaving view is the safer failure, and the estimator is robust to it anyway.
    @param   positions     ``(N, 2)`` or ``(N,)`` position samples in order.
    @param   window_coeffs Odd-length symmetric kernel.
    @param   normalize     Divide by the coefficient sum so a constant signal is preserved.
    @return Array of the same shape as ``positions``.
    @raises  ValueError for a non-positive or even kernel length.
    """
    samples = np.asarray(positions, dtype=np.float64)
    coeffs = np.asarray(window_coeffs, dtype=np.float64)
    if coeffs.size == 0 or coeffs.size % 2 == 0:
        raise ValueError(f"the smoothing kernel must have a positive odd length, got {coeffs.size}")
    if samples.shape[0] < coeffs.size:
        return samples.copy()
    if normalize:
        total = float(coeffs.sum())
        if abs(total) > 1e-12:
            coeffs = coeffs / total
    half = coeffs.size // 2
    # @step Edge replication rather than zero padding: zero padding would drag the first and
    #       last samples toward the origin and manufacture a deceleration at both ends of
    #       every track, which is exactly where a car enters and leaves the measurement zone.
    padded = np.pad(samples, [(half, half)] + [(0, 0)] * (samples.ndim - 1), mode="edge")
    out = np.zeros_like(samples, dtype=np.float64)
    for offset, weight in enumerate(coeffs):
        out += weight * padded[offset : offset + samples.shape[0]]
    return out


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Locate maximal runs of consecutive ``True`` values, as inclusive index ranges.

    @brief   Group valid samples so each run can be filtered independently.
    @details Used to smooth a track only across steps that are themselves plausible, so that a
             rejected step acts as a hard boundary the smoothing kernel cannot reach across.
             Runs are returned as inclusive ``(start, end)`` sample-index pairs; a run of length
             one is a single isolated sample and is passed through untouched by the caller.
    @param   mask ``(N,)`` boolean array.
    @return  List of inclusive ``(start, end)`` index pairs in ascending order.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(np.asarray(mask, dtype=bool)):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _sampling_is_regular(dt: np.ndarray, max_cv: float) -> bool:
    """Decide whether position smoothing is safe for this sampling pattern.

    @brief   Uniformity gate for the smoothing kernel.
    @details The Savitzky-Golay coefficients used here are derived for evenly spaced samples.  On
             irregular samples they are not conservative: a constant-velocity track whose
             intervals vary by a factor of two passes through the filter with per-step speeds
             spanning roughly 1.4 to 2.7 times the true speed instead of all being equal, which
             silently degrades the estimate.  Measuring the spread of the intervals and skipping
             smoothing when it is large avoids applying a filter outside its assumptions.

             The coefficient of variation (standard deviation over mean) is used rather than a
             min/max ratio because it responds to the overall irregularity rather than to a
             single outlier interval.
    @param   dt     ``(M,)`` consecutive time intervals in seconds.
    @param   max_cv Largest tolerated coefficient of variation.
    @return  ``True`` when smoothing may be applied.
    """
    if dt.size < 2:
        return False
    mean = float(np.mean(dt))
    if mean <= 0.0:
        return False
    return bool(float(np.std(dt)) / mean <= max_cv)


@dataclass
class SpeedEstimatorConfig:
    """Tuning for the velocity estimator.

    @brief   All thresholds in one place so they can be set per deployment.
    @details Defaults are chosen for a table-top Hot Wheels track filmed with a phone or a Pi
             camera at 10-30 FPS, where cars cross the frame in roughly a second.  A real
             street deployment at 30-60 FPS would want a shorter ``window_s``.

    @param  window_s          Length of the sliding measurement window, seconds.
    @param  min_span_s        Minimum time span before any speed is reported, seconds.
    @param  max_step_m        Largest plausible ground displacement between two consecutive
                              observations, metres.  Anything larger is treated as an
                              identity switch or a detection glitch rather than motion.
    @param  min_samples       Minimum observations before a speed is reported.
    @param  max_speed_mps     Hard plausibility bound; measurements above it are rejected.
    @param  use_smoothing     Apply the Savitzky-Golay kernel before differencing.
    @param  robust_percentile Percentile of the per-step speeds used for the robust estimate.
                              The median (50) is the default; a higher value biases toward the
                              faster end, which is occasionally useful when a car is partly
                              occluded for part of its transit.
    @param  max_interval_cv   Largest coefficient of variation of the sample intervals for
                              which position smoothing is still applied.  The smoothing kernel
                              assumes uniformly spaced samples, so on a variable-frame-rate
                              source it is not conservative -- it distorts a constant-velocity
                              track instead of passing it through.  Above this threshold
                              smoothing is skipped and the robust statistic is relied on alone,
                              which is the honest choice: it is better to accept noisier input
                              than to apply a filter whose assumptions do not hold.
    """

    window_s: float = 0.5
    min_span_s: float = 0.08
    max_step_m: float = 0.5
    min_samples: int = 3
    max_speed_mps: float = 12.0
    use_smoothing: bool = True
    robust_percentile: float = 50.0
    max_interval_cv: float = 0.2

    def __post_init__(self) -> None:
        if self.window_s <= 0.0:
            raise ValueError(f"window_s must be > 0, got {self.window_s}")
        if self.min_span_s <= 0.0:
            raise ValueError(f"min_span_s must be > 0, got {self.min_span_s}")
        if self.max_step_m <= 0.0:
            raise ValueError(f"max_step_m must be > 0, got {self.max_step_m}")
        if not 0.0 <= self.robust_percentile <= 100.0:
            raise ValueError(f"robust_percentile must be in [0, 100], got {self.robust_percentile}")
        if self.max_interval_cv < 0.0:
            raise ValueError(f"max_interval_cv must be >= 0, got {self.max_interval_cv}")


@dataclass
class SpeedEstimate:
    """A single speed measurement with the context needed to trust or discount it.

    @brief   Speed plus its uncertainty and provenance.
    @details Reporting a bare number invites over-interpretation.  Every field here exists
             because a plausible-looking speed can be produced from too few samples, over too
             short an interval, or at a ground scale where the resolution alone explains the
             variation.

    @param  speed_mps       Robust speed estimate in metres per second, or ``None`` if not
                            yet reportable.
    @param  speed_kmh       The same value in km/h, for readability.
    @param  samples         Number of observations used.
    @param  span_s          Time span of the measurement window, seconds.
    @param  std_mps         Standard deviation of the per-step speeds, m/s.  A large value
                            relative to ``speed_mps`` means a noisy or manoeuvring track.
    @param  noise_floor_mps Lower bound on the speed uncertainty implied by the ground
                            resolution and the sample interval, m/s.
    @param  distance_m      Straight-line ground distance covered across the window, metres.
    @param  direction_rad   Travel direction in the ground plane, radians, or ``None``.
    @param  rejected_steps  Count of consecutive-observation steps discarded as implausible.
    @param  reason          Why no speed was reported, when ``speed_mps`` is ``None``.
    """

    speed_mps: float | None = None
    speed_kmh: float | None = None
    samples: int = 0
    span_s: float = 0.0
    std_mps: float = 0.0
    noise_floor_mps: float = 0.0
    distance_m: float = 0.0
    direction_rad: float | None = None
    rejected_steps: int = 0
    reason: str = ""

    @property
    def is_reportable(self) -> bool:
        """@brief Whether a speed value was produced."""
        return self.speed_mps is not None

    def to_row(self) -> dict[str, float | int | str | None]:
        """@brief Flatten to a CSV/JSON-friendly mapping."""
        return {
            "speed_mps": self.speed_mps,
            "speed_kmh": self.speed_kmh,
            "samples": self.samples,
            "span_s": self.span_s,
            "std_mps": self.std_mps,
            "noise_floor_mps": self.noise_floor_mps,
            "distance_m": self.distance_m,
            "direction_rad": self.direction_rad,
            "rejected_steps": self.rejected_steps,
            "reason": self.reason,
        }


@dataclass
class TrackHistory:
    """Observations of one tracked vehicle over time.

    @brief   Append-only record of ground positions, timestamps and scales.
    @details Positions are stored in the road frame in metres, already projected through the
             ground geometry; the tracker layer is responsible for using the *ground contact
             point* of each detection rather than the box centre, since projecting a box centre
             places the vehicle's roof on the road ahead of it and biases every speed high.

    @param  track_id          Tracker-assigned identity.
    @param  times             Observation timestamps in seconds, monotonically non-decreasing.
    @param  positions         ``(N, 2)`` ground positions ``(X, Z)`` in metres.
    @param  scales            ``(N,)`` metres-per-pixel at each observation, for the noise floor.
    @param  confidences       ``(N,)`` detector confidences.
    @param  class_ids         ``(N,)`` detector class indices.
    """

    track_id: int
    times: list[float] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        """@brief Number of observations recorded."""
        return len(self.times)

    def append(
        self,
        time_s: float,
        position_xz: np.ndarray,
        scale_m_per_px: float = math.nan,
        confidence: float = math.nan,
        class_id: int = -1,
    ) -> None:
        """Add one observation.

        @brief   Record a ground position and its timestamp.
        @param   time_s            Timestamp in seconds.
        @param   position_xz       Ground position ``(X, Z)`` in metres.
        @param   scale_m_per_px    Local metres-per-pixel, for the noise floor.
        @param   confidence        Detector confidence.
        @param   class_id          Detector class index.
        @raises  ValueError when the timestamp is not monotonic, because a non-monotonic
                 history silently produces negative or wildly wrong durations.
        """
        if self.times and time_s < self.times[-1]:
            raise ValueError(
                f"timestamps must be non-decreasing for track {self.track_id}: "
                f"got {time_s} after {self.times[-1]}"
            )
        if not np.all(np.isfinite(position_xz)):
            raise ValueError(f"non-finite ground position for track {self.track_id}: {position_xz}")
        self.times.append(float(time_s))
        self.positions.append(np.asarray(position_xz, dtype=np.float64).reshape(2))
        self.scales.append(float(scale_m_per_px))
        self.confidences.append(float(confidence))
        self.class_ids.append(int(class_id))

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """@brief Return ``(times, positions, scales)`` as stacked arrays."""
        return (
            np.asarray(self.times, dtype=np.float64),
            np.asarray(self.positions, dtype=np.float64).reshape(-1, 2),
            np.asarray(self.scales, dtype=np.float64),
        )

    def recent(self, window_s: float) -> "TrackHistory":
        """Extract the trailing time window.

        @brief   Sub-history covering the last ``window_s`` seconds.
        @details Keeping the estimate local in time is what allows a car that slows, stops or
                 accelerates to be tracked honestly instead of averaging its whole transit into
                 one meaningless number.  At least two samples are always retained if available.
        @param   window_s Window length in seconds.
        @return  A new :class:`TrackHistory` sharing no mutable state with this one.
        """
        if not self.times:
            return TrackHistory(self.track_id)
        cutoff = self.times[-1] - float(window_s)
        start = 0
        for idx, time_s in enumerate(self.times):
            if time_s >= cutoff:
                start = idx
                break
        start = max(0, min(start, max(0, len(self.times) - 2)))
        return TrackHistory(
            track_id=self.track_id,
            times=self.times[start:],
            positions=[p.copy() for p in self.positions[start:]],
            scales=self.scales[start:],
            confidences=self.confidences[start:],
            class_ids=self.class_ids[start:],
        )


class SpeedEstimator:
    """Estimate a track's speed from its ground-position history.

    @brief   Robust, windowed, outlier-gated metric speed.
    @details The method, and why each part of it is there:

             1. **Restrict to a recent window** so the estimate describes what the vehicle is
                doing now rather than an average over its whole transit.
             2. **Optionally smooth** the positions to suppress detector jitter.  Smoothing is
                applied to positions, not to speeds, because differencing a noisy position
                series amplifies noise whereas a local quadratic fit suppresses it while
                preserving acceleration.
             3. **Gate consecutive steps** on plausibility.  A step larger than
                ``max_step_m`` is far more likely to be an identity switch than real motion,
                and leaving one in place corrupts a median only slightly but a mean badly --
                nevertheless dropping it explicitly makes the failure visible through
                ``rejected_steps``.
             4. **Take a robust statistic** over the per-step speeds rather than the endpoint
                difference.  The median resists a single outlier; the endpoint difference does
                not, because it is exactly the difference of two noisy samples.
             5. **Report the noise floor** implied by the ground scale and the sample interval,
                so a speed whose standard deviation is below that floor is known to be
                resolution-limited rather than genuinely precise.

    @param  config Tuning parameters; see :class:`SpeedEstimatorConfig`.
    """

    def __init__(self, config: SpeedEstimatorConfig | None = None) -> None:
        self.config = config or SpeedEstimatorConfig()

    def estimate(self, history: TrackHistory, pixel_sigma_px: float = 2.0) -> SpeedEstimate:
        """Compute a speed estimate for one track.

        @brief   Robust windowed speed with diagnostics.
        @details Returns an estimate with ``speed_mps is None`` and a human-readable
                 ``reason`` whenever the data cannot support a number.  Refusing to answer is a
                 valid outcome here and is far better than emitting a confident value computed
                 from two samples a millisecond apart.
        @param   history        Full observation history for the track.
        @param   pixel_sigma_px Assumed detector jitter, used for the noise floor.
        @return  :class:`SpeedEstimate`.
        """
        windowed = history.recent(self.config.window_s)
        times, positions, scales = windowed.as_arrays()

        if len(times) < self.config.min_samples:
            return SpeedEstimate(
                samples=len(times),
                reason=f"only {len(times)} observation(s); need {self.config.min_samples}",
            )
        span = float(times[-1] - times[0])
        if span < self.config.min_span_s:
            return SpeedEstimate(
                samples=len(times),
                span_s=span,
                reason=f"span {span:.4f}s shorter than the {self.config.min_span_s}s minimum",
            )

        dt = np.diff(times)
        plausible = (dt > 0.0) & (np.linalg.norm(np.diff(positions, axis=0), axis=1) <= self.config.max_step_m)
        rejected = int(np.count_nonzero(~plausible))

        # @step 1: smooth positions, never speeds, and only *within* runs of plausible steps.
        #       The ordering matters and was wrong in the first implementation: smoothing before
        #       gating lets a single teleport -- the signature of an identity switch -- bleed
        #       into its neighbours through the kernel, converting one bad sample into three.
        #       Gating first confines the contamination to the step that actually contains the
        #       outlier, which the robust statistic then rejects on its own.
        if self.config.use_smoothing and _sampling_is_regular(dt, self.config.max_interval_cv):
            working = positions.copy()
            for lo, hi in _contiguous_runs(plausible):
                if hi - lo >= 1:
                    working[lo : hi + 1] = smooth_positions(positions[lo : hi + 1])
        else:
            # @note Skipped on irregular sampling.  The kernel is derived for evenly spaced
            #       samples and is not conservative otherwise, so applying it would corrupt the
            #       very quantity being measured.  See SpeedEstimatorConfig.max_interval_cv.
            working = positions

        # @step 2: per-step velocities over the surviving steps.
        deltas = np.diff(working, axis=0)
        steps = np.linalg.norm(deltas, axis=1)
        good = plausible
        if not np.any(good):
            return SpeedEstimate(
                samples=len(times),
                span_s=span,
                rejected_steps=rejected,
                reason="every consecutive step was implausible; likely tracking instability",
            )

        # @step 3: robust speed, plus the spread, from the surviving steps.
        step_speeds = steps[good] / dt[good]
        speeds_ok = step_speeds[step_speeds <= self.config.max_speed_mps]
        dropped_fast = int(step_speeds.size - speeds_ok.size)
        if speeds_ok.size == 0:
            return SpeedEstimate(
                samples=len(times),
                span_s=span,
                rejected_steps=rejected + dropped_fast,
                reason=f"all steps exceeded the {self.config.max_speed_mps} m/s plausibility bound",
            )

        speed = float(np.percentile(speeds_ok, self.config.robust_percentile))
        std = float(np.std(speeds_ok)) if speeds_ok.size > 1 else 0.0

        # @step 4: net displacement and direction over the window, which is what a human
        #       reading a "car went this way at this speed" label actually wants.
        net = working[-1] - working[0]
        distance = float(np.linalg.norm(net))
        direction = float(math.atan2(net[0], net[1])) if distance > 1e-9 else None

        # @step 5: noise floor from the local ground resolution and the sample interval.  The
        #       median interval is used because irregular sampling is normal on a Pi and a mean
        #       would be dragged by a single long gap.
        valid_scales = scales[np.isfinite(scales)]
        if valid_scales.size:
            median_scale = float(np.median(valid_scales))
            median_dt = float(np.median(dt[good]))
            noise_floor = math.sqrt(2.0) * pixel_sigma_px * median_scale / max(median_dt, 1e-9)
        else:
            noise_floor = 0.0

        return SpeedEstimate(
            speed_mps=speed,
            speed_kmh=speed * 3.6,
            samples=len(times),
            span_s=span,
            std_mps=std,
            noise_floor_mps=noise_floor,
            distance_m=distance,
            direction_rad=direction,
            rejected_steps=rejected + dropped_fast,
        )

    def estimate_all(
        self, histories: dict[int, TrackHistory], pixel_sigma_px: float = 2.0
    ) -> dict[int, SpeedEstimate]:
        """@brief Estimate speeds for every track in a mapping."""
        return {
            track_id: self.estimate(history, pixel_sigma_px)
            for track_id, history in histories.items()
        }
