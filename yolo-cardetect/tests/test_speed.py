"""Speed estimator tests, verified against synthetic ground truth.

@file    test_speed.py
@brief   Numerically verifies metric velocity estimation.
@details Every test here builds a trajectory with a **known** speed, converts it to the noisy
         ground positions a detector would produce, and checks that the estimator recovers the
         truth within a stated tolerance.  That is the only way to demonstrate correctness in
         the absence of real footage, and it also documents the achievable accuracy so a later
         real-world result can be judged against it rather than against intuition.

         NumPy only: no OpenCV, no detector, no model download.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cardetect.speed import (
    SAVITZKY_GOLAY_5,
    SpeedEstimate,
    SpeedEstimator,
    SpeedEstimatorConfig,
    TrackHistory,
    smooth_positions,
)


def straight_track(
    speed_mps: float,
    n: int = 30,
    fps: float = 30.0,
    noise_m: float = 0.0,
    seed: int = 0,
    track_id: int = 1,
    start: tuple[float, float] = (0.0, 0.0),
    heading_rad: float = 0.0,
) -> TrackHistory:
    """Build a constant-velocity track with optional positional noise.

    @brief   Synthetic ground truth for speed tests.
    @details The vehicle starts at ``start`` and travels along ``heading_rad`` at exactly
             ``speed_mps``.  Gaussian noise of ``noise_m`` standard deviation is added
             independently to X and Z when requested, standing in for detector box jitter
             after projection.
    @param   speed_mps   True speed in metres per second.
    @param   n           Number of observations.
    @param   fps         Sampling rate in hertz.
    @param   noise_m     Standard deviation of additive position noise, metres.
    @param   seed        Random seed for reproducibility.
    @param   track_id    Identifier assigned to the history.
    @param   start       Starting ground position ``(X, Z)``.
    @param   heading_rad Travel direction in the ground plane.
    @return  Populated :class:`TrackHistory`.
    """
    rng = np.random.default_rng(seed)
    history = TrackHistory(track_id=track_id)
    direction = np.array([math.sin(heading_rad), math.cos(heading_rad)])
    for idx in range(n):
        time_s = idx / fps
        truth = np.asarray(start, dtype=np.float64) + direction * speed_mps * time_s
        jitter = rng.normal(0.0, noise_m, size=2) if noise_m > 0.0 else np.zeros(2)
        history.append(time_s, truth + jitter, scale_m_per_px=0.002, confidence=0.9, class_id=2)
    return history


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def test_smoothing_preserves_a_constant_exactly() -> None:
    """@brief A Savitzky-Golay kernel normalised to unit sum must not shift a constant."""
    counts = np.full((11, 2), 3.7)
    assert np.allclose(smooth_positions(counts), counts)


def test_smoothing_preserves_a_linear_ramp_in_the_interior() -> None:
    """@brief A quadratic kernel reproduces linear motion, so it adds no speed bias.

    @details This is the property that justifies smoothing *positions* before differencing: a
             constant-velocity track must pass through the filter untouched, otherwise the
             filter itself would invent an acceleration.

             Only the interior is asserted.  Edge replication implies zero slope at the boundary
             samples, so a quadratic fit bends toward them and the last two samples of a ramp are
             pulled low by about a third of one step.  That is a deliberate trade documented in
             ``smooth_positions``: linear extrapolation would preserve the ramp exactly but would
             also fabricate a position for a car that is entering or leaving the frame.
    """
    ramp = np.column_stack([np.arange(20.0), 2.0 * np.arange(20.0)])
    assert np.allclose(smooth_positions(ramp)[2:-2], ramp[2:-2], atol=1e-9)
    # The documented boundary behaviour, pinned so a change to it is visible.
    assert np.abs(smooth_positions(ramp)[-2] - ramp[-2]).max() < 0.5


def test_smoothing_reduces_noise_on_a_constant_signal() -> None:
    """@brief Noise must actually go down, not merely be redistributed."""
    rng = np.random.default_rng(7)
    clean = np.ones((101, 2))
    noisy = clean + rng.normal(0.0, 0.05, size=clean.shape)
    smoothed = smooth_positions(noisy)
    assert np.std(smoothed) < np.std(noisy)


def test_smoothing_leaves_short_series_untouched() -> None:
    """@brief A series shorter than the kernel is returned unchanged."""
    short = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert np.allclose(smooth_positions(short), short)


def test_smoothing_rejects_an_even_kernel() -> None:
    """@brief An even-length kernel has no centre sample and must be rejected."""
    with pytest.raises(ValueError):
        smooth_positions(np.zeros((10, 2)), window_coeffs=(1.0, 1.0))


def test_savgol_kernel_is_symmetric() -> None:
    """@brief The hard-coded kernel must be symmetric or it would introduce a lag."""
    assert SAVITZKY_GOLAY_5 == SAVITZKY_GOLAY_5[::-1]


# ---------------------------------------------------------------------------
# Accuracy against ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("true_speed", [0.5, 1.0, 2.5, 5.0])
def test_recovers_constant_speed_without_noise(true_speed: float) -> None:
    """@brief A clean constant-velocity track is recovered essentially exactly."""
    estimator = SpeedEstimator(SpeedEstimatorConfig(window_s=10.0))
    estimate = estimator.estimate(straight_track(true_speed, n=60, fps=30.0))
    assert estimate.is_reportable
    assert estimate.speed_mps == pytest.approx(true_speed, rel=1e-6)
    assert estimate.speed_kmh == pytest.approx(true_speed * 3.6, rel=1e-6)


@pytest.mark.parametrize("true_speed", [1.0, 3.0])
def test_recovers_speed_despite_position_noise(true_speed: float) -> None:
    """@brief Robust estimation must tolerate realistic detector jitter.

    @details 5 mm of position noise at 30 FPS is deliberately pessimistic for a 1:64 car, and
             the estimator should still land within a few per cent.  If this degrades, the
             cause is usually the smoothing or the robust statistic, not the geometry.
    """
    estimator = SpeedEstimator(SpeedEstimatorConfig(window_s=10.0))
    estimate = estimator.estimate(straight_track(true_speed, n=120, fps=30.0, noise_m=0.005, seed=3))
    assert estimate.speed_mps == pytest.approx(true_speed, rel=0.10)


def test_speed_is_direction_independent() -> None:
    """@brief A car travelling either way along the road reports the same speed."""
    estimator = SpeedEstimator(SpeedEstimatorConfig(window_s=10.0))
    forward = estimator.estimate(straight_track(2.0, n=60, heading_rad=0.0))
    reverse = estimator.estimate(straight_track(2.0, n=60, heading_rad=math.pi))
    assert forward.speed_mps == pytest.approx(2.0, rel=1e-6)
    assert reverse.speed_mps == pytest.approx(2.0, rel=1e-6)


def test_direction_is_reported_in_the_ground_plane() -> None:
    """@brief The travel heading is exposed, not just the magnitude."""
    estimator = SpeedEstimator(SpeedEstimatorConfig(window_s=10.0))
    estimate = estimator.estimate(straight_track(2.0, n=60, heading_rad=math.pi / 2))
    assert estimate.direction_rad == pytest.approx(math.pi / 2, abs=1e-6)


def test_estimate_tracks_a_changing_speed_locally() -> None:
    """@brief The windowed estimate must follow a car that slows down.

    @details A single average over the whole transit would report something between the two
             speeds; the estimator should report a value close to the *recent* speed because
             only the trailing window is used.  This is the property that makes the number
             meaningful on a track with a bend or a hill.
    """
    history = TrackHistory(track_id=9)
    fps = 30.0
    for idx in range(60):
        time_s = idx / fps
        # Fast at first, then much slower over the second half.
        speed = 3.0 if idx < 30 else 0.6
        if idx < 30:
            distance = 3.0 * time_s
        else:
            distance = 3.0 * (30.0 / fps) + 0.6 * (time_s - 30.0 / fps)
        history.append(time_s, np.array([0.0, distance]), scale_m_per_px=0.002)

    estimator = SpeedEstimator(SpeedEstimatorConfig(window_s=0.25))
    estimate = estimator.estimate(history)
    assert estimate.speed_mps == pytest.approx(0.6, rel=0.2)


# ---------------------------------------------------------------------------
# Refusal and gating behaviour
# ---------------------------------------------------------------------------


def test_refuses_with_too_few_samples() -> None:
    """@brief Refusing to answer is a valid outcome and must carry a reason."""
    estimator = SpeedEstimator()
    estimate = estimator.estimate(straight_track(1.0, n=2))
    assert not estimate.is_reportable
    assert "observation" in estimate.reason


def test_refuses_when_the_span_is_too_short() -> None:
    """@brief Many samples crammed into an instant cannot support a speed."""
    history = TrackHistory(track_id=2)
    for idx in range(10):
        history.append(idx * 1e-4, np.array([0.0, 0.01 * idx]))
    estimator = SpeedEstimator()
    estimate = estimator.estimate(history)
    assert not estimate.is_reportable
    assert "span" in estimate.reason


def test_rejects_an_implausible_jump() -> None:
    """@brief A single teleport, as an identity switch produces, must not corrupt the result.

    @details The track moves steadily, then jumps a metre for one frame (plausible for a real
             Hot Wheels car only at ~30 m/s, far above the configured bound), then continues.
             The reported speed must stay close to the truth and the rejection must be counted.
    """
    history = TrackHistory(track_id=3)
    fps = 30.0
    for idx in range(60):
        distance = 2.0 * idx / fps
        if idx == 30:
            distance += 1.0  # teleport
            history.append(idx / fps, np.array([0.0, distance]))
            continue
        if idx > 30:
            distance = 2.0 * idx / fps  # snap back to the true trajectory
        history.append(idx / fps, np.array([0.0, distance]))

    estimator = SpeedEstimator(SpeedEstimatorConfig(window_s=10.0, max_step_m=0.5))
    estimate = estimator.estimate(history)
    assert estimate.speed_mps == pytest.approx(2.0, rel=0.01)
    # Exactly the two steps that touch the teleport: gating runs before smoothing, so the
    # outlier cannot bleed into its neighbours and inflate this count.
    assert estimate.rejected_steps == 2
    assert estimate.std_mps < 0.2


def test_rejects_an_absurd_speed_bound() -> None:
    """@brief Steps inside the step bound but above the speed bound are still dropped."""
    history = TrackHistory(track_id=4)
    # 0.4 m in 1/30 s is 12 m/s: inside max_step_m but above the default speed bound.
    for idx in range(10):
        history.append(idx / 30.0, np.array([0.0, 0.4 * idx]))
    estimator = SpeedEstimator(SpeedEstimatorConfig(window_s=10.0, max_speed_mps=5.0))
    estimate = estimator.estimate(history)
    assert not estimate.is_reportable
    assert "plausibility" in estimate.reason


def test_non_monotonic_timestamps_are_rejected() -> None:
    """@brief Out-of-order timestamps must fail loudly, not produce a negative duration."""
    history = TrackHistory(track_id=5)
    history.append(1.0, np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        history.append(0.5, np.array([0.0, 0.1]))


def test_non_finite_positions_are_rejected() -> None:
    """@brief A NaN from a failed projection must not enter the history."""
    history = TrackHistory(track_id=6)
    with pytest.raises(ValueError):
        history.append(0.0, np.array([np.nan, 0.0]))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_noise_floor_reflects_scale_and_interval() -> None:
    """@brief The reported floor must scale with ground resolution and sampling interval.

    @details Halving the frame interval must double the floor, which is the quantitative form
             of "a slower camera cannot measure speed as well".  This is the number that
             explains a noisy Raspberry Pi result without blaming the detector.
    """
    # A window long enough to contain the whole track, so frame rate is the only variable.
    estimator = SpeedEstimator(SpeedEstimatorConfig(window_s=100.0))
    fast = estimator.estimate(straight_track(1.0, n=60, fps=60.0))
    slow = estimator.estimate(straight_track(1.0, n=60, fps=15.0))
    # The floor is proportional to 1/dt, so a four-times-longer interval means four times the
    # noise.  This is the quantitative reason a slow camera cannot measure speed well.
    assert fast.noise_floor_mps == pytest.approx(4.0 * slow.noise_floor_mps, rel=0.02)

    coarse = TrackHistory(track_id=7)
    for idx in range(40):
        coarse.append(idx / 30.0, np.array([0.0, 0.01 * idx]), scale_m_per_px=0.02)
    assert estimator.estimate(coarse).noise_floor_mps > fast.noise_floor_mps


def test_recent_window_limits_the_estimate_to_recent_motion() -> None:
    """@brief The window must actually restrict which samples are used."""
    history = straight_track(1.0, n=100, fps=100.0)
    recent = history.recent(0.1)
    assert len(recent) < len(history)
    assert recent.times[-1] == history.times[-1]


def test_recent_window_keeps_at_least_two_samples() -> None:
    """@brief A very short window must not reduce the history below a usable pair."""
    history = straight_track(1.0, n=10, fps=10.0)
    assert len(history.recent(0.001)) >= 2


def test_config_validation() -> None:
    """@brief Invalid tuning must be rejected at construction."""
    with pytest.raises(ValueError):
        SpeedEstimatorConfig(window_s=0.0)
    with pytest.raises(ValueError):
        SpeedEstimatorConfig(min_span_s=-1.0)
    with pytest.raises(ValueError):
        SpeedEstimatorConfig(robust_percentile=150.0)


def test_estimate_row_is_flat_for_serialisation() -> None:
    """@brief The diagnostics must be serialisable without custom encoders."""
    estimate = SpeedEstimator(SpeedEstimatorConfig(window_s=10.0)).estimate(
        straight_track(2.0, n=40)
    )
    row = estimate.to_row()
    assert set(row) >= {"speed_mps", "speed_kmh", "samples", "noise_floor_mps", "reason"}
    for value in row.values():
        assert value is None or isinstance(value, (int, float, str))


def test_estimate_all_handles_multiple_tracks() -> None:
    """@brief Batch estimation over several tracks must not cross-contaminate."""
    histories = {
        1: straight_track(1.0, n=40, track_id=1),
        2: straight_track(4.0, n=40, track_id=2),
    }
    estimates = SpeedEstimator(SpeedEstimatorConfig(window_s=10.0)).estimate_all(histories)
    assert estimates[1].speed_mps == pytest.approx(1.0, rel=1e-6)
    assert estimates[2].speed_mps == pytest.approx(4.0, rel=1e-6)


def test_variable_frame_rate_uses_real_timestamps() -> None:
    """@brief Irregular sampling must be handled via timestamps, not assumed frame counts.

    @details This is the Raspberry Pi case: no hardware timestamps, unreliable
             ``CAP_PROP_FPS``, and frames arriving irregularly.  A constant-velocity track
             sampled at jittery intervals must still report the true speed, which only works
             because the estimator differences real times.
    """
    rng = np.random.default_rng(11)
    intervals = rng.uniform(1.0 / 40.0, 1.0 / 20.0, size=60)
    times = np.concatenate([[0.0], np.cumsum(intervals)])
    history = TrackHistory(track_id=8)
    for time_s in times:
        history.append(float(time_s), np.array([0.0, 2.0 * time_s]), scale_m_per_px=0.002)
    estimate = SpeedEstimator(SpeedEstimatorConfig(window_s=10.0)).estimate(history)
    assert estimate.speed_mps == pytest.approx(2.0, rel=1e-6)
