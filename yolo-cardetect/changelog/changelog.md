# Changelog

All changes to `yolo-cardetect`, newest first. One document per change, per `AGENTS.md` §1,
named `YYYYMMDD-HHMMSS-change.md` after the time the change was made.

| Date and time | Change | Document | Status |
|---|---|---|---|
| 2026-10-06 20:04:25 | Project bootstrap, geometry core for both mounting configurations, 57-test geometry suite | [20261006-200425-change.md](20261006-200425-change.md) | complete |
| 2026-10-06 20:06:49 | Robust velocity estimator (28 tests) and configuration files for both rigs | [20261006-200649-change.md](20261006-200649-change.md) | complete |
| 2026-10-06 21:37:06 | Full pipeline, CLI, calibration workflow, synthetic end-to-end validation, rig diagrams | [20261006-213706-change.md](20261006-213706-change.md) | complete |

Test suite: **105 tests, all passing** (`uv run pytest`).

## Usage

```bash
uv sync --extra vision                                   # workstation
uv run cardetect doctor                                  # what can this device do?
uv run cardetect check    --config config/camera_topdown.yaml
uv run cardetect preview  --config config/camera_roadside.yaml --save-geometry
uv run cardetect calibrate topdown  --config config/camera_topdown.yaml --write-config
uv run cardetect run      --config config/camera_topdown.yaml --source 0
uv run python -m cardetect.tools.make_synthetic_video --config config/camera_roadside.yaml
```

## Current state

**Implemented and tested**

* `geometry.py` — camera models, analytic ground homography, affine and projective projectors,
  closed-form local resolution and error budget, calibration solvers for the tape-measure, horizon,
  height-prior and ArUco workflows.
* `speed.py` — robust windowed metric velocity, outlier gating, uniformity-gated Savitzky-Golay
  smoothing, per-measurement noise floor and refusal reasons.
* `source.py` — camera, file and stream capture with explicit timestamp regimes, plus
  motion-triggered burst capture for constrained devices.
* `detect.py` — YOLO wrapper, vehicle class filtering, configurable ground-contact extraction.
* `track.py` — association, per-track metric histories, identity lifetime, index-space safe.
* `pipeline.py` — run orchestration, artefact writing, plots.
* `calibration.py`, `cli.py` — interactive two-mark tape calibration and the command line.
* `visualize.py` — annotation overlays, minimap, speed, trajectory and ground-resolution figures.
* `config/`, `tools/`, `docs/rig-diagrams.svg`, `tests/` (105 tests).

**Pending**

* `tests/test_detect.py`, `tests/test_source.py` — unit coverage for the detector wrapper and the
  timestamp regimes. The behaviour is exercised end to end, but not unit-tested in isolation.
* Live webcam validation — the code path is implemented but has not been exercised against a real
  camera, and the detector has not yet been run on real vehicles.

## Known constraints carried forward

* A single measured distance cannot determine a 2D ground homography; the roadside tape workflow
  requires an additional horizon, height or tilt constraint. Enforced by
  `RoadsideProjector.from_measured_distance`, not defaulted.
* The overhead pose is a deliberate reflection (`det = -1`) so that image "down" means increasing
  road distance. A reflection preserves lengths, so speeds are unaffected.
* The two rigs use opposite `Z` sign conventions at the vertical limit; velocity magnitudes are
  unaffected. Documented in `geometry.py`.
* Position smoothing is skipped above `speed.max_interval_cv` (default 0.2) because the
  Savitzky-Golay kernel assumes uniform sampling and is not conservative on variable-frame-rate
  input.
* An overhead rig close enough to fill the frame with a table-top track sees only a few tens of
  centimetres of road (0.67 m at 0.85 m height with a 70° lens), which bounds the usable measurement
  window and therefore the measurable speeds.
* On a pitched camera, a distant vehicle's box compresses until its bottom edge no longer
  corresponds to the near side of the footprint. Such detections are gated, and this bounds the
  usable range independently of the error budget.
