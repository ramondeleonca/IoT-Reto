# Deploying on Raspberry Pi and Orange Pi

@file    deployment.md
@brief   Getting `cardetect` running on the two single-board targets, with honest expectations.
@details This is written for someone who has a board, a camera and none of the tooling, in the order
         they will need it. Where a figure is an estimate rather than a measurement it says so,
         because an unrealistic expectation about frame rate leads to blaming the detector for a
         limitation imposed by the hardware.

## Which board for what

| | **Raspberry Pi Zero 2 W** | **Orange Pi 3B** |
|---|---|---|
| CPU | 4× Cortex-A53 @ 1 GHz | 4× Cortex-A55 @ 2.4 GHz |
| RAM | 512 MB | 2–8 GB |
| NPU | **none** | **~6 TOPS** (RK3588S) |
| Realistic detection rate | **1–4 FPS** at 320–416 px | **15–30 FPS** with RKNN |
| Capture strategy | motion-triggered **burst** | continuous **stream** |
| Verdict | works, as an event recorder | works as a live meter |

The Zero 2 W is not a video processing device, and no amount of tuning changes that. It *is* a
perfectly good event recorder, and because speed is a per-track quantity, an event recorder loses
nothing: only the frames containing a car carry information, and those are exactly the frames a
burst captures. That is why `source.mode: burst` exists.

## 1. Install

```bash
sudo apt update && sudo apt install -y python3-venv git libgl1 libglib2.0-0
git clone <your-repo> && cd yolo-cardetect
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: pipx install uv
uv sync --extra vision
uv run cardetect doctor
```

`libgl1` and `libglib2.0-0` are needed by OpenCV's shared objects on a headless Debian image; without
them `import cv2` fails with a message about `libGL.so.1` that does not mention OpenCV.

`cardetect doctor` is the first thing to run on any new board. It reports which layers are usable, and
it distinguishes "the geometry works" from "the detector can run", which on these devices are
genuinely different questions. A NumPy-only environment can still do calibration, `check` and
`preview`.

## 2. Enable the camera

### Pi Zero 2 W

```bash
sudo raspi-config        # Interface Options -> Legacy Camera, or use libcamera
libcamera-still -o test.jpg          # confirms the sensor works
```

OpenCV reaches a libcamera sensor through the `libcamera` V4L2 compatibility layer, which exposes it
as `/dev/video0`. Then `--source 0` works as usual.

### Orange Pi 3B

The MIPI camera appears as `/dev/video*` once the device tree overlay is enabled. Check with
`v4l2-ctl --list-devices`, and prefer the `rkisp` node over the raw one.

## 3. Export a model the board can actually run

The default `yolo26n.pt` is PyTorch, which is far too heavy for a Zero 2 W. Export something
smaller. Do this on a workstation, not on the board.

### NCNN — the Zero 2 W path

```bash
# on a workstation
uv run yolo export model=yolo11n.pt format=ncnn imgsz=320 half=False
# copy the resulting directory to the board
scp -r yolo11n_ncnn_model pi@zero:/home/pi/yolo-cardetect/models/
```

Then point the config at it:

```yaml
detector:
  model: models/yolo11n_ncnn_model
  imgsz: 320
  device: cpu
  quantize: off        # NCNN handles its own precision
```

Why **YOLO11n** and not YOLO26n here: NCNN export support lags new architectures, and a model that
exports cleanly at 320 px beats a newer one that does not run. This is a deliberate trade of accuracy
for a working deployment, and it is the right trade on this board because the ground resolution, not
the detector, is usually what limits speed accuracy.

### RKNN — the Orange Pi path

```bash
# on a workstation, or on the board with the RKNN toolkit
uv run yolo export model=yolo11n.pt format=rknn imgsz=640
```

RKNN conversion needs Rockchip's toolchain and a matching board target string. Follow their
documentation; the important point is that the same `detect.py` handles the result, because
Ultralytics dispatches on the model directory rather than on the file extension.

## 4. Calibrate on the board

Calibration needs a display for the two clicks. If the board is headless, do one of these instead:

* **Measure the height and skip clicking.** For an overhead rig this is the most accurate option and
  needs no display at all:

  ```yaml
  kind: topdown
  height_m: 0.85      # to the LENS, from the surface the cars run on
  ```

* **Calibrate on a workstation**, commit the resulting `calibration/*.yaml`, and copy it to the board.

## 5. Run

```bash
# Continuous, for the Orange Pi
uv run cardetect run --config config/camera_topdown.yaml --source 0 \
    --model models/yolo11n_ncnn_model --imgsz 640 --no-video

# Burst, for the Zero 2 W
uv run cardetect run --config config/camera_topdown.yaml --source 0 \
    --mode burst --model models/yolo11n_ncnn_model --imgsz 320 --no-video
```

`--no-video` matters: writing an annotated video on the Zero 2 W costs a meaningful fraction of the
frame budget and fills the SD card. The CSV and JSON output carry the measurements; run with video
enabled only when reviewing.

The `annotated.mp4` writer uses `mp4v`, which is available in every OpenCV build. On a board with a
hardware H.264 encoder you can gain throughput by piping to `ffmpeg`, but it is not worth the
complexity until the detector is no longer the bottleneck — and on a Zero 2 W it never will be.

## 6. Running as a service

```ini
# /etc/systemd/system/cardetect.service
[Unit]
Description=cardetect speed measurement
After=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/yolo-cardetect
# Burst mode, because this board cannot run continuously.
ExecStart=/home/pi/.local/bin/uv run cardetect run \
    --config config/camera_topdown.yaml --source 0 \
    --mode burst --no-video --no-metric
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Note `--no-metric` in the template: it is there to make the service start cleanly on a rig that has
not been calibrated yet, and should be removed once a calibration exists. Running uncalibrated
produces tracks and an annotated video but no speeds, and the CLI says so explicitly.

## 7. What to expect, and why

| Symptom | Cause | Fix |
|---|---|---|
| Speeds noisy, jumping by 50% | Frame interval too long for the window | Shorten `speed.window_s`, raise `imgsz`, or move the camera closer |
| `no track produced a speed` | Too few samples, or the car crosses the view too fast | Enable burst mode; slow the cars; raise the camera |
| Speeds consistently high or low by a fixed factor | **Height measured to the wrong reference** | Re-measure to the lens pupil, above the track surface |
| Detections vanish every few frames | Confidence threshold too high for a toy-sized object | Lower `detector.conf` to 0.15–0.20 |
| Everything is slow | Model too large, or precision enabled on CPU | Export NCNN, drop `imgsz`, ensure `quantize: off` on CPU |
| Far cars give wild speeds | Beyond the error budget | Lower `max_error_m`, or ignore the far field |

The second row is by far the most common. `speed.window_s` divided by the frame interval gives the
number of samples behind a measurement: at 3 FPS and a 0.5 s window that is **1.5 samples**, which
cannot support a speed. The estimator refuses rather than inventing one, and reports why. Either
shorten the window's *time* span or raise the frame rate; burst mode raises the effective rate for
exactly the frames that matter.

The third row is the reason this project makes so much of how the height is measured. A wrong height
scales every speed by a constant, and a constant factor is invisible in the output — the numbers all
look entirely reasonable.

## 8. Thermal and power notes

* A Zero 2 W running inference continuously will thermally throttle within minutes without a
  heatsink. Burst mode is as much a thermal strategy as a throughput one: the board is idle most of
  the time.
* Brownouts under load cause `read()` to fail and produce a corrupt frame, not a clean error. Use a
  supply rated above 2.5 A; a phone charger is often not enough.
* Write the CSV to a ramdisk or a USB device if running for days. Continuous writes to an SD card
  will eventually destroy it, and the failure appears as random corruption rather than a disk error.
