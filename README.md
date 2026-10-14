# Drone Car Tracker (simple_remake)

Detects vehicles in drone footage with two YOLO models, tracks every vehicle with BoT-SORT (each car keeps its ID,
even when YOLO misses it for a moment), and follows one car as the **target**. The target is chosen automatically, or
you lock one by clicking it.

## How to run

You need Windows, [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (a CPU works, but slowly).

```
uv sync          # first time only: creates .venv and installs the dependencies (CUDA 13 build of PyTorch)
uv run main.py   # or double-click run.bat
```

1. A **menu** opens. Pick a video (the first one in `videos/` is preselected), or switch Source to
   **Live stream** and pick a camera (see [Live stream](#live-stream-obs)), and press **Start**.
2. The models load (missing ones are downloaded into `models/` on first use) and a video window opens.
3. Press **Q** in the video window to return to the menu. Every run writes a log.

The defaults are the ones that were tuned on the sample footage. If you are unsure, leave them alone.

## Menu options

| Option | Default | Meaning |
|---|---|---|
| Source | Video file | **Video file** plays a file; **Live stream** reads a camera such as OBS Virtual Camera |
| Video | last used, or first in `videos/` | dropdown of every video in `videos/`, plus **Other...**. Choosing Other enables the path box and **Browse...** so you can pick a video from anywhere on the PC |
| Camera | last used, else the first OBS camera | the camera to read in Live stream mode. **Scan cameras** refreshes the list (e.g. after starting OBS) |
| Main model / size | `yolo26m.pt` / 960 | COCO model: good on low and oblique views. Lists detection models only. Choose "None" to run only the aerial model (no bus, motorcycle or people, since the aerial model lacks those classes) |
| Aerial model / size | `yolo26m-obb.pt` / 1280 | Lists OBB models only. DOTA aerial model: good on cars seen from straight above. Choose "None" to run only the main model (faster, misses more top-down cars) |
| Confidence floor | 0.10 | low on purpose: the tracker applies its own stricter thresholds and uses weak detections to keep tracks alive |
| Classes | all four | car, truck, bus, motorcycle |
| Track people | off | also detect and track pedestrians (boxes read `person #7`). Only the main model has a person class. People can be locked with a click or Tab, but are never chosen as the automatic target |
| Show trails | on | 2-second ground-pinned trail behind every car |
| Start at max speed | off | run uncapped instead of at the video's frame rate (video files only) |
| Device | Auto | Auto / CPU / GPU |

Your last choices are saved in `last_settings.json`, including keys pressed during the run.

## Live stream (OBS)

Any camera works, but OBS lets you replay a video as if it were a live feed:

1. In OBS, add a **Media Source** with the video and tick **Loop**.
2. In **Settings > Video**, set the Output resolution to 1920x1080 and the FPS to 30.
3. Click **Start Virtual Camera** (the camera cannot be opened until it is started).
4. In the menu, choose **Live stream (camera)**, press **Scan cameras** if needed, pick **OBS Virtual Camera**, Start.

In live mode the reader keeps only the newest frame, so if the models are slower than the stream, frames are skipped
instead of the picture falling behind. R (restart) and F (speed) do nothing, and the run ends ("STREAM ENDED") if
the camera sends no frames for 2 seconds. When OBS loops the video, the jump back to the start is detected as a
scene cut and the tracks are cleared. The console prints the size the camera actually delivers; if it is 640x480,
check the OBS output resolution.

## Keys and mouse (video window)

| Key / mouse | Action |
|---|---|
| Left-click a car, or **Tab** | lock that car as the target (Tab cycles left to right) |
| Right-click or **C** | back to the automatic target |
| Space | pause / resume |
| F | real-time / max speed |
| R | restart the video |
| T | show / hide trails |
| I | show / hide the info panel |
| H | show / hide the help bar |
| Q | back to the menu |

On screen: every car has a box labeled `car #12 0.71` (class, track ID, confidence). A car YOLO has just missed is
drawn dashed as `car #12 lost` for up to a second while the tracker keeps predicting where it is. The target is red
(`TARGET` when chosen automatically, `LOCKED` when you chose it), and says `searching` while it is being predicted.
The FPS counter is green while processing keeps up with the video's frame rate, yellow or red when it falls behind.

## How it works

For every video frame, `tracking_session.py` does this, in order:

1. **Detect** (`detection/`): both YOLO models run on the frame. Their boxes are merged so a car found by both is
   reported once (`duplicate_merging.py`). Overlapping boxes of vehicles are merged; people are only compared with other people. Detections are plain NumPy arrays of shape (N, 6): `x1, y1, x2, y2,
   confidence, class index` (the column names are in `vehicle_types.py`).
2. **Camera motion** (`tracking/camera_motion.py`): corner features on the background (cars masked out) are tracked
   between frames, and a shift/rotation/zoom transform is fitted to them. The same measurements detect hard cuts in
   edited footage, and a cut clears all tracks.
3. **Track** (`tracking/vehicle_tracker.py`): BoT-SORT matches detections to existing tracks. It is given the camera
   motion measured in step 2, so a missed car's predicted box moves with the scene instead of staying put. Weak
   detections (confidence 0.1 to 0.3) can still extend existing tracks. A missed car is remembered for 2.5 seconds
   and, if it reappears, gets its old ID back.
4. **Trails** (`tracking/trails.py`): each track's recent centers, moved by the camera motion so they stay on the
   ground instead of following the screen.
5. **Target** (`tracking/target_selector.py`): automatic mode prefers a track that is large, central, confident and
   long-lived, and sticks with it until that track ends. Click or Tab (`track_picking.py`) locks a track instead.
6. **Draw and input** (`display/`): boxes, IDs, trails, info panel, FPS; then keys and clicks are handled.

## Code layout

Each file does one job. Folders group related files; there are no `__init__.py` files, and imports are absolute
(`from src.detection.vehicle_detector import VehicleDetector`).

```
main.py                         entry point: DPI awareness -> menu -> load models -> session -> save settings -> menu
run.bat                         launches `uv run main.py`
pyproject.toml                  dependencies (CUDA 13 PyTorch build; `lap` is needed by BoT-SORT;
                                `cv2-enumerate-cameras` gives camera names)
videos/  models/  logs/         input footage, model weights (downloaded here), run logs (created on demand)
src/
  project_paths.py              folder and file locations
  video_reader.py               threaded video file decoder, or live camera reader (newest frame only)
  camera_scanner.py             lists cameras by name (OBS Virtual Camera, webcams)
  playback_pacer.py             real-time pacing and frame-interval measurement
  tracking_session.py           the per-frame loop described above (the only file that wires everything together)
  menu/                         the tkinter menu
    tracker_settings.py         TrackerSettings: every option, its default and limits
    settings_storage.py         load/save last_settings.json (bad or missing files fall back to defaults)
    menu_variables.py           tkinter variables behind the fields
    menu_window.py              the menu window
  detection/                    finding vehicles
    vehicle_types.py            VehicleClass, the (N, 6) detection array columns, model class-name mapping
    model_catalog.py            installed / downloadable models, where weights live
    yolo_model.py               ONE YOLO model -> detections
    duplicate_merging.py        removes duplicate boxes between the two models
    vehicle_detector.py         runs both models and merges them
    vram_monitor.py             GPU memory readout
  tracking/                     following the cars
    camera_motion.py            camera motion estimate and hard-cut detection
    track.py                    the Track record
    vehicle_tracker.py          BoT-SORT with our camera motion, plus coasting of missed tracks
    trails.py                   ground-pinned trails
    target_selector.py          automatic target and manual lock
    track_picking.py            click hit-test and Tab cycling
  display/                      everything drawn or shown
    drawing.py                  low-level drawing helpers
    overlay.py                  boxes with IDs, target marker, trails
    hud.py                      FPS counter, info panel, help bar, PAUSED/END banner
    fps_counter.py              smoothed FPS
    display_window.py           the OpenCV window, fit-to-screen, mouse clicks
  run_logging/                  per-run logging
    run_logger.py               the API the session calls; writes the JSON log and console summary
    run_statistics.py           running statistics (FPS, detection, tracking)
    run_history.py              one summary row per run in logs/run_history.csv
    environment_info.py         GPU, CPU and library versions
```

Rules that keep it easy to debug: `menu/` never imports torch (so the menu opens instantly), `tracking/` does not
import from the other folders, and only `main.py` and `tracking_session.py` connect the folders.

## Logs

Every run writes `logs/run_<date>_<time>.json` (video, models, settings and live changes, FPS with the 1% low,
detection time and per-class confidence, tracks created, target changes and scene cuts with timestamps, hardware
and library versions) and appends a row to `logs/run_history.csv` for comparing runs. A short summary is printed when
the run ends. The log is written even if the run crashes.

## Tips

- Slow? Run one model instead of two, or lower the model sizes. For straight-down footage, setting the Main model to "None" usually keeps most cars; setting the Aerial model to "None" misses cars seen from straight above.
- Cars missed in a particular clip? Try lowering the confidence floor (0.05) or raising the model sizes.
- Models are stored in `models/`; copy any Ultralytics detection or OBB `.pt` file there and it appears in the menu.
