# Design: Real-Time Game Object-Detection & Overlay Tool

**Date:** 2026-07-16
**Status:** Approved design, pending implementation plan

## Summary

A Python computer-vision tool that takes frames from a **single-player game** —
either a live screen capture or a recorded video — runs real-time object
detection on them (enemies / game elements), and renders a live **mirror-window
overlay** with bounding boxes, labels, and a small HUD. Detections can
optionally be logged to disk to produce post-session analytics (heatmaps,
timelines, auto-highlights).

## Scope & boundaries

**In scope:**
- Live screen / capture-card input and recorded-video input, interchangeable.
- Real-time detection using a YOLO model (pretrained first, custom-trained later).
- A mirror-window overlay (boxes + labels + HUD).
- Optional structured detection log for offline analytics.
- A custom-model training workflow for game-specific classes.

**Out of scope / explicit non-goals:**
- **No multiplayer/online use.** This tool targets single-player games only.
  Running detection against live competitive online matches (ESP-style overlays)
  is out of scope and will not be built.
- **No automated input control** (mouse/keyboard auto-aim). Detections drive
  *display and analytics only*, never game input.

## Technology choices

- **Language:** Python. Rationale: neural-net inference runs as native CUDA
  kernels regardless of the calling language, so at 30-60 fps Python's
  per-frame overhead is negligible; the Python CV/ML ecosystem is the
  industry standard and vastly faster to build in. C++ is a possible *later*
  optimization for a single hot module, not a starting point.
- **Hardware:** NVIDIA GPU (CUDA) for real-time inference.
- **Key libraries:**
  - `dxcam` / `bettercam` — fast Windows desktop-duplication screen capture.
  - `ultralytics` (YOLO) — detection + built-in tracking (ByteTrack).
  - `opencv-python` — video-file/device input, drawing, overlay window.
  - `numpy` — frame/array handling.

## Architecture

One-directional real-time pipeline; overlay and logger are independent
consumers of the detection stream.

```
                                          +--> Overlay renderer (mirror window)
Frame Source --> Detection --> Tracking --+
                                          +--> File logger (optional)
                                                     |
                                                     v
                                          Analysis & visualization (offline)
```

### Modules

1. **Frame source (`sources/`)** — a common interface that yields
   `(frame, timestamp, is_live)`. Implementations:
   - `ScreenSource` — `dxcam` capture of a monitor/window region (live, droppable).
   - `VideoFileSource` — `cv2.VideoCapture` over a file; supports `realtime`
     (paced to the recording's fps) and `fast` (as fast as the GPU allows)
     modes; complete (never drops frames).
   - `DeviceSource` — `cv2.VideoCapture` over a capture-card device index
     (live, droppable).
   Selected at startup via `--source {screen,video,device}` (+ `--path` / `--index`).

2. **Detection (`detection/`)** — loads a YOLO model and runs it on each frame,
   returning a list of `Detection{class, confidence, box}`. Model path is
   configurable so Phase 1 (pretrained) and Phase 2 (custom) use the same code.

3. **Tracking (`tracking/`)** — assigns stable IDs across frames via
   Ultralytics' built-in ByteTrack, so objects keep a consistent ID over time
   (needed for paths, dwell time, and stable overlay labels).

4. **Overlay renderer (`overlay/`)** — draws boxes + class/ID/confidence labels
   onto the frame and shows it in an OpenCV mirror window, plus a corner HUD
   (current detection count, FPS). **Mirror-window approach** chosen over a
   transparent click-through overlay for v1: far simpler, works over any game
   including fullscreen, and touches nothing in the game itself.

5. **File logger (`logging/`)** — optional; appends each frame's detections to a
   timestamped JSONL/CSV log. Off by default; enabled with `--log`.

6. **Analysis & visualization (`analysis/`)** — offline scripts that read a log
   to produce screen-space heatmaps, engagement timelines, multi-enemy flags,
   and auto-highlight clip lists. Not part of the real-time path.

7. **Training pipeline (`training/`)** — Phase 2 workflow: extract frames from
   recordings, label them (Roboflow / CVAT / LabelImg), train a custom YOLO
   model via Ultralytics, evaluate (mAP / precision / recall), iterate on
   failure cases, export weights for the Detection module.

### Live vs. recorded handling

- **Timing:** live = wall-clock, arrives in real time; video = `frame_number /
  fps`, with selectable realtime/fast playback.
- **Frame dropping:** live sources may drop stale frames to stay current; video
  sources process every frame.
- **End condition:** video ends at EOF; live runs until quit.
- Downstream modules are identical for both — only the source differs.

## Phasing

- **Phase 1 — Prove the pipeline.** Frame source (start with `VideoFileSource`
  for reproducible dev, then `ScreenSource`) → **pretrained** YOLO → tracking →
  mirror overlay + HUD. Detects generic COCO classes only; goal is a working,
  real-time end-to-end loop at good FPS.
- **Phase 2 — Custom detection.** Collect + label game frames, train a custom
  YOLO model, swap it into the Detection module. Now detects the game's actual
  enemies/elements. Expect an iterative label → train → review loop.
- **Phase 3 (optional) — Analytics.** Enable the logger and build the offline
  analysis/visualization scripts.
- **Later (optional) — C++ stretch.** Rewrite one hot module (capture or
  inference) in C++/TensorRT as a focused learning exercise, benchmarked
  against the working Python version.

## Error handling

- Source open failure (missing file, invalid monitor/device) → clear error and
  exit before the loop starts.
- Missing/failed model load → explicit error naming the model path.
- Per-frame detection exceptions → logged and skipped so one bad frame doesn't
  crash a live session.
- Video EOF → clean shutdown; live-source read failure → retry with backoff,
  then exit if persistent.

## Testing strategy

- **Frame sources:** unit-test `VideoFileSource` against a short fixture clip
  (deterministic frame count, timestamps, ordering; fast vs realtime pacing).
- **Detection:** test the frame→`Detection[]` mapping against a fixed image with
  known objects using the pretrained model.
- **Overlay:** test box/label drawing produces expected pixels on a synthetic
  frame (no live window needed).
- **Logger:** assert log rows match input detections.
- **End-to-end:** run the full pipeline on the fixture clip and assert it
  completes and emits a non-empty log.

## Open questions / deferred

- Specific game and class list for Phase 2 (drives the dataset, not the code).
- Whether to add a transparent click-through overlay after v1.
- Minimap reading for world-space (vs screen-space) positions — a separate,
  harder CV task; deferred.
