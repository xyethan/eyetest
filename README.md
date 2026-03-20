## eyetest

`eyetest` integrates EllSeg-based iris segmentation with a geometry-based gaze estimator.

### Batch mode

```bash
python main.py batch \
  --face-video /path/to/face_video.mp4 \
  --output-json outputs/gaze-results.json \
  --output-video outputs/gaze-overlay.mp4 \
  --output-overlay-video outputs/gaze-side-by-side.mp4 \
  --eval-on-cpu
```

`main.py` now bootstraps the local `src` package path, so no manual `PYTHONPATH` export is required.

`batch` now accepts a single face video. The system extracts left/right eye crops per frame, marks missing-eye frames as invalid, and keeps processing.

`--output-video` writes the screen-only gaze map video. `--output-overlay-video` writes the side-by-side extracted eye video with ellipse overlays, a fused gaze inset, and a status bar.

### Realtime mode

```bash
python main.py realtime --camera-index 0 --eval-on-cpu
```

### Calibration

The project currently ships with default calibration values migrated from the legacy gaze-estimation project.
They are bootstrapping defaults only and are not guaranteed to match your camera, monitor, or physical setup.
