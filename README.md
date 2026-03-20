## eyetest

`eyetest` integrates EllSeg-based iris segmentation with a geometry-based gaze estimator.

### Batch mode

```bash
python main.py batch \
  --left-video /path/to/left_eye.mp4 \
  --right-video /path/to/right_eye.mp4 \
  --output-json outputs/gaze-results.json \
  --eval-on-cpu
```

### Realtime mode

```bash
python main.py realtime --camera-index 0 --eval-on-cpu
```

### Calibration

The project currently ships with default calibration values migrated from the legacy gaze-estimation project.
They are bootstrapping defaults only and are not guaranteed to match your camera, monitor, or physical setup.
