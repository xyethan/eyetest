# Eyetest Integration Design

**Date:** 2026-03-20

**Goal**

Integrate the existing EllSeg iris-segmentation pipeline and the reusable geometry-based gaze-estimation pipeline into the `eyetest` project as one maintainable Python package. The first deliverable must support batch processing and then support realtime camera input using the same internal pipeline.

## Scope

This integration will:

- use EllSeg as the segmentation and ellipse-generation backend
- consume both left-eye and right-eye iris ellipse outputs
- reuse the geometry-based gaze-estimation core from `gazeVisualization-main (3)`
- ship with default calibration values copied from the original gaze project
- support two entry modes:
  - batch processing for videos or image sequences
  - realtime camera processing

This integration will not:

- migrate EllSeg training code
- migrate old Tk UI code
- migrate old iris-segmentation code from `gazeVisualization-main (3)`
- claim that bundled calibration values are valid for the user's hardware setup

## Source Projects

### EllSeg source

Accessible at [EllSeg](/mnt/c/Project/EllSeg).

Relevant files confirmed in the source project:

- [evaluate_ellseg.py](/mnt/c/Project/EllSeg/evaluate_ellseg.py)
- [helperfunctions.py](/mnt/c/Project/EllSeg/helperfunctions.py)
- [loss.py](/mnt/c/Project/EllSeg/loss.py)
- [utils.py](/mnt/c/Project/EllSeg/utils.py)
- [modelSummary.py](/mnt/c/Project/EllSeg/modelSummary.py)
- [models/RITnet_v3.py](/mnt/c/Project/EllSeg/models/RITnet_v3.py)
- [weights/all.git_ok](/mnt/c/Project/EllSeg/weights/all.git_ok)

### Gaze-estimation source

Accessible at [gazeVisualization-main (3)](/home/ethansky/demo/gazeVisualization-main%20%283%29/gazeVisualization-main).

Relevant files confirmed in the source project:

- [interact1.py](/home/ethansky/demo/gazeVisualization-main%20%283%29/gazeVisualization-main/interact1.py)
- [submodule.py](/home/ethansky/demo/gazeVisualization-main%20%283%29/gazeVisualization-main/submodule.py)
- [eyetracking_utils.py](/home/ethansky/demo/gazeVisualization-main%20%283%29/gazeVisualization-main/eyetracking_utils.py)
- [calipara.txt](/home/ethansky/demo/gazeVisualization-main%20%283%29/gazeVisualization-main/calipara.txt)

## Architecture

The integrated project will be organized as a pipeline:

`input source -> EllSeg inference -> ellipse adapter -> gaze estimation -> output writer`

The implementation will follow a package-first structure under `src/eyetest` instead of carrying over the original script-heavy layouts.

## Proposed Project Layout

```text
eyetest/
  src/eyetest/
    cli.py
    config.py
    models/types.py
    inputs/
      video.py
      camera.py
    segmentation/
      ellseg_preprocess.py
      ellseg_model.py
      ellseg_pipeline.py
    adapters/
      ellipse_adapter.py
    gaze/
      geometry.py
      calibration.py
      estimator.py
    outputs/
      overlay.py
      writers.py
    pipelines/
      batch_pipeline.py
      realtime_pipeline.py
  configs/
    default.yaml
    calibration.default.yaml
  tests/
```

## Module Responsibilities

### `segmentation`

This layer wraps EllSeg inference only.

It will be extracted primarily from [evaluate_ellseg.py](/mnt/c/Project/EllSeg/evaluate_ellseg.py):

- `preprocess_frame`
- `evaluate_ellseg_on_image`
- `rescale_to_original`

It will output normalized in-memory results, not write videos directly.

### `adapters`

This layer converts EllSeg ellipse output into the gaze module's expected format.

Reason:

- EllSeg returns 5-value ellipse arrays
- the gaze code expects ellipse parameters in a specific geometry-oriented convention
- this conversion must be isolated so the segmentation backend can be replaced later without touching gaze logic

### `gaze`

This layer contains only reusable geometry and estimation logic.

Primary functions to migrate:

- `chos_uniq_norandcen` from [interact1.py](/home/ethansky/demo/gazeVisualization-main%20%283%29/gazeVisualization-main/interact1.py)
- `get_los` from [interact1.py](/home/ethansky/demo/gazeVisualization-main%20%283%29/gazeVisualization-main/interact1.py)
- `gaze_estimation` from [interact1.py](/home/ethansky/demo/gazeVisualization-main%20%283%29/gazeVisualization-main/interact1.py)
- supporting math from [submodule.py](/home/ethansky/demo/gazeVisualization-main%20%283%29/gazeVisualization-main/submodule.py)
- supporting math from [eyetracking_utils.py](/home/ethansky/demo/gazeVisualization-main%20%283%29/gazeVisualization-main/eyetracking_utils.py)

UI code, camera widgets, and legacy segmentation code will not be migrated.

### `pipelines`

These modules will orchestrate the end-to-end flow.

- `batch_pipeline.py`: video files and image sequences
- `realtime_pipeline.py`: camera stream

Both pipelines must share the same segmentation, adapter, gaze, and output layers.

## Input and Output Contract

The first version will assume both eyes are available for each frame.

Core internal types:

- `Ellipse2D`
- `FrameSegmentation`
- `GazeEstimate`
- `CalibrationParams`

Per-frame processing must support:

- left iris ellipse
- right iris ellipse
- optional left/right pupil ellipses
- frame index and timestamp metadata

Output artifacts for batch mode:

- overlay video
- per-frame structured results
- optional CSV summary

Output artifacts for realtime mode:

- onscreen visualization
- optional recording of overlays
- optional structured per-frame export

## Calibration Strategy

The user currently has no personal calibration results.

The first version will therefore ship with copied default values from the original gaze project:

- screen plane corners from [interact1.py:123](/home/ethansky/demo/gazeVisualization-main%20%283%29/gazeVisualization-main/interact1.py#L123)
- camera parameters from [interact1.py:148](/home/ethansky/demo/gazeVisualization-main%20%283%29/gazeVisualization-main/interact1.py#L148)
- radius and kappa values from [calipara.txt](/home/ethansky/demo/gazeVisualization-main%20%283%29/gazeVisualization-main/calipara.txt)

These values must be stored as configuration, not hardcoded inside estimator logic.

Important limitation:

These are default bootstrapping values only. They are not guaranteed to match the user's hardware, camera placement, or screen geometry.

## Minimal EllSeg Dependency Set

The integration does not need the whole EllSeg training stack, but it does need the minimum inference stack:

- [evaluate_ellseg.py](/mnt/c/Project/EllSeg/evaluate_ellseg.py)
- [helperfunctions.py](/mnt/c/Project/EllSeg/helperfunctions.py)
- [loss.py](/mnt/c/Project/EllSeg/loss.py)
- [utils.py](/mnt/c/Project/EllSeg/utils.py)
- [modelSummary.py](/mnt/c/Project/EllSeg/modelSummary.py)
- [models/RITnet_v3.py](/mnt/c/Project/EllSeg/models/RITnet_v3.py)
- [weights/all.git_ok](/mnt/c/Project/EllSeg/weights/all.git_ok)

If additional direct imports are discovered during extraction, they should be copied only if they are required for inference.

## Migration Rules

### Migrate

- inference-only EllSeg logic
- ellipse fitting and ellipse output formatting needed for inference
- geometry-based gaze estimation core
- calibration loading
- output rendering and persistence

### Do not migrate

- training scripts
- dataset generation code
- legacy GUI code
- old gaze project segmentation code
- unrelated YOLO eye-detection code
- hardcoded Windows paths

## Known Risks

- EllSeg ellipse format may need explicit normalization before feeding gaze estimation
- the original gaze project contains device-specific defaults and at least one suspicious left/right reuse bug in the estimator path
- default calibration values may let the pipeline run while still producing inaccurate gaze points on the user's setup

## Validation Plan

Validation will happen in this order:

1. unit tests for config loading and ellipse format conversion
2. unit tests for gaze geometry entry points using fixed ellipse fixtures
3. integration test for `Frame -> segmentation result -> gaze result`
4. batch-mode smoke test on a short sample input
5. realtime-mode smoke test after batch mode is stable

## Recommended Implementation Sequence

1. Create package structure in `src/eyetest`
2. Add config loading and calibration defaults
3. Extract EllSeg inference into reusable modules
4. Extract gaze geometry into reusable modules
5. Implement the ellipse adapter
6. Implement batch pipeline
7. Implement realtime pipeline
8. Add tests and smoke verification

## Decision Summary

The recommended approach remains:

- implement the first version in an A-to-B transition style
- start from the existing EllSeg inference script
- extract the gaze geometry core from the old gaze project
- land the result in a clean `eyetest` package structure
- use original calibration values only as configurable defaults
