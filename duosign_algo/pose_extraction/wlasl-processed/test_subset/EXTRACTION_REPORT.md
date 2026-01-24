# WLASL Test Subset — Pose Extraction Results

**Date**: January 24, 2026  
**Pipeline**: `wlasl_pose_pipeline.py`  
**MediaPipe Version**: 0.10.32 (Tasks API)

---

## Overview

This document describes the test run of the pose extraction pipeline on a random subset of WLASL videos.

## Dataset Selection

### Method
- Selected 10 random classes from `nslt_100.json` using seed 42 for reproducibility
- Selected up to 3 videos per class (train/val/test subsets)
- Total: 30 videos

### Selected Classes

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 3 | before | Temporal sign |
| 13 | no | Common response |
| 14 | thin | Descriptive adjective |
| 17 | yes | Common response |
| 28 | table | Noun/object |
| 31 | woman | Noun/person |
| 35 | can | Modal verb |
| 81 | cheat | Action verb |
| 86 | how | Question word |
| 94 | purple | Color adjective |

---

## Pipeline Configuration

```json
{
  "input_dir": "./wlasl-processed/test_subset/videos",
  "output_dir": "./wlasl-processed/test_subset/poses",
  "num_workers": 1,
  "model_complexity": 1,
  "min_detection_confidence": 0.5,
  "min_tracking_confidence": 0.5,
  "skip_existing": true,
  "log_level": "INFO"
}
```

---

## Results Summary

### Extraction Statistics

| Metric | Value |
|--------|-------|
| Total Videos | 30 |
| Successfully Processed | 30 |
| Failed | 0 |
| Total Frames | 1,883 |
| Processing Time | ~90 seconds |

### Detection Rates

| Component | Average Detection Rate |
|-----------|----------------------|
| **Pose** | 99.7% |
| **Face** | 85.7% |
| **Left Hand** | 28.4% |
| **Right Hand** | 61.0% |

> **Note**: Lower hand detection rates are expected for signs that don't prominently feature hands in the camera view, or when hands are occluded.

---

## Detailed Results

| Video ID | Class | Frames | FPS | Pose% | Face% | Left Hand% | Right Hand% |
|----------|-------|--------|-----|-------|-------|------------|-------------|
| 05730 | before | 31 | 24.0 | 100.0 | 100.0 | 71.0 | 71.0 |
| 05731 | before | 45 | 24.0 | 100.0 | 100.0 | 11.1 | 68.9 |
| 05734 | before | 81 | 30.0 | 98.8 | 98.8 | 0.0 | 77.8 |
| 08935 | can | 58 | 30.0 | 100.0 | 0.0 | 75.9 | 62.1 |
| 08936 | can | 63 | 30.0 | 100.0 | 100.0 | 23.8 | 28.6 |
| 08937 | can | 89 | 30.0 | 98.9 | 98.9 | 43.8 | 42.7 |
| 10147 | cheat | 72 | 25.0 | 100.0 | 100.0 | 18.1 | 38.9 |
| 10158 | cheat | 41 | 30.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 10166 | cheat | 94 | 30.3 | 98.9 | 98.9 | 23.4 | 38.3 |
| 28210 | how | 27 | 30.0 | 100.0 | 100.0 | 96.3 | 96.3 |
| 28214 | how | 81 | 30.3 | 98.8 | 98.8 | 38.3 | 35.8 |
| 38544 | no | 83 | 30.3 | 98.8 | 0.0 | 1.2 | 30.1 |
| 45436 | purple | 92 | 30.0 | 98.9 | 98.9 | 6.5 | 59.8 |
| 45438 | purple | 34 | 30.0 | 100.0 | 100.0 | 0.0 | 100.0 |
| 45443 | purple | 77 | 30.4 | 98.7 | 98.7 | 1.3 | 42.9 |
| 56556 | table | 72 | 30.0 | 100.0 | 0.0 | 54.2 | 48.6 |
| 56557 | table | 39 | 24.0 | 100.0 | 100.0 | 25.6 | 64.1 |
| 56563 | table | 40 | 30.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 57934 | thin | 90 | 30.0 | 100.0 | 33.3 | 2.2 | 45.6 |
| 57935 | thin | 90 | 30.0 | 100.0 | 45.6 | 50.0 | 37.8 |
| 57937 | thin | 31 | 30.0 | 100.0 | 100.0 | 0.0 | 29.0 |
| 63679 | woman | 69 | 30.4 | 98.6 | 98.6 | 0.0 | 49.3 |
| 64291 | yes | 55 | 30.0 | 100.0 | 100.0 | 0.0 | 100.0 |
| 64292 | yes | 47 | 30.0 | 100.0 | 100.0 | 0.0 | 97.9 |
| 64293 | yes | 53 | 30.0 | 100.0 | 100.0 | 18.9 | 94.3 |
| 66183 | no | 68 | 30.0 | 100.0 | 100.0 | 2.9 | 38.2 |
| 66798 | woman | 49 | 24.0 | 100.0 | 100.0 | 2.0 | 53.1 |
| 66799 | woman | 55 | 24.0 | 100.0 | 100.0 | 3.6 | 38.2 |
| 69370 | how | 90 | 30.0 | 100.0 | 100.0 | 81.1 | 74.4 |
| 69411 | no | 67 | 30.0 | 100.0 | 100.0 | 0.0 | 67.2 |

---

## Output File Structure

```
wlasl-processed/test_subset/
├── selection.json          # Selected videos and classes
├── videos/                 # Input videos (30 files)
│   ├── 05730.mp4
│   ├── 05731.mp4
│   └── ...
└── poses/                  # Extracted pose files
    ├── config.json         # Pipeline configuration
    ├── manifest.jsonl      # Success log
    ├── failures.jsonl      # Failure log
    ├── pipeline.log        # Detailed logs
    ├── summary.json        # Run statistics
    ├── 05730.pose          # Pose data files
    ├── 05731.pose
    └── ...
```

---

## Pose File Format

Each `.pose` file is a NumPy compressed archive (`.npz`) containing:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `landmarks` | (T, 523, 3) | float32 | x, y, z coordinates |
| `confidence` | (T, 523) | float32 | Detection confidence |
| `presence_mask` | (T, 523) | bool | True where detected |
| `fps` | scalar | float | Video frame rate |
| `frame_count` | scalar | int | Number of frames |
| `source_video` | scalar | str | Original video path |
| `landmark_layout` | dict | object | Index ranges |
| `version` | scalar | str | Format version |

### Loading Example

```python
import numpy as np

data = np.load("poses/05730.pose", allow_pickle=True)
landmarks = data["landmarks"]  # Shape: (31, 523, 3)
fps = float(data["fps"])       # 24.0
```

---

## Observations

1. **Pose detection is excellent** (99.7%) — the upper body is reliably detected in all videos
2. **Face detection is strong** (85.7%) — some videos have low face detection when the signer is not facing the camera
3. **Hand detection varies** — left hand (28.4%) vs right hand (61.0%) suggests many signers are right-handed dominant
4. **Zero detection cases** — some frames show 0% for hands, which is expected when hands are out of frame or occluded

---

## Files

- [selection.json](file:///Users/nanaamoako/Desktop/duosign-frontend/duosign_algo/pose_extraction/wlasl-processed/test_subset/selection.json) — Video selection metadata
- [config.json](file:///Users/nanaamoako/Desktop/duosign-frontend/duosign_algo/pose_extraction/wlasl-processed/test_subset/poses/config.json) — Pipeline configuration
- [Pose files](file:///Users/nanaamoako/Desktop/duosign-frontend/duosign_algo/pose_extraction/wlasl-processed/test_subset/poses/) — 30 extracted pose files
