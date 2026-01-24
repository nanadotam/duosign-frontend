# Pose Extraction Methodology - Executive Summary

## Quick Reference Guide

This document provides a concise summary of the video-to-pose extraction methodology. For complete details, see `POSE_EXTRACTION_METHODOLOGY_ANALYSIS.md`.

---

## Pipeline Overview

```
Video → Keyframe Extraction → Landmark Detection → Structured Output
```

### Stage 1: Motion-Based Keyframe Extraction
- **Input**: Raw video files
- **Method**: Dense optical flow (Farneback)
- **Output**: Keyframes representing stable poses
- **Reduction**: 70-90% fewer frames

### Stage 2: Landmark Detection
- **Hands**: 21 landmarks × 2 hands (42 total)
- **Pose**: 33 body landmarks
- **Face**: 468+ facial landmarks (optional)
- **Tool**: MediaPipe (Holistic or separate models)

### Stage 3: Structured Output
- **Format**: JSON with temporal synchronization
- **Schema**: Frame-indexed landmark sequences
- **Coordinates**: Normalized (0-1) or pixel-based

---

## Core Algorithm: Keyframe Extraction

### Step 1: Motion Score Calculation
```python
# Dense optical flow between consecutive frames
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, curr_gray, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.1, flags=0
)

# Motion magnitude (mean across frame)
magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
motion_score = np.mean(magnitude)
```

### Step 2: Active Region Detection
```python
# Find motion peaks (high activity)
peaks, _ = find_peaks(motion_scores, prominence=0.5)

# Define active region with buffer
active_start = max(0, first_peak - 5)
active_end = min(len(motion_scores) - 1, last_peak + 5)
```

### Step 3: Stable Pose Detection
```python
# Find troughs (low motion = stable poses)
troughs, _ = find_peaks(-motion_scores, distance=5)

# Separate active/inactive regions
active_troughs = [t for t in troughs if active_start <= t <= active_end]
```

### Step 4: Sharpness Refinement
```python
# Select sharpest frame in window around each trough
for trough_idx in troughs:
    window = frames[trough_idx-1 : trough_idx+2]  # 3-frame window
    sharpness_scores = [cv2.Laplacian(f, cv2.CV_64F).var() for f in window]
    best_frame = window[np.argmax(sharpness_scores)]
```

---

## Landmark Extraction

### MediaPipe Holistic (Recommended)
```python
import mediapipe as mp

holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,     # Video mode with tracking
    model_complexity=1,           # Balance accuracy/speed
    smooth_landmarks=True,        # Temporal smoothing
    refine_face_landmarks=True,   # 468 → 478 with iris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Process frame
results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Extract landmarks
left_hand = results.left_hand_landmarks    # 21 landmarks
right_hand = results.right_hand_landmarks  # 21 landmarks
pose = results.pose_landmarks              # 33 landmarks
face = results.face_landmarks              # 468+ landmarks
```

### Landmark Structure
```python
{
    "frame_index": 42,
    "timestamp_ms": 1400.0,
    "hand_landmarks": {
        "left": [
            {"x": 0.523, "y": 0.612, "z": -0.045, "label": "WRIST"},
            {"x": 0.531, "y": 0.589, "z": -0.052, "label": "THUMB_CMC"},
            # ... 19 more landmarks
        ],
        "right": [...]
    },
    "pose_landmarks": [
        {"x": 0.501, "y": 0.234, "z": 0.012, "visibility": 0.98, "label": "NOSE"},
        {"x": 0.423, "y": 0.456, "z": -0.123, "visibility": 0.95, "label": "LEFT_SHOULDER"},
        # ... 31 more landmarks
    ],
    "face_landmarks": [
        {"x": 0.498, "y": 0.245, "z": 0.001, "index": 0},
        # ... 467 more landmarks
    ]
}
```

---

## Key Parameters

### Keyframe Extraction
| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `peak_prominence` | 0.5 | 0.0-1.0 | Motion peak sensitivity |
| `active_region_buffer` | 5 | 0-10 | Context frames |
| `trough_distance` | 5 | 1-15 | Pose separation |
| `trough_window_size` | 3 | 1-7 (odd) | Sharpness window |

### MediaPipe Detection
| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `min_detection_confidence` | 0.5 | 0.0-1.0 | Detection threshold |
| `min_tracking_confidence` | 0.5 | 0.0-1.0 | Tracking threshold |
| `model_complexity` | 1 | 0-2 | Accuracy vs. speed |

---

## Coordinate Systems

### Normalized (Recommended)
- **Range**: 0.0 to 1.0
- **Advantages**: Resolution-independent, animation-friendly
- **Use Case**: Cross-video comparison, 3D avatar animation

```python
# Normalized coordinates
x_norm = landmark.x  # 0.0 to 1.0 (left to right)
y_norm = landmark.y  # 0.0 to 1.0 (top to bottom)
z_norm = landmark.z  # Depth relative to reference point
```

### Pixel-Based (Alternative)
- **Range**: 0 to image_width/height
- **Advantages**: Direct overlay on images, annotation-friendly
- **Use Case**: Visualization, annotation pipelines

```python
# Pixel coordinates
x_pixel = int(landmark.x * image_width)
y_pixel = int(landmark.y * image_height)
z_pixel = landmark.z  # Still normalized depth
```

---

## Implementation Comparison

| Aspect | NVIDIA | DuoSign | Recommendation |
|--------|--------|---------|----------------|
| **MediaPipe Model** | Separate (Hands, Pose, Face) | Holistic (unified) | **Holistic** (efficient) |
| **Image Mode** | Static (per-frame) | Video (tracking) | **Video** (coherent) |
| **Coordinates** | Pixel (int) | Normalized (float) | **Normalized** (flexible) |
| **Output** | Distributed files | Unified JSON | **Depends on use case** |
| **Architecture** | Production (S3) | Research (local) | **Depends on scale** |

---

## Quick Start: MediaPipe Implementation

```python
import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks

class PoseExtractor:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_keyframes(self, video_path):
        """Extract keyframes using motion analysis."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        motion_scores = []
        
        # Read frames and calculate motion
        ret, prev_frame = cap.read()
        frames.append(prev_frame)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.1, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_scores.append(np.mean(magnitude))
            prev_gray = gray
        
        cap.release()
        
        # Find stable poses (troughs)
        motion_array = np.array(motion_scores)
        troughs, _ = find_peaks(-motion_array, distance=5)
        
        # Refine with sharpness
        keyframe_indices = []
        for idx in troughs:
            start = max(0, idx - 1)
            end = min(len(frames), idx + 2)
            window = frames[start:end]
            
            sharpness = [cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() 
                        for f in window]
            best_idx = start + np.argmax(sharpness)
            if best_idx not in keyframe_indices:
                keyframe_indices.append(best_idx)
        
        keyframes = [frames[i] for i in sorted(keyframe_indices)]
        return keyframes, keyframe_indices
    
    def extract_landmarks(self, keyframes):
        """Extract landmarks from keyframes."""
        results = []
        
        for frame in keyframes:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection = self.holistic.process(rgb)
            
            frame_data = {
                "hand_landmarks": {},
                "pose_landmarks": [],
                "face_landmarks": []
            }
            
            # Hands
            if detection.left_hand_landmarks:
                frame_data["hand_landmarks"]["left"] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in detection.left_hand_landmarks.landmark
                ]
            
            if detection.right_hand_landmarks:
                frame_data["hand_landmarks"]["right"] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in detection.right_hand_landmarks.landmark
                ]
            
            # Pose
            if detection.pose_landmarks:
                frame_data["pose_landmarks"] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                    for lm in detection.pose_landmarks.landmark
                ]
            
            # Face
            if detection.face_landmarks:
                frame_data["face_landmarks"] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in detection.face_landmarks.landmark
                ]
            
            results.append(frame_data)
        
        return results
    
    def process_video(self, video_path):
        """Full pipeline: keyframes + landmarks."""
        keyframes, indices = self.extract_keyframes(video_path)
        landmarks = self.extract_landmarks(keyframes)
        
        return {
            "keyframe_indices": indices,
            "landmarks": landmarks
        }

# Usage
extractor = PoseExtractor()
result = extractor.process_video("sign_video.mp4")
print(f"Extracted {len(result['landmarks'])} keyframes")
```

---

## Quality Metrics

### Detection Rates
```python
def calculate_quality(landmarks):
    total_frames = len(landmarks)
    hands_detected = sum(1 for f in landmarks if f["hand_landmarks"])
    pose_detected = sum(1 for f in landmarks if f["pose_landmarks"])
    face_detected = sum(1 for f in landmarks if f["face_landmarks"])
    
    return {
        "hands_rate": hands_detected / total_frames,
        "pose_rate": pose_detected / total_frames,
        "face_rate": face_detected / total_frames,
    }
```

### Expected Rates (Good Quality Video)
- **Hands**: 80-95%
- **Pose**: 90-98%
- **Face**: 85-95%

---

## Common Issues & Solutions

### Issue: Too Many Keyframes
- **Cause**: `peak_prominence` too low
- **Solution**: Increase to 0.5-0.7

### Issue: Missing Important Poses
- **Cause**: `peak_prominence` too high
- **Solution**: Decrease to 0.2-0.3

### Issue: Blurry Keyframes
- **Cause**: `trough_window_size` too small
- **Solution**: Increase to 5 or 7

### Issue: Low Hand Detection
- **Cause**: Poor lighting, occlusions, or low confidence threshold
- **Solution**: Improve lighting, lower `min_detection_confidence` to 0.3

### Issue: Jittery Landmarks
- **Cause**: `smooth_landmarks` disabled or `min_tracking_confidence` too high
- **Solution**: Enable smoothing, lower tracking confidence to 0.3

---

## Validation Checklist

- [ ] Keyframe count is 10-30% of original frames
- [ ] Active region captures entire sign
- [ ] Stable poses are clearly visible (no motion blur)
- [ ] Hand detection rate > 80%
- [ ] Pose detection rate > 90%
- [ ] Landmarks align with visual features
- [ ] Temporal sequence is coherent
- [ ] Output format matches schema

---

## Next Steps

1. **Implement** the `MediaPipePoseExtractor` class
2. **Test** on sample videos from WLASL dataset
3. **Compare** outputs with reference implementation
4. **Tune** parameters for your specific use case
5. **Validate** with downstream tasks (animation, recognition)

---

## Resources

- **Full Analysis**: `POSE_EXTRACTION_METHODOLOGY_ANALYSIS.md`
- **MediaPipe Docs**: https://google.github.io/mediapipe/
- **WLASL Dataset**: https://github.com/dxli94/WLASL
- **Reference Code**: 
  - DuoSign: `/duosign_pipeline/`
  - NVIDIA: `/ASL Developer Community/`

---

**Last Updated**: 2026-01-23  
**Status**: Ready for implementation
