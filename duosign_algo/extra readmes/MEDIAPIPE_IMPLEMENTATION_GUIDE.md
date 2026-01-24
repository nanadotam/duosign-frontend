# MediaPipe Pose Extraction Implementation Guide
## Quick Reference for Faithful Replication

**Based on:** spoken-to-signed-translation codebase analysis  
**Target:** MediaPipe-based reimplementation  
**Compatibility:** Downstream model training

---

## TL;DR — Core Requirements

```python
# Landmark Configuration
POSE_LANDMARKS = 13      # Upper body only (reduced from 33)
FACE_LANDMARKS = 468     # Full face mesh
LEFT_HAND = 21           # Full hand
RIGHT_HAND = 21          # Full hand
TOTAL = 523 landmarks

# Processing Pipeline
Video → MediaPipe Holistic → Reduce → Align Wrists → Normalize → Trim → Smooth → Output

# Key Parameters
FPS_TARGET = 15-30
SMOOTHING_WINDOW = 3
SMOOTHING_ORDER = 1
PADDING_SECONDS = 0.2
STITCHING_WINDOW = 0.3
```

---

## 1. MediaPipe Configuration

```python
import mediapipe as mp

holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,           # Video mode
    model_complexity=2,                 # Highest accuracy
    smooth_landmarks=False,             # Apply custom smoothing instead
    enable_segmentation=False,          # Not needed
    refine_face_landmarks=True,         # 468-point face mesh
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

---

## 2. Landmark Extraction Order

**CRITICAL:** Maintain this exact order for compatibility

```python
# Index ranges (after reduction)
POSE_RANGE = 0:13          # Upper body
FACE_RANGE = 13:481        # 468 face points
LEFT_HAND_RANGE = 481:502  # 21 hand points
RIGHT_HAND_RANGE = 502:523 # 21 hand points
```

### Upper Body Pose Indices (from MediaPipe Pose 33-point model)

```python
UPPER_BODY_INDICES = [
    0,   # Nose
    11,  # Left shoulder
    12,  # Right shoulder
    13,  # Left elbow
    14,  # Right elbow
    15,  # Left wrist
    16,  # Right wrist
    # Optional for head orientation:
    # 1, 2, 3, 4  # Left eye, right eye, left ear, right ear
]
```

---

## 3. Processing Pipeline (Step-by-Step)

### Step 1: Extract Raw Landmarks

```python
def extract_frame(frame, holistic_model):
    """Extract landmarks from single frame."""
    results = holistic_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    landmarks = np.zeros((523, 3))
    confidence = np.zeros(523)
    
    # Pose (13 points)
    if results.pose_landmarks:
        for i, idx in enumerate(UPPER_BODY_INDICES):
            lm = results.pose_landmarks.landmark[idx]
            landmarks[i] = [lm.x, lm.y, lm.z]
            confidence[i] = lm.visibility
    
    # Face (468 points)
    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            landmarks[13 + i] = [lm.x, lm.y, lm.z]
            confidence[13 + i] = 1.0
    
    # Left hand (21 points)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            landmarks[481 + i] = [lm.x, lm.y, lm.z]
            confidence[481 + i] = 1.0
    
    # Right hand (21 points)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            landmarks[502 + i] = [lm.x, lm.y, lm.z]
            confidence[502 + i] = 1.0
    
    return landmarks, confidence
```

### Step 2: Align Wrists

```python
def align_wrists(pose_data, confidence):
    """Align body wrists with hand wrists."""
    BODY_LEFT_WRIST = 5   # Index in reduced pose
    BODY_RIGHT_WRIST = 6
    HAND_LEFT_WRIST = 481  # First point of left hand
    HAND_RIGHT_WRIST = 502 # First point of right hand
    
    for frame in range(len(pose_data)):
        # Left wrist
        if confidence[frame, HAND_LEFT_WRIST] > 0:
            pose_data[frame, BODY_LEFT_WRIST] = pose_data[frame, HAND_LEFT_WRIST]
        
        # Right wrist
        if confidence[frame, HAND_RIGHT_WRIST] > 0:
            pose_data[frame, BODY_RIGHT_WRIST] = pose_data[frame, HAND_RIGHT_WRIST]
    
    return pose_data
```

### Step 3: Normalize Pose

```python
def normalize_pose(pose_data):
    """Center and scale pose."""
    LEFT_SHOULDER = 1
    RIGHT_SHOULDER = 2
    
    # Calculate shoulder midpoint and width per frame
    left_shoulder = pose_data[:, LEFT_SHOULDER, :]
    right_shoulder = pose_data[:, RIGHT_SHOULDER, :]
    
    midpoint = (left_shoulder + right_shoulder) / 2
    width = np.linalg.norm(right_shoulder - left_shoulder, axis=1, keepdims=True)
    
    # Center and scale
    centered = pose_data - midpoint[:, np.newaxis, :]
    normalized = centered / width[:, np.newaxis, np.newaxis]
    
    return normalized
```

### Step 4: Trim to Active Signing

```python
def trim_pose(pose_data, confidence):
    """Trim to frames where hands are active."""
    LEFT_WRIST = 5
    LEFT_ELBOW = 3
    RIGHT_WRIST = 6
    RIGHT_ELBOW = 4
    
    boundaries = []
    
    for wrist_idx, elbow_idx in [(LEFT_WRIST, LEFT_ELBOW), (RIGHT_WRIST, RIGHT_ELBOW)]:
        # Find where wrist exists
        exists = confidence[:, wrist_idx] > 0
        if not exists.any():
            continue
        
        first_exist = np.argmax(exists)
        last_exist = len(exists) - np.argmax(exists[::-1])
        
        # Find where wrist is above elbow (Y decreases upward)
        wrist_y = pose_data[:, wrist_idx, 1]
        elbow_y = pose_data[:, elbow_idx, 1]
        above = wrist_y < elbow_y
        
        if not above.any():
            continue
        
        first_active = np.argmax(above)
        last_active = len(above) - np.argmax(above[::-1])
        
        # Combine with 5-frame buffer
        start = max(first_exist, first_active - 5)
        end = min(last_exist, last_active + 5)
        boundaries.append((start, end))
    
    if not boundaries:
        return pose_data, confidence
    
    # Take union
    start = min(b[0] for b in boundaries)
    end = max(b[1] for b in boundaries)
    
    return pose_data[start:end], confidence[start:end]
```

### Step 5: Smooth Motion

```python
from scipy.signal import savgol_filter

def smooth_pose(pose_data):
    """Apply Savitzky-Golay filter (skip face)."""
    FACE_START = 13
    FACE_END = 481
    
    frames, landmarks, dims = pose_data.shape
    
    for lm in range(landmarks):
        # Skip face landmarks
        if FACE_START <= lm < FACE_END:
            continue
        
        for d in range(dims):
            pose_data[:, lm, d] = savgol_filter(
                pose_data[:, lm, d],
                window_length=3,
                polyorder=1
            )
    
    return pose_data
```

---

## 4. Concatenation (for Multi-Sign Sequences)

```python
from scipy.spatial.distance import cdist

def find_stitch_point(pose1, pose2, window_sec=0.3, fps=25):
    """Find optimal connection point via L2 distance."""
    window_frames = int(window_sec * fps)
    
    # Get windows
    p1_window = min(window_frames, int(len(pose1) * 0.3))
    p2_window = min(window_frames, int(len(pose2) * 0.3))
    
    last_frames = pose1[-p1_window:].reshape(p1_window, -1)
    first_frames = pose2[:p2_window].reshape(p2_window, -1)
    
    # Find minimum L2 distance
    distances = cdist(last_frames, first_frames)
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    
    return len(pose1) - p1_window + min_idx[0], min_idx[1]

def concatenate_poses(pose_list, padding_sec=0.2, fps=25):
    """Concatenate with optimal stitching."""
    if len(pose_list) == 1:
        return pose_list[0]
    
    padding_frames = int(padding_sec * fps)
    padding = np.zeros((padding_frames, pose_list[0].shape[1], 3))
    
    result = pose_list[0]
    
    for next_pose in pose_list[1:]:
        trim1, trim2 = find_stitch_point(result, next_pose, fps=fps)
        result = np.concatenate([
            result[:trim1],
            padding,
            next_pose[trim2:]
        ])
    
    return result
```

---

## 5. Complete Pipeline Function

```python
def video_to_pose(video_path):
    """Complete extraction pipeline."""
    # Initialize MediaPipe
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames_data = []
    frames_conf = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks, confidence = extract_frame(frame, holistic)
        frames_data.append(landmarks)
        frames_conf.append(confidence)
    
    cap.release()
    holistic.close()
    
    pose_data = np.array(frames_data)
    confidence = np.array(frames_conf)
    
    # Process
    pose_data = align_wrists(pose_data, confidence)
    pose_data = normalize_pose(pose_data)
    pose_data, confidence = trim_pose(pose_data, confidence)
    pose_data = smooth_pose(pose_data)
    
    return pose_data, confidence, fps
```

---

## 6. Validation Checklist

```python
def validate_output(pose_data, confidence):
    """Ensure output matches reference format."""
    
    # Shape checks
    assert pose_data.shape[1] == 523, f"Expected 523 landmarks, got {pose_data.shape[1]}"
    assert pose_data.shape[2] == 3, f"Expected 3D coordinates, got {pose_data.shape[2]}"
    assert confidence.shape == pose_data.shape[:2], "Confidence shape mismatch"
    
    # Normalization checks
    mean = np.abs(pose_data.mean())
    std = pose_data.std()
    assert mean < 0.5, f"Pose not centered (mean={mean:.3f})"
    assert 0.3 < std < 3.0, f"Pose not normalized (std={std:.3f})"
    
    # Confidence checks
    assert (confidence >= 0).all() and (confidence <= 1).all(), "Invalid confidence values"
    
    print("✓ All validation checks passed")
```

---

## 7. Common Pitfalls

### ❌ DON'T:
- Smooth face landmarks (degrades expressions)
- Use all 33 body landmarks (incompatible)
- Apply different normalization (breaks compatibility)
- Change landmark order (breaks model input)
- Skip wrist alignment (causes spatial inconsistency)

### ✅ DO:
- Use MediaPipe Holistic (not separate Pose/Hands/Face)
- Reduce to 13 upper-body pose landmarks
- Skip face in Savitzky-Golay smoothing
- Maintain exact landmark order
- Validate output format

---

## 8. Performance Optimization

```python
# Batch processing
def process_video_batch(video_paths, num_workers=4):
    """Process multiple videos in parallel."""
    from multiprocessing import Pool
    
    with Pool(num_workers) as pool:
        results = pool.map(video_to_pose, video_paths)
    
    return results

# GPU acceleration (if available)
# MediaPipe automatically uses GPU when available
# No additional configuration needed
```

---

## 9. Output Format

### Recommended Structure

```python
output = {
    'landmarks': pose_data,        # (frames, 523, 3)
    'confidence': confidence,       # (frames, 523)
    'fps': fps,                    # float
    'metadata': {
        'video_path': str,
        'num_frames': int,
        'duration_sec': float,
        'landmark_order': ['pose', 'face', 'left_hand', 'right_hand'],
        'normalization': 'shoulder_width',
        'smoothing': 'savgol_3_1'
    }
}
```

### Save to NumPy

```python
np.savez_compressed(
    'output.npz',
    landmarks=pose_data,
    confidence=confidence,
    fps=fps
)
```

### Save to pose-format (for compatibility)

```python
# Requires pose-format library
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody

# Create header (define components)
# Create body
# Save to .pose file
# See pose-format documentation for details
```

---

## 10. Testing Your Implementation

```python
# Test on sample video
pose_data, confidence, fps = video_to_pose('sample_sign.mp4')

# Validate
validate_output(pose_data, confidence)

# Visualize (optional)
import matplotlib.pyplot as plt

frame_idx = 15
plt.figure(figsize=(10, 10))
plt.scatter(pose_data[frame_idx, :, 0], -pose_data[frame_idx, :, 1], c=confidence[frame_idx])
plt.title(f'Frame {frame_idx}')
plt.colorbar(label='Confidence')
plt.axis('equal')
plt.show()
```

---

## 11. Troubleshooting

| Issue | Solution |
|-------|----------|
| Landmarks all zeros | Check video format, ensure face/hands visible |
| Confidence all zeros | MediaPipe not detecting person, check lighting |
| Shape mismatch | Verify UPPER_BODY_INDICES list |
| Normalization fails | Check for NaN/inf in shoulder positions |
| Trimming removes all frames | Adjust wrist-above-elbow threshold |
| Smoothing artifacts | Verify face landmarks are skipped |

---

## 12. Next Steps

1. **Implement core pipeline** using code above
2. **Test on reference videos** from SignSuisse dataset
3. **Compare outputs** to original `.pose` files (if available)
4. **Integrate with training pipeline**
5. **Optimize for batch processing**

---

## 13. Key Differences from Original

| Aspect | Original | Your Implementation |
|--------|----------|---------------------|
| Library | pose-format (wrapper) | Direct MediaPipe |
| Reduction | Via pose-format utility | Manual index selection |
| File format | Binary .pose | NumPy .npz or custom |
| Validation | Implicit | Explicit checks |

**Compatibility:** Outputs should be numerically equivalent after accounting for format differences.

---

## References

- **Full Analysis:** See `POSE_EXTRACTION_METHODOLOGY_ANALYSIS.md`
- **Original Code:** `concatenate.py`, `smoothing.py`, `preprocess_files.py`
- **MediaPipe Docs:** https://google.github.io/mediapipe/solutions/holistic
- **Research Paper:** Moryossef et al. (2023), arXiv:2305.17714

---

**Version:** 1.0  
**Last Updated:** 2026-01-23  
**Status:** Ready for implementation
