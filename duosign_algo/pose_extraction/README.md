# WLASL Pose Extraction Pipeline

A robust, modular pipeline for extracting `.pose` files from WLASL (World-Level American Sign Language) dataset videos using MediaPipe Holistic.

## Features

- **Deterministic Output**: Every video produces exactly one `.pose` file (or a logged failure)
- **Fixed-Shape Tensors**: Every frame produces a `(523, 4)` tensor with NaN padding for missing landmarks
- **Resumable & Idempotent**: Skips existing valid outputs; crash-safe restart
- **Modular Design**: Separate components for decoding, extraction, serialization, and orchestration
- **Scalable**: Multiprocessing support for 21,000+ videos
- **Auditable**: Configuration logging, failure tracking, manifest generation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install mediapipe opencv-python numpy tqdm
```

## Usage

### Desktop

```bash
# Basic usage
python wlasl_pose_pipeline.py \
    --input_dir "/path/to/WLASL/videos" \
    --output_dir "/path/to/output_pose" \
    --num_workers 8

# With all options
python wlasl_pose_pipeline.py \
    --input_dir "/path/to/WLASL/videos" \
    --output_dir "/path/to/output_pose" \
    --num_workers 8 \
    --model_complexity 2 \
    --recursive \
    --log_level INFO
```

### Google Colab

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Install dependencies
!pip install mediapipe opencv-python tqdm

# Cell 3: Upload or clone the pipeline
# Option A: Upload wlasl_pose_pipeline.py to Colab
# Option B: Clone from your repository

# Cell 4: Run the pipeline
!python wlasl_pose_pipeline.py \
    --input_dir "/content/drive/MyDrive/WLASL/videos" \
    --output_dir "/content/drive/MyDrive/WLASL/pose_out" \
    --num_workers 4
```

> **Note**: Colab usually has fewer CPU cores available. Use `--num_workers 2` or `4` for optimal performance.

## Command-Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input_dir` | `-i` | Required | Directory containing input video files |
| `--output_dir` | `-o` | Required | Directory for output .pose files |
| `--num_workers` | `-w` | 4 | Number of parallel worker processes |
| `--model_complexity` | `-m` | 2 | MediaPipe complexity: 0=lite, 1=full, 2=heavy |
| `--min_detection_confidence` | | 0.5 | Detection confidence threshold [0-1] |
| `--min_tracking_confidence` | | 0.5 | Tracking confidence threshold [0-1] |
| `--skip_existing` | | True | Skip videos with existing valid .pose files |
| `--no_skip_existing` | | | Force reprocessing of all videos |
| `--recursive` | `-r` | False | Search input directory recursively |
| `--log_level` | | INFO | Logging level: DEBUG, INFO, WARNING, ERROR |

## Output Format

Each `.pose` file is a compressed NumPy archive (`.npz`) containing:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `landmarks` | (T, 523, 3) | float32 | x, y, z coordinates |
| `confidence` | (T, 523) | float32 | Detection confidence (NaN = missing) |
| `presence_mask` | (T, 523) | bool | True where landmark detected |
| `fps` | scalar | float | Video frame rate |
| `frame_count` | scalar | int | Number of frames |
| `source_video` | scalar | str | Original video path |
| `landmark_layout` | dict | object | Index ranges per component |
| `version` | scalar | str | Format version ("1.0") |

### Landmark Layout

| Component | Index Range | Count | Source |
|-----------|-------------|-------|--------|
| Pose (upper body) | 0 – 12 | 13 | BlazePose |
| Face | 13 – 480 | 468 | Face Mesh |
| Left Hand | 481 – 501 | 21 | Hand Tracking |
| Right Hand | 502 – 522 | 21 | Hand Tracking |
| **Total** | 0 – 522 | **523** | MediaPipe Holistic |

### Loading Pose Files

```python
import numpy as np

# Load a pose file
data = np.load("output/example.pose", allow_pickle=True)

# Access components
landmarks = data["landmarks"]      # Shape: (T, 523, 3)
confidence = data["confidence"]    # Shape: (T, 523)
fps = float(data["fps"])
frame_count = int(data["frame_count"])

# Get specific landmark ranges
pose_range = (0, 13)
face_range = (13, 481)
left_hand_range = (481, 502)
right_hand_range = (502, 523)

# Extract just hands for a specific frame
frame_idx = 10
left_hand = landmarks[frame_idx, 481:502, :]   # (21, 3)
right_hand = landmarks[frame_idx, 502:523, :]  # (21, 3)
```

## Output Directory Structure

```
output_dir/
├── config.json           # Pipeline configuration
├── manifest.jsonl        # Successful extractions log
├── failures.jsonl        # Failed extractions log
├── summary.json          # Run statistics
├── pipeline.log          # Detailed log file
└── *.pose                # Extracted pose files
```

## Resumability

The pipeline is designed to be crash-safe and resumable:

1. **Skip existing**: Valid `.pose` files are skipped by default
2. **Atomic writes**: Files are written to `.tmp` then renamed
3. **Validation**: Corrupted files are automatically reprocessed
4. **Logging**: All successes/failures are logged to JSONL files

To resume an interrupted run, simply re-run the same command:

```bash
# This will continue from where it left off
python wlasl_pose_pipeline.py --input_dir ./videos --output_dir ./poses
```

## Utility Functions

The pipeline includes utility functions for inspecting pose files:

```python
from wlasl_pose_pipeline import print_pose_info, visualize_pose

# Print pose file information
print_pose_info("output/example.pose")

# Visualize landmarks (requires matplotlib)
visualize_pose("output/example.pose", frame_idx=15)
```

## Performance Tips

### Desktop
- Use `--num_workers` equal to your CPU core count
- Use SSD storage for faster I/O
- Model complexity 2 is most accurate but slower; 1 is a good balance

### Colab
- Use `--num_workers 2-4` (Colab has limited cores)
- Process videos in batches to avoid Drive timeouts
- Consider using Colab Pro for more resources

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PoseExtractionPipeline                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ VideoDecoder │→ │  Landmark    │→ │   PoseSerializer     │  │
│  │  (OpenCV)    │  │  Extractor   │  │  (.pose writing)     │  │
│  └──────────────┘  │ (MediaPipe)  │  └──────────────────────┘  │
│                    └──────────────┘                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  ManifestLogger                          │  │
│  │           (manifest.jsonl + failures.jsonl)              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## References

- **Methodology**: Based on [Moryossef et al. (2023)](https://arxiv.org/abs/2305.17714) - "An Open-Source Gloss-Based Baseline for Spoken to Signed Language Translation"
- **MediaPipe Holistic**: https://google.github.io/mediapipe/solutions/holistic
- **WLASL Dataset**: https://github.com/dxli94/WLASL

## Author

Nana Kwaku Amoako

## License

This project is provided for research and educational purposes.
