# Complete Video-to-Pose Pipeline for WLASL Dataset

Based on the codebase analysis and WLASL JSON structure, here's your comprehensive pipeline:

---

## 📊 Understanding WLASL Data Splits

### **How WLASL is Organized:**

Each gloss has **multiple video instances** with **split assignments**:

```json
{
  "gloss": "book",
  "instances": [
    {
      "video_id": "xyz123",
      "split": "train",    // Use for training
      "bbox": [...],
      "fps": 25
    },
    {
      "video_id": "abc456",
      "split": "val",      // Use for validation/tuning
      "bbox": [...]
    },
    {
      "video_id": "def789",
      "split": "test",     // Use for final evaluation
      "bbox": [...]
    }
  ]
}
```

**Key insights:**
- ✅ Multiple videos per gloss (typically 5-10 videos)
- ✅ Each video is pre-assigned to train/val/test
- ✅ Different people signing the same word (good for generalization)

---

## 🏗️ Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Video Download & Organization                    │
│  Input:  WLASL_v0.3.json                                    │
│  Output: Organized video files by split                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Pose Extraction (MediaPipe Holistic)             │
│  Input:  MP4 videos                                         │
│  Output: Raw .pose files                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: Pose Preprocessing & Quality Control             │
│  - Trim neutral positions                                   │
│  - Normalize coordinates                                    │
│  - Remove low-quality poses                                 │
│  Output: Cleaned .pose files                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: Lexicon Building                                  │
│  - Create train/val/test lexicons                           │
│  - Map glosses to pose files                                │
│  Output: index_train.csv, index_val.csv, index_test.csv    │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Complete Implementation

### **Project Structure:**

```
asl_pose_pipeline/
├── config/
│   └── config.yaml                 # Configuration
├── data/
│   ├── raw/
│   │   └── WLASL_v0.3.json        # Original metadata
│   ├── videos/
│   │   ├── train/                  # Training videos
│   │   ├── val/                    # Validation videos
│   │   └── test/                   # Test videos
│   ├── poses/
│   │   ├── raw/                    # Raw extracted poses
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── processed/              # Cleaned poses
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── lexicons/
│       ├── index_train.csv
│       ├── index_val.csv
│       └── index_test.csv
├── src/
│   ├── __init__.py
│   ├── download.py                 # Phase 1
│   ├── extract_poses.py            # Phase 2
│   ├── preprocess_poses.py         # Phase 3
│   ├── build_lexicon.py            # Phase 4
│   └── utils.py                    # Utilities
├── scripts/
│   └── run_pipeline.py             # Main orchestrator
├── logs/
│   └── pipeline.log
└── requirements.txt
```

---

### **Phase 1: Download & Organize Videos**

```python
# src/download.py
import json
import subprocess
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WLASLDownloader:
    """
    Downloads WLASL videos and organizes them by split (train/val/test)
    """
    
    def __init__(self, json_path: str, output_dir: str):
        self.json_path = Path(json_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load WLASL metadata
        with open(self.json_path) as f:
            self.wlasl_data = json.load(f)
        
        logger.info(f"Loaded {len(self.wlasl_data)} glosses")
    
    def parse_instances_by_split(self) -> Dict[str, List]:
        """
        Organize video instances by split (train/val/test)
        
        Returns:
            Dict mapping split -> list of (gloss, video_id, instance_data)
        """
        splits = {'train': [], 'val': [], 'test': []}
        
        for entry in self.wlasl_data:
            gloss = entry['gloss']
            
            for instance in entry['instances']:
                split = instance.get('split', 'train')  # Default to train if missing
                video_id = instance['video_id']
                
                splits[split].append({
                    'gloss': gloss,
                    'video_id': video_id,
                    'instance': instance
                })
        
        logger.info(f"Found {len(splits['train'])} train, "
                   f"{len(splits['val'])} val, "
                   f"{len(splits['test'])} test instances")
        
        return splits
    
    def download_video(self, video_id: str, output_path: Path) -> bool:
        """
        Download a single video from YouTube
        
        Args:
            video_id: YouTube video ID
            output_path: Where to save the video
            
        Returns:
            True if successful, False otherwise
        """
        if output_path.exists():
            return True  # Already downloaded
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        try:
            # Use yt-dlp (better than youtube-dl)
            cmd = [
                'yt-dlp',
                '-f', 'best[height<=480]',  # Download 480p max (smaller files)
                '-o', str(output_path),
                '--quiet',
                '--no-warnings',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0 and output_path.exists():
                return True
            else:
                logger.warning(f"Failed to download {video_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {video_id}: {e}")
            return False
    
    def crop_video(self, input_path: Path, output_path: Path, 
                   bbox: List[int], start_frame: int, end_frame: int) -> bool:
        """
        Crop video to signing region using bbox and frame range
        
        Args:
            input_path: Input video path
            output_path: Output cropped video path
            bbox: [x, y, width, height] bounding box
            start_frame: Start frame number
            end_frame: End frame number
        """
        try:
            # Calculate frame times (assuming 25 fps - adjust if needed)
            fps = 25
            start_time = start_frame / fps
            duration = (end_frame - start_frame) / fps
            
            x, y, w, h = bbox
            
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-vf', f'crop={w}:{h}:{x}:{y}',
                '-c:v', 'libx264',
                '-crf', '23',
                '-an',  # No audio
                '-y',   # Overwrite
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error cropping video: {e}")
            return False
    
    def download_all(self, limit: int = None):
        """
        Download all videos organized by split
        
        Args:
            limit: Optional limit on number of videos per split (for testing)
        """
        splits = self.parse_instances_by_split()
        
        for split_name, instances in splits.items():
            split_dir = self.output_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Apply limit if specified
            if limit:
                instances = instances[:limit]
            
            logger.info(f"Downloading {len(instances)} videos for {split_name} split...")
            
            success_count = 0
            
            for item in tqdm(instances, desc=f"{split_name} videos"):
                gloss = item['gloss']
                video_id = item['video_id']
                instance = item['instance']
                
                # Create gloss subdirectory
                gloss_dir = split_dir / gloss
                gloss_dir.mkdir(exist_ok=True)
                
                # Download full video first
                temp_video = gloss_dir / f"{video_id}_temp.mp4"
                final_video = gloss_dir / f"{video_id}.mp4"
                
                if self.download_video(video_id, temp_video):
                    # Crop to signing region
                    bbox = instance.get('bbox', [0, 0, 480, 360])
                    start_frame = instance.get('frame_start', 0)
                    end_frame = instance.get('frame_end', 100)
                    
                    if self.crop_video(temp_video, final_video, 
                                      bbox, start_frame, end_frame):
                        success_count += 1
                        temp_video.unlink()  # Delete temp file
                    else:
                        logger.warning(f"Failed to crop {video_id}, keeping full video")
                        temp_video.rename(final_video)
                        success_count += 1
            
            logger.info(f"{split_name}: Downloaded {success_count}/{len(instances)} videos")


# Usage
if __name__ == "__main__":
    downloader = WLASLDownloader(
        json_path="data/raw/WLASL_v0.3.json",
        output_dir="data/videos"
    )
    
    # Download limited set for testing (remove limit for full download)
    downloader.download_all(limit=100)  # 100 videos per split for testing
```

---

### **Phase 2: Extract Poses from Videos**

```python
# src/extract_poses.py
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import logging
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderDimensions, PoseHeaderComponent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaPipePoseExtractor:
    """
    Extract poses from videos using MediaPipe Holistic
    Following Sign.MT paper methodology (Section 5)
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1  # 0=lite, 1=full, 2=heavy
        )
        
        # MediaPipe Holistic components
        self.component_names = [
            'POSE_LANDMARKS',      # 33 body landmarks
            'LEFT_HAND_LANDMARKS', # 21 left hand landmarks
            'RIGHT_HAND_LANDMARKS',# 21 right hand landmarks
            'FACE_LANDMARKS'       # 468 face landmarks
        ]
    
    def extract_from_video(self, video_path: Path) -> Dict:
        """
        Extract pose sequence from a single video
        
        Args:
            video_path: Path to MP4 video
            
        Returns:
            Dict with 'landmarks', 'fps', 'frame_count'
        """
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB (MediaPipe expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.holistic.process(frame_rgb)
            
            # Extract landmarks
            frame_data = self._extract_landmarks(results)
            
            if frame_data is not None:
                all_frames.append(frame_data)
        
        cap.release()
        
        if len(all_frames) == 0:
            logger.warning(f"No valid frames extracted from {video_path}")
            return None
        
        return {
            'landmarks': np.array(all_frames),  # Shape: [frames, points, 4]
            'fps': fps,
            'frame_count': len(all_frames),
            'video_path': str(video_path)
        }
    
    def _extract_landmarks(self, results) -> np.ndarray:
        """
        Extract all landmarks from MediaPipe results
        
        Returns:
            Array of shape [total_points, 4] where 4 = (x, y, z, confidence)
        """
        landmarks = []
        
        # Body pose (33 points)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
        else:
            # Fill with zeros if not detected
            landmarks.extend([[0, 0, 0, 0]] * 33)
        
        # Left hand (21 points)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, 1.0])
        else:
            landmarks.extend([[0, 0, 0, 0]] * 21)
        
        # Right hand (21 points)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, 1.0])
        else:
            landmarks.extend([[0, 0, 0, 0]] * 21)
        
        # Face (468 points) - optional, can skip for ASL
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, 1.0])
        else:
            landmarks.extend([[0, 0, 0, 0]] * 468)
        
        return np.array(landmarks, dtype=np.float32)
    
    def save_as_pose_file(self, landmarks_data: Dict, output_path: Path):
        """
        Save extracted landmarks as .pose file
        Compatible with pose-format library
        
        Args:
            landmarks_data: Dict from extract_from_video()
            output_path: Where to save .pose file
        """
        # Create pose-format header
        header = self._create_pose_header()
        
        # Prepare body data
        # Shape: [frames, people, points, dimensions]
        data = landmarks_data['landmarks']
        frames, points, dims = data.shape
        
        # Reshape to add "people" dimension (always 1 person)
        data = data.reshape(frames, 1, points, dims)
        
        # Separate data and confidence
        pose_data = data[:, :, :, :3]  # x, y, z
        confidence = data[:, :, :, 3]   # visibility/confidence
        
        # Create NumPyPoseBody
        body = NumPyPoseBody(
            fps=landmarks_data['fps'],
            data=pose_data,
            confidence=confidence
        )
        
        # Create Pose object
        pose = Pose(header=header, body=body)
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pose.write(f)
    
    def _create_pose_header(self) -> PoseHeader:
        """
        Create pose-format header matching MediaPipe Holistic structure
        """
        # Define components
        components = [
            PoseHeaderComponent(
                name='POSE_LANDMARKS',
                points=['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', ...]  # All 33 points
            ),
            PoseHeaderComponent(
                name='LEFT_HAND_LANDMARKS',
                points=[f'LEFT_HAND_{i}' for i in range(21)]
            ),
            PoseHeaderComponent(
                name='RIGHT_HAND_LANDMARKS',
                points=[f'RIGHT_HAND_{i}' for i in range(21)]
            ),
            PoseHeaderComponent(
                name='FACE_LANDMARKS',
                points=[f'FACE_{i}' for i in range(468)]
            )
        ]
        
        dimensions = PoseHeaderDimensions(
            width=1, height=1, depth=1  # Normalized coordinates
        )
        
        return PoseHeader(
            version=0.1,
            dimensions=dimensions,
            components=components
        )
    
    def process_split(self, video_dir: Path, output_dir: Path, split: str):
        """
        Process all videos in a split (train/val/test)
        
        Args:
            video_dir: Directory containing videos (e.g., data/videos/train)
            output_dir: Where to save poses (e.g., data/poses/raw/train)
            split: 'train', 'val', or 'test'
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all videos
        video_files = list(video_dir.rglob('*.mp4'))
        logger.info(f"Found {len(video_files)} videos in {split} split")
        
        success_count = 0
        failed_videos = []
        
        for video_path in tqdm(video_files, desc=f"Extracting {split} poses"):
            # Maintain directory structure
            relative_path = video_path.relative_to(video_dir)
            output_path = output_dir / relative_path.with_suffix('.pose')
            
            # Skip if already processed
            if output_path.exists():
                success_count += 1
                continue
            
            # Extract poses
            landmarks_data = self.extract_from_video(video_path)
            
            if landmarks_data:
                self.save_as_pose_file(landmarks_data, output_path)
                success_count += 1
            else:
                failed_videos.append(str(video_path))
        
        logger.info(f"{split}: Successfully processed {success_count}/{len(video_files)} videos")
        
        if failed_videos:
            logger.warning(f"Failed videos ({len(failed_videos)}):")
            for vid in failed_videos[:10]:  # Show first 10
                logger.warning(f"  - {vid}")


# Usage
if __name__ == "__main__":
    extractor = MediaPipePoseExtractor()
    
    # Process each split
    for split in ['train', 'val', 'test']:
        video_dir = Path(f"data/videos/{split}")
        output_dir = Path(f"data/poses/raw/{split}")
        
        extractor.process_split(video_dir, output_dir, split)
```

---

### **Phase 3: Preprocess Poses**

```python
# src/preprocess_poses.py
"""
Adapt from the codebase's preprocess_files.py and concatenate.py
"""
from pathlib import Path
from tqdm import tqdm
import logging
from pose_format import Pose
from pose_format.utils.generic import reduce_holistic
from pose_anonymization.appearance import remove_appearance

# Import from existing codebase
import sys
sys.path.append('path/to/spoken-to-signed-translation')
from spoken_to_signed.gloss_to_pose.concatenate import (
    trim_pose, normalize_pose, scale_normalized_pose
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PosePreprocessor:
    """
    Clean and normalize extracted poses
    Following Sign.MT preprocessing pipeline
    """
    
    def __init__(self, quality_threshold: float = 0.3):
        """
        Args:
            quality_threshold: Minimum average confidence to keep pose
        """
        self.quality_threshold = quality_threshold
    
    def check_quality(self, pose: Pose) -> bool:
        """
        Check if pose has sufficient quality
        
        Returns:
            True if pose should be kept
        """
        # Calculate average confidence
        avg_confidence = pose.body.confidence.mean()
        
        if avg_confidence < self.quality_threshold:
            return False
        
        # Check if we have enough valid frames
        if len(pose.body.data) < 10:  # At least 10 frames
            return False
        
        return True
    
    def preprocess_pose(self, pose: Pose) -> Pose:
        """
        Apply full preprocessing pipeline from Sign.MT
        
        Steps:
        1. Trim to active signing region
        2. Reduce holistic (remove unnecessary landmarks)
        3. Normalize coordinates
        4. Scale
        5. Remove appearance for anonymization
        """
        # 1. Trim (removes neutral positions at start/end)
        pose = trim_pose(pose)
        
        # 2. Reduce holistic (70% file size reduction)
        # Keeps only essential landmarks
        pose = reduce_holistic(pose)
        
        # 3. Normalize coordinates
        pose = normalize_pose(pose)
        
        # 4. Scale to standard size
        scale_normalized_pose(pose)
        
        # 5. Remove appearance data (for privacy/generalization)
        pose = remove_appearance(pose)
        
        return pose
    
    def process_split(self, input_dir: Path, output_dir: Path, split: str):
        """
        Preprocess all poses in a split
        
        Args:
            input_dir: Raw poses directory (e.g., data/poses/raw/train)
            output_dir: Processed poses directory (e.g., data/poses/processed/train)
            split: 'train', 'val', or 'test'
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pose_files = list(input_dir.rglob('*.pose'))
        logger.info(f"Processing {len(pose_files)} poses in {split} split")
        
        success_count = 0
        low_quality_count = 0
        
        for pose_path in tqdm(pose_files, desc=f"Preprocessing {split}"):
            # Maintain directory structure
            relative_path = pose_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            
            try:
                # Load pose
                with open(pose_path, 'rb') as f:
                    pose = Pose.read(f.read())
                
                # Quality check
                if not self.check_quality(pose):
                    low_quality_count += 1
                    logger.debug(f"Skipping low quality pose: {pose_path}")
                    continue
                
                # Preprocess
                pose = self.preprocess_pose(pose)
                
                # Save
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    pose.write(f)
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {pose_path}: {e}")
        
        logger.info(f"{split}: Preprocessed {success_count}/{len(pose_files)} poses")
        logger.info(f"  Filtered out {low_quality_count} low-quality poses")


# Usage
if __name__ == "__main__":
    preprocessor = PosePreprocessor(quality_threshold=0.3)
    
    for split in ['train', 'val', 'test']:
        input_dir = Path(f"data/poses/raw/{split}")
        output_dir = Path(f"data/poses/processed/{split}")
        
        preprocessor.process_split(input_dir, output_dir, split)
```

---

### **Phase 4: Build Lexicons**

```python
# src/build_lexicon.py
import csv
import json
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LexiconBuilder:
    """
    Build lexicon CSV files mapping glosses to pose files
    Creates separate lexicons for train/val/test splits
    """
    
    def __init__(self, wlasl_json: Path, poses_dir: Path, output_dir: Path):
        self.wlasl_json = wlasl_json
        self.poses_dir = poses_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load WLASL metadata
        with open(wlasl_json) as f:
            self.wlasl_data = json.load(f)
    
    def build_lexicon_for_split(self, split: str):
        """
        Build lexicon CSV for a specific split
        
        Args:
            split: 'train', 'val', or 'test'
        """
        split_poses_dir = self.poses_dir / split
        
        if not split_poses_dir.exists():
            logger.error(f"Poses directory not found: {split_poses_dir}")
            return
        
        # Collect all pose files
        pose_files = list(split_poses_dir.rglob('*.pose'))
        logger.info(f"Found {len(pose_files)} pose files in {split} split")
        
        # Build mapping: video_id -> gloss
        video_to_gloss = {}
        for entry in self.wlasl_data:
            gloss = entry['gloss']
            for instance in entry['instances']:
                if instance.get('split') == split:
                    video_to_gloss[instance['video_id']] = gloss
        
        # Create lexicon rows
        rows = []
        for pose_path in pose_files:
            # Extract video_id from filename
            video_id = pose_path.stem  # Removes .pose extension
            
            # Get gloss
            gloss = video_to_gloss.get(video_id)
            if not gloss:
                logger.warning(f"No gloss found for {video_id}, skipping")
                continue
            
            # Create row
            # path relative to lexicon directory
            relative_path = pose_path.relative_to(split_poses_dir)
            
            rows.append({
                'path': str(relative_path),
                'spoken_language': 'en',
                'signed_language': 'ase',  # ASE = American Sign Language
                'start': 0,
                'end': 0,
                'words': gloss.lower(),
                'glosses': gloss.upper(),
                'priority': 0
            })
        
        # Write to CSV
        output_path = self.output_dir / f'index_{split}.csv'
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['path', 'spoken_language', 'signed_language', 
                         'start', 'end', 'words', 'glosses', 'priority']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Created {split} lexicon with {len(rows)} entries")
        logger.info(f"Saved to: {output_path}")
        
        # Print statistics
        gloss_counts = defaultdict(int)
        for row in rows:
            gloss_counts[row['glosses']] += 1
        
        logger.info(f"Unique glosses: {len(gloss_counts)}")
        logger.info(f"Avg videos per gloss: {len(rows) / len(gloss_counts):.1f}")
    
    def build_all_lexicons(self):
        """Build lexicons for all splits"""
        for split in ['train', 'val', 'test']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Building {split} lexicon")
            logger.info(f"{'='*60}")
            self.build_lexicon_for_split(split)


# Usage
if __name__ == "__main__":
    builder = LexiconBuilder(
        wlasl_json=Path("data/raw/WLASL_v0.3.json"),
        poses_dir=Path("data/poses/processed"),
        output_dir=Path("data/lexicons")
    )
    
    builder.build_all_lexicons()
```

---

### **Main Pipeline Orchestrator**

```python
# scripts/run_pipeline.py
"""
Main pipeline orchestrator
Runs all phases in sequence
"""
import argparse
import logging
from pathlib import Path

from src.download import WLASLDownloader
from src.extract_poses import MediaPipePoseExtractor
from src.preprocess_poses import PosePreprocessor
from src.build_lexicon import LexiconBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main(args):
    """Run complete pipeline"""
    
    # Phase 1: Download videos
    if args.download:
        logger.info("="*60)
        logger.info("PHASE 1: Downloading videos")
        logger.info("="*60)
        
        downloader = WLASLDownloader(
            json_path=args.wlasl_json,
            output_dir=args.video_dir
        )
        downloader.download_all(limit=args.limit)
    
    # Phase 2: Extract poses
    if args.extract:
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Extracting poses")
        logger.info("="*60)
        
        extractor = MediaPipePoseExtractor()
        
        for split in ['train', 'val', 'test']:
            video_dir = Path(args.video_dir) / split
            output_dir = Path(args.raw_poses_dir) / split
            
            extractor.process_split(video_dir, output_dir, split)
    
    # Phase 3: Preprocess poses
    if args.preprocess:
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: Preprocessing poses")
        logger.info("="*60)
        
        preprocessor = PosePreprocessor(quality_threshold=0.3)
        
        for split in ['train', 'val', 'test']:
            input_dir = Path(args.raw_poses_dir) / split
            output_dir = Path(args.processed_poses_dir) / split
            
            preprocessor.process_split(input_dir, output_dir, split)
    
    # Phase 4: Build lexicons
    if args.build_lexicon:
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: Building lexicons")
        logger.info("="*60)
        
        builder = LexiconBuilder(
            wlasl_json=Path(args.wlasl_json),
            poses_dir=Path(args.processed_poses_dir),
            output_dir=Path(args.lexicon_dir)
        )
        builder.build_all_lexicons()
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WLASL Video-to-Pose Pipeline")
    
    # Input/output paths
    parser.add_argument("--wlasl-json", default="data/raw/WLASL_v0.3.json")
    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--raw-poses-dir", default="data/poses/raw")
    parser.add_argument("--processed-poses-dir", default="data/poses/processed")
    parser.add_argument("--lexicon-dir", default="data/lexicons")
    
    # Phase toggles
    parser.add_argument("--download", action="store_true", help="Run download phase")
    parser.add_argument("--extract", action="store_true", help="Run pose extraction")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--build-lexicon", action="store_true", help="Build lexicons")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    
    # Options
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit videos per split (for testing)")
    
    args = parser.parse_args()
    
    # If --all is specified, enable all phases
    if args.all:
        args.download = True
        args.extract = True
        args.preprocess = True
        args.build_lexicon = True
    
    main(args)
```

---

## 🚀 Usage

### **Installation:**

```bash
# Install dependencies
pip install -r requirements.txt

# Install yt-dlp for video download
pip install yt-dlp

# Install ffmpeg (macOS)
brew install ffmpeg
# Or Ubuntu
sudo apt-get install ffmpeg
```

### **requirements.txt:**

```
mediapipe>=0.10.0
opencv-python>=4.8.0
pose-format>=0.4.1
pose-anonymization>=0.1.0
numpy>=1.24.0
scipy>=1.11.0
tqdm>=4.66.0
pyyaml>=6.0
```

### **Run Complete Pipeline:**

```bash
# Test with 100 videos per split
python scripts/run_pipeline.py --all --limit 100

# Run specific phases
python scripts/run_pipeline.py --download --limit 500
python scripts/run_pipeline.py --extract
python scripts/run_pipeline.py --preprocess
python scripts/run_pipeline.py --build-lexicon

# Full production run (all 21k videos)
python scripts/run_pipeline.py --all
```

---

## 📈 Expected Timeline & Output

| Phase | Time (100 videos/split) | Time (Full 21k) | Output |
|-------|------------------------|-----------------|---------|
| **1. Download** | 2-4 hours | 2-3 days | 300 videos (100×3 splits) |
| **2. Extract** | 4-6 hours | 20-30 hours | 300 .pose files |
| **3. Preprocess** | 1-2 hours | 8-12 hours | Cleaned .pose files |
| **4. Lexicon** | < 5 minutes | < 10 minutes | 3 CSV files |
| **TOTAL** | **8-12 hours** | **3-5 days** | Complete dataset |

---

## ✅ Final Output Structure

```
data/
├── lexicons/
│   ├── index_train.csv     ← Use for training
│   ├── index_val.csv       ← Use for validation
│   └── index_test.csv      ← Use for evaluation
├── poses/processed/
│   ├── train/              ← 70% of data
│   ├── val/                ← 15% of data
│   └── test/               ← 15% of data
```

**This is exactly what you need to plug into the Sign.MT pipeline!** 🎉

Start with `--limit 100` to test the pipeline end-to-end before running the full dataset.