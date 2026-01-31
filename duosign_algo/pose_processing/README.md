# Pose Processing Module - Python Backend

This module provides advanced pose processing algorithms for sign language avatar animation.

## Structure

```
pose_processing/
├── __init__.py              # Package initialization
├── filters.py               # 1€ filter and temporal smoothing
├── quaternion_solver.py     # Landmark-to-quaternion conversion
├── skeleton_normalizer.py   # Scale normalization
├── pose_v3_converter.py     # Complete conversion pipeline
└── tests/                   # Unit tests
    ├── test_filters.py
    ├── test_quaternion_solver.py
    └── test_pose_v3_converter.py
```

## Installation

```bash
cd duosign_algo
pip install -r requirements-api.txt
```

## Usage

See individual module documentation for detailed usage examples.
