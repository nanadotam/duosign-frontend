# DuoSign Python Backend - Quick Start Guide

## 🚀 Starting the API Server

### Prerequisites
```bash
# Install dependencies (one-time setup)
cd duosign_algo
pip install -r requirements-api.txt
```

### Start the Server

**Option 1: Development Mode (Auto-reload)**
```bash
cd duosign_algo
uvicorn api.main:app --reload --port 8000
```

**Option 2: Production Mode**
```bash
cd duosign_algo
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Verify Server is Running

Open your browser to:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

Or use curl:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "3.0.0",
  "timestamp": "2026-01-31T16:30:00",
  "signs_available": 30
}
```

---

## 📡 API Endpoints

### 1. List All Signs
```bash
curl http://localhost:8000/api/signs
```

Returns metadata for all available signs:
```json
[
  {
    "gloss": "05730",
    "frame_count": 47,
    "duration_sec": 1.57,
    "file_size_kb": 1.03
  },
  ...
]
```

### 2. Get Pose Data for a Sign
```bash
curl http://localhost:8000/api/sign/05730
```

Returns complete pose data in V3 format with quaternions.

### 3. Get Metadata Only (Faster)
```bash
curl http://localhost:8000/api/sign/05730/metadata
```

Returns just the metadata without frame data.

---

## 🔄 Complete Pipeline Workflow

### 1. Extract Poses from Videos
```bash
cd duosign_algo
python pose_extraction/wlasl_pose_pipeline.py \
    --input_dir pose_extraction/wlasl-processed/videos \
    --output_dir pose_extraction/poses_kalidokit \
    --num_workers 4
```

### 2. Convert to V3 Format
```bash
python scripts/convert_to_v3.py \
    --input_dir pose_extraction/poses_kalidokit \
    --output_dir pose_processing/poses_v3 \
    --min-cutoff 1.0 \
    --beta 0.007
```

### 3. Copy to Frontend
```bash
cp pose_processing/poses_v3/*.json ../public/poses_v3/
```

### 4. Start API Server
```bash
uvicorn api.main:app --reload --port 8000
```

---

## 🎯 One-Command Pipeline

Run the complete pipeline with one command:
```bash
python run_complete_pipeline.py --num_videos 10
```

Options:
- `--num_videos N` - Process N videos (default: 5)
- `--skip_extraction` - Skip pose extraction, use existing .pose files

---

## 🛠️ Filter Configuration

Adjust smoothing for different sign types:

**Fingerspelling (more smoothing)**
```bash
python scripts/convert_to_v3.py \
    --input_dir pose_extraction/poses_kalidokit \
    --output_dir pose_processing/poses_v3 \
    --min-cutoff 0.5 \
    --beta 0.001
```

**Dynamic Signs (balanced - default)**
```bash
--min-cutoff 1.0 --beta 0.007
```

**Classifiers (less smoothing)**
```bash
--min-cutoff 1.5 --beta 0.015
```

---

## 🧪 Testing

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# List signs
curl http://localhost:8000/api/signs

# Get specific sign
curl http://localhost:8000/api/sign/05730 | python -m json.tool
```

### Test in Frontend
1. Start API server: `uvicorn api.main:app --reload`
2. Start frontend: `npm run dev`
3. Open http://localhost:3000

---

## 📁 Directory Structure

```
duosign_algo/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI application
├── pose_processing/
│   ├── filters.py           # 1€ filter
│   ├── quaternion_solver.py # Quaternion computation
│   ├── skeleton_normalizer.py
│   └── pose_v3_converter.py # Complete pipeline
├── scripts/
│   └── convert_to_v3.py     # Batch conversion CLI
├── requirements-api.txt      # Python dependencies
└── run_complete_pipeline.py  # One-command pipeline
```

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn api.main:app --reload --port 8001
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements-api.txt --force-reinstall
```

### CORS Issues
Update `api/main.py` to add your frontend URL:
```python
allow_origins=[
    "http://localhost:3000",
    "http://localhost:3001",
    "https://your-domain.com",  # Add your domain
]
```

---

## 📊 Performance

- **Throughput**: 15-20k requests/second
- **Latency**: <5ms (p50) for static file serving
- **File Size**: ~1KB per pose (81% smaller than raw landmarks)

---

## 🔗 Related Documentation

- **Implementation Plan**: `brain/implementation_plan.md`
- **Walkthrough**: `brain/walkthrough.md`
- **Algorithm Details**: `brain/algorithm_*.md`
- **API Docs**: http://localhost:8000/docs (when server running)
