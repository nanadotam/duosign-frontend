# Python Backend Quick Start Guide

## Installation

```bash
cd duosign_algo

# Install dependencies
pip install -r requirements-api.txt
```

## Running the API Server

```bash
# Development mode (auto-reload)
cd duosign_algo
uvicorn api.main:app --reload --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Converting Poses to V3 Format

### Single File

```bash
python -m pose_processing.pose_v3_converter \
    --input pose_extraction/poses_kalidokit/hello.pose \
    --output pose_processing/poses_v3/hello.json
```

### Batch Conversion

```bash
python scripts/convert_to_v3.py \
    --input_dir pose_extraction/poses_kalidokit \
    --output_dir pose_processing/poses_v3 \
    --num-workers 4
```

### Custom Filter Settings

```bash
# For fingerspelling (more smoothing)
python scripts/convert_to_v3.py \
    --input_dir pose_extraction/poses_kalidokit \
    --output_dir pose_processing/poses_v3 \
    --min-cutoff 0.5 \
    --beta 0.001

# For dynamic signs (balanced)
python scripts/convert_to_v3.py \
    --input_dir pose_extraction/poses_kalidokit \
    --output_dir pose_processing/poses_v3 \
    --min-cutoff 1.0 \
    --beta 0.007

# For classifiers (less smoothing)
python scripts/convert_to_v3.py \
    --input_dir pose_extraction/poses_kalidokit \
    --output_dir pose_processing/poses_v3 \
    --min-cutoff 1.5 \
    --beta 0.015
```

## API Endpoints

### Get Sign Data
```bash
curl http://localhost:8000/api/sign/hello
```

### List All Signs
```bash
curl http://localhost:8000/api/signs
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Testing

```bash
# Run all tests
pytest pose_processing/tests/

# Run specific test file
pytest pose_processing/tests/test_filters.py -v

# Run with coverage
pytest --cov=pose_processing pose_processing/tests/
```
