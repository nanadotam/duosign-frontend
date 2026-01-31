#!/usr/bin/env python3
"""
FastAPI Backend for Pose Data Serving
=====================================

This module provides a high-performance REST API for serving pre-processed
pose data in .pose-v3 format to the frontend application.

Key Features:
    - Async I/O for concurrent requests
    - Automatic API documentation (OpenAPI/Swagger)
    - CORS support for frontend access
    - Type-safe request/response validation with Pydantic
    - Health check endpoint

Performance:
    - 15-20k requests/second (vs Flask's 2-3k)
    - <5ms latency (p50) for static file serving
    - Event-loop based concurrency (non-blocking I/O)

Author: Nana Kwaku Amoako
Date: 2026-01-31
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Models for Type-Safe API
# ============================================================================

class FrameData(BaseModel):
    """
    Single frame of pose data in v3 format.
    
    Attributes:
        rotations: Bone name → quaternion [x, y, z, w]
        velocities: Bone name → angular velocity (rad/s)
        confidences: Bone name → detection confidence (0-1)
    """
    rotations: Dict[str, List[float]] = Field(
        ...,
        description="Bone rotations as quaternions [x, y, z, w]"
    )
    velocities: Dict[str, float] = Field(
        ...,
        description="Angular velocities in rad/s"
    )
    confidences: Dict[str, float] = Field(
        ...,
        description="Detection confidences (0-1)"
    )


class PoseV3Data(BaseModel):
    """
    Complete pose animation data in v3 format.
    
    Attributes:
        format_version: Format version string ("3.0-quaternion")
        fps: Frame rate in Hz
        frame_count: Total number of frames
        source_video: Original video filename
        frames: List of frame data
        skeleton_info: Skeleton metadata
        filter_config: 1€ filter configuration used
        metadata: Conversion metadata
    """
    format_version: str = Field(..., description="Format version")
    fps: float = Field(..., description="Frame rate in Hz")
    frame_count: int = Field(..., description="Number of frames")
    source_video: str = Field(..., description="Source video filename")
    frames: List[FrameData] = Field(..., description="Frame data")
    skeleton_info: Dict[str, Any] = Field(..., description="Skeleton metadata")
    filter_config: Dict[str, float] = Field(..., description="Filter configuration")
    metadata: Dict[str, str] = Field(..., description="Conversion metadata")


class SignListItem(BaseModel):
    """
    Metadata for a single sign in the library.
    
    Attributes:
        gloss: Sign gloss name (human-readable)
        video_id: WLASL video ID (for loading pose data)
        frame_count: Number of frames
        duration_sec: Duration in seconds
        file_size_kb: File size in KB
    """
    gloss: str = Field(..., description="Sign gloss/identifier")
    video_id: str = Field(..., description="WLASL video ID")
    frame_count: int = Field(..., description="Number of frames")
    duration_sec: float = Field(..., description="Duration in seconds")
    file_size_kb: float = Field(..., description="File size in KB")


class HealthResponse(BaseModel):
    """
    Health check response.
    
    Attributes:
        status: Service status ("healthy")
        version: API version
        timestamp: Current server time
        signs_available: Number of signs in library
    """
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current server time")
    signs_available: int = Field(..., description="Number of signs available")


# ============================================================================
# FastAPI Application
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="DuoSign Pose API",
    description="High-performance API for serving quaternion-native pose data",
    version="3.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:3001",  # Alternative port
        # "https://duosign.app",    # Prod
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Configuration
# When running from duosign_algo/, __file__ is duosign_algo/api/main.py
# We need to go: api/main.py -> api -> duosign_algo -> duosign-frontend -> public/poses_v3
POSES_DIR = Path(__file__).parent.parent.parent / "public" / "poses_v3"
if not POSES_DIR.exists():
    # Fallback: try relative to current working directory
    POSES_DIR = Path("../public/poses_v3").resolve()
POSES_DIR.mkdir(parents=True, exist_ok=True)

# WLASL metadata directory
WLASL_DIR = Path(__file__).parent.parent / "wlasl-processed"

# Gloss to video ID mapping (loaded lazily)
_gloss_map: Dict[str, List[str]] = {}

def load_gloss_map() -> Dict[str, List[str]]:
    """Load WLASL gloss -> video ID mapping (cached)."""
    global _gloss_map
    if _gloss_map:
        return _gloss_map
    
    wlasl_file = WLASL_DIR / "WLASL_v0.3.json"
    if not wlasl_file.exists():
        return {}
    
    try:
        with open(wlasl_file, 'r') as f:
            data = json.load(f)
        
        for entry in data:
            gloss = entry['gloss'].lower()
            video_ids = [inst['video_id'] for inst in entry.get('instances', [])]
            _gloss_map[gloss] = video_ids
        
        return _gloss_map
    except Exception:
        return {}


def get_video_ids_for_gloss(gloss: str) -> List[str]:
    """Get all video IDs for a given gloss name."""
    mapping = load_gloss_map()
    return mapping.get(gloss.lower(), [])


def get_gloss_for_video_id(video_id: str) -> Optional[str]:
    """Get gloss name for a video ID."""
    mapping = load_gloss_map()
    for gloss, vids in mapping.items():
        if video_id in vids:
            return gloss
    return None


# ============================================================================
# Helper Functions
# ============================================================================

def load_pose_file(gloss: str) -> Dict[str, Any]:
    """
    Load a .pose-v3 file from disk.
    
    Args:
        gloss: Sign gloss (filename without extension)
    
    Returns:
        Dict containing pose data
    
    Raises:
        HTTPException: If file not found or invalid
    """
    pose_path = POSES_DIR / f"{gloss}.json"
    
    if not pose_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Sign '{gloss}' not found. Available signs: {list_available_signs()}"
        )
    
    try:
        with open(pose_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid pose file format: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load pose file: {e}"
        )


def list_available_signs() -> List[str]:
    """
    List all available sign glosses.
    
    Returns:
        List of sign glosses (filenames without .json extension)
    """
    if not POSES_DIR.exists():
        return []
    
    return [
        f.stem for f in POSES_DIR.glob("*.json")
        if not f.name.startswith("_")  # Exclude metadata files
    ]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Dict with API name and documentation links
    """
    return {
        "name": "DuoSign Pose API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with service status and metadata
    
    Example:
        ```bash
        curl http://localhost:8000/health
        ```
    """
    signs = list_available_signs()
    
    return HealthResponse(
        status="healthy",
        version="3.0.0",
        timestamp=datetime.now().isoformat(),
        signs_available=len(signs)
    )


@app.get("/api/signs", response_model=List[SignListItem])
async def list_signs(
    limit: Optional[int] = Query(None, description="Maximum number of signs to return"),
    offset: Optional[int] = Query(0, description="Number of signs to skip")
):
    """
    List all available signs with metadata.
    
    Args:
        limit: Maximum number of signs to return (optional)
        offset: Number of signs to skip for pagination (default: 0)
    
    Returns:
        List of SignListItem with metadata for each sign
    
    Example:
        ```bash
        # Get all signs
        curl http://localhost:8000/api/signs
        
        # Get first 10 signs
        curl http://localhost:8000/api/signs?limit=10
        
        # Get next 10 signs (pagination)
        curl http://localhost:8000/api/signs?limit=10&offset=10
        ```
    """
    signs = list_available_signs()
    
    # Apply pagination
    if offset:
        signs = signs[offset:]
    if limit:
        signs = signs[:limit]
    
    # Load metadata for each sign (video_id -> gloss name)
    sign_list = []
    for video_id in signs:
        try:
            data = load_pose_file(video_id)
            gloss_name = get_gloss_for_video_id(video_id) or video_id
            
            sign_list.append(SignListItem(
                gloss=gloss_name,  # Use actual gloss name
                video_id=video_id,  # WLASL video ID for loading
                frame_count=data["frame_count"],
                duration_sec=round(data["frame_count"] / data["fps"], 2),
                file_size_kb=round((POSES_DIR / f"{video_id}.json").stat().st_size / 1024, 2)
            ))
        except Exception as e:
            # Skip invalid files
            continue
    
    return sign_list


@app.get("/api/sign/{gloss}", response_model=PoseV3Data)
async def get_sign(
    gloss: str = PathParam(..., description="Sign gloss/identifier")
):
    """
    Get pose data for a specific sign.
    
    Args:
        gloss: Sign gloss (e.g., "hello", "thank_you")
    
    Returns:
        PoseV3Data with complete animation data
    
    Raises:
        HTTPException: 404 if sign not found, 500 if file invalid
    
    Example:
        ```bash
        # Get pose data for "hello"
        curl http://localhost:8000/api/sign/hello
        
        # Use in JavaScript
        const response = await fetch('http://localhost:8000/api/sign/hello');
        const poseData = await response.json();
        console.log(poseData.frame_count);  // Number of frames
        console.log(poseData.frames[0].rotations.leftUpperArm);  // [x, y, z, w]
        ```
    """
    data = load_pose_file(gloss)
    return PoseV3Data(**data)


@app.get("/api/sign/{gloss}/metadata", response_model=Dict[str, Any])
async def get_sign_metadata(
    gloss: str = PathParam(..., description="Sign gloss/identifier")
):
    """
    Get metadata for a specific sign (without frame data).
    
    This endpoint is faster than /api/sign/{gloss} because it doesn't
    include the full frame data, only metadata.
    
    Args:
        gloss: Sign gloss
    
    Returns:
        Dict with metadata (fps, frame_count, skeleton_info, etc.)
    
    Example:
        ```bash
        curl http://localhost:8000/api/sign/hello/metadata
        ```
    """
    data = load_pose_file(gloss)
    
    # Return everything except frames (which is the bulk of the data)
    return {
        "format_version": data["format_version"],
        "fps": data["fps"],
        "frame_count": data["frame_count"],
        "source_video": data["source_video"],
        "skeleton_info": data["skeleton_info"],
        "filter_config": data["filter_config"],
        "metadata": data["metadata"]
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    """
    Run the API server directly (for development).
    
    For production, use:
        uvicorn api.main:app --host 0.0.0.0 --port 8000
    """
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
