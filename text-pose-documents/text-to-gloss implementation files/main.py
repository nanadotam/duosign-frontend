"""
Text-to-Gloss API
=================
Convert English text to ASL gloss notation.

Author: Nana Amoako
Date: February 2026
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import logging

from text_to_gloss import TextToGlossConverter, load_vocabularies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Text-to-Gloss API",
    description="Convert English text to ASL gloss notation",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class AppState:
    def __init__(self):
        self.converter = None
        self.is_initialized = False

state = AppState()


# Request/Response Models
class TextToGlossRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    use_llm_fallback: bool = Field(default=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Tomorrow I will go to school",
                "use_llm_fallback": True
            }
        }


class GlossItem(BaseModel):
    index: int
    gloss: str
    original_word: Optional[str] = None
    confidence: float = 1.0


class TextToGlossResponse(BaseModel):
    text: str
    glosses: List[GlossItem]
    gloss_string: str
    method: str
    confidence: float
    processing_time_ms: float


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., max_items=100)
    use_llm_fallback: bool = Field(default=True)


class BatchResponse(BaseModel):
    results: List[TextToGlossResponse]
    total_time_ms: float
    successful: int
    failed: int


class HealthResponse(BaseModel):
    status: str
    version: str
    cache_size: int


# Startup
@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting Text-to-Gloss API...")
    
    try:
        wlasl_vocab, asl_lex_vocab = load_vocabularies()
        state.converter = TextToGlossConverter(wlasl_vocab, asl_lex_vocab)
        state.is_initialized = True
        logger.info("✅ API ready!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Text-to-Gloss API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        cache_size=len(state.converter.cache)
    )


@app.post("/api/v1/text-to-gloss", response_model=TextToGlossResponse)
async def convert_text_to_gloss(request: TextToGlossRequest):
    """
    Convert English text to ASL gloss notation.
    
    Returns ordered glosses with indices for lookup in your pose directory.
    """
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = time.time()
    
    try:
        result = state.converter.convert(
            text=request.text,
            use_llm_fallback=request.use_llm_fallback
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Format as indexed glosses for lookup
        gloss_items = [
            GlossItem(
                index=i,
                gloss=token["gloss"],
                original_word=token.get("word"),
                confidence=token.get("confidence", 1.0)
            )
            for i, token in enumerate(result["tokens"])
        ]
        
        response = TextToGlossResponse(
            text=request.text,
            glosses=gloss_items,
            gloss_string=result["gloss"],
            method=result["method"],
            confidence=result["confidence"],
            processing_time_ms=processing_time
        )
        
        logger.info(f"✅ '{request.text}' → '{result['gloss']}' ({processing_time:.1f}ms)")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/batch", response_model=BatchResponse)
async def batch_convert(request: BatchRequest):
    """Batch convert multiple texts"""
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = time.time()
    results = []
    successful = 0
    failed = 0
    
    for text in request.texts:
        try:
            result = state.converter.convert(text, request.use_llm_fallback)
            
            gloss_items = [
                GlossItem(
                    index=i,
                    gloss=token["gloss"],
                    original_word=token.get("word"),
                    confidence=token.get("confidence", 1.0)
                )
                for i, token in enumerate(result["tokens"])
            ]
            
            results.append(TextToGlossResponse(
                text=text,
                glosses=gloss_items,
                gloss_string=result["gloss"],
                method=result["method"],
                confidence=result["confidence"],
                processing_time_ms=0
            ))
            successful += 1
        except Exception as e:
            logger.error(f"❌ Batch item failed: {e}")
            failed += 1
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchResponse(
        results=results,
        total_time_ms=total_time,
        successful=successful,
        failed=failed
    )


@app.post("/api/v1/cache/clear")
async def clear_cache():
    """Clear conversion cache"""
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    cleared = len(state.converter.cache)
    state.converter.cache.clear()
    
    return {"status": "success", "cleared": cleared}


@app.get("/api/v1/stats")
async def get_stats():
    """Get API statistics"""
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return state.converter.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
