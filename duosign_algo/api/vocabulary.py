"""
Vocabulary Management for Text-to-Gloss Pipeline
=================================================
Load available gloss vocabulary from pose file inventory.

Author: Nana Amoako
Date: February 2026
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class VocabularyManager:
    """
    Manages the available gloss vocabulary from pose files.
    
    Loads from selection.json which maps gloss names to video IDs.
    """
    
    def __init__(self, selection_json_path: Optional[Path] = None):
        """
        Initialize vocabulary manager.
        
        Args:
            selection_json_path: Path to selection.json. If None, uses default.
        """
        self.selection_path = selection_json_path or self._find_selection_json()
        self.gloss_to_videos: Dict[str, List[str]] = {}
        self.video_to_gloss: Dict[str, str] = {}
        self.available_glosses: Set[str] = set()
        
        self._load_vocabulary()
    
    def _find_selection_json(self) -> Path:
        """Find selection.json in expected locations."""
        candidates = [
            Path(__file__).parent.parent.parent / "old-git-v-gloss-poses-subset" / "test_subset" / "selection.json",
            Path("old-git-v-gloss-poses-subset/test_subset/selection.json"),
        ]
        
        for path in candidates:
            if path.exists():
                return path
        
        # Return default even if doesn't exist (will log warning)
        return candidates[0]
    
    def _load_vocabulary(self) -> None:
        """Load vocabulary from selection.json."""
        if not self.selection_path.exists():
            logger.warning(f"Selection file not found: {self.selection_path}")
            return
        
        try:
            with open(self.selection_path, 'r') as f:
                data = json.load(f)
            
            # Build mappings
            for video_entry in data.get("videos", []):
                gloss = video_entry["class_name"].upper()
                video_id = video_entry["video_id"]
                
                # Gloss -> list of video IDs
                if gloss not in self.gloss_to_videos:
                    self.gloss_to_videos[gloss] = []
                self.gloss_to_videos[gloss].append(video_id)
                
                # Video ID -> gloss
                self.video_to_gloss[video_id] = gloss
            
            self.available_glosses = set(self.gloss_to_videos.keys())
            
            logger.info(f"Loaded {len(self.available_glosses)} glosses from vocabulary")
            logger.info(f"Available: {sorted(self.available_glosses)}")
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
    
    def is_available(self, gloss: str) -> bool:
        """Check if a gloss has available pose data."""
        return gloss.upper() in self.available_glosses
    
    def get_video_id(self, gloss: str) -> Optional[str]:
        """Get a video ID for a gloss (first available)."""
        gloss_upper = gloss.upper()
        videos = self.gloss_to_videos.get(gloss_upper, [])
        return videos[0] if videos else None
    
    def get_all_video_ids(self, gloss: str) -> List[str]:
        """Get all video IDs for a gloss."""
        return self.gloss_to_videos.get(gloss.upper(), [])
    
    def get_gloss_for_video(self, video_id: str) -> Optional[str]:
        """Get gloss name for a video ID."""
        return self.video_to_gloss.get(video_id)
    
    def list_available(self) -> List[str]:
        """List all available glosses."""
        return sorted(self.available_glosses)
    
    def get_stats(self) -> Dict:
        """Get vocabulary statistics."""
        return {
            "total_glosses": len(self.available_glosses),
            "total_videos": len(self.video_to_gloss),
            "glosses": self.list_available()
        }


# Global instance (lazy loaded)
_vocab_manager: Optional[VocabularyManager] = None


def get_vocabulary() -> VocabularyManager:
    """Get the global vocabulary manager instance."""
    global _vocab_manager
    if _vocab_manager is None:
        _vocab_manager = VocabularyManager()
    return _vocab_manager


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    vocab = get_vocabulary()
    print(f"\nAvailable glosses: {vocab.list_available()}")
    print(f"\nStats: {vocab.get_stats()}")
    
    # Test lookup
    test_glosses = ["YES", "HELLO", "CAN", "BEFORE"]
    for g in test_glosses:
        available = vocab.is_available(g)
        video_id = vocab.get_video_id(g)
        print(f"  {g}: {'✓' if available else '✗'} (video: {video_id})")
