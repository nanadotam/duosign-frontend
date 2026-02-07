"""
Vocabulary Management for Text-to-Gloss Pipeline
=================================================
Load available gloss vocabulary from selected_poses/gloss_video_mapping.json.

Author: Nana Amoako
Date: February 2026
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


# Alphabet for fingerspelling (letters A-Z)
FINGERSPELL_ALPHABET = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


class VocabularyManager:
    """
    Manages the available gloss vocabulary from pose files.
    
    Loads from gloss_video_mapping.json which maps gloss names to video info.
    """
    
    def __init__(self, mapping_json_path: Optional[Path] = None):
        """
        Initialize vocabulary manager.
        
        Args:
            mapping_json_path: Path to gloss_video_mapping.json. If None, uses default.
        """
        self.mapping_path = mapping_json_path or self._find_mapping_json()
        self.gloss_to_info: Dict[str, Dict] = {}
        self.available_glosses: Set[str] = set()
        
        self._load_vocabulary()
    
    def _find_mapping_json(self) -> Path:
        """Find gloss_video_mapping.json in expected locations."""
        candidates = [
            Path(__file__).parent.parent / "selected_poses" / "gloss_video_mapping.json",
            Path("duosign_algo/selected_poses/gloss_video_mapping.json"),
            Path("selected_poses/gloss_video_mapping.json"),
        ]
        
        for path in candidates:
            if path.exists():
                logger.info(f"Found vocabulary at: {path}")
                return path
        
        # Return default even if doesn't exist (will log warning)
        return candidates[0]
    
    def _load_vocabulary(self) -> None:
        """Load vocabulary from gloss_video_mapping.json."""
        if not self.mapping_path.exists():
            logger.warning(f"Vocabulary file not found: {self.mapping_path}")
            return
        
        try:
            with open(self.mapping_path, 'r') as f:
                data = json.load(f)
            
            # gloss_video_mapping.json format:
            # { "hello": { "best_video": "123.pose", "gloss_id": "hello", ... }, ... }
            for gloss_name, info in data.items():
                gloss_upper = gloss_name.upper()
                self.gloss_to_info[gloss_upper] = {
                    "video_file": info.get("best_video", ""),
                    "gloss_id": info.get("gloss_id", gloss_name),
                    "detection_rate": info.get("detection_rate", 0),
                    "confidence_avg": info.get("confidence_avg", 0),
                    "alternatives": info.get("alternatives", []),
                }
            
            self.available_glosses = set(self.gloss_to_info.keys())
            
            # Also add alphabet letters for fingerspelling
            # These will use .pose files named a.pose, b.pose, etc.
            for letter in FINGERSPELL_ALPHABET:
                if letter not in self.available_glosses:
                    # Check if letter.pose exists
                    letter_lower = letter.lower()
                    if letter_lower in data:
                        self.gloss_to_info[letter] = {
                            "video_file": data[letter_lower].get("best_video", f"{letter_lower}.pose"),
                            "gloss_id": letter,
                            "detection_rate": data[letter_lower].get("detection_rate", 0),
                            "confidence_avg": data[letter_lower].get("confidence_avg", 0),
                            "alternatives": [],
                        }
                        self.available_glosses.add(letter)
            
            logger.info(f"Loaded {len(self.available_glosses)} glosses from vocabulary")
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
    
    def is_available(self, gloss: str) -> bool:
        """Check if a gloss has available pose data."""
        return gloss.upper() in self.available_glosses
    
    def is_letter(self, gloss: str) -> bool:
        """Check if a gloss is a single letter (for fingerspelling)."""
        return gloss.upper() in FINGERSPELL_ALPHABET
    
    def get_video_id(self, gloss: str) -> Optional[str]:
        """Get a video ID/file for a gloss."""
        gloss_upper = gloss.upper()
        info = self.gloss_to_info.get(gloss_upper)
        if info:
            video_file = info.get("video_file", "")
            # Return just the filename without extension for API compatibility
            return video_file.replace(".pose", "") if video_file else None
        return None
    
    def get_pose_filename(self, gloss: str) -> Optional[str]:
        """Get the .pose filename for a gloss."""
        gloss_upper = gloss.upper()
        info = self.gloss_to_info.get(gloss_upper)
        if info:
            return info.get("video_file")
        return None
    
    def get_gloss_info(self, gloss: str) -> Optional[Dict]:
        """Get full info for a gloss."""
        return self.gloss_to_info.get(gloss.upper())
    
    def list_available(self) -> List[str]:
        """List all available glosses."""
        return sorted(self.available_glosses)
    
    def search(self, query: str, limit: int = 50) -> List[str]:
        """Search glosses by prefix."""
        query_upper = query.upper()
        matches = [g for g in self.available_glosses if g.startswith(query_upper)]
        return sorted(matches)[:limit]
    
    def get_stats(self) -> Dict:
        """Get vocabulary statistics."""
        return {
            "total_glosses": len(self.available_glosses),
            "has_alphabet": all(l in self.available_glosses for l in FINGERSPELL_ALPHABET),
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


def reset_vocabulary() -> None:
    """Reset the global vocabulary manager (useful after file changes)."""
    global _vocab_manager
    _vocab_manager = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    vocab = get_vocabulary()
    print(f"\nAvailable glosses: {len(vocab.list_available())}")
    print(f"\nFirst 20: {vocab.list_available()[:20]}")
    print(f"\nStats: {vocab.get_stats()}")
    
    # Test lookup
    test_glosses = ["HELLO", "FRIEND", "YES", "A", "B", "C"]
    for g in test_glosses:
        available = vocab.is_available(g)
        video_id = vocab.get_video_id(g)
        print(f"  {g}: {'✓' if available else '✗'} (video: {video_id})")

