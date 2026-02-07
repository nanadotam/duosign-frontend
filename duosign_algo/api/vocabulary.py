"""
Vocabulary Management for Text-to-Gloss Pipeline
=================================================
Load available gloss vocabulary from public/lexicon/ase/*.json files.

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
    Manages the available gloss vocabulary from JSON pose files.
    
    Scans public/lexicon/ase/*.json to find available glosses.
    """
    
    def __init__(self, lexicon_dir: Optional[Path] = None):
        """
        Initialize vocabulary manager.
        
        Args:
            lexicon_dir: Path to lexicon directory. If None, auto-detects.
        """
        self.lexicon_dir = lexicon_dir or self._find_lexicon_dir()
        self.gloss_to_info: Dict[str, Dict] = {}
        self.available_glosses: Set[str] = set()
        
        self._load_vocabulary()
    
    def _find_lexicon_dir(self) -> Path:
        """Find the public/lexicon/ase directory."""
        candidates = [
            Path(__file__).parent.parent.parent / "public" / "lexicon" / "ase",
            Path("public/lexicon/ase"),
            Path("../public/lexicon/ase"),
        ]
        
        for path in candidates:
            if path.exists():
                logger.info(f"Found lexicon directory at: {path}")
                return path
        
        # Return default even if doesn't exist (will log warning)
        logger.warning(f"Lexicon directory not found, tried: {candidates}")
        return candidates[0]
    
    def _load_vocabulary(self) -> None:
        """Load vocabulary by scanning JSON files in lexicon directory."""
        if not self.lexicon_dir.exists():
            logger.warning(f"Lexicon directory not found: {self.lexicon_dir}")
            return
        
        try:
            # Scan all .json files in the lexicon directory
            json_files = list(self.lexicon_dir.glob("*.json"))
            
            for json_file in json_files:
                # Skip metadata files (starting with _)
                if json_file.name.startswith("_"):
                    continue
                
                gloss_name = json_file.stem  # filename without extension
                gloss_upper = gloss_name.upper()
                
                self.gloss_to_info[gloss_upper] = {
                    "json_file": json_file.name,
                    "gloss_id": gloss_name,
                    "path": str(json_file),
                }
                self.available_glosses.add(gloss_upper)
            
            # Mark alphabet letters that are available
            for letter in FINGERSPELL_ALPHABET:
                if letter in self.available_glosses:
                    self.gloss_to_info[letter]["is_letter"] = True
            
            logger.info(f"Loaded {len(self.available_glosses)} glosses from {self.lexicon_dir}")
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
    
    def is_available(self, gloss: str) -> bool:
        """Check if a gloss has available pose data."""
        return gloss.upper() in self.available_glosses
    
    def is_letter(self, gloss: str) -> bool:
        """Check if a gloss is a single letter (for fingerspelling)."""
        return gloss.upper() in FINGERSPELL_ALPHABET
    
    def get_video_id(self, gloss: str) -> Optional[str]:
        """Get a video ID/file for a gloss (returns the gloss name for loading)."""
        gloss_upper = gloss.upper()
        info = self.gloss_to_info.get(gloss_upper)
        if info:
            # Return the gloss_id (lowercase filename) for API compatibility
            return info.get("gloss_id", gloss.lower())
        return None
    
    def get_pose_filename(self, gloss: str) -> Optional[str]:
        """Get the .json filename for a gloss."""
        gloss_upper = gloss.upper()
        info = self.gloss_to_info.get(gloss_upper)
        if info:
            return info.get("json_file")
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

