"""
Tests for Text-to-Gloss Pipeline
================================
Unit tests for the text-to-gloss converter.

Run with: python -m pytest duosign_algo/api/tests/test_text_to_gloss.py -v
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestVocabulary:
    """Test vocabulary loading and lookup."""
    
    def test_vocabulary_loads(self):
        """Vocabulary should load from selection.json."""
        from vocabulary import get_vocabulary
        vocab = get_vocabulary()
        
        assert vocab is not None
        assert len(vocab.list_available()) > 0
    
    def test_known_glosses_available(self):
        """Known glosses from selection.json should be available."""
        from vocabulary import get_vocabulary
        vocab = get_vocabulary()
        
        # These are in the test subset
        known_glosses = ["YES", "NO", "HOW", "CAN", "WOMAN", "BEFORE"]
        
        for gloss in known_glosses:
            assert vocab.is_available(gloss), f"{gloss} should be available"
    
    def test_unknown_gloss_not_available(self):
        """Unknown glosses should not be available."""
        from vocabulary import get_vocabulary
        vocab = get_vocabulary()
        
        assert not vocab.is_available("XYZNOTEXIST")
    
    def test_get_video_id(self):
        """Should return video ID for available gloss."""
        from vocabulary import get_vocabulary
        vocab = get_vocabulary()
        
        video_id = vocab.get_video_id("YES")
        assert video_id is not None
        assert len(video_id) == 5  # WLASL video IDs are 5 digits


class TestTextToGlossConverter:
    """Test text-to-gloss conversion."""
    
    @pytest.fixture
    def converter(self):
        from text_to_gloss import TextToGlossConverter
        return TextToGlossConverter()
    
    def test_simple_word(self, converter):
        """Single word should convert correctly."""
        result = converter.convert("Yes")
        
        assert result["gloss_string"] is not None
        assert len(result["glosses"]) >= 1
    
    def test_temporal_fronting(self, converter):
        """Temporal markers should be moved to front."""
        result = converter.convert("I go tomorrow")
        
        # TOMORROW should be first
        assert result["gloss_string"].startswith("TOMORROW")
    
    def test_negation_end(self, converter):
        """Negation should be at end."""
        result = converter.convert("I don't like pizza")
        
        # Should end with NOT
        assert result["gloss_string"].endswith("NOT")
    
    def test_pronoun_indexing(self, converter):
        """Pronouns should be converted to IX notation."""
        result = converter.convert("I help you")
        
        assert "IX-1" in result["gloss_string"]  # I
        assert "IX-2" in result["gloss_string"]  # you
    
    def test_availability_info(self, converter):
        """Result should include availability info."""
        result = converter.convert("Can you help me?")
        
        assert "debug" in result
        assert "available_count" in result["debug"]
        assert "missing_count" in result["debug"]
        assert "missing_glosses" in result["debug"]
    
    def test_known_gloss_marked_available(self, converter):
        """Known glosses should be marked as available."""
        result = converter.convert("Yes")
        
        # Find YES in glosses
        yes_gloss = None
        for g in result["glosses"]:
            if g["gloss"] == "YES":
                yes_gloss = g
                break
        
        if yes_gloss:
            assert yes_gloss["available"] is True
            assert yes_gloss["video_id"] is not None
    
    def test_empty_string_handling(self, converter):
        """Empty string should return empty result."""
        result = converter.convert("")
        
        assert result["glosses"] == []
        assert result["gloss_string"] == ""
    
    def test_caching(self, converter):
        """Same input should use cache."""
        text = "Hello there"
        
        result1 = converter.convert(text)
        result2 = converter.convert(text)
        
        # Both should be equal
        assert result1["gloss_string"] == result2["gloss_string"]
        
        # Cache should have increased
        assert converter.stats["cache_hits"] >= 1


class TestASLGrammar:
    """Test ASL grammar rules."""
    
    def test_pronoun_mapping(self):
        from text_to_gloss import ASLGrammar
        
        assert ASLGrammar.PRONOUNS["i"] == "IX-1"
        assert ASLGrammar.PRONOUNS["you"] == "IX-2"
        assert ASLGrammar.PRONOUNS["she"] == "IX-3"
        assert ASLGrammar.PRONOUNS["we"] == "IX-1+"
    
    def test_temporal_detection(self):
        from text_to_gloss import ASLGrammar
        
        assert ASLGrammar.is_temporal("tomorrow")
        assert ASLGrammar.is_temporal("YESTERDAY")
        assert ASLGrammar.is_temporal("before")
        assert not ASLGrammar.is_temporal("pizza")
    
    def test_idiom_detection(self):
        from text_to_gloss import ASLGrammar
        
        result = ASLGrammar.check_idiom("It's raining cats and dogs")
        assert result == "RAIN HEAVY"
        
        result = ASLGrammar.check_idiom("Hello world")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
