"""
Text-to-Gloss Converter
=======================
Rule-based English to ASL gloss conversion using spaCy NLP.

Author: Nana Amoako
Date: February 2026
"""

import spacy
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from functools import lru_cache
import logging

from .vocabulary import get_vocabulary

logger = logging.getLogger(__name__)


# spaCy Model
@lru_cache(maxsize=1)
def load_spacy():
    """Load spaCy model (cached)."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading spaCy model...")
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")


# ASL Grammar Rules
class ASLGrammar:
    """ASL grammar transformation rules."""
    
    PRONOUNS = {
        'i': 'IX-1', 'me': 'IX-1', 'my': 'IX-1', 'mine': 'IX-1',
        'you': 'IX-2', 'your': 'IX-2', 'yours': 'IX-2',
        'he': 'IX-3', 'him': 'IX-3', 'his': 'IX-3',
        'she': 'IX-3', 'her': 'IX-3', 'hers': 'IX-3',
        'it': 'IX-3', 'its': 'IX-3',
        'we': 'IX-1+', 'us': 'IX-1+', 'our': 'IX-1+', 'ours': 'IX-1+',
        'they': 'IX-3+', 'them': 'IX-3+', 'their': 'IX-3+', 'theirs': 'IX-3+'
    }
    
    TEMPORAL = {
        'yesterday', 'today', 'tomorrow', 'now', 'then', 'later',
        'morning', 'afternoon', 'evening', 'night', 'week', 'month', 'year',
        'before', 'after', 'always', 'never', 'sometimes'
    }
    
    IDIOMS = {
        'raining cats and dogs': 'RAIN HEAVY',
        'piece of cake': 'EASY',
        'under the weather': 'SICK',
    }
    
    # Skip words (articles, prepositions, auxiliaries)
    SKIP_WORDS = {'a', 'an', 'the', 'to', 'of', 'for', 'in', 'on', 'at', 'with', 'by'}
    SKIP_VERBS = {'BE', 'AM', 'IS', 'ARE', 'WAS', 'WERE', 'DO', 'DOES', 'DID', 'WILL', 'WOULD', 'COULD', 'SHOULD'}
    
    @staticmethod
    def is_temporal(word: str) -> bool:
        return word.lower() in ASLGrammar.TEMPORAL
    
    @staticmethod
    def check_idiom(text: str) -> Optional[str]:
        text_lower = text.lower()
        for idiom, gloss in ASLGrammar.IDIOMS.items():
            if idiom in text_lower:
                return gloss
        return None


# Helper Functions
def filter_content(tokens) -> List:
    """Keep only content words."""
    # Include INTJ for standalone words like 'Yes', 'No', 'Hello'
    keep_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'NUM', 'PROPN', 'PRON', 'INTJ'}
    content = []
    
    for token in tokens:
        if token.is_punct:
            continue
        if token.text.lower() in ASLGrammar.SKIP_WORDS:
            continue
        if token.pos_ in keep_pos:
            content.append(token)
        # Only treat as negation if it's marked as negation (not standalone 'no')
        elif token.dep_ == "neg" or token.text.lower() in ["n't", "not"]:
            content.append(token)
    
    return content


def extract_temporal_spatial(doc) -> Tuple[List, List]:
    """Extract temporal and spatial markers."""
    temporal = []
    spatial = []
    
    for token in doc:
        if ASLGrammar.is_temporal(token.text):
            temporal.append(token)
        if token.ent_type_ in ["GPE", "LOC", "FAC"]:
            spatial.append(token)
    
    return temporal, spatial


def find_subject(tokens) -> Optional[Any]:
    for token in tokens:
        if token.dep_ in ["nsubj", "nsubjpass"]:
            return token
    return None


def find_object(tokens) -> Optional[Any]:
    for token in tokens:
        if token.dep_ in ["dobj", "pobj", "attr"]:
            return token
    return None


def find_verb(tokens) -> Optional[Any]:
    for token in tokens:
        if token.pos_ == "VERB" and token.dep_ not in ["aux", "auxpass"]:
            return token
    return None


def find_negation(tokens) -> Optional[Any]:
    for token in tokens:
        # Only match actual negation markers, not standalone 'no' (which is INTJ)
        if token.dep_ == "neg" or token.text.lower() in ["n't", "not"]:
            return token
    return None


# Main Converter
class TextToGlossConverter:
    """
    Convert English text to ASL gloss using rule-based NLP.
    
    Pipeline:
    1. Parse with spaCy
    2. Extract temporal/spatial markers
    3. Filter content words
    4. Apply ASL grammar (Time-Topic-Subject-Object-Verb-Negation)
    5. Check vocabulary coverage
    """
    
    def __init__(self):
        self.nlp = load_spacy()
        self.vocab = get_vocabulary()
        self.cache = {}
        self.stats = {"total": 0, "cache_hits": 0}
        
        logger.info(f"Initialized TextToGlossConverter with {len(self.vocab.list_available())} available glosses")
    
    def convert(self, text: str) -> Dict[str, Any]:
        """
        Convert text to gloss.
        
        Args:
            text: English text
        
        Returns:
            {
                "text": "...",
                "gloss_string": "TOMORROW IX-1 SCHOOL GO",
                "glosses": [{"index": 0, "gloss": "TOMORROW", "available": true, "video_id": "..."}, ...],
                "method": "rule_based",
                "confidence": 0.8,
                "debug": {"available_count": 3, "missing_count": 1, ...}
            }
        """
        self.stats["total"] += 1
        
        # Check cache
        if text in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[text]
        
        # Check for idioms
        idiom_gloss = ASLGrammar.check_idiom(text)
        if idiom_gloss:
            tokens = self._build_gloss_tokens(idiom_gloss.split(), text)
            return self._build_result(text, tokens, "idiom")
        
        # Parse with spaCy
        doc = self.nlp(text)
        
        # Extract temporal/spatial
        temporal, spatial = extract_temporal_spatial(doc)
        
        # Filter content
        content = filter_content(doc)
        
        # Build gloss tokens in ASL order
        gloss_items = []
        seen_glosses = set()
        
        # 1. Time first (temporal topicalization)
        for t in temporal:
            gloss = t.lemma_.upper()
            if gloss not in seen_glosses:
                gloss_items.append({"word": t.text, "gloss": gloss})
                seen_glosses.add(gloss)
        
        # 2. Location/topic
        for s in spatial:
            gloss = s.lemma_.upper()
            if gloss not in seen_glosses:
                gloss_items.append({"word": s.text, "gloss": gloss})
                seen_glosses.add(gloss)
        
        # 3. Subject
        subject = find_subject(content)
        if subject and subject.text.lower() not in seen_glosses:
            if subject.pos_ == "PRON":
                gloss = ASLGrammar.PRONOUNS.get(subject.lemma_.lower(), subject.lemma_.upper())
            else:
                gloss = subject.lemma_.upper()
                if subject.morph.get("Number") == ["Plur"]:
                    gloss += "+"
            gloss_items.append({"word": subject.text, "gloss": gloss})
            seen_glosses.add(gloss)
        
        # 4. Object
        obj = find_object(content)
        if obj and obj.text.lower() not in seen_glosses:
            if obj.pos_ == "PRON":
                gloss = ASLGrammar.PRONOUNS.get(obj.lemma_.lower(), obj.lemma_.upper())
            else:
                gloss = obj.lemma_.upper()
                if obj.morph.get("Number") == ["Plur"]:
                    gloss += "+"
            gloss_items.append({"word": obj.text, "gloss": gloss})
            seen_glosses.add(gloss)
        
        # 5. Verb
        verb = find_verb(content)
        if verb:
            lemma = verb.lemma_.upper()
            if lemma not in ASLGrammar.SKIP_VERBS and lemma not in seen_glosses:
                gloss_items.append({"word": verb.text, "gloss": lemma})
                seen_glosses.add(lemma)
        
        # 6. Negation at end
        negation = find_negation(content)
        if negation and "NOT" not in seen_glosses:
            gloss_items.append({"word": negation.text, "gloss": "NOT"})
        
        # Handle cases where no structure was found - just use content words
        if not gloss_items:
            for token in content:
                lemma = token.lemma_.upper()
                if lemma not in ASLGrammar.SKIP_VERBS and lemma not in seen_glosses:
                    if token.pos_ == "PRON":
                        gloss = ASLGrammar.PRONOUNS.get(token.lemma_.lower(), lemma)
                    else:
                        gloss = lemma
                    gloss_items.append({"word": token.text, "gloss": gloss})
                    seen_glosses.add(gloss)
        
        result = self._build_result(text, gloss_items, "rule_based")
        
        # Cache
        self.cache[text] = result
        
        return result
    
    def _build_gloss_tokens(self, glosses: List[str], original_text: str) -> List[Dict]:
        """Build gloss tokens from a list of gloss strings."""
        return [{"word": original_text, "gloss": g} for g in glosses]
    
    def _build_result(self, text: str, gloss_items: List[Dict], method: str) -> Dict:
        """Build the final result dict with availability info."""
        # Enrich with availability info
        glosses = []
        available_count = 0
        missing_glosses = []
        
        for i, item in enumerate(gloss_items):
            gloss = item["gloss"]
            is_available = self.vocab.is_available(gloss)
            video_id = self.vocab.get_video_id(gloss) if is_available else None
            
            if is_available:
                available_count += 1
            else:
                missing_glosses.append(gloss)
            
            glosses.append({
                "index": i,
                "gloss": gloss,
                "original_word": item["word"],
                "available": is_available,
                "video_id": video_id
            })
        
        gloss_string = ' '.join(item["gloss"] for item in gloss_items)
        total_count = len(glosses)
        
        return {
            "text": text,
            "gloss_string": gloss_string,
            "glosses": glosses,
            "method": method,
            "confidence": available_count / total_count if total_count > 0 else 0,
            "debug": {
                "available_count": available_count,
                "missing_count": len(missing_glosses),
                "total_count": total_count,
                "missing_glosses": missing_glosses,
                "available_glosses": [g["gloss"] for g in glosses if g["available"]]
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversion statistics."""
        return {
            "total_requests": self.stats["total"],
            "cache_hits": self.stats["cache_hits"],
            "cache_size": len(self.cache),
            "available_vocabulary": self.vocab.list_available()
        }


# Global instance (lazy loaded)
_converter: Optional[TextToGlossConverter] = None


def get_converter() -> TextToGlossConverter:
    """Get the global converter instance."""
    global _converter
    if _converter is None:
        _converter = TextToGlossConverter()
    return _converter


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    converter = TextToGlossConverter()
    
    tests = [
        "Hello",
        "Yes",
        "No",
        "Can you help me?",
        "Tomorrow I will go to school",
        "She doesn't like pizza",
        "I am searching for a doctor",
        "The woman is thin",
        "How can I help you?",
    ]
    
    print("\n" + "=" * 70)
    print("Text-to-Gloss Converter Test")
    print("=" * 70)
    
    for text in tests:
        result = converter.convert(text)
        print(f"\n📝 Input: {text}")
        print(f"   Gloss: {result['gloss_string']}")
        print(f"   Available: {result['debug']['available_count']}/{result['debug']['total_count']}")
        if result['debug']['missing_glosses']:
            print(f"   Missing: {', '.join(result['debug']['missing_glosses'])}")
        print(f"   Method: {result['method']}")
