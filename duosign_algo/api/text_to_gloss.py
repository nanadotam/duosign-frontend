"""
Text-to-Gloss Converter
=======================
Rule-based English to ASL gloss conversion using spaCy NLP.

Author: Nana Amoako
Date: February 2026
"""

import re
import spacy
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from functools import lru_cache
import logging

from .vocabulary import get_vocabulary, FINGERSPELL_ALPHABET

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
    
    # Map pronouns to actual ASL glosses (use glosses that exist in vocabulary)
    # Falls back to literal pronoun if available, otherwise uses ASL notation
    PRONOUNS = {
        'i': 'I', 'me': 'I', 'my': 'I', 'mine': 'I',
        'you': 'YOU', 'your': 'YOUR', 'yours': 'YOUR',
        'he': 'HE', 'him': 'HE', 'his': 'HE',
        'she': 'SHE', 'her': 'SHE', 'hers': 'SHE',
        'it': 'IT', 'its': 'IT',
        'we': 'WE', 'us': 'WE', 'our': 'WE', 'ours': 'WE',
        'they': 'THEY', 'them': 'THEY', 'their': 'THEY', 'theirs': 'THEY'
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
    
    # Contraction expansion mapping
    CONTRACTIONS = {
        "let's": "we go", "lets": "we go",
        "can't": "can not", "cannot": "can not",
        "won't": "will not", "wont": "will not",
        "don't": "do not", "dont": "do not",
        "doesn't": "does not", "doesnt": "does not",
        "didn't": "did not", "didnt": "did not",
        "i'm": "i am", "im": "i am",
        "you're": "you are", "youre": "you are",
        "we're": "we are", "were": "we are",
        "they're": "they are", "theyre": "they are",
        "he's": "he is", "hes": "he is",
        "she's": "she is", "shes": "she is",
        "it's": "it is", "its": "it is",
        "i've": "i have", "ive": "i have",
        "you've": "you have", "youve": "you have",
        "we've": "we have", "weve": "we have",
        "they've": "they have", "theyve": "they have",
        "i'll": "i will", "ill": "i will",
        "you'll": "you will", "youll": "you will",
        "we'll": "we will", "well": "we will",
        "they'll": "they will", "theyll": "they will",
        "i'd": "i would", "id": "i would",
        "you'd": "you would", "youd": "you would",
        "we'd": "we would", "wed": "we would",
        "they'd": "they would", "theyd": "they would",
        "isn't": "is not", "isnt": "is not",
        "aren't": "are not", "arent": "are not",
        "wasn't": "was not", "wasnt": "was not",
        "weren't": "were not", "werent": "were not",
        "haven't": "have not", "havent": "have not",
        "hasn't": "has not", "hasnt": "has not",
        "hadn't": "had not", "hadnt": "had not",
        "shouldn't": "should not", "shouldnt": "should not",
        "wouldn't": "would not", "wouldnt": "would not",
        "couldn't": "could not", "couldnt": "could not",
        "that's": "that is", "thats": "that is",
        "there's": "there is", "theres": "there is",
        "here's": "here is", "heres": "here is",
        "what's": "what is", "whats": "what is",
        "who's": "who is", "whos": "who is",
        "where's": "where is", "wheres": "where is",
    }
    
    # Synonym fallback mapping (primary word -> list of alternatives)
    SYNONYMS = {
        "GO": ["LEAVE", "WALK", "MOVE", "TRAVEL", "DEPART", "RUN"],
        "EAT": ["FOOD", "HUNGRY", "MEAL", "BREAKFAST", "LUNCH", "DINNER"],
        "HELP": ["ASSIST", "SUPPORT", "AID"],
        "WANT": ["WISH", "DESIRE", "NEED", "HOPE"],
        "LIKE": ["LOVE", "ENJOY", "PREFER", "FAVORITE"],
        "GOOD": ["GREAT", "FINE", "NICE", "OKAY", "WELL", "EXCELLENT"],
        "BAD": ["WRONG", "TERRIBLE", "AWFUL", "POOR"],
        "BIG": ["LARGE", "HUGE", "GIANT", "WIDE"],
        "SMALL": ["LITTLE", "TINY", "MINI", "SHORT"],
        "FAST": ["QUICK", "RAPID", "SPEED", "HURRY"],
        "SLOW": ["CAREFUL", "PATIENT", "WAIT"],
        "HAPPY": ["GLAD", "JOYFUL", "PLEASED", "EXCITED", "CELEBRATE"],
        "SAD": ["UPSET", "DEPRESSED", "DOWN", "UNHAPPY", "CRY"],
        "THINK": ["BELIEVE", "CONSIDER", "WONDER", "KNOW", "UNDERSTAND"],
        "SEE": ["LOOK", "WATCH", "VIEW", "OBSERVE", "NOTICE"],
        "SAY": ["TELL", "SPEAK", "TALK", "COMMUNICATE"],
        "COME": ["ARRIVE", "APPROACH", "VISIT"],
        "GIVE": ["OFFER", "PROVIDE", "SHARE"],
        "TAKE": ["GET", "GRAB", "RECEIVE", "ACCEPT"],
        "MAKE": ["CREATE", "BUILD", "PRODUCE", "PREPARE"],
        "USE": ["APPLY", "UTILIZE"],
        "FIND": ["SEARCH", "DISCOVER", "LOCATE"],
        "PUT": ["PLACE", "SET", "POSITION"],
        "SHOW": ["DEMONSTRATE", "DISPLAY", "PRESENT"],
    }
    
    # Skip words (articles, prepositions, auxiliaries)
    SKIP_WORDS = {'a', 'an', 'the', 'to', 'of', 'for', 'in', 'on', 'at', 'with', 'by'}
    # SKIP_VERBS = {'BE', 'AM', 'IS', 'ARE', 'WAS', 'WERE', 'DO', 'DOES', 'DID', 'WILL', 'WOULD', 'COULD', 'SHOULD'}
    
    # things to leave:
    SKIP_VERBS = {'BE', 'AM', 'IS', 'ARE', 'WAS', 'WERE'}
    
    @staticmethod
    def expand_contractions(text: str) -> str:
        """Expand contractions to full forms."""
        result = text
        for contraction, expansion in ASLGrammar.CONTRACTIONS.items():
            # Use word boundary regex for accurate replacement
            pattern = r'\b' + re.escape(contraction) + r'\b'
            result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
        return result
    
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
        
        # Expand contractions before parsing (e.g., "let's" -> "we go")
        expanded_text = ASLGrammar.expand_contractions(text)
        
        # Parse with spaCy
        doc = self.nlp(expanded_text)
        
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
    
    def _build_result(self, text: str, gloss_items: List[Dict], method: str, enable_fingerspelling: bool = True) -> Dict:
        """
        Build the final result dict with availability info.
        
        Priority order for unavailable glosses:
        1. Direct match in vocabulary
        2. Synonym lookup
        3. Fingerspelling (if enabled)
        4. Mark as missing
        """
        # Enrich with availability info, optionally fingerspelling
        glosses = []
        available_count = 0
        missing_glosses = []
        fingerspelled_count = 0
        synonym_used_count = 0
        
        # Special glosses that should not be fingerspelled or synonym-replaced (only NOT)
        special_glosses = {'NOT'}
        
        for i, item in enumerate(gloss_items):
            gloss = item["gloss"]
            is_available = self.vocab.is_available(gloss)
            video_id = self.vocab.get_video_id(gloss) if is_available else None
            
            if is_available:
                # Direct match found
                available_count += 1
                glosses.append({
                    "index": len(glosses),
                    "gloss": gloss,
                    "original_word": item["word"],
                    "available": True,
                    "video_id": video_id,
                    "fingerspelled": False,
                    "synonym_of": None
                })
            elif gloss not in special_glosses:
                # Try to find a synonym
                synonym = self._find_synonym(gloss)
                
                if synonym:
                    # Synonym found
                    available_count += 1
                    synonym_used_count += 1
                    glosses.append({
                        "index": len(glosses),
                        "gloss": synonym,
                        "original_word": item["word"],
                        "available": True,
                        "video_id": self.vocab.get_video_id(synonym),
                        "fingerspelled": False,
                        "synonym_of": gloss  # Track original word
                    })
                elif enable_fingerspelling:
                    # Fingerspell this word: expand to individual letters
                    word_letters = gloss.upper()
                    spelled_letters = []
                    for letter in word_letters:
                        if letter in FINGERSPELL_ALPHABET and self.vocab.is_available(letter):
                            spelled_letters.append(letter)
                            fingerspelled_count += 1
                    
                    if spelled_letters:
                        for letter in spelled_letters:
                            glosses.append({
                                "index": len(glosses),
                                "gloss": letter,
                                "original_word": item["word"],
                                "available": True,
                                "video_id": self.vocab.get_video_id(letter),
                                "fingerspelled": True,
                                "synonym_of": None
                            })
                        available_count += len(spelled_letters)
                    else:
                        # No letters available, mark as missing
                        missing_glosses.append(gloss)
                        glosses.append({
                            "index": len(glosses),
                            "gloss": gloss,
                            "original_word": item["word"],
                            "available": False,
                            "video_id": None,
                            "fingerspelled": False,
                            "synonym_of": None
                        })
                else:
                    # Neither synonym nor fingerspelling available
                    missing_glosses.append(gloss)
                    glosses.append({
                        "index": len(glosses),
                        "gloss": gloss,
                        "original_word": item["word"],
                        "available": False,
                        "video_id": None,
                        "fingerspelled": False,
                        "synonym_of": None
                    })
            else:
                # Special gloss (IX-1, NOT, etc.) - mark as missing but don't modify
                missing_glosses.append(gloss)
                glosses.append({
                    "index": len(glosses),
                    "gloss": gloss,
                    "original_word": item["word"],
                    "available": False,
                    "video_id": None,
                    "fingerspelled": False,
                    "synonym_of": None
                })
        
        gloss_string = ' '.join(g["gloss"] for g in glosses)
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
                "fingerspelled_count": fingerspelled_count,
                "total_count": total_count,
                "missing_glosses": missing_glosses,
                "available_glosses": [g["gloss"] for g in glosses if g["available"]]
            }
        }
    
    def _find_synonym(self, gloss: str) -> Optional[str]:
        """
        Find an available synonym for a missing gloss.
        
        Checks ASLGrammar.SYNONYMS for alternatives that are available in vocabulary.
        
        Args:
            gloss: The missing gloss to find a synonym for
            
        Returns:
            An available synonym, or None if no synonym found
        """
        gloss_upper = gloss.upper()
        
        # Check if this word has synonyms defined
        synonyms = ASLGrammar.SYNONYMS.get(gloss_upper, [])
        for syn in synonyms:
            if self.vocab.is_available(syn):
                logger.debug(f"Found synonym for {gloss_upper}: {syn}")
                return syn
        
        # Check reverse mapping: is this word a synonym for something available?
        for root, syns in ASLGrammar.SYNONYMS.items():
            if gloss_upper in syns and self.vocab.is_available(root):
                logger.debug(f"Found root word for {gloss_upper}: {root}")
                return root
        
        return None
    
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
