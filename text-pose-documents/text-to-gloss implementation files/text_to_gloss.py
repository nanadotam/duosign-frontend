"""
Text-to-Gloss Converter
======================
Rule-based English to ASL gloss conversion using spaCy NLP.
"""

import spacy
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


# Vocabulary Loading
def load_vocabularies() -> Tuple[Dict, Dict]:
    """Load WLASL and ASL-LEX vocabularies"""
    
    # Sample vocabulary - replace with your actual vocabulary files
    wlasl_vocab = {
        "HELLO": {"id": "hello"},
        "DOCTOR": {"id": "doctor"},
        "SEARCH": {"id": "search"},
        "SCHOOL": {"id": "school"},
        "GO": {"id": "go"},
        "TOMORROW": {"id": "tomorrow"},
        "YESTERDAY": {"id": "yesterday"},
        "TODAY": {"id": "today"},
        "LIKE": {"id": "like"},
        "PIZZA": {"id": "pizza"},
        "NOT": {"id": "not"},
        "BATHROOM": {"id": "bathroom"},
        "WHERE": {"id": "where"},
        "WHAT": {"id": "what"},
        "WHEN": {"id": "when"},
        "WHO": {"id": "who"},
        "WHY": {"id": "why"},
        "HOW": {"id": "how"},
        "EAT": {"id": "eat"},
        "DRINK": {"id": "drink"},
        "WORK": {"id": "work"},
        "HOME": {"id": "home"},
        "FRIEND": {"id": "friend"},
        "FAMILY": {"id": "family"},
        "HAPPY": {"id": "happy"},
        "HELP": {"id": "help"},
        "THANK": {"id": "thank"},
        "SORRY": {"id": "sorry"},
        "YES": {"id": "yes"},
        "NO": {"id": "no"},
    }
    
    asl_lex_vocab = {
        "MORNING": {"id": "morning"},
        "AFTERNOON": {"id": "afternoon"},
        "EVENING": {"id": "evening"},
        "NIGHT": {"id": "night"},
    }
    
    logger.info(f"Loaded {len(wlasl_vocab)} WLASL + {len(asl_lex_vocab)} ASL-LEX signs")
    return wlasl_vocab, asl_lex_vocab


# spaCy Model
@lru_cache(maxsize=1)
def load_spacy():
    """Load spaCy model (cached)"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading spaCy model...")
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")


# ASL Grammar Rules
class ASLGrammar:
    """ASL grammar transformation rules"""
    
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
        'morning', 'afternoon', 'evening', 'night', 'week', 'month', 'year'
    }
    
    IDIOMS = {
        'raining cats and dogs': 'RAIN HEAVY',
        'piece of cake': 'EASY',
        'under the weather': 'SICK',
    }
    
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
    """Keep only content words"""
    keep_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'NUM', 'PROPN', 'PRON'}
    content = []
    
    for token in tokens:
        if token.is_punct:
            continue
        if token.pos_ in keep_pos:
            content.append(token)
        elif token.text.lower() in ['not', "n't", 'no']:
            content.append(token)
    
    return content


def extract_temporal_spatial(doc) -> Tuple[List, List]:
    """Extract temporal and spatial markers"""
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
        if token.dep_ == "neg" or token.text.lower() in ['not', "n't", 'no']:
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
    
    def __init__(self, wlasl_vocab: Dict, asl_lex_vocab: Dict):
        self.nlp = load_spacy()
        self.vocab = set(wlasl_vocab.keys()) | set(asl_lex_vocab.keys())
        self.vocab.update(['IX-1', 'IX-2', 'IX-3', 'IX-1+', 'IX-2+', 'IX-3+'])
        self.cache = {}
        self.stats = {"total": 0, "rule_based": 0, "cache_hits": 0}
        
        logger.info(f"Initialized with {len(self.vocab)} vocabulary items")
    
    def convert(self, text: str, use_llm_fallback: bool = False) -> Dict[str, Any]:
        """
        Convert text to gloss.
        
        Args:
            text: English text
            use_llm_fallback: Not implemented (for future use)
        
        Returns:
            {
                "gloss": "TOMORROW IX-1 SCHOOL GO",
                "tokens": [{"gloss": "TOMORROW", "word": "tomorrow", ...}, ...],
                "method": "rule_based",
                "confidence": 1.0
            }
        """
        self.stats["total"] += 1
        
        # Check cache
        if text in self.cache:
            self.stats["cache_hits"] += 1
            result = self.cache[text].copy()
            result["method"] = "cached"
            return result
        
        # Check for idioms
        idiom_gloss = ASLGrammar.check_idiom(text)
        if idiom_gloss:
            result = {
                "gloss": idiom_gloss,
                "tokens": [{"gloss": g, "word": text, "confidence": 1.0} 
                          for g in idiom_gloss.split()],
                "method": "rule_based",
                "confidence": 1.0
            }
            self.cache[text] = result
            return result
        
        # Parse with spaCy
        doc = self.nlp(text)
        
        # Extract temporal/spatial
        temporal, spatial = extract_temporal_spatial(doc)
        
        # Filter content
        content = filter_content(doc)
        
        # Build gloss tokens
        gloss_tokens = []
        
        # 1. Time first (temporal topicalization)
        for t in temporal:
            gloss_tokens.append({
                "word": t.text,
                "gloss": t.lemma_.upper(),
                "confidence": 1.0
            })
        
        # 2. Location/topic
        for s in spatial:
            gloss_tokens.append({
                "word": s.text,
                "gloss": s.lemma_.upper(),
                "confidence": 1.0
            })
        
        # 3. Subject
        subject = find_subject(content)
        if subject:
            if subject.pos_ == "PRON":
                gloss = ASLGrammar.PRONOUNS.get(subject.lemma_.lower(), subject.lemma_.upper())
            else:
                gloss = subject.lemma_.upper()
                if subject.morph.get("Number") == ["Plur"]:
                    gloss += "+"
            
            gloss_tokens.append({
                "word": subject.text,
                "gloss": gloss,
                "confidence": 1.0
            })
        
        # 4. Object
        obj = find_object(content)
        if obj:
            if obj.pos_ == "PRON":
                gloss = ASLGrammar.PRONOUNS.get(obj.lemma_.lower(), obj.lemma_.upper())
            else:
                gloss = obj.lemma_.upper()
                if obj.morph.get("Number") == ["Plur"]:
                    gloss += "+"
            
            gloss_tokens.append({
                "word": obj.text,
                "gloss": gloss,
                "confidence": 1.0
            })
        
        # 5. Verb
        verb = find_verb(content)
        if verb:
            lemma = verb.lemma_.upper()
            
            # Skip auxiliary verbs
            if lemma not in ['BE', 'AM', 'IS', 'ARE', 'WAS', 'WERE', 'DO', 'DOES', 'DID']:
                if lemma in ['HAVE', 'HAS', 'HAD'] and verb.dep_ == "aux":
                    pass  # Skip auxiliary have
                else:
                    gloss_tokens.append({
                        "word": verb.text,
                        "gloss": lemma,
                        "confidence": 1.0
                    })
        
        # 6. Negation at end
        negation = find_negation(content)
        if negation:
            gloss_tokens.append({
                "word": negation.text,
                "gloss": "NOT",
                "confidence": 1.0
            })
        
        # Calculate confidence
        unknown = sum(1 for t in gloss_tokens 
                     if t["gloss"] not in self.vocab and not t["gloss"].startswith("IX-"))
        confidence = (len(gloss_tokens) - unknown) / len(gloss_tokens) if gloss_tokens else 0
        
        # Build result
        gloss_string = ' '.join(t["gloss"] for t in gloss_tokens)
        
        result = {
            "gloss": gloss_string,
            "tokens": gloss_tokens,
            "method": "rule_based",
            "confidence": confidence
        }
        
        # Cache
        self.cache[text] = result
        self.stats["rule_based"] += 1
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversion statistics"""
        return {
            "total_requests": self.stats["total"],
            "rule_based": self.stats["rule_based"],
            "cache_hits": self.stats["cache_hits"],
            "cache_size": len(self.cache),
            "vocab_size": len(self.vocab)
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    wlasl, asl_lex = load_vocabularies()
    converter = TextToGlossConverter(wlasl, asl_lex)
    
    tests = [
        "I am searching for a doctor",
        "Tomorrow I will go to school",
        "She doesn't like pizza"
    ]
    
    for text in tests:
        result = converter.convert(text)
        print(f"\nInput: {text}")
        print(f"Gloss: {result['gloss']}")
        print(f"Tokens: {result['tokens']}")
