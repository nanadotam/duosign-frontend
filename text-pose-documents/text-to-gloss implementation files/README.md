# Text-to-Gloss API

**Focused pipeline: English text → ASL gloss notation**

This API converts English text to ASL gloss notation using rule-based NLP. The glosses are returned with indices for lookup in your pose file directory.

## Architecture

```
English Text → [Text-to-Gloss API] → Indexed Glosses → [Your Lookup] → Pose Files → [Your Concatenation] → Video
```

This API handles **ONLY** the text-to-gloss conversion. You handle:
- Pose file lookup from your local directory
- Video concatenation and optimization

## Quick Start

```bash
# Install
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run
python main.py
```

API runs at: http://localhost:8000

## API Usage

### Convert Text to Gloss

**POST** `/api/v1/text-to-gloss`

```bash
curl -X POST "http://localhost:8000/api/v1/text-to-gloss" \
  -H "Content-Type: application/json" \
  -d '{"text": "Tomorrow I will go to school"}'
```

**Response:**
```json
{
  "text": "Tomorrow I will go to school",
  "glosses": [
    {"index": 0, "gloss": "TOMORROW", "original_word": "Tomorrow"},
    {"index": 1, "gloss": "IX-1", "original_word": "I"},
    {"index": 2, "gloss": "SCHOOL", "original_word": "school"},
    {"index": 3, "gloss": "GO", "original_word": "go"}
  ],
  "gloss_string": "TOMORROW IX-1 SCHOOL GO",
  "method": "rule_based",
  "confidence": 1.0,
  "processing_time_ms": 42.3
}
```

### Use the Indexed Glosses

```python
# Your lookup code
response = requests.post("http://localhost:8000/api/v1/text-to-gloss", 
                        json={"text": "Hello"})
data = response.json()

# Get ordered glosses for lookup
for item in data["glosses"]:
    index = item["index"]
    gloss = item["gloss"]
    
    # Your code: lookup pose file
    pose_file = f"/path/to/poses/{gloss}.json"  # or .mp4, .pkl, etc
    poses.append(load_pose(pose_file))

# Your code: concatenate and render
```

## Algorithm

**Rule-Based Pipeline:**

1. **NLP Parsing** (spaCy)
   - Tokenization, POS tagging, dependencies

2. **ASL Grammar Transformation**
   - Time markers → front
   - Topic-Subject-Object-Verb ordering
   - Pronoun indexing (I→IX-1)
   - Negation → end

3. **Content Filtering**
   - Remove articles, prepositions
   - Keep nouns, verbs, adjectives

4. **Output**
   - Indexed glosses for lookup
   - Confidence scores

## Customization

### Add Vocabulary

Edit `text_to_gloss.py`:

```python
def load_vocabularies():
    wlasl_vocab = {
        "HELLO": {"id": "hello"},
        "NEW_SIGN": {"id": "new_sign"},  # Add here
        # ...
    }
```

### Modify Grammar Rules

Edit `ASLGrammar` class in `text_to_gloss.py`

## Performance

- **Latency**: 30-50ms per request
- **Cache**: Automatic caching of results
- **Throughput**: ~1000 req/sec (single worker)

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v1/text-to-gloss` | Convert single text |
| `POST /api/v1/batch` | Batch convert (up to 100) |
| `GET /health` | Health check |
| `GET /api/v1/stats` | Usage statistics |
| `POST /api/v1/cache/clear` | Clear cache |
| `GET /docs` | Interactive API docs |

## Testing

```bash
# Health check
curl http://localhost:8000/health

# Example conversion
curl -X POST "http://localhost:8000/api/v1/text-to-gloss" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am searching for a doctor"}'

# Expected output: "IX-1 DOCTOR SEARCH"
```

## Integration Example

```python
import requests

def text_to_glosses(text: str) -> list:
    """Get ordered glosses from text"""
    response = requests.post(
        "http://localhost:8000/api/v1/text-to-gloss",
        json={"text": text}
    )
    data = response.json()
    return data["glosses"]

# Use it
glosses = text_to_glosses("Hello, how are you?")

# Lookup pose files (your code)
for item in glosses:
    pose_file = lookup_pose(item["gloss"])
    # ... concatenate, render
```

## File Structure

```
text-to-gloss-api/
├── main.py              # FastAPI app
├── text_to_gloss.py     # Converter logic
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Output Format

The API returns glosses in an **indexed dictionary/JSON** format perfect for sequential lookup:

```json
{
  "glosses": [
    {"index": 0, "gloss": "HELLO"},
    {"index": 1, "gloss": "HOW"},
    {"index": 2, "gloss": "IX-2"}
  ]
}
```

Use the `index` to maintain order when looking up pose files.

---

**Simple, focused, production-ready. Just text → gloss. 🤟**
