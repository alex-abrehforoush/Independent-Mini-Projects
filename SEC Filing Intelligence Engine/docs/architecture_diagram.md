# System Architecture

## High-Level Flow
```
┌─────────────────────────────────────────────────────────────────────┐
│                        SEC FILING INTELLIGENCE ENGINE                │
└─────────────────────────────────────────────────────────────────────┘

INPUT LAYER
───────────
┌─────────────────┐
│  SEC 10-K HTML  │  ← Downloaded from EDGAR
│  (e.g., Apple)  │
└────────┬────────┘
         │
         ▼

PARSING LAYER
─────────────
┌─────────────────────────────────────────────────────────────┐
│  Document Parser (BeautifulSoup)                            │
│  ─────────────────────────────────────────────────────────  │
│  • Extract text from HTML                                   │
│  • Remove XBRL metadata                                     │
│  • Identify section boundaries (Item 1A, Item 7, etc.)     │
│  • Extract: Risk Factors, Financial Performance sections   │
└────────┬────────────────────────────────────────────────────┘
         │
         ├─────────────────────┬─────────────────────┐
         ▼                     ▼                     ▼
    Risk Text           Revenue Text          Metrics Text
    (69K chars)         (5K chars)            (10K chars)
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                               │
                               ▼

EXTRACTION LAYER
────────────────
┌─────────────────────────────────────────────────────────────┐
│  LLM Extractor (Azure OpenAI GPT-4o-mini)                   │
│  ─────────────────────────────────────────────────────────  │
│  • Temperature = 0 (deterministic)                          │
│  • Structured outputs (Pydantic schemas)                    │
│  • Concurrent extractions (3 API calls)                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Risk         │  │ Revenue      │  │ Financial    │     │
│  │ Extractor    │  │ Extractor    │  │ Extractor    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│         └─────────────────┴─────────────────┘              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼

GOVERNANCE LAYER (Optional)
───────────────────────────
┌─────────────────────────────────────────────────────────────┐
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ PII          │  │ Confidence   │  │ Data         │     │
│  │ Detection    │  │ Scoring      │  │ Validation   │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│         └─────────────────┴─────────────────┘              │
│                           │                                 │
│                           ▼                                 │
│              ┌────────────────────────┐                     │
│              │   Audit Logger         │                     │
│              │   (JSONL format)       │                     │
│              └────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼

OUTPUT LAYER
────────────
┌─────────────────────────────────────────────────────────────┐
│  Structured JSON Output                                      │
│  ─────────────────────────────────────────────────────────  │
│  {                                                           │
│    "risks": [...],          # 10 categorized risks          │
│    "revenue": {             # 5 product segments            │
│      "segments": [...],                                      │
│      "total": 416161.0                                       │
│    },                                                        │
│    "financials": [...],     # Key metrics                   │
│    "metadata": {                                             │
│      "governance": {...},   # Audit trail, validations      │
│      "cost": 0.0021         # USD per extraction            │
│    }                                                         │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow Details

### 1. Document Parsing Strategy

**Challenge**: SEC 10-Ks contain mixed content:
- Human-readable narrative (what we need)
- XBRL metadata tags (noise for LLMs)
- Inconsistent section formatting

**Solution**:
```python
# Multi-pattern matching for robustness
patterns = [
    r'Item\s*1A\.\s*Risk\s*Factors\s+The\s+following',
    r'following\s+table\s+shows\s+net\s+sales\s+by\s+category'
]

# Validate extraction quality
if len(extracted_text) < 1000:
    # Likely caught table of contents, not actual content
    fallback_to_alternative_pattern()
```

### 2. LLM Extraction Pipeline

**Why structured outputs?**
- Prevents hallucination through schema enforcement
- Type safety (revenue must be float, not string)
- Consistent output format for downstream processing

**Token optimization**:
```
Risk section:    30,000 chars → ~7,500 tokens
Revenue section:  8,000 chars → ~2,000 tokens  
Financial:       10,000 chars → ~2,500 tokens
─────────────────────────────────────────────
Total input:                   ~12,000 tokens
Structured output:              ~1,500 tokens
─────────────────────────────────────────────
Cost: $0.0021 per 10-K
```

### 3. Governance Controls

**Audit Log Format** (JSONL - one JSON per line):
```json
{
  "timestamp": "2026-01-20T16:07:06.810348",
  "operation": "extract_revenue",
  "document_id": "AAPL-10K-FY2025",
  "input_hash": "af82f395...",  // SHA256 for reproducibility
  "output_data": {"segment_count": 5},
  "metadata": {"model": "gpt-4o-mini", "temperature": 0}
}
```

**Why this matters for RBC**:
- **Regulatory compliance**: MiFID II/Dodd-Frank require audit trails
- **Reproducibility**: Input hash allows re-running exact extraction
- **Debugging**: Track which documents cause extraction failures
- **Cost attribution**: Per-document cost tracking for chargebacks

### 4. Validation Pipeline
```
Extracted Data
      ↓
┌──────────────────────────────────────┐
│ 1. Schema Validation (Pydantic)     │ ← Type checking
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ 2. Reasonableness Checks             │ ← Revenue > 0, YoY < 100%
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ 3. Confidence Scoring                │ ← Field completeness, value validity
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ 4. Ground Truth Comparison (if avail)│ ← Accuracy metrics
└──────────────────────────────────────┘
```

---

## Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **LLM** | GPT-4o-mini | Cost ($0.15/1M tokens) vs quality for extraction tasks |
| **Framework** | LangChain | Structured outputs, multi-provider support |
| **Parser** | BeautifulSoup | Handles messy SEC HTML better than regex |
| **Schema** | Pydantic | Type safety, validation, easy serialization |
| **UI** | Streamlit | Rapid prototyping, no frontend needed |
| **Logging** | JSONL | Line-oriented for streaming, easy to parse |

---

## Performance Characteristics

**Latency** (Apple 10-K, 416K revenue):
- Document parsing: ~0.5s
- Risk extraction: ~2.5s (30K chars)
- Revenue extraction: ~1.5s (8K chars)
- Financial extraction: ~1.5s (10K chars)
- **Total**: ~6 seconds end-to-end

**Accuracy** (validated against manual review):
- Revenue segments: 100% (5/5 exact matches)
- Risk categories: 80% (4/5 key risks identified)
- Financial metrics: 100% (5/5 exact matches)

**Cost**:
- Per 10-K: $0.0021
- Per 1000 10-Ks: $2.10
- Scales linearly (no batching in current implementation)

---

## Failure Modes & Mitigations

| Failure Mode | Cause | Mitigation |
|--------------|-------|------------|
| **XBRL contamination** | Parser extracts metadata instead of narrative | Multi-pattern matching, content length validation |
| **Hallucinated numbers** | LLM invents data when input is unclear | Strict prompts ("only extract stated numbers"), validation checks |
| **Missing sections** | Non-standard 10-K formatting | Fallback patterns, manual flagging for review |
| **Encoding errors** | Non-UTF-8 characters | Multi-encoding attempt (UTF-8, Latin-1, CP1252) |
| **Rate limiting** | Too many API calls | Built-in retry logic (not yet implemented but needed for production) |

---

## Future Architecture Enhancements

1. **Async processing**: Concurrent API calls to reduce latency to ~3s
2. **Caching layer**: Redis for repeated document extractions
3. **RAG integration**: Vector DB for multi-document Q&A
4. **Model comparison**: A/B test GPT vs Claude vs fine-tuned model
5. **Real-time pipeline**: Monitor SEC RSS feed, extract on filing