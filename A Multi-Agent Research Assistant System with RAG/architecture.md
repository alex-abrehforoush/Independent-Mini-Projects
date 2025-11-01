# Architecture & Technical Design

## System Overview

The Multi-Agent Research Assistant is built on a modular, production-ready architecture that separates concerns and enables independent scaling of components.

## Core Components

### 1. Agent Layer

#### Coordinator Agent
**Responsibility**: Query decomposition and research planning

**Design Pattern**: Decomposition strategy
- Receives high-level research questions
- Uses GPT-3.5/4 to break down into 3-5 specific subtasks
- Outputs structured ResearchPlan with prioritized tasks
- Employs JSON-constrained output for reliable parsing

**Why this matters**: Complex questions require structured decomposition. A single-shot query often misses nuances that emerge from systematic breakdown.

**Trade-offs**:
- Pro: Better coverage of multi-faceted questions
- Pro: Enables parallel research execution
- Con: Additional API call and latency
- Mitigation: Use cheaper GPT-3.5 for planning, reserve GPT-4 for synthesis

#### RAG Researcher Agent
**Responsibility**: Information retrieval and grounded answer generation

**Design Pattern**: Retrieval-Augmented Generation
- Embeds query using text-embedding-ada-002
- Performs semantic search in ChromaDB (top-k=5, cosine similarity)
- Retrieves document chunks with relevance > 0.3 threshold
- Constructs grounded prompts with explicit citation instructions
- Generates answers based on retrieved context

**Why RAG over fine-tuning**:
- Dynamic knowledge: Documents can be updated without retraining
- Source attribution: Critical for trust in financial/research contexts
- Cost-effective: No fine-tuning compute costs
- Hallucination reduction: Answers grounded in actual documents

**Vector Search Strategy**:
```python
Query → Embedding (1536-dim) → ChromaDB HNSW index → Top-K results → Filter by threshold
```

**Optimization considerations**:
- Chunk size: 1000 chars with 200-char overlap (context preservation)
- Top-k: 5 (balance between context richness and token limits)
- Threshold: 0.3 (tuned empirically, adjustable for precision/recall)

#### Analyzer Agent
**Responsibility**: Insight extraction and information synthesis

**Design Pattern**: Structured analysis with confidence scoring
- Receives all search results from Researcher
- Uses lower temperature (0.5) for consistent analysis
- Outputs JSON-structured insights with:
  - Theme categorization
  - Confidence levels (high/medium/low)
  - Supporting source attribution

**Why separate analyzer**: 
- Separation of concerns: Retrieval vs. reasoning
- Enables different prompting strategies
- Facilitates A/B testing of analysis approaches

#### Writer Agent
**Responsibility**: Report synthesis and presentation

**Design Pattern**: Structured document generation
- Synthesizes analyzed insights into narrative form
- Maintains proper citation format
- Generates executive summary + detailed findings
- Optimizes for readability and coherence

**Design choice**: Why not combine Analyzer + Writer?
- Modularity: Each can be improved independently
- Debugging: Easier to identify quality issues
- Cost optimization: Writer uses more tokens; can skip if only insights needed

### 2. Data Layer

#### Document Processing Pipeline

```
Input Document (PDF/TXT)
    ↓
Text Extraction (PyPDF2)
    ↓
Semantic Chunking (overlapping windows)
    ↓
Embedding Generation (batch processing)
    ↓
Vector Storage (ChromaDB)
```

**Chunking Strategy**:
- Base size: 1000 characters
- Overlap: 200 characters
- Boundary detection: Sentence-aware splitting
- Metadata: Source, page number, chunk index

**Why this chunking approach**:
- 1000 chars ≈ 250 tokens: Fits well within context windows
- Overlap preserves context across chunks
- Sentence boundaries prevent mid-sentence cuts

**Embedding Batching**:
- Batch size: 16 (Azure OpenAI limit)
- Async processing: Multiple batches in parallel
- Cost: ~$0.0001 per 1K tokens (very cheap)

#### Vector Database (ChromaDB)

**Why ChromaDB**:
- Embedded option: No separate infrastructure
- Persistent storage: Data survives restarts
- HNSW indexing: Fast approximate nearest neighbor search
- Open source: No vendor lock-in

**Production alternatives**:
- Azure Cognitive Search: Managed, enterprise-ready
- Pinecone: Fully managed vector DB
- Weaviate: Open-source with cloud options

**Index Configuration**:
- Distance metric: Cosine similarity
- Index type: HNSW (Hierarchical Navigable Small World)
- Persistence: DuckDB backend with Parquet storage

### 3. Evaluation & Monitoring Layer

#### Quality Evaluator

**LLM-as-a-Judge Pattern**:
- Uses GPT-3.5 to evaluate GPT-3.5 outputs
- Lower temperature (0.3) for consistent scoring
- Four dimensions: Relevance, Completeness, Citation Quality, Groundedness

**Why this works**:
- Correlation studies show LLM judges align well with human evaluation
- Scalable: Automated evaluation for every query
- Consistent: No inter-annotator disagreement
- Cost-effective: $0.001-0.002 per evaluation

**Metric Design**:
- **Relevance**: Answer-question alignment
- **Completeness**: Coverage of all aspects
- **Citation Quality**: Proper source usage
- **Groundedness**: Document-based vs. hallucinated

**Weighted scoring**:
```
Overall = 0.35*Relevance + 0.25*Completeness + 0.20*Citation + 0.20*Groundedness
```

Weights reflect priorities: Relevance most critical, groundedness ensures trust.

#### Performance Monitor

**Metrics tracked**:
- Latency: Total and per-agent
- Tokens: Input/output by agent
- Cost: Real-time estimation
- Success rate: Error tracking

**Why track per-agent**:
- Bottleneck identification
- Cost attribution
- Optimization targeting

**Structured Logging**:
- Format: JSON Lines (JSONL)
- Trace IDs: Request tracking across agents
- Separate logs: evaluations.jsonl, agent_executions.jsonl, errors.jsonl

**Why JSONL over database**:
- Append-only: Fast writes
- Easy parsing: Standard tools (jq, pandas)
- No infrastructure: File-based
- Migration path: Easy to load into databases later

### 4. Orchestration Layer

**Async Execution Model**:
```python
async def conduct_research():
    plan = await coordinator.create_plan()
    results = await asyncio.gather(*[researcher.search(task) for task in plan.tasks])
    insights = await analyzer.analyze(results)
    report = await writer.write(insights)
```

**Why async**:
- Parallel research tasks: 3x faster than sequential
- Non-blocking I/O: Efficient API waiting
- Cost savings: Faster execution = fewer wasted cycles

**Error Handling Strategy**:
- Try-except at each agent level
- Graceful fallbacks: Return partial results
- Error logging: Structured context capture
- User communication: Clear error messages

## Data Flow Example

### Document Ingestion Flow
```
User uploads "paper.pdf"
    ↓
DocumentProcessor extracts text by page
    ↓
Chunker creates 47 chunks with overlap
    ↓
EmbeddingGenerator creates 47 x 1536-dim vectors (batched)
    ↓
VectorStore.add_documents() stores in ChromaDB
    ↓
Success: 47 chunks indexed
```

### Research Query Flow
```
User asks: "What are sparse attention mechanisms?"
    ↓
Coordinator creates plan:
  - Task 1: Define sparse attention
  - Task 2: Find key papers
  - Task 3: Identify applications
    ↓
Researcher (parallel for each task):
  - Embeds task query
  - Searches ChromaDB
  - Retrieves top-5 chunks per task
  - Generates grounded answer
    ↓
Analyzer receives 3 search results:
  - Extracts 5 key insights
  - Categorizes by theme
  - Assigns confidence scores
    ↓
Writer synthesizes:
  - Executive summary (2-3 sentences)
  - Detailed findings (organized by theme)
  - Citations (source + page)
    ↓
Evaluator assesses:
  - Quality metrics (0.82 overall)
  - Performance metrics (23.4s, 3847 tokens, $0.008)
    ↓
Results displayed to user with visualizations
```

## Design Decisions & Trade-offs

### Why Multi-Agent vs. Single Prompt?

**Multi-Agent Pros**:
- Specialization: Each agent optimized for its task
- Modularity: Easy to improve/replace agents
- Observability: Track performance per stage
- Flexibility: Different models/parameters per agent

**Multi-Agent Cons**:
- Latency: Sequential pipeline adds time
- Cost: More API calls
- Complexity: More code to maintain

**Decision**: Multi-agent chosen for production systems where observability and modularity outweigh latency concerns.

### Why ChromaDB vs. Pinecone/Weaviate?

**Factors**:
- Deployment simplicity: Embedded > Managed service (for prototypes)
- Cost: Free > Paid (for development)
- Migration path: Easy to swap later

**Decision**: ChromaDB for development, with architecture designed for easy migration to managed services in production.

### Why GPT-3.5 vs. GPT-4?

**Analysis**:
- GPT-4: 10-20x more expensive, 30% higher quality
- GPT-3.5: Fast, cheap, sufficient for most tasks

**Strategy**:
- Use GPT-3.5 by default
- Reserve GPT-4 for:
  - Complex reasoning (Coordinator, if needed)
  - Final synthesis (Writer, for critical reports)
  - Configurable per use case

### Why Async vs. Synchronous?

**Measurement**:
- Sequential execution: 45s for 3-task research
- Async execution: 18s for same research (2.5x faster)

**Decision**: Async for all I/O-bound operations. Cost savings from faster execution justify complexity.

## Scalability Considerations

### Horizontal Scaling
- **Stateless design**: Each request independent
- **Load balancing**: Multiple Streamlit instances
- **Shared vector DB**: Centralized ChromaDB or cloud alternative

### Vertical Scaling
- **Batch processing**: Process multiple docs in parallel
- **Caching**: Cache embeddings for repeated queries
- **Model optimization**: Quantization for faster inference

### Cost Optimization
- **Tiered models**: GPT-3.5 default, GPT-4 on-demand
- **Prompt compression**: Minimize token usage
- **Caching**: Semantic caching for similar queries
- **Batch embeddings**: Reduce API calls

## Security Considerations

### Data Privacy
- Documents stored locally (ChromaDB)
- No data sent to third parties beyond Azure OpenAI
- API keys in environment variables (never committed)

### Access Control
- Streamlit: Add authentication layer (streamlit-authenticator)
- Azure: RBAC for OpenAI resource access
- Network: Private endpoints for production

### Audit Logging
- All operations logged with trace IDs
- User actions tracked
- Cost attribution per user/query

## Testing Strategy

### Unit Tests
- Mock Azure OpenAI responses
- Test document chunking logic
- Validate evaluation metrics calculation

### Integration Tests
- End-to-end research pipeline
- Document ingestion workflow
- Error handling scenarios

### Performance Tests
- Latency benchmarks
- Token usage validation
- Cost estimation accuracy

## Monitoring & Alerting

### Key Metrics
- **Quality**: Overall score < 0.6 → Alert
- **Latency**: P95 > 60s → Alert
- **Cost**: Daily spend > threshold → Alert
- **Errors**: Error rate > 5% → Alert

### Dashboards
- Real-time: Streamlit dashboard
- Historical: Grafana + Prometheus
- Business: Cost and usage reports

## Future Architecture Enhancements

### Short-term (1-3 months)
- Add caching layer (Redis)
- Implement rate limiting
- Add user authentication
- Enhanced error recovery

### Medium-term (3-6 months)
- Migrate to Azure Cognitive Search
- Add streaming responses
- Multi-modal support (images, tables)
- Fine-tuned embeddings

### Long-term (6-12 months)
- Distributed vector database
- Custom fine-tuned models
- Real-time collaboration
- Advanced analytics pipeline

---

This architecture balances production readiness with development velocity. Every design decision considers:
1. Cost efficiency
2. Observability
3. Maintainability
4. Scalability
5. User experience

The result is a system that demonstrates understanding of real-world ML system requirements beyond basic API integration.
