# Multi-Agent Research Assistant with RAG

An intelligent research system powered by Azure OpenAI that autonomously conducts comprehensive research through coordinated multi-agent collaboration, retrieval-augmented generation (RAG), and automated quality evaluation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-0078D4.svg)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)](https://streamlit.io/)

## 🎯 Overview

This project demonstrates production-ready AI application development by combining multiple cutting-edge techniques:

- **Multi-Agent Architecture**: Specialized agents (Coordinator, Researcher, Analyzer, Writer) collaborate to decompose complex queries, gather information, extract insights, and synthesize coherent reports
- **Retrieval-Augmented Generation (RAG)**: Grounds responses in actual documents using semantic search with vector embeddings, reducing hallucinations
- **MLOps Integration**: Automated quality evaluation, performance monitoring, structured logging, and trend analysis
- **Async Orchestration**: Parallel task execution for optimal performance and cost efficiency
- **Interactive UI**: Production-ready Streamlit interface for document management, research queries, and system monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                        │
│  (Document Upload │ Research Interface │ Dashboard)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              Research Orchestrator                          │
│  (Async coordination, monitoring, evaluation)               │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│  Coordinator   │  │  Researcher │  │    Analyzer     │
│   Agent        │  │   Agent     │  │     Agent       │
│                │  │   (RAG)     │  │                 │
│ • Plan tasks   │  │ • Retrieve  │  │ • Extract       │
│ • Decompose    │  │ • Search    │  │   insights      │
│   queries      │  │ • Cite      │  │ • Evaluate      │
└────────────────┘  └──────┬──────┘  └─────────────────┘
                           │
                   ┌───────▼────────┐
                   │  Vector Store  │
                   │   (ChromaDB)   │
                   │                │
                   │ • Embeddings   │
                   │ • Semantic     │
                   │   search       │
                   └────────────────┘
```

### Data Flow

1. **Document Ingestion**: PDFs/text files → chunking → embedding generation → vector storage
2. **Query Processing**: User question → task decomposition → parallel retrieval
3. **RAG Pipeline**: Query embedding → semantic search → context retrieval → grounded generation
4. **Synthesis**: Retrieved contexts → insight extraction → report generation
5. **Evaluation**: LLM-as-judge quality assessment → performance metrics → structured logging

## ✨ Key Features

### Multi-Agent System
- **Coordinator Agent**: Breaks down research questions into specific, actionable subtasks
- **RAG Researcher Agent**: Retrieves relevant document passages using semantic search and generates grounded answers
- **Analyzer Agent**: Extracts key insights, identifies patterns, and evaluates information quality
- **Writer Agent**: Synthesizes findings into coherent, well-cited research reports

### RAG Implementation
- Document processing pipeline (PDF, TXT, Markdown support)
- Intelligent chunking with overlap for context preservation
- Azure OpenAI embeddings (text-embedding-ada-002)
- ChromaDB vector database with cosine similarity search
- Source attribution with page-level citations

### MLOps & Monitoring
- **Quality Evaluation**: Automated assessment using LLM-as-judge pattern
  - Relevance scoring
  - Completeness analysis
  - Citation quality validation
  - Hallucination detection (groundedness)
- **Performance Monitoring**: 
  - Per-agent latency tracking
  - Token usage and cost estimation
  - Success/failure rate monitoring
- **Structured Logging**: JSON-formatted logs for all operations with trace IDs
- **Trend Analysis**: Quality degradation detection and performance comparison

### Production Engineering
- Async/await for concurrent agent execution
- Robust error handling with graceful fallbacks
- Token usage optimization and cost tracking
- Type-safe data structures (dataclasses)
- Comprehensive documentation and code comments

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Azure OpenAI Service access with deployed models:
  - `gpt-35-turbo` (or `gpt-4`)
  - `text-embedding-ada-002`

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd multi-agent-research-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set environment variables**
```bash
export AZURE_OPENAI_KEY="your-azure-openai-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.cognitiveservices.azure.com/"
```

Or create a `.env` file:
```
AZURE_OPENAI_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
```

### Running the Application

**Option 1: Streamlit UI (Recommended)**
```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser.

**Option 2: Command Line**
```bash
python research_agent.py
```

## 📖 Usage Guide

### Document Management
1. Upload research papers, articles, or documents (PDF, TXT, MD)
2. System automatically:
   - Extracts text
   - Chunks documents intelligently
   - Generates embeddings
   - Stores in vector database

### Conducting Research
1. Enter your research question
2. Watch agents execute in real-time:
   - Coordinator plans the research
   - Researcher retrieves relevant information
   - Analyzer extracts key insights
   - Writer synthesizes the report
3. View results with:
   - Executive summary
   - Detailed findings
   - Quality metrics (radar chart)
   - Performance breakdown
   - Source citations

### Monitoring Performance
- Dashboard shows quality trends over time
- Per-query cost and latency metrics
- Token usage breakdown by agent
- Quality score distribution

## 📊 Evaluation Metrics

### Quality Metrics (0-1 scale)
- **Relevance**: How well the answer addresses the question
- **Completeness**: Coverage of all question aspects
- **Citation Quality**: Proper source attribution
- **Groundedness**: Information grounded in documents vs. hallucinated

### Performance Metrics
- Total and per-agent latency
- Token consumption by agent
- Cost per query
- Throughput (characters/second)

## 🏭 Production Considerations

### Deployment Options

**Azure Container Instances**
```bash
# Build Docker image
docker build -t research-agent .

# Push to Azure Container Registry
az acr build --registry <your-acr> --image research-agent:v1 .

# Deploy to ACI
az container create --resource-group <rg> \
  --name research-agent \
  --image <your-acr>.azurecr.io/research-agent:v1 \
  --environment-variables AZURE_OPENAI_KEY=<key>
```

**Azure App Service**
- Package as web app
- Configure environment variables
- Enable continuous deployment from GitHub

**Databricks (for large-scale processing)**
- Deploy as Databricks job for batch processing
- Use Delta Lake for document storage
- MLflow for model/experiment tracking

### Scaling Considerations

1. **Vector Database**: 
   - Migrate ChromaDB to Azure Cognitive Search for production scale
   - Or use Pinecone/Weaviate for managed vector database

2. **Model Optimization**:
   - Cache frequent queries
   - Implement request batching
   - Use GPT-3.5-turbo for cost-sensitive operations
   - Reserve GPT-4 for complex reasoning tasks

3. **Monitoring & Alerting**:
   - Integrate with Azure Application Insights
   - Set up alerts for quality degradation
   - Monitor token usage and cost budgets

4. **Security**:
   - Use Azure Key Vault for credential management
   - Implement RBAC for document access
   - Enable audit logging
   - Network isolation with Private Endpoints

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Generate coverage report:
```bash
pytest --cov=. --cov-report=html
```

## 📈 Performance Benchmarks

Typical performance on sample workload:
- **Latency**: 15-30s for complex research queries
- **Cost**: $0.01-0.05 per query (GPT-3.5-turbo)
- **Quality**: 75-85% average quality score
- **Throughput**: ~50 chars/second generation

## 🗺️ Roadmap

- [ ] Multi-modal support (images, tables in PDFs)
- [ ] Integration with external APIs (arXiv, PubMed, Google Scholar)
- [ ] Fine-tuning specialized models for domain-specific research
- [ ] Collaborative research (multiple users, shared document pools)
- [ ] Advanced citation formats (BibTeX, APA, MLA)
- [ ] Conversation history and follow-up questions
- [ ] Export reports to PDF/DOCX

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Submit a pull request

## 📄 License

BSD 3-Clause License - see LICENSE file for details

## 🙏 Acknowledgments

- Azure OpenAI Service for LLM infrastructure
- ChromaDB for vector database
- Streamlit for rapid UI development
- OpenAI for foundational model research

## 📧 Contact

For questions or collaboration opportunities, reach out via:
- **Email**: alex.abrehforoush@gmail.com
- **LinkedIn**: [linkedin.com/in/alex-abrehforoush](https://www.linkedin.com/in/alex-abrehforoush/)
- **GitHub**: [github.com/alex-abrehforoush](https://github.com/alex-abrehforoush)

---

**Built with passion for production ML systems** 🚀

This project demonstrates real-world AI application development practices including multi-agent orchestration, RAG implementation, MLOps integration, and production-ready engineering. Perfect for roles in ML Engineering, AI Application Development, and Quantitative Research.
