# Multi-Agent Research System

An intelligent research assistant powered by Azure OpenAI that autonomously conducts research through coordinated multi-agent collaboration.

## Architecture

The system uses four specialized agents:
- **Coordinator Agent**: Breaks down research questions into specific subtasks
- **Researcher Agent**: Gathers information through parallel search execution
- **Analyzer Agent**: Extracts key insights and evaluates information quality
- **Writer Agent**: Synthesizes findings into coherent research reports

## Features

- Asynchronous agent orchestration for parallel task execution
- Token usage tracking and cost monitoring
- Structured data flow using typed dataclasses
- Comprehensive error handling and logging
- Production-ready code architecture

## Setup

1. Install dependencies:
```bash
   pip install -r requirements.txt
```

2. Set environment variables:
```bash
   export AZURE_OPENAI_KEY="your-key"
   export AZURE_OPENAI_ENDPOINT="https://your-resource.cognitiveservices.azure.com/"
```

3. Run the system:
```bash
   python research_agent.py
```

## Technical Stack

- Python 3.8+
- Azure OpenAI Service (GPT-3.5-turbo)
- Async/await for concurrent execution
- Dataclasses for type-safe data structures

## Cost Optimization

The system implements several cost optimization strategies:
- Default use of GPT-3.5-turbo for cost efficiency
- Token usage tracking across all agents
- Real-time cost estimation
- Configurable token limits per agent

## Future Enhancements

- Integration with real search APIs (web search, arxiv, etc.)
- Vector database for document retrieval (RAG)
- Deployment to Azure with monitoring dashboards
- Support for multiple LLM providers