import asyncio
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from openai import AsyncAzureOpenAI
from datetime import datetime
import chromadb
import PyPDF2
from pathlib import Path
import hashlib
import uuid
from evaluation_monitoring import (
    PerformanceMonitor,
    EvaluationOrchestrator,
    AgentExecutionLog,
    evaluate_research_pipeline
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    chunk_id: str
    content: str
    source: str
    page_number: Optional[int]
    chunk_index: int
    metadata: Dict[str, Any]


@dataclass
class RetrievedContext:
    """Context retrieved from vector database."""
    content: str
    source: str
    page_number: Optional[int]
    relevance_score: float


@dataclass
class ResearchTask:
    """Represents a specific subtask identified by the coordinator."""
    task_id: int
    description: str
    search_query: str
    priority: str


@dataclass
class ResearchPlan:
    """The coordinator's output: a structured plan for research."""
    original_question: str
    tasks: List[ResearchTask]
    estimated_complexity: str


@dataclass
class SearchResult:
    """Enhanced search result with RAG context."""
    task_id: int
    query: str
    content: str
    retrieved_contexts: List[RetrievedContext]
    source: str
    relevance_score: float


@dataclass
class AnalyzedInsight:
    """The analyzer's output: key insights extracted from search results."""
    insight: str
    supporting_sources: List[str]
    confidence: str
    theme: str


@dataclass
class ResearchReport:
    """Final output: a complete research report with citations."""
    question: str
    executive_summary: str
    detailed_findings: str
    key_insights: List[str]
    sources: List[str]
    citations: List[Dict[str, Any]]  # Detailed citation information
    timestamp: str


# ============================================================================
# DOCUMENT PROCESSING AND VECTOR STORE
# ============================================================================

class DocumentProcessor:
    """
    Handles document ingestion, chunking, and text extraction.
    Supports PDFs and text files.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with page numbers."""
        pages = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    text = page.extract_text()
                    if text.strip():
                        pages.append({
                            'page_number': page_num,
                            'text': text,
                            'source': os.path.basename(pdf_path)
                        })
        except Exception as e:
            print(f"[ERROR] Failed to extract PDF {pdf_path}: {e}")
            raise
        return pages
    
    def extract_text_from_txt(self, txt_path: str) -> List[Dict[str, Any]]:
        """Extract text from plain text file."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
                return [{
                    'page_number': None,
                    'text': text,
                    'source': os.path.basename(txt_path)
                }]
        except Exception as e:
            print(f"[ERROR] Failed to read text file {txt_path}: {e}")
            raise
    
    def chunk_text(self, text: str, source: str, page_number: Optional[int]) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        Uses simple character-based chunking with overlap to preserve context.
        """
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start, end - 200), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # Generate unique ID for this chunk
                chunk_id = hashlib.md5(
                    f"{source}_{page_number}_{chunk_index}_{chunk_text[:50]}".encode()
                ).hexdigest()
                
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    source=source,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    metadata={
                        'char_count': len(chunk_text),
                        'word_count': len(chunk_text.split())
                    }
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
        return chunks
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a document (PDF or text) into chunks.
        """
        file_ext = Path(file_path).suffix.lower()
        
        print(f"[DocumentProcessor] Processing {file_path}...")
        
        if file_ext == '.pdf':
            pages = self.extract_text_from_pdf(file_path)
        elif file_ext in ['.txt', '.md']:
            pages = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Chunk each page
        all_chunks = []
        for page_data in pages:
            chunks = self.chunk_text(
                page_data['text'],
                page_data['source'],
                page_data['page_number']
            )
            all_chunks.extend(chunks)
        
        print(f"[DocumentProcessor] Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks


class VectorStore:
    """
    Manages vector storage and retrieval using ChromaDB.
    Handles embeddings and semantic search.
    """
    
    def __init__(self, collection_name: str = "research_documents"):
        # Initialize ChromaDB with persistent storage (new API)
        self.client = chromadb.PersistentClient(path=".chromadb")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
    async def generate_embeddings(
        self, 
        texts: List[str], 
        azure_client: AsyncAzureOpenAI
    ) -> List[List[float]]:
        """
        Generate embeddings using Azure OpenAI.
        Note: Azure OpenAI embedding endpoint is synchronous, but we keep async
        signature for consistency with the rest of the codebase.
        """
        try:
            # Azure OpenAI embeddings are created synchronously
            # We use the async client but call it in a way that works
            embeddings = []
            
            # Batch process embeddings (max 16 at a time for Azure)
            batch_size = 16
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Note: embedding creation doesn't have async version in current SDK
                # This is a known limitation - embeddings are fast enough that it's acceptable
                response = await azure_client.embeddings.create(
                    model="text-embedding-ada-002",  # You need this deployment in Azure
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            print(f"[ERROR] Failed to generate embeddings: {e}")
            raise
    
    async def add_documents(
        self, 
        chunks: List[DocumentChunk],
        azure_client: AsyncAzureOpenAI
    ):
        """
        Add document chunks to the vector store with embeddings.
        """
        if not chunks:
            return
        
        print(f"[VectorStore] Generating embeddings for {len(chunks)} chunks...")
        
        # Extract texts and generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.generate_embeddings(texts, azure_client)
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {
                'source': chunk.source,
                'page_number': str(chunk.page_number) if chunk.page_number else 'N/A',
                'chunk_index': chunk.chunk_index,
                **chunk.metadata
            }
            for chunk in chunks
        ]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"[VectorStore] Added {len(chunks)} chunks to vector store")
    
    async def search(
        self,
        query: str,
        azure_client: AsyncAzureOpenAI,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[RetrievedContext]:
        """
        Search for relevant document chunks using semantic similarity.
        """
        # Generate query embedding
        query_embedding = (await self.generate_embeddings([query], azure_client))[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Convert to RetrievedContext objects
        contexts = []
        if results['documents'] and results['documents'][0]:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                # Convert distance to similarity score (cosine distance -> similarity)
                similarity = 1 - distance
                
                if similarity >= score_threshold:
                    contexts.append(RetrievedContext(
                        content=doc,
                        source=metadata['source'],
                        page_number=int(metadata['page_number']) if metadata['page_number'] != 'N/A' else None,
                        relevance_score=similarity
                    ))
        
        return contexts
    
    def clear(self):
        """Clear all documents from the vector store."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("[VectorStore] Cleared all documents")


# ============================================================================
# TOKEN TRACKING AND CLIENT
# ============================================================================

class TokenTracker:
    """Tracks API usage and costs across all agents."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.calls_by_agent = {}
        self.tokens_by_model = {"gpt-35-turbo": 0, "gpt-4": 0, "text-embedding-ada-002": 0}
        
    def add_usage(self, agent_name: str, model: str, input_tokens: int, output_tokens: int = 0):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        if agent_name not in self.calls_by_agent:
            self.calls_by_agent[agent_name] = 0
        self.calls_by_agent[agent_name] += 1
        
        self.tokens_by_model[model] = self.tokens_by_model.get(model, 0) + input_tokens + output_tokens
        
    def get_estimated_cost(self) -> float:
        """Calculate cost based on different model pricing."""
        gpt35_tokens = self.tokens_by_model.get("gpt-35-turbo", 0)
        gpt4_tokens = self.tokens_by_model.get("gpt-4", 0)
        embedding_tokens = self.tokens_by_model.get("text-embedding-ada-002", 0)
        
        # Pricing estimates
        gpt35_cost = (gpt35_tokens / 1000) * 0.002
        gpt4_cost = (gpt4_tokens / 1000) * 0.045
        embedding_cost = (embedding_tokens / 1000) * 0.0001  # Very cheap
        
        return gpt35_cost + gpt4_cost + embedding_cost
    
    def report(self):
        print(f"\n{'='*60}")
        print(f"TOKEN USAGE REPORT")
        print(f"{'='*60}")
        print(f"Total input tokens:  {self.total_input_tokens:,}")
        print(f"Total output tokens: {self.total_output_tokens:,}")
        print(f"\nCalls by agent:")
        for agent, count in self.calls_by_agent.items():
            print(f"  {agent}: {count}")
        print(f"\nTokens by model:")
        for model, tokens in self.tokens_by_model.items():
            if tokens > 0:
                print(f"  {model}: {tokens:,}")
        print(f"\nEstimated cost: ${self.get_estimated_cost():.4f}")
        print(f"{'='*60}\n")


class AzureOpenAIClient:
    """Handles all Azure OpenAI API interactions with token tracking."""
    
    def __init__(self, api_key: str, endpoint: str, tracker: TokenTracker):
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=endpoint
        )
        self.tracker = tracker
        
    async def create_completion(
        self,
        agent_name: str,
        messages: List[Dict[str, str]],
        deployment_name: str = "gpt-35-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> str:
        """Make an async API call with automatic token tracking."""
        try:
            response = await self.client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            usage = response.usage
            self.tracker.add_usage(
                agent_name, 
                deployment_name,
                usage.prompt_tokens, 
                usage.completion_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[ERROR] API call failed in {agent_name}: {e}")
            raise


# ============================================================================
# ENHANCED AGENTS WITH RAG
# ============================================================================

class CoordinatorAgent:
    """Coordinator breaks down research questions into specific subtasks."""
    
    def __init__(self, client: AzureOpenAIClient):
        self.client = client
        self.name = "Coordinator"
        
    async def create_plan(self, question: str) -> ResearchPlan:
        """Creates a structured research plan."""
        
        system_prompt = """You are a research coordinator. Break down research questions into specific, actionable subtasks.

For each subtask:
1. Create a clear, specific description
2. Formulate a targeted search query
3. Assign a priority (high/medium/low)

Aim for 3-5 subtasks that comprehensively address the question.
Respond in JSON format:
{
    "tasks": [
        {
            "task_id": 1,
            "description": "...",
            "search_query": "...",
            "priority": "high"
        }
    ],
    "estimated_complexity": "simple/moderate/complex"
}"""

        user_prompt = f"""Research question: {question}

Break this down into specific research subtasks."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"[{self.name}] Creating research plan...")
        
        response = await self.client.create_completion(
            agent_name=self.name,
            messages=messages,
            deployment_name="gpt-35-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        
        try:
            plan_data = json.loads(response)
            tasks = [ResearchTask(**task) for task in plan_data["tasks"]]
            
            plan = ResearchPlan(
                original_question=question,
                tasks=tasks,
                estimated_complexity=plan_data.get("estimated_complexity", "moderate")
            )
            
            print(f"[{self.name}] Created plan with {len(tasks)} tasks")
            return plan
            
        except json.JSONDecodeError as e:
            print(f"[{self.name}] Failed to parse plan: {e}")
            return ResearchPlan(
                original_question=question,
                tasks=[ResearchTask(1, question, question, "high")],
                estimated_complexity="simple"
            )


class RAGResearcherAgent:
    """
    Enhanced researcher that uses RAG to ground answers in documents.
    This is the key improvement over the basic version.
    """
    
    def __init__(self, client: AzureOpenAIClient, vector_store: VectorStore):
        self.client = client
        self.vector_store = vector_store
        self.name = "RAG_Researcher"
        
    async def execute_search(self, task: ResearchTask) -> SearchResult:
        """
        Execute search using RAG: retrieve relevant contexts, then generate answer.
        """
        print(f"[{self.name}] Researching: {task.description}")
        
        # Step 1: Retrieve relevant contexts from vector store
        contexts = await self.vector_store.search(
            query=task.search_query,
            azure_client=self.client.client,
            top_k=5,
            score_threshold=0.3
        )
        
        if not contexts:
            print(f"[{self.name}] No relevant documents found, generating from model knowledge")
            # Fallback to model knowledge if no documents found
            context_str = "No specific documents available. Use your general knowledge."
        else:
            print(f"[{self.name}] Retrieved {len(contexts)} relevant contexts")
            # Format contexts for the prompt
            context_str = "\n\n".join([
                f"[Source: {ctx.source}, Page: {ctx.page_number or 'N/A'}, Relevance: {ctx.relevance_score:.2f}]\n{ctx.content}"
                for ctx in contexts
            ])
        
        # Step 2: Generate answer grounded in retrieved contexts
        system_prompt = """You are a research agent that provides detailed, accurate information.

IMPORTANT: Base your answer primarily on the provided document contexts. If the contexts contain relevant information, cite the specific sources and page numbers. If the contexts don't contain sufficient information, clearly state this and use your general knowledge cautiously."""

        user_prompt = f"""Search query: {task.search_query}
Task context: {task.description}

Retrieved document contexts:
{context_str}

Provide a detailed, well-organized answer based on the contexts above. Include specific citations (source and page number) when referencing information from the documents."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.client.create_completion(
            agent_name=self.name,
            messages=messages,
            deployment_name="gpt-35-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Package as search result with retrieved contexts
        result = SearchResult(
            task_id=task.task_id,
            query=task.search_query,
            content=response,
            retrieved_contexts=contexts,
            source=f"rag_search_{task.task_id}",
            relevance_score=contexts[0].relevance_score if contexts else 0.0
        )
        
        return result
    
    async def gather_all_information(self, plan: ResearchPlan) -> List[SearchResult]:
        """Execute all research tasks concurrently."""
        print(f"[{self.name}] Starting research on {len(plan.tasks)} tasks...")
        
        # Run all searches in parallel
        search_coroutines = [self.execute_search(task) for task in plan.tasks]
        results = await asyncio.gather(*search_coroutines)
        
        print(f"[{self.name}] Completed all research tasks")
        return list(results)


class AnalyzerAgent:
    """Analyzes search results and extracts key insights."""
    
    def __init__(self, client: AzureOpenAIClient):
        self.client = client
        self.name = "Analyzer"
        
    async def analyze_results(
        self, 
        question: str, 
        results: List[SearchResult]
    ) -> List[AnalyzedInsight]:
        """Analyzes search results to extract key insights."""
        
        # Combine all search results with citation information
        context = "\n\n".join([
            f"Result {i+1} (Query: {r.query}):\n{r.content}\n"
            f"Sources: {', '.join(set(ctx.source for ctx in r.retrieved_contexts))}"
            for i, r in enumerate(results)
        ])
        
        system_prompt = """You are an analytical research agent. Your job is to:
1. Identify key insights from the provided information
2. Group related information into themes
3. Evaluate the confidence level of each insight based on source quality
4. Note which sources support each insight

Respond in JSON format:
{
    "insights": [
        {
            "insight": "Clear statement of the insight",
            "supporting_sources": ["source_1", "source_2"],
            "confidence": "high/medium/low",
            "theme": "category or theme name"
        }
    ]
}

Aim for 4-6 key insights that directly address the research question."""

        user_prompt = f"""Research question: {question}

Information gathered:
{context}

Analyze this information and extract key insights."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"[{self.name}] Analyzing results...")
        
        response = await self.client.create_completion(
            agent_name=self.name,
            messages=messages,
            deployment_name="gpt-35-turbo",
            temperature=0.5,
            max_tokens=1200
        )
        
        try:
            # Try to extract JSON from response (sometimes LLM adds explanatory text)
            response_clean = response.strip()
            
            # Look for JSON content between braces if there's extra text
            if response_clean.startswith('{'):
                json_end = response_clean.rfind('}') + 1
                response_clean = response_clean[:json_end]
            elif '{' in response_clean:
                json_start = response_clean.index('{')
                json_end = response_clean.rfind('}') + 1
                response_clean = response_clean[json_start:json_end]
            
            analysis_data = json.loads(response_clean)
            insights = [AnalyzedInsight(**insight) for insight in analysis_data["insights"]]
            print(f"[{self.name}] Extracted {len(insights)} insights")
            return insights
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[{self.name}] Failed to parse analysis: {e}")
            print(f"[{self.name}] Raw response: {response[:200]}...")
            
            # Fallback: create basic insights from the response text
            return [AnalyzedInsight(
                insight=f"Analysis response was not properly structured. Raw content available.",
                supporting_sources=["multiple sources"],
                confidence="low",
                theme="general"
            )]


class WriterAgent:
    """Synthesizes insights into a coherent research report with proper citations."""
    
    def __init__(self, client: AzureOpenAIClient):
        self.client = client
        self.name = "Writer"
        
    async def write_report(
        self,
        question: str,
        plan: ResearchPlan,
        insights: List[AnalyzedInsight],
        results: List[SearchResult]
    ) -> ResearchReport:
        """Creates a final research report with citations."""
        
        insights_text = "\n".join([
            f"- {insight.insight} (Theme: {insight.theme}, Confidence: {insight.confidence})"
            for insight in insights
        ])
        
        # Collect all citations
        all_citations = []
        for result in results:
            for ctx in result.retrieved_contexts:
                all_citations.append({
                    'source': ctx.source,
                    'page': ctx.page_number,
                    'relevance': ctx.relevance_score,
                    'query': result.query
                })
        
        system_prompt = """You are a technical writer creating research reports.
Your report should:
1. Start with a concise executive summary (2-3 sentences)
2. Present detailed findings organized logically
3. Include proper citations to source documents
4. Highlight key insights clearly
5. Be clear, well-organized, and professional

Use citations in format: (Source: filename, Page: N) when referencing specific information."""

        user_prompt = f"""Research question: {question}

Key insights identified:
{insights_text}

Write a comprehensive research report with proper citations."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"[{self.name}] Writing final report...")
        
        response = await self.client.create_completion(
            agent_name=self.name,
            messages=messages,
            deployment_name="gpt-35-turbo",
            temperature=0.7,
            max_tokens=1500
        )
        
        key_insights_list = [insight.insight for insight in insights[:5]]
        sources_list = list(set([citation['source'] for citation in all_citations]))
        
        report = ResearchReport(
            question=question,
            executive_summary=response.split("\n\n")[0],
            detailed_findings=response,
            key_insights=key_insights_list,
            sources=sources_list,
            citations=all_citations,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"[{self.name}] Report completed with {len(all_citations)} citations")
        return report


# ============================================================================
# MONITORED ORCHESTRATOR (Enhanced version)
# ============================================================================

class MonitoredRAGResearchOrchestrator:
    """
    Enhanced orchestrator with integrated monitoring and evaluation.
    """
    
    def __init__(self, client, vector_store):
        self.client = client
        self.vector_store = vector_store
        self.coordinator = CoordinatorAgent(client)
        self.researcher = RAGResearcherAgent(client, vector_store)
        self.analyzer = AnalyzerAgent(client)
        self.writer = WriterAgent(client)
        
        # Add evaluation components
        self.evaluator = EvaluationOrchestrator(client.client)
        
    async def ingest_documents(self, document_paths: List[str]):
        """Document ingestion with monitoring."""
        print(f"\n{'='*60}")
        print(f"DOCUMENT INGESTION PIPELINE")
        print(f"{'='*60}\n")
        
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        
        all_chunks = []
        for doc_path in document_paths:
            try:
                chunks = processor.process_document(doc_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"[ERROR] Failed to process {doc_path}: {e}")
                continue
        
        if all_chunks:
            await self.vector_store.add_documents(all_chunks, self.client.client)
            print(f"\n[SUCCESS] Ingested {len(all_chunks)} chunks from {len(document_paths)} documents")
        else:
            print("[WARNING] No chunks were created from documents")
        
        print(f"{'='*60}\n")
        
    async def conduct_research(self, question: str, enable_evaluation: bool = True):
        """
        Execute research with integrated monitoring and evaluation.
        """
        trace_id = str(uuid.uuid4())
        monitor = PerformanceMonitor()
        monitor.start_operation()
        
        print(f"\n{'='*60}")
        print(f"STARTING MONITORED RESEARCH")
        print(f"Trace ID: {trace_id}")
        print(f"Question: {question}")
        print(f"{'='*60}\n")
        
        all_retrieved_contexts = []
        
        try:
            # Step 1: Coordinator
            monitor.start_agent("Coordinator")
            tokens_before = self.client.tracker.total_input_tokens + self.client.tracker.total_output_tokens
            
            plan = await self.coordinator.create_plan(question)
            
            tokens_after = self.client.tracker.total_input_tokens + self.client.tracker.total_output_tokens
            monitor.end_agent("Coordinator", tokens_after - tokens_before)
            
            self.evaluator.logger.log_agent_execution(AgentExecutionLog(
                trace_id=trace_id,
                agent_name="Coordinator",
                timestamp=datetime.now().isoformat(),
                input_summary=f"Question: {question[:100]}...",
                output_summary=f"Created {len(plan.tasks)} tasks",
                latency_seconds=monitor.agent_latencies.get("Coordinator", 0),
                tokens_used=tokens_after - tokens_before,
                success=True
            ))
            
            # Step 2: Researcher
            monitor.start_agent("Researcher")
            tokens_before = self.client.tracker.total_input_tokens + self.client.tracker.total_output_tokens
            
            results = await self.researcher.gather_all_information(plan)
            
            # Collect all retrieved contexts for evaluation
            for result in results:
                all_retrieved_contexts.extend(result.retrieved_contexts)
            
            tokens_after = self.client.tracker.total_input_tokens + self.client.tracker.total_output_tokens
            monitor.end_agent("Researcher", tokens_after - tokens_before)
            
            self.evaluator.logger.log_agent_execution(AgentExecutionLog(
                trace_id=trace_id,
                agent_name="Researcher",
                timestamp=datetime.now().isoformat(),
                input_summary=f"Researching {len(plan.tasks)} tasks",
                output_summary=f"Found {len(all_retrieved_contexts)} relevant contexts",
                latency_seconds=monitor.agent_latencies.get("Researcher", 0),
                tokens_used=tokens_after - tokens_before,
                success=True
            ))
            
            # Step 3: Analyzer
            monitor.start_agent("Analyzer")
            tokens_before = self.client.tracker.total_input_tokens + self.client.tracker.total_output_tokens
            
            insights = await self.analyzer.analyze_results(question, results)
            
            tokens_after = self.client.tracker.total_input_tokens + self.client.tracker.total_output_tokens
            monitor.end_agent("Analyzer", tokens_after - tokens_before)
            
            self.evaluator.logger.log_agent_execution(AgentExecutionLog(
                trace_id=trace_id,
                agent_name="Analyzer",
                timestamp=datetime.now().isoformat(),
                input_summary=f"Analyzing {len(results)} search results",
                output_summary=f"Extracted {len(insights)} insights",
                latency_seconds=monitor.agent_latencies.get("Analyzer", 0),
                tokens_used=tokens_after - tokens_before,
                success=True
            ))
            
            # Step 4: Writer
            monitor.start_agent("Writer")
            tokens_before = self.client.tracker.total_input_tokens + self.client.tracker.total_output_tokens
            
            report = await self.writer.write_report(question, plan, insights, results)
            
            tokens_after = self.client.tracker.total_input_tokens + self.client.tracker.total_output_tokens
            monitor.end_agent("Writer", tokens_after - tokens_before)
            
            self.evaluator.logger.log_agent_execution(AgentExecutionLog(
                trace_id=trace_id,
                agent_name="Writer",
                timestamp=datetime.now().isoformat(),
                input_summary=f"Synthesizing {len(insights)} insights",
                output_summary=f"Generated report ({len(report.detailed_findings)} chars)",
                latency_seconds=monitor.agent_latencies.get("Writer", 0),
                tokens_used=tokens_after - tokens_before,
                success=True
            ))
            
            print(f"\n{'='*60}")
            print(f"RESEARCH COMPLETED")
            print(f"{'='*60}\n")
            
            # Step 5: Evaluation (if enabled)
            if enable_evaluation:
                eval_result = await evaluate_research_pipeline(
                    self.client.client,
                    question,
                    report,
                    all_retrieved_contexts,
                    monitor
                )
            
            return report
            
        except Exception as e:
            monitor.record_error(str(e))
            self.evaluator.logger.log_error(
                trace_id=trace_id,
                error_type=type(e).__name__,
                error_message=str(e),
                context={'question': question}
            )
            raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main function with evaluation enabled."""
    
    # Setup
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not api_key or not endpoint:
        print("Error: Please set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT")
        return
    
    # Initialize
    tracker = TokenTracker()
    client = AzureOpenAIClient(api_key, endpoint, tracker)
    vector_store = VectorStore(collection_name="research_documents")
    
    # Create MONITORED orchestrator (this is the key change)
    orchestrator = MonitoredRAGResearchOrchestrator(client, vector_store)
    
    # Ingest documents
    document_folder = "documents"
    if os.path.exists(document_folder):
        document_paths = [
            os.path.join(document_folder, f) 
            for f in os.listdir(document_folder) 
            if f.endswith(('.pdf', '.txt', '.md'))
        ]
        
        if document_paths:
            await orchestrator.ingest_documents(document_paths)
        else:
            # Create sample document
            os.makedirs(document_folder, exist_ok=True)
            sample_doc = os.path.join(document_folder, "sample.txt")
            with open(sample_doc, 'w') as f:
                f.write("""
Transformer Architecture Improvements

Recent developments in transformer architectures have focused on several key areas:

1. Sparse Attention Mechanisms: Longformer and BigBird introduced sparse attention patterns 
that reduce the quadratic complexity of standard transformers to linear complexity.

2. Mixture of Experts (MoE): Switch Transformers demonstrated that using sparse MoE layers 
can dramatically increase model capacity while keeping computational costs manageable.

3. Efficient Attention: Flash Attention optimized memory access patterns of attention 
computation, achieving 2-4x speedups.

4. Rotary Position Embeddings (RoPE): Encodes position information more effectively than 
absolute or learned position embeddings.
                """)
            await orchestrator.ingest_documents([sample_doc])
    
    # Run multiple queries to build evaluation history
    questions = [
        "What are the recent improvements in AI for handling cyber security challenges?",
        "How do machines can interact with humans?",
        "What are the latest unresolved issues in AI?"
    ]
    
    for question in questions:
        print(f"\n\n{'#'*60}")
        print(f"QUERY: {question}")
        print(f"{'#'*60}")
        
        report = await orchestrator.conduct_research(question, enable_evaluation=True)
        
        # Display report
        print("\n" + "="*60)
        print("RESEARCH REPORT")
        print("="*60 + "\n")
        print(f"Question: {report.question}\n")
        print(f"Executive Summary:\n{report.executive_summary}\n")
        print(f"\nKey Insights:")
        for i, insight in enumerate(report.key_insights, 1):
            print(f"{i}. {insight}")
        
        print(f"\n\nSources: {', '.join(report.sources)}")
    
    # Show performance trends
    print("\n\n" + "="*60)
    print("ANALYZING PERFORMANCE TRENDS")
    print("="*60)
    orchestrator.evaluator.compare_recent_performance(n=len(questions))
    
    # Show cost tracking
    tracker.report()


if __name__ == "__main__":
    asyncio.run(main())