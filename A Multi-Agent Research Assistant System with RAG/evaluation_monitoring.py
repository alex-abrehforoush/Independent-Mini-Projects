import json
import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import asyncio
from openai import AsyncAzureOpenAI


# ============================================================================
# EVALUATION DATA STRUCTURES
# ============================================================================

@dataclass
class QualityMetrics:
    """Quality assessment metrics for research outputs."""
    relevance_score: float  # 0-1: How well does answer address the question?
    completeness_score: float  # 0-1: Are all aspects covered?
    citation_quality_score: float  # 0-1: Are sources properly used?
    groundedness_score: float  # 0-1: Is info grounded in documents vs hallucinated?
    overall_score: float  # Weighted average
    reasoning: str  # Explanation of scores


@dataclass
class PerformanceMetrics:
    """Performance metrics for system execution."""
    total_latency_seconds: float
    agent_latencies: Dict[str, float]
    total_tokens: int
    tokens_by_agent: Dict[str, int]
    cost_usd: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class EvaluationResult:
    """Complete evaluation of a research query."""
    trace_id: str
    timestamp: str
    question: str
    quality_metrics: QualityMetrics
    performance_metrics: PerformanceMetrics
    report_length: int
    num_citations: int
    num_sources: int
    

@dataclass
class AgentExecutionLog:
    """Detailed log of agent execution."""
    trace_id: str
    agent_name: str
    timestamp: str
    input_summary: str
    output_summary: str
    latency_seconds: float
    tokens_used: int
    success: bool
    error: Optional[str] = None


# ============================================================================
# QUALITY EVALUATOR
# ============================================================================

class QualityEvaluator:
    """
    Evaluates the quality of research outputs using LLM-as-a-judge pattern.
    This is critical for production ML systems - you need automated quality checks.
    """
    
    def __init__(self, azure_client: AsyncAzureOpenAI):
        self.client = azure_client
        
    async def evaluate_quality(
        self,
        question: str,
        report: Any,  # ResearchReport object
        retrieved_contexts: List[Any]  # List of RetrievedContext
    ) -> QualityMetrics:
        """
        Comprehensive quality evaluation using LLM as a judge.
        This is a standard pattern in production LLM systems.
        """
        
        # Prepare evaluation prompt
        contexts_text = "\n".join([
            f"- {ctx.content[:200]}... (Source: {ctx.source})"
            for ctx in retrieved_contexts[:5]
        ])
        
        eval_prompt = f"""You are an expert evaluator assessing the quality of a research report.

QUESTION: {question}

REPORT:
{report.detailed_findings}

AVAILABLE SOURCE DOCUMENTS:
{contexts_text}

CITATIONS IN REPORT:
{json.dumps(report.citations[:10], indent=2)}

Evaluate the report on these dimensions (score 0-10 for each):

1. RELEVANCE: Does the report directly address the question asked?
2. COMPLETENESS: Are all aspects of the question covered adequately?
3. CITATION_QUALITY: Are sources properly cited and attributions accurate?
4. GROUNDEDNESS: Is the information grounded in the provided sources vs. potentially hallucinated?

Respond in JSON format:
{{
    "relevance_score": <0-10>,
    "completeness_score": <0-10>,
    "citation_quality_score": <0-10>,
    "groundedness_score": <0-10>,
    "reasoning": "Brief explanation of scores focusing on specific strengths and weaknesses"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-35-turbo",
                messages=[
                    {"role": "system", "content": "You are a rigorous research quality evaluator. Be critical and precise."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent evaluation
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if '{' in result_text:
                json_start = result_text.index('{')
                json_end = result_text.rfind('}') + 1
                result_text = result_text[json_start:json_end]
            
            scores = json.loads(result_text)
            
            # Normalize scores to 0-1 range
            relevance = scores['relevance_score'] / 10.0
            completeness = scores['completeness_score'] / 10.0
            citation_quality = scores['citation_quality_score'] / 10.0
            groundedness = scores['groundedness_score'] / 10.0
            
            # Calculate weighted overall score
            # Weight groundedness and relevance more heavily for RAG systems
            overall = (
                relevance * 0.35 +
                completeness * 0.25 +
                citation_quality * 0.20 +
                groundedness * 0.20
            )
            
            return QualityMetrics(
                relevance_score=relevance,
                completeness_score=completeness,
                citation_quality_score=citation_quality,
                groundedness_score=groundedness,
                overall_score=overall,
                reasoning=scores['reasoning']
            )
            
        except Exception as e:
            print(f"[QualityEvaluator] Evaluation failed: {e}")
            # Return neutral scores on failure
            return QualityMetrics(
                relevance_score=0.5,
                completeness_score=0.5,
                citation_quality_score=0.5,
                groundedness_score=0.5,
                overall_score=0.5,
                reasoning=f"Evaluation failed: {str(e)}"
            )


# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """
    Monitors system performance including latency, tokens, and costs.
    Essential for production ML systems.
    """
    
    def __init__(self):
        self.start_time = None
        self.agent_start_times = {}
        self.agent_latencies = {}
        self.agent_tokens = {}
        self.total_tokens = 0
        self.total_cost = 0.0
        self.errors = []
        
    def start_operation(self):
        """Start timing the overall operation."""
        self.start_time = time.time()
        
    def start_agent(self, agent_name: str):
        """Start timing a specific agent."""
        self.agent_start_times[agent_name] = time.time()
        
    def end_agent(self, agent_name: str, tokens_used: int):
        """Record agent completion."""
        if agent_name in self.agent_start_times:
            latency = time.time() - self.agent_start_times[agent_name]
            self.agent_latencies[agent_name] = latency
            self.agent_tokens[agent_name] = tokens_used
            self.total_tokens += tokens_used
            
    def record_error(self, error_message: str):
        """Record an error."""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'message': error_message
        })
        
    def get_metrics(self) -> PerformanceMetrics:
        """Get complete performance metrics."""
        total_latency = time.time() - self.start_time if self.start_time else 0
        
        # Estimate cost (simplified pricing)
        cost = (self.total_tokens / 1000) * 0.002  # $0.002 per 1K tokens for GPT-3.5
        
        return PerformanceMetrics(
            total_latency_seconds=total_latency,
            agent_latencies=self.agent_latencies.copy(),
            total_tokens=self.total_tokens,
            tokens_by_agent=self.agent_tokens.copy(),
            cost_usd=cost,
            success=len(self.errors) == 0,
            error_message='; '.join([e['message'] for e in self.errors]) if self.errors else None
        )


# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class StructuredLogger:
    """
    Structured logging system for production ML monitoring.
    Logs are in JSON format for easy parsing and analysis.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create separate log files for different types
        self.evaluation_log = self.log_dir / "evaluations.jsonl"
        self.agent_log = self.log_dir / "agent_executions.jsonl"
        self.error_log = self.log_dir / "errors.jsonl"
        
    def log_evaluation(self, eval_result: EvaluationResult):
        """Log evaluation result."""
        self._append_jsonl(self.evaluation_log, asdict(eval_result))
        
    def log_agent_execution(self, log_entry: AgentExecutionLog):
        """Log agent execution details."""
        self._append_jsonl(self.agent_log, asdict(log_entry))
        
    def log_error(self, trace_id: str, error_type: str, error_message: str, context: Dict[str, Any]):
        """Log error with context."""
        error_entry = {
            'trace_id': trace_id,
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'context': context
        }
        self._append_jsonl(self.error_log, error_entry)
        
    def _append_jsonl(self, filepath: Path, data: Dict[str, Any]):
        """Append JSON line to log file."""
        with open(filepath, 'a') as f:
            f.write(json.dumps(data) + '\n')
            
    def get_recent_evaluations(self, n: int = 10) -> List[EvaluationResult]:
        """Retrieve recent evaluation results."""
        if not self.evaluation_log.exists():
            return []
            
        evaluations = []
        with open(self.evaluation_log, 'r') as f:
            lines = f.readlines()
            for line in lines[-n:]:
                data = json.loads(line)
                # Reconstruct nested objects
                data['quality_metrics'] = QualityMetrics(**data['quality_metrics'])
                data['performance_metrics'] = PerformanceMetrics(**data['performance_metrics'])
                evaluations.append(EvaluationResult(**data))
                
        return evaluations


# ============================================================================
# EVALUATION ORCHESTRATOR
# ============================================================================

class EvaluationOrchestrator:
    """
    Main orchestrator that integrates evaluation and monitoring into the research system.
    """
    
    def __init__(self, azure_client: AsyncAzureOpenAI):
        self.quality_evaluator = QualityEvaluator(azure_client)
        self.logger = StructuredLogger()
        
    async def evaluate_research(
        self,
        trace_id: str,
        question: str,
        report: Any,
        retrieved_contexts: List[Any],
        performance_metrics: PerformanceMetrics
    ) -> EvaluationResult:
        """
        Complete evaluation of a research operation.
        """
        print(f"\n[Evaluation] Starting quality assessment (trace: {trace_id})...")
        
        # Evaluate quality
        quality_metrics = await self.quality_evaluator.evaluate_quality(
            question, report, retrieved_contexts
        )
        
        # Create evaluation result
        eval_result = EvaluationResult(
            trace_id=trace_id,
            timestamp=datetime.now().isoformat(),
            question=question,
            quality_metrics=quality_metrics,
            performance_metrics=performance_metrics,
            report_length=len(report.detailed_findings),
            num_citations=len(report.citations),
            num_sources=len(report.sources)
        )
        
        # Log the evaluation
        self.logger.log_evaluation(eval_result)
        
        print(f"[Evaluation] Quality Assessment Complete:")
        print(f"  Overall Score: {quality_metrics.overall_score:.2f}")
        print(f"  Relevance: {quality_metrics.relevance_score:.2f}")
        print(f"  Completeness: {quality_metrics.completeness_score:.2f}")
        print(f"  Citation Quality: {quality_metrics.citation_quality_score:.2f}")
        print(f"  Groundedness: {quality_metrics.groundedness_score:.2f}")
        print(f"  Reasoning: {quality_metrics.reasoning[:150]}...")
        
        return eval_result
    
    def generate_performance_report(self, eval_result: EvaluationResult):
        """Generate a detailed performance report."""
        print(f"\n{'='*60}")
        print(f"PERFORMANCE REPORT")
        print(f"{'='*60}")
        print(f"Trace ID: {eval_result.trace_id}")
        print(f"Timestamp: {eval_result.timestamp}")
        print(f"\nQuery: {eval_result.question}")
        
        print(f"\n--- Quality Metrics ---")
        print(f"Overall Score: {eval_result.quality_metrics.overall_score:.2%}")
        print(f"  Relevance:        {eval_result.quality_metrics.relevance_score:.2%}")
        print(f"  Completeness:     {eval_result.quality_metrics.completeness_score:.2%}")
        print(f"  Citation Quality: {eval_result.quality_metrics.citation_quality_score:.2%}")
        print(f"  Groundedness:     {eval_result.quality_metrics.groundedness_score:.2%}")
        
        print(f"\n--- Performance Metrics ---")
        perf = eval_result.performance_metrics
        print(f"Total Latency: {perf.total_latency_seconds:.2f}s")
        print(f"Agent Breakdown:")
        for agent, latency in perf.agent_latencies.items():
            tokens = perf.tokens_by_agent.get(agent, 0)
            print(f"  {agent:20s}: {latency:6.2f}s | {tokens:,} tokens")
        
        print(f"\nTotal Tokens: {perf.total_tokens:,}")
        print(f"Estimated Cost: ${perf.cost_usd:.4f}")
        print(f"Success: {perf.success}")
        
        print(f"\n--- Output Metrics ---")
        print(f"Report Length: {eval_result.report_length:,} characters")
        print(f"Citations: {eval_result.num_citations}")
        print(f"Unique Sources: {eval_result.num_sources}")
        
        # Calculate efficiency metrics
        if perf.total_latency_seconds > 0:
            throughput = eval_result.report_length / perf.total_latency_seconds
            print(f"\nThroughput: {throughput:.0f} chars/second")
        
        if perf.total_tokens > 0:
            cost_per_char = perf.cost_usd / eval_result.report_length if eval_result.report_length > 0 else 0
            print(f"Cost Efficiency: ${cost_per_char*1000:.4f} per 1K chars")
        
        print(f"{'='*60}\n")
        
    def compare_recent_performance(self, n: int = 5):
        """Compare performance across recent queries."""
        evaluations = self.logger.get_recent_evaluations(n)
        
        if not evaluations:
            print("No evaluation data available yet.")
            return
            
        print(f"\n{'='*60}")
        print(f"PERFORMANCE TRENDS (Last {len(evaluations)} queries)")
        print(f"{'='*60}\n")
        
        # Calculate averages
        avg_quality = sum(e.quality_metrics.overall_score for e in evaluations) / len(evaluations)
        avg_latency = sum(e.performance_metrics.total_latency_seconds for e in evaluations) / len(evaluations)
        avg_cost = sum(e.performance_metrics.cost_usd for e in evaluations) / len(evaluations)
        avg_tokens = sum(e.performance_metrics.total_tokens for e in evaluations) / len(evaluations)
        
        print(f"Average Quality Score: {avg_quality:.2%}")
        print(f"Average Latency: {avg_latency:.2f}s")
        print(f"Average Cost: ${avg_cost:.4f}")
        print(f"Average Tokens: {avg_tokens:,.0f}")
        
        print(f"\nQuality Score Distribution:")
        for i, eval_result in enumerate(evaluations, 1):
            score = eval_result.quality_metrics.overall_score
            bar = '█' * int(score * 50)
            print(f"  Query {i}: {bar} {score:.2%}")
        
        # Identify potential issues
        low_quality = [e for e in evaluations if e.quality_metrics.overall_score < 0.6]
        if low_quality:
            print(f"\n⚠️  Warning: {len(low_quality)} queries had quality scores below 60%")
            
        high_latency = [e for e in evaluations if e.performance_metrics.total_latency_seconds > 30]
        if high_latency:
            print(f"⚠️  Warning: {len(high_latency)} queries exceeded 30s latency")
            
        print(f"{'='*60}\n")


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

async def evaluate_research_pipeline(
    azure_openai_client,
    question: str,
    report,
    all_retrieved_contexts: List,
    performance_monitor: PerformanceMonitor
):
    """
    Helper function to evaluate a completed research pipeline.
    Call this after your research completes.
    """
    trace_id = str(uuid.uuid4())
    
    # Get performance metrics
    perf_metrics = performance_monitor.get_metrics()
    
    # Run evaluation
    evaluator = EvaluationOrchestrator(azure_openai_client)
    eval_result = await evaluator.evaluate_research(
        trace_id=trace_id,
        question=question,
        report=report,
        retrieved_contexts=all_retrieved_contexts,
        performance_metrics=perf_metrics
    )
    
    # Generate detailed report
    evaluator.generate_performance_report(eval_result)
    
    return eval_result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """
    Example showing how to integrate evaluation into your research system.
    """
    from openai import AsyncAzureOpenAI
    import os
    
    # Setup
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    client = AsyncAzureOpenAI(
        api_key=api_key,
        api_version="2024-02-15-preview",
        azure_endpoint=endpoint
    )
    
    # Create mock data for demonstration
    from dataclasses import dataclass
    
    @dataclass
    class MockReport:
        question: str
        detailed_findings: str
        citations: List[Dict]
        sources: List[str]
    
    @dataclass
    class MockContext:
        content: str
        source: str
        relevance_score: float
    
    # Simulate a research operation
    monitor = PerformanceMonitor()
    monitor.start_operation()
    
    # Simulate agent executions
    monitor.start_agent("Coordinator")
    await asyncio.sleep(0.5)  # Simulate work
    monitor.end_agent("Coordinator", 150)
    
    monitor.start_agent("Researcher")
    await asyncio.sleep(1.0)
    monitor.end_agent("Researcher", 800)
    
    # Create mock report
    report = MockReport(
        question="What are recent transformer improvements?",
        detailed_findings="Transformers have seen improvements in efficiency through sparse attention...",
        citations=[{'source': 'paper1.pdf', 'page': 3}],
        sources=['paper1.pdf']
    )
    
    contexts = [
        MockContext("Sparse attention reduces complexity...", "paper1.pdf", 0.9)
    ]
    
    # Run evaluation
    eval_result = await evaluate_research_pipeline(
        client, 
        report.question,
        report,
        contexts,
        monitor
    )
    
    # Show trends
    evaluator = EvaluationOrchestrator(client)
    evaluator.compare_recent_performance(n=5)


if __name__ == "__main__":
    asyncio.run(example_usage())