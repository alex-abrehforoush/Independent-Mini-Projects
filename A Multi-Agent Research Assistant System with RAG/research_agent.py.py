import asyncio
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from openai import AsyncAzureOpenAI
from datetime import datetime


# ============================================================================
# DATA STRUCTURES
# ============================================================================
# We use dataclasses to define structured data that flows between agents.
# This is better than passing raw strings because it enforces a schema.

@dataclass
class ResearchTask:
    """Represents a specific subtask identified by the coordinator."""
    task_id: int
    description: str
    search_query: str
    priority: str  # "high", "medium", "low"


@dataclass
class ResearchPlan:
    """The coordinator's output: a structured plan for research."""
    original_question: str
    tasks: List[ResearchTask]
    estimated_complexity: str


@dataclass
class SearchResult:
    """Simulated search result (in production, would come from real search API)."""
    task_id: int
    query: str
    content: str
    source: str
    relevance_score: float


@dataclass
class AnalyzedInsight:
    """The analyzer's output: key insights extracted from search results."""
    insight: str
    supporting_sources: List[str]
    confidence: str  # "high", "medium", "low"
    theme: str


@dataclass
class ResearchReport:
    """Final output: a complete research report."""
    question: str
    executive_summary: str
    detailed_findings: str
    key_insights: List[str]
    sources: List[str]
    timestamp: str


# ============================================================================
# TOKEN TRACKING AND CLIENT
# ============================================================================

class TokenTracker:
    """Tracks API usage and costs across all agents."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.calls_by_agent = {}
        self.tokens_by_model = {"gpt-35-turbo": 0, "gpt-4": 0}
        
    def add_usage(self, agent_name: str, model: str, input_tokens: int, output_tokens: int):
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
        
        # GPT-3.5: ~$0.002 per 1K tokens (average of input/output)
        # GPT-4: ~$0.045 per 1K tokens (average of input/output)
        gpt35_cost = (gpt35_tokens / 1000) * 0.002
        gpt4_cost = (gpt4_tokens / 1000) * 0.045
        
        return gpt35_cost + gpt4_cost
    
    def report(self):
        print(f"\n{'='*60}")
        print(f"TOKEN USAGE REPORT")
        print(f"{'='*60}")
        print(f"Total input tokens:  {self.total_input_tokens:,}")
        print(f"Total output tokens: {self.total_output_tokens:,}")
        print(f"\nCalls by agent:")
        for agent, count in self.calls_by_agent.items():
            print(f"  {agent}: {count}")
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
# AGENTS
# ============================================================================

class CoordinatorAgent:
    """
    The coordinator breaks down research questions into specific subtasks.
    This agent uses GPT-4 because planning requires strong reasoning.
    """
    
    def __init__(self, client: AzureOpenAIClient):
        self.client = client
        self.name = "Coordinator"
        
    async def create_plan(self, question: str) -> ResearchPlan:
        """
        Takes a research question and creates a structured plan.
        Returns a ResearchPlan object with specific tasks.
        """
        
        system_prompt = """You are a research coordinator. Your job is to break down research questions into specific, actionable subtasks.

For each subtask, you should:
1. Create a clear, specific description
2. Formulate a targeted search query
3. Assign a priority (high/medium/low)

Aim for 3-5 subtasks that together comprehensively address the question.
Respond in JSON format with this structure:
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

Break this down into specific research subtasks. Think about:
- What are the key components of this question?
- What specific information needs to be gathered?
- What's the logical order to investigate these components?"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"[{self.name}] Creating research plan...")
        
        # Use GPT-4 for better planning (you can use gpt-35-turbo to save cost in testing)
        response = await self.client.create_completion(
            agent_name=self.name,
            messages=messages,
            deployment_name="gpt-35-turbo",  # Switch to "gpt-4" if available for better results
            temperature=0.7,
            max_tokens=1000
        )
        
        # Parse the JSON response
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
            # Fallback: create a simple single-task plan
            return ResearchPlan(
                original_question=question,
                tasks=[ResearchTask(1, question, question, "high")],
                estimated_complexity="simple"
            )


class ResearcherAgent:
    """
    The researcher executes search tasks and gathers information.
    In production, this would interface with real search APIs.
    """
    
    def __init__(self, client: AzureOpenAIClient):
        self.client = client
        self.name = "Researcher"
        
    async def execute_search(self, task: ResearchTask) -> List[SearchResult]:
        """
        Simulates searching for information related to a task.
        In production, this would call actual search APIs (web search, arxiv, etc.)
        """
        
        system_prompt = """You are a research agent that finds relevant information for specific queries.
Generate detailed, informative content that would be found when searching for the given query.
Include specific facts, data, and examples. Write 2-3 paragraphs of relevant information."""

        user_prompt = f"""Search query: {task.search_query}

Task context: {task.description}

Provide detailed information that would be found when researching this topic."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"[{self.name}] Researching: {task.description}")
        
        # Use cheaper model for information gathering
        response = await self.client.create_completion(
            agent_name=self.name,
            messages=messages,
            deployment_name="gpt-35-turbo",
            temperature=0.7,
            max_tokens=800
        )
        
        # Package as a search result
        result = SearchResult(
            task_id=task.task_id,
            query=task.search_query,
            content=response,
            source=f"simulated_search_{task.task_id}",
            relevance_score=0.9  # In production, this would come from search API
        )
        
        return [result]
    
    async def gather_all_information(self, plan: ResearchPlan) -> List[SearchResult]:
        """
        Executes all research tasks concurrently.
        This demonstrates async parallelism - multiple searches run simultaneously.
        """
        print(f"[{self.name}] Starting research on {len(plan.tasks)} tasks...")
        
        # Create coroutines for all tasks
        search_coroutines = [self.execute_search(task) for task in plan.tasks]
        
        # Run all searches in parallel and flatten results
        results_list = await asyncio.gather(*search_coroutines)
        all_results = [result for results in results_list for result in results]
        
        print(f"[{self.name}] Gathered {len(all_results)} results")
        return all_results


class AnalyzerAgent:
    """
    The analyzer processes raw search results and extracts key insights.
    This agent performs critical evaluation and synthesis.
    """
    
    def __init__(self, client: AzureOpenAIClient):
        self.client = client
        self.name = "Analyzer"
        
    async def analyze_results(
        self, 
        question: str, 
        results: List[SearchResult]
    ) -> List[AnalyzedInsight]:
        """
        Analyzes search results to extract key insights.
        Groups information by themes and evaluates quality.
        """
        
        # Combine all search results into context
        context = "\n\n".join([
            f"Source {i+1} (Query: {r.query}):\n{r.content}"
            for i, r in enumerate(results)
        ])
        
        system_prompt = """You are an analytical research agent. Your job is to:
1. Identify key insights from the provided information
2. Group related information into themes
3. Evaluate the confidence level of each insight
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
            temperature=0.5,  # Lower temperature for more focused analysis
            max_tokens=1200
        )
        
        try:
            analysis_data = json.loads(response)
            insights = [AnalyzedInsight(**insight) for insight in analysis_data["insights"]]
            print(f"[{self.name}] Extracted {len(insights)} insights")
            return insights
            
        except json.JSONDecodeError as e:
            print(f"[{self.name}] Failed to parse analysis: {e}")
            # Fallback: create a simple insight
            return [AnalyzedInsight(
                insight="Analysis could not be structured properly",
                supporting_sources=["all"],
                confidence="low",
                theme="general"
            )]


class WriterAgent:
    """
    The writer synthesizes insights into a coherent research report.
    This agent focuses on clarity, organization, and readability.
    """
    
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
        """
        Creates a final research report from analyzed insights.
        """
        
        # Format insights for the prompt
        insights_text = "\n".join([
            f"- {insight.insight} (Theme: {insight.theme}, Confidence: {insight.confidence})"
            for insight in insights
        ])
        
        system_prompt = """You are a technical writer creating research reports.
Your report should:
1. Start with a concise executive summary (2-3 sentences)
2. Present detailed findings organized logically
3. Highlight key insights clearly
4. Be clear, well-organized, and professional
5. Use specific information from the provided insights

Write in a clear, engaging style appropriate for technical stakeholders."""

        user_prompt = f"""Research question: {question}

Key insights identified:
{insights_text}

Write a comprehensive research report that addresses the original question."""

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
        
        # Structure the response
        # In a more sophisticated version, we'd ask the LLM to return JSON
        # For now, we'll use the full response as detailed findings
        
        key_insights_list = [insight.insight for insight in insights[:5]]
        sources_list = list(set([r.source for r in results]))
        
        report = ResearchReport(
            question=question,
            executive_summary=response.split("\n\n")[0],  # First paragraph as summary
            detailed_findings=response,
            key_insights=key_insights_list,
            sources=sources_list,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"[{self.name}] Report completed")
        return report


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class ResearchOrchestrator:
    """
    The orchestrator coordinates all agents and manages the overall workflow.
    This is the main entry point for the multi-agent system.
    """
    
    def __init__(self, client: AzureOpenAIClient):
        self.client = client
        self.coordinator = CoordinatorAgent(client)
        self.researcher = ResearcherAgent(client)
        self.analyzer = AnalyzerAgent(client)
        self.writer = WriterAgent(client)
        
    async def conduct_research(self, question: str) -> ResearchReport:
        """
        Executes the full research pipeline:
        1. Coordinator creates a plan
        2. Researcher gathers information
        3. Analyzer extracts insights
        4. Writer produces final report
        """
        print(f"\n{'='*60}")
        print(f"STARTING RESEARCH: {question}")
        print(f"{'='*60}\n")
        
        # Step 1: Planning
        plan = await self.coordinator.create_plan(question)
        
        # Step 2: Research
        results = await self.researcher.gather_all_information(plan)
        
        # Step 3: Analysis
        insights = await self.analyzer.analyze_results(question, results)
        
        # Step 4: Writing
        report = await self.writer.write_report(question, plan, insights, results)
        
        print(f"\n{'='*60}")
        print(f"RESEARCH COMPLETED")
        print(f"{'='*60}\n")
        
        return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """
    Main function demonstrating the complete research system.
    """
    
    # Setup Azure OpenAI client
    # You need to set these environment variables:
    # export AZURE_OPENAI_KEY="your-key"
    # export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
    
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not api_key or not endpoint:
        print("Error: Please set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT environment variables")
        return
    
    # Initialize tracking and client
    tracker = TokenTracker()
    client = AzureOpenAIClient(api_key, endpoint, tracker)
    
    # Create orchestrator
    orchestrator = ResearchOrchestrator(client)
    
    # Example research question
    question = "What are the latest developments in transformer architecture improvements for large language models?"
    
    # Conduct research
    report = await orchestrator.conduct_research(question)
    
    # Display results
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60 + "\n")
    print(f"Question: {report.question}\n")
    print(f"Executive Summary:\n{report.executive_summary}\n")
    print(f"\nDetailed Findings:\n{report.detailed_findings}\n")
    print(f"\nKey Insights:")
    for i, insight in enumerate(report.key_insights, 1):
        print(f"{i}. {insight}")
    
    # Show cost tracking
    tracker.report()


if __name__ == "__main__":
    asyncio.run(main()) 
