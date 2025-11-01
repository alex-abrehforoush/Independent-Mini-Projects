import streamlit as st
import asyncio
import os
from pathlib import Path
import json
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Import your research system components
# Make sure these imports match your actual file structure
from research_agent import (
    TokenTracker, AzureOpenAIClient, VectorStore,
    DocumentProcessor, CoordinatorAgent, RAGResearcherAgent,
    AnalyzerAgent, WriterAgent, ResearchReport
)
from evaluation_monitoring import (
    PerformanceMonitor, EvaluationOrchestrator,
    evaluate_research_pipeline
)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .agent-status {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-weight: bold;
    }
    .agent-running {
        background-color: #ffd700;
        color: #000;
    }
    .agent-complete {
        background-color: #90ee90;
        color: #000;
    }
    .quality-excellent {
        color: #008000;
        font-weight: bold;
    }
    .quality-good {
        color: #808000;
        font-weight: bold;
    }
    .quality-poor {
        color: #ff0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.documents_ingested = []
        st.session_state.research_history = []
        st.session_state.agent_status = {}
        st.session_state.current_report = None
        st.session_state.evaluation_results = []


# ============================================================================
# AZURE CLIENT SETUP
# ============================================================================

@st.cache_resource
def get_azure_client():
    """Get or create Azure OpenAI client (cached)."""
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not api_key or not endpoint:
        st.error("⚠️ Azure OpenAI credentials not found in environment variables!")
        st.stop()
    
    tracker = TokenTracker()
    client = AzureOpenAIClient(api_key, endpoint, tracker)
    return client, tracker


@st.cache_resource
def get_vector_store():
    """Get or create vector store (cached)."""
    return VectorStore(collection_name="research_documents_ui")


# ============================================================================
# DOCUMENT INGESTION
# ============================================================================

async def ingest_documents_async(uploaded_files, vector_store, client):
    """Ingest uploaded documents into vector store."""
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    # Create temporary directory for uploads
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    all_chunks = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Save uploaded file temporarily
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        status_text.text(f"Processing {uploaded_file.name}...")
        
        try:
            chunks = processor.process_document(str(file_path))
            all_chunks.extend(chunks)
            st.session_state.documents_ingested.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {e}")
        finally:
            # Clean up temp file
            file_path.unlink(missing_ok=True)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    if all_chunks:
        status_text.text("Generating embeddings and storing in vector database...")
        await vector_store.add_documents(all_chunks, client.client)
        status_text.success(f"✅ Successfully ingested {len(all_chunks)} chunks from {len(uploaded_files)} documents!")
    
    progress_bar.empty()
    return len(all_chunks)


# ============================================================================
# RESEARCH EXECUTION
# ============================================================================

async def conduct_research_async(question, client, vector_store):
    """Execute research with real-time status updates."""
    
    # Initialize agents
    coordinator = CoordinatorAgent(client)
    researcher = RAGResearcherAgent(client, vector_store)
    analyzer = AnalyzerAgent(client)
    writer = WriterAgent(client)
    evaluator = EvaluationOrchestrator(client.client)
    
    # Performance monitoring
    monitor = PerformanceMonitor()
    monitor.start_operation()
    
    # Create status containers
    status_container = st.container()
    
    with status_container:
        st.markdown("### 🤖 Agent Execution Pipeline")
        coord_status = st.empty()
        research_status = st.empty()
        analyze_status = st.empty()
        write_status = st.empty()
    
    all_retrieved_contexts = []
    
    try:
        # Step 1: Coordinator
        coord_status.markdown('<div class="agent-status agent-running">🔄 Coordinator: Creating research plan...</div>', unsafe_allow_html=True)
        monitor.start_agent("Coordinator")
        tokens_before = client.tracker.total_input_tokens + client.tracker.total_output_tokens
        
        plan = await coordinator.create_plan(question)
        
        tokens_after = client.tracker.total_input_tokens + client.tracker.total_output_tokens
        monitor.end_agent("Coordinator", tokens_after - tokens_before)
        coord_status.markdown(f'<div class="agent-status agent-complete">✅ Coordinator: Created {len(plan.tasks)} research tasks</div>', unsafe_allow_html=True)
        
        # Step 2: Researcher
        research_status.markdown('<div class="agent-status agent-running">🔄 Researcher: Gathering information from documents...</div>', unsafe_allow_html=True)
        monitor.start_agent("Researcher")
        tokens_before = client.tracker.total_input_tokens + client.tracker.total_output_tokens
        
        results = await researcher.gather_all_information(plan)
        
        for result in results:
            all_retrieved_contexts.extend(result.retrieved_contexts)
        
        tokens_after = client.tracker.total_input_tokens + client.tracker.total_output_tokens
        monitor.end_agent("Researcher", tokens_after - tokens_before)
        research_status.markdown(f'<div class="agent-status agent-complete">✅ Researcher: Found {len(all_retrieved_contexts)} relevant contexts</div>', unsafe_allow_html=True)
        
        # Step 3: Analyzer
        analyze_status.markdown('<div class="agent-status agent-running">🔄 Analyzer: Extracting key insights...</div>', unsafe_allow_html=True)
        monitor.start_agent("Analyzer")
        tokens_before = client.tracker.total_input_tokens + client.tracker.total_output_tokens
        
        insights = await analyzer.analyze_results(question, results)
        
        tokens_after = client.tracker.total_input_tokens + client.tracker.total_output_tokens
        monitor.end_agent("Analyzer", tokens_after - tokens_before)
        analyze_status.markdown(f'<div class="agent-status agent-complete">✅ Analyzer: Extracted {len(insights)} key insights</div>', unsafe_allow_html=True)
        
        # Step 4: Writer
        write_status.markdown('<div class="agent-status agent-running">🔄 Writer: Synthesizing final report...</div>', unsafe_allow_html=True)
        monitor.start_agent("Writer")
        tokens_before = client.tracker.total_input_tokens + client.tracker.total_output_tokens
        
        report = await writer.write_report(question, plan, insights, results)
        
        tokens_after = client.tracker.total_input_tokens + client.tracker.total_output_tokens
        monitor.end_agent("Writer", tokens_after - tokens_before)
        write_status.markdown('<div class="agent-status agent-complete">✅ Writer: Report completed</div>', unsafe_allow_html=True)
        
        # Step 5: Evaluation
        st.markdown("### 📊 Evaluating Quality...")
        eval_result = await evaluate_research_pipeline(
            client.client,
            question,
            report,
            all_retrieved_contexts,
            monitor
        )
        
        return report, eval_result
        
    except Exception as e:
        st.error(f"❌ Research failed: {str(e)}")
        return None, None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_quality_radar(quality_metrics):
    """Create radar chart for quality metrics."""
    categories = ['Relevance', 'Completeness', 'Citation Quality', 'Groundedness']
    values = [
        quality_metrics.relevance_score,
        quality_metrics.completeness_score,
        quality_metrics.citation_quality_score,
        quality_metrics.groundedness_score
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='#1f77b4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Quality Assessment",
        height=400
    )
    
    return fig


def plot_agent_performance(performance_metrics):
    """Create bar chart for agent latency."""
    agents = list(performance_metrics.agent_latencies.keys())
    latencies = list(performance_metrics.agent_latencies.values())
    tokens = [performance_metrics.tokens_by_agent.get(agent, 0) for agent in agents]
    
    fig = go.Figure(data=[
        go.Bar(name='Latency (s)', x=agents, y=latencies, yaxis='y', offsetgroup=1),
        go.Bar(name='Tokens', x=agents, y=tokens, yaxis='y2', offsetgroup=2)
    ])
    
    fig.update_layout(
        title='Agent Performance',
        yaxis=dict(title='Latency (seconds)'),
        yaxis2=dict(title='Tokens', overlaying='y', side='right'),
        barmode='group',
        height=400
    )
    
    return fig


def plot_quality_trends(evaluation_results):
    """Plot quality score trends over time."""
    if not evaluation_results:
        return None
    
    df = pd.DataFrame([
        {
            'timestamp': eval_res.timestamp,
            'overall_score': eval_res.quality_metrics.overall_score,
            'relevance': eval_res.quality_metrics.relevance_score,
            'completeness': eval_res.quality_metrics.completeness_score,
            'groundedness': eval_res.quality_metrics.groundedness_score
        }
        for eval_res in evaluation_results
    ])
    
    fig = px.line(df, x='timestamp', y=['overall_score', 'relevance', 'completeness', 'groundedness'],
                  title='Quality Metrics Over Time',
                  labels={'value': 'Score', 'variable': 'Metric'},
                  height=400)
    
    return fig


# ============================================================================
# UI PAGES
# ============================================================================

def document_upload_page():
    """Document upload and management page."""
    st.markdown('<div class="main-header">📚 Document Management</div>', unsafe_allow_html=True)
    
    # Get clients
    client, tracker = get_azure_client()
    vector_store = get_vector_store()
    
    # Upload section
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or text files",
        type=['pdf', 'txt', 'md'],
        accept_multiple_files=True,
        help="Upload research papers, articles, or any documents you want to query"
    )
    
    if uploaded_files:
        if st.button("🚀 Ingest Documents", type="primary"):
            with st.spinner("Processing documents..."):
                num_chunks = asyncio.run(ingest_documents_async(uploaded_files, vector_store, client))
                st.success(f"✅ Successfully processed {num_chunks} chunks!")
    
    # Show ingested documents
    if st.session_state.documents_ingested:
        st.markdown("### 📄 Ingested Documents")
        for doc in st.session_state.documents_ingested:
            st.text(f"✓ {doc}")
    
    # Clear database option
    st.markdown("---")
    if st.button("🗑️ Clear Document Database", help="Remove all ingested documents"):
        vector_store.clear()
        st.session_state.documents_ingested = []
        st.success("Database cleared!")
        st.rerun()


def research_page():
    """Main research interface."""
    st.markdown('<div class="main-header">🔬 Research Assistant</div>', unsafe_allow_html=True)
    
    # Get clients
    client, tracker = get_azure_client()
    vector_store = get_vector_store()
    
    # Research input
    st.markdown("### Ask a Research Question")
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., What are the recent developments in transformer architectures?",
        help="Ask questions about your uploaded documents"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_research = st.button("🔍 Research", type="primary", disabled=not question)
    
    if run_research and question:
        # Execute research
        with st.spinner("Conducting research..."):
            report, eval_result = asyncio.run(conduct_research_async(question, client, vector_store))
            
            if report and eval_result:
                st.session_state.current_report = report
                st.session_state.evaluation_results.append(eval_result)
                st.session_state.research_history.append({
                    'question': question,
                    'timestamp': datetime.now().isoformat(),
                    'report': report,
                    'evaluation': eval_result
                })
    
    # Display results
    if st.session_state.current_report:
        st.markdown("---")
        display_research_results(st.session_state.current_report, 
                                st.session_state.evaluation_results[-1] if st.session_state.evaluation_results else None)


def display_research_results(report, eval_result):
    """Display research results with quality metrics."""
    
    # Quality score badge
    if eval_result:
        quality_score = eval_result.quality_metrics.overall_score
        if quality_score >= 0.8:
            quality_class = "quality-excellent"
            emoji = "🌟"
        elif quality_score >= 0.6:
            quality_class = "quality-good"
            emoji = "✅"
        else:
            quality_class = "quality-poor"
            emoji = "⚠️"
        
        st.markdown(f'### {emoji} Overall Quality: <span class="{quality_class}">{quality_score:.1%}</span>', unsafe_allow_html=True)
    
    # Report content
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Report", "📊 Quality Metrics", "⚡ Performance", "📎 Citations"])
    
    with tab1:
        st.markdown("#### Executive Summary")
        st.info(report.executive_summary)
        
        st.markdown("#### Detailed Findings")
        st.markdown(report.detailed_findings)
        
        st.markdown("#### Key Insights")
        for i, insight in enumerate(report.key_insights, 1):
            st.markdown(f"{i}. {insight}")
    
    with tab2:
        if eval_result:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(plot_quality_radar(eval_result.quality_metrics), use_container_width=True)
            
            with col2:
                st.markdown("#### Score Breakdown")
                metrics = eval_result.quality_metrics
                st.metric("Relevance", f"{metrics.relevance_score:.1%}")
                st.metric("Completeness", f"{metrics.completeness_score:.1%}")
                st.metric("Citation Quality", f"{metrics.citation_quality_score:.1%}")
                st.metric("Groundedness", f"{metrics.groundedness_score:.1%}")
                
                st.markdown("#### Evaluator Reasoning")
                st.write(metrics.reasoning)
    
    with tab3:
        if eval_result:
            perf = eval_result.performance_metrics
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Latency", f"{perf.total_latency_seconds:.2f}s")
            col2.metric("Total Tokens", f"{perf.total_tokens:,}")
            col3.metric("Est. Cost", f"${perf.cost_usd:.4f}")
            
            st.plotly_chart(plot_agent_performance(perf), use_container_width=True)
    
    with tab4:
        st.markdown("#### Sources Used")
        for source in report.sources:
            st.text(f"📄 {source}")
        
        if report.citations:
            st.markdown("#### Citation Details")
            citation_df = pd.DataFrame([
                {
                    'Source': c['source'],
                    'Page': c.get('page', 'N/A'),
                    'Relevance': f"{c['relevance']:.2f}"
                }
                for c in report.citations[:10]
            ])
            st.dataframe(citation_df, use_container_width=True)


def dashboard_page():
    """Monitoring dashboard with analytics."""
    st.markdown('<div class="main-header">📊 Performance Dashboard</div>', unsafe_allow_html=True)
    
    client, tracker = get_azure_client()
    
    if not st.session_state.evaluation_results:
        st.info("No evaluation data yet. Conduct some research to see analytics!")
        return
    
    # Summary metrics
    st.markdown("### Overall Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_queries = len(st.session_state.evaluation_results)
    avg_quality = sum(e.quality_metrics.overall_score for e in st.session_state.evaluation_results) / total_queries
    total_cost = sum(e.performance_metrics.cost_usd for e in st.session_state.evaluation_results)
    avg_latency = sum(e.performance_metrics.total_latency_seconds for e in st.session_state.evaluation_results) / total_queries
    
    col1.metric("Total Queries", total_queries)
    col2.metric("Avg Quality", f"{avg_quality:.1%}")
    col3.metric("Total Cost", f"${total_cost:.4f}")
    col4.metric("Avg Latency", f"{avg_latency:.2f}s")
    
    # Quality trends
    st.markdown("### Quality Trends")
    fig = plot_quality_trends(st.session_state.evaluation_results)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent queries
    st.markdown("### Recent Queries")
    if st.session_state.research_history:
        for entry in reversed(st.session_state.research_history[-5:]):
            with st.expander(f"Q: {entry['question'][:100]}... ({entry['timestamp'][:19]})"):
                eval_res = entry['evaluation']
                col1, col2 = st.columns(2)
                col1.metric("Quality Score", f"{eval_res.quality_metrics.overall_score:.1%}")
                col2.metric("Cost", f"${eval_res.performance_metrics.cost_usd:.4f}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application."""
    init_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🔬 Multi-Agent Research")
        st.markdown("### Navigation")
        page = st.radio(
            "Go to:",
            ["📚 Document Management", "🔍 Research", "📊 Dashboard"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### System Info")
        client, tracker = get_azure_client()
        st.metric("Documents Loaded", len(st.session_state.documents_ingested))
        st.metric("Queries Completed", len(st.session_state.research_history))
        
        if tracker.total_input_tokens > 0:
            st.metric("Total Tokens Used", f"{tracker.total_input_tokens + tracker.total_output_tokens:,}")
            st.metric("Estimated Cost", f"${tracker.get_estimated_cost():.4f}")
    
    # Route to selected page
    if page == "📚 Document Management":
        document_upload_page()
    elif page == "🔍 Research":
        research_page()
    elif page == "📊 Dashboard":
        dashboard_page()


if __name__ == "__main__":
    main()