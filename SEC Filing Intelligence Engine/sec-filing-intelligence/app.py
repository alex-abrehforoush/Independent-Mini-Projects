import streamlit as st
from src.document_parser import SECFilingParser
from src.extractor import SECExtractor
from src.evaluator import ExtractionEvaluator
from src.schemas import SECFilingExtraction
from datetime import datetime
import json
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="SEC Filing Intelligence Engine",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 SEC Filing Intelligence Engine")
st.markdown("**Extract structured data from 10-K filings using LLMs with governance controls**")

# Sidebar
st.sidebar.header("Configuration")
run_mode = st.sidebar.radio(
    "Mode",
    ["Quick Demo (Pre-loaded Apple 10-K)", "Upload Custom Filing"],
    help="Demo mode uses Apple's 2025 10-K for instant results"
)

enable_governance = st.sidebar.checkbox("Enable Governance Controls", value=True)
show_audit_log = st.sidebar.checkbox("Show Audit Trail", value=False)

# Main content
if run_mode == "Quick Demo (Pre-loaded Apple 10-K)":
    st.info("Using pre-loaded Apple Inc. 10-K (FY2025)")
    
    if st.button("🚀 Run Extraction", type="primary"):
        with st.spinner("Processing SEC filing..."):
            # Load document
            parser = SECFilingParser("data/apple_10k_2025.html")
            parser.load()
            
            risk_text = parser.extract_section("risk factors")
            financial_text = parser.extract_section("financial performance")
            
            # Progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Extract with or without governance
            extractor = SECExtractor()
            
            if enable_governance:
                status_text.text("Extracting with governance controls...")
                progress_bar.progress(25)
                
                results = extractor.extract_with_governance(
                    document_id="AAPL-10K-FY2025",
                    risk_text=risk_text,
                    financial_text=financial_text
                )
                
                risks = results["extractions"]["risks"]
                revenue = results["extractions"]["revenue"]
                financials = results["extractions"]["financials"]
                governance = results["governance"]
                
            else:
                status_text.text("Extracting data...")
                progress_bar.progress(25)
                
                risks = extractor.extract_risks(risk_text)
                progress_bar.progress(50)
                revenue = extractor.extract_revenue(financial_text)
                progress_bar.progress(75)
                financials = extractor.extract_financials(financial_text)
                governance = None
            
            progress_bar.progress(100)
            status_text.text("✓ Extraction complete!")
            
            # Display results
            st.success("Extraction completed successfully!")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Risk Categories", len(risks.risks))
            with col2:
                st.metric("Revenue Segments", len(revenue.segments))
            with col3:
                st.metric("Financial Metrics", len(financials.metrics))
            with col4:
                cost = extractor.get_cost_summary()
                st.metric("Cost (USD)", f"${cost['estimated_cost_usd']:.4f}")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Revenue Breakdown",
                "⚠️ Risk Factors",
                "💰 Financial Metrics",
                "🔒 Governance"
            ])
            
            with tab1:
                st.subheader("Revenue by Segment (FY2025)")
                
                # Create DataFrame
                revenue_data = []
                for seg in revenue.segments:
                    revenue_data.append({
                        "Segment": seg.segment_name,
                        "FY2025 ($M)": f"${seg.revenue_current:,.0f}",
                        "FY2024 ($M)": f"${seg.revenue_prior:,.0f}" if seg.revenue_prior else "N/A",
                        "YoY Change": f"{seg.percentage_change:+.1f}%" if seg.percentage_change else "N/A"
                    })
                
                df = pd.DataFrame(revenue_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Bar chart
                chart_data = pd.DataFrame({
                    "Segment": [seg.segment_name for seg in revenue.segments],
                    "Revenue": [seg.revenue_current for seg in revenue.segments]
                })
                st.bar_chart(chart_data.set_index("Segment"))
                
                if revenue.total_revenue:
                    st.info(f"**Total Revenue:** ${revenue.total_revenue:,.0f}M")
            
            with tab2:
                st.subheader("Identified Risk Categories")
                
                for i, risk in enumerate(risks.risks, 1):
                    severity_color = {
                        "High": "🔴",
                        "Medium": "🟡",
                        "Low": "🟢"
                    }.get(risk.severity, "⚪")
                    
                    with st.expander(f"{severity_color} {risk.category} ({risk.severity})"):
                        st.write(risk.description)
            
            with tab3:
                st.subheader("Key Financial Metrics (FY2025)")
                
                metrics_data = []
                for metric in financials.metrics:
                    metrics_data.append({
                        "Metric": metric.metric_name,
                        "Value": metric.value,
                        "Fiscal Year": metric.fiscal_year
                    })
                
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            
            with tab4:
                if enable_governance and governance:
                    st.subheader("Governance & Compliance")
                    
                    # PII Detection
                    st.markdown("#### PII Detection")
                    if governance.get("pii_detected"):
                        st.warning("⚠️ PII detected in document")
                        st.json(governance["pii_detected"])
                    else:
                        st.success("✓ No PII detected")
                    
                    # Confidence Scores
                    st.markdown("#### Confidence Scores")
                    for extraction_type, score in governance.get("confidence_scores", {}).items():
                        st.metric(extraction_type, f"{score:.1%}")
                    
                    # Validation Results
                    st.markdown("#### Data Validation")
                    for validation_type, result in governance.get("validations", {}).items():
                        if result["valid"]:
                            st.success(f"✓ {validation_type}: PASSED")
                        else:
                            st.error(f"✗ {validation_type}: FAILED")
                            for issue in result.get("issues", []):
                                st.write(f"  - {issue}")
                    
                    # Audit Trail
                    if show_audit_log:
                        st.markdown("#### Audit Trail")
                        audit_summary = extractor.audit_logger.get_session_summary()
                        st.json(audit_summary)
                        
                        with open(audit_summary['log_file'], 'r') as f:
                            st.code(f.read(), language="json")
                else:
                    st.info("Enable governance controls in sidebar to see compliance features")
            
            # Download button
            st.markdown("---")
            
            extraction_output = SECFilingExtraction(
                company="Apple Inc.",
                filing_type="10-K",
                fiscal_year="2025",
                extraction_timestamp=datetime.now().isoformat(),
                risks=risks,
                revenue=revenue,
                financials=financials,
                metadata={
                    "governance": governance if enable_governance else {},
                    "cost": extractor.get_cost_summary()
                }
            )
            
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json.dumps(extraction_output.model_dump(), indent=2),
                file_name=f"sec_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

else:
    st.warning("Custom file upload not implemented in this demo. Use 'Quick Demo' mode.")
    st.markdown("""
    **To use custom filings:**
    1. Download 10-K HTML from SEC EDGAR
    2. Place in `data/` folder
    3. Update `SECFilingParser` path in code
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>SEC Filing Intelligence Engine | Built with LangChain + Azure OpenAI | 
    <a href='https://github.com/alex-abrehforoush/sec-filing-intelligence'>GitHub</a>
    </small>
</div>
""", unsafe_allow_html=True)