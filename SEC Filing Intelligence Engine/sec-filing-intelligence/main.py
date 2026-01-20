from src.document_parser import SECFilingParser
from src.extractor import SECExtractor
from src.schemas import SECFilingExtraction
from datetime import datetime
import json

def main():
    print("=" * 80)
    print("SEC FILING INTELLIGENCE ENGINE")
    print("=" * 80)
    
    # Load document
    print("\n[1/4] Loading Apple 10-K...")
    parser = SECFilingParser("data/apple_10k_2025.html")
    parser.load()
    print("✓ Document loaded")
    
    # Extract sections
    print("\n[2/4] Extracting sections...")
    risk_text = parser.extract_section("risk factors")
    financial_text = parser.extract_section("financial performance")
    
    print(f"  - Risk Factors: {len(risk_text):,} chars")
    print(f"  - Financial Data: {len(financial_text):,} chars")
    
    if not risk_text:
        print("ERROR: Could not extract risk factors")
        return
    
    # Extract structured data
    print("\n[3/4] Extracting structured data with LLM...")
    extractor = SECExtractor()
    
    print("  - Extracting risks...")
    risks = extractor.extract_risks(risk_text)
    print(f"    ✓ {len(risks.risks)} risk categories")
    
    print("  - Extracting revenue breakdown...")
    revenue = extractor.extract_revenue(financial_text) if financial_text else None
    if revenue:
        print(f"    ✓ {len(revenue.segments)} revenue segments")
    else:
        print("    ⚠ No revenue data extracted")
    
    print("  - Extracting financial metrics...")
    financials = extractor.extract_financials(financial_text) if financial_text else None
    if financials:
        print(f"    ✓ {len(financials.metrics)} metrics")
    else:
        print("    ⚠ No financial metrics extracted")
    
    # Create complete extraction
    extraction = SECFilingExtraction(
        company="Apple Inc.",
        filing_type="10-K",
        fiscal_year="2025",
        extraction_timestamp=datetime.now().isoformat(),
        risks=risks,
        revenue=revenue or RevenueBreakdown(segments=[]),
        financials=financials or FinancialMetrics(metrics=[]),
        metadata={
            "model": "gpt-4o-mini",
            "risk_text_length": len(risk_text),
            "financial_text_length": len(financial_text)
        }
    )
    
    # Save
    print("\n[4/4] Saving results...")
    output_path = "output/complete_extraction.json"
    with open(output_path, "w") as f:
        json.dump(extraction.model_dump(), f, indent=2)
    
    print(f"✓ Saved to {output_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Risks: {len(risks.risks)}")
    if revenue:
        print(f"Revenue Segments: {len(revenue.segments)}")
    if financials:
        print(f"Financial Metrics: {len(financials.metrics)}")
    
    print("\nSample data:")
    print(f"\nTop Risk: {risks.risks[0].category} ({risks.risks[0].severity})")
    if revenue and revenue.segments:
        print(f"Top Revenue Segment: {revenue.segments[0].segment_name} - ${revenue.segments[0].revenue_current}M")

if __name__ == "__main__":
    main()