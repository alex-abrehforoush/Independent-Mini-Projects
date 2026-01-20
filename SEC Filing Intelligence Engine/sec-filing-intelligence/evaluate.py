from src.document_parser import SECFilingParser
from src.extractor import SECExtractor
from src.evaluator import ExtractionEvaluator
from src.schemas import SECFilingExtraction
from datetime import datetime
import json

def main():
    print("=" * 80)
    print("EXTRACTION EVALUATION")
    print("=" * 80)
    
    # Load and extract
    print("\n[1/3] Running extraction...")
    parser = SECFilingParser("data/apple_10k_2025.html")
    parser.load()
    
    risk_text = parser.extract_section("risk factors")
    financial_text = parser.extract_section("financial performance")
    
    extractor = SECExtractor()
    risks = extractor.extract_risks(risk_text)
    revenue = extractor.extract_revenue(financial_text)
    financials = extractor.extract_financials(financial_text)
    
    extraction = SECFilingExtraction(
        company="Apple Inc.",
        filing_type="10-K",
        fiscal_year="2025",
        extraction_timestamp=datetime.now().isoformat(),
        risks=risks,
        revenue=revenue,
        financials=financials,
        metadata={}
    )
    
    # Evaluate accuracy
    print("\n[2/3] Evaluating accuracy...")
    evaluator = ExtractionEvaluator("ground_truth/apple_2025_revenue.json")
    
    revenue_eval = evaluator.evaluate_revenue(extraction)
    risk_eval = evaluator.evaluate_risks(extraction)
    
    print(f"\n  Revenue Extraction Accuracy: {revenue_eval['accuracy_percentage']:.1f}%")
    print(f"  Correct segments: {revenue_eval['correct_segments']}/{revenue_eval['total_segments']}")
    
    if revenue_eval['errors']:
        print("\n  Errors:")
        for error in revenue_eval['errors']:
            print(f"    - {error}")
    
    print(f"\n  Risk Coverage: {risk_eval['coverage_percentage']:.1f}%")
    print(f"  Key risks identified: {risk_eval['identified_risks']}/{risk_eval['total_key_risks']}")
    
    # Cost analysis
    print("\n[3/3] Cost analysis...")
    cost_summary = extractor.get_cost_summary()
    
    print(f"\n  Total API calls: {cost_summary['api_calls']}")
    print(f"  Total tokens: {cost_summary['total_tokens']:,}")
    print(f"    - Input: {cost_summary['total_input_tokens']:,}")
    print(f"    - Output: {cost_summary['total_output_tokens']:,}")
    print(f"  Estimated cost: ${cost_summary['estimated_cost_usd']:.4f}")
    
    print("\n  Per-operation breakdown:")
    for call in cost_summary['breakdown']:
        print(f"    - {call['operation']}: {call['total_tokens']:,} tokens")
    
    # Save full report
    report = {
        "extraction_timestamp": extraction.extraction_timestamp,
        "accuracy": {
            "revenue": revenue_eval,
            "risks": risk_eval
        },
        "cost": cost_summary
    }
    
    with open("output/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Revenue accuracy: {revenue_eval['accuracy_percentage']:.1f}%")
    print(f"✓ Risk coverage: {risk_eval['coverage_percentage']:.1f}%")
    print(f"✓ Cost per extraction: ${cost_summary['estimated_cost_usd']:.4f}")
    print(f"✓ Report saved to output/evaluation_report.json")

if __name__ == "__main__":
    main()