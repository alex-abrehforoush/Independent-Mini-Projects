from src.document_parser import SECFilingParser
from src.extractor import SECExtractor
from src.schemas import SECFilingExtraction
from datetime import datetime
import json

def main():
    print("=" * 80)
    print("GOVERNANCE & COMPLIANCE DEMONSTRATION")
    print("=" * 80)
    
    # Load document
    print("\n[1/4] Loading Apple 10-K...")
    parser = SECFilingParser("data/apple_10k_2025.html")
    parser.load()
    
    risk_text = parser.extract_section("risk factors")
    financial_text = parser.extract_section("financial performance")
    print("✓ Document loaded")
    
    # Extract with governance
    print("\n[2/4] Extracting with governance controls...")
    extractor = SECExtractor()
    
    results = extractor.extract_with_governance(
        document_id="AAPL-10K-FY2025",
        risk_text=risk_text,
        financial_text=financial_text
    )
    
    # Display governance metrics
    print("\n[3/4] Governance Summary:")
    gov = results["governance"]
    
    print("\n  PII Detection:")
    if gov["pii_detected"]:
        for section, findings in gov["pii_detected"].items():
            if findings:
                print(f"    ⚠ {section}: {findings}")
            else:
                print(f"    ✓ {section}: No PII detected")
    else:
        print("    ✓ No PII detected in any section")
    
    print("\n  Confidence Scores:")
    for extraction_type, score in gov["confidence_scores"].items():
        status = "✓" if score > 0.8 else "⚠"
        print(f"    {status} {extraction_type}: {score:.2%}")
    
    print("\n  Data Validation:")
    for validation_type, result in gov["validations"].items():
        status = "✓" if result["valid"] else "✗"
        print(f"    {status} {validation_type}: {'PASSED' if result['valid'] else 'FAILED'}")
        if not result["valid"] and result.get("issues"):
            for issue in result["issues"]:
                print(f"       - {issue}")
    
    # Audit trail summary
    print("\n[4/4] Audit Trail:")
    audit_summary = extractor.audit_logger.get_session_summary()
    print(f"  Session ID: {audit_summary['session_id']}")
    print(f"  Total operations logged: {audit_summary['total_operations']}")
    print(f"  Total validations: {audit_summary['total_validations']}")
    print(f"  Validations passed: {audit_summary['validations_passed']}/{audit_summary['total_validations']}")
    print(f"  Audit log: {audit_summary['log_file']}")
    
    # Save complete output
    extraction = SECFilingExtraction(
        company="Apple Inc.",
        filing_type="10-K",
        fiscal_year="2025",
        extraction_timestamp=datetime.now().isoformat(),
        risks=results["extractions"]["risks"],
        revenue=results["extractions"]["revenue"],
        financials=results["extractions"]["financials"],
        metadata={
            "governance": gov,
            "audit_session": audit_summary['session_id']
        }
    )
    
    with open("output/governed_extraction.json", "w") as f:
        json.dump(extraction.model_dump(), f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓ Governance-enabled extraction complete")
    print("✓ Results saved to output/governed_extraction.json")
    print(f"✓ Audit trail: {audit_summary['log_file']}")

if __name__ == "__main__":
    main()