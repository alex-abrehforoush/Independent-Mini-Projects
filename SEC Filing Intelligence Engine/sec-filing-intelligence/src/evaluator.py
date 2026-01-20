import json
from typing import Dict, Any
from src.schemas import SECFilingExtraction

class ExtractionEvaluator:
    def __init__(self, ground_truth_path: str):
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)
    
    def evaluate_revenue(self, extraction: SECFilingExtraction) -> Dict[str, Any]:
        """
        Compare extracted revenue against ground truth.
        Returns accuracy metrics.
        """
        results = {
            "total_segments": 5,
            "correct_segments": 0,
            "errors": [],
            "accuracy_percentage": 0.0
        }
        
        gt_revenue = self.ground_truth["revenue_segments"]
        extracted = {seg.segment_name: seg for seg in extraction.revenue.segments}
        
        # Map extracted names to ground truth keys
        name_mapping = {
            "iPhone": "iPhone",
            "Mac": "Mac",
            "iPad": "iPad",
            "Wearables/Home/Accessories": "Wearables",
            "Wearables, Home and Accessories": "Wearables",
            "Services": "Services"
        }
        
        for ext_name, ext_seg in extracted.items():
            gt_key = name_mapping.get(ext_name)
            
            if not gt_key or gt_key not in gt_revenue:
                results["errors"].append(f"Unknown segment: {ext_name}")
                continue
            
            gt = gt_revenue[gt_key]
            
            # Check FY2025 revenue (allow 0.1% tolerance for rounding)
            if ext_seg.revenue_current:
                diff_pct = abs(ext_seg.revenue_current - gt["fy2025"]) / gt["fy2025"] * 100
                
                if diff_pct < 0.1:  # Within 0.1%
                    results["correct_segments"] += 1
                else:
                    results["errors"].append(
                        f"{ext_name}: Expected ${gt['fy2025']}M, got ${ext_seg.revenue_current}M (diff: {diff_pct:.2f}%)"
                    )
            else:
                results["errors"].append(f"{ext_name}: No current revenue extracted")
        
        results["accuracy_percentage"] = (results["correct_segments"] / results["total_segments"]) * 100
        
        return results
    
    def evaluate_risks(self, extraction: SECFilingExtraction) -> Dict[str, Any]:
        """
        Check if key risk categories were identified.
        """
        gt_risks = set(r.lower() for r in self.ground_truth["key_risks"])
        extracted_risks = set(r.category.lower() for r in extraction.risks.risks)
        
        # Fuzzy matching - check if ground truth concepts appear in extracted categories
        matches = 0
        for gt_risk in gt_risks:
            for ext_risk in extracted_risks:
                if gt_risk in ext_risk or any(word in ext_risk for word in gt_risk.split()):
                    matches += 1
                    break
        
        return {
            "total_key_risks": len(gt_risks),
            "identified_risks": matches,
            "coverage_percentage": (matches / len(gt_risks)) * 100,
            "total_extracted": len(extracted_risks)
        }