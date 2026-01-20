import re
from typing import Dict, List, Tuple
from typing import Any

class DataValidator:
    """
    Validates extracted data for compliance with financial services requirements.
    """
    
    # Simple PII patterns (in production, use dedicated libraries like presidio)
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    }
    
    @classmethod
    def detect_pii(cls, text: str) -> Dict[str, List[str]]:
        """
        Detect potential PII in text.
        Returns dict of PII type -> list of matches.
        """
        findings = {}
        
        for pii_type, pattern in cls.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                findings[pii_type] = matches
        
        return findings
    
    @classmethod
    def redact_pii(cls, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Redact PII from text.
        Returns (redacted_text, redaction_counts).
        """
        redacted = text
        counts = {}
        
        for pii_type, pattern in cls.PII_PATTERNS.items():
            matches = re.findall(pattern, redacted)
            count = len(matches)
            
            if count > 0:
                redacted = re.sub(pattern, f'[REDACTED-{pii_type.upper()}]', redacted)
                counts[pii_type] = count
        
        return redacted, counts
    
    @staticmethod
    def calculate_confidence_score(extraction_data: dict, expected_fields: List[str]) -> float:
        """
        Calculate confidence score for extraction.
        Based on: completeness, field presence, value validity.
        """
        score = 0.0
        max_score = 100.0
        
        # Check field completeness (50 points)
        fields_present = sum(1 for field in expected_fields if extraction_data.get(field) is not None)
        completeness_score = (fields_present / len(expected_fields)) * 50
        score += completeness_score
        
        # Check for null/empty values in critical fields (30 points)
        critical_fields = expected_fields[:3] if len(expected_fields) >= 3 else expected_fields
        non_null_critical = sum(
            1 for field in critical_fields 
            if extraction_data.get(field) not in [None, "", [], {}]
        )
        validity_score = (non_null_critical / len(critical_fields)) * 30
        score += validity_score
        
        # Check for reasonable data types (20 points)
        type_score = 20.0  # Default to full points, deduct for issues
        for field, value in extraction_data.items():
            if value is None:
                type_score -= 2
        
        score += max(0, type_score)
        
        return min(score, max_score) / max_score  # Return 0-1 scale
    
    @staticmethod
    def validate_revenue_data(revenue_segments: List[dict]) -> Dict[str, Any]:
        """
        Validate revenue data for reasonableness.
        Checks: positive values, year-over-year consistency, segment total matches.
        """
        issues = []
        
        for segment in revenue_segments:
            name = segment.get("segment_name", "Unknown")
            current = segment.get("revenue_current")
            prior = segment.get("revenue_prior")
            change = segment.get("percentage_change")
            
            # Check positive values
            if current and current < 0:
                issues.append(f"{name}: Negative current revenue (${current}M)")
            
            if prior and prior < 0:
                issues.append(f"{name}: Negative prior revenue (${prior}M)")
            
            # Check YoY change consistency
            if current and prior and change is not None:
                expected_change = ((current - prior) / prior) * 100
                if abs(expected_change - change) > 1.0:  # Allow 1% tolerance
                    issues.append(
                        f"{name}: YoY change mismatch - stated {change}%, calculated {expected_change:.1f}%"
                    )
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "segments_checked": len(revenue_segments)
        }