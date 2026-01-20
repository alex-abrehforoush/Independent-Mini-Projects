from src.audit_logger import AuditLogger
from src.validators import DataValidator
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from typing import Any
from src.schemas import (
    RiskFactors, RevenueBreakdown, FinancialMetrics,
    SECFilingExtraction
)
from datetime import datetime

load_dotenv()

class SECExtractor:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0,
        )
        self.api_calls = []
        self.total_tokens = 0
        self.audit_logger = AuditLogger()
        self.validator = DataValidator()
    
    def _track_usage(self, operation: str, input_text: str, response: Any):
        """Estimate token usage (rough approximation: 1 token ≈ 4 chars)"""
        input_tokens = len(input_text) // 4
        # Response tokens are harder to estimate, use conservative multiplier
        output_tokens = len(str(response)) // 4
        
        total = input_tokens + output_tokens
        self.total_tokens += total
        
        self.api_calls.append({
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total
        })
        
        return total
    
    def extract_risks(self, text: str) -> RiskFactors:
        """Extract structured risk factors"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst extracting risk factors from SEC filings.
            Categorize each risk and assess severity based on potential business impact.
            Focus on distinct risk categories, not individual sentences."""),
            ("user", "Extract and categorize the main risk factors:\n\n{text}")
        ])
        
        structured_llm = self.llm.with_structured_output(RiskFactors)
        chain = prompt | structured_llm
        
        truncated = text[:30000]
        result = chain.invoke({"text": truncated})
        
        self._track_usage("extract_risks", truncated, result)
        
        return result
    
    def extract_revenue(self, text: str) -> RevenueBreakdown:
        """Extract revenue breakdown by segment"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst extracting revenue data from 10-K filings.
            Extract the EXACT numbers from the revenue table for each product segment.
            Apple segments: iPhone, Mac, iPad, Wearables/Home/Accessories, Services.
            
            CRITICAL: Only extract numbers that are explicitly stated in the text.
            Do not estimate, interpolate, or make up numbers.
            Revenue should be in millions of dollars (the tables say "in millions").
            
            If a number is not clearly stated, set that field to null."""),
            ("user", "Extract the exact revenue numbers from this table:\n\n{text}")
        ])
        
        structured_llm = self.llm.with_structured_output(RevenueBreakdown)
        chain = prompt | structured_llm
        
        result = chain.invoke({"text": text[:8000]})
        
        self._track_usage("extract_revenue", text[:8000], result)
        
        return result
    
    def extract_financials(self, text: str) -> FinancialMetrics:
        """Extract key financial metrics"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst extracting key metrics from 10-K filings.
            Extract: Net Income, EPS (earnings per share), Operating Margin, Gross Margin, R&D spending.
            Include the fiscal year for each metric."""),
            ("user", "Extract key financial metrics:\n\n{text}")
        ])
        
        structured_llm = self.llm.with_structured_output(FinancialMetrics)
        chain = prompt | structured_llm
        
        result = chain.invoke({"text": text[:10000]})
        
        self._track_usage("extract_financials", text[:10000], result)
        
        return result
    
    def get_cost_summary(self) -> dict:
        """
        Calculate estimated API costs.
        GPT-4o-mini pricing (as of 2025):
        - Input: $0.150 per 1M tokens
        - Output: $0.600 per 1M tokens
        """
        total_input = sum(call["input_tokens"] for call in self.api_calls)
        total_output = sum(call["output_tokens"] for call in self.api_calls)
        
        input_cost = (total_input / 1_000_000) * 0.150
        output_cost = (total_output / 1_000_000) * 0.600
        total_cost = input_cost + output_cost
        
        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "estimated_cost_usd": round(total_cost, 4),
            "api_calls": len(self.api_calls),
            "breakdown": self.api_calls
        }
    
    def extract_with_governance(
        self,
        document_id: str,
        risk_text: str,
        financial_text: str
    ) -> dict:
        """
        Extract data with full governance controls:
        - PII detection
        - Audit logging
        - Confidence scoring
        - Data validation
        """
        results = {
            "extractions": {},
            "governance": {
                "pii_detected": {},
                "confidence_scores": {},
                "validations": {}
            }
        }
        
        # 1. PII Detection
        risk_pii = self.validator.detect_pii(risk_text)
        financial_pii = self.validator.detect_pii(financial_text)
        
        if risk_pii or financial_pii:
            results["governance"]["pii_detected"] = {
                "risk_section": risk_pii,
                "financial_section": financial_pii
            }
            # Log PII detection
            self.audit_logger.log_validation(
                "pii_detection",
                document_id,
                passed=(len(risk_pii) == 0 and len(financial_pii) == 0),
                details={"risk_pii": risk_pii, "financial_pii": financial_pii}
            )
        
        # 2. Extract data with audit trail
        print("  - Extracting risks (with audit trail)...")
        risks = self.extract_risks(risk_text)
        input_hash = AuditLogger.hash_text(risk_text)
        self.audit_logger.log_extraction(
            "extract_risks",
            document_id,
            input_hash,
            {"risk_count": len(risks.risks)},
            {"model": "gpt-4o-mini", "temperature": 0}
        )
        
        print("  - Extracting revenue (with audit trail)...")
        revenue = self.extract_revenue(financial_text)
        input_hash = AuditLogger.hash_text(financial_text)
        self.audit_logger.log_extraction(
            "extract_revenue",
            document_id,
            input_hash,
            {"segment_count": len(revenue.segments)},
            {"model": "gpt-4o-mini", "temperature": 0}
        )
        
        print("  - Extracting financials (with audit trail)...")
        financials = self.extract_financials(financial_text)
        self.audit_logger.log_extraction(
            "extract_financials",
            document_id,
            input_hash,
            {"metric_count": len(financials.metrics)},
            {"model": "gpt-4o-mini", "temperature": 0}
        )
        
        # 3. Confidence scoring
        revenue_confidence = self.validator.calculate_confidence_score(
            {seg.segment_name: seg.revenue_current for seg in revenue.segments},
            ["iPhone", "Mac", "iPad", "Wearables/Home/Accessories", "Services"]
        )
        
        results["governance"]["confidence_scores"] = {
            "revenue_extraction": revenue_confidence
        }
        
        # 4. Data validation
        revenue_validation = self.validator.validate_revenue_data(
            [seg.model_dump() for seg in revenue.segments]
        )
        results["governance"]["validations"]["revenue"] = revenue_validation
        
        self.audit_logger.log_validation(
            "revenue_validation",
            document_id,
            revenue_validation["valid"],
            revenue_validation
        )
        
        # 5. Store results
        results["extractions"] = {
            "risks": risks,
            "revenue": revenue,
            "financials": financials
        }
        
        return results