from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class RiskCategory(BaseModel):
    category: str = Field(description="Type of risk")
    description: str = Field(description="Brief description")
    severity: str = Field(description="High, Medium, or Low")

class RiskFactors(BaseModel):
    risks: list[RiskCategory]

class RevenueSegment(BaseModel):
    segment_name: str = Field(description="Product or service category (e.g., iPhone, Services)")
    revenue_current: Optional[float] = Field(None, description="Revenue in millions for current year")
    revenue_prior: Optional[float] = Field(None, description="Revenue in millions for prior year")
    percentage_change: Optional[float] = Field(None, description="Year-over-year percentage change")

class RevenueBreakdown(BaseModel):
    segments: list[RevenueSegment]
    total_revenue: Optional[float] = Field(None, description="Total net sales in millions")

class FinancialMetric(BaseModel):
    metric_name: str = Field(description="e.g., 'Net Income', 'EPS', 'Operating Margin'")
    value: str = Field(description="Value with units (e.g., '$93.7B', '15.3%')")
    fiscal_year: str = Field(description="e.g., '2025'")

class FinancialMetrics(BaseModel):
    metrics: list[FinancialMetric]

class SECFilingExtraction(BaseModel):
    """Complete extraction from a 10-K filing"""
    company: str
    filing_type: str
    fiscal_year: str
    extraction_timestamp: str
    risks: RiskFactors
    revenue: RevenueBreakdown
    financials: FinancialMetrics
    metadata: dict = Field(default_factory=dict, description="Token counts, costs, model used")