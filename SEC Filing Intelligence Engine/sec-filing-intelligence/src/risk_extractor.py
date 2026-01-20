from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import json

load_dotenv()

class RiskCategory(BaseModel):
    category: str = Field(description="Type of risk (e.g., Market Risk, Regulatory Risk, Operational Risk)")
    description: str = Field(description="Brief description of the risk")
    severity: str = Field(description="High, Medium, or Low")

class RiskFactors(BaseModel):
    risks: list[RiskCategory] = Field(description="List of identified risks")

class RiskExtractor:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0,
        )
    
    def extract(self, risk_section_text: str) -> RiskFactors:
        """Extract structured risk factors from text"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst extracting risk factors from SEC filings.
            Categorize each risk and assess severity based on:
            - High: Could significantly impact business operations or financials
            - Medium: Notable impact but manageable
            - Low: Minor or theoretical risk
            
            Focus on distinct risk categories, not individual sentences."""),
            ("user", "Extract and categorize the main risk factors from this text:\n\n{text}")
        ])
        
        # Use structured output
        structured_llm = self.llm.with_structured_output(RiskFactors)
        chain = prompt | structured_llm
        
        # Truncate if too long (gpt4o-mini has 128k context, but let's be safe)
        truncated_text = risk_section_text[:30000]
        
        result = chain.invoke({"text": truncated_text})
        return result