from bs4 import BeautifulSoup
from pathlib import Path
import re

class SECFilingParser:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.soup = None
        self.text = None
        
    def load(self):
        """Load and parse HTML filing"""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(self.filepath, 'r', encoding=encoding) as f:
                    html_content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not decode file")
        
        self.soup = BeautifulSoup(html_content, 'html.parser')
        self.text = self.soup.get_text(separator=' ', strip=True)
        return self
    
    def extract_section(self, section_name: str) -> str:
        """Extract specific sections from 10-K"""
        text = self.text
        
        if section_name.lower() == 'risk factors':
            return self._extract_risk_factors()
        elif section_name.lower() == 'financial performance':
            return self._extract_financial_performance()
        else:
            return ""
    
    def _extract_risk_factors(self) -> str:
        """Extract Item 1A - Risk Factors"""
        text = self.text
        pattern = r'Item\s*1A\.\s*Risk\s*Factors\s+The\s+following'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if not match:
            pattern = r'Item\s*1A\.\s*Risk\s*Factors\s+\w{20,}'
            match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            start_idx = match.start()
            end_patterns = [r'Item\s*1B', r'Item\s*2\.\s*Properties']
            end_idx = None
            
            for end_pattern in end_patterns:
                end_match = re.search(end_pattern, text[start_idx + 100:], re.IGNORECASE)
                if end_match:
                    end_idx = start_idx + 100 + end_match.start()
                    break
            
            if end_idx:
                return text[start_idx:end_idx].strip()
            else:
                return text[start_idx:start_idx + 70000].strip()
        
        return ""
    
    def _extract_financial_performance(self) -> str:
        """
        Extract financial data - target the actual revenue table.
        """
        text = self.text
        
        # Look for the specific table header that precedes Apple's revenue breakdown
        # "The following table shows net sales by category"
        pattern = r'following\s+table\s+shows\s+net\s+sales\s+by\s+category'
        
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            start = match.start()
            # Extract from 200 chars before (to get header) to 5000 chars after (full table + context)
            section = text[max(0, start - 200):start + 5000]
            return section
        
        # Fallback: look for product names with dollar amounts in sequence
        pattern2 = r'iPhone\s+\$\s*[\d,]+.*?Mac\s+\$\s*[\d,]+.*?iPad\s+\$\s*[\d,]+'
        match2 = re.search(pattern2, text, re.IGNORECASE | re.DOTALL)
        
        if match2:
            start = match2.start()
            section = text[max(0, start - 500):start + 3000]
            return section
        
        return ""