import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class AuditLogger:
    """
    Logs all extraction operations for compliance and auditability.
    In production, this would write to a database or centralized logging system.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"audit_{self.session_id}.jsonl"
    
    def log_extraction(
        self,
        operation: str,
        document_id: str,
        input_hash: str,
        output_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ):
        """
        Log an extraction operation.
        
        Args:
            operation: Type of extraction (e.g., 'extract_risks', 'extract_revenue')
            document_id: Unique identifier for source document
            input_hash: SHA256 hash of input text (for data lineage)
            output_data: Extracted structured data
            metadata: Model info, token counts, timestamps, etc.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "operation": operation,
            "document_id": document_id,
            "input_hash": input_hash,
            "output_data": output_data,
            "metadata": metadata
        }
        
        # Append to JSONL file (one JSON object per line)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_validation(
        self,
        validation_type: str,
        document_id: str,
        passed: bool,
        details: Dict[str, Any]
    ):
        """Log validation checks (PII detection, confidence scores, etc.)"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "type": "validation",
            "validation_type": validation_type,
            "document_id": document_id,
            "passed": passed,
            "details": details
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    @staticmethod
    def hash_text(text: str) -> str:
        """Create SHA256 hash of text for data lineage tracking"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session's activity"""
        if not self.log_file.exists():
            return {"total_operations": 0}
        
        operations = []
        validations = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("type") == "validation":
                    validations.append(entry)
                else:
                    operations.append(entry)
        
        return {
            "session_id": self.session_id,
            "total_operations": len(operations),
            "total_validations": len(validations),
            "validations_passed": sum(1 for v in validations if v["passed"]),
            "log_file": str(self.log_file)
        }