# schemas.py
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class EvidenceRecord:
    evidence_id: str
    source_type: str
    source_level: str
    report_id: str
    report_date: Optional[str]
    issue_date: Optional[str]
    tunnel_name: Optional[str]
    start_num: float
    end_num: float
    face_num: Optional[float]
    next_forecast_num: Optional[float]
    confidence: str
    attrs_json: str
    raw_text: Optional[str] = None

    def attrs(self) -> Dict[str, Any]:
        try:
            return json.loads(self.attrs_json)
        except:
            return {}