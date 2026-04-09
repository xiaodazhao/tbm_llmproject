# sketch_parser.py
import re
import json
from pathlib import Path
from typing import List

from schemas.schemas import EvidenceRecord
from utils.utils import mileage_to_num

def parse_sketch_pdf(pdf_path: Path) -> List[EvidenceRecord]:
    import fitz

    doc = fitz.open(pdf_path)
    text = "\n".join([p.get_text() for p in doc])

    mileage = re.search(r"DyK\d+\+\d+\.?\d*", text)
    if not mileage:
        return []

    chainage = mileage_to_num(mileage.group())

    attrs = {
        "rock_grade": "Ⅴ" if "Ⅴ" in text else None,
        "water_flag": int("出水" in text),
        "collapse_flag": int("掉块" in text),
        "risk_level": "high"
    }

    return [
        EvidenceRecord(
            evidence_id=pdf_path.stem,
            source_type="sketch",
            source_level="point",
            report_id=pdf_path.stem,
            report_date=None,
            issue_date=None,
            tunnel_name=None,
            start_num=chainage,
            end_num=chainage,
            face_num=chainage,
            next_forecast_num=None,
            confidence="high",
            attrs_json=json.dumps(attrs, ensure_ascii=False),
            raw_text=text
        )
    ]