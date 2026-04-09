# drill_parser.py   
import re
import json
from pathlib import Path
from typing import List

from schemas.schemas import EvidenceRecord
from utils.utils import mileage_to_num

def parse_drill_pdf(pdf_path: Path) -> List[EvidenceRecord]:
    import fitz

    doc = fitz.open(pdf_path)
    text = "\n".join([p.get_text() for p in doc])

    mileage_match = re.search(r"DyK\d+\+\d+\.?\d*", text)
    if not mileage_match:
        return []

    start = mileage_to_num(mileage_match.group())

    # 提取进尺
    lengths = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)m", text)]
    lengths = sorted(set(lengths))

    records = []
    last = 0

    for i, l in enumerate(lengths):
        seg_start = start + last
        seg_end = start + l
        last = l

        segment_text = text  # 简化版本（你后面可以细分）

        attrs = {
            "water_flag": int("出水" in segment_text),
            "collapse_flag": int("卡钻" in segment_text or "破碎" in segment_text),
            "risk_level": "high" if "出水" in segment_text else "medium"
        }

        records.append(
            EvidenceRecord(
                evidence_id=f"{pdf_path.stem}_{i}",
                source_type="drill",
                source_level="segment",
                report_id=pdf_path.stem,
                report_date=None,
                issue_date=None,
                tunnel_name=None,
                start_num=seg_start,
                end_num=seg_end,
                face_num=start,
                next_forecast_num=None,
                confidence="high",
                attrs_json=json.dumps(attrs, ensure_ascii=False),
                raw_text=segment_text
            )
        )

    return records