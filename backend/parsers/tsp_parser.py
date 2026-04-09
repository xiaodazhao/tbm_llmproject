# tsp_parser.py
import re
import json
from pathlib import Path
from typing import List, Optional, Dict

from schemas.schemas import EvidenceRecord
from utils.utils import mileage_to_num, safe_float, compact_text


def extract_pdf_text(pdf_path: Path):
    import fitz
    doc = fitz.open(pdf_path)
    return "\n".join([p.get_text() for p in doc])


def extract_meta(text: str, name: str):
    def g(p):
        m = re.search(p, text, re.S)
        return m.group(1).strip() if m else None

    forecast = g(r"预报范围[:：]?\s*(DyK[^\n]+)")
    face = g(r"掌子面里程[:：]?\s*(DyK\d+\+\d+\.?\d*)")
    next_ = g(r"下次.*?里程.*?(DyK\d+\+\d+\.?\d*)")

    start, end = None, None
    if forecast:
        m = re.search(r"(DyK\d+\+\d+\.?\d*)\s*~\s*(DyK\d+\+\d+\.?\d*)", forecast)
        if m:
            start = mileage_to_num(m.group(1))
            end = mileage_to_num(m.group(2))

    return dict(
        report_id=name.replace(".pdf",""),
        start=start,
        end=end,
        face=mileage_to_num(face),
        next=mileage_to_num(next_)
    )


def split_blocks(text: str):
    pattern = r"(DyK\d+\+\d+\.?\d*\s*~\s*DyK\d+\+\d+\.?\d*)"
    idx = [m.start() for m in re.finditer(pattern, text)]
    blocks = []
    for i in range(len(idx)):
        s = idx[i]
        e = idx[i+1] if i+1<len(idx) else len(text)
        blocks.append(text[s:e])
    return blocks


def parse_block(block: str, meta, i):
    m = re.search(r"(DyK\d+\+\d+\.?\d*)\s*~\s*(DyK\d+\+\d+\.?\d*)", block)
    if not m:
        return None

    start = mileage_to_num(m.group(1))
    end = mileage_to_num(m.group(2))

    attrs = {
        "rock_grade": None,
        "water_flag": int("出水" in block),
        "collapse_flag": int("掉块" in block),
        "risk_level": None
    }

    if "Ⅴ" in block:
        attrs["rock_grade"] = "Ⅴ"

    if attrs["water_flag"] or attrs["collapse_flag"]:
        attrs["risk_level"] = "high"
    else:
        attrs["risk_level"] = "medium"

    return EvidenceRecord(
        evidence_id=f"{meta['report_id']}_{i}",
        source_type="tsp",
        source_level="segment",
        report_id=meta["report_id"],
        report_date=None,
        issue_date=None,
        tunnel_name=None,
        start_num=start,
        end_num=end,
        face_num=meta["face"],
        next_forecast_num=meta["next"],
        confidence="medium",
        attrs_json=json.dumps(attrs, ensure_ascii=False),
        raw_text=block
    )


def parse_tsp_pdf(pdf_path: Path) -> List[EvidenceRecord]:
    text = extract_pdf_text(pdf_path)
    meta = extract_meta(text, pdf_path.name)
    blocks = split_blocks(text)

    records = []
    for i, b in enumerate(blocks):
        r = parse_block(b, meta, i)
        if r:
            records.append(r)

    return records