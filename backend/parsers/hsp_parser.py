# hsp_parser.py
import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import pdfplumber
import fitz

from schemas.schemas import EvidenceRecord
from utils.utils import mileage_to_num


def _norm_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u3000", " ")
    text = text.replace("～", "~")
    text = text.replace("—", "-").replace("−", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _flat_text(text: str) -> str:
    text = _norm_text(text)
    text = text.replace("\n", "")
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _safe_search(pattern: str, text: str, flags=re.S) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _safe_mileage(x: Optional[str]) -> Optional[float]:
    if not x:
        return None
    try:
        return float(mileage_to_num(x))
    except Exception:
        return None


def _extract_meta_from_text(text: str, pdf_name: str) -> Dict[str, Any]:
    t = _norm_text(text)

    report_id = pdf_name.replace(".pdf", "")
    tunnel_name = _safe_search(r"(伯舒拉岭隧道进口右线)", t) or "伯舒拉岭隧道进口右线"

    report_date = (
        _safe_search(r"检测日期[:：]?\s*([0-9]{4}\s*年\s*[0-9]{1,2}\s*月\s*[0-9]{1,2}\s*日)", t)
        or _safe_search(r"(二〇[^\n]+)", t)
    )

    forecast = _safe_search(r"预报范围\s*(DyK\d+\+\d+\.?\d*\s*~\s*DyK\d+\+\d+\.?\d*)", t)
    face = _safe_search(r"开挖面里程\s*(DyK\d+\+\d+\.?\d*)", t)
    next_ = _safe_search(r"下次物探预报里程为\s*(DyK\d+\+\d+\.?\d*)", t)

    start_num, end_num = None, None
    if forecast:
        m = re.search(r"(DyK\d+\+\d+\.?\d*)\s*~\s*(DyK\d+\+\d+\.?\d*)", forecast)
        if m:
            start_num = _safe_mileage(m.group(1))
            end_num = _safe_mileage(m.group(2))

    return {
        "report_id": report_id,
        "tunnel_name": tunnel_name,
        "report_date": report_date,
        "issue_date": report_date,
        "forecast_start_num": start_num,
        "forecast_end_num": end_num,
        "face_num": _safe_mileage(face),
        "next_forecast_num": _safe_mileage(next_),
    }


def _parse_range_cell(text: str):
    if not text:
        return None

    flat = _flat_text(text)
    miles = re.findall(r"DyK\d+\+\d+\.?\d*", flat)
    if len(miles) < 2:
        return None

    return {
        "start_text": miles[0],
        "end_text": miles[1],
        "start_num": _safe_mileage(miles[0]),
        "end_num": _safe_mileage(miles[1]),
    }


def _infer_anomaly_level(text: str) -> str:
    flat = _flat_text(text)

    # 顺序非常关键：先判断“未见”
    if "未见明显反射异常" in flat:
        return "none"
    if "较明显反射异常" in flat:
        return "medium"
    if "明显反射异常" in flat:
        return "strong"
    return "none"


def _extract_support_grade(text: str) -> Optional[str]:
    flat = _flat_text(text)
    m = re.search(r"([ⅠⅡⅢⅣⅤIVX]+)级围岩", flat)
    if m:
        return m.group(1)
    return None


def _extract_joint_degree(text: str) -> Optional[str]:
    flat = _flat_text(text)
    if "节理裂隙发育密集" in flat:
        return "发育密集"
    if "节理裂隙较发育" in flat:
        return "较发育"
    if "节理裂隙发育" in flat:
        return "发育"
    return None


def _extract_rock_mass_state(text: str) -> Optional[str]:
    flat = _flat_text(text)
    if "岩体破碎-极破碎" in flat or "岩体破碎极破碎" in flat:
        return "破碎-极破碎"
    if "岩体极破碎" in flat:
        return "极破碎"
    if "岩体较破碎" in flat or "岩体相对破碎" in flat:
        return "较破碎"
    if "岩体破碎" in flat:
        return "破碎"
    return None


def _extract_weathering(text: str) -> Optional[str]:
    flat = _flat_text(text)
    for x in ["全风化", "强风化", "弱风化", "微风化", "未风化"]:
        if x in flat:
            return x
    return None


def _extract_rock_uniformity(text: str) -> Optional[str]:
    flat = _flat_text(text)
    if "软硬不均" in flat:
        return "软硬不均"
    return None


def _extract_stability(text: str) -> Optional[str]:
    flat = _flat_text(text)
    if "围岩整体稳定性差" in flat or "围岩整体稳定性较差" in flat or "围岩自稳性差" in flat:
        return "较差"
    if "围岩整体稳定性一般" in flat:
        return "一般"
    return None


def _extract_lithology(text: str) -> Optional[str]:
    flat = _flat_text(text)
    if "板岩夹变质砂岩" in flat:
        return "板岩夹变质砂岩"
    return None


def _extract_collapse_info(risk_hint: str, conclusion: str):
    flat_hint = _flat_text(risk_hint or "")
    flat_conc = _flat_text(conclusion or "")
    flat = flat_hint + flat_conc

    collapse_flag = 1 if "掉块" in flat else 0

    collapse_points = []
    # 只从风险提示列抽里程点，不从range列误抽
    pts = re.findall(r"\+(\d+\.?\d*)", flat_hint)
    for p in pts:
        try:
            collapse_points.append(float(p))
        except Exception:
            pass

    return collapse_flag, collapse_points


def _infer_risk_level(
    anomaly_level: str,
    collapse_flag: int,
    support_grade: Optional[str],
    rock_mass_state: Optional[str]
) -> str:
    if anomaly_level == "strong" or collapse_flag == 1 or rock_mass_state in {"破碎-极破碎", "极破碎"}:
        return "high"
    if anomaly_level == "medium" or support_grade == "Ⅴ" or rock_mass_state in {"破碎", "较破碎"}:
        return "medium"
    return "low"


def _build_risk_tags(
    anomaly_level: str,
    collapse_flag: int,
    joint_degree: Optional[str],
    rock_mass_state: Optional[str],
    support_grade: Optional[str]
):
    tags = []

    if anomaly_level == "strong":
        tags.append("明显反射异常")
    elif anomaly_level == "medium":
        tags.append("较明显反射异常")

    if collapse_flag:
        tags.append("掉块")

    if joint_degree == "发育密集":
        tags.append("裂隙密集")
    elif joint_degree in {"发育", "较发育"}:
        tags.append("裂隙发育")

    if rock_mass_state in {"破碎-极破碎", "极破碎"}:
        tags.append("围岩极破碎")
    elif rock_mass_state in {"破碎", "较破碎"}:
        tags.append("围岩破碎")

    if support_grade:
        tags.append("围岩等级建议")

    out = []
    seen = set()
    for t in tags:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _score_range_cell(text: str) -> int:
    flat = _flat_text(text)
    if re.search(r"DyK\d+\+\d+\.?\d*", flat) and "~" in flat:
        return 3
    return 0


def _score_detect_cell(text: str) -> int:
    flat = _flat_text(text)
    keys = ["未见明显反射异常", "较明显反射异常", "明显反射异常", "反射异常"]
    return sum(1 for k in keys if k in flat)


def _score_conclusion_cell(text: str) -> int:
    flat = _flat_text(text)
    keys = ["围岩", "岩性", "弱风化", "软硬不均", "裂隙", "岩体", "稳定性", "变差", "变好", "掌子面相当"]
    return sum(1 for k in keys if k in flat)


def _score_risk_hint_cell(text: str) -> int:
    flat = _flat_text(text)
    keys = ["掉块风险", "附近有掉块", "风险提示"]
    return sum(1 for k in keys if k in flat)


def _score_grade_cell(text: str) -> int:
    flat = _flat_text(text)
    if re.search(r"[ⅠⅡⅢⅣⅤIVX]+级围岩", flat):
        return 3
    return 0


def _pick_cells_from_row(row: List[str]) -> Dict[str, str]:
    cells = [("" if c is None else str(c).strip()) for c in row]
    non_empty = [c for c in cells if c]

    if not non_empty:
        return {"range": "", "detect": "", "conclusion": "", "risk_hint": "", "grade": ""}

    range_cell = max(non_empty, key=_score_range_cell, default="")
    detect_cell = max(non_empty, key=_score_detect_cell, default="")
    conclusion_cell = max(non_empty, key=_score_conclusion_cell, default="")
    risk_hint_cell = max(non_empty, key=_score_risk_hint_cell, default="")
    grade_cell = max(non_empty, key=_score_grade_cell, default="")

    return {
        "range": range_cell,
        "detect": detect_cell,
        "conclusion": conclusion_cell,
        "risk_hint": risk_hint_cell,
        "grade": grade_cell,
    }


def _is_valid_hsp_row(picked: Dict[str, str]) -> bool:
    range_text = picked.get("range", "") or ""
    detect_text = picked.get("detect", "") or ""
    conclusion_text = picked.get("conclusion", "") or ""
    grade_text = picked.get("grade", "") or ""

    # 必须有里程段
    if not range_text:
        return False

    flat_range = _flat_text(range_text)
    if "~" not in flat_range:
        return False

    miles = re.findall(r"DyK\d+\+\d+\.?\d*", flat_range)
    if len(miles) < 2:
        return False

    # 排除“预报范围”这种假行
    flat_detect = _flat_text(detect_text)
    flat_conc = _flat_text(conclusion_text)
    flat_grade = _flat_text(grade_text)

    if "预报范围" in flat_range or "预报范围" in flat_detect or "预报范围" in flat_conc:
        return False

    # 必须至少包含 检测结果 / 结论 / 等级 其中之一
    useful = False
    if any(x in flat_detect for x in ["未见明显反射异常", "较明显反射异常", "明显反射异常", "反射异常"]):
        useful = True
    if "围岩" in flat_conc or "岩性" in flat_conc:
        useful = True
    if re.search(r"[ⅠⅡⅢⅣⅤIVX]+级围岩", flat_grade):
        useful = True

    return useful


def _parse_hsp_row_to_record(row_data: Dict[str, str], meta: Dict[str, Any], idx: int) -> Optional[EvidenceRecord]:
    range_info = _parse_range_cell(row_data.get("range", ""))
    if not range_info:
        return None

    detect_text = row_data.get("detect", "")
    conclusion_text = row_data.get("conclusion", "")
    risk_hint_text = row_data.get("risk_hint", "")
    grade_text = row_data.get("grade", "")

    anomaly_level = _infer_anomaly_level(detect_text)
    support_grade = _extract_support_grade(grade_text or conclusion_text)
    joint_degree = _extract_joint_degree(conclusion_text)
    rock_mass_state = _extract_rock_mass_state(conclusion_text)
    weathering = _extract_weathering(conclusion_text)
    rock_uniformity = _extract_rock_uniformity(conclusion_text)
    stability = _extract_stability(conclusion_text)
    lithology = _extract_lithology(conclusion_text)
    collapse_flag, collapse_points = _extract_collapse_info(risk_hint_text, conclusion_text)

    risk_level = _infer_risk_level(
        anomaly_level=anomaly_level,
        collapse_flag=collapse_flag,
        support_grade=support_grade,
        rock_mass_state=rock_mass_state,
    )

    risk_tags = _build_risk_tags(
        anomaly_level=anomaly_level,
        collapse_flag=collapse_flag,
        joint_degree=joint_degree,
        rock_mass_state=rock_mass_state,
        support_grade=support_grade,
    )

    attrs = {
        "fact_type": "hsp_segment",
        "lithology": lithology,
        "weathering": weathering,
        "support_grade": support_grade,
        "rock_hardness": None,
        "rock_uniformity": rock_uniformity,
        "joint_degree": joint_degree,
        "rock_mass_state": rock_mass_state,
        "mud_filling_flag": 0,
        "stability": stability,
        "water_flag": 0,
        "water_type": None,
        "collapse_flag": collapse_flag,
        "deformation_flag": 0,
        "anomaly_level": anomaly_level,
        "risk_hint_text": risk_hint_text or None,
        "collapse_points": collapse_points,
        "risk_level": risk_level,
        "risk_tags": risk_tags,
    }

    raw_text = "\n".join([
        f"里程范围: {row_data.get('range', '')}",
        f"物探探测结果: {detect_text}",
        f"预报结论: {conclusion_text}",
        f"风险提示: {risk_hint_text}",
        f"建议围岩等级: {grade_text}",
    ]).strip()

    return EvidenceRecord(
        evidence_id=f"{meta['report_id']}_hsp_{idx}",
        source_type="sonic",
        source_level="segment",
        report_id=meta["report_id"],
        report_date=meta["report_date"],
        issue_date=meta["issue_date"],
        tunnel_name=meta["tunnel_name"],
        start_num=range_info["start_num"],
        end_num=range_info["end_num"],
        face_num=meta["face_num"],
        next_forecast_num=meta["next_forecast_num"],
        confidence="medium",
        attrs_json=json.dumps(attrs, ensure_ascii=False),
        raw_text=raw_text
    )


def parse_hsp_pdf(pdf_path: Path) -> List[EvidenceRecord]:
    doc = fitz.open(pdf_path)
    try:
        text = "\n".join([p.get_text() for p in doc])
    finally:
        doc.close()

    meta = _extract_meta_from_text(text, pdf_path.name)

    records: List[EvidenceRecord] = []

    with pdfplumber.open(pdf_path) as pdf:
        idx = 0
        for page in pdf.pages:
            page_text = page.extract_text() or ""

            # 只看表1那页
            if ("表 1" not in page_text and "表1" not in page_text) or "隧道超前地质预报报表" not in page_text:
                continue

            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                table_text = str(table)

                if "里程范围" not in table_text and "预报结论" not in table_text:
                    continue

                for row in table:
                    if not row:
                        continue

                    row = [("" if c is None else str(c)) for c in row]
                    joined = "".join(row)

                    if "里程范围" in joined and "预报结论" in joined:
                        continue
                    if "下一次超前预报里程" in joined:
                        continue
                    if "备注" in joined:
                        continue

                    picked = _pick_cells_from_row(row)
                    if not _is_valid_hsp_row(picked):
                        continue

                    rec = _parse_hsp_row_to_record(picked, meta, idx)
                    if rec is not None:
                        records.append(rec)
                        idx += 1

    return records