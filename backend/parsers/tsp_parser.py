# tsp_parser.py
import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import pdfplumber
import fitz  # PyMuPDF

from schemas.schemas import EvidenceRecord
from utils.utils import mileage_to_num


# =========================
# 1. PDF全文提取
# =========================
def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    try:
        pages = [p.get_text() for p in doc]
        return "\n".join(pages)
    finally:
        doc.close()


# =========================
# 2. 基础工具
# =========================
def _norm_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u3000", " ")
    text = text.replace("～", "~")
    text = text.replace("￾", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _flat_text(text: str) -> str:
    """
    用于跨行正则匹配
    """
    text = _norm_text(text)
    text = text.replace("\n", "")
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _safe_search(pattern: str, text: str, flags=re.S) -> Optional[str]:
    if not text:
        return None
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _safe_mileage(x: Optional[str]) -> Optional[float]:
    if not x:
        return None
    try:
        return float(mileage_to_num(x))
    except Exception:
        return None


def _first_match(candidates: List[str], text: str) -> Optional[str]:
    for c in candidates:
        if c in text:
            return c
    return None


def _extract_range_values(label: str, text: str) -> Dict[str, Optional[float]]:
    """
    提取：
    纵波速度Vp：4871m/s
    纵波速度Vp：4793~5312m/s
    """
    if not text:
        return {"min": None, "max": None}

    pattern = rf"{label}[:：]?\s*([0-9.]+(?:\s*~\s*[0-9.]+)?)"
    m = re.search(pattern, text, re.S)
    if not m:
        return {"min": None, "max": None}

    raw = re.sub(r"\s+", "", m.group(1))
    nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", raw)
    if not nums:
        return {"min": None, "max": None}

    vals = [float(x) for x in nums]
    if len(vals) == 1:
        return {"min": vals[0], "max": vals[0]}
    return {"min": min(vals), "max": max(vals)}


# =========================
# 3. 报告级元信息提取
# =========================
def extract_meta(text: str, pdf_name: str) -> Dict[str, Any]:
    t = _norm_text(text)

    report_id = pdf_name.replace(".pdf", "")
    tunnel_name = _safe_search(r"(伯舒拉岭隧道进口右线)", t)

    report_date = (
        _safe_search(r"检测日期[:：]?\s*([^\n]+)", t)
        or _safe_search(r"测试日期[:：]?\s*([^\n]+)", t)
        or _safe_search(r"(二〇[^\n]+)", t)
    )

    forecast = (
        _safe_search(r"预报范围\s*(DyK\d+\+\d+\.?\d*\s*~\s*DyK\d+\+\d+\.?\d*)", t)
        or _safe_search(r"\((DyK\d+\+\d+\.?\d*\s*~\s*DyK\d+\+\d+\.?\d*)\)", t)
    )

    face = (
        _safe_search(r"开挖面里程\s*(DyK\d+\+\d+\.?\d*)", t)
        or _safe_search(r"掌子面（(DyK\d+\+\d+\.?\d*)）", t)
    )

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


# =========================
# 4. 目标区间：从“7 结论”到“备注前”
# =========================
def extract_target_section(text: str) -> str:
    """
    起点：7 结论
    终点：备注：本报表掌子面里程由施工单位提供...
    """
    t = _norm_text(text)

    start_pat = r"7\s*结论"
    end_pat = r"备注[:：]?\s*本报表掌子面里程由施工单位提供"

    start_m = re.search(start_pat, t, re.S)
    if not start_m:
        return ""

    end_m = re.search(end_pat, t, re.S)
    if end_m:
        return t[start_m.start():end_m.start()].strip()

    fallback = re.search(r"附图|附表", t, re.S)
    end_idx = fallback.start() if fallback else len(t)
    return t[start_m.start():end_idx].strip()


def extract_conclusion_text_from_section(section_text: str) -> str:
    """
    从总截取块里提取：7结论 到 表2 之前
    """
    if not section_text:
        return ""

    m = re.search(r"7\s*结论(.*?)(?:表\s*2|表2)", section_text, re.S)
    if m:
        return m.group(1).strip()

    m2 = re.search(r"7\s*结论(.*)", section_text, re.S)
    return m2.group(1).strip() if m2 else ""


def extract_table2_text_from_section(section_text: str) -> str:
    """
    这里只用来判断 section 中是否存在表2
    """
    if not section_text:
        return ""

    m = re.search(
        r"(表\s*2[\s\S]*?)(?=备注[:：]|为满足隧道超前地质预报资料搭接的要求|附图|附表|$)",
        section_text
    )
    return m.group(1).strip() if m else ""





def _score_param_cell(text: str) -> int:
    if not text:
        return 0
    flat = re.sub(r"\s+", "", text)
    keys = ["纵波速度", "横波速度", "速度比", "泊松比", "动态杨氏模量"]
    return sum(1 for k in keys if k in flat)


def _score_conclusion_cell(text: str) -> int:
    if not text:
        return 0
    flat = re.sub(r"\s+", "", text)
    keys = [
        "地层岩性", "板岩夹变质砂岩", "弱风化", "岩质",
        "节理裂隙", "岩体", "围岩", "施工", "出水",
        "掉块", "泥质填充", "稳定性"
    ]
    return sum(1 for k in keys if k in flat)


def _pick_param_and_conclusion_cells(row: List[str]) -> Dict[str, str]:
    """
    从整行里自动挑选最像【物性参数】和【预报结论】的单元格
    """
    cells = [("" if c is None else str(c).strip()) for c in row]

    # 去掉空单元格
    non_empty = [c for c in cells if c]
    if not non_empty:
        return {"params": "", "conclusion": ""}

    # 参数列：命中参数关键词最多的那个
    param_cell = max(non_empty, key=_score_param_cell, default="")

    # 结论列：命中结论关键词最多的那个，且不要和参数列完全一样
    conclusion_candidates = [c for c in non_empty if c != param_cell]
    conclusion_cell = max(conclusion_candidates, key=_score_conclusion_cell, default="")

    # 如果没挑出来，兜底：取最长的非参数 cell
    if _score_conclusion_cell(conclusion_cell) == 0:
        conclusion_cell = max(conclusion_candidates, key=len, default="")

    return {
        "params": param_cell,
        "conclusion": conclusion_cell,
    }








# =========================
# 5. overview 提取
# =========================
def _parse_overview_record(text: str, meta: Dict[str, Any]) -> List[EvidenceRecord]:
    t = _norm_text(text)
    m = re.search(
        r"本次预报当前掌子面.*?(?=5\s*现场工作布置及数据采集|5\.1|6\s*资料处理与解释)",
        t,
        re.S
    )
    if not m:
        return []

    block = m.group(0)
    flat = _flat_text(block)

    attrs = {
        "fact_type": "overview",
        "lithology": "板岩夹变质砂岩" if "板岩夹变质砂岩" in flat else None,
        "weathering": "弱风化" if "弱风化" in flat else None,
        "support_grade": None,
        "rock_hardness": None,
        "rock_uniformity": None,
        "joint_degree": "发育" if "节理裂隙发育" in flat else None,
        "rock_mass_state": "破碎" if "岩体破碎" in flat else None,
        "mud_filling_flag": 0,
        "stability": "较差" if ("自稳性较差" in flat or "稳定性较差" in flat) else None,
        "water_flag": 1 if ("线状出水" in flat or "线-股状出水" in flat) else 0,
        "water_type": "线-股状出水" if "线-股状出水" in flat else ("线状出水" if "线状出水" in flat else None),
        "collapse_flag": 0,
        "vp_min": None,
        "vp_max": None,
        "vs_min": None,
        "vs_max": None,
        "vp_vs_min": None,
        "vp_vs_max": None,
        "poisson_min": None,
        "poisson_max": None,
        "dynamic_modulus_min": None,
        "dynamic_modulus_max": None,
        "risk_note_text": None,
        "risk_level": "medium",
        "risk_tags": [],
    }

    if attrs["water_flag"]:
        attrs["risk_tags"].append("出水")
    if attrs["rock_mass_state"] == "破碎":
        attrs["risk_tags"].append("围岩破碎")

    rec = EvidenceRecord(
        evidence_id=f"{meta['report_id']}_overview_0",
        source_type="tsp",
        source_level="overview",
        report_id=meta["report_id"],
        report_date=meta["report_date"],
        issue_date=meta["issue_date"],
        tunnel_name=meta["tunnel_name"],
        start_num=meta["forecast_start_num"],
        end_num=meta["forecast_end_num"],
        face_num=meta["face_num"],
        next_forecast_num=meta["next_forecast_num"],
        confidence="medium",
        attrs_json=json.dumps(attrs, ensure_ascii=False),
        raw_text=block
    )
    return [rec]


# =========================
# 6. 解析 7结论
# =========================
def _parse_grade_records(conclusion_text: str, meta: Dict[str, Any]) -> List[EvidenceRecord]:
    flat = _flat_text(conclusion_text)

    matches = re.findall(
        r"(DyK\d+\+\d+\.?\d*)~(DyK\d+\+\d+\.?\d*)段建议按([ⅠⅡⅢⅣⅤIVX]+)级围岩施工",
        flat
    )

    records = []
    for i, (s, e, grade) in enumerate(matches):
        attrs = {
            "fact_type": "support_grade_conclusion",
            "support_grade": grade,
            "water_flag": 0,
            "water_type": None,
            "collapse_flag": 0,
            "risk_note_text": None,
            "vp_min": None,
            "vp_max": None,
            "vs_min": None,
            "vs_max": None,
            "vp_vs_min": None,
            "vp_vs_max": None,
            "poisson_min": None,
            "poisson_max": None,
            "dynamic_modulus_min": None,
            "dynamic_modulus_max": None,
            "risk_level": "medium" if grade == "Ⅴ" else "low",
            "risk_tags": ["围岩等级建议"],
        }

        records.append(
            EvidenceRecord(
                evidence_id=f"{meta['report_id']}_grade_{i}",
                source_type="tsp",
                source_level="report_conclusion",
                report_id=meta["report_id"],
                report_date=meta["report_date"],
                issue_date=meta["issue_date"],
                tunnel_name=meta["tunnel_name"],
                start_num=_safe_mileage(s),
                end_num=_safe_mileage(e),
                face_num=meta["face_num"],
                next_forecast_num=meta["next_forecast_num"],
                confidence="high",
                attrs_json=json.dumps(attrs, ensure_ascii=False),
                raw_text=f"{s}~{e} 段建议按{grade}级围岩施工"
            )
        )

    return records


def _parse_collapse_records(conclusion_text: str, meta: Dict[str, Any]) -> List[EvidenceRecord]:
    flat = _flat_text(conclusion_text)

    matches = re.findall(
        r"(DyK\d+\+\d+\.?\d*)~(DyK\d+\+\d+\.?\d*)段([^。；]*?)(?=；|。|DyK\d+\+\d+|$)",
        flat
    )

    records = []
    idx = 0
    for s, e, desc in matches:
        if not any(k in desc for k in ["裂隙", "破碎", "泥质填充", "软硬不均", "掉块"]):
            continue

        if "软硬不均" in desc:
            rock_hardness = None
            rock_uniformity = "软硬不均"
        else:
            rock_hardness = _first_match(["岩质较硬", "岩质硬", "岩质较软", "岩质软"], desc)
            rock_uniformity = None

        attrs = {
            "fact_type": "collapse_risk_conclusion",
            "support_grade": None,
            "rock_hardness": rock_hardness,
            "rock_uniformity": rock_uniformity,
            "joint_degree": "发育密集" if "节理裂隙发育密集" in desc else ("发育" if "节理裂隙发育" in desc else None),
            "rock_mass_state": (
                "破碎-极破碎" if ("破碎-极破碎" in desc or "破碎极破碎" in desc) else
                "破碎" if "岩体破碎" in desc else None
            ),
            "mud_filling_flag": 1 if "泥质填充" in desc else 0,
            "stability": "较差" if "稳定性较差" in flat else None,
            "water_flag": 0,
            "water_type": None,
            "collapse_flag": 1 if ("掉块" in flat or "掉块" in desc) else 0,
            "risk_note_text": "存在掉块风险" if "掉块风险" in flat else None,
            "vp_min": None,
            "vp_max": None,
            "vs_min": None,
            "vs_max": None,
            "vp_vs_min": None,
            "vp_vs_max": None,
            "poisson_min": None,
            "poisson_max": None,
            "dynamic_modulus_min": None,
            "dynamic_modulus_max": None,
            "risk_level": "high",
            "risk_tags": [],
        }

        if attrs["collapse_flag"]:
            attrs["risk_tags"].append("掉块")
        if attrs["rock_mass_state"] in {"破碎-极破碎"}:
            attrs["risk_tags"].append("围岩极破碎")
        elif attrs["rock_mass_state"] == "破碎":
            attrs["risk_tags"].append("围岩破碎")
        if attrs["mud_filling_flag"]:
            attrs["risk_tags"].append("泥质填充")
        if attrs["joint_degree"] == "发育密集":
            attrs["risk_tags"].append("裂隙密集")

        records.append(
            EvidenceRecord(
                evidence_id=f"{meta['report_id']}_collapse_{idx}",
                source_type="tsp",
                source_level="report_conclusion",
                report_id=meta["report_id"],
                report_date=meta["report_date"],
                issue_date=meta["issue_date"],
                tunnel_name=meta["tunnel_name"],
                start_num=_safe_mileage(s),
                end_num=_safe_mileage(e),
                face_num=meta["face_num"],
                next_forecast_num=meta["next_forecast_num"],
                confidence="high",
                attrs_json=json.dumps(attrs, ensure_ascii=False),
                raw_text=f"{s}~{e} 段{desc}"
            )
        )
        idx += 1

    return records


def _parse_water_records(conclusion_text: str, meta: Dict[str, Any]) -> List[EvidenceRecord]:
    flat = _flat_text(conclusion_text)

    records = []
    grouped_matches = re.findall(
        r"((?:DyK\d+\+\d+\.?\d*~DyK\d+\+\d+\.?\d*[、，]?)+)段掌子面存在(线-股状出水|线状出水|股状出水)",
        flat
    )

    idx = 0
    for ranges_text, water_type in grouped_matches:
        pairs = re.findall(
            r"(DyK\d+\+\d+\.?\d*)~(DyK\d+\+\d+\.?\d*)",
            ranges_text
        )

        for s, e in pairs:
            attrs = {
                "fact_type": "water_risk_conclusion",
                "support_grade": None,
                "water_flag": 1,
                "water_type": water_type,
                "collapse_flag": 0,
                "risk_note_text": "存在出水风险" if "出水风险" in flat else None,
                "vp_min": None,
                "vp_max": None,
                "vs_min": None,
                "vs_max": None,
                "vp_vs_min": None,
                "vp_vs_max": None,
                "poisson_min": None,
                "poisson_max": None,
                "dynamic_modulus_min": None,
                "dynamic_modulus_max": None,
                "risk_level": "high" if water_type == "线-股状出水" else "medium",
                "risk_tags": ["出水"],
            }

            records.append(
                EvidenceRecord(
                    evidence_id=f"{meta['report_id']}_water_{idx}",
                    source_type="tsp",
                    source_level="report_conclusion",
                    report_id=meta["report_id"],
                    report_date=meta["report_date"],
                    issue_date=meta["issue_date"],
                    tunnel_name=meta["tunnel_name"],
                    start_num=_safe_mileage(s),
                    end_num=_safe_mileage(e),
                    face_num=meta["face_num"],
                    next_forecast_num=meta["next_forecast_num"],
                    confidence="high",
                    attrs_json=json.dumps(attrs, ensure_ascii=False),
                    raw_text=f"{s}~{e} 段掌子面存在{water_type}"
                )
            )
            idx += 1

    return records


# =========================
# 7. 表2：使用 pdfplumber 结构化解析
# =========================
def parse_table2_structured_row(
    row_data: Dict[str, str],
    meta: Dict[str, Any],
    idx: int
) -> Optional[EvidenceRecord]:
    mileage_text = row_data.get("mileage", "") or ""
    params_text = row_data.get("params", "") or ""
    conclusion_text = row_data.get("conclusion", "") or ""

    mileage_flat = re.sub(r"\s+", "", mileage_text)
    m = re.search(r"(DyK\d+\+\d+\.?\d*)(?:~|-)(DyK\d+\+\d+\.?\d*)", mileage_flat)
    if not m:
        return None

    s, e = m.group(1), m.group(2)

    vp = _extract_range_values(r"纵波速度\s*Vp", params_text)
    vs = _extract_range_values(r"横波速度\s*Vs", params_text)
    vp_vs = _extract_range_values(r"速度比\s*Vp/Vs", params_text)
    poisson = _extract_range_values(r"泊松比[^\d]*", params_text)
    modulus = _extract_range_values(r"动态杨氏模量\s*E", params_text)

    flat_conc = re.sub(r"\s+", "", conclusion_text)

    support_grade = _safe_search(r"按([ⅠⅡⅢⅣⅤIVX]+)级围岩施工", flat_conc, 0)

    rock_uniformity = "软硬不均" if "软硬不均" in flat_conc else None
    if rock_uniformity == "软硬不均":
        rock_hardness = None
    else:
        rock_hardness = _safe_search(r"(岩质较硬|岩质硬|岩质较软|岩质软)", flat_conc, 0)

    if "节理裂隙发育密集" in flat_conc:
        joint_degree = "发育密集"
    elif "节理裂隙较发育" in flat_conc:
        joint_degree = "较发育"
    elif "节理裂隙发育" in flat_conc:
        joint_degree = "发育"
    else:
        joint_degree = None

    if "岩体破碎-极破碎" in flat_conc or "岩体破碎极破碎" in flat_conc:
        rock_mass_state = "破碎-极破碎"
    elif "岩体较破碎" in flat_conc:
        rock_mass_state = "较破碎"
    elif "岩体极破碎" in flat_conc:
        rock_mass_state = "极破碎"
    elif "岩体破碎" in flat_conc:
        rock_mass_state = "破碎"
    else:
        rock_mass_state = None

    mud_filling_flag = 1 if "泥质填充" in flat_conc else 0

    if "围岩整体稳定性较差" in flat_conc:
        stability = "较差"
    elif "围岩整体稳定性一般" in flat_conc:
        stability = "一般"
    elif "围岩整体稳定性较好" in flat_conc:
        stability = "较好"
    else:
        stability = None

    if "线-股状出水" in flat_conc:
        water_type = "线-股状出水"
        water_flag = 1
    elif "线状出水" in flat_conc:
        water_type = "线状出水"
        water_flag = 1
    elif "股状出水" in flat_conc:
        water_type = "股状出水"
        water_flag = 1
    else:
        water_type = None
        water_flag = 0

    collapse_flag = 1 if "掉块风险" in flat_conc else 0

    notes = []
    if "掉块风险" in flat_conc:
        notes.append("存在掉块风险")
    if "出水风险" in flat_conc:
        notes.append("存在出水风险")
    risk_note_text = "；".join(notes) if notes else None

    risk_tags = []
    if water_flag:
        risk_tags.append("出水")
    if collapse_flag:
        risk_tags.append("掉块")
    if rock_mass_state in {"破碎-极破碎", "极破碎"}:
        risk_tags.append("围岩极破碎")
    elif rock_mass_state in {"破碎", "较破碎"}:
        risk_tags.append("围岩破碎")
    if mud_filling_flag:
        risk_tags.append("泥质填充")
    if joint_degree == "发育密集":
        risk_tags.append("裂隙密集")

    if collapse_flag == 1 or water_type == "线-股状出水" or rock_mass_state in {"破碎-极破碎", "极破碎"}:
        risk_level = "high"
    elif water_flag == 1 or support_grade == "Ⅴ" or stability == "较差" or rock_mass_state in {"破碎", "较破碎"}:
        risk_level = "medium"
    else:
        risk_level = "low"

    attrs = {
        "fact_type": "table2_segment",
        "lithology": "板岩夹变质砂岩" if "板岩夹变质砂岩" in flat_conc else None,
        "weathering": "弱风化" if "弱风化" in flat_conc else None,
        "support_grade": support_grade,
        "rock_hardness": rock_hardness,
        "rock_uniformity": rock_uniformity,
        "joint_degree": joint_degree,
        "rock_mass_state": rock_mass_state,
        "mud_filling_flag": mud_filling_flag,
        "stability": stability,
        "water_flag": water_flag,
        "water_type": water_type,
        "collapse_flag": collapse_flag,
        "risk_note_text": risk_note_text,
        "vp_min": vp["min"],
        "vp_max": vp["max"],
        "vs_min": vs["min"],
        "vs_max": vs["max"],
        "vp_vs_min": vp_vs["min"],
        "vp_vs_max": vp_vs["max"],
        "poisson_min": poisson["min"],
        "poisson_max": poisson["max"],
        "dynamic_modulus_min": modulus["min"],
        "dynamic_modulus_max": modulus["max"],
        "risk_level": risk_level,
        "risk_tags": risk_tags,
    }

    return EvidenceRecord(
        evidence_id=f"{meta['report_id']}_table2_{idx}",
        source_type="tsp",
        source_level="segment",
        report_id=meta["report_id"],
        report_date=meta["report_date"],
        issue_date=meta["issue_date"],
        tunnel_name=meta["tunnel_name"],
        start_num=_safe_mileage(s),
        end_num=_safe_mileage(e),
        face_num=meta["face_num"],
        next_forecast_num=meta["next_forecast_num"],
        confidence="high",
        attrs_json=json.dumps(attrs, ensure_ascii=False),
        raw_text=conclusion_text
    )


def parse_table2_records_by_pdfplumber(pdf_path: Path, meta: Dict[str, Any]) -> List[EvidenceRecord]:
    records: List[EvidenceRecord] = []
    idx = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                table_text = str(table)
                if ("纵波速度" not in table_text) and ("物性参数" not in table_text):
                    continue

                for row in table:
                    if not row:
                        continue

                    row = [("" if c is None else str(c)) for c in row]
                    joined = "".join(row)

                    # 跳过表头
                    if "里程范围" in joined and "预报结论" in joined:
                        continue

                    mileage_cell = row[0].strip() if len(row) >= 1 else ""
                    mileage_flat = re.sub(r"\s+", "", mileage_cell)

                    # 第一列必须像里程段
                    if not re.search(r"DyK\d+\+\d+\.?\d*", mileage_flat):
                        continue

                    picked = _pick_param_and_conclusion_cells(row)
                    params_cell = picked["params"]
                    conclusion_cell = picked["conclusion"]

                    # 如果参数列还是空，就用整行兜底
                    if not params_cell:
                        params_cell = " ".join(row)

                    row_data = {
                        "mileage": mileage_cell,
                        "params": params_cell,
                        "conclusion": conclusion_cell,
                    }

                    rec = parse_table2_structured_row(row_data, meta, idx)
                    if rec is not None:
                        records.append(rec)
                        idx += 1

    return records


# =========================
# 8. 冲突标记：结论 vs 表2
# =========================
def attach_grade_conflicts(records: List[EvidenceRecord]) -> List[EvidenceRecord]:
    conclusion_map = {}
    segment_map = {}

    for rec in records:
        try:
            attrs = rec.attrs() if callable(getattr(rec, "attrs", None)) else json.loads(rec.attrs_json)
        except Exception:
            try:
                attrs = json.loads(rec.attrs_json)
            except Exception:
                continue

        key = (rec.start_num, rec.end_num)

        if rec.source_level == "report_conclusion" and attrs.get("fact_type") == "support_grade_conclusion":
            conclusion_map[key] = attrs.get("support_grade")

        if rec.source_level == "segment" and attrs.get("fact_type") == "table2_segment":
            segment_map[key] = attrs.get("support_grade")

    for rec in records:
        try:
            attrs = rec.attrs() if callable(getattr(rec, "attrs", None)) else json.loads(rec.attrs_json)
        except Exception:
            try:
                attrs = json.loads(rec.attrs_json)
            except Exception:
                continue

        key = (rec.start_num, rec.end_num)
        if key in conclusion_map and key in segment_map:
            g1 = conclusion_map[key]
            g2 = segment_map[key]
            if g1 and g2 and g1 != g2:
                attrs["grade_conflict"] = 1
                attrs["support_grade_conclusion"] = g1
                attrs["support_grade_table2"] = g2
                attrs["support_grade_final"] = g2
                rec.attrs_json = json.dumps(attrs, ensure_ascii=False)

    return records


# =========================
# 9. 主入口
# =========================
def parse_tsp_pdf(pdf_path: Path) -> List[EvidenceRecord]:
    text = extract_pdf_text(pdf_path)
    meta = extract_meta(text, pdf_path.name)

    records: List[EvidenceRecord] = []

    # 1) overview
    records.extend(_parse_overview_record(text, meta))

    # 2) 目标区段
    section_text = extract_target_section(text)
    if not section_text:
        return records

    # 3) 7结论
    conclusion_text = extract_conclusion_text_from_section(section_text)
    if conclusion_text:
        records.extend(_parse_grade_records(conclusion_text, meta))
        records.extend(_parse_collapse_records(conclusion_text, meta))
        records.extend(_parse_water_records(conclusion_text, meta))

    # 4) 表2：只要 section 里有表2，就用 pdfplumber 解析 segment
    table2_text = extract_table2_text_from_section(section_text)
    if table2_text:
        records.extend(parse_table2_records_by_pdfplumber(pdf_path, meta))

    # 5) 冲突标记
    records = attach_grade_conflicts(records)

    return records