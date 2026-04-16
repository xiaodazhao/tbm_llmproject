# sketch_parser.py
import re
import json
from pathlib import Path
from typing import List, Optional

from schemas.schemas import EvidenceRecord
from utils.utils import mileage_to_num


def _norm_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u3000", " ")
    text = text.replace("～", "~")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _flat_text(text: str) -> str:
    text = _norm_text(text)
    text = text.replace("\n", "")
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _safe_mileage(x: Optional[str]) -> Optional[float]:
    if not x:
        return None
    try:
        return float(mileage_to_num(x))
    except Exception:
        return None


def _safe_search(pattern: str, text: str, flags=re.S) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _extract_grade(flat: str) -> Optional[str]:
    m = re.search(r"建议围岩级别([ⅠⅡⅢⅣⅤIVX]+)", flat)
    if m:
        return m.group(1)

    m = re.search(r"判定洞身围岩为([ⅠⅡⅢⅣⅤIVX]+)级", flat)
    if m:
        return m.group(1)

    m = re.search(r"设计围岩级别([ⅠⅡⅢⅣⅤIVX]+)", flat)
    if m:
        return m.group(1)

    return None


def _extract_weathering(flat: str) -> Optional[str]:
    # 优先识别带 √ 的
    for x in ["未风化", "微风化", "弱风化", "强风化", "全风化"]:
        if f"{x}√" in flat:
            return x

    # fallback：正文描述
    for x in ["全风化", "强风化", "弱风化", "微风化", "未风化"]:
        if x in flat:
            return x
    return None


def _extract_joint_degree(flat: str) -> Optional[str]:
    if "裂隙发育密集" in flat or "相对密集" in flat:
        return "发育密集"
    if "裂隙较发育" in flat:
        return "较发育"
    if "裂隙发育" in flat:
        return "发育"

    # 表格“间距”项兜底
    if "0.06~0.2√" in flat or "＜0.06√" in flat:
        return "发育密集"
    if "0.2~0.6√" in flat:
        return "发育"
    if "0.6~1.5√" in flat:
        return "较发育"

    return None


def _extract_rock_mass_state(flat: str) -> Optional[str]:
    if "破碎-极破碎" in flat or "破碎极破碎" in flat:
        return "破碎-极破碎"
    if "岩体极破碎" in flat:
        return "极破碎"
    # 优先识别明确“岩体破碎”
    if "岩体破碎" in flat or "压碎结构" in flat:
        return "破碎"
    if "岩体相对破碎" in flat or "岩体较破碎" in flat:
        return "较破碎"
    return None


def _extract_rock_uniformity(flat: str) -> Optional[str]:
    if "软硬不均" in flat:
        return "软硬不均"
    return None


def _extract_stability(flat: str) -> Optional[str]:
    if "不能自稳" in flat or "自稳困难" in flat or "自稳性较差" in flat:
        return "较差"
    if "稳定" in flat or "自稳" in flat:
        return "一般"
    return None


def _extract_water_info(flat: str):
    water_flag = 0
    water_type = None

    # 优先文本描述
    if "线-股状出水" in flat:
        water_flag = 1
        water_type = "线-股状出水"
    elif "线状出水" in flat:
        water_flag = 1
        water_type = "线状出水"
    elif "股状出水" in flat:
        water_flag = 1
        water_type = "股状出水"
    elif "渗滴水" in flat:
        water_flag = 1
        water_type = "渗滴水"
    elif "渗水" in flat:
        water_flag = 1
        water_type = "渗水"
    elif "涌出或喷出√" in flat or "涌出或喷出" in flat:
        water_flag = 1
        water_type = "涌出或喷出"
    elif "湿润√" in flat or "湿润" in flat:
        water_flag = 1
        water_type = "湿润"

    return water_flag, water_type


def _extract_collapse_flag(flat: str) -> int:
    if "掉块" in flat:
        return 1
    return 0


def _extract_lithology(flat: str) -> Optional[str]:
    if "板岩夹变质砂岩" in flat:
        return "板岩夹变质砂岩"
    return None


def _build_risk_tags(
    water_flag: int,
    water_type: Optional[str],
    collapse_flag: int,
    joint_degree: Optional[str],
    rock_mass_state: Optional[str],
    rock_uniformity: Optional[str],
    mud_filling_flag: int,
    support_grade: Optional[str],
):
    tags = []

    if water_flag:
        tags.append("出水")
    if water_type:
        tags.append(str(water_type))
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

    if rock_uniformity == "软硬不均":
        tags.append("软硬不均")

    if mud_filling_flag:
        tags.append("泥质填充")

    if support_grade:
        tags.append("围岩等级建议")

    out = []
    seen = set()
    for t in tags:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _infer_risk_level(
    support_grade: Optional[str],
    water_flag: int,
    water_type: Optional[str],
    collapse_flag: int,
    joint_degree: Optional[str],
    rock_mass_state: Optional[str],
    rock_uniformity: Optional[str],
    stability: Optional[str],
    mud_filling_flag: int,
) -> str:
    if (
        collapse_flag == 1
        or water_type in {"线-股状出水", "涌出或喷出"}
        or rock_mass_state in {"破碎-极破碎", "极破碎"}
        or (joint_degree == "发育密集" and stability == "较差")
        or (rock_uniformity == "软硬不均" and stability == "较差")
    ):
        return "high"

    if (
        water_flag == 1
        or support_grade == "Ⅴ"
        or stability == "较差"
        or rock_mass_state in {"破碎", "较破碎"}
        or joint_degree in {"发育", "较发育"}
        or mud_filling_flag == 1
    ):
        return "medium"

    return "low"


def parse_sketch_pdf(pdf_path: Path) -> List[EvidenceRecord]:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        text = "\n".join([p.get_text() for p in doc])
    finally:
        doc.close()

    text = _norm_text(text)
    flat = _flat_text(text)

    m = re.search(r"DyK\d+\+\d+\.?\d*", text)
    if not m:
        return []

    mileage_text = m.group()
    chainage = _safe_mileage(mileage_text)

    report_id = pdf_path.stem
    tunnel_name = _safe_search(r"(伯舒拉岭.*?)(?:洞身段地质素描记录表)", text)
    if not tunnel_name:
        tunnel_name = "伯舒拉岭隧道进口右线"

    issue_date = _safe_search(r"日期[:：]?\s*([0-9]{4}\s*年\s*[0-9]{1,2}\s*月\s*[0-9]{1,2}\s*日)", text)

    support_grade = _extract_grade(flat)
    weathering = _extract_weathering(flat)
    joint_degree = _extract_joint_degree(flat)
    rock_mass_state = _extract_rock_mass_state(flat)
    rock_uniformity = _extract_rock_uniformity(flat)
    stability = _extract_stability(flat)
    water_flag, water_type = _extract_water_info(flat)
    collapse_flag = _extract_collapse_flag(flat)
    lithology = _extract_lithology(flat)

    mud_filling_flag = 1 if "泥质填充" in flat else 0

    risk_tags = _build_risk_tags(
        water_flag=water_flag,
        water_type=water_type,
        collapse_flag=collapse_flag,
        joint_degree=joint_degree,
        rock_mass_state=rock_mass_state,
        rock_uniformity=rock_uniformity,
        mud_filling_flag=mud_filling_flag,
        support_grade=support_grade,
    )

    risk_level = _infer_risk_level(
        support_grade=support_grade,
        water_flag=water_flag,
        water_type=water_type,
        collapse_flag=collapse_flag,
        joint_degree=joint_degree,
        rock_mass_state=rock_mass_state,
        rock_uniformity=rock_uniformity,
        stability=stability,
        mud_filling_flag=mud_filling_flag,
    )

    attrs = {
        "fact_type": "sketch_observation",
        "lithology": lithology,
        "weathering": weathering,
        "support_grade": support_grade,
        "rock_hardness": None,
        "rock_uniformity": rock_uniformity,
        "joint_degree": joint_degree,
        "rock_mass_state": rock_mass_state,
        "mud_filling_flag": mud_filling_flag,
        "stability": stability,
        "water_flag": water_flag,
        "water_type": water_type,
        "collapse_flag": collapse_flag,
        "deformation_flag": 0,
        "risk_level": risk_level,
        "risk_tags": risk_tags,
    }

    rec = EvidenceRecord(
        evidence_id=f"{report_id}_sketch_0",
        source_type="sketch",
        source_level="point",
        report_id=report_id,
        report_date=issue_date,
        issue_date=issue_date,
        tunnel_name=tunnel_name,
        start_num=chainage,
        end_num=chainage,
        face_num=chainage,
        next_forecast_num=None,
        confidence="high",
        attrs_json=json.dumps(attrs, ensure_ascii=False),
        raw_text=text
    )

    return [rec]