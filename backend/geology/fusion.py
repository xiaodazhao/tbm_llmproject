# geology/fusion.py
import json
import re
import pandas as pd
from config import TOLERANCE_M


RISK_SCORE_MAP = {
    "low": 1,
    "medium": 2,
    "high": 3
}

GRADE_ORDER = {
    "Ⅰ": 1,
    "Ⅱ": 2,
    "Ⅲ": 3,
    "Ⅳ": 4,
    "Ⅴ": 5,
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
}


def get_active(chainage, evidence):
    """
    获取当前里程命中的所有地质证据
    """
    return evidence[
        (evidence["start_num"] - TOLERANCE_M <= chainage) &
        (evidence["end_num"] + TOLERANCE_M >= chainage)
    ]


def normalize_report_id(report_id: str) -> str:
    """
    规范化报告ID，避免同一报告因命名差异被重复统计
    """
    if pd.isna(report_id):
        return ""

    report_id = str(report_id).strip()

    report_id = re.sub(r"_[0-9]+$", "", report_id)
    report_id = report_id.replace("-_", "_")
    report_id = re.sub(r"[-_]+", "_", report_id)
    report_id = report_id.rstrip("_")

    return report_id


def _safe_load_attrs(attrs_json_str):
    try:
        data = json.loads(attrs_json_str)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalize_grade(x):
    """
    统一围岩等级字段，兼容旧字段 rock_grade 和新字段 support_grade
    """
    if x is None:
        return None

    x = str(x).strip()
    if not x:
        return None

    mapping = {
        "1": "Ⅰ",
        "2": "Ⅱ",
        "3": "Ⅲ",
        "4": "Ⅳ",
        "5": "Ⅴ",
        "I": "Ⅰ",
        "II": "Ⅱ",
        "III": "Ⅲ",
        "IV": "Ⅳ",
        "V": "Ⅴ",
        "Ⅰ": "Ⅰ",
        "Ⅱ": "Ⅱ",
        "Ⅲ": "Ⅲ",
        "Ⅳ": "Ⅳ",
        "Ⅴ": "Ⅴ",
    }
    return mapping.get(x, x)


def _record_weight(row, attrs):
    """
    给不同证据一个简单权重：
    - 报告结论层 > 普通区段层
    - 高置信度 > 中 > 低
    """
    weight = 1.0

    source_level = str(row.get("source_level", "")).strip()
    confidence = str(row.get("confidence", "")).strip().lower()

    if source_level == "report_conclusion":
        weight *= 1.20
    elif source_level == "segment":
        weight *= 1.00

    if confidence == "high":
        weight *= 1.15
    elif confidence == "medium":
        weight *= 1.00
    elif confidence == "low":
        weight *= 0.90

    # 若已有冲突/不一致标记，稍降权
    if attrs.get("consistency_flag") == 0:
        weight *= 0.90
    if attrs.get("grade_conflict") == 1:
        weight *= 0.95

    return weight


def _infer_risk_if_missing(attrs):
    """
    当某条证据缺失 risk_level 时，按结构字段兜底推断一个
    """
    risk = attrs.get("risk_level")
    if risk in RISK_SCORE_MAP:
        return risk

    water_flag = int(bool(attrs.get("water_flag", 0)))
    collapse_flag = int(bool(attrs.get("collapse_flag", 0)))
    deformation_flag = int(bool(attrs.get("deformation_flag", 0)))

    water_type = attrs.get("water_type")
    support_grade = _normalize_grade(
        attrs.get("support_grade") or attrs.get("rock_grade")
    )
    rock_mass_state = str(attrs.get("rock_mass_state", "") or "").strip()
    joint_degree = str(attrs.get("joint_degree", "") or "").strip()
    stability = str(attrs.get("stability", "") or "").strip()

    if (
        collapse_flag == 1
        or deformation_flag == 1
        or water_type == "线-股状出水"
        or rock_mass_state in {"破碎极破碎", "破碎-极破碎", "极破碎"}
        or (joint_degree == "发育密集" and stability == "较差")
    ):
        return "high"

    if (
        water_flag == 1
        or support_grade == "Ⅴ"
        or stability == "较差"
        or rock_mass_state in {"破碎", "较破碎"}
        or joint_degree in {"发育", "较发育"}
    ):
        return "medium"

    return "low"


# =========================
# 新增：更细的结构化评分
# =========================
def _compute_detail_scores(attrs):
    """
    把 parser 抽出来的细粒度字段真正转成多个子风险分量
    输出 0~3 的分值
    """
    water_flag = int(bool(attrs.get("water_flag", 0)))
    collapse_flag = int(bool(attrs.get("collapse_flag", 0)))
    deformation_flag = int(bool(attrs.get("deformation_flag", 0)))
    mud_filling_flag = int(bool(attrs.get("mud_filling_flag", 0)))

    water_type = str(attrs.get("water_type", "") or "").strip()
    support_grade = _normalize_grade(
        attrs.get("support_grade") or attrs.get("rock_grade")
    )
    rock_mass_state = str(attrs.get("rock_mass_state", "") or "").strip()
    joint_degree = str(attrs.get("joint_degree", "") or "").strip()
    stability = str(attrs.get("stability", "") or "").strip()
    rock_uniformity = str(attrs.get("rock_uniformity", "") or "").strip()

    risk_tags = attrs.get("risk_tags", [])
    if not isinstance(risk_tags, list):
        risk_tags = []

    # 1) 出水风险
    if water_type == "线-股状出水":
        water_risk_score = 3
    elif water_type in {"股状出水", "线状出水"}:
        water_risk_score = 2
    elif water_flag == 1:
        water_risk_score = 1
    else:
        water_risk_score = 0

    # 2) 掉块/变形风险
    collapse_risk_score = 0
    if collapse_flag == 1:
        collapse_risk_score = max(collapse_risk_score, 3)
    if deformation_flag == 1:
        collapse_risk_score = max(collapse_risk_score, 3)
    if "掉块" in risk_tags:
        collapse_risk_score = max(collapse_risk_score, 2)

    # 3) 围岩质量风险
    rockmass_risk_score = 0
    if rock_mass_state in {"破碎极破碎", "破碎-极破碎", "极破碎"}:
        rockmass_risk_score = max(rockmass_risk_score, 3)
    elif rock_mass_state in {"破碎", "较破碎"}:
        rockmass_risk_score = max(rockmass_risk_score, 2)

    if joint_degree == "发育密集":
        rockmass_risk_score = max(rockmass_risk_score, 3 if rockmass_risk_score >= 2 else 2)
    elif joint_degree in {"发育", "较发育"}:
        rockmass_risk_score = max(rockmass_risk_score, 1)

    if stability == "较差":
        rockmass_risk_score = max(rockmass_risk_score, 2)
    elif stability == "一般":
        rockmass_risk_score = max(rockmass_risk_score, 1)

    if mud_filling_flag == 1:
        rockmass_risk_score = min(3, rockmass_risk_score + 1)

    if rock_uniformity == "软硬不均":
        rockmass_risk_score = min(3, rockmass_risk_score + 1)

    # 4) 围岩等级风险
    if support_grade == "Ⅴ":
        grade_risk_score = 3
    elif support_grade == "Ⅳ":
        grade_risk_score = 2
    elif support_grade == "Ⅲ":
        grade_risk_score = 1
    else:
        grade_risk_score = 0

    return {
        "water_risk_score": int(min(3, max(0, water_risk_score))),
        "collapse_risk_score": int(min(3, max(0, collapse_risk_score))),
        "rockmass_risk_score": int(min(3, max(0, rockmass_risk_score))),
        "grade_risk_score": int(min(3, max(0, grade_risk_score))),
    }


def _detail_scores_to_risk(detail_scores, attrs):
    """
    用细粒度分量反推出一条证据的风险等级
    """
    m = max(detail_scores.values()) if detail_scores else 0
    s = sum(detail_scores.values()) if detail_scores else 0

    # 若原始 risk_level 已有，作为参考上界
    raw_risk = attrs.get("risk_level")
    raw_score = RISK_SCORE_MAP.get(raw_risk, 0)

    final_score = max(raw_score, 0)

    if m >= 3:
        final_score = max(final_score, 3)
    elif s >= 5:
        final_score = max(final_score, 3)
    elif m >= 2 or s >= 3:
        final_score = max(final_score, 2)
    else:
        final_score = max(final_score, 1)

    if final_score >= 3:
        return "high"
    elif final_score >= 2:
        return "medium"
    return "low"


def _risk_from_scores(mean_score, max_score):
    """
    区分'平均危险程度'与'最坏情形'
    当前策略偏保守：
    - 只要 max_score 很高，整体风险不低于 medium/high
    """
    if max_score >= 2.8:
        return "high"
    if mean_score >= 2.4:
        return "high"
    if max_score >= 2.2:
        return "medium"
    if mean_score >= 1.8:
        return "medium"
    return "low"


def fuse(chainage, df):
    """
    对某一里程的命中证据进行融合
    """
    # ===== 无证据 =====
    if df.empty:
        return {
            "chainage": chainage,
            "coverage": "none",
            "risk": "low",
            "risk_score": 1,
            "hazard": "无明显异常",
            "water_flag_fused": 0,
            "collapse_flag_fused": 0,
            "deformation_flag_fused": 0,
            "active_source_count": 0,
            "active_sources": "",
            "active_evidence_ids": "",
            "active_report_ids": "",
            "fused_grade": "",
            "uncertainty": "high",
            "water_risk_score": 0,
            "collapse_risk_score": 0,
            "rockmass_risk_score": 0,
            "grade_risk_score": 0,
            "detail_score_mean": 0.0,
            "detail_score_max": 0.0,
        }

    parsed_rows = []
    for _, row in df.iterrows():
        attrs = _safe_load_attrs(row.get("attrs_json", ""))
        parsed_rows.append((row, attrs))

    # ===== 逐条证据：细粒度评分 + 加权风险 =====
    weighted_scores = []
    total_weight = 0.0
    raw_scores = []

    water_score_weighted = 0.0
    collapse_score_weighted = 0.0
    rockmass_score_weighted = 0.0
    grade_score_weighted = 0.0

    water_score_max = 0
    collapse_score_max = 0
    rockmass_score_max = 0
    grade_score_max = 0

    for row, attrs in parsed_rows:
        detail_scores = _compute_detail_scores(attrs)
        inferred_risk = _detail_scores_to_risk(detail_scores, attrs)

        # 双保险：如果细粒度没覆盖到，再走旧兜底
        if inferred_risk not in RISK_SCORE_MAP:
            inferred_risk = _infer_risk_if_missing(attrs)

        score = RISK_SCORE_MAP.get(inferred_risk, 1)
        weight = _record_weight(row, attrs)

        weighted_scores.append(score * weight)
        raw_scores.append(score)
        total_weight += weight

        water_score_weighted += detail_scores["water_risk_score"] * weight
        collapse_score_weighted += detail_scores["collapse_risk_score"] * weight
        rockmass_score_weighted += detail_scores["rockmass_risk_score"] * weight
        grade_score_weighted += detail_scores["grade_risk_score"] * weight

        water_score_max = max(water_score_max, detail_scores["water_risk_score"])
        collapse_score_max = max(collapse_score_max, detail_scores["collapse_risk_score"])
        rockmass_score_max = max(rockmass_score_max, detail_scores["rockmass_risk_score"])
        grade_score_max = max(grade_score_max, detail_scores["grade_risk_score"])

    if total_weight <= 0:
        risk = "low"
        mean_score = 1.0
        max_score = 1.0
    else:
        mean_score = sum(weighted_scores) / total_weight
        max_score = max(raw_scores) if raw_scores else 1.0
        risk = _risk_from_scores(mean_score, max_score)

    risk_score = RISK_SCORE_MAP.get(risk, 1)

    # ===== 灾害融合 =====
    hazard_set = set()
    grades = []

    water_flag_fused = 0
    collapse_flag_fused = 0
    deformation_flag_fused = 0

    for row, attrs in parsed_rows:
        water_type = attrs.get("water_type")
        if water_type:
            hazard_set.add(str(water_type))
            water_flag_fused = 1
        elif attrs.get("water_flag"):
            hazard_set.add("出水")
            water_flag_fused = 1

        if attrs.get("collapse_flag"):
            hazard_set.add("掉块")
            collapse_flag_fused = 1

        if attrs.get("deformation_flag"):
            hazard_set.add("变形")
            deformation_flag_fused = 1

        tags = attrs.get("risk_tags", [])
        if isinstance(tags, list):
            for t in tags:
                if t:
                    hazard_set.add(str(t))

        grade = _normalize_grade(
            attrs.get("support_grade") or attrs.get("rock_grade")
        )
        if grade:
            grades.append(grade)

    hazard = "+".join(sorted(hazard_set)) if hazard_set else "无明显异常"

    # ===== 围岩等级融合（从严原则）=====
    fused_grade = ""
    if grades:
        fused_grade = sorted(
            grades,
            key=lambda x: GRADE_ORDER.get(str(x), 0),
            reverse=True
        )[0]

    # ===== 按 source_type + report_id 去重统计 =====
    unique_report_keys = set()
    unique_report_ids = set()

    for _, row in df.iterrows():
        source_type = str(row.get("source_type", "")).strip()
        report_id = normalize_report_id(row.get("report_id", ""))
        unique_report_keys.add((source_type, report_id))
        if report_id:
            unique_report_ids.add(report_id)

    active_source_count = len(unique_report_keys)
    unique_sources = sorted(set(df["source_type"].astype(str)))

    # ===== 不确定性 =====
    if active_source_count >= 3:
        uncertainty = "low"
    elif active_source_count == 2:
        uncertainty = "medium"
    else:
        uncertainty = "high"

    # ===== 子分量融合输出 =====
    if total_weight > 0:
        water_risk_score = round(water_score_weighted / total_weight, 2)
        collapse_risk_score = round(collapse_score_weighted / total_weight, 2)
        rockmass_risk_score = round(rockmass_score_weighted / total_weight, 2)
        grade_risk_score = round(grade_score_weighted / total_weight, 2)
    else:
        water_risk_score = 0.0
        collapse_risk_score = 0.0
        rockmass_risk_score = 0.0
        grade_risk_score = 0.0

    detail_score_mean = round(
        (water_risk_score + collapse_risk_score + rockmass_risk_score + grade_risk_score) / 4.0,
        2
    )
    detail_score_max = round(
        max(water_score_max, collapse_score_max, rockmass_score_max, grade_score_max),
        2
    )

    return {
        "chainage": chainage,
        "coverage": "multi" if active_source_count > 1 else "single",
        "risk": risk,
        "risk_score": risk_score,
        "hazard": hazard,
        "water_flag_fused": water_flag_fused,
        "collapse_flag_fused": collapse_flag_fused,
        "deformation_flag_fused": deformation_flag_fused,
        "active_source_count": active_source_count,
        "active_sources": ";".join(unique_sources),
        "active_evidence_ids": ";".join(df["evidence_id"].astype(str).tolist()),
        "active_report_ids": ";".join(sorted(unique_report_ids)),
        "fused_grade": fused_grade,
        "uncertainty": uncertainty,

        # 新增，更细粒度的融合输出
        "water_risk_score": water_risk_score,
        "collapse_risk_score": collapse_risk_score,
        "rockmass_risk_score": rockmass_risk_score,
        "grade_risk_score": grade_risk_score,
        "detail_score_mean": detail_score_mean,
        "detail_score_max": detail_score_max,
    }


def annotate_unique_chainage(unique_chainage_df, evidence):
    """
    对唯一里程表做逐里程注记
    """
    results = []
    total = len(unique_chainage_df)

    for i, x in enumerate(unique_chainage_df["chainage"], start=1):
        if i % 5000 == 0:
            print(f"已处理唯一里程: {i}/{total}")

        hit_df = get_active(x, evidence)
        results.append(fuse(x, hit_df))

    return pd.DataFrame(results)