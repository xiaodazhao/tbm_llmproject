import glob
import os
from typing import Optional
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# 自定义模块
# =========================
from analysis.dataprocess import (
    load_and_process, segments_to_text, compute_stats, stats_to_text
)
from analysis.excavation_state import (
    detect_excavation_state, excavation_state_segments, explain_excavation_states,
    excavation_state_to_text, excavation_state_efficiency, excavation_state_stats,
    excavation_state_stats_to_text
)
from analysis.gas_analysis import compute_gas_stats, gas_stats_to_text

from geology.geology_fusion_backend import attach_geology_labels, load_evidence_db
from geology.geology_summary import (
    summarize_geology_record_level,
    summarize_geology_segment_level,
    geology_summary_to_text
)
from geology.segment_analysis import (
    run_segment_analysis,
    build_typical_segments_table
)

from analysis.forward_risk_advisor import (
    generate_forward_risk_summary,
    forward_risk_to_text,
)

from llm.prompt_builder import build_prompt
from llm.prompt_builder_timewindow import build_prompt_timewindow
from llm.llm_api import call_llm

from utils.time_window_utils import load_df_by_time

from geology.geology_summary import build_face_geo_text


# =========================
# FastAPI 初始化
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 路径配置
# =========================
from config import DATA_DIR, EVIDENCE_DB_PATH

# =========================
# 请求体
# =========================
class DailyReportRequest(BaseModel):
    date: str  # YYYY-MM-DD


class TimeWindowRequest(BaseModel):
    start_time: str
    end_time: str


# =========================
# 工具函数
# =========================
def get_file_path_by_date(date_str: str) -> Path:
    date_compact = date_str.replace("-", "")
    file_path = DATA_DIR / f"tbm_data_{date_compact}.csv"
    if not file_path.exists():
        raise FileNotFoundError
    return file_path


def load_data_from_path(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if "运行时间-time" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "运行时间-time"})

    if "运行时间-time" not in df.columns:
        raise ValueError("缺少时间列：运行时间-time")

    df["运行时间-time"] = pd.to_datetime(df["运行时间-time"], errors="coerce")
    df = (
        df.dropna(subset=["运行时间-time"])
        .sort_values("运行时间-time")
        .reset_index(drop=True)
    )
    return df


def get_df_by_date(date_str: str) -> pd.DataFrame:
    return load_data_from_path(get_file_path_by_date(date_str))


def get_latest_df() -> pd.DataFrame:
    files = glob.glob(str(DATA_DIR / "tbm_data_*.csv"))
    if not files:
        raise FileNotFoundError
    latest_file = max(files)
    return load_data_from_path(Path(latest_file))


def serialize_for_json(obj):
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [serialize_for_json(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


def semantic_efficiency_to_text(eff_df: pd.DataFrame) -> str:
    """
    语义聚合后的施工状态效率表 -> 文本
    """
    if eff_df is None or eff_df.empty:
        return "数据量不足，无法计算施工状态效率统计。"

    lines = ["不同施工状态语义下的效率统计如下："]

    for _, row in eff_df.iterrows():
        label = row.get("label_text", "未知状态")
        parts = [f"{label}："]

        if "平均推进速度" in row and pd.notna(row["平均推进速度"]):
            parts.append(f"平均推进速度 {row['平均推进速度']:.2f}")

        if "平均推力" in row and pd.notna(row["平均推力"]):
            parts.append(f"平均推力 {row['平均推力']:.2f}")

        if "平均刀盘扭矩" in row and pd.notna(row["平均刀盘扭矩"]):
            parts.append(f"平均刀盘扭矩 {row['平均刀盘扭矩']:.2f}")

        if "平均刀盘实际转速" in row and pd.notna(row["平均刀盘实际转速"]):
            parts.append(f"平均刀盘实际转速 {row['平均刀盘实际转速']:.2f}")

        lines.append("，".join(parts) + "。")

    return "\n".join(lines)


# =========================
# 自适应状态识别参数
# =========================
STATE_FEATURES = ("推力", "刀盘扭矩", "刀盘实际转速", "推进速度")


def estimate_valid_samples(
    df: pd.DataFrame,
    feature_cols=STATE_FEATURES
) -> int:
    valid_mask = pd.Series(True, index=df.index)

    if "掘进状态" in df.columns:
        valid_mask &= (pd.to_numeric(df["掘进状态"], errors="coerce").fillna(0) != 0)
    else:
        temp_mask = pd.Series(False, index=df.index)
        if "推力" in df.columns:
            temp_mask |= (pd.to_numeric(df["推力"], errors="coerce").fillna(0).abs() > 1e-8)
        if "推进速度" in df.columns:
            temp_mask |= (pd.to_numeric(df["推进速度"], errors="coerce").fillna(0).abs() > 1e-8)
        valid_mask &= temp_mask

    for col in feature_cols:
        if col in df.columns:
            valid_mask &= pd.to_numeric(df[col], errors="coerce").notna()

    return int(valid_mask.sum())


def choose_state_params(n_valid: int):
    if n_valid < 5:
        return {
            "do_cluster": False,
            "n_states": 0,
            "min_duration_sec": 0
        }
    elif n_valid < 10:
        return {
            "do_cluster": True,
            "n_states": 2,
            "min_duration_sec": 0
        }
    elif n_valid < 30:
        return {
            "do_cluster": True,
            "n_states": 3,
            "min_duration_sec": 20
        }
    else:
        return {
            "do_cluster": True,
            "n_states": 4,
            "min_duration_sec": 60
        }


def risk_probability_to_text(df_geo):
    """
    简化版：基于已有字段生成风险评估文字
    不依赖 risk_prob / risk_prob_smooth
    """
    try:
        if df_geo is None or df_geo.empty:
            return "未进行区段风险概率评估分析。"

        df = df_geo.copy()

        if "chainage" not in df.columns:
            return "缺少里程信息，无法进行区段风险概率分析。"

        df["chainage"] = pd.to_numeric(df["chainage"], errors="coerce")
        df = df.dropna(subset=["chainage"]).copy()

        if df.empty:
            return "缺少有效里程数据，无法进行区段风险概率分析。"

        score = pd.Series(0.0, index=df.index)

        if "active_source_count" in df.columns:
            score += pd.to_numeric(df["active_source_count"], errors="coerce").fillna(0)

        if "risk_score" in df.columns:
            score += pd.to_numeric(df["risk_score"], errors="coerce").fillna(0)

        if "weighted_evidence_strength" in df.columns:
            score += pd.to_numeric(df["weighted_evidence_strength"], errors="coerce").fillna(0)

        df["risk_prob_like"] = score

        top = (
            df.sort_values("risk_prob_like", ascending=False)
              .drop_duplicates(subset=["chainage"])
              .head(5)
        )

        if top.empty or float(top["risk_prob_like"].fillna(0).max()) <= 0:
            return "基于现有多源地质信息，当前未识别出表现特别突出的高关注区段，整体风险评估结果相对平稳。"

        lines = []
        lines.append("基于多源地质信息及区段响应特征，对沿线区段进行了综合风险评估。结果表明：")

        for _, row in top.iterrows():
            ch = row["chainage"]
            active_cnt = int(pd.to_numeric(row.get("active_source_count", 0), errors="coerce") or 0)
            hazard = str(row.get("hazard", "")).strip()

            text = f"里程约 DK{ch/1000:.3f} 附近区段表现出相对较高关注特征"
            if active_cnt > 0:
                text += f"，多源关注数为 {active_cnt}"
            if hazard and hazard.lower() != "nan":
                text += f"，主要关注表现为{hazard}"
            text += "。"
            lines.append(text)

        lines.append(
            "总体来看，上述结果反映的是多源信息综合关注程度，不代表实际灾害已发生，相关结论仍需结合现场监测、掌子面揭示情况及施工响应进一步核实。"
        )

        return "\n".join(lines)

    except Exception as e:
        print("[Risk Prob Text Error]", e)
        return "区段风险概率分析不可用。"
# =========================
# 核心分析引擎
# =========================
def analyze_tbm_data(df: pd.DataFrame):
    # ===== 0. 地质融合 =====
    try:
        evidence_df = load_evidence_db(EVIDENCE_DB_PATH)
        df_geo = attach_geology_labels(df, evidence_df)

        geo_summary_record = summarize_geology_record_level(df_geo)
        segment_df = run_segment_analysis(df_geo, segment_length=10)
        typical_segments_df = build_typical_segments_table(segment_df, top_n=20)
        geo_summary_segment = summarize_geology_segment_level(segment_df)
        geo_text = geology_summary_to_text(geo_summary_segment)
        face_geo_text = build_face_geo_text(evidence_df)
        forward_risk_summary = generate_forward_risk_summary(
            df_plc=df_geo,
            evidence_df=evidence_df,
            lookahead_m=30
        )
        forward_risk_text = forward_risk_to_text(forward_risk_summary)

    except Exception as e:
        print(f"[Geology Error] {e}")
        df_geo = df.copy()
        segment_df = pd.DataFrame()
        typical_segments_df = pd.DataFrame()
        geo_summary_record = {"has_geology": False, "summary_text": "地质融合分析不可用。"}
        geo_summary_segment = {"has_geology": False, "summary_text": "地质融合分析不可用。"}
        geo_text = "地质融合分析不可用。"
        forward_risk_summary = {"has_forward_risk": False}
        forward_risk_text = "前方风险提示不可用。"

    # ===== 1. 基础工况分析 =====
    segments = load_and_process(df_geo)
    seg_text = segments_to_text(segments)
    stats = compute_stats(segments)
    stats_text = stats_to_text(stats)

    # ===== 2. 自适应施工状态识别 =====
    n_valid = estimate_valid_samples(df_geo, STATE_FEATURES)
    state_cfg = choose_state_params(n_valid)

    state_labels = {}
    state_segments = {}
    state_text = "当日有效工作样本过少，未进行隐含施工状态识别。"
    eff_df = pd.DataFrame()
    eff_text = "当日有效工作样本过少，无法计算施工状态效率统计。"
    state_stats = {}
    state_stats_text = "当日有效工作样本过少，无施工状态统计结果。"
    df_state = df_geo.copy()

    if state_cfg["do_cluster"]:
        try:
            df_state, _ = detect_excavation_state(
                df_geo.copy(),
                features=STATE_FEATURES,
                n_states=state_cfg["n_states"]
            )

            state_labels = explain_excavation_states(df_state)

            # 原始 state_id 分段结果
            state_segments = excavation_state_segments(
                df_state,
                min_duration_sec=state_cfg["min_duration_sec"]
            )
            state_text = excavation_state_to_text(state_segments, state_labels)

            # 原始效率统计（按 state_id）
            raw_eff_df = excavation_state_efficiency(df_state)

            # 语义聚合后的效率表
            if not raw_eff_df.empty:
                eff_df = raw_eff_df.reset_index().rename(columns={"state_id": "label"})
                eff_df["label_text"] = eff_df["label"].map(state_labels).fillna("未知状态")

                agg_dict = {}
                for col in ["平均推进速度", "平均推力", "平均刀盘扭矩", "平均刀盘实际转速", "平均推进给定速度"]:
                    if col in eff_df.columns:
                        agg_dict[col] = "mean"

                if agg_dict:
                    eff_df = (
                        eff_df
                        .groupby("label_text", as_index=False)
                        .agg(agg_dict)
                    )
                else:
                    eff_df = pd.DataFrame()

                eff_text = semantic_efficiency_to_text(eff_df)
            else:
                eff_df = pd.DataFrame()
                eff_text = "数据量不足，无法计算施工状态效率统计。"

            # 状态统计仍按原始 state_id
            state_stats = excavation_state_stats(df_state, state_segments)
            state_stats_text = excavation_state_stats_to_text(state_stats, state_labels)

        except Exception as e:
            print(f"[State Error] {e}")
            df_state = df_geo.copy()
            state_labels = {}
            state_segments = {}
            state_text = "施工状态分析不可用。"
            eff_df = pd.DataFrame()
            eff_text = "施工效率分析不可用。"
            state_stats = {}
            state_stats_text = "施工状态统计不可用。"

    # ===== 3. 气体分析 =====
    try:
        gas_stats = compute_gas_stats(df_geo, df_state=df_state)
        gas_text = gas_stats_to_text(gas_stats)
    except Exception as e:
        print(f"[Gas Error] {e}")
        gas_stats = {}
        gas_text = "无气体监测数据。"

    # ===== 4. LLM 结构化摘要 =====
    llm_summary = {
        "基础工况统计": stats,
        "施工状态标签": state_labels,
        "施工状态统计": state_stats,
        "施工状态效率表": eff_df.to_dict(orient="records") if not eff_df.empty else [],
        "气体统计": gas_stats,
        "地质摘要_记录级": geo_summary_record,
        "地质摘要_区段级": geo_summary_segment,
        "典型地质区段": typical_segments_df.to_dict(orient="records") if not typical_segments_df.empty else [],
        "前方风险提示摘要": forward_risk_summary,
        "前方风险提示文本": forward_risk_text,
        "有效状态样本数": n_valid,
        "状态识别配置": state_cfg
    }
    risk_prob_text = risk_probability_to_text(df_geo)
    return {
        "segments": segments,
        "seg_text": seg_text,
        "stats": stats,
        "stats_text": stats_text,
        "df_geo": df_geo,
        "df_state": df_state,
        "state_labels": state_labels,
        "state_segments": state_segments,
        "state_text": state_text,
        "eff_df": eff_df,
        "eff_text": eff_text,
        "state_stats": state_stats,
        "state_stats_text": state_stats_text,
        "gas_stats": gas_stats,
        "gas_text": gas_text,
        "geo_summary_record": geo_summary_record,
        "geo_summary_segment": geo_summary_segment,
        "geo_summary": geo_summary_segment,
        "geo_text": geo_text,
        "segment_df": segment_df,
        "typical_segments_df": typical_segments_df,
        "forward_risk_summary": forward_risk_summary,
        "forward_risk_text": forward_risk_text,
        "llm_summary": llm_summary,
        "face_geo_text": face_geo_text,
        "risk_prob_text": risk_prob_text,
    }


# =========================
# 空间风险剖面
# =========================
def build_risk_profile(df_geo: pd.DataFrame):
    if "chainage" not in df_geo.columns:
        return {
            "profile": [],
            "high_segments": [],
            "has_data": False,
            "message": "缺少 chainage 字段"
        }

    use_cols = [c for c in [
        "chainage",
        "active_source_count",
        "risk",
        "risk_score",
        "hazard",
        "coverage",
        "active_sources",
        "fused_grade"
    ] if c in df_geo.columns]

    if not use_cols:
        return {
            "profile": [],
            "high_segments": [],
            "has_data": False,
            "message": "缺少风险剖面所需字段"
        }

    prof_raw = df_geo[use_cols].copy()
    prof_raw["chainage"] = pd.to_numeric(prof_raw["chainage"], errors="coerce")
    prof_raw = prof_raw.dropna(subset=["chainage"]).copy()

    if prof_raw.empty:
        return {
            "profile": [],
            "high_segments": [],
            "has_data": False,
            "message": "当前日期无可用剖面数据"
        }

    if "active_source_count" in prof_raw.columns:
        prof_raw["active_source_count"] = pd.to_numeric(
            prof_raw["active_source_count"], errors="coerce"
        ).fillna(0)

    if "risk_score" in prof_raw.columns:
        prof_raw["risk_score"] = pd.to_numeric(
            prof_raw["risk_score"], errors="coerce"
        ).fillna(0)

    for col in ["risk", "hazard", "coverage", "active_sources", "fused_grade"]:
        if col in prof_raw.columns:
            prof_raw[col] = prof_raw[col].fillna("")

    agg_dict = {}
    if "active_source_count" in prof_raw.columns:
        agg_dict["active_source_count"] = "max"
    if "risk_score" in prof_raw.columns:
        agg_dict["risk_score"] = "max"
    if "risk" in prof_raw.columns:
        agg_dict["risk"] = "first"
    if "hazard" in prof_raw.columns:
        agg_dict["hazard"] = "first"
    if "coverage" in prof_raw.columns:
        agg_dict["coverage"] = "first"
    if "active_sources" in prof_raw.columns:
        agg_dict["active_sources"] = "first"
    if "fused_grade" in prof_raw.columns:
        agg_dict["fused_grade"] = "first"

    prof = (
        prof_raw
        .groupby("chainage", as_index=False)
        .agg(agg_dict)
        .sort_values("chainage")
        .reset_index(drop=True)
    )

    if prof.empty:
        return {
            "profile": [],
            "high_segments": [],
            "has_data": False,
            "message": "当前日期无可用剖面数据"
        }

    if "risk_score" in prof.columns:
        prof["risk_value"] = prof["risk_score"]
    elif "risk" in prof.columns:
        risk_map = {"low": 1, "medium": 2, "high": 3}
        prof["risk_value"] = prof["risk"].map(risk_map).fillna(0)
    else:
        prof["risk_value"] = 0

    ch_min = prof["chainage"].min()
    prof["chainage_rel"] = prof["chainage"] - ch_min

    high_segments = []
    if "active_source_count" in prof.columns:
        high_df = prof[prof["active_source_count"] >= 4].copy()
    else:
        high_df = pd.DataFrame()

    if not high_df.empty:
        high_df["gap"] = high_df["chainage"].diff().fillna(0)
        high_df["group"] = (high_df["gap"] > 2).cumsum()

        for _, g in high_df.groupby("group"):
            hazards = sorted(set(
                str(x) for x in g.get("hazard", pd.Series(dtype=str)).dropna().tolist() if str(x).strip()
            ))
            sources = sorted(set(
                str(x) for x in g.get("active_sources", pd.Series(dtype=str)).dropna().tolist() if str(x).strip()
            ))

            high_segments.append({
                "start_chainage": float(g["chainage"].min()),
                "end_chainage": float(g["chainage"].max()),
                "start_rel": float(g["chainage_rel"].min()),
                "end_rel": float(g["chainage_rel"].max()),
                "max_attention": int(g["active_source_count"].max()),
                "hazards": " / ".join(hazards[:3]) if hazards else "",
                "sources": " / ".join(sources[:3]) if sources else "",
            })

    return {
        "has_data": True,
        "profile": serialize_for_json(prof.to_dict(orient="records")),
        "high_segments": serialize_for_json(high_segments),
        "message": "ok"
    }


def build_speed_profile(df_geo: pd.DataFrame):
    if "chainage" not in df_geo.columns or "推进速度" not in df_geo.columns:
        return []

    plc = df_geo.copy()
    plc["chainage"] = pd.to_numeric(plc["chainage"], errors="coerce")
    plc["推进速度"] = pd.to_numeric(plc["推进速度"], errors="coerce")
    plc = plc.dropna(subset=["chainage", "推进速度"])

    if plc.empty:
        return []

    if "掘进状态" in plc.columns:
        plc = plc[pd.to_numeric(plc["掘进状态"], errors="coerce").fillna(0) != 0].copy()

    if plc.empty:
        return []

    grp = (
        plc.groupby("chainage", as_index=False)["推进速度"]
        .mean()
        .sort_values("chainage")
        .reset_index(drop=True)
    )

    global_min = pd.to_numeric(df_geo["chainage"], errors="coerce").dropna().min()
    grp["chainage_rel"] = grp["chainage"] - global_min

    return serialize_for_json(grp.to_dict(orient="records"))



# =========================
# API 接口
# =========================
@app.get("/api/tbm/dates")
def get_available_dates():
    dates = []
    for f in glob.glob(str(DATA_DIR / "tbm_data_*.csv")):
        try:
            d = os.path.basename(f).replace("tbm_data_", "").replace(".csv", "")
            dates.append(datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d"))
        except Exception:
            pass
    dates.sort(reverse=True)
    return {"dates": dates}


@app.post("/api/tbm/report")
def generate_daily_report(req: DailyReportRequest):
    try:
        df = get_df_by_date(req.date)
        result = analyze_tbm_data(df)

        prompt = build_prompt(
            seg_text=result["seg_text"],
            stats_text=result["stats_text"],
            state_text=result["state_text"],
            eff_text=result["eff_text"],
            state_stats_text=result["state_stats_text"],
            gas_text=result["gas_text"],
            geo_text=result["geo_text"],
            face_geo_text=result["face_geo_text"],
            llm_summary=result["llm_summary"],
            risk_prob_text=result["risk_prob_text"] 
        )
        

        report = call_llm(prompt)
        return {"report": report}

    except FileNotFoundError:
        return {"report": f"❌ 找不到 {req.date} 的数据文件"}
    except Exception as e:
        return {"report": f"❌ 服务器错误：{e}"}


@app.get("/api/tbm/summary")
def tbm_summary(date: Optional[str] = None):
    try:
        df = get_df_by_date(date) if date else get_latest_df()
        result = analyze_tbm_data(df)

        stats = result["stats"]
        geo_summary = result.get("geo_summary_segment", {})

        return {
            "stop_count": stats.get("stop_count", 0),
            "transition_count": stats.get("transition_count", 0),
            "work_count": stats.get("work_count", 0),
            "abnormal_count": stats.get("abnormal_count", 0),
            "stop_total_min": round(stats.get("stop_total_min", 0), 1),
            "transition_total_min": round(stats.get("transition_total_min", 0), 1),
            "work_total_min": round(stats.get("work_total_min", 0), 1),
            "abnormal_total_min": round(stats.get("abnormal_total_min", 0), 1),
            "geology_has": geo_summary.get("has_geology", False),
            "geology_high_risk_segment_count": geo_summary.get("high_risk_segment_count", 0),
            "geology_multi_source_segment_count": geo_summary.get("multi_source_segment_count", 0),
        }
    except Exception as e:
        print(f"[Summary Error] {e}")
        return {
            "stop_count": 0,
            "transition_count": 0,
            "work_count": 0,
            "abnormal_count": 0,
            "stop_total_min": 0,
            "transition_total_min": 0,
            "work_total_min": 0,
            "abnormal_total_min": 0,
            "geology_has": False,
            "geology_high_risk_segment_count": 0,
            "geology_multi_source_segment_count": 0,
        }


@app.get("/api/tbm/state")
def state_api(date: Optional[str] = None):
    try:
        df = get_df_by_date(date) if date else get_latest_df()
        result = analyze_tbm_data(df)

        segments = []
        for state, pairs in result["state_segments"].items():
            label_text = result["state_labels"].get(int(state), f"施工状态 {int(state)}")
            for s, e in pairs:
                segments.append({
                    "label": int(state),
                    "label_text": label_text,
                    "start": s.strftime("%H:%M:%S"),
                    "end": e.strftime("%H:%M:%S"),
                    "duration": (e - s).total_seconds(),
                })

        efficiency = []
        if not result["eff_df"].empty:
            efficiency = result["eff_df"].to_dict(orient="records")

        return {
            "segments": serialize_for_json(segments),
            "efficiency": serialize_for_json(efficiency),
            "state_labels": serialize_for_json(result["state_labels"]),
            "state_stats": serialize_for_json(result["state_stats"]),
            "valid_samples": result["llm_summary"]["有效状态样本数"],
            "state_config": serialize_for_json(result["llm_summary"]["状态识别配置"]),
        }

    except Exception as e:
        print(f"[State API Error] {e}")
        return {
            "segments": [],
            "efficiency": [],
            "state_labels": {},
            "state_stats": {},
            "valid_samples": 0,
            "state_config": {},
        }


@app.get("/api/tbm/gas")
def gas_api(date: Optional[str] = None):
    try:
        df = get_df_by_date(date) if date else get_latest_df()
        result = analyze_tbm_data(df)
        return serialize_for_json(result["gas_stats"])
    except Exception as e:
        print(f"[Gas API Error] {e}")
        return {}


@app.get("/api/tbm/geology")
def geology_api(date: Optional[str] = None):
    try:
        df = get_df_by_date(date) if date else get_latest_df()
        result = analyze_tbm_data(df)

        segment_df = result.get("segment_df", pd.DataFrame())
        typical_df = result.get("typical_segments_df", pd.DataFrame())

        preferred_cols = [
            "segment",
            "segment_start_first",
            "segment_end_first",
            "risk_mode",
            "risk_score_max",
            "active_source_count_max",
            "hazard_mode",
            "fused_grade_mode",
            "推进速度_mean",
            "推进速度_std",
            "推力_mean",
            "刀盘扭矩_mean",
            "efficiency",
            "interpretation",
        ]

        if not segment_df.empty:
            keep_cols = [c for c in preferred_cols if c in segment_df.columns]
            if keep_cols:
                segment_df = segment_df[keep_cols].copy()
            if "segment_start_first" in segment_df.columns:
                segment_df = segment_df.sort_values("segment_start_first").reset_index(drop=True)

        if not typical_df.empty:
            keep_cols2 = [c for c in preferred_cols if c in typical_df.columns]
            if keep_cols2:
                typical_df = typical_df[keep_cols2].copy()
            if "segment_start_first" in typical_df.columns:
                typical_df = typical_df.sort_values("segment_start_first").reset_index(drop=True)

        return {
            "record_summary": serialize_for_json(result["geo_summary_record"]),
            "segment_summary": serialize_for_json(result["geo_summary_segment"]),
            "segment_table": serialize_for_json(
                segment_df.to_dict(orient="records") if not segment_df.empty else []
            ),
            "typical_segments": serialize_for_json(
                typical_df.to_dict(orient="records") if not typical_df.empty else []
            )
        }

    except Exception as e:
        print(f"[Geology API Error] {e}")
        return {
            "record_summary": {"has_geology": False},
            "segment_summary": {
                "has_geology": False,
                "summary_text": "地质融合分析不可用。"
            },
            "segment_table": [],
            "typical_segments": []
        }


@app.post("/api/tbm/report_by_time")
def generate_report_by_time(req: TimeWindowRequest):
    try:
        start = req.start_time.replace("T", " ")
        end = req.end_time.replace("T", " ")
        date = start.split(" ")[0]

        df_day = get_df_by_date(date)
        df = load_df_by_time(df_day, start, end)

        if df.empty:
            return {"report": "⚠️ 该时间段无数据"}

        result = analyze_tbm_data(df)

        prompt = build_prompt_timewindow(
            start_time=start,
            end_time=end,
            seg_text=result["seg_text"],
            stats_text=result["stats_text"],
            state_text=result["state_text"],
            eff_text=result["eff_text"],
            state_stats_text=result["state_stats_text"],
            gas_text=result["gas_text"],
            geo_text=result["geo_text"],
            llm_summary=result["llm_summary"]
        )

        return {"report": call_llm(prompt)}

    except Exception as e:
        return {"report": f"❌ 出错：{e}"}


@app.get("/api/tbm/risk_profile")
def risk_profile_api(date: Optional[str] = None):
    try:
        df = get_df_by_date(date) if date else get_latest_df()
        result = analyze_tbm_data(df)

        risk_profile = build_risk_profile(result["df_geo"])
        speed_profile = build_speed_profile(result["df_geo"])

        return {
            "date": date,
            "risk_profile": risk_profile,
            "speed_profile": speed_profile
        }

    except Exception as e:
        print(f"[Risk Profile API Error] {e}")
        return {
            "date": date,
            "risk_profile": {
                "has_data": False,
                "profile": [],
                "high_segments": [],
                "message": str(e)
            },
            "speed_profile": []
        }


# 启动：
# uvicorn app:app --reload --port 8000