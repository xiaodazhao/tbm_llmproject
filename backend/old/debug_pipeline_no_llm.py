# debug_pipeline_no_llm.py
from pathlib import Path
import glob
import os
import traceback
import pandas as pd

from config import DATA_DIR, EVIDENCE_DB_PATH

from analysis.dataprocess import (
    load_and_process,
    segments_to_text,
    compute_stats,
    stats_to_text,
)

from analysis.excavation_state import (
    detect_excavation_state,
    excavation_state_segments,
    explain_excavation_states,
    excavation_state_to_text,
    excavation_state_efficiency,
    excavation_state_stats,
    excavation_state_stats_to_text,
)

from analysis.gas_analysis import (
    compute_gas_stats,
    gas_stats_to_text,
)

from analysis.forward_risk_advisor import (
    generate_forward_risk_summary,
    forward_risk_to_text,
)

from geology.geology_fusion_backend import (
    attach_geology_labels,
    load_evidence_db,
)

from geology.geology_summary import (
    summarize_geology_record_level,
    summarize_geology_segment_level,
    geology_summary_to_text,
)

from geology.segment_analysis import (
    run_segment_analysis,
    build_typical_segments_table,
)

from llm.prompt_builder import build_prompt


# =========================
# 配置
# =========================
STATE_FEATURES = ("推力", "刀盘扭矩", "刀盘实际转速", "推进速度")


# =========================
# 工具函数
# =========================
def print_title(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_subtitle(title: str):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def safe_preview_df(df: pd.DataFrame, name: str, n: int = 5):
    if df is None:
        print(f"[{name}] = None")
        return
    print(f"[{name}] shape = {df.shape}")
    if df.empty:
        print(f"[{name}] 为空")
        return
    print(df.head(n).to_string(index=False))


def get_latest_csv():
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"在 {DATA_DIR} 下没有找到任何 csv 文件")
    latest_file = max(files)
    return Path(latest_file)


def load_data(file_path: Path) -> pd.DataFrame:
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


def estimate_valid_samples(df: pd.DataFrame, feature_cols=STATE_FEATURES) -> int:
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
        return {"do_cluster": False, "n_states": 0, "min_duration_sec": 0}
    elif n_valid < 10:
        return {"do_cluster": True, "n_states": 2, "min_duration_sec": 0}
    elif n_valid < 30:
        return {"do_cluster": True, "n_states": 3, "min_duration_sec": 20}
    else:
        return {"do_cluster": True, "n_states": 4, "min_duration_sec": 60}


def semantic_efficiency_to_text(eff_df: pd.DataFrame) -> str:
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
# 主流程
# =========================
def main():
    print_title("TBM 本地全流程调试（不调用 LLM API）")

    # 0. 环境检查
    print_subtitle("步骤 0：环境与路径检查")
    print(f"DATA_DIR           : {DATA_DIR}")
    print(f"EVIDENCE_DB_PATH   : {EVIDENCE_DB_PATH}")
    print(f"DATA_DIR exists    : {DATA_DIR.exists()}")
    print(f"EVIDENCE_DB exists : {EVIDENCE_DB_PATH.exists()}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR 不存在：{DATA_DIR}")

    latest_csv = get_latest_csv()
    print(f"最新 CSV 文件      : {latest_csv.name}")

    # 1. 读取原始 CSV
    print_subtitle("步骤 1：读取原始 CSV")
    df = load_data(latest_csv)
    print(f"原始数据 shape: {df.shape}")
    print(f"列数: {len(df.columns)}")
    print(f"前 20 列: {df.columns[:20].tolist()}")
    print(f"时间范围: {df['运行时间-time'].min()} ~ {df['运行时间-time'].max()}")
    safe_preview_df(df, "原始数据预览", n=3)

    # 2. 读取 evidence_db
    print_subtitle("步骤 2：读取 evidence_db.csv")
    if not EVIDENCE_DB_PATH.exists():
        raise FileNotFoundError(
            f"找不到 evidence_db.csv：{EVIDENCE_DB_PATH}\n"
            f"请先运行：python scripts/build_evidence_db.py"
        )

    evidence_df = load_evidence_db(EVIDENCE_DB_PATH)
    print(f"evidence_db shape: {evidence_df.shape}")
    print(f"evidence_db 列: {evidence_df.columns.tolist()}")
    safe_preview_df(evidence_df, "evidence_db预览", n=5)

    # 3. 地质融合
    print_subtitle("步骤 3：PLC + 地质证据 融合")
    df_geo = attach_geology_labels(df, evidence_df)
    print(f"融合后 shape: {df_geo.shape}")

    geo_cols = [
        "chainage", "coverage", "risk", "risk_score", "hazard",
        "active_source_count", "active_sources", "fused_grade", "uncertainty"
    ]
    geo_cols = [c for c in geo_cols if c in df_geo.columns]
    if geo_cols:
        unique_geo = df_geo[geo_cols].drop_duplicates(subset=["chainage"]) if "chainage" in df_geo.columns else df_geo[geo_cols]
        safe_preview_df(unique_geo, "融合后地质字段预览", n=10)
    else:
        print("未生成地质扩展字段，请检查 chainage / evidence_db 是否正常。")

    # 4. 地质摘要
    print_subtitle("步骤 4：地质摘要与区段分析")
    geo_summary_record = summarize_geology_record_level(df_geo)
    print("[记录级地质摘要]")
    print(geo_summary_record)

    segment_df = run_segment_analysis(df_geo, segment_length=10)
    print(f"\n区段分析结果 shape: {segment_df.shape}")
    safe_preview_df(segment_df, "区段分析表", n=10)

    geo_summary_segment = summarize_geology_segment_level(segment_df)
    geo_text = geology_summary_to_text(geo_summary_segment)

    print("\n[区段级地质摘要文本]")
    print(geo_text)

    typical_segments_df = build_typical_segments_table(segment_df, top_n=10)
    print("\n[典型区段 Top10]")
    safe_preview_df(typical_segments_df, "典型区段", n=10)

    # 5. 前方风险提示
    print_subtitle("步骤 5：前方风险提示")
    forward_risk_summary = generate_forward_risk_summary(
        df_plc=df_geo,
        evidence_df=evidence_df,
        lookahead_m=30
    )
    forward_risk_text = forward_risk_to_text(forward_risk_summary)

    print("[前方风险结构化摘要]")
    print(forward_risk_summary)

    print("\n[前方风险文本]")
    print(forward_risk_text)

    # 6. 基础工况分析
    print_subtitle("步骤 6：基础工况分析")
    segments = load_and_process(df_geo)
    seg_text = segments_to_text(segments)

    stats = compute_stats(segments)
    stats_text = stats_to_text(stats)

    print(f"工况段数量: {len(segments)}")
    print("\n[基础工况分段文本]")
    print(seg_text[:3000] + ("\n...(已截断)" if len(seg_text) > 3000 else ""))

    print("\n[基础工况统计文本]")
    print(stats_text)

    # 7. 隐含施工状态识别
    print_subtitle("步骤 7：隐含施工状态识别")
    n_valid = estimate_valid_samples(df_geo, STATE_FEATURES)
    state_cfg = choose_state_params(n_valid)

    print(f"有效状态样本数: {n_valid}")
    print(f"状态识别配置: {state_cfg}")

    state_labels = {}
    state_segments = {}
    state_text = "当日有效工作样本过少，未进行隐含施工状态识别。"
    eff_df = pd.DataFrame()
    eff_text = "当日有效工作样本过少，无法计算施工状态效率统计。"
    state_stats = {}
    state_stats_text = "当日有效工作样本过少，无施工状态统计结果。"
    df_state = df_geo.copy()

    if state_cfg["do_cluster"]:
        df_state, _ = detect_excavation_state(
            df_geo.copy(),
            features=STATE_FEATURES,
            n_states=state_cfg["n_states"]
        )

        state_labels = explain_excavation_states(df_state)
        state_segments = excavation_state_segments(
            df_state,
            min_duration_sec=state_cfg["min_duration_sec"]
        )
        state_text = excavation_state_to_text(state_segments, state_labels)

        raw_eff_df = excavation_state_efficiency(df_state)
        if not raw_eff_df.empty:
            eff_df = raw_eff_df.reset_index().rename(columns={"state_id": "label"})
            eff_df["label_text"] = eff_df["label"].map(state_labels).fillna("未知状态")

            agg_dict = {}
            for col in ["平均推进速度", "平均推力", "平均刀盘扭矩", "平均刀盘实际转速", "平均推进给定速度"]:
                if col in eff_df.columns:
                    agg_dict[col] = "mean"

            if agg_dict:
                eff_df = eff_df.groupby("label_text", as_index=False).agg(agg_dict)
            else:
                eff_df = pd.DataFrame()

            eff_text = semantic_efficiency_to_text(eff_df)

        state_stats = excavation_state_stats(df_state, state_segments)
        state_stats_text = excavation_state_stats_to_text(state_stats, state_labels)

        print("[state_labels]")
        print(state_labels)

        print("\n[state_id 分布]")
        if "state_id" in df_state.columns:
            print(df_state["state_id"].value_counts(dropna=False))

        print("\n[施工状态文本]")
        print(state_text[:3000] + ("\n...(已截断)" if len(state_text) > 3000 else ""))

        print("\n[施工状态效率文本]")
        print(eff_text)

        print("\n[施工状态统计文本]")
        print(state_stats_text)
    else:
        print("样本量不足，跳过隐含施工状态识别。")

    # 8. 气体分析
    print_subtitle("步骤 8：气体分析")
    gas_stats = compute_gas_stats(df_geo, df_state=df_state)
    gas_text = gas_stats_to_text(gas_stats)

    print("[气体统计结构]")
    print(gas_stats)

    print("\n[气体统计文本]")
    print(gas_text if gas_text.strip() else "无可用气体文本输出")

    # 9. 组装 llm_summary
    print_subtitle("步骤 9：组装 llm_summary")
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
        "状态识别配置": state_cfg,
    }

    print("llm_summary 组装成功。")
    print(f"llm_summary 顶层键: {list(llm_summary.keys())}")

    # 10. 生成 Prompt（到此结束，不调 API）
    print_subtitle("步骤 10：生成 Prompt（不调用 LLM）")
    prompt = build_prompt(
        seg_text=seg_text,
        stats_text=stats_text,
        state_text=state_text,
        eff_text=eff_text,
        state_stats_text=state_stats_text,
        gas_text=gas_text,
        geo_text=geo_text,
        llm_summary=llm_summary,
    )

    print(f"Prompt 长度: {len(prompt)} 字符")
    print("\n[Prompt 前 3000 字预览]")
    print(prompt[:50000] + ("\n...(已截断)" if len(prompt) > 50000 else ""))

    print_title("调试完成：已到 Prompt 生成步骤，未调用 LLM API")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_title("程序报错")
        print(str(e))
        print("\n完整报错栈：")
        traceback.print_exc()