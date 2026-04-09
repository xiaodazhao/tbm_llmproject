import pandas as pd
from pathlib import Path

from analysis.dataprocess import (
    load_and_process,
    segments_to_text,
    compute_stats,
    stats_to_text
)

from analysis.gas_analysis import (
    compute_gas_stats,
    gas_stats_to_text
)

from analysis.excavation_state import (
    detect_excavation_state,
    excavation_state_segments,
    explain_excavation_states,
    excavation_state_to_text,
    excavation_state_efficiency,
    efficiency_to_text,
    excavation_state_stats,
    excavation_state_stats_to_text
)

from llm.prompt_builder import build_prompt
from llm.llm_api import call_llm


# =========================================================
# 0. 路径配置 (已修复为 Mac 兼容的路径)
# =========================================================
# 自动获取当前脚本所在的目录 (Backend文件夹)
from config import DATA_DIR
csv_path = DATA_DIR / "tbm_data (63).csv"

# 💡 如果运行依然提示找不到文件，请取消下面这行的注释，强制使用你电脑的绝对路径：
# csv_path = Path("/Users/zhaoxiaoda/Desktop/LLM_20260402/LLM_20251219/Backend/tbm_data (87).csv")


# =========================================================
# 1. 数据读取
# =========================================================
def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # 兼容性处理：如果列名叫 time，重命名一下
    if "运行时间-time" not in df.columns and "time" in df.columns:
        df.rename(columns={"time": "运行时间-time"}, inplace=True)
        
    if "运行时间-time" not in df.columns:
        raise ValueError("缺少时间列：运行时间-time")

    df["运行时间-time"] = pd.to_datetime(df["运行时间-time"], errors="coerce")
    df = df.dropna(subset=["运行时间-time"]).sort_values("运行时间-time").reset_index(drop=True)

    return df


# =========================================================
# 2. 估计有效样本数（用于自适应状态识别）
# =========================================================
def estimate_valid_samples(
    df: pd.DataFrame,
    feature_cols=("推力", "刀盘扭矩", "刀盘实际转速", "推进速度")
) -> int:
    valid_mask = pd.Series(True, index=df.index)

    # 优先用掘进状态筛工作样本
    if "掘进状态" in df.columns:
        valid_mask &= (pd.to_numeric(df["掘进状态"], errors="coerce").fillna(0) != 0)
    else:
        temp_mask = pd.Series(False, index=df.index)
        if "推力" in df.columns:
            temp_mask |= (pd.to_numeric(df["推力"], errors="coerce").fillna(0).abs() > 1e-8)
        if "推进速度" in df.columns:
            temp_mask |= (pd.to_numeric(df["推进速度"], errors="coerce").fillna(0).abs() > 1e-8)
        valid_mask &= temp_mask

    # 再要求状态特征列非空
    for col in feature_cols:
        if col in df.columns:
            valid_mask &= pd.to_numeric(df[col], errors="coerce").notna()

    return int(valid_mask.sum())


# =========================================================
# 3. 自适应参数选择
# =========================================================
def choose_state_params(n_valid: int):
    """
    根据有效样本数，自适应决定聚类状态数和最短状态段时长
    """
    if n_valid < 5:
        return {"do_cluster": False, "n_states": 0, "min_duration_sec": 0}
    elif n_valid < 10:
        return {"do_cluster": True, "n_states": 2, "min_duration_sec": 0}
    elif n_valid < 30:
        return {"do_cluster": True, "n_states": 3, "min_duration_sec": 20}
    else:
        return {"do_cluster": True, "n_states": 4, "min_duration_sec": 60}


# =========================================================
# 4. 主流程
# =========================================================
def main():
    # -----------------------------------------------------
    # 4.1 读取数据
    # -----------------------------------------------------
    print(f"正在读取文件: {csv_path}")
    df = load_data(csv_path)

    print("=" * 80)
    print("【数据读取完成】")
    print(f"数据行数：{len(df)}")
    print(f"数据列数：{len(df.columns)}")
    print("前20个字段：")
    print(df.columns[:20].tolist())

    # -----------------------------------------------------
    # 4.2 基础工况分析（多工况版）
    # -----------------------------------------------------
    # 💡 修复：直接传入清洗后的 df，而不是传路径，避免重复读取导致时间格式丢失
    segments = load_and_process(df)  
    seg_text = segments_to_text(segments)

    stats = compute_stats(segments)
    stats_text = stats_to_text(stats)

    # -----------------------------------------------------
    # 4.3 隐含施工状态识别（自适应版）
    # -----------------------------------------------------
    state_features = ("推力", "刀盘扭矩", "刀盘实际转速", "推进速度")
    n_valid = estimate_valid_samples(df, state_features)
    state_cfg = choose_state_params(n_valid)

    print("\n" + "=" * 80)
    print("【状态识别自适应参数】")
    print(f"有效样本数：{n_valid}")
    print(f"是否进行聚类：{state_cfg['do_cluster']}")
    print(f"自动选择 n_states = {state_cfg['n_states']}")
    print(f"自动选择 min_duration_sec = {state_cfg['min_duration_sec']}")

    state_labels = {}
    state_segments = {}
    state_text = "当日有效工作样本过少，未进行隐含施工状态识别。"
    eff_df = pd.DataFrame()
    eff_text = "当日有效工作样本过少，无法计算施工状态效率统计。"
    state_stats = {}
    state_stats_text = "当日有效工作样本过少，无施工状态统计结果。"

    if state_cfg["do_cluster"]:
        df_state, labels = detect_excavation_state(
            df.copy(),
            features=state_features,
            n_states=state_cfg["n_states"]
        )

        print("\n" + "=" * 80)
        print("【状态识别调试信息】")
        for col in state_features:
            print(f"{col} ->", col in df_state.columns)

        if "state_id" in df_state.columns:
            print("state_id 非空数量：", int(df_state["state_id"].notna().sum()))
            print("state_id 分布：")
            print(df_state["state_id"].value_counts(dropna=False))
        else:
            print("没有生成 state_id 列")

        state_labels = explain_excavation_states(df_state)

        state_segments = excavation_state_segments(
            df_state,
            min_duration_sec=state_cfg["min_duration_sec"]
        )
        state_text = excavation_state_to_text(state_segments, state_labels)

        eff_df = excavation_state_efficiency(df_state)
        eff_text = efficiency_to_text(eff_df, state_labels)

        state_stats = excavation_state_stats(df_state, state_segments)
        state_stats_text = excavation_state_stats_to_text(state_stats, state_labels)
    else:
        df_state = df.copy()

    # -----------------------------------------------------
    # 4.4 气体分析
    # -----------------------------------------------------
    gas_stats = compute_gas_stats(df, df_state=df_state)
    gas_text = gas_stats_to_text(gas_stats)

    # -----------------------------------------------------
    # 4.5 给 LLM 的结构化摘要
    # -----------------------------------------------------
    llm_summary = {
        "基础工况统计": stats,
        "施工状态标签": state_labels,
        "施工状态统计": state_stats,
        "施工状态效率表": eff_df.to_dict(orient="index") if not eff_df.empty else {},
        "气体统计": gas_stats,
        "有效状态样本数": n_valid,
        "状态识别配置": state_cfg
    }

    # -----------------------------------------------------
    # 4.6 打印结果
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("【给 LLM 的结构化摘要】")
    print(llm_summary)

    # -----------------------------------------------------
    # 4.7 构造 Prompt 并调用 LLM
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("🚀 正在组装 Prompt 并请求大模型 API，请稍候...")
    
    prompt = build_prompt(
        seg_text=seg_text,
        stats_text=stats_text,
        state_text=state_text,
        eff_text=eff_text,
        state_stats_text=state_stats_text,
        gas_text=gas_text,
        geo_text="单文件独立测试阶段，暂无地质预警数据输入。",  # 💡 修复：补齐缺失的参数
        llm_summary=llm_summary
    )

    report = call_llm(prompt)

    print("\n" + "=" * 80)
    print("【🎉 LLM 最终生成报告 🎉】")
    print(report)


# =========================================================
# 5. 程序入口
# =========================================================
if __name__ == "__main__":
    main()