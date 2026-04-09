# debug_runner.py
import traceback
import pandas as pd

from utils.io_utils import check_data_environment, load_latest_csv, load_evidence
from app import analyze_tbm_data
from llm.prompt_builder import build_prompt


def print_title(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_subtitle(title: str):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def preview_text(text: str, max_len: int = 2500) -> str:
    if not text:
        return ""
    return text[:max_len] + ("\n...(已截断)" if len(text) > max_len else "")


def preview_df(df: pd.DataFrame, name: str, n: int = 5):
    if df is None:
        print(f"[{name}] = None")
        return
    print(f"[{name}] shape = {df.shape}")
    if df.empty:
        print(f"[{name}] 为空")
        return
    print(df.head(n).to_string(index=False))


def run_debug_without_llm():
    print_title("TBM 本地调试入口（调用 analyze_tbm_data，不调用 LLM API）")

    # 0. 环境检查
    print_subtitle("步骤 0：环境检查")
    env_info = check_data_environment()
    for k, v in env_info.items():
        print(f"{k}: {v}")

    # 1. 读取数据
    print_subtitle("步骤 1：读取最新 CSV")
    csv_path, df = load_latest_csv()
    print(f"当前 CSV: {csv_path.name}")
    print(f"CSV shape: {df.shape}")
    print(f"列数: {len(df.columns)}")
    print(f"前 20 列: {df.columns[:20].tolist()}")
    print(f"时间范围: {df['运行时间-time'].min()} ~ {df['运行时间-time'].max()}")

    # 2. 检查 evidence_db
    print_subtitle("步骤 2：读取 evidence_db.csv")
    evidence_df = load_evidence()
    print(f"evidence_db shape: {evidence_df.shape}")
    print(f"evidence_db 列: {evidence_df.columns.tolist()}")

    # 3. 跑总流程
    print_subtitle("步骤 3：调用 analyze_tbm_data(df)")
    result = analyze_tbm_data(df)
    print("analyze_tbm_data 执行完成。")
    print(f"result keys: {list(result.keys())}")

    # 4. 看融合后的 df_geo
    print_subtitle("步骤 4：检查地质融合结果")
    df_geo = result.get("df_geo", pd.DataFrame())
    geo_cols = [
        "chainage", "coverage", "risk", "risk_score", "hazard",
        "active_source_count", "active_sources", "fused_grade", "uncertainty"
    ]
    geo_cols = [c for c in geo_cols if c in df_geo.columns]
    if geo_cols:
        temp = df_geo[geo_cols].copy()
        if "chainage" in temp.columns:
            temp = temp.drop_duplicates(subset=["chainage"])
        preview_df(temp, "融合后地质字段", n=10)
    else:
        print("未发现地质扩展字段。")

    # 5. 看基础工况
    print_subtitle("步骤 5：基础工况结果")
    print("[stats_text]")
    print(result.get("stats_text", ""))

    print("\n[seg_text 预览]")
    print(preview_text(result.get("seg_text", "")))

    # 6. 看状态识别
    print_subtitle("步骤 6：施工状态识别结果")
    print("[state_labels]")
    print(result.get("state_labels", {}))

    print("\n[state_text 预览]")
    print(preview_text(result.get("state_text", "")))

    print("\n[eff_text]")
    print(result.get("eff_text", ""))

    print("\n[state_stats_text]")
    print(result.get("state_stats_text", ""))

    # 7. 看地质区段分析
    print_subtitle("步骤 7：地质区段分析")
    print("[geo_text]")
    print(result.get("geo_text", ""))

    typical_df = result.get("typical_segments_df", pd.DataFrame())
    preview_df(typical_df, "典型区段", n=10)

    # 8. 看前方风险
    print_subtitle("步骤 8：前方风险提示")
    print("[forward_risk_summary]")
    print(result.get("forward_risk_summary", {}))

    print("\n[forward_risk_text]")
    print(result.get("forward_risk_text", ""))

    # 9. 看气体分析
    print_subtitle("步骤 9：气体分析")
    print("[gas_text]")
    print(result.get("gas_text", ""))

    # 10. 组装 Prompt（不调 API）
    print_subtitle("步骤 10：生成 Prompt（不调用 LLM API）")
    prompt = build_prompt(
        seg_text=result.get("seg_text", ""),
        stats_text=result.get("stats_text", ""),
        state_text=result.get("state_text", ""),
        eff_text=result.get("eff_text", ""),
        state_stats_text=result.get("state_stats_text", ""),
        gas_text=result.get("gas_text", ""),
        geo_text=result.get("geo_text", ""),
        llm_summary=result.get("llm_summary", {}),
    )

    print(f"Prompt 长度: {len(prompt)}")
    print("\n[Prompt 前 2500 字]")
    print(preview_text(prompt, max_len=2500))

    print_title("调试结束：主流程正常，已成功生成 Prompt（未调用 API）")
    return result, prompt


if __name__ == "__main__":
    try:
        run_debug_without_llm()
    except Exception as e:
        print_title("程序报错")
        print(str(e))
        print("\n完整报错栈：")
        traceback.print_exc()