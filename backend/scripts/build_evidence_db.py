# build_evidence_db.py
import sys
import os

# 把当前脚本的上一级目录(backend)加入环境变量，防止找不到包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from pathlib import Path
from typing import Dict, List
import pandas as pd

from config import TSP_DIR, HSP_DIR, SKETCH_DIR, DB_DIR, LOG_DIR
from scripts.db import records_to_dataframe
from schemas.schemas import EvidenceRecord
from parsers.tsp_parser import parse_tsp_pdf
from parsers.hsp_parser import parse_hsp_pdf
from parsers.sketch_parser import parse_sketch_pdf


# ==============================
# 1️⃣ 文件名规范化（用于去重）
# ==============================
def normalize_name_for_dedup(pdf_path: Path) -> str:
    """
    规范化文件名，用于文件级去重
    """
    name = pdf_path.stem.strip()

    # 去掉末尾 _数字
    name = re.sub(r"_[0-9]+$", "", name)

    # 去掉空白
    name = re.sub(r"\s+", "", name)

    # 去掉末尾多余 -
    name = re.sub(r"-+$", "", name)

    # 连续下划线合并
    name = re.sub(r"_+", "_", name)

    # 全角括号转半角
    name = name.replace("（", "(").replace("）", ")")

    return name.lower()


# ==============================
# 2️⃣ 文件过滤
# ==============================
def is_valid_pdf(pdf_path: Path) -> bool:
    """
    基础过滤规则
    """
    if pdf_path.suffix.lower() != ".pdf":
        return False

    name = pdf_path.name

    # 剔除左线
    if "左线" in name:
        return False

    return True


# ==============================
# 3️⃣ 收集并去重 PDF
# ==============================
def collect_unique_pdfs(folder: Path) -> List[Path]:
    seen: Dict[str, Path] = {}
    duplicates = []

    for pdf_path in sorted(folder.glob("*.pdf")):
        if not is_valid_pdf(pdf_path):
            continue

        key = normalize_name_for_dedup(pdf_path)

        if key not in seen:
            seen[key] = pdf_path
        else:
            duplicates.append((seen[key].name, pdf_path.name))

    if duplicates:
        print(f"\n[{folder.name}] 发现重复 {len(duplicates)} 组")
        for a, b in duplicates[:10]:
            print(f"  保留: {a}")
            print(f"  跳过: {b}")

    return list(seen.values())


# ==============================
# 4️⃣ 解析一个文件夹
# ==============================
def parse_folder(folder: Path, parser_func, source_name: str) -> List[EvidenceRecord]:
    records: List[EvidenceRecord] = []
    files = collect_unique_pdfs(folder)

    print(f"\n--- 📁 处理 {source_name}：{len(files)} 个 PDF ---")

    fail_logs = []

    for pdf_path in files:
        try:
            recs = parser_func(pdf_path)
            if recs:
                records.extend(recs)
                print(f"[OK] {pdf_path.name} -> {len(recs)} 条")
            else:
                print(f"[EMPTY] {pdf_path.name}")
                fail_logs.append(f"[EMPTY] {pdf_path}")
        except Exception as e:
            print(f"[FAIL] {pdf_path.name}: {e}")
            fail_logs.append(f"[FAIL] {pdf_path} :: {e}")

    if fail_logs:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{source_name}_fail_log.txt"
        log_path.write_text("\n".join(fail_logs), encoding="utf-8")

    return records


# ==============================
# 5️⃣ 数据清洗
# ==============================
def clean_evidence_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    对证据库做基本清洗：
    - 去掉关键字段缺失
    - 去掉非法区间
    - 对 segment 做长度约束
    - 保留 report_conclusion 等其他层级
    """
    if df.empty:
        return df

    # 关键字段检查
    required_cols = ["evidence_id", "start_num", "end_num"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"证据库缺少必要字段: {missing_cols}")

    # 记录级去重
    df = df.drop_duplicates(subset=["evidence_id"]).copy()

    # 去掉区间为空
    df = df.dropna(subset=["start_num", "end_num"]).copy()

    # 数值化，防御性处理
    df["start_num"] = pd.to_numeric(df["start_num"], errors="coerce")
    df["end_num"] = pd.to_numeric(df["end_num"], errors="coerce")
    df = df.dropna(subset=["start_num", "end_num"]).copy()

    # start <= end
    df = df[df["start_num"] <= df["end_num"]].copy()

    # 分层清洗
    if "source_level" in df.columns:
        seg_mask = df["source_level"] == "segment"
        df_seg = df[seg_mask].copy()
        df_other = df[~seg_mask].copy()

        # 只对 segment 做长度约束
        if not df_seg.empty:
            df_seg["seg_len"] = df_seg["end_num"] - df_seg["start_num"]
            df_seg = df_seg[
                (df_seg["seg_len"] >= 0) &
                (df_seg["seg_len"] <= 300)
            ].copy()
            df_seg.drop(columns=["seg_len"], inplace=True, errors="ignore")

        df = pd.concat([df_seg, df_other], ignore_index=True)
    else:
        # 如果没有 source_level，就统一做长度清洗
        df["seg_len"] = df["end_num"] - df["start_num"]
        df = df[
            (df["seg_len"] >= 0) &
            (df["seg_len"] <= 300)
        ].copy()
        df.drop(columns=["seg_len"], inplace=True, errors="ignore")

    # 排序，方便后续查看
    sort_cols = [c for c in ["source_type", "report_id", "start_num", "end_num"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


# ==============================
# 6️⃣ 主流程
# ==============================
def main():
    all_records: List[EvidenceRecord] = []

    # 三类证据
    all_records.extend(parse_folder(TSP_DIR, parse_tsp_pdf, "tsp"))
    all_records.extend(parse_folder(HSP_DIR, parse_hsp_pdf, "sonic"))
    all_records.extend(parse_folder(SKETCH_DIR, parse_sketch_pdf, "sketch"))

    # drill 暂不启用
    print("\n⚠️ 已禁用超前水平钻（drill），原因：图片PDF无法可靠解析")

    print("\n==============================")
    print(f"原始记录数: {len(all_records)}")

    if not all_records:
        print("未解析到任何证据记录，终止保存。")
        return

    # 转表
    df = records_to_dataframe(all_records)

    # 清洗
    df = clean_evidence_dataframe(df)

    if df.empty:
        print("清洗后证据库为空，终止保存。")
        return

    # 保存
    DB_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = DB_DIR / "evidence_db.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print(f"清洗后记录数: {len(df)}")

    if "source_type" in df.columns:
        print("\n===== 各 source_type 统计 =====")
        print(df["source_type"].value_counts(dropna=False))

    if "source_level" in df.columns:
        print("\n===== 各 source_level 统计 =====")
        print(df["source_level"].value_counts(dropna=False))

    if {"source_type", "source_level"}.issubset(df.columns):
        print("\n===== source_type × source_level 交叉统计 =====")
        print(pd.crosstab(df["source_type"], df["source_level"]))

    print(f"\n证据库已保存: {out_csv}")
    print("==============================")


if __name__ == "__main__":
    main()