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
    name = pdf_path.stem.strip()

    name = re.sub(r"_[0-9]+$", "", name)
    name = re.sub(r"\s+", "", name)
    name = re.sub(r"-+$", "", name)
    name = re.sub(r"_+", "_", name)

    name = name.replace("（", "(").replace("）", ")")

    return name.lower()


# ==============================
# 2️⃣ 文件过滤
# ==============================
def is_valid_pdf(pdf_path: Path) -> bool:
    if pdf_path.suffix.lower() != ".pdf":
        return False

    name = pdf_path.name

    # ❗剔除左线
    if "左线" in name:
        return False

    return True


# ==============================
# 3️⃣ 去重
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
# 4️⃣ 解析函数
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
        log_path = LOG_DIR / f"{source_name}_fail_log.txt"
        log_path.write_text("\n".join(fail_logs), encoding="utf-8")

    return records


# ==============================
# 5️⃣ 主流程（无 drill）
# ==============================
def main():
    all_records: List[EvidenceRecord] = []

    # ✅ 保留三类
    all_records.extend(parse_folder(TSP_DIR, parse_tsp_pdf, "tsp"))
    all_records.extend(parse_folder(HSP_DIR, parse_hsp_pdf, "sonic"))
    all_records.extend(parse_folder(SKETCH_DIR, parse_sketch_pdf, "sketch"))

    # ❌ 已移除 drill
    print("\n⚠️ 已禁用超前水平钻（drill），原因：图片PDF无法可靠解析")

    print("\n==============================")
    print(f"原始记录数: {len(all_records)}")

    df = records_to_dataframe(all_records)

    # ==============================
    # 6️⃣ 清洗
    # ==============================
    df = df[df["start_num"] <= df["end_num"]].copy()

    df["seg_len"] = df["end_num"] - df["start_num"]

    df = df[
        (df["seg_len"] >= 0) &
        (df["seg_len"] <= 300)
    ].copy()

    df.drop(columns=["seg_len"], inplace=True)

    # ==============================
    # 7️⃣ 保存
    # ==============================
    out_csv = DB_DIR / "evidence_db.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print(f"清洗后记录数: {len(df)}")

    print("\n===== 各类型统计 =====")
    print(df["source_type"].value_counts())

    print(f"\n证据库已保存: {out_csv}")
    print("==============================")


if __name__ == "__main__":
    main()