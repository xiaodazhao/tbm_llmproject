# db.py
from dataclasses import asdict
from pathlib import Path
from typing import List
import pandas as pd

# 修改这里：从 schemas 文件夹导入 schemas.py 里的 EvidenceRecord
from schemas.schemas import EvidenceRecord


def records_to_dataframe(records: List[EvidenceRecord]) -> pd.DataFrame:
    """
    把 EvidenceRecord 列表转成 pandas DataFrame
    """
    return pd.DataFrame([asdict(r) for r in records])


def save_evidence_db(records: List[EvidenceRecord], out_csv: Path) -> None:
    """
    保存证据库到 CSV
    """
    df = records_to_dataframe(records)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def load_evidence_db(csv_path: Path) -> pd.DataFrame:
    """
    从 CSV 读取证据库
    """
    return pd.read_csv(csv_path)