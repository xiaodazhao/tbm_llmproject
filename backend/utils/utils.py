# utils.py
import re
from typing import Optional, Any

def mileage_to_num(mileage: str) -> Optional[float]:
    if mileage is None:
        return None
    mileage = str(mileage).strip().replace(" ", "")
    m = re.search(r"DyK(\d+)\+(\d+(?:\.\d+)?)", mileage)
    if not m:
        return None
    return float(m.group(1)) * 1000 + float(m.group(2))


def num_to_mileage(num: float) -> str:
    km = int(num // 1000)
    m = num - km * 1000
    return f"DyK{km}+{m:.1f}"


def safe_float(v: Any):
    try:
        return float(v)
    except:
        return None


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")