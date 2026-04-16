# test_tsp_parser.py
from pathlib import Path
import json
from collections import Counter

from parsers.tsp_parser import parse_tsp_pdf

pdf_path = Path(r"C:\Users\22923\Desktop\伯舒拉岭_plc_超报数据\数据\自动提取的超报报告_PDF汇总\tsp\DyK1013+124.20_伯舒拉岭隧道进口右线地震波反射法超前地质预报报告(DyK1013+080.2～DyK1013+200.2)-20230603.pdf")
records = parse_tsp_pdf(pdf_path)

print(f"共解析出 {len(records)} 条记录\n")

# 先统计类型
level_counter = Counter(r.source_level for r in records)
print("source_level 统计：")
for k, v in level_counter.items():
    print(f"  {k}: {v}")
print()

# 再打印全部
for i, r in enumerate(records, start=1):
    print("=" * 80)
    print(f"[{i}] evidence_id: {r.evidence_id}")
    print(f"source_type: {r.source_type}")
    print(f"source_level: {r.source_level}")
    print(f"report_id: {r.report_id}")
    print(f"start_num: {r.start_num}")
    print(f"end_num: {r.end_num}")
    print(f"face_num: {r.face_num}")
    print(f"next_forecast_num: {r.next_forecast_num}")
    print(f"confidence: {r.confidence}")

    attrs = r.attrs()
    print("attrs:")
    print(json.dumps(attrs, ensure_ascii=False, indent=2))

    print("raw_text:")
    print((r.raw_text or "")[:500])
    print()