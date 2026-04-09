# hsp_parser.py
from pathlib import Path
from parsers.tsp_parser import parse_tsp_pdf

def parse_hsp_pdf(pdf_path: Path):
    records = parse_tsp_pdf(pdf_path)
    
    # 修改类型
    for r in records:
        r.source_type = "sonic"
    
    return records