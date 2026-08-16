"""원형 도감(dataset/proto_dex) 45종 전체를 이름표 붙여 한 장에 — 어떤 종이
별로인지 짚기 위한 대조표. curate.py의 contact_sheets()와 같은 목적, 원형
도감용으로 재사용. GPU 불필요.

사용법: python tools/contact_sheet_dex.py
출력:   dataset/review/proto_dex_sheet.png
"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DEX = ROOT / "dataset" / "proto_dex"
OUT = ROOT / "dataset" / "review" / "proto_dex_sheet.png"

species = sorted(p for p in DEX.glob("*.png"))
cols, cell, label_h = 8, 180, 22
rows = -(-len(species) // cols)

sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
d = ImageDraw.Draw(sheet)
for i, p in enumerate(species):
    x, y = (i % cols) * cell, (i // cols) * (cell + label_h)
    img = Image.open(p).convert("RGB").resize((cell, cell))
    sheet.paste(img, (x, y))
    d.rectangle([x, y, x + cell, y + label_h], fill="white")
    d.text((x + 2, y + 2), p.stem, fill="black")

OUT.parent.mkdir(exist_ok=True)
sheet.save(OUT)
print(f"{len(species)}종 -> {OUT}")
