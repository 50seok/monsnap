"""dataset/cute → 학습용 metadata.jsonl 생성 (색 보존 캡션).

고정 캡션으로 학습하면 색이 '데이터셋 평균'(파스텔)으로 수렴한다 — 공개 LoRA에서
검은 머리가 금발로 나오던 원인. 이미지별 주요 색상을 캡션에 넣어 색을 프롬프트
제어 축으로 남긴다.

사용법: python tools/build_captions.py
출력:   dataset/cute/metadata.jsonl  (diffusers imagefolder 형식)
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

CUTE = Path(__file__).resolve().parent.parent / "dataset" / "cute"
TRIGGER = "cutemon"  # tools/curate.py와 동일 트리거

# 기본 색상 팔레트 (이름, RGB) — 세밀한 색 구분은 불필요, 프롬프트 어휘로 흔한 단어만
PALETTE = [
    ("red", (210, 60, 50)), ("orange", (235, 140, 45)), ("yellow", (235, 205, 80)),
    ("green", (95, 175, 90)), ("blue", (75, 120, 210)), ("purple", (150, 95, 195)),
    ("pink", (235, 150, 185)), ("brown", (135, 100, 65)), ("black", (40, 40, 40)),
    ("gray", (150, 150, 150)), ("white", (240, 240, 240)),
]


def dominant_colors(img: Image.Image, top: int = 2) -> list[str]:
    arr = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32).reshape(-1, 3)
    # 배경(순백 근처) 제외 — 몸통이 흰 경우는 순백이 아니라 음영이 섞여 살아남는다
    arr = arr[arr.min(axis=1) < 235]
    if len(arr) == 0:
        return []
    pal = np.array([c for _, c in PALETTE], dtype=np.float32)
    nearest = np.argmin(((arr[:, None] - pal[None]) ** 2).sum(-1), axis=1)
    counts = np.bincount(nearest, minlength=len(PALETTE)).astype(np.float32)
    counts /= counts.sum()
    order = counts.argsort()[::-1]
    return [PALETTE[i][0] for i in order[:top] if counts[i] >= 0.15]


def main():
    rows = []
    for png in sorted(CUTE.glob("*.png")):
        colors = dominant_colors(Image.open(png))
        color_txt = f"{' and '.join(colors)} " if colors else ""
        rows.append({
            "file_name": png.name,
            "text": f"{TRIGGER}, a cute round {color_txt}creature, "
                    "big eyes, simple shapes, white background",
        })
    out = CUTE / "metadata.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"{len(rows)} captions -> {out}")
    for r in rows[:5]:
        print(" ", r["text"])


if __name__ == "__main__":
    main()
