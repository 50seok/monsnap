"""팔레트 견인 실패(probe_palette 결과) 대응안 검증 — 결정론적 색상 후처리.

계기: probe_palette_strength07.png 실측 — 물/페어리 타입은 파랑/분홍이 전혀
안 나타남(원형의 원래 초록·갈색이 지배적). 프롬프트만으로는 strength 0.7에서
한계로 판단, 생성 후 HSV 색조 이동으로 타입 색을 강제하는 안을 GPU 없이
(이미 생성된 이미지 재사용) 저비용 검증한다.

방법: 채도 있는 픽셀(배경·라인아트 제외)만 목표 색조로 이동, 명도·질감(음영)은
보존 — 셀 셰이딩 특유의 밝기 그라데이션이 살아있으면 입체감이 유지된다.
"""
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "dataset" / "review" / "probe_palette_strength07.png"
CELL, HEADER = 320, 30

# 타입별 목표 색조(0~1) — TYPES 팔레트 설명을 대표 색상 하나로 근사
TARGET_HUE = {
    "불": 20 / 360, "물": 200 / 360, "풀": 100 / 360,
    "전기": 48 / 360, "페어리": 330 / 360,
}
ORDER = ["불", "물", "풀", "전기", "페어리"]
SAT_THRESHOLD = 0.15  # 이 미만은 배경/라인아트로 보고 색조 이동 안 함


def recolor(img: Image.Image, target_hue: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*px) for px in arr.reshape(-1, 3)]).reshape(arr.shape)
    mask = hsv[..., 1] > SAT_THRESHOLD
    hsv[mask, 0] = target_hue
    hsv[mask, 1] = np.clip(hsv[mask, 1] * 1.15, 0, 1)  # 채도 살짝 부스트(선명한 발색)
    rgb = np.array([colorsys.hsv_to_rgb(*px) for px in hsv.reshape(-1, 3)]).reshape(arr.shape)
    return Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))


grid = Image.open(GRID)
before = [grid.crop((CELL * (i + 1), HEADER, CELL * (i + 2), HEADER + CELL)) for i in range(5)]
after = [recolor(img, TARGET_HUE[t]) for img, t in zip(before, ORDER)]

out = Image.new("RGB", (CELL * 5, CELL * 2 + HEADER * 2), "white")
d = ImageDraw.Draw(out)
d.text((4, 4), "원본(strength 0.7)", fill="black")
d.text((4, CELL + HEADER + 4), "색조 후처리 적용", fill="black")
for i, (b, a, t) in enumerate(zip(before, after, ORDER)):
    out.paste(b, (i * CELL, HEADER))
    out.paste(a, (i * CELL, CELL + HEADER * 2))
    d.text((i * CELL + 4, HEADER + 2), t, fill="black")

out_path = ROOT / "dataset" / "review" / "probe_recolor_compare.png"
out.save(out_path)
print(f"saved: {out_path}")
