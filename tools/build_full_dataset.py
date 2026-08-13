"""dataset/raw 1025장 → dataset/full (v3 학습용: 흰 배경 512 + 색 캡션).

큐레이션 200장(v1·v2)보다 스타일 어휘를 넓히기 위한 전체 데이터셋.
"cute round" 같은 서술은 뺀다 — 전체 세트에는 거짓인 이미지가 많아 노이즈가 되고,
귀여움 유도는 추론 프롬프트·네거티브가 담당한다(검증된 구조). 트리거·색만 남긴다.

사용법: python tools/build_full_dataset.py
"""
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from build_captions import TRIGGER, dominant_colors  # noqa: E402

RAW = Path(__file__).resolve().parent.parent / "dataset" / "raw"
OUT = Path(__file__).resolve().parent.parent / "dataset" / "full"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in sorted(RAW.glob("*.png")):
        dst = OUT / src.name
        if not dst.exists():
            im = Image.open(src).convert("RGBA")
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im, (0, 0), im)
            bg.resize((512, 512), Image.LANCZOS).save(dst)
        colors = dominant_colors(Image.open(dst))
        color_txt = f"{' and '.join(colors)} " if colors else ""
        rows.append({
            "file_name": src.name,
            "text": f"{TRIGGER}, a {color_txt}creature, white background",
        })
    (OUT / "metadata.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"{len(rows)} images -> {OUT}")


if __name__ == "__main__":
    main()
