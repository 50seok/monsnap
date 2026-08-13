"""dataset/cute 200장의 도감 번호·체형(shape) 매니페스트 생성.

큐레이션 때 파일명이 000~199로 리네임돼 원본 도감 번호가 소실됐다.
① 16px 썸네일 MSE로 raw(도감번호=파일명)와 재매칭 ② PokéAPI species.shape 조회.
shape은 공식 체형 분류(ball/fish/quadruped/wings…) — 생성 프롬프트에 부위 어휘로
주입해 "꼬리인지 날개인지 알 수 없는 덩어리" 문제를 해결한다(정답 데이터 주입).

사용법: python tools/build_ref_manifest.py
출력:   dataset/cute/manifest.json  {"000.png": {"id": 535, "shape": "fish"}, ...}
"""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CUTE, RAW = ROOT / "dataset" / "cute", ROOT / "dataset" / "raw"
API = "https://pokeapi.co/api/v2/pokemon-species/{}/"


def thumb(path):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, (0, 0), im if im.mode == "RGBA" else None)
    return np.asarray(bg.resize((16, 16)), dtype=np.float32)


def main():
    raw_paths = sorted(RAW.glob("*.png"))
    raw_thumbs = np.stack([thumb(p) for p in raw_paths])  # (N,16,16,3)

    mapping = {}
    for p in sorted(CUTE.glob("*.png")):
        t = thumb(p)
        idx = int(((raw_thumbs - t) ** 2).mean(axis=(1, 2, 3)).argmin())
        mapping[p.name] = int(raw_paths[idx].stem)

    def fetch(dex):
        req = urllib.request.Request(
            API.format(dex), headers={"User-Agent": "monsnap-dataset-tool/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return dex, json.load(r)["shape"]["name"]

    shapes = dict(ThreadPoolExecutor(8).map(fetch, set(mapping.values())))
    out = {name: {"id": dex, "shape": shapes[dex]} for name, dex in mapping.items()}
    (CUTE / "manifest.json").write_text(json.dumps(out, indent=0), encoding="utf-8")
    from collections import Counter
    print(len(out), "entries,", Counter(v["shape"] for v in out.values()))


if __name__ == "__main__":
    main()
