"""dataset/full 1025장 → v4 학습 캡션 (설계 문법: 체형 + 색 + 부위 태그).

v3까지의 캡션("cutemon, a {색} creature, white background")은 색 말고는 정보가
없어 모델이 그림체만 배우고 체형↔부위↔시점 설계 문법을 못 배웠다 — 추론에서
부위 문구가 절반만 듣고, img2img 밑그림에 구조를 의존하다 형태가 붕괴한 근본
원인(플랜 2026-08-16). v4는 이미지마다 실제로 참인 설계 정보를 단다:

  ① 체형: PokéAPI species.shape(14종) → 고정 구절(SHAPE_TAGS) — 추론 프롬프트와
     동일 어휘로 사용해 "체형 토큰"으로 학습시킨다.
  ② 색: 기존 dominant_colors 재사용(픽셀 실측 — 타입 색은 실제와 다른 종이 많아
     캡션 거짓이 되므로 쓰지 않는다. 예: 물타입 고라파덕은 노랑).
  ③ 부위: WD14 danbooru 태거(wd-swinv2-tagger-v3) general 태그 — Animagine
     베이스의 네이티브 어휘라 정렬 최적. 캐릭터 태그(category 4)는 전부 드롭
     (§10: 포켓몬 고유명 어휘 배제), "pokemon" 포함 태그도 드롭.

캡션은 CLIPTokenizer로 실측해 75토큰 이내로 태그를 잘라낸다(94토큰 초과로
꼬리 유실됐던 SD1.5 실측 재발 방지 — 추정 금지, 실측).

사용법: cd streamlit_app && .venv\Scripts\python ..\tools\build_captions_v4.py
출력:   dataset/full/metadata.jsonl (덮어씀 — v3 캡션은 git에 없음, 재생성은
        build_full_dataset.py로 언제든 가능)
캐시:   dataset/meta_shapes.json, dataset/wd14_tags.json (재실행 무료)
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
from build_captions import TRIGGER, dominant_colors  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "dataset" / "full"
SHAPE_CACHE = ROOT / "dataset" / "meta_shapes.json"
TAG_CACHE = ROOT / "dataset" / "wd14_tags.json"
API = "https://pokeapi.co/api/v2/pokemon-species/{}/"

WD14_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
TAG_THRESHOLD = 0.35
MAX_TOKENS = 75  # CLIP 77 - 시작/끝 특수토큰

# 체형 → 캡션 구절. ⚠ 추론(app.py) 프롬프트와 반드시 동일 어휘 — 여기서 학습된
# 토큰을 추론에서 그대로 불러 쓰는 구조라, 한쪽만 바꾸면 견인력이 사라진다.
SHAPE_TAGS = {
    "ball": "spherical round body",
    "squiggle": "serpentine body",
    "fish": "aquatic finned body",
    "arms": "torso with arms",
    "blob": "soft blob body",
    "upright": "upright bipedal body",
    "legs": "long-legged body",
    "quadruped": "quadruped body",
    "wings": "winged body",
    "tentacles": "tentacled body",
    "heads": "head-shaped body",
    "humanoid": "humanoid body",
    "bug-wings": "winged insect body",
    "armor": "armored shell body",
}

# 수동 프레이밍 구절과 중복되거나 노이즈인 WD14 태그
TAG_DROP = {
    "solo", "no humans", "full body", "white background", "simple background",
    "transparent background", "artist name", "signature", "watermark",
    "traditional media", "standing", "looking at viewer",
}


def fetch_meta(ids: list[int]) -> dict[str, dict]:
    """종별 {shape, genus}. genus = 공식 '분류'(Mouse Pokémon → mouse) — 디자인
    모티브를 한 단어로 요약한 정답 데이터(사용자 제안). 모티브↔해부 구조 연결을
    학습시키는 핵심 축. 'Pokémon' 접미어는 제거(§10: 고유명 어휘 배제)."""
    if SHAPE_CACHE.exists():
        return json.loads(SHAPE_CACHE.read_text("utf-8"))

    def fetch(dex):
        req = urllib.request.Request(
            API.format(dex), headers={"User-Agent": "monsnap-dataset-tool/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.load(r)
            genus = next((g["genus"] for g in data.get("genera", [])
                          if g["language"]["name"] == "en"), "")
            genus = genus.replace("Pokémon", "").replace("Pokemon", "").strip().lower()
            shape = (data.get("shape") or {}).get("name", "")
            return str(dex), {"shape": shape, "genus": genus}
        except Exception:
            return str(dex), {"shape": "", "genus": ""}  # 결번·null은 구절 생략

    meta = dict(ThreadPoolExecutor(8).map(fetch, ids))
    SHAPE_CACHE.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def wd14_tag_all(paths: list[Path]) -> dict[str, list[str]]:
    if TAG_CACHE.exists():
        return json.loads(TAG_CACHE.read_text("utf-8"))

    import csv

    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(WD14_REPO, "model.onnx")
    tags_path = hf_hub_download(WD14_REPO, "selected_tags.csv")
    with open(tags_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # category: 0=general(부위·외형), 4=character(고유명 — §10 정합상 전부 배제), 9=rating
    general = [(i, r["name"].replace("_", " ")) for i, r in enumerate(rows)
               if r["category"] == "0"]

    sess = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = inp.shape[1] if isinstance(inp.shape[1], int) else 448

    def preprocess(p: Path) -> np.ndarray:
        img = Image.open(p).convert("RGB")
        side = max(img.size)
        canvas = Image.new("RGB", (side, side), "white")
        canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        arr = np.asarray(canvas.resize((size, size), Image.BICUBIC), dtype=np.float32)
        return arr[:, :, ::-1][None]  # RGB→BGR (SmilingWolf 태거 규약)

    result = {}
    for n, p in enumerate(paths, 1):
        probs = sess.run(None, {inp.name: preprocess(p)})[0][0]
        picked = [(name, float(probs[i])) for i, name in general
                  if probs[i] >= TAG_THRESHOLD
                  and "pokemon" not in name and name not in TAG_DROP]
        picked.sort(key=lambda t: t[1], reverse=True)
        result[p.name] = [name for name, _ in picked]
        if n % 100 == 0 or n == len(paths):
            print(f"  wd14 {n}/{len(paths)}", flush=True)

    TAG_CACHE.write_text(json.dumps(result), encoding="utf-8")
    return result


def main():
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def n_tokens(text: str) -> int:
        return len(tok(text).input_ids) - 2

    paths = sorted(FULL.glob("*.png"))
    ids = [int(p.stem) for p in paths]
    meta = fetch_meta(ids)
    n_shape = sum(1 for v in meta.values() if v["shape"])
    n_genus = sum(1 for v in meta.values() if v["genus"])
    print(f"메타: 체형 {n_shape} · 모티브(genus) {n_genus} / {len(paths)}", flush=True)
    tags = wd14_tag_all(paths)

    rows, token_max = [], 0
    for p in paths:
        m = meta.get(str(int(p.stem)), {"shape": "", "genus": ""})
        shape_txt = SHAPE_TAGS.get(m["shape"], "")
        genus_txt = f"{m['genus']} motif" if m["genus"] else ""
        colors = dominant_colors(Image.open(p))
        color_txt = f"a {' and '.join(colors)} creature" if colors else "a creature"

        # 순서: 트리거 → 모티브(디자인 컨셉) → 체형 → 색 — CLIP 앞토큰이 강하므로
        # 설계 축(모티브·체형)을 앞에 배치. 부위 태그가 뒤에서 세부를 채운다.
        parts = [TRIGGER, genus_txt, shape_txt, color_txt]
        head = ", ".join(x for x in parts if x)
        tail = "solo, full body, white background"
        body_tags = list(tags.get(p.name, []))
        # 토큰 실측으로 태그를 뒤에서부터 잘라 75토큰 이내 보장(초과분 무음 유실 방지)
        while True:
            caption = f"{head}, {', '.join(body_tags)}, {tail}" if body_tags else f"{head}, {tail}"
            if n_tokens(caption) <= MAX_TOKENS or not body_tags:
                break
            body_tags.pop()
        token_max = max(token_max, n_tokens(caption))
        rows.append({"file_name": p.name, "text": caption})

    (FULL / "metadata.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    from collections import Counter
    freq = Counter(t for r in rows for t in r["text"].split(", "))
    print(f"\n{len(rows)} captions -> {FULL / 'metadata.jsonl'} (최대 {token_max}토큰)")
    print("상위 태그 25:", freq.most_common(25))
    for r in rows[:3] + rows[500:502]:
        print(" ", r["text"])


if __name__ == "__main__":
    main()
