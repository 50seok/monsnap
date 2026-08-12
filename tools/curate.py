"""귀여운 포켓몬만 골라 LoRA 학습용 데이터셋을 만든다.

833장 전체로 학습하면 '온갖 포켓몬의 평균'이 나온다(= 지금 베이스 모델의 문제).
CLIP으로 '귀여움 - 무서움' 점수를 매겨 상위 N장만 남기고, 사람이 컨택트시트로
최종 확인하는 구조. 833장을 눈으로 다 보지 않아도 된다.

사용법:
    python tools/curate.py                 # 상위 200장 추출 + 컨택트시트 생성
    python tools/curate.py --top 150
    python tools/curate.py --drop 3,17,42  # 검토 후 제외할 번호 지정해 재생성

출력:
    dataset/cute/000.png, 000.txt ...      # 학습용 (이미지 + 캡션 쌍)
    dataset/review/sheet_00.jpg ...        # 검토용 컨택트시트 (번호 표기)
"""
import argparse
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
OUT = ROOT / "dataset" / "cute"
REVIEW = ROOT / "dataset" / "review"

# PokéAPI 공식 아트워크 (PRD §6.1 데이터셋 2안). HF의 pokemon-blip-captions는
# 게이트 처리돼서 사용 불가 + 공식 아트워크가 그림체가 훨씬 균일하다.
ART_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{}.png"
MAX_ID = 1025
TRIGGER = "cutemon"  # LoRA 호출용 트리거 토큰 — 학습 후 프롬프트에 이 단어를 넣으면 스타일 발동

# 여러 문장을 평균 내면 단일 문장보다 판정이 안정적이다
CUTE = [
    "a cute round friendly mascot creature",
    "a small chubby adorable cartoon animal",
    "a kawaii pastel creature with big sparkling eyes",
    "a soft simple cheerful character with a smiling face",
]
SCARY = [
    "a scary monster with fangs and sharp claws",
    "a dark evil demon creature",
    "a menacing muscular armored beast",
    "a serpent dragon with sharp teeth",
    "a complex mechanical robot machine",
]


def download_artwork():
    """공식 아트워크를 dataset/raw에 캐시. 이미 받은 건 건너뛴다."""
    RAW.mkdir(parents=True, exist_ok=True)

    def fetch(i):
        path = RAW / f"{i:04d}.png"
        if path.exists():
            return path
        try:
            urllib.request.urlretrieve(ART_URL.format(i), path)
            return path
        except Exception:
            path.unlink(missing_ok=True)  # 결번(미출시 ID 등)은 조용히 건너뜀
            return None

    with ThreadPoolExecutor(max_workers=16) as ex:
        done = [p for p in ex.map(fetch, range(1, MAX_ID + 1)) if p]
    return sorted(done)


def load_rgb(path: Path) -> Image.Image:
    """공식 아트워크는 투명 배경 PNG — 흰 배경으로 합성한다(학습셋 배경 통일)."""
    img = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(canvas, img).convert("RGB")


def score_all(ds, device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # get_text_features/get_image_features는 transformers 버전에 따라 반환 타입이
    # 달라진다(5.x에서 객체로 변경). 전체 forward의 text_embeds/image_embeds는
    # 버전에 무관하게 동일한 CLIP 공동 임베딩 공간이라 이쪽이 안전하다.
    scores = []
    for i in range(0, len(ds), 32):
        batch = [load_rgb(p) for p in ds[i : i + 32]]
        with torch.no_grad():
            inp = proc(text=CUTE + SCARY, images=batch, return_tensors="pt", padding=True).to(device)
            out = model(**inp)
            ifeat = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            tfeat = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            sim = ifeat @ tfeat.T  # (batch, len(CUTE)+len(SCARY))
        scores += (sim[:, : len(CUTE)].mean(1) - sim[:, len(CUTE) :].mean(1)).tolist()
        print(f"  scored {min(i + 32, len(ds))}/{len(ds)}", flush=True)
    return scores


def contact_sheets(items, cols=8, rows=5, cell=160):
    """번호가 찍힌 컨택트시트. 사용자는 빼고 싶은 번호만 알려주면 된다."""
    REVIEW.mkdir(parents=True, exist_ok=True)
    per = cols * rows
    for s in range(0, len(items), per):
        chunk = items[s : s + per]
        sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (idx, img, _) in enumerate(chunk):
            x, y = (k % cols) * cell, (k // cols) * cell
            sheet.paste(img.resize((cell, cell)), (x, y))
            draw.rectangle([x, y, x + 34, y + 16], fill="black")
            draw.text((x + 3, y + 3), str(idx), fill="white")
        path = REVIEW / f"sheet_{s // per:02d}.jpg"
        sheet.save(path, quality=90)
        print(f"  {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--drop", default="", help="제외할 번호 (쉼표 구분)")
    args = ap.parse_args()
    drop = {int(x) for x in args.drop.split(",") if x.strip()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/4] 공식 아트워크 다운로드 (device={device})…", flush=True)
    ds = download_artwork()
    print(f"  {len(ds)}장", flush=True)

    print("[2/4] CLIP 점수 계산…", flush=True)
    scores = score_all(ds, device)

    order = sorted(range(len(ds)), key=lambda i: scores[i], reverse=True)
    picked = [i for i in order[: args.top + len(drop)]][: args.top + len(drop)]

    print(f"[3/4] 상위 {args.top}장 추출 (제외 {len(drop)}개)…", flush=True)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    items, n = [], 0
    for rank, i in enumerate(picked):
        if rank in drop:
            continue
        if n >= args.top:
            break
        img = load_rgb(ds[i]).resize((512, 512), Image.LANCZOS)
        caption = "a cute round creature, big eyes, simple shapes, white background"
        img.save(OUT / f"{n:03d}.png")
        # 트리거 토큰을 앞에 붙인다 — 학습 후 이 단어로 스타일을 호출한다.
        # 스타일 LoRA는 캡션을 통일하는 게 오히려 유리하다(스타일이 트리거에 집중됨).
        (OUT / f"{n:03d}.txt").write_text(f"{TRIGGER}, {caption}", encoding="utf-8")
        items.append((rank, img, caption))
        n += 1

    print(f"[4/4] 컨택트시트 생성…", flush=True)
    contact_sheets(items)

    print(f"\n완료: {n}장 -> {OUT}")
    print(f"검토: {REVIEW}\\sheet_*.jpg 를 열어 빼고 싶은 번호 확인")
    print(f"재생성: python tools/curate.py --drop 3,17,42")


if __name__ == "__main__":
    main()
