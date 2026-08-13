import hashlib
import io
import logging
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

# 모듈형 스택: 표준 SD1.5 + 스타일 LoRA + IP-Adapter FaceID + ControlNet lineart.
# 풀 파인튜닝 베이스(sd-pokemon-diffusers)는 FaceID를 막는다는 게 실측 확인돼
# (PRD §13) 베이스는 순정으로 두고 스타일은 LoRA로 얹는다.
# 기본 LoRA = 자체 학습 v1(트리거 cutemon, 색 보존 캡션). 가중치는 커밋 금지(*.safetensors)라
# 파일이 없는 환경에선 공개 LoRA로 폴백 — 그땐 PROMPT 트리거를 "pokemon"으로 맞출 것.
BASE_MODEL = st.secrets.get("BASE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
_LOCAL_LORA = Path(__file__).parent / "models" / "style_lora"
STYLE_LORA = st.secrets.get(
    "STYLE_LORA",
    str(_LOCAL_LORA) if (_LOCAL_LORA / "pytorch_lora_weights.safetensors").exists()
    else "pcuenq/pokemon-lora",
)
# 보조 스타일 = v2(큐레이션 200장 학습). v3(전체 1025장)는 포켓몬 비례를 주지만
# 무생물형 포켓몬의 '얼굴 없는 모드'가 섞여서, 귀여움·얼굴 편향의 v2를 낮은 가중치로
# 혼합해 상쇄한다(probe9: v3 1.1 + v2 0.7이 최적). 없으면 v3 단독.
_LOCAL_LORA2 = Path(__file__).parent / "models" / "style_lora_v2"
STYLE_LORA2 = (str(_LOCAL_LORA2)
               if (_LOCAL_LORA2 / "pytorch_lora_weights.safetensors").exists() else "")
STYLE2_WEIGHT = 0.7
SIZE = 512  # SD1.5 네이티브 해상도
# 레퍼런스 원형 후보 = 큐레이션된 귀여운 200장. 사람 얼굴과 CLIP 유사도 top-1 한 마리를
# img2img 밑그림으로 써서 "여러 포켓몬 평균"이 만들던 키메라를 없앤다(probe7, 사용자 제안).
# ⚠ IP: 특정 원형에 앵커하므로 실서비스 전 클린 데이터로 교체 필수(PRD §10).
REF_DIR = Path(__file__).parent.parent / "dataset" / "cute"

# 레퍼런스 합성 모드 프롬프트. 특징 구절을 맨 앞에 — CLIP 텍스트 인코더는 앞 토큰이
# 강해서, 뒤에 두면 img2img가 레퍼런스에 없는 사물(안경 등)을 안 그린다(probe7b·c 실측).
# 트리거 "cutemon" = 자체 LoRA 학습 캡션의 첫 토큰(빼면 카툰화된 사람 — probe2·3 실측).
# "monster"는 절대 넣지 말 것(공개 데이터셋 캡션의 악타입 클러스터 소환 — 이전 실측).
def build_prompt(feature: str | None, palette: str) -> str:
    feat = f"{feature}, " if feature else ""
    # 얼굴 구절("cute simple face, round dot eyes, tiny smiling mouth")을 앞에 고정 —
    # v3에 섞인 무생물형 포켓몬의 '얼굴 없는 모드'를 차단(probe9, 네거티브 faceless와 세트).
    # 특징은 중간 위치로 강등 — 맨 앞에 두면 안경 하나가 디자인 전체를 지배한다(사용자 피드백).
    # "a tiny clean simple smile, clean thin lineart": 512px에서 입이 작은 영역이라
    # 선이 뭉개지던 문제 — 이 어휘로 깨끗한 곡선 미소가 나온다(probe10, 정제 패스 불필요)
    return ("cutemon creature, full body, standing, "
            f"a cute round {palette} creature with a cute simple face, "
            f"round dot eyes, a tiny clean simple smile, clean thin lineart, {feat}"
            "chubby simple body, flat colors")


# 속성 시스템(PRD §12-8 계층 시드 완성형): 이름 해시 → 속성 → 몸 색 팔레트.
# (한글명, 뱃지 RGB, 프롬프트 팔레트)
TYPES = [
    ("불", (239, 118, 61), "fiery orange and red"),
    ("물", (88, 154, 240), "blue and aqua"),
    ("풀", (120, 200, 80), "fresh green"),
    ("전기", (247, 200, 46), "bright yellow"),
    ("페어리", (238, 153, 200), "pastel pink and white"),
]

# CLIP 제로샷 특징 후보: (프롬프트 구절, 한국어 표시, 긍정 문장, 부정 문장)
# 점수 = P(긍정) - P(부정) 마진. 전 후보 중 최고 마진 1개만 채택(임계 0.12 미달 시 없음).
FEATURES = [
    ("wearing tiny round glasses", "안경", "a person wearing glasses", "a person without glasses"),
    ("with a fluffy beard", "수염", "a person with a beard", "a clean-shaven person"),
    ("with long flowing hair", "긴 머리", "a person with long hair", "a person with short hair"),
    ("with spiky hair", "뻗친 머리", "a person with spiky messy hair", "a person with neat flat hair"),
    ("with big round eyes", "큰 눈", "a person with big round eyes", "a person with small narrow eyes"),
    ("with narrow sleepy eyes", "가는 눈", "a person with small narrow eyes", "a person with big round eyes"),
    ("with a big round nose", "큰 코", "a person with a big prominent nose", "a person with a small flat nose"),
    ("with a wide smiling mouth", "큰 입", "a person with a wide big mouth", "a person with a small mouth"),
    ("with chubby cheeks", "통통한 볼", "a person with chubby round cheeks", "a person with a slim narrow face"),
    ("with thick bold eyebrows", "진한 눈썹", "a person with thick dark eyebrows", "a person with thin light eyebrows"),
    ("with a small beauty mark", "점", "a person with a facial mole or beauty mark", "a person with clear smooth skin"),
    ("with big round ears", "큰 귀", "a person with big prominent ears", "a person with small flat ears"),
    ("with a wide broad forehead", "넓은 이마", "a person with a wide broad forehead", "a person with a forehead covered by bangs"),
]

# 앞쪽 = 악타입 억제(귀여움 확보), 뒤쪽 = 사람이 아니라 크리처로 채우게 만드는 장치
NEG_PROMPT = (
    "monster, demon, scary, evil, fangs, sharp teeth, claws, bat wings, "
    "dark, muscular, horror, "
    "human, person, human face, realistic face, photograph, text, watermark, "
    "blurry, deformed, extra limbs, ugly, "
    "realistic eyes, detailed iris, human eyes, "
    "faceless, no face, mechanical, robot, pokeball, orb, machine, "
    "messy mouth, smudged face, distorted mouth, extra mouth, noisy lines"
)
MAX_UPLOAD = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("monsnap")

st.set_page_config(page_title="MonSnap", page_icon="🐲")


@st.cache_resource(show_spinner="모델 로딩 중… (최초 1회 ~5GB 다운로드)")
def pipeline():
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,  # 얼굴 입력에서 오탐이 잦고 VRAM 1.2GB를 더 먹는다
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(
        "h94/IP-Adapter-FaceID", subfolder=None,
        weight_name="ip-adapter-faceid_sd15.bin", image_encoder_folder=None,
    )
    if STYLE_LORA:
        pipe.load_lora_weights(_style_lora_state(STYLE_LORA), adapter_name="style")
    if STYLE_LORA2:
        pipe.load_lora_weights(STYLE_LORA2, adapter_name="style2")
    # ponytail: VRAM 8GB — 전체 상주 대신 레이어 단위 오프로드. 더 빠르게 하려면
    # VRAM 12GB 이상에서 pipe.to("cuda")로 교체.
    pipe.enable_model_cpu_offload()
    return pipe


def _style_lora_state(repo: str):
    """임시 공개 LoRA(2023 attn-proc 포맷)를 PEFT 키로 변환해 state dict로 반환.

    모던 포맷(자체 학습본)은 변환 없이 그대로 통과시킨다. 구식 포맷을 그냥
    load_lora_weights에 넘기면 에러 없이 '키 0개 매칭' 경고만 내고 무시된다(실측) —
    스타일이 조용히 빠진 채 사람 그림이 나오므로 여기서 미리 변환한다.
    """
    import torch
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo, "pytorch_lora_weights.bin")
    except Exception:  # .bin이 없으면 모던 포맷 레포 — 그대로 로드
        return repo
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not any(".processor." in k for k in sd):
        return repo
    return {
        "unet." + k.replace(".processor.", ".").replace("to_out_lora", "to_out.0")
        .replace("to_q_lora", "to_q").replace("to_k_lora", "to_k")
        .replace("to_v_lora", "to_v")
        .replace(".down.weight", ".lora_A.weight").replace(".up.weight", ".lora_B.weight"): v
        for k, v in sd.items()
    }


@st.cache_resource(show_spinner=False)
def face_embedder():
    """insightface — FaceID용 정체성 임베딩. YuNet(박스)과 용도가 다르다."""
    from insightface.app import FaceAnalysis

    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))
    return fa


@st.cache_resource(show_spinner=False)
def clip_model():
    from transformers import CLIPModel, CLIPProcessor

    return (CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval(),
            CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"))


def pick_traits(crop: Image.Image):
    """CLIP 제로샷 특징 점수판. (채택 구절, 채택 한국어, [(한국어, 마진점수)] 내림차순) 반환.

    점수 = P(긍정문장) - P(부정문장) 마진. 최고 마진 1개만 채택, 0.12 미달이면 순수 원형.
    """
    import torch

    model, proc = clip_model()
    scores = []
    with torch.no_grad():
        for phrase, ko, pos, neg in FEATURES:
            inp = proc(text=[pos, neg], images=crop, return_tensors="pt", padding=True)
            p = model(**inp).logits_per_image.softmax(-1)[0]
            scores.append((phrase, ko, float(p[0] - p[1])))
    scores.sort(key=lambda s: s[2], reverse=True)
    best = scores[0] if scores[0][2] > 0.12 else None
    return (best[0] if best else None, best[1] if best else None,
            [(ko, m) for _, ko, m in scores])


def _clip_image_features(imgs):
    """CLIP 이미지 임베딩(정규화). transformers 5.x는 get_image_features가
    출력 객체를 반환하므로 vision_model→visual_projection을 직접 탄다."""
    import torch

    model, proc = clip_model()
    with torch.no_grad():
        inp = proc(images=imgs, return_tensors="pt")
        vis = model.vision_model(pixel_values=inp["pixel_values"])
        f = model.visual_projection(vis.pooler_output)
        return f / f.norm(dim=-1, keepdim=True)


@st.cache_resource(show_spinner="원형 도감 인덱싱 중… (최초 1회 ~20초)")
def ref_index():
    """레퍼런스 200장의 CLIP 임베딩 인덱스. (경로 리스트, 임베딩 텐서)"""
    import torch

    paths = sorted(REF_DIR.glob("*.png"))
    embs = [_clip_image_features([Image.open(p).convert("RGB") for p in paths[i:i + 64]])
            for i in range(0, len(paths), 64)]
    return paths, torch.cat(embs)


def match_reference(crop: Image.Image) -> tuple[Image.Image, float]:
    """사람 얼굴과 CLIP 유사도 top-1 원형 한 마리. (512px 이미지, 유사도) 반환."""
    paths, embs = ref_index()
    sim = (embs @ _clip_image_features([crop]).T).squeeze(1)
    idx = int(sim.argmax())
    return Image.open(paths[idx]).convert("RGB").resize((SIZE, SIZE)), float(sim[idx])


YUNET = Path(__file__).parent / "models" / "yunet.onnx"
DETECT_MAX = 640  # 검출은 축소본으로 — 휴대폰 원본(4000px)을 그대로 넣으면 느리고 불안정


@st.cache_resource(show_spinner=False)
def face_detector():
    """opencv 5의 YuNet. Haar cascade는 opencv 5에서 번들 제외됐고,
    YuNet이 기울어진 얼굴에도 강해서 촬영 제약을 덜 걸어도 된다."""
    import cv2

    return cv2.FaceDetectorYN.create(str(YUNET), "", (DETECT_MAX, DETECT_MAX), score_threshold=0.6)


def face_box(img: Image.Image) -> tuple[int, int, int, int] | None:
    """가장 큰 얼굴을 감싸는 정사각 영역. 못 찾으면 None."""
    import cv2
    import numpy as np

    scale = min(1.0, DETECT_MAX / max(img.size))
    small = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))

    det = face_detector()
    det.setInputSize((small.width, small.height))
    _, faces = det.detect(cv2.cvtColor(np.array(small), cv2.COLOR_RGB2BGR))
    if faces is None or len(faces) == 0:
        return None

    # 축소본 좌표를 원본 좌표로 되돌린다
    x, y, w, h = (max(faces, key=lambda f: f[2] * f[3])[:4] / scale).astype(int)
    # 헤어스타일이 핵심 특징이라 얼굴 박스보다 넉넉히 잡는다(1.8배).
    side = min(int(max(w, h) * 1.8), img.width, img.height)
    cx, cy = x + w // 2, y + h // 2
    left = int(min(max(0, cx - side // 2), img.width - side))
    top = int(min(max(0, cy - side // 2), img.height - side))
    return (left, top, left + side, top + side)


def prepare(img: Image.Image, size: int = SIZE) -> tuple[Image.Image, bool]:
    """회전 보정 → 얼굴 크롭 → 정사각 리사이즈. (이미지, 얼굴검출여부) 반환.

    EXIF를 먼저 적용하되 그것만 믿지 않는다. 메신저를 거친 사진은 EXIF가 통째로
    날아가고, 태그가 실제 방향과 어긋난 사진도 흔하다. 그래서 얼굴을 못 찾으면
    90도씩 돌려가며 재시도한다 — 얼굴이 잡히는 방향이 곧 올바른 방향이다.
    실패한 검출 1회는 640px 축소본 기준 수 ms라 비용은 무시할 수준.
    """
    img = ImageOps.exif_transpose(img).convert("RGB")

    for angle in (0, 270, 90, 180):  # 0 = EXIF가 맞은 경우(대부분 여기서 끝)
        candidate = img if angle == 0 else img.rotate(angle, expand=True)
        box = face_box(candidate)
        if box is not None:
            return candidate.crop(box).resize((size, size), Image.LANCZOS), True

    side = min(img.size)  # 폴백: 중앙 정사각 크롭
    left, top = (img.width - side) // 2, (img.height - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS), False


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def identity(photo_bytes: bytes, photo: Image.Image):
    """FaceID 임베딩 (2,1,512)와 시드 소스 바이트를 반환. 얼굴 못 찾으면 (None, 사진바이트).

    임베딩은 원본에서 뽑는다 — 1.8배 타이트 크롭은 insightface 검출이 불안정하고,
    ControlNet용 크롭(형태)과 FaceID용 임베딩(정체성)은 애초에 용도가 다르다.
    """
    import cv2
    import numpy as np
    import torch

    src = ImageOps.exif_transpose(Image.open(io.BytesIO(photo_bytes))).convert("RGB")
    faces = face_embedder().get(cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR))
    if not faces:
        faces = face_embedder().get(cv2.cvtColor(np.array(photo), cv2.COLOR_RGB2BGR))
    if not faces:
        return None, photo_bytes

    normed = max(faces, key=lambda f: f.det_score).normed_embedding
    pos = torch.from_numpy(normed).unsqueeze(0)  # (1, 512)
    # CFG용 네거티브(0벡터)를 dim 0에 쌓아 (2, 1, 512) — 빼면 chunk(2) 언패킹 실패
    emb = torch.cat([torch.zeros_like(pos), pos], dim=0).unsqueeze(1)
    return emb.to(dtype=torch.float16, device="cuda"), normed.tobytes()


def generate(photo_bytes: bytes, strength: float, steps: int,
             lora_weight: float, faceid_scale: float, name: str):
    """(결과 PNG, 레퍼런스 PNG, 전처리 입력 PNG, 얼굴검출여부, 카드정보 dict) 반환.

    레퍼런스 합성(사용자 설계): ① 특징 1개 인식(CLIP 점수) → ② 가장 닮은 원형
    1마리 선택 → ③ img2img로 특징을 배합. 단일 원형 밑그림이라 키메라가 안 나온다.
    """
    import torch

    photo, face_found = prepare(Image.open(io.BytesIO(photo_bytes)))
    emb, seed_src = identity(photo_bytes, photo)
    feature, feature_ko, scores = pick_traits(photo)
    ref, ref_sim = match_reference(photo)

    # 계층 시드(PRD §12-8 완성형): 얼굴 임베딩(주) → 시드·레퍼런스(종족),
    # 이름(보조) → 속성(색·장식)만. 이름을 바꿔도 종족은 유지되고 색만 바뀐다.
    seed = int.from_bytes(hashlib.sha256(seed_src).digest()[:4], "big")
    type_src = name.strip().encode() if name.strip() else seed_src
    type_ko, type_rgb, palette = TYPES[
        int.from_bytes(hashlib.sha256(type_src).digest()[:4], "big") % len(TYPES)]
    info = {
        "traits": feature_ko or "없음 (순수 원형)",
        "scores": scores, "ref_sim": ref_sim,
        "type_ko": type_ko, "type_rgb": type_rgb,
        "dex": seed % 1000, "name": name.strip(),
    }

    pipe = pipeline()
    if STYLE_LORA:
        # faceid_0(FaceID 체크포인트에 내장된 LoRA)도 같이 켜야 한다 —
        # style만 넘기면 set_adapters가 faceid_0을 비활성화해 닮음이 사라진다(실측)
        names = ["faceid_0", "style"] + (["style2"] if STYLE_LORA2 else [])
        weights = [1.0, lora_weight] + ([STYLE2_WEIGHT] if STYLE_LORA2 else [])
        pipe.set_adapters(names, adapter_weights=weights)
    if emb is None:  # 얼굴 없음 — FaceID 끄고 0벡터로 채워 파이프라인 요구만 충족
        pipe.set_ip_adapter_scale(0.0)
        emb = torch.zeros((2, 1, 512), dtype=torch.float16, device="cuda")
    else:
        pipe.set_ip_adapter_scale(faceid_scale)

    result = pipe(
        prompt=build_prompt(feature, palette),
        negative_prompt=NEG_PROMPT,
        image=ref,
        strength=strength,
        ip_adapter_image_embeds=[emb],
        num_inference_steps=steps,
        # 9.0 = 기본(7.5)보다 강한 프롬프트·네거티브 견인 — "human, person" 억제가
        # 세져 시드가 바뀌어도 사람 캐리커처로 안 떨어진다(probe4 실측)
        guidance_scale=9.0,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    return to_png(result), to_png(ref), to_png(photo), face_found, info


def make_card(result_png: bytes, info: dict) -> bytes:
    """결과를 도감 카드로 합성 — 속성색 테두리·뱃지, 이름, 도감 번호."""
    from PIL import ImageDraw, ImageFont

    W, H, M = 640, 840, 24
    r, g, b = info["type_rgb"]
    card = Image.new("RGB", (W, H), (r // 6 + 213, g // 6 + 213, b // 6 + 213))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle([8, 8, W - 8, H - 8], radius=24, outline=info["type_rgb"], width=6)
    art = Image.open(io.BytesIO(result_png)).resize((W - 2 * M, W - 2 * M))
    card.paste(art, (M, 116))
    try:  # 한국어 렌더링 — 맑은고딕(한국어 Windows 표준). 없으면 PIL 기본 폰트.
        f_big = ImageFont.truetype(r"C:\Windows\Fonts\malgunbd.ttf", 44)
        f_small = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 24)
    except OSError:
        f_big = f_small = ImageFont.load_default()
    d.text((M, 40), info["name"] or "이름 없는 몬스터", font=f_big, fill=(40, 40, 40))
    badge = f"{info['type_ko']} 타입"
    bw = d.textlength(badge, font=f_small)
    d.rounded_rectangle([W - M - bw - 28, 46, W - M, 92], radius=22, fill=info["type_rgb"])
    d.text((W - M - bw - 14, 55), badge, font=f_small, fill="white")
    d.text((M, 116 + (W - 2 * M) + 16), f"특징: {info['traits']}", font=f_small, fill=(90, 90, 90))
    d.text((M, H - 56), f"MonSnap 도감 No.{info['dex']:03d}", font=f_small, fill=(120, 120, 120))
    return to_png(card)


st.title("MonSnap 🐲")
st.caption("사진을 찍으면 나만의 몬스터가 태어납니다")

with st.sidebar:
    st.subheader("생성 설정")
    # probe8 실측: 0.85 = 눈·색이 확실히 새 창작으로 바뀌면서 특징(안경)은 유지.
    # 0.7 이하는 원형과 너무 닮고(사용자 피드백 + IP 리스크), 0.9부터 특징이 소실됨.
    strength = st.slider(
        "변신 정도", 0.5, 0.95, 0.85, 0.05,
        help="낮으면 매칭된 원형 그대로, 높으면 완전히 새로운 창작",
    )
    # probe9: v3 1.1 + v2 0.7 혼합이 최적 — 슬라이더는 주 스타일(v3)만 조절
    lora_weight = st.slider(
        "스타일 강도", 0.0, 1.5, 1.1, 0.05,
        help="크리처 그림체로 미는 힘 — 약하면 사람 그림이 됩니다",
    )
    faceid_scale = st.slider(
        "닮음 강도", 0.0, 1.0, 0.2, 0.05,
        help="원본 얼굴 분위기를 반영하는 힘 — 세면 사람에 가까워집니다",
    )
    steps = st.slider("스텝 수", 10, 40, 30, 5, help="높을수록 느리고 정교함")
    show_debug = st.checkbox("매칭 과정 보기", value=True)

st.info(
    "**이렇게 찍으면 잘 나옵니다**\n"
    "- 얼굴이 **정면**을 보게 (옆얼굴은 인식되지 않습니다)\n"
    "- 화면에 **얼굴이 크게** 차게 — 배경이 넓으면 배경 윤곽이 특징으로 섞입니다\n"
    "- **머리 전체**가 잘리지 않게 — 헤어스타일이 가장 큰 특징입니다\n"
    "- **밝은 곳**에서, 안경·앞머리로 얼굴을 가리지 않게",
    icon="📸",
)

source = st.radio(
    "사진 방법", ["앨범에서 선택", "카메라로 촬영"], horizontal=True, label_visibility="collapsed"
)
uploaded = (
    st.file_uploader("사진 선택", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
    if source == "앨범에서 선택"
    else st.camera_input("사진 촬영", label_visibility="collapsed")
)

if uploaded is not None:
    photo_bytes = uploaded.getvalue()
    mime_type = uploaded.type or "image/jpeg"

    # FR-1: 파일 검증
    if not mime_type.startswith("image/"):
        st.error("이미지 파일만 업로드할 수 있습니다.")
    elif len(photo_bytes) > MAX_UPLOAD:
        st.error("10MB 이하 이미지만 지원합니다.")
    else:
        # 새 사진이면 이전 결과를 지운다 (다운로드 버튼 클릭 등으로 재실행돼도
        # 같은 사진인 동안은 session_state의 결과가 유지되어 화면이 안 비워진다)
        if st.session_state.get("photo_bytes") != photo_bytes:
            st.session_state.pop("result", None)
            st.session_state["photo_bytes"] = photo_bytes

        col1, col2 = st.columns(2)
        col1.image(photo_bytes, caption="원본", use_container_width=True)

        # 계층 시드의 보조 키(PRD §12-8) — 실루엣은 사진이, 속성(색)은 이름이 결정
        name = st.text_input(
            "몬스터 이름 (선택)", max_chars=20,
            placeholder="이름을 지어 주세요 — 이름이 몬스터의 속성(색)을 결정합니다",
        )

        label = "다시 생성" if "result" in st.session_state else "몬스터 생성"
        if st.button(label, type="primary"):
            t0 = time.monotonic()
            try:
                with st.spinner("몬스터 소환 중…"):
                    result, ref, crop, face_found, info = generate(
                        photo_bytes, strength, steps, lora_weight, faceid_scale, name
                    )
                st.session_state["result"] = result
                st.session_state["ref"] = ref
                st.session_state["crop"] = crop
                st.session_state["face_found"] = face_found
                st.session_state["info"] = info
                st.session_state["card"] = make_card(result, info)
                log.info(
                    "generate ok elapsed=%.1fs strength=%.2f lora=%.2f face=%.2f steps=%d type=%s",
                    time.monotonic() - t0, strength, lora_weight, faceid_scale, steps,
                    info["type_ko"],
                )
            except Exception as exc:  # FR-4
                st.session_state.pop("result", None)
                log.exception("generate failed elapsed=%.1fs", time.monotonic() - t0)
                if "out of memory" in str(exc).lower():
                    st.error("GPU 메모리가 부족합니다. 스텝 수를 줄이거나 다른 앱을 종료해 주세요.")
                else:
                    st.error("생성에 실패했습니다. 잠시 후 다시 시도해 주세요.")

        if "result" in st.session_state:
            col2.image(st.session_state["result"], caption="몬스터", use_container_width=True)
            if "info" in st.session_state:
                i = st.session_state["info"]
                st.caption(f"속성: {i['type_ko']} · 반영된 특징: {i['traits']} · 도감 No.{i['dex']:03d}")

            if not st.session_state.get("face_found", True):
                st.warning(
                    "얼굴을 찾지 못해 사진 중앙을 잘라 썼습니다. 정면·밝은 곳에서 "
                    "얼굴이 크게 나오도록 다시 찍으면 특징이 훨씬 잘 반영됩니다.",
                    icon="⚠️",
                )

            if show_debug and "ref" in st.session_state:
                i = st.session_state["info"]
                st.caption("매칭 과정 — ① 얼굴 → ② 가장 닮은 원형 1마리 → ③ 특징 배합")
                e1, e2 = st.columns(2)
                e1.image(st.session_state["crop"], caption="얼굴 크롭", use_container_width=True)
                e2.image(st.session_state["ref"],
                         caption=f"매칭된 원형 (유사도 {i['ref_sim']:.2f})", use_container_width=True)
                score_txt = " · ".join(f"{ko} {m:+.2f}" for ko, m in i["scores"][:4])
                st.caption(f"특징 점수(상위 4): {score_txt} — 0.12 이상 최고점만 반영")
            if "card" in st.session_state:
                st.image(st.session_state["card"], caption="도감 카드", width=360)
            b1, b2 = st.columns(2)
            b1.download_button(
                "이미지 저장",
                data=st.session_state["result"],
                file_name="monster.png",
                mime="image/png",
            )
            if "card" in st.session_state:
                b2.download_button(
                    "도감 카드 저장",
                    data=st.session_state["card"],
                    file_name="monsnap_card.png",
                    mime="image/png",
                )
            st.caption(
                "같은 사진이면 항상 같은 몬스터가 나옵니다. "
                "이름을 바꾸면 속성(색)이 바뀝니다."
            )

st.divider()
st.caption("사진은 이 PC 안에서만 처리됩니다. 외부로 전송되지 않습니다.")
