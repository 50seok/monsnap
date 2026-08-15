import hashlib
import io
import logging
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

# SDXL 스택: Animagine XL 4.0(danbooru 학습 애니 베이스) + 자체 cutemon SDXL LoRA
# + IP-Adapter FaceID SDXL. SD1.5의 남은 격차(입 뭉개짐·팔다리 소실·특징 소실)가
# 베이스 디테일 한계로 확인돼 전환(PRD §13, probe_sdxl 1·2: 입·팔다리·안경 전부 우세).
# LoRA = dataset/full 1025장 × 5000스텝 중 checkpoint-4000(스윕 실측, v3 때와 동일 패턴).
# SD1.5 롤백: BASE_MODEL=stable-diffusion-v1-5/… + STYLE_LORA=models/style_lora (secrets).
BASE_MODEL = st.secrets.get("BASE_MODEL", "cagliostrolab/animagine-xl-4.0")
_LOCAL_LORA = Path(__file__).parent / "models" / "style_lora_sdxl"
STYLE_LORA = st.secrets.get(
    "STYLE_LORA",
    str(_LOCAL_LORA) if (_LOCAL_LORA / "pytorch_lora_weights.safetensors").exists() else "",
)
SIZE = 512       # 얼굴 크롭·원형 인덱스 해상도(CLIP·데이터셋과 일치)
GEN_SIZE = 1024  # SDXL 네이티브 생성 해상도
# 레퍼런스 원형 = img2img 밑그림 풀. 사람 얼굴과 CLIP 유사도 상위에서 시드로 1마리 —
# 단일 밑그림이라 "여러 후보 평균"이 만들던 키메라를 없앤다(probe7, 사용자 제안).
# 레퍼런스 풀 = dataset/proto_dex(동물 사진 기반 큐레이션 원형, 30종) — 특정 포켓몬
# 닮음·IP 문제를 모티브 원천(동물)에서 출발해 구조적으로 회피(probe_animal).
# secrets REF_DIR로 다른 풀(예: dataset/cute 포켓몬 아트) 실험 가능하나, 그 경우
# STRUCTURED_SHAPES를 {"upright","quadruped","humanoid","legs","arms","wings"}로
# 좁혀야 함 — dataset/cute는 fish도 무사지 덩어리라 넓은 필터에서 블롭 재발(PRD §13).
REF_DIR = Path(st.secrets.get(
    "REF_DIR", str(Path(__file__).parent.parent / "dataset" / "proto_dex")))

# 레퍼런스 합성 모드 프롬프트. 특징 구절을 맨 앞에 — CLIP 텍스트 인코더는 앞 토큰이
# 강해서, 뒤에 두면 img2img가 레퍼런스에 없는 사물(안경 등)을 안 그린다(probe7b·c 실측).
# 트리거 "cutemon" = 자체 LoRA 학습 캡션의 첫 토큰(빼면 카툰화된 사람 — probe2·3 실측).
# "monster"는 절대 넣지 말 것(공개 데이터셋 캡션의 악타입 클러스터 소환 — 이전 실측).
# PokéAPI 공식 체형(shape) → 부위 어휘. img2img는 레퍼런스의 형태 덩어리만 물려받고
# 그 덩어리가 무엇인지는 모른다 — 정답 데이터로 "이 덩어리는 꼬리"라고 알려준다(probe13).
# ⚠ 종을 암시하는 명사(fish 등)는 금지 — 얼굴까지 물고기가 된다(probe12 실측). 사지 어휘만.
SHAPE_PHRASES = {
    "ball": "a simple round body", "squiggle": "a curled body with a clear tail",
    "fish": "one clear round tail", "arms": "two clearly separated arms",
    "blob": "a soft round body", "upright": "clearly separated arms and legs",
    "legs": "two clearly separated legs", "quadruped": "four clearly separated legs",
    "wings": "two clearly separated wings", "tentacles": "clearly separated tentacles",
    "heads": "a round head body", "humanoid": "two clear arms and two clear legs",
    "bug-wings": "clearly separated wings", "armor": "a shell-covered body",
}


# 다리가 없는 체형에 "standing"을 강제하면 모델이 없는 다리를 억지로 만들어 붙여
# 비례가 깨지고 눈이 묻힌다(물고기형 실측, 사용자 피드백) — 체형별 자세어로 대체.
POSE_PHRASES = {"fish": "swimming pose", "squiggle": "swimming pose",
                "tentacles": "floating pose"}


def build_prompt(feature: str | None, palette: str, shape: str = "",
                 signature: str = "") -> str:
    feat = f"{feature}, " if feature else ""
    part = f"{SHAPE_PHRASES[shape]}, " if shape in SHAPE_PHRASES else ""
    sig = f"{signature}, " if signature else ""
    pose = POSE_PHRASES.get(shape, "standing")
    # 학습 캡션 어형("cutemon, a {색} creature, white background")을 맨 앞에 그대로 —
    # 색 견인은 캡션 분포와 같은 위치·어형일 때 가장 강하다. "no humans, solo"는
    # danbooru 태그(Animagine 네이티브 어휘). 특징은 중간 위치 유지 — 맨 앞이면
    # 안경 하나가 디자인 전체를 지배한다(사용자 피드백). 전체 77토큰 이내 유지 필수 —
    # 초과분은 에러 없이 잘린다(SD1.5 시절 94토큰으로 꼬리 유실 실측).
    # 시그니처(속성 부위)가 들어가며 "chubby simple body"는 토큰 예산에서 제외됨.
    return (f"cutemon, a {palette} creature, white background, "
            f"no humans, solo, full body, {pose}, a cute simple face, "
            f"round dot eyes, a tiny clean simple smile, {feat}"
            f"{part}{sig}flat colors, clean thin lineart")


# 속성 시스템(PRD §12-8 계층 시드 완성형): 이름 해시 → 속성 → 몸 색 + 시그니처 부위.
# 시그니처 = 파이리의 꼬리 불꽃처럼 "또렷한 정체성 부위"(사용자 피드백 — 일반 어휘만으론
# '그냥 만들어진 무언가'가 됨). 속성이 색뿐 아니라 디자인 정체성을 결정한다.
# (한글명, 뱃지 RGB, 프롬프트 팔레트, 시그니처 구절)
TYPES = [
    ("불", (239, 118, 61), "fiery orange and red",
     "a small flame on its tail tip, tiny claws on its paws"),
    ("물", (88, 154, 240), "blue and aqua",
     "fin ears and a droplet-shaped tail"),
    ("풀", (120, 200, 80), "fresh green",
     "a leaf sprout on its head and a leafy tail"),
    ("전기", (247, 200, 46), "bright yellow",
     "a lightning-shaped tail and bright cheek marks"),
    ("페어리", (238, 153, 200), "pastel pink and white",
     "ribbon ears and a fluffy curled tail"),
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

# 77토큰 이내로 압축(초과분은 조용히 잘린다 — SD1.5 시절 실측). 앞쪽 = 악타입 억제,
# 중간 = 사람 차단, 뒤쪽 = XL 스윕에서 확인된 아티팩트(빈 눈·입 뭉개짐) + 품질 태그.
NEG_PROMPT = (
    # "claws"는 네거티브 금지 — 시그니처(불 타입 발톱)를 억제한다. 악타입 차단은 나머지로.
    "monster, scary, evil, sharp teeth, dark, horror, "
    "human, person, realistic, photograph, text, watermark, "
    "blurry, deformed, extra limbs, faceless, empty eyes, blank eyes, "
    "messy mouth, distorted mouth, extra mouth, noisy lines, "
    "lowres, bad anatomy, worst quality, low quality"
)
MAX_UPLOAD = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("monsnap")

st.set_page_config(page_title="MonSnap", page_icon="🐲")


@st.cache_resource(show_spinner="모델 로딩 중… (최초 1회 ~7GB 다운로드)")
def pipeline():
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline, UniPCMultistepScheduler

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(
        "h94/IP-Adapter-FaceID", subfolder=None,
        weight_name="ip-adapter-faceid_sdxl.bin", image_encoder_folder=None,
    )
    if STYLE_LORA:
        pipe.load_lora_weights(STYLE_LORA, adapter_name="style")
    # ponytail: VRAM 8GB — 전체 상주 대신 레이어 단위 오프로드 + 1024 디코드 타일링.
    # VRAM 12GB 이상이면 pipe.to("cuda")로 교체해 가속 가능.
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    return pipe


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
    """레퍼런스 200장의 CLIP 임베딩 인덱스. (경로 리스트, 임베딩 텐서, 체형 dict)"""
    import json

    import torch

    paths = sorted(REF_DIR.glob("*.png"))
    embs = [_clip_image_features([Image.open(p).convert("RGB") for p in paths[i:i + 64]])
            for i in range(0, len(paths), 64)]
    manifest_path = REF_DIR / "manifest.json"  # tools/build_ref_manifest.py 산출물
    shapes = (json.loads(manifest_path.read_text("utf-8"))
              if manifest_path.exists() else {})
    return paths, torch.cat(embs), shapes


# 부위가 안 읽히는 체형(heads/ball/blob)만 밑그림 후보에서 배제 — 무사지 원형이
# 밑그림이 되면 변신 후 '이도저도 아닌 덩어리'가 된다(probe14 진단).
# proto_dex(사전 생성·큐레이션 도감)는 부위 가독성이 선별 기준이라 fish·armor·
# tentacles 등도 안전. ⚠ dataset/cute(포켓몬 아트)로 롤백 시엔 구 필터
# {"upright","quadruped","humanoid","legs","arms","wings"} 복원 권장(구 풀은 fish도 덩어리).
STRUCTURED_SHAPES = {"upright", "quadruped", "humanoid", "legs", "arms", "wings",
                     "fish", "squiggle", "armor", "bug-wings", "tentacles"}
REF_TOP_K = 5  # top-1 고정이 아니라 상위 K 중 시드로 결정론 선택 — 원형 다양성 확보


def match_reference(crop: Image.Image, pick: int = 0) -> tuple[Image.Image, float, str]:
    """구조형 원형 중 CLIP 상위 K에서 pick번째. (512px 이미지, 유사도, 체형) 반환."""
    import torch

    paths, embs, shapes = ref_index()
    ok = [i for i, p in enumerate(paths)
          if shapes.get(p.name, {}).get("shape") in STRUCTURED_SHAPES]
    if not ok:  # manifest 없음 — 전체에서 top-1 (구버전 동작)
        ok = list(range(len(paths)))
    sim = (embs[ok] @ _clip_image_features([crop]).T).squeeze(1)
    top = torch.topk(sim, min(REF_TOP_K, len(ok))).indices.tolist()
    idx = ok[top[pick % len(top)]]
    shape = shapes.get(paths[idx].name, {}).get("shape", "")
    chosen_sim = float((embs[idx] @ _clip_image_features([crop]).T).squeeze())
    return Image.open(paths[idx]).convert("RGB").resize((SIZE, SIZE)), chosen_sim, shape


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

    # 계층 시드(PRD §12-8 완성형): 얼굴 임베딩(주) → 시드·레퍼런스(종족),
    # 이름(보조) → 속성(색·장식)만. 이름을 바꿔도 종족은 유지되고 색만 바뀐다.
    seed = int.from_bytes(hashlib.sha256(seed_src).digest()[:4], "big")
    ref, ref_sim, ref_shape = match_reference(photo, pick=seed)
    type_src = name.strip().encode() if name.strip() else seed_src
    type_ko, type_rgb, palette, signature = TYPES[
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
        pipe.set_adapters(["faceid_0", "style"], adapter_weights=[1.0, lora_weight])
    if emb is None:  # 얼굴 없음 — FaceID 끄고 0벡터로 채워 파이프라인 요구만 충족
        pipe.set_ip_adapter_scale(0.0)
        emb = torch.zeros((2, 1, 512), dtype=torch.float16, device="cuda")
    else:
        pipe.set_ip_adapter_scale(faceid_scale)

    result = pipe(
        prompt=build_prompt(feature, palette, ref_shape, signature),
        negative_prompt=NEG_PROMPT,
        image=ref.resize((GEN_SIZE, GEN_SIZE)),  # 원형 512 → SDXL 네이티브 1024
        strength=strength,
        ip_adapter_image_embeds=[emb],
        num_inference_steps=steps,
        # 9.0 = XL 기본(5~7)보다 강한 견인 — CFG 6에선 입·특징이 흐릿, 9에서 또렷
        # (체크포인트 스윕 타이브레이커 실측. SD1.5 시절 probe4와 같은 결론)
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
    # probe8(0.85 최적) 실측은 REF_DIR=dataset/cute(포켓몬 아트) 시절 값 — 그때는
    # 원형과 닮는 게 IP 리스크라 일부러 높였다. REF_DIR=proto_dex(동물 기반) 전환
    # 후 재실측(probe_strength_grid): 0.85는 lora_weight를 낮춰도 원형 구조가 전부
    # 지워지고 포켓몬 학습 LoRA로 재구성됨(물고기 레퍼런스가 발톱 4족 생물이 됨).
    # 0.5~0.7에서는 원형 실루엣이 살아남으면서 색·무늬·표정은 창작됨 — 0.7 채택.
    strength = st.slider(
        "변신 정도", 0.5, 0.95, 0.7, 0.05,
        help="낮으면 매칭된 원형 그대로, 높으면 완전히 새로운 창작",
    )
    lora_weight = st.slider(
        "스타일 강도", 0.0, 1.5, 1.0, 0.05,
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
