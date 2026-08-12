import io
import logging
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageFilter, ImageOps

# 로컬 생성 스택: SD1.5 파생 모델 + ControlNet lineart(형태·특징 유지)
# ponytail: 베이스 모델은 secrets로 교체 가능 — LoRA 자체 학습 후 여기만 바꾸면 됨
BASE_MODEL = st.secrets.get("BASE_MODEL", "lambdalabs/sd-pokemon-diffusers")
# lineart > softedge (실측). softedge(HED)는 선이 뭉툭해서 눈매·입술·헤어라인이
# 소실되고 강도를 올리면 뭉개져 붕괴한다. lineart는 쌍꺼풀·코 윤곽까지 선으로 남고
# 강도를 올려도 붕괴 대신 "사람에 가까워지는" 방향이라 제어 구간이 넓다(0.5~0.8).
CONTROLNET = "lllyasviel/control_v11p_sd15_lineart"
SIZE = 512  # SD1.5 네이티브 해상도

# "monster"는 절대 넣지 말 것. 학습 데이터(pokemon-blip-captions)는 BLIP 자동 캡션이라
# 무서운 포켓몬들이 "a monster"/"a demon"으로 캡션됐다. 그래서 이 단어 하나로 악타입
# 클러스터가 소환되고, "cute"를 붙여도 못 이긴다(실측 확인). 귀여움은 형용사가 아니라
# 명사 선택의 문제 — round/chubby/mascot 계열 어휘로 부를 것.
PROMPT = (
    "a cute round creature, big sparkling eyes, smiling face, "
    "chubby simple body, bright cheerful colors, mascot character"
)
# 앞쪽 = 악타입 억제(귀여움 확보), 뒤쪽 = 사람이 아니라 크리처로 채우게 만드는 장치
NEG_PROMPT = (
    "monster, demon, scary, evil, fangs, sharp teeth, claws, bat wings, "
    "dark, muscular, horror, "
    "human, person, realistic face, photograph, text, watermark, "
    "blurry, deformed, extra limbs, ugly"
)
MAX_UPLOAD = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("monsnap")

st.set_page_config(page_title="MonSnap", page_icon="🐲")


@st.cache_resource(show_spinner="모델 로딩 중… (최초 1회 ~5GB 다운로드)")
def pipeline():
    import torch
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    controlnet = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,  # 얼굴 입력에서 오탐이 잦고 VRAM 1.2GB를 더 먹는다
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # ponytail: VRAM 8GB — 전체 상주 대신 레이어 단위 오프로드. 더 빠르게 하려면
    # VRAM 12GB 이상에서 pipe.to("cuda")로 교체.
    pipe.enable_model_cpu_offload()
    return pipe


@st.cache_resource(show_spinner=False)
def edge_detector():
    from controlnet_aux import LineartDetector

    return LineartDetector.from_pretrained("lllyasviel/Annotators")


@st.cache_resource(show_spinner=False)
def bg_remover():
    from rembg import new_session

    return new_session("u2net")


def mask_background(photo: Image.Image, edge: Image.Image) -> Image.Image:
    """인물 영역 밖의 선을 지운다.

    얼굴 크롭을 1.8배로 넉넉히 잡다 보니(헤어 보존 목적) 선반·포스터·의자 같은
    배경 모서리가 같이 딸려와 ControlNet에 '특징'으로 들어갔다. 타원 마스크로도
    모서리는 지워지지만 타원 안쪽 배경이 남아서, 인물 분리(u2net)를 쓴다.
    헤어 라인을 정확히 보존하면서 배경만 없애는 게 핵심.
    """
    import numpy as np
    from rembg import remove

    alpha = remove(photo, session=bg_remover()).getchannel("A")
    alpha = alpha.filter(ImageFilter.GaussianBlur(4))  # 경계 계단 방지

    # rembg는 '인물 전경'을 남기므로 목에 두른 수건·옷이 그대로 통과한다. 천 질감은
    # 잔선을 대량 만들어 ControlNet에 가짜 구조로 들어간다. 크롭이 얼굴 1.8배 정사각이라
    # 턱은 대략 높이의 0.78 지점 — 그 아래를 페이드아웃시켜 목/어깨/옷을 버린다.
    arr = np.array(alpha, dtype=np.float32)
    height = arr.shape[0]
    cut, fade = int(height * 0.80), max(1, int(height * 0.10))
    ramp = np.ones(height, dtype=np.float32)
    ramp[cut : cut + fade] = np.linspace(1.0, 0.0, min(fade, height - cut))
    ramp[cut + fade :] = 0.0
    alpha = Image.fromarray((arr * ramp[:, None]).astype(np.uint8))

    masked = Image.new("RGB", edge.size, "black")
    masked.paste(edge, (0, 0), alpha.resize(edge.size))
    return masked


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


def generate(photo_bytes: bytes, control_scale: float, steps: int):
    """(결과 PNG, 윤곽선 PNG, 전처리 입력 PNG, 얼굴검출여부) 반환."""
    photo, face_found = prepare(Image.open(io.BytesIO(photo_bytes)))
    control = mask_background(photo, edge_detector()(photo))
    result = pipeline()(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        image=control,
        controlnet_conditioning_scale=control_scale,
        num_inference_steps=steps,
        guidance_scale=7.5,
    ).images[0]
    return to_png(result), to_png(control), to_png(photo), face_found


st.title("MonSnap 🐲")
st.caption("사진을 찍으면 나만의 몬스터가 태어납니다")

with st.sidebar:
    st.subheader("생성 설정")
    # PRD §12-4(마스코트형 ↔ 크리처형)는 결국 이 숫자 하나다
    control_scale = st.slider(
        "형태 반영 강도", 0.0, 1.2, 0.65, 0.05,
        help="0.5=귀엽지만 덜 닮음 · 0.65=권장(균형) · 0.8=닮지만 사람에 가까움",
    )
    steps = st.slider("스텝 수", 10, 40, 25, 5, help="높을수록 느리고 정교함")
    show_edge = st.checkbox("윤곽선 같이 보기", value=True)

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

        label = "다시 생성" if "result" in st.session_state else "몬스터 생성"
        if st.button(label, type="primary"):
            t0 = time.monotonic()
            try:
                with st.spinner("몬스터 소환 중…"):
                    result, edge, crop, face_found = generate(photo_bytes, control_scale, steps)
                st.session_state["result"] = result
                st.session_state["edge"] = edge
                st.session_state["crop"] = crop
                st.session_state["face_found"] = face_found
                log.info(
                    "generate ok elapsed=%.1fs scale=%.2f steps=%d",
                    time.monotonic() - t0, control_scale, steps,
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

            if not st.session_state.get("face_found", True):
                st.warning(
                    "얼굴을 찾지 못해 사진 중앙을 잘라 썼습니다. 정면·밝은 곳에서 "
                    "얼굴이 크게 나오도록 다시 찍으면 특징이 훨씬 잘 반영됩니다.",
                    icon="⚠️",
                )

            if show_edge and "edge" in st.session_state:
                st.caption("모델이 실제로 받은 입력 — 특징이 여기 안 잡혔으면 사진을 다시 찍는 게 빠릅니다")
                e1, e2 = st.columns(2)
                e1.image(st.session_state["crop"], caption="얼굴 크롭", use_container_width=True)
                e2.image(st.session_state["edge"], caption="윤곽선", use_container_width=True)
            st.download_button(
                "이미지 저장",
                data=st.session_state["result"],
                file_name="monster.png",
                mime="image/png",
            )
            st.caption("마음에 안 드시면 '다시 생성'을 눌러 보세요 — 매번 다른 결과가 나옵니다.")

st.divider()
st.caption("사진은 이 PC 안에서만 처리됩니다. 외부로 전송되지 않습니다.")
