import io
import logging
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

# 로컬 생성 스택: SD1.5 파생 모델 + ControlNet softedge(형태 유지)
# ponytail: 베이스 모델은 secrets로 교체 가능 — LoRA 자체 학습 후 여기만 바꾸면 됨
BASE_MODEL = st.secrets.get("BASE_MODEL", "lambdalabs/sd-pokemon-diffusers")
CONTROLNET = "lllyasviel/control_v11p_sd15_softedge"
SIZE = 512  # SD1.5 네이티브 해상도

PROMPT = (
    "a cute original monster creature, chibi, big expressive eyes, "
    "vibrant colors, simple clean background, character art"
)
# "human/person" 억제 = 윤곽은 ControlNet이 주고 내용물은 몬스터로 채우게 만드는 핵심
NEG_PROMPT = (
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
    from controlnet_aux import HEDdetector

    return HEDdetector.from_pretrained("lllyasviel/Annotators")


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
    """EXIF 회전 보정 → 얼굴 크롭 → 정사각 리사이즈. (이미지, 얼굴검출여부) 반환."""
    img = ImageOps.exif_transpose(img).convert("RGB")  # 휴대폰 사진은 회전정보가 EXIF에 있다
    box = face_box(img)
    found = box is not None
    if box is None:
        side = min(img.size)  # 폴백: 중앙 정사각 크롭
        box = ((img.width - side) // 2, (img.height - side) // 2)
        box = (box[0], box[1], box[0] + side, box[1] + side)
    return img.crop(box).resize((size, size), Image.LANCZOS), found


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate(photo_bytes: bytes, control_scale: float, steps: int):
    """(결과 PNG, 윤곽선 PNG, 전처리 입력 PNG, 얼굴검출여부) 반환."""
    photo, face_found = prepare(Image.open(io.BytesIO(photo_bytes)))
    control = edge_detector()(photo, scribble=False)
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
        "형태 반영 강도", 0.0, 1.2, 0.6, 0.05,
        help="낮음 = 크리처형(형태 힌트만) · 높음 = 마스코트형(사람 윤곽 뚜렷)",
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
