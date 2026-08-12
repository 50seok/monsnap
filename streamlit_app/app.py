import logging
import time

import streamlit as st
from google import genai
from google.genai import types

MODEL = st.secrets.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
PROMPT = (
    "Transform the person in this photo into a cute, round, original monster "
    "creature character, like a caricature: keep their distinctive features "
    "(hairstyle, glasses, face shape, expression) clearly recognizable. "
    "Vibrant colors, anime-inspired, full body, clean white background. "
    "Original design only - do not imitate any copyrighted characters."
)
MAX_UPLOAD = 10 * 1024 * 1024
TIMEOUT_MS = 150_000  # PRD §5: 클라이언트 타임아웃 150초

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("monsnap")

st.set_page_config(page_title="MonSnap", page_icon="🐲")


class SafetyBlockedError(Exception):
    pass


@st.cache_resource
def client() -> genai.Client:
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=TIMEOUT_MS),
    )


def generate(photo_bytes: bytes, mime_type: str) -> bytes:
    """생성 성공 시 PNG 바이트 반환. 실패 시 예외."""
    resp = client().models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=photo_bytes, mime_type=mime_type), PROMPT],
    )
    cand = resp.candidates[0] if resp.candidates else None
    parts = cand.content.parts if cand and cand.content else None
    for part in parts or []:
        if part.inline_data and part.inline_data.data:
            return part.inline_data.data
    raise SafetyBlockedError()  # 응답은 왔지만 이미지가 없음 = 안전 필터 차단 (FR-4)


st.title("MonSnap 🐲")
st.caption("사진을 찍으면 나만의 몬스터가 태어납니다")

source = st.radio("사진 방법", ["앨범에서 선택", "카메라로 촬영"], horizontal=True, label_visibility="collapsed")
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
                with st.spinner("몬스터 소환 중… (최대 2분)"):
                    st.session_state["result"] = generate(photo_bytes, mime_type)
                log.info("generate ok elapsed=%.1fs", time.monotonic() - t0)
            except SafetyBlockedError:
                st.session_state.pop("result", None)
                log.info("generate blocked elapsed=%.1fs", time.monotonic() - t0)
                st.error("이 사진은 생성할 수 없습니다. 다른 사진을 선택해 주세요.")
            except Exception:
                st.session_state.pop("result", None)
                log.exception("generate failed elapsed=%.1fs", time.monotonic() - t0)
                st.error("일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")

        if "result" in st.session_state:
            col2.image(st.session_state["result"], caption="몬스터", use_container_width=True)
            st.download_button(
                "이미지 저장",
                data=st.session_state["result"],
                file_name="monster.png",
                mime="image/png",
            )
            st.caption("마음에 안 드시면 '다시 생성'을 눌러 보세요 — 매번 다른 결과가 나옵니다.")

st.divider()
st.caption("생성을 위해 사진이 구글 Gemini API로 전송됩니다. 서버에는 저장되지 않습니다.")
