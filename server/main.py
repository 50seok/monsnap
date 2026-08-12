import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
PROMPT = (
    "Transform the person in this photo into a cute, round, original monster "
    "creature character, like a caricature: keep their distinctive features "
    "(hairstyle, glasses, face shape, expression) clearly recognizable. "
    "Vibrant colors, anime-inspired, full body, clean white background. "
    "Original design only - do not imitate any copyrighted characters."
)
MAX_UPLOAD = 10 * 1024 * 1024

app = FastAPI(title="monsnap")

_client: genai.Client | None = None


def client() -> genai.Client:
    # 지연 초기화: 키 없이도 서버가 뜨고, /api/generate 호출 시점에만 키를 요구한다
    global _client
    if _client is None:
        _client = genai.Client()  # GEMINI_API_KEY 환경변수 사용
    return _client


@app.post("/api/generate")
def generate(photo: UploadFile):
    # sync def → FastAPI 스레드풀 실행이라 30~120초 생성 동안 이벤트루프를 막지 않는다
    if not (photo.content_type or "").startswith("image/"):
        raise HTTPException(400, "이미지 파일만 업로드할 수 있습니다.")
    data = photo.file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(400, "10MB 이하 이미지만 지원합니다.")

    try:
        resp = client().models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=data, mime_type=photo.content_type),
                PROMPT,
            ],
        )
    except Exception as e:  # noqa: BLE001 — 외부 API 실패는 502로 표면화
        raise HTTPException(502, f"이미지 생성 API 오류: {e}")

    cand = resp.candidates[0] if resp.candidates else None
    parts = cand.content.parts if cand and cand.content else None
    for part in parts or []:
        if part.inline_data and part.inline_data.data:
            return Response(
                content=part.inline_data.data,
                media_type=part.inline_data.mime_type or "image/png",
            )
    raise HTTPException(502, "생성 결과에 이미지가 없습니다 (안전 필터 차단 가능성).")
