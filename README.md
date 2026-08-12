# MonSnap

> **문서 버전:** v0.1 · 최종 수정: 2026-08-08

인물 사진을 찍으면 그 사람의 특징(캐리커처식)을 반영한 오리지널 몬스터 캐릭터를 생성하는 앱.
비상업 · 학습/포트폴리오 목적 프로젝트.

## 구조

```
frontend/  React 19 + Vite (카메라·업로드·결과 화면)
server/    FastAPI 프록시 (API 키 은닉, Gemini 이미지 API 호출)
```

## 실행 (로컬)

### 1. 서버

```bash
cd server
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env         # GEMINI_API_KEY 채우기 (https://aistudio.google.com/apikey)
uvicorn main:app --reload
```

### 2. 프론트

```bash
cd frontend
npm install
npm run dev
```

http://localhost:5173 접속 → 사진 선택 → "몬스터 생성".
모바일 실기기 테스트: `npm run dev -- --host` 후 같은 Wi-Fi에서 PC IP로 접속.

## 로드맵

1. **Phase 1 (현재)** — Gemini 이미지 API 기반 웹 MVP. 배포: Cloudflare Pages(프론트) + OCI 무료 VM(서버)
2. **Phase 2** — 직접 학습: fal.ai에서 FLUX.1-dev 스타일 LoRA 파인튜닝 + PuLID 정체성 어댑터로 교체 (`server/main.py`의 생성 호출부만 교체하면 됨)
3. **Phase 3** — Expo(React Native) 앱 전환, 스토어 배포

## 주의

- API 키는 `server/.env`에만 둔다. 프론트 코드·커밋에 절대 포함 금지.
- 생성 프롬프트는 오리지널 크리처 컨셉만 사용 — 타사 IP(포켓몬 등) 명칭·디자인 모방 금지.
