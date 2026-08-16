"""app.py의 실제 generate()/make_card()를 Streamlit 서버 없이 배치로 돌린다.

지금까지 이 세션은 실사진이 없어 팔레트·색상 실험을 격리된 이미지로만
검증했다 — 이 스크립트로 실사진 다수를 실제 프로덕션 코드 경로(app.py) 그대로
돌려 e2e 품질(눈 가시성 등)을 확인한다. app.py를 재구현하지 않고 AST로
함수 정의부만 그대로 가져온다(st.title(...) 이전까지) — 원본과 드리프트 없음.

⚠ 포그라운드 전용(PRD §13 실행 함정과 동일 이유 — 백그라운드 래퍼는 조용히 죽음).

사용법: cd streamlit_app && ..\.venv\Scripts\python ..\tools\batch_generate.py "<사진 폴더>"
출력:   dataset/review/batch_e2e/<원본파일명>_card.png
"""
import ast
import hashlib
import sys
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "streamlit_app" / "app.py"
SECRETS_PATH = ROOT / "streamlit_app" / ".streamlit" / "secrets.toml"


class _St:
    """app.py가 함수 정의부에서 쓰는 st.secrets / st.cache_resource만 흉내낸다."""

    def __init__(self, secrets: dict):
        self.secrets = secrets

    @staticmethod
    def cache_resource(*_a, **_k):
        def deco(fn):
            return lru_cache(maxsize=None)(fn)
        return deco


def load_core_namespace() -> dict:
    """app.py에서 st.title(...) 이전(함수·상수 정의부)까지만 exec — UI 스크립트 본문은 제외."""
    src = APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cut = next(
        i for i, node in enumerate(tree.body)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "attr", "") == "title"
    )
    core = ast.Module(body=tree.body[:cut], type_ignores=[])
    secrets = tomllib.load(open(SECRETS_PATH, "rb")) if SECRETS_PATH.exists() else {}
    ns = {"st": _St(secrets), "__name__": "monsnap_core", "__file__": str(APP)}
    exec(compile(core, str(APP), "exec"), ns)
    return ns


def dedup(photos: list[Path]) -> list[Path]:
    """바이트 동일한 중복(카톡 재전송 등)은 첫 장만 남긴다."""
    seen, unique = set(), []
    for p in photos:
        h = hashlib.sha256(p.read_bytes()).digest()
        if h in seen:
            continue
        seen.add(h)
        unique.append(p)
    return unique


def main():
    folder = Path(sys.argv[1])
    # "start:end" 슬라이스 — 한 번에 다 돌리면 시간 초과라 여러 배치로 나눠 실행
    sl = sys.argv[2] if len(sys.argv) > 2 else ":"
    start, end = (int(x) if x else None for x in sl.split(":"))

    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    unique = dedup(photos)
    print(f"발견 {len(photos)}장 -> 중복 제외 {len(unique)}장", flush=True)
    unique = unique[start:end]
    print(f"이번 배치: {len(unique)}장 (슬라이스 {sl})", flush=True)

    ns = load_core_namespace()
    generate, make_card = ns["generate"], ns["make_card"]

    out_dir = ROOT / "dataset" / "review" / "batch_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in unique:
        print(f"\n=== {p.name} ===", flush=True)
        try:
            result, ref, crop, face_found, info = generate(
                p.read_bytes(), strength=0.7, steps=30, lora_weight=1.0,
                faceid_scale=0.2, name="",
            )
        except Exception as e:
            print(f"  실패: {e!r}", flush=True)
            continue
        card = make_card(result, info)
        out_path = out_dir / f"{p.stem}_card.png"
        out_path.write_bytes(card)
        print(
            f"  얼굴검출={face_found} 속성={info['type_ko']} 특징={info['traits']} "
            f"원형유사도={info['ref_sim']:.2f} -> {out_path}", flush=True,
        )


if __name__ == "__main__":
    main()
