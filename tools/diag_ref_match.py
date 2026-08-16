"""23장 배치에서 실제로 어떤 원형(proto_dex)이 매칭됐는지 집계 — GPU 불필요(CLIP만).

계기: 사용자가 생성물에 같은 고리·코 모양 부위가 반복된다고 지적. 추측(코끼리 코?
tree_frog 특징?) 대신 match_reference()를 실제로 돌려 매칭 분포를 직접 센다.

사용법: cd streamlit_app && ..\.venv\Scripts\python ..\tools\diag_ref_match.py <사진 폴더>
"""
import ast
import sys
import tomllib
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "streamlit_app" / "app.py"
SECRETS_PATH = ROOT / "streamlit_app" / ".streamlit" / "secrets.toml"


class _St:
    def __init__(self, secrets):
        self.secrets = secrets

    @staticmethod
    def cache_resource(*_a, **_k):
        return lambda fn: lru_cache(maxsize=None)(fn)


def load_core():
    src = APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cut = next(
        i for i, n in enumerate(tree.body)
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "attr", "") == "title"
    )
    secrets = tomllib.load(open(SECRETS_PATH, "rb")) if SECRETS_PATH.exists() else {}
    ns = {"st": _St(secrets), "__name__": "monsnap_core", "__file__": str(APP)}
    exec(compile(ast.Module(body=tree.body[:cut], type_ignores=[]), str(APP), "exec"), ns)
    return ns


def main():
    folder = Path(sys.argv[1])
    ns = load_core()
    prepare, match_reference = ns["prepare"], ns["match_reference"]
    import hashlib

    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    counts = Counter()
    for p in photos:
        from PIL import Image
        photo, face_found = prepare(Image.open(p))
        seed = int.from_bytes(hashlib.sha256(p.read_bytes()).digest()[:4], "big")
        ref, sim, shape = match_reference(photo, pick=seed)
        # match_reference는 이미지만 반환 — 어떤 파일인지 알려면 인덱스 재조회 필요
        paths, embs, shapes = ns["ref_index"]()
        ok = [i for i, pp in enumerate(paths) if shapes.get(pp.name, {}).get("shape") in ns["STRUCTURED_SHAPES"]]
        import torch
        clip_feat = ns["_clip_image_features"]([photo])
        simv = (embs[ok] @ clip_feat.T).squeeze(1)
        top = torch.topk(simv, min(ns["REF_TOP_K"], len(ok))).indices.tolist()
        idx = ok[top[seed % len(top)]]
        name = paths[idx].name
        counts[name] += 1
        print(f"{p.name:45s} face={face_found!s:5s} -> {name} (shape={shape}, sim={sim:.2f})")

    print("\n=== 매칭 분포 ===")
    for name, c in counts.most_common():
        print(f"  {c:2d}x  {name}")


if __name__ == "__main__":
    main()
