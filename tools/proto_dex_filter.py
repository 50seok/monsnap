"""원형 도감 후보(dataset/animals/*.png) 사전 필터 — 배치 생성(GPU) 전에 걸러낸다.

계기: 49→45 사후 정리에서 제외된 4종 중 budgerigar는 rembg가 배경 제거에
실패해 원본이 사실상 빈 캔버스였고, 그 빈 원본이 생성 후보를 색 빠진 흐릿한
결과로 만들었다(원인→결과 인과관계 실측 확인). deer_fawn·hamster는 원본에
동물이 2마리(hamster는 거울 반사 포함) 겹쳐 있었다. 이 세 가지는 원본 사진
단계에서 걸러낼 수 있었던 것 — GPU 배치 생성(종당 4시드)을 돌리기 전에 걸러
그만큼의 GPU 시간을 아낀다.

⚠ 범위 밖: bear_cub(원본·생성본 다 멀쩡하나 결과 디자인이 "부위 경계 없는
덩어리"라는 구도적 판단으로 제외)류는 이 필터로 못 잡는다 — 그건 여전히
사람이 생성 결과를 보고 판단해야 한다. 이 필터는 딱 두 가지, ① 빈 캔버스
② 다중 피사체만 자동 검출한다.

사용법:
    python tools/proto_dex_filter.py                    # dataset/animals/*.png 전체 점검
    python tools/proto_dex_filter.py dataset/animals/new_species.png  # 개별 파일

종료 코드: 결함 있는 파일이 하나라도 있으면 1 (CI/스크립트 체이닝용).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")  # Windows cp949 콘솔에서 한글 깨짐/크래시 방지

ROOT = Path(__file__).resolve().parent.parent
NEAR_WHITE = 245          # 이 값 이상인 채널을 "배경"으로 취급 (흰 배경 합성 관례와 일치)
BLANK_FG_RATIO = 0.03     # 전경 비율이 이 미만이면 "거의 빈 캔버스"
DUP_AREA_RATIO = 0.15     # 2번째로 큰 덩어리가 1번째의 이 비율 이상이면 "다중 피사체 의심"


def _fg_mask(img: Image.Image) -> np.ndarray:
    """흰 배경이 아닌(=전경) 픽셀 마스크. RGBA면 알파도 함께 고려."""
    arr = np.asarray(img.convert("RGB"))
    non_white = (arr < NEAR_WHITE).any(axis=-1)
    if img.mode == "RGBA":
        alpha = np.asarray(img.getchannel("A"))
        non_white &= alpha > 10
    return non_white


def _components(mask: np.ndarray) -> list[int]:
    """4-연결 성분 크기 목록(내림차순). scipy 없이 BFS로 직접 구현(의존성 최소화)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys, xs):
        if seen[sy, sx]:
            continue
        stack, size = [(sy, sx)], 0
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            size += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def check(path: Path) -> dict:
    """한 이미지를 점검. {blank, multi_subject, fg_ratio, second_ratio} 반환."""
    img = Image.open(path)
    mask = _fg_mask(img)
    fg_ratio = mask.mean()
    if fg_ratio < BLANK_FG_RATIO:
        return {"blank": True, "multi_subject": False, "fg_ratio": fg_ratio, "second_ratio": 0.0}

    # 다운샘플 후 성분 분석 — 원본 크기(512²)에서 BFS는 느리고 불필요
    small = np.asarray(Image.fromarray(mask).resize((128, 128), Image.NEAREST))
    sizes = _components(small)
    second_ratio = (sizes[1] / sizes[0]) if len(sizes) > 1 and sizes[0] > 0 else 0.0
    return {
        "blank": False,
        "multi_subject": second_ratio >= DUP_AREA_RATIO,
        "fg_ratio": fg_ratio,
        "second_ratio": second_ratio,
    }


def main():
    targets = (
        [Path(a) for a in sys.argv[1:]]
        if len(sys.argv) > 1
        else sorted((ROOT / "dataset" / "animals").glob("*.png"))
    )
    bad = []
    for p in targets:
        r = check(p)
        flag = "BLANK" if r["blank"] else "DUP" if r["multi_subject"] else "ok"
        print(f"{flag:>5}  fg={r['fg_ratio']:.3f}  2nd={r['second_ratio']:.3f}  {p.name}")
        if flag != "ok":
            bad.append(p.name)

    print(f"\n{len(targets)}개 중 {len(bad)}개 결함: {', '.join(bad) if bad else '없음'}")
    print("(참고) 구도상 '덩어리형' 판단은 이 필터 범위 밖 — 생성 후 육안 확인 필요")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
