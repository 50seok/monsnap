"""strength=0.7에서 팔레트(속성 색) 견인력 실측 그리드.

계기(PRD §13): strength를 0.85→0.7로 낮춰 원형 구조가 잘 살아남게 됐는데,
그 구조에는 원형의 원래 색도 포함돼 있어 팔레트 견인이 오히려 약해졌을
가능성 — 검증 없이 보류됐던 항목.

FaceID는 0으로 꺼서(identity 임베딩 무관) strength+프롬프트+LoRA만으로 팔레트가
얼마나 살아나는지 순수 격리해서 본다. 레퍼런스는 원본이 초록(tree_frog)이라
불(주황)·물(파랑)·전기(노랑)·페어리(분홍) 4개 타입과 뚜렷이 대비되는 케이스.

⚠ 포그라운드 전용 — 백그라운드 래퍼로 띄우면 첫 생성에서 조용히 죽는 사례
3회 확인됨(PRD §13 실행 함정). 모델 로드 ~1분 + 5장 생성 ~수 분 예상.

사용법: cd streamlit_app && ..\.venv\Scripts\python ..\tools\probe_palette.py
"""
import sys
import tomllib
from pathlib import Path

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, UniPCMultistepScheduler
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ROOT / "streamlit_app" / ".streamlit" / "secrets.toml"
_LOCAL_LORA = ROOT / "streamlit_app" / "models" / "style_lora_sdxl"

_secrets = tomllib.load(open(SECRETS, "rb")) if SECRETS.exists() else {}
BASE_MODEL = _secrets.get("BASE_MODEL", "cagliostrolab/animagine-xl-4.0")
STYLE_LORA = _secrets.get(
    "STYLE_LORA",
    str(_LOCAL_LORA) if (_LOCAL_LORA / "pytorch_lora_weights.safetensors").exists() else "",
)

# app.py의 SHAPE_PHRASES/POSE_PHRASES/TYPES/NEG_PROMPT/build_prompt와 동일 로직
# (app.py는 최상단에 st.* 호출이 있어 모듈로 import하면 Streamlit 컨텍스트 밖에서
# 깨진다 — 프롬프트 로직만 발췌. app.py 프롬프트 변경 시 여기도 같이 고칠 것)
SHAPE_PHRASES = {"legs": "two clearly separated legs"}
TYPES = [
    ("불", "fiery orange and red", "a small flame on its tail tip, tiny claws on its paws"),
    ("물", "blue and aqua", "fin ears and a droplet-shaped tail"),
    ("풀", "fresh green", "a leaf sprout on its head and a leafy tail"),
    ("전기", "bright yellow", "a lightning-shaped tail and bright cheek marks"),
    ("페어리", "pastel pink and white", "ribbon ears and a fluffy curled tail"),
]
NEG_PROMPT = (
    "monster, scary, evil, sharp teeth, dark, horror, "
    "human, person, realistic, photograph, text, watermark, "
    "blurry, deformed, extra limbs, faceless, empty eyes, blank eyes, "
    "messy mouth, distorted mouth, extra mouth, noisy lines, "
    "lowres, bad anatomy, worst quality, low quality"
)


def build_prompt(palette: str, shape: str, signature: str) -> str:
    part = f"{SHAPE_PHRASES[shape]}, " if shape in SHAPE_PHRASES else ""
    return (f"cutemon, a {palette} creature, {signature}, white background, "
            f"no humans, solo, full body, standing, a cute simple face, "
            f"round dot eyes, a tiny clean simple smile, "
            f"{part}flat colors, clean thin lineart")


REF_NAME, REF_SHAPE = "tree_frog", "legs"
STRENGTH, STEPS, SEED, GEN_SIZE = 0.7, 30, 12345, 1024

print(f"BASE_MODEL={BASE_MODEL}\nSTYLE_LORA={STYLE_LORA or '(none)'}", flush=True)

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter(
    "h94/IP-Adapter-FaceID", subfolder=None,
    weight_name="ip-adapter-faceid_sdxl.bin", image_encoder_folder=None,
)
if STYLE_LORA:
    pipe.load_lora_weights(STYLE_LORA, adapter_name="style")
    pipe.set_adapters(["faceid_0", "style"], adapter_weights=[1.0, 1.0])
pipe.set_ip_adapter_scale(0.0)  # 팔레트 순수 격리 — identity 조건 없음
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()
print("pipeline ready", flush=True)

zero_emb = torch.zeros((2, 1, 512), dtype=torch.float16, device="cuda")
ref_img = Image.open(ROOT / "dataset" / "proto_dex" / f"{REF_NAME}.png").convert("RGB").resize((GEN_SIZE, GEN_SIZE))

cell = 320
grid = Image.new("RGB", (cell * (len(TYPES) + 1), cell + 30), "white")
d = ImageDraw.Draw(grid)
grid.paste(ref_img.resize((cell, cell)), (0, 30))
d.text((4, 4), f"ref:{REF_NAME}", fill="black")

for i, (type_ko, palette, signature) in enumerate(TYPES, start=1):
    prompt = build_prompt(palette, REF_SHAPE, signature)
    print(f"[{i}/{len(TYPES)}] {type_ko}: {prompt}", flush=True)
    out = pipe(
        prompt=prompt, negative_prompt=NEG_PROMPT, image=ref_img,
        strength=STRENGTH, ip_adapter_image_embeds=[zero_emb],
        num_inference_steps=STEPS, guidance_scale=9.0,
        generator=torch.Generator("cpu").manual_seed(SEED),
    ).images[0]
    grid.paste(out.resize((cell, cell)), (i * cell, 30))
    d.text((i * cell + 4, 4), type_ko, fill="black")

REVIEW = ROOT / "dataset" / "review"
REVIEW.mkdir(exist_ok=True)
out_path = REVIEW / "probe_palette_strength07.png"
grid.save(out_path)
print(f"saved: {out_path}", flush=True)
