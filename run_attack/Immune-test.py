import torch
import sys
import os
import yaml
from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.data.video import save_video, VideoData, LowMemoryImageFolder
from diffsynth.utils import setup_pipe_modules
from PIL import Image

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float16, # Image Encoder is loaded with float16
)
model_manager.load_models(
    [
        [
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")


# --------------------------------------------- Testing ---------------------------------------------

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

h = cfg["video"]["height"]
w = cfg["video"]["width"]
num_frames = cfg["video"]["num_frames"]
seed = 0

image_path = cfg["data"]["image_path"]
image_name = os.path.basename(image_path)
name_wo_ext = os.path.splitext(image_name)[0]

good_test_prompts = cfg["prompt"]["good_test"]
assert len(good_test_prompts) == 4, f"config.yaml prompt.good_test must have exactly 4 entries, got {len(good_test_prompts)}"

adv_image_path = os.path.join("attacked/images", f"{name_wo_ext}.jpg")
if not os.path.exists(adv_image_path):
    raise FileNotFoundError(
        f"Adversarial image not found at {adv_image_path}. Please run Immune-attack.py first."
    )

image_variants = {
    "clean": Image.open(image_path).resize((w, h)),
    "adv": Image.open(adv_image_path).resize((w, h)),
}

# pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
pipe = setup_pipe_modules(pipe)


os.makedirs("attacked/videos", exist_ok=True)
os.makedirs("attacked/frames", exist_ok=True)

for variant_name, variant_image in image_variants.items():
    for prompt_idx, prompt_text in enumerate(good_test_prompts):
        video = pipe(
            prompt=prompt_text,
            input_image=variant_image,
            num_inference_steps=25, height=h, width=w,
            seed=seed, tiled=False, num_frames=num_frames, cfg_scale=5,
        )

        tag = f"{name_wo_ext}_{variant_name}_test{prompt_idx}_seed{seed}"

        video_path = os.path.join("attacked/videos", f"{tag}.mp4")
        save_video(video, video_path, fps=10, quality=5)

        out_dir = os.path.join("attacked/frames", tag)
        os.makedirs(out_dir, exist_ok=True)
        for i, frame in enumerate(video):
            frame.save(os.path.join(out_dir, f"{i:04d}.png"))

        print(f"Saved {video_path}")
