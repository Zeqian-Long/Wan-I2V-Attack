import torch
import sys
diffsynth_path = "/workspace/DiffSynth-Studio"
sys.path.append(diffsynth_path)
from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline, model_fn_wan_video_attack, prompt_clip_attn_loss
from diffsynth.data.video import save_video, VideoData, LowMemoryImageFolder
from PIL import Image

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["/workspace/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float16, # Image Encoder is loaded with float16
)
model_manager.load_models(
    [
        [
            "/workspace/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "/workspace/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "/workspace/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "/workspace/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "/workspace/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "/workspace/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "/workspace/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "/workspace/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
        "/workspace/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")


# --------------------------------------------- Testing ---------------------------------------------

h = 480
w = 832

image = Image.open("I_adv_final.jpg")
image = image.resize((w, h))
pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

video = pipe(
    prompt="A swan is swimming in the river.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=image, target_image=Image.open("data/dog.png").resize((w, h)),
    num_inference_steps=15, height=h, width=w,
    seed=0, tiled=True, num_frames=17
)
save_video(video, "swan_attacked.mp4", fps=15, quality=5)

