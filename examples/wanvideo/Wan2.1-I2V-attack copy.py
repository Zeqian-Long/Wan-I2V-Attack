import torch
import sys
diffsynth_path = "/workspace/DiffSynth-Studio"
sys.path.append(diffsynth_path)
from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline, prompt_clip_attn_loss, model_fn_wan_video
from diffsynth.data.video import save_video, VideoData, LowMemoryImageFolder
from PIL import Image

from diffsynth.utils import crop_and_resize, register_vae_hooks, setup_pipe_modules, init_adv_image, plot_loss_curve, save_adv_result

from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random


# ======================================================================
#                          LOADING MODELS
# ======================================================================
def load_all_models():
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        ["/workspace/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float16, 
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
        torch_dtype=torch.bfloat16,
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe = setup_pipe_modules(pipe)
    return pipe


# ======================================================================
#                       PREPARE IMAGES, PROMPTS, EMBEDDINGS
# ======================================================================
def prepare_data(pipe):
    h = 480
    w = 832

    image = Image.open("data/image/lucia.jpg").resize((w, h))
    mask = Image.open("data/mask/swan_224.jpg").convert("L")

    num_frames = 5

    prompt = "在一间无限延伸的绿色实验室中，空气被冰冷的光线分割成无数几何面。地面反射着荧光蓝与猩红的光带，像液态的玻璃在呼吸。墙壁上悬浮着巨大的数字与方程，它们在慢慢旋转，投下灰绿色的阴影。中央站着一个身披银灰外衣的身影，眼中闪烁着金光，周围漂浮着数以百计的透明立方体，每一个立方体里都封存着一段记忆的闪光。突然，顶端的灯光爆裂成黑白交错的涡流，空间开始塌陷，色彩被撕裂成纯粹的蓝与橙的对撞。声音消失，只剩下一种光的震颤，像宇宙在深呼吸。"

    pipe.load_models_to_device(["text_encoder"])
    with torch.no_grad():
        prompt_emb_posi = pipe.encode_prompt(prompt=prompt, positive=True)
    print(prompt_emb_posi['context'].shape)

    tiler_kwargs = {"tiled": False, "tile_size": (h / 16, w / 16), "tile_stride": (h / 32, w / 32)}

    pipe.load_models_to_device(["image_encoder", "vae"])
    with torch.no_grad():
        image_emb_src = pipe.encode_image(
            image, num_frames=num_frames, height=h, width=w, **tiler_kwargs
        )
    print("Source Image Embedding Shape:", image_emb_src["clip_feature"].shape)

    noise = pipe.generate_noise(
        (1, 16, (num_frames - 1) // 4 + 1, h//8, w//8), 
        seed=0, device="cpu", dtype=torch.float32
    )
    noise = noise.to(dtype=pipe.torch_dtype, device=pipe.device)
    return h, w, num_frames, tiler_kwargs, image, prompt_emb_posi, image_emb_src, noise


# ======================================================================
#                           MAIN ATTACK LOGIC
# ======================================================================
def run_attack(pipe, h, w, num_frames, tiler_kwargs,
               prompt_emb_posi, image_emb_src, noise):

    latents_list = []
    latents = noise
    extra_input = pipe.prepare_extra_input(latents)

    pipe.scheduler.set_timesteps(num_inference_steps=25, denoising_strength=1.0, shift=5.0)
    pipe.load_models_to_device(["dit"])

    # ------------------- Create latent list -------------------
    with torch.no_grad():
        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
            latents_list.append(latents.detach().cpu())
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = model_fn_wan_video(
                pipe.dit, latents, timestep=timestep,
                **prompt_emb_posi, **image_emb_src, **extra_input
            )
            latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], latents)

    pipe = setup_pipe_modules(pipe)
    saved_features = register_vae_hooks(pipe)

    target_image = Image.open("data/image/MIST_Repeated.png").convert("RGB").resize((w, h))
    with torch.no_grad():
        image_emb_tgt = pipe.encode_image(
            target_image, num_frames=num_frames, height=h, width=w, **tiler_kwargs
        )

    target_features = copy.deepcopy(saved_features)
    saved_features = {}

    # ------------------- PGD Attack -------------------
    I_adv = pipe.preprocess_image(Image.open("data/image/lucia.jpg").resize((w, h))).to(pipe.device).detach().requires_grad_(True)
    I_adv_before = I_adv.clone().detach()

    num_steps = 400
    epsilon = 20.0 / 255 * 2
    step_size = epsilon / 10
    I_adv = init_adv_image(I_adv, epsilon=epsilon, value_range=(-1.0, 1.0))

    loss_history = []

    for step in tqdm(range(num_steps), desc="Optimizing"):
        if I_adv.grad is not None:
            I_adv.grad.zero_()

        pipe.load_models_to_device(["vae", "image_encoder"])
        image_emb_adv = pipe.encode_image(I_adv, num_frames=num_frames, height=h, width=w, **tiler_kwargs)

        L_enc_1 = torch.nn.functional.mse_loss(image_emb_adv["y"][:, 4:, :], image_emb_tgt["y"][:, 4:, :])

        pipe.scheduler.set_timesteps(num_inference_steps=25, denoising_strength=1.0, shift=5.0)
        idx = random.randrange(len(pipe.scheduler.timesteps))

        adv_latents = latents_list[idx].to(dtype=pipe.torch_dtype, device=pipe.device)
        timestep = pipe.scheduler.timesteps[idx].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        extra_input = pipe.prepare_extra_input(adv_latents)

        pipe.load_models_to_device(["dit"])
        attn_loss = prompt_clip_attn_loss(
            pipe.dit, adv_latents, timestep=timestep,
            **prompt_emb_posi, **image_emb_adv, **extra_input
        )

        L = attn_loss
        print(f"Step {step+1}/{num_steps}, Loss: {L.item():.6f}")
        loss_history.append(L.item())
        L.backward()

        if step == 100:
            step_size *= 0.5
        if step == 200:
            step_size *= 0.5

        sgn = I_adv.grad.data.sign()
        I_adv.data = I_adv.data - step_size * sgn
        delta = torch.clamp(I_adv - I_adv_before, min=-epsilon, max=epsilon)
        I_adv.data = torch.clamp(I_adv_before + delta, -1.0, 1.0)

    plot_loss_curve(loss_history)
    metrics = save_adv_result(I_adv, I_adv_before, save_path="I_adv_final_lucia.jpg")



# ======================================================================
#                                MAIN
# ======================================================================
def main():
    pipe = load_all_models()
    h, w, num_frames, tiler_kwargs, image, prompt_emb_posi, image_emb_src, noise = prepare_data(pipe)
    run_attack(pipe, h, w, num_frames, tiler_kwargs, prompt_emb_posi, image_emb_src, noise)


if __name__ == "__main__":
    main()
