import torch
import sys
from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline, model_fn_wan_video
from diffsynth.utils import register_vae_hooks, setup_pipe_modules
from PIL import Image
from tqdm import tqdm
import yaml
import os


def load_all_models():
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        ["models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float16,
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
        torch_dtype=torch.bfloat16,
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe = setup_pipe_modules(pipe, attack=False)
    return pipe


def prepare_data(pipe, image, good_train_prompts, h=480, w=832, num_frames=1):

    # Encode the good-training prompt bank
    pipe.load_models_to_device(["text_encoder"])
    with torch.no_grad():
        good_bank = [pipe.encode_prompt(prompt=p, positive=True) for p in good_train_prompts]   # each: {"context": [1, 512, 4096]}

    # Modify if needed, usually tiled = False
    tiler_kwargs = {"tiled": False, "tile_size": (h / 16, w / 16), "tile_stride": (h / 32, w / 32)}

    # Encode Image (computed once, reused for good_bank's structural reference
    # and for every collapse prompt's trajectory generation below)
    pipe.load_models_to_device(["image_encoder", "vae"])
    with torch.no_grad():
        image_emb_src = pipe.encode_image(image, num_frames=num_frames, height=h, width=w, **tiler_kwargs)   # clip: [1, 1 + 256, 1280], y: [1, C (4+16), 1+T/4, 60, 104]
        # image_emb_src["y"][0, 4:] is also used as the fixed clean reference z_t(I)
        # for the structural-conditioning attack loss in Immune-attack.py.

    return good_bank, image_emb_src


def obtain_latent_sequence(pipe, h, w, num_frames, prompt_emb, image_emb_src, num_inference_steps=25, preprocess_cfg_scale=5.0):

    noise = pipe.generate_noise(
        (1, 16, (num_frames - 1) // 4 + 1, h//8, w//8),
        seed=0, device="cpu", dtype=torch.float32
    )
    noise = noise.to(dtype=pipe.torch_dtype, device=pipe.device)

    prompt_emb_nega = pipe.encode_prompt("", positive=False)

    latents_list = []
    latents = noise

    extra_input = pipe.prepare_extra_input(latents)

    pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, denoising_strength=1.0, shift=5.0)
    pipe.load_models_to_device(["dit"])


    with torch.no_grad():
        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
            latents_list.append(latents.detach().cpu())
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            noise_pred_posi = model_fn_wan_video(pipe.dit, latents, timestep=timestep, **prompt_emb, **image_emb_src, **extra_input)
            noise_pred_nega = model_fn_wan_video(pipe.dit, latents, timestep=timestep, **prompt_emb_nega, **image_emb_src, **extra_input)

            # cfg scale
            noise_pred = noise_pred_nega + preprocess_cfg_scale * (noise_pred_posi - noise_pred_nega)
            latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], latents)

    return latents_list


def prepare_collapse_bank(pipe, collapse_prompts, h, w, num_frames, image_emb_src, num_inference_steps=25, preprocess_cfg_scale=5.0):
    """
    For each collapse prompt, encode the prompt and generate its own real
    denoising trajectory from the clean source image (image_emb_src is computed
    once by prepare_data and reused unchanged here for every collapse prompt).
    The prompt embedding and its trajectory are kept together in a single dict
    per collapse prompt so the pairing (p_k^c <-> T_k^c) can never drift out of
    sync, unlike two independently-indexed parallel lists.
    """
    pipe.load_models_to_device(["text_encoder"])
    collapse_bank = []
    for i, prompt in enumerate(collapse_prompts):
        print(f"Generating collapse trajectory {i + 1}/{len(collapse_prompts)}...")
        with torch.no_grad():
            prompt_emb = pipe.encode_prompt(prompt=prompt, positive=True)
        latents_list = obtain_latent_sequence(
            pipe, h, w, num_frames, prompt_emb, image_emb_src,
            num_inference_steps=num_inference_steps, preprocess_cfg_scale=preprocess_cfg_scale,
        )
        collapse_bank.append({"prompt_emb": prompt_emb, "latents_list": latents_list})
    return collapse_bank


def main():
    pipe = load_all_models()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    h = cfg["video"]["height"]
    w = cfg["video"]["width"]

    num_frames = cfg["video"]["num_frames"]
    num_inference_steps = cfg["video"]["denoising_steps"]
    preprocess_cfg_scale = cfg["video"]["preprocess_cfg_scale"]

    image = Image.open(cfg["data"]["image_path"]).resize((w, h))

    good_train_prompts = cfg["prompt"]["good_train"]
    good_test_prompts = cfg["prompt"]["good_test"]
    collapse_prompts = cfg["prompt"]["collapse"]

    assert len(good_train_prompts) == 6, f"config.yaml prompt.good_train must have exactly 6 entries, got {len(good_train_prompts)}"
    assert len(good_test_prompts) == 4, f"config.yaml prompt.good_test must have exactly 4 entries, got {len(good_test_prompts)}"
    assert len(collapse_prompts) == 4, f"config.yaml prompt.collapse must have exactly 4 entries, got {len(collapse_prompts)}"

    # good_test_prompts are intentionally validated but NOT encoded/cached here:
    # they must never participate in attack preprocessing or PGD optimization,
    # only in the held-out evaluation pass performed later by Immune-test.py.

    good_bank, image_emb_src = prepare_data(pipe, image, good_train_prompts, h=h, w=w, num_frames=num_frames)

    collapse_bank = prepare_collapse_bank(
        pipe, collapse_prompts, h, w, num_frames, image_emb_src,
        num_inference_steps=num_inference_steps, preprocess_cfg_scale=preprocess_cfg_scale,
    )

    os.makedirs("cache", exist_ok=True)
    torch.save(good_bank, "cache/good_bank.pt")
    torch.save(collapse_bank, "cache/collapse_bank.pt")
    torch.save(image_emb_src, "cache/image_emb_src.pt")
    print("Saved to cache/")

if __name__ == "__main__":
    main()
