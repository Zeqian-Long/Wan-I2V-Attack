import torch
import sys
from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline, prompt_img_sem_loss
from diffsynth.utils import setup_pipe_modules, plot_loss_curve, save_adv_result
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import random
import yaml
import os
import math
from collections import Counter

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
    pipe = setup_pipe_modules(pipe, attack=True)
    return pipe


def init_adv_image(I, epsilon=0.03, value_range=(-1.0, 1.0), device=None):
    if not isinstance(I, torch.Tensor):
        raise TypeError("I must be a torch.Tensor")
    I = I.detach().clone()
    if device is not None:
        I = I.to(device)
    noise = torch.empty_like(I).uniform_(-epsilon, epsilon)
    I_adv = I + noise
    I_adv = torch.clamp(I_adv, value_range[0], value_range[1])
    I_adv.requires_grad_(True)
    return I_adv


# Default hyperparameters for the untargeted Wan Structural Conditioning Attack loss.
# Overridable per-run via config.yaml: attack.lambda_cov / attack.temperature / attack.rms_epsilon.
DEFAULT_LAMBDA_COV = 0.5    # weight of the smooth-min temporal coverage term J_cov
DEFAULT_TEMPERATURE = 0.1   # softmin temperature tau used in J_cov
DEFAULT_RMS_EPSILON = 1e-6  # numerical epsilon inside the per-position RMS deviation sqrt

# Default size of the high-noise timestep pool sampled from each collapse-prompt
# trajectory for the Semantic Conditioning Attack. Overridable via
# config.yaml: attack.semantic_timestep_pool.
DEFAULT_SEMANTIC_TIMESTEP_POOL = 10


def compute_structural_loss(z_adv, z_clean, lambda_cov, temperature, rms_epsilon):
    """
    Untargeted Wan Structural Conditioning Attack loss.

    z_adv, z_clean: (C, T, H, W) VAE structural-latent tensors with the 4 mask
    channels already removed. z_clean is treated as a fixed reference (no grad).

    d_t     = RMS(z_adv_t - z_clean_t) over (C, H, W), per temporal position t
    J_avg   = mean_t d_t
    J_cov   = smooth (log-sum-exp) approximation of min_t d_t, temperature-scaled
    L_struct = -(J_avg + lambda_cov * J_cov)   # negated so PGD descent maximizes disruption

    All arithmetic is done in float32 regardless of the VAE's compute dtype.
    """
    z_adv = z_adv.float()
    z_clean = z_clean.detach().float()

    d = torch.sqrt(((z_adv - z_clean) ** 2).mean(dim=(0, 2, 3)) + rms_epsilon)  # (T,)

    j_avg = d.mean()

    num_positions = d.shape[0]
    log_mean_exp = torch.logsumexp(-d / temperature, dim=0) - math.log(num_positions)
    j_cov = -temperature * log_mean_exp

    loss = -(j_avg + lambda_cov * j_cov)

    return {
        "loss": loss,
        "j_avg": j_avg.detach(),
        "j_cov": j_cov.detach(),
        "d": d.detach(),
    }


def run_attack(pipe, image, h, w, num_frames, good_bank, collapse_bank, image_emb_src,
               num_steps=400, epsilon=20.0 / 255 * 2, step_size=2.0 / 255 * 2,
               lambda_cov=DEFAULT_LAMBDA_COV, temperature=DEFAULT_TEMPERATURE, rms_epsilon=DEFAULT_RMS_EPSILON,
               semantic_timestep_pool=DEFAULT_SEMANTIC_TIMESTEP_POOL):

    I_adv = pipe.preprocess_image(image).to(pipe.device).detach().requires_grad_(True)
    I_adv_before = I_adv.clone().detach()
    I_adv = init_adv_image(I_adv, epsilon=epsilon, value_range=(-1.0, 1.0))
    history = {
        "total_loss": [], "struct_loss": [], "j_avg": [], "j_cov": [], "attn_loss": [], "d": [],
        "good_idx": [], "collapse_idx": [], "timestep_idx": [],
    }
    pbar = tqdm(range(num_steps), desc="Attacking")

    for step in pbar:
        if I_adv.grad is not None:
            I_adv.grad.zero_()

        pipe.load_models_to_device(["vae", "image_encoder"])

        image_emb_adv = pipe.encode_image(I_adv, num_frames=num_frames, height=h, width=w)

        # Structural loss: untargeted clean-vs-adversarial temporal deviation
        # (Wan Structural Conditioning Attack). z_clean is the cached, gradient-free
        # reference; z_adv carries the graph back to I_adv through the VAE encoder.
        z_adv = image_emb_adv["y"][0, 4:]
        z_clean = image_emb_src["y"][0, 4:]
        struct = compute_structural_loss(z_adv, z_clean, lambda_cov=lambda_cov, temperature=temperature, rms_epsilon=rms_epsilon)
        struct_loss = struct["loss"]

        pipe.scheduler.set_timesteps(num_inference_steps=25, denoising_strength=1.0, shift=5.0)

        # Semantic Conditioning Attack: sample one good-training prompt and one
        # (collapse prompt, own trajectory) pair, then sample one high-noise
        # timestep from within that trajectory's own leading pool. Both semantic
        # branches below use this exact same adv_latents / timestep.
        good_idx = random.randrange(len(good_bank))
        collapse_idx = random.randrange(len(collapse_bank))

        good_emb = good_bank[good_idx]
        collapse_item = collapse_bank[collapse_idx]
        bad_emb = collapse_item["prompt_emb"]
        trajectory = collapse_item["latents_list"]

        pool_size = min(semantic_timestep_pool, len(trajectory))
        timestep_idx = random.randrange(pool_size)

        # Sample the clean latent
        adv_latents = trajectory[timestep_idx].to(dtype=pipe.torch_dtype, device=pipe.device)
        timestep = pipe.scheduler.timesteps[timestep_idx].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

        pipe.load_models_to_device(["dit"])

        noise_pred = prompt_img_sem_loss(pipe.dit, adv_latents, timestep=timestep, **good_emb, **image_emb_adv)
        noise_pred_tar = prompt_img_sem_loss(pipe.dit, adv_latents, timestep=timestep, **bad_emb, **image_emb_src)

        B, S, N = noise_pred.shape
        assert S % 1560 == 0
        A_split = 5 * noise_pred.view(B, S // 1560, 1560, N)
        B_split = 5 * noise_pred_tar.view(B, S // 1560, 1560, N)

        D = A_split[:, 1:, :, :] - B_split[:, 1:, :, :]
        attn_loss = torch.sqrt((D ** 2).sum(dim=(0, 2, 3))).sum()

        # scale the loss if needed
        w1 = 1
        w2 = 0.125
        L = w1 * struct_loss + w2 * attn_loss

        pbar.set_postfix(
            loss=f"{L.item():.4f}",
            struct=f"{struct_loss.item():.4f}",
            j_avg=f"{struct['j_avg'].item():.4f}",
            j_cov=f"{struct['j_cov'].item():.4f}",
            attn=f"{attn_loss.item():.4f}",
            g=good_idx, c=collapse_idx, ti=timestep_idx,
        )
        history["total_loss"].append(L.item())
        history["struct_loss"].append(struct_loss.item())
        history["j_avg"].append(struct["j_avg"].item())
        history["j_cov"].append(struct["j_cov"].item())
        history["attn_loss"].append(attn_loss.item())
        history["d"].append(struct["d"].tolist())
        history["good_idx"].append(good_idx)
        history["collapse_idx"].append(collapse_idx)
        history["timestep_idx"].append(timestep_idx)
        L.backward()

        # PGD, Clamp
        sgn = I_adv.grad.data.sign()

        # step size adjustment (optional)
        # step_size = step_size * 0.5 * (1 + math.cos(math.pi * step / num_steps))

        I_adv.data = I_adv.data - step_size * sgn
        delta = torch.clamp(I_adv - I_adv_before, min=-epsilon, max=epsilon)
        I_adv.data = torch.clamp(I_adv_before + delta, -1.0, 1.0)

    good_idx_counts = dict(sorted(Counter(history["good_idx"]).items()))
    collapse_idx_counts = dict(sorted(Counter(history["collapse_idx"]).items()))
    history["good_idx_counts"] = good_idx_counts
    history["collapse_idx_counts"] = collapse_idx_counts
    print(f"Good-prompt sampling counts: {good_idx_counts}")
    print(f"Collapse-prompt sampling counts: {collapse_idx_counts}")

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    image_path = cfg["data"]["image_path"]
    image_name = os.path.basename(image_path)
    name_wo_ext = os.path.splitext(image_name)[0]

    os.makedirs("attacked/images", exist_ok=True)
    os.makedirs("attacked/loss_curve", exist_ok=True)

    loss_save_path = os.path.join("attacked/loss_curve", f"{name_wo_ext}_loss.png")
    plot_loss_curve(history["total_loss"], save_path=loss_save_path)

    diagnostics_save_path = os.path.join("attacked/loss_curve", f"{name_wo_ext}_diagnostics.pt")
    torch.save(history, diagnostics_save_path)

    adv_save_path = os.path.join("attacked/images", f"{name_wo_ext}.jpg")
    metrics = save_adv_result(I_adv, I_adv_before, save_path=adv_save_path)

    print(f"Saved adversarial image to {adv_save_path}")
    print(f"Saved loss curve to {loss_save_path}")
    print(f"Saved structural-loss diagnostics (J_avg, J_cov, per-position d, attn_loss, total loss, sampling indices/counts) to {diagnostics_save_path}")

    return history


def main():
    pipe = load_all_models()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    h = cfg["video"]["height"]
    w = cfg["video"]["width"]
    num_frames = cfg["video"]["num_frames"]

    image = Image.open(cfg["data"]["image_path"]).resize((w, h))

    cache_files = {
        'good_bank': 'cache/good_bank.pt',
        'collapse_bank': 'cache/collapse_bank.pt',
        'image_emb_src': 'cache/image_emb_src.pt',
    }

    if all(os.path.exists(f) for f in cache_files.values()):
        print("Loading cached data...")
        good_bank = torch.load(cache_files['good_bank'])
        collapse_bank = torch.load(cache_files['collapse_bank'])
        image_emb_src = torch.load(cache_files['image_emb_src'])
        print("Loaded successfully!")
    else:
        raise FileNotFoundError(
            "Data not found in cache/. Please prepare and preprocess data first by running preprocess_data.py!"
        )

    assert len(good_bank) == 6, f"cache/good_bank.pt must contain exactly 6 entries, got {len(good_bank)}. Re-run preprocess_data.py."
    assert len(collapse_bank) == 4, f"cache/collapse_bank.pt must contain exactly 4 entries, got {len(collapse_bank)}. Re-run preprocess_data.py."
    for item in collapse_bank:
        assert "prompt_emb" in item and "latents_list" in item, "Each collapse_bank entry must contain its own paired prompt_emb and latents_list."

    num_steps = cfg["attack"]["num_steps"]
    epsilon = eval(cfg["attack"]["epsilon"])
    step_size = epsilon / 50

    lambda_cov = cfg["attack"].get("lambda_cov", DEFAULT_LAMBDA_COV)
    temperature = cfg["attack"].get("temperature", DEFAULT_TEMPERATURE)
    rms_epsilon = cfg["attack"].get("rms_epsilon", DEFAULT_RMS_EPSILON)
    semantic_timestep_pool = cfg["attack"].get("semantic_timestep_pool", DEFAULT_SEMANTIC_TIMESTEP_POOL)

    random.seed(0)
    run_attack(pipe, image, h, w, num_frames, good_bank, collapse_bank, image_emb_src,
                num_steps=num_steps, epsilon=epsilon, step_size=step_size,
                lambda_cov=lambda_cov, temperature=temperature, rms_epsilon=rms_epsilon,
                semantic_timestep_pool=semantic_timestep_pool)


if __name__ == "__main__":
    main()
