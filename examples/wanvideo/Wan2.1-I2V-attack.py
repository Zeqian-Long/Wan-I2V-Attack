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

# Ensure training
pipe = setup_pipe_modules(pipe)
# pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) 

# --------------------------------------------- Preprocessing ---------------------------------------------

h = 480
w = 832

image = Image.open("data/image/hike.jpg").resize((w, h))
# image = Image.open("I_adv_final.jpg")
mask = Image.open("data/mask/swan_224.jpg").convert("L")


num_frames = 5  
# 1, 5, 9, 13, ....

# def find_farthest_words_auto(img_emb, clip_model, tokenizer, k=20, device="cuda"):
#     img_emb = F.normalize(img_emb, dim=-1)
#     vocab = list(tokenizer.get_vocab().keys())
#     all_text_embs = []
#     batch_size = 256 
#     for i in tqdm(range(0, len(vocab), batch_size)):
#         batch = vocab[i:i+batch_size]
#         text_tokens = tokenizer(batch, padding=True, return_tensors="pt").to(device)
#         with torch.no_grad():
#             text_emb = clip_model.get_text_features(**text_tokens)
#             text_emb = F.normalize(text_emb, dim=-1)
#         all_text_embs.append(text_emb)
#     all_text_embs = torch.cat(all_text_embs, dim=0)
#     sim = img_emb @ all_text_embs.T  # [1, vocab_size]
#     farthest_idx = torch.topk(-sim, k, dim=-1).indices.squeeze(0)
#     farthest_words = [vocab[i] for i in farthest_idx.tolist()]
#     return farthest_words


# from transformers import CLIPProcessor, CLIPModel
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# img_emb = model.get_image_features(**processor(images=image, return_tensors="pt").to("cuda"))
# farthest = find_farthest_words_auto(img_emb, model, processor.tokenizer, k=512)

# def tokens_to_sentence(tokens):
#     clean = [t.replace("</w>", "") for t in tokens]
#     merged = []
#     for t in clean:
#         if not merged or merged[-1] != t:
#             merged.append(t)
#     sentence = " ".join(merged)
#     return sentence

# sentence = tokens_to_sentence(farthest)
# print(sentence)



# Encode Prompt
# prompt = "一艘小船正勇敢地乘风破浪前行。蔚蓝的大海波涛汹涌，白色的浪花拍打着船身，但小船毫不畏惧，坚定地驶向远方。阳光洒在水面上，闪烁着金色的光芒，为这壮丽的场景增添了一抹温暖。镜头拉近，可以看到船上的旗帜迎风飘扬，象征着不屈的精神与冒险的勇气。这段画面充满力量，激励人心，展现了面对挑战时的无畏与执着。"

prompt = "在一间无限延伸的绿色实验室中，空气被冰冷的光线分割成无数几何面。地面反射着荧光蓝与猩红的光带，像液态的玻璃在呼吸。墙壁上悬浮着巨大的数字与方程，它们在慢慢旋转，投下灰绿色的阴影。中央站着一个身披银灰外衣的身影，眼中闪烁着金光，周围漂浮着数以百计的透明立方体，每一个立方体里都封存着一段记忆的闪光。突然，顶端的灯光爆裂成黑白交错的涡流，空间开始塌陷，色彩被撕裂成纯粹的蓝与橙的对撞。声音消失，只剩下一种光的震颤，像宇宙在深呼吸。"

# prompt = "A black swan is swimming gracefully in the river."

# prompt = "A miniature blue train pulling several colorful wagons moves along curved railway tracks in a detailed model village. Small trees, grass, and tiny human figures surround the tracks, with a station building and people nearby, creating a realistic diorama scene."

# prompt = sentence

pipe.load_models_to_device(["text_encoder"])
with torch.no_grad():
    prompt_emb_posi = pipe.encode_prompt(prompt=prompt, positive=True)
print(prompt_emb_posi['context'].shape)
# [1, 512, 4096]

tiler_kwargs = {"tiled": False, "tile_size": (h / 16, w / 16), "tile_stride": (h / 32, w / 32)}




# # Encode Image
pipe.load_models_to_device(["image_encoder", "vae"])
with torch.no_grad():
    image_emb_src = pipe.encode_image(
        image, num_frames=num_frames, height=h, width=w, **tiler_kwargs
    )
print("Source Image Embedding Shape:", image_emb_src["clip_feature"].shape)
# [1, 1 + 256, 1280]
# [1, C (4+16), 1+T/4, 60, 104]
# src_decoded = pipe.decode_video(image_emb_src["y"][:, 4:, :], **tiler_kwargs)




latents_list = []

noise = pipe.generate_noise((1, 16, (num_frames - 1) // 4 + 1, h//8, w//8), seed=0, device="cpu", dtype=torch.float32)
noise = noise.to(dtype=pipe.torch_dtype, device=pipe.device)
latents = noise
extra_input = pipe.prepare_extra_input(latents)
pipe.scheduler.set_timesteps(num_inference_steps=25, denoising_strength=1.0, shift=5.0)
pipe.load_models_to_device(["dit"])

with torch.no_grad():
    for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
        latents_list.append(latents.detach().cpu())
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = model_fn_wan_video(pipe.dit, latents, timestep=timestep, **prompt_emb_posi, **image_emb_src, **extra_input)
        latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], latents)


pipe = setup_pipe_modules(pipe)
saved_features = register_vae_hooks(pipe)

# # Target Image
# # target_image = Image.new("RGB", (832, 480), color=(0, 0, 0))
target_image = Image.open("data/image/MIST_Repeated.png").convert("RGB")
target_image = target_image.resize((w, h))
with torch.no_grad():
    image_emb_tgt = pipe.encode_image(
        target_image, num_frames=num_frames, height=h, width=w, **tiler_kwargs
    )
print("Target Image Embedding Shape:", image_emb_tgt["y"].shape)

# # tgt_decoded = pipe.decode_video(image_emb_tgt["y"][:, 4:, :], **tiler_kwargs)

# # source_features = copy.deepcopy(saved_features)
target_features = copy.deepcopy(saved_features)
saved_features = {}


# --------------------------------------------- Attack ---------------------------------------------

I_adv = pipe.preprocess_image(image).to(pipe.device).detach().requires_grad_(True)
# print("Initial Image Shape:", I_adv.shape)
I_adv_before = I_adv.clone().detach()

num_steps = 400

ATTN_BLOCK_ID = 3

epsilon = 20.0 / 255 * 2
step_size = epsilon / 10 

I_adv = init_adv_image(I_adv, epsilon=epsilon, value_range=(-1.0, 1.0))


# lpips_fn = lpips.LPIPS(net='vgg').cuda()

loss_history = []

for step in tqdm(range(num_steps), desc="Optimizing"):
    if I_adv.grad is not None:
        I_adv.grad.zero_()

    pipe.load_models_to_device(["vae", "image_encoder"])
    image_emb_adv = pipe.encode_image(I_adv, num_frames=num_frames, height=h, width=w, **tiler_kwargs)

    # # Pixel reconstruction loss
    # adv_decoded = pipe.decode_video(image_emb_adv["y"][:, 4:, :], **tiler_kwargs)

    # Encoder loss
    L_enc_1 = torch.nn.functional.mse_loss(image_emb_adv["y"][:, 4:, :], image_emb_tgt["y"][:, 4:, :])

    # L_enc_2 = torch.nn.functional.mse_loss(adv_decoded, tgt_decoded)
    
    # L_enc = 0
    # for i in range(len(list(encoder.downsamples))):
    #     L_enc += 0.001 * torch.nn.functional.l1_loss(saved_features[f'downsample_{i}'], target_features[f'downsample_{i}'])

    # L_dec = 0
    # for i in range(len(list(decoder.upsamples))):
    #     L_dec += 0.001 * torch.nn.functional.l1_loss(saved_features[f'upsample_{i}'], target_features[f'upsample_{i}'])



    pipe.scheduler.set_timesteps(num_inference_steps=25, denoising_strength=1.0, shift=5.0)
    assert len(latents_list) == len(pipe.scheduler.timesteps), "Latent list and timesteps length mismatch!"
    idx = random.randrange(len(pipe.scheduler.timesteps))

    adv_latents = latents_list[idx].to(dtype=pipe.torch_dtype, device=pipe.device)
    timestep = pipe.scheduler.timesteps[idx].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
    extra_input = pipe.prepare_extra_input(adv_latents)

    pipe.load_models_to_device(["dit"])
    attn_loss = prompt_clip_attn_loss(pipe.dit, adv_latents, timestep=timestep, **prompt_emb_posi, **image_emb_adv, **extra_input)

    L = attn_loss + L_enc_1
    print(f"Step {step+1}/{num_steps}, Loss: {L.item():.6f}")
    loss_history.append(L.item())
    L.backward()


    # # decay
    # if step == 100:
    #     step_size *= 0.5
    # if step == 200:
    #     step_size *= 0.5

    sgn = I_adv.grad.data.sign()
    I_adv.data = I_adv.data - step_size * sgn
    delta = torch.clamp(I_adv - I_adv_before, min=-epsilon, max=epsilon)
    I_adv.data = torch.clamp(I_adv_before + delta, -1.0, 1.0)


plot_loss_curve(loss_history)
metrics = save_adv_result(I_adv, I_adv_before, save_path="I_adv_final_hike.jpg")




    # vae_latent = image_emb_adv["y"][0, 4:, 0]
    # os.makedirs("logs", exist_ok=True) 
    # if step % 10 == 0:
    #     C, H, W = vae_latent.shape
    #     latent_2d = vae_latent.permute(1, 2, 0).reshape(-1, C).detach().cpu().to(torch.float32).numpy()  # (H*W, C)
    #     pca = PCA(n_components=3)
    #     latent_pca = pca.fit_transform(latent_2d)  # (H*W, 3)
    #     latent_pca = (latent_pca - latent_pca.min()) / (latent_pca.max() - latent_pca.min())
    #     latent_rgb = latent_pca.reshape(H, W, 3)
    #     plt.imsave(f"logs/emb_adv_step{step:04d}_pca.png", latent_rgb)