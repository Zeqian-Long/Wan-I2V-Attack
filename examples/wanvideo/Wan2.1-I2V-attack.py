import torch
import sys
diffsynth_path = "/workspace/DiffSynth-Studio"
sys.path.append(diffsynth_path)
from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline, model_fn_wan_video_attack, prompt_clip_attn_loss
from diffsynth.data.video import save_video, VideoData, LowMemoryImageFolder
from PIL import Image

from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import copy
import lpips
import numpy as np
from sklearn.manifold import TSNE
import torchvision.transforms.functional as TF

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

# pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
# pipe.enable_vram_management()
h = 480
w = 832

image = Image.open("data/boat.jpg")
# image = Image.open("I_adv_final.jpg")
image = image.resize((w, h))

num_frames = 1  
# 1, 5, 9, 13, ....

# Ensure training
pipe.dit.to(pipe.device)
pipe.vae.to(pipe.device)
pipe.image_encoder.to(pipe.device)
pipe.text_encoder.to(pipe.device)


# Encode Prompt
# prompt = "一艘小船正勇敢地乘风破浪前行。蔚蓝的大海波涛汹涌，白色的浪花拍打着船身，但小船毫不畏惧，坚定地驶向远方。阳光洒在水面上，闪烁着金色的光芒，为这壮丽的场景增添了一抹温暖。镜头拉近，可以看到船上的旗帜迎风飘扬，象征着不屈的精神与冒险的勇气。这段画面充满力量，激励人心，展现了面对挑战时的无畏与执着。"

prompt = "在一间无限延伸的绿色实验室中，空气被冰冷的光线分割成无数几何面。地面反射着荧光蓝与猩红的光带，像液态的玻璃在呼吸。墙壁上悬浮着巨大的数字与方程，它们在慢慢旋转，投下灰绿色的阴影。中央站着一个身披银灰外衣的身影，眼中闪烁着金光，周围漂浮着数以百计的透明立方体，每一个立方体里都封存着一段记忆的闪光。突然，顶端的灯光爆裂成黑白交错的涡流，空间开始塌陷，色彩被撕裂成纯粹的蓝与橙的对撞。声音消失，只剩下一种光的震颤，像宇宙在深呼吸。"

# prompt = "A black swan swimming gracefully in the river."
pipe.load_models_to_device(["text_encoder"])
with torch.no_grad():
    prompt_emb_posi = pipe.encode_prompt(prompt=prompt, positive=True)

print(prompt_emb_posi['context'].shape)
# [1, 512, 4096]

tiler_kwargs = {"tiled": False, "tile_size": (h / 16, w / 16), "tile_stride": (h / 32, w / 32)}


saved_features = {}
def make_hook(name):
    def hook(module, inp, out):
        # print(f"Hook for {name}, out.shape={out.shape}")
        saved_features[name] = out
    return hook

vae = pipe.vae.model
encoder = pipe.vae.model.encoder

encoder.conv1.register_forward_hook(make_hook("conv1"))
for i in range(len(list(encoder.downsamples))):
    list(encoder.downsamples)[i].register_forward_hook(make_hook(f"downsample_{i}"))
list(encoder.middle)[-1].register_forward_hook(make_hook("middle"))
vae.conv1.register_forward_hook(make_hook("mu_logvar"))
decoder = pipe.vae.model.decoder
for i in range(len(list(decoder.upsamples))):
    list(decoder.upsamples)[i].register_forward_hook(make_hook(f"upsample_{i}"))



# # Encode Image
# pipe.load_models_to_device(["image_encoder", "vae"])
# image_emb_src = pipe.encode_image(
#     image, num_frames=num_frames, height=h, width=w, **tiler_kwargs
# )
# print("Source Image Embedding Shape:", image_emb_src["clip_feature"].shape)
# # [1, 1 + 256, 1280]

# # [1, C (4+16), 1+T/4, 60, 104]

# src_decoded = pipe.decode_video(image_emb_src["y"][:, 4:, :], **tiler_kwargs)


# -------------------------------------------------------------------------------------------------------
# decoder check

# frames = pipe.decode_video(image_emb_src["y"][:, 4:, :], **tiler_kwargs)
# pipe.load_models_to_device([])
# frames = pipe.tensor2video(frames[0])
# save_video(frames, "check.mp4", fps=15)
# import pdb; pdb.set_trace()
# -------------------------------------------------------------------------------------------------------


# # Black image for target
# # target_image = Image.new("RGB", (832, 480), color=(0, 0, 0))
target_image = Image.open("data/MIST_Repeated.png").convert("RGB")
target_image = target_image.resize((w, h))
with torch.no_grad():
    image_emb_tgt = pipe.encode_image(
        target_image, num_frames=num_frames, height=h, width=w, **tiler_kwargs
    )
print("Target Image Embedding Shape:", image_emb_tgt["y"].shape)



# image_emb_tgt = pipe.encode_image(
#     target_image, num_frames=num_frames, height=h, width=w, **tiler_kwargs
# )
# print("Target Image Embedding Shape:", image_emb_tgt["y"].shape)
# # [1, 20, 21, 60, 104]

# # tgt_decoded = pipe.decode_video(image_emb_tgt["y"][:, 4:, :], **tiler_kwargs)

# # # source_features = copy.deepcopy(saved_features)
# # target_features = copy.deepcopy(saved_features)
# # saved_features = {}



# # def random_resized_crop(img_tensor, target_size=(224, 224), scale=(0.5, 1.0)):
# #     transform = T.Compose([
# #         T.RandomResizedCrop(target_size, scale=scale),
# #     ])
# #     return transform(img_tensor)


I_adv = pipe.preprocess_image(image).to(pipe.device).detach().requires_grad_(True)
print("Initial Image Shape:", I_adv.shape)
I_adv_before = I_adv.clone().detach()

num_steps = 400

ATTN_BLOCK_ID = 3

epsilon = 16.0 / 255 * 2
step_size = epsilon / 10 


# Random Initialization within the epsilon-ball
noise = torch.empty_like(I_adv).uniform_(-epsilon, epsilon) 
I_adv = I_adv + noise
I_adv = torch.clamp(I_adv, -1.0, 1.0).detach()
I_adv.requires_grad_(True)

# lpips_fn = lpips.LPIPS(net='vgg').cuda()

loss_history = []

for step in tqdm(range(num_steps), desc="Optimizing"):
    if I_adv.grad is not None:
        I_adv.grad.zero_()

    # i, j, h_crop, w_crop = T.RandomResizedCrop.get_params(I_adv, scale=(0.5, 1.0), ratio=(3. / 4., 4. / 3.))
    # I_adv_crop = TF.resized_crop(I_adv, i, j, h_crop, w_crop, size=(224, 224))
    # # I_tgt_crop = TF.resized_crop(target_image, i, j, h_crop, w_crop, size=(224, 224))
    # I_tgt_crop = TF.resize(image, size=(224, 224))

    pipe.load_models_to_device(["vae", "image_encoder"])
    image_emb_adv = pipe.encode_image(I_adv, num_frames=num_frames, height=h, width=w, **tiler_kwargs)

    # # Pixel reconstruction loss
    # adv_decoded = pipe.decode_video(image_emb_adv["y"][:, 4:, :], **tiler_kwargs)

    # Encoder loss
    L_enc_1 = torch.nn.functional.mse_loss(image_emb_adv["y"][:, 4:, :], image_emb_tgt["y"][:, 4:, :])

    # L_enc_clip = torch.nn.functional.cosine_similarity(image_emb_adv["clip_feature"].flatten(1), image_emb_tgt["clip_feature"].flatten(1), dim=1).mean()

    # L_enc_2 = torch.nn.functional.mse_loss(adv_decoded, tgt_decoded)

    # # L_enc_2 = lpips_fn(adv_decoded[:, :, 0], src_decoded[:, :, 0]).mean()

    
    # L_enc_3 = torch.nn.functional.l1_loss(saved_features['conv1'], target_features['conv1'])
    # L_enc = 0
    # for i in range(len(list(encoder.downsamples))):
    #     L_enc += torch.nn.functional.l1_loss(saved_features[f'downsample_{i}'], target_features[f'downsample_{i}'])
    # L_enc_7 = torch.nn.functional.l1_loss(saved_features['middle'], target_features['middle'])
    # L_dec = 0
    # for i in range(len(list(decoder.upsamples))):
    #     L_dec += torch.nn.functional.l1_loss(saved_features[f'upsample_{i}'], target_features[f'upsample_{i}'])


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

    noise = pipe.generate_noise((1, 16, (num_frames - 1) // 4 + 1, 480//8, 832//8), seed=0, device="cpu", dtype=torch.float32)
    noise = noise.to(dtype=pipe.torch_dtype, device=pipe.device)
    adv_latents = noise
    extra_input = pipe.prepare_extra_input(adv_latents)

    pipe.scheduler.set_timesteps(num_inference_steps=25, denoising_strength=1.0, shift=5.0)

    pipe.load_models_to_device(["dit"])
    for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps[:1])):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        attn_loss = prompt_clip_attn_loss(pipe.dit, adv_latents, timestep=timestep, **prompt_emb_posi, **image_emb_adv, **extra_input)

    L = attn_loss + L_enc_1
    print(f"Step {step+1}/{num_steps}, Loss: {L.item():.6f}")
    loss_history.append(L.item())
    L.backward()

    sgn = I_adv.grad.data.sign()
    I_adv.data = I_adv.data - step_size * sgn
    delta = torch.clamp(I_adv - I_adv_before, min=-epsilon, max=epsilon)
    I_adv.data = torch.clamp(I_adv_before + delta, -1.0, 1.0)




plt.plot(loss_history, label="Total Loss L")
plt.xlabel("Step")
plt.ylabel("Loss Value")
plt.title("Loss Curve during Optimization")
plt.legend()
plt.savefig("loss_curve.png")

I_adv_out = I_adv.detach().cpu().squeeze(0)
I_adv_before = I_adv_before.detach().cpu()
diff = (I_adv_out - I_adv_before).abs()
max_diff = diff.max()
mean_diff = diff.mean()
print(f"Max Diff: {max_diff.item():.6f}")
print(f"Mean Diff: {mean_diff.item():.6f}")

I_adv_out = (I_adv_out + 1.0) / 2.0         
I_adv_out = I_adv_out.to(torch.float32).clamp(0, 1)  
I_adv_pil = to_pil_image(I_adv_out)
I_adv_pil.save("I_adv_final.jpg")

# --------------------------------------------- Testing ---------------------------------------------

# image = Image.open("I_adv_final.jpg")
# image = image.resize((w, h))
# pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

# video = pipe(
#     prompt="一艘白色的游船正在湖面上正常地行驶",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     input_image=image, target_image=Image.open("data/MIST_Repeated.png").resize((w, h)),
#     num_inference_steps=25, height=h, width=w,
#     seed=0, tiled=True, num_frames=17
# )
# save_video(video, "video_attacked.mp4", fps=15, quality=5)

