import torch
import sys
diffsynth_path = "/workspace/DiffSynth-Studio"
sys.path.append(diffsynth_path)
from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline, model_fn_wan_video_attack
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

pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

image = Image.open("data/swan.jpg")
image = Image.open("I_adv_final.jpg")
image = image.resize((832, 480))

num_frames = 1  
# 1, 5, 9, 13, ....

# pipe.dit.to(pipe.device)
# pipe.vae.to(pipe.device)
# pipe.image_encoder.to(pipe.device)
# pipe.text_encoder.to(pipe.device)
# for m in [pipe.dit, pipe.vae, pipe.image_encoder, pipe.text_encoder]:
#     for p in m.parameters():
#         p.requires_grad_(False)


# Encode Prompt
# prompt = "一艘小船正勇敢地乘风破浪前行。蔚蓝的大海波涛汹涌，白色的浪花拍打着船身，但小船毫不畏惧，坚定地驶向远方。阳光洒在水面上，闪烁着金色的光芒，为这壮丽的场景增添了一抹温暖。镜头拉近，可以看到船上的旗帜迎风飘扬，象征着不屈的精神与冒险的勇气。这段画面充满力量，激励人心，展现了面对挑战时的无畏与执着。"
prompt = "In a documentary photography style, the camera follows a graceful black swan gliding slowly across a calm river. On the swan's head, its bright red beak contrasts vividly with the sleek dark feathers, while its neck curves elegantly as it moves forward. The water ripples gently in its wake, reflecting subtle shades of green from the overhanging leaves and plants along the riverbank. The scene is tranquil and natural, captured from a medium shot in a tracking perspective."
pipe.load_models_to_device(["text_encoder"])
with torch.no_grad():
    prompt_emb_posi = pipe.encode_prompt(prompt=prompt, positive=True)

tiler_kwargs = {"tiled": False, "tile_size": (30, 52), "tile_stride": (15, 26)}



saved_features = {}

def make_hook(name):
    def hook(module, inp, out):
        # print(f"Hook for {name}, out.shape={out.shape}")
        saved_features[name] = out
    return hook

encoder = pipe.vae.model.encoder

encoder.conv1.register_forward_hook(make_hook("conv1"))
list(encoder.downsamples)[2].register_forward_hook(make_hook("downsample_2"))
list(encoder.downsamples)[5].register_forward_hook(make_hook("downsample_5"))
list(encoder.downsamples)[8].register_forward_hook(make_hook("downsample_8"))
list(encoder.middle)[-1].register_forward_hook(make_hook("middle"))

# Encode Image
pipe.load_models_to_device(["image_encoder", "vae"])
image_emb_src = pipe.encode_image(
    image, num_frames=num_frames, height=480, width=832, **tiler_kwargs
)
print("Source Image Embedding Shape:", image_emb_src["y"].shape)
# [1, C (4+16), 1+T/4, 60, 104]


# -------------------------------------------------------------------------------------------------------
# decoder check
frames = pipe.decode_video(image_emb_src["y"][:, 4:, :], **tiler_kwargs)
pipe.load_models_to_device([])
frames = pipe.tensor2video(frames[0])
save_video(frames, "check.mp4", fps=15)
import pdb; pdb.set_trace()
# -------------------------------------------------------------------------------------------------------

# latent = image_emb_src["y"] 
# latent_2d = latent[0, 4:, 3]  
# C, H, W = latent_2d.shape
# latent_2d = latent_2d.permute(1, 2, 0).reshape(-1, C).cpu().to(torch.float32).numpy()
# pca = PCA(n_components=3)
# latent_rgb = pca.fit_transform(latent_2d)  # (H*W, 3)
# latent_rgb = (latent_rgb - latent_rgb.min()) / (latent_rgb.max() - latent_rgb.min())
# latent_rgb = latent_rgb.reshape(H, W, 3)
# plt.imsave("vae_latent.png", latent_rgb)
# import pdb; pdb.set_trace()


# Black image for target
# target_image = Image.new("RGB", (832, 480), color=(0, 0, 0))
# target_image = Image.open("data/MIST_Repeated.png").convert("RGB")
# target_image = target_image.resize((832, 480))


# to_tensor = T.ToTensor()
# to_pil = T.ToPILImage()
# def shuffle_patches(image, patch_size=32):
#     img_tensor = to_tensor(image)
#     C, H, W = img_tensor.shape
#     patches = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
#     patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, C, patch_size, patch_size)
#     idx = torch.randperm(patches.size(0))
#     patches = patches[idx]
#     num_h = H // patch_size
#     num_w = W // patch_size
#     patches = patches.reshape(num_h, num_w, C, patch_size, patch_size)
#     patches = patches.permute(2, 0, 3, 1, 4).reshape(C, H, W)
#     return to_pil(patches)


# frames = []
# for _ in range(num_frames):
#     shuffled = shuffle_patches(image, patch_size=32)
#     frames.append(shuffled)
# save_video(frames, "target_image_input.mp4", fps=15)

# # import pdb; pdb.set_trace()


# target_image = Image.open("target_patch_shuffle.png").convert("RGB")
# target_image = target_image.resize((832, 480))
# target_image_input = [target_image] * num_frames
# save_video(target_image_input, "target_image_input.mp4", fps=15)
# target_image_input = VideoData("/workspace/DiffSynth-Studio/target_image_input.mp4", height=480, width=832)
# target_image_input = pipe.preprocess_images(target_image_input)
# target_image_input = torch.stack(target_image_input, dim=2).to(dtype=pipe.torch_dtype, device=pipe.device)
# image_emb_tgt = pipe.encode_video(target_image_input, **tiler_kwargs).to(dtype=pipe.torch_dtype, device=pipe.device)
# print("Target Image Input Shape:", image_emb_tgt.shape)



# image_emb_tgt = pipe.encode_image(
#     target_image, num_frames=num_frames, height=480, width=832, **tiler_kwargs
# )
# print("Target Image Embedding Shape:", image_emb_tgt["y"].shape)
# [1, 20, 21, 60, 104]


# source_features = copy.deepcopy(saved_features)
target_features = copy.deepcopy(saved_features)

# target_features = {}
# for name, feat in saved_features.items():
#     target_features[name] = torch.randn_like(feat)

saved_features = {}

# latent = image_emb_tgt 
# latent_2d = latent[0, : , 5]  
# C, H, W = latent_2d.shape
# latent_2d = latent_2d.permute(1, 2, 0).reshape(-1, C).cpu().to(torch.float32).numpy()
# pca = PCA(n_components=3)
# latent_rgb = pca.fit_transform(latent_2d)  # (H*W, 3)
# latent_rgb = (latent_rgb - latent_rgb.min()) / (latent_rgb.max() - latent_rgb.min())
# latent_rgb = latent_rgb.reshape(H, W, 3)
# plt.imsave("tar_latent.png", latent_rgb)
# import pdb; pdb.set_trace()

# latent = image_emb_tgt 
# vae_latent = latent[0, 5, 5]   # [16, 60, 104]
# vae_mean = vae_latent
# plt.imsave("tar_latent_mean.png", vae_mean.detach().cpu().to(torch.float32).numpy(), cmap="viridis")
# import pdb; pdb.set_trace()


I_adv = pipe.preprocess_image(image).to(pipe.device).detach().requires_grad_(True)
# print("Initial Image Shape:", I_adv.shape)

I_adv_before = I_adv.clone().detach()



num_steps = 400
ATTN_BLOCK_ID = 3

alpha = 0.4  
beta = 0.4   
gamma = 1.0 
tao_1 = 1.0     
tao_2 = 1.0 

epsilon = 16.0 / 255 * 2
step_size = epsilon / 10  

noise = torch.empty_like(I_adv).uniform_(-epsilon, epsilon) 
I_adv = I_adv + noise
I_adv = torch.clamp(I_adv, -1.0, 1.0).detach()
I_adv.requires_grad_(True)


# noise = torch.empty_like(I_adv).uniform_(-epsilon, epsilon)
# I_adv = torch.clamp(I_adv + noise, -1.0, 1.0)
# I_adv.requires_grad_(True)

lpips_fn = lpips.LPIPS(net='vgg').cuda()

for step in tqdm(range(num_steps), desc="Optimizing"):
    # use for pgd attack
    I_orig = I_adv.clone().detach()
    if I_adv.grad is not None:
        I_adv.grad.zero_()

    pipe.load_models_to_device(["vae", "image_encoder"])
    # with torch.no_grad():
    image_emb_adv = pipe.encode_image(I_adv, num_frames=num_frames, height=480, width=832, **tiler_kwargs)


    

    # Pixel reconstruction loss
    adv_decoded = pipe.decode_video(image_emb_adv["y"][:, 4:, :], **tiler_kwargs)
    src_decoded = pipe.decode_video(image_emb_src["y"][:, 4:, :], **tiler_kwargs)

    # tgt_decoded = pipe.decode_video(image_emb_tgt["y"][:, 4:, :], **tiler_kwargs)

    # print(adv_decoded.shape)
    # import pdb; pdb.set_trace()


    L_enc_1 = torch.nn.functional.mse_loss(image_emb_adv["y"][:, 4:, :], image_emb_src["y"][:, 4:, :])
    # L_enc_2 = torch.nn.functional.mse_loss(adv_decoded, tgt_decoded)

    L_enc_2 = lpips_fn(adv_decoded[:, :, 0], src_decoded[:, :, 0]).mean()


    
    L_enc_3 = torch.nn.functional.mse_loss(saved_features['conv1'], target_features['conv1'])
    L_enc_4 = torch.nn.functional.mse_loss(saved_features['downsample_2'], target_features['downsample_2'])
    L_enc_5 = torch.nn.functional.mse_loss(saved_features['downsample_5'], target_features['downsample_5'])
    L_enc_6 = torch.nn.functional.mse_loss(saved_features['downsample_8'], target_features['downsample_8'])
    L_enc_7 = torch.nn.functional.mse_loss(saved_features['middle'], target_features['middle'])

    # L_enc = torch.nn.functional.l1_loss(image_emb_adv["y"][:, 4:, :], image_emb_tgt[:, :, :])


    vae_latent = image_emb_adv["y"][0, 4:, 0]
    os.makedirs("logs", exist_ok=True) 
    if step % 10 == 0:
        C, H, W = vae_latent.shape
        latent_2d = vae_latent.permute(1, 2, 0).reshape(-1, C).detach().cpu().to(torch.float32).numpy()  # (H*W, C)
        pca = PCA(n_components=3)
        latent_pca = pca.fit_transform(latent_2d)  # (H*W, 3)
        latent_pca = (latent_pca - latent_pca.min()) / (latent_pca.max() - latent_pca.min())
        latent_rgb = latent_pca.reshape(H, W, 3)
        plt.imsave(f"logs/emb_adv_step{step:04d}_pca.png", latent_rgb)

    # noise = pipe.generate_noise((1, 16, (num_frames - 1) // 4 + 1, 480//8, 832//8), seed=0, device="cpu", dtype=torch.float32)
    # noise = noise.to(dtype=pipe.torch_dtype, device=pipe.device)
    # adv_latents = noise
    # extra_input = pipe.prepare_extra_input(adv_latents)

    # pipe.scheduler.set_timesteps(num_inference_steps=50, denoising_strength=1.0, shift=5.0)

    # pipe.load_models_to_device(["dit"])
    # for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps[:1])):
    #     timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
    #     noise_pred, attn_loss = model_fn_wan_video_attack(pipe.dit, adv_latents, timestep=timestep, **prompt_emb_posi, **image_emb_adv, **extra_input)

    # L = L_enc + attn_loss
    L = L_enc_1
    print(L)
    L.backward()
    sgn = I_adv.grad.data.sign()
    I_adv.data = I_adv.data + step_size * sgn
    delta = torch.clamp(I_adv - I_adv_before, min=-epsilon, max=epsilon)
    I_adv.data = torch.clamp(I_adv_before + delta, -1.0, 1.0)

    # L_enc = alpha * torch.nn.functional.mse_loss(image_emb_adv["y"][:, 4:, :], image_emb_tgt)

    # Generate noisy frames X_t
    
    # timestep = pipe.scheduler.timesteps[0]
    # timestep = timestep.to(dtype=torch.float32, device=pipe.device)
    # noise = pipe.generate_noise(
    #     shape=X_0.shape,
    #     seed=0,
    #     device=pipe.device,
    #     dtype=torch.float32
    # ).to(dtype=pipe.torch_dtype).to(device=pipe.device)
    # X_t = pipe.scheduler.add_noise(X_0, noise, timestep)



    # # pipe.load_models_to_device(["vae", "image_encoder"])
    # # image_emb_adv = pipe.encode_image(I_orig, num_frames=num_frames, height=480, width=832)


    # Diffusion / Flow Attack
    

    # # # with torch.no_grad():
    # timestep_tensor = pipe.scheduler.timesteps[0].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
    

    # with torch.no_grad():
    #     X_t_src = X_t
    #     X_t_adv = X_t
    #     for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
    #         timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
    #         X0_adv_hat = pipe.dit(X_t_adv, timestep=timestep, **prompt_emb_posi, **image_emb_tgt).to(dtype=pipe.torch_dtype, device=pipe.device)

    # # # import pdb; pdb.set_trace()

    # # # # Attention hook
    # # # attn_adv = pipe.dit.blocks[2].self_attn.last_attn.clone()

    #         X0_src_hat = pipe.dit(X_t_src, timestep=timestep, **prompt_emb_posi, **image_emb_src).to(dtype=pipe.torch_dtype, device=pipe.device)
    #         print ("X0_adv_hat:", X0_adv_hat.shape)
    #         print ("X0_src_hat:", X0_src_hat.shape)

    #         diff = (X0_adv_hat[:,:,0] - X0_src_hat[:,:,0]).norm(dim=1).squeeze().detach().cpu().to(torch.float32)
    #         diff = (diff - diff.min()) / (diff.max() - diff.min())
    #         to_pil_image(diff.unsqueeze(0)).save(f"noise_pred_diff_{progress_id}.png")
    #         print("Saved noise_pred_diff.png")

    #         X_t_adv = pipe.scheduler.step(X0_adv_hat, pipe.scheduler.timesteps[progress_id], X_t_adv)
    #         X_t_src = pipe.scheduler.step(X0_src_hat, pipe.scheduler.timesteps[progress_id], X_t_src)

    #     import pdb; pdb.set_trace()
    



    # # # # Attention hook
    # # # attn_src = pipe.dit.blocks[2].self_attn.last_attn.clone()

    # # L_con = beta * (
    # #     torch.nn.functional.mse_loss(X0_adv_hat, X_tg) +
    # #     torch.clamp(tao_1 - torch.nn.functional.mse_loss(X0_adv_hat, X_0), min=0)
    # # )
    # # L_att = gamma * (tao_2 - ((attn_adv - attn_src) ** 2).mean())

    # # L = L_att + L_con + L_enc + torch.nn.functional.mse_loss(I_adv, pipe.preprocess_image(image).to(pipe.device))

    # L = L_enc + torch.nn.functional.mse_loss(I_adv, pipe.preprocess_image(image).to(pipe.device))
    # print(L)

    # # import pdb; pdb.set_trace()

    # L.backward()

    # sgn = I_adv.grad.data.sign()
    # I_adv.data = I_adv.data + step_size* sgn
    
    # # optimizer.step()

    # delta = torch.clamp(I_adv - I_adv_before, min=-epsilon, max=epsilon)
    # I_adv.data = torch.clamp(I_adv_before + delta, -1.0, 1.0)


# epsilon = 64.0 / 255 * 2
# delta = torch.clamp(I_adv - I_adv_before, min=-epsilon, max=epsilon)
# I_adv.data = torch.clamp(I_adv_before + delta, -1.0, 1.0)

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
# image = image.resize((832, 480))

# video = pipe(
#     prompt="In a documentary photography style, the camera follows a graceful black swan gliding slowly across a calm river. On the swan’s head, its bright red beak contrasts vividly with the sleek dark feathers, while its neck curves elegantly as it moves forward. The water ripples gently in its wake, reflecting subtle shades of green from the overhanging leaves and plants along the riverbank. The scene is tranquil and natural, captured from a medium shot in a tracking perspective.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     input_image=image,
#     num_inference_steps=50,
#     seed=0, tiled=True, num_frames=10
# )
# save_video(video, "video_attacked.mp4", fps=15, quality=5)



# Encode Frames
# pipe.load_models_to_device(['vae'])
# video = pipe.preprocess_images(video)
# video = torch.stack(video, dim=2).to(dtype=pipe.torch_dtype, device=pipe.device)
# X_0 = pipe.encode_video(video, **tiler_kwargs).to(dtype=pipe.torch_dtype, device=pipe.device)
# print("Source Video Embedding Shape:", X_0.shape)
# # [1, 16, 21, 60, 104]

# # Black Frames
# frames = [target_image] * num_frames
# save_video(frames, "video_black.mp4", fps=15)

# target_video = VideoData("/workspace/DiffSynth-Studio-clean/video_black.mp4", height=480, width=832)
# target_video = pipe.preprocess_images(target_video)
# target_video = torch.stack(target_video, dim=2).to(dtype=pipe.torch_dtype, device=pipe.device)
# X_tg = pipe.encode_video(target_video, **tiler_kwargs).to(dtype=pipe.torch_dtype, device=pipe.device)
# print("Target Video Embedding Shape:", X_tg.shape)





# Encoder attack
# pipe.load_models_to_device(["vae"])
# img_tensor = I_adv[0]  # [3, H, W]
# vae_input = torch.cat([
#     img_tensor.unsqueeze(1),  # [3, 1, H, W]
#     torch.zeros(3, num_frames - 1, 480, 832, device=img_tensor.device)
# ], dim=1)  # [3, num_frames, 480, 832]
# vae_input = vae_input.to(pipe.device).to(pipe.torch_dtype)
# vae_output = pipe.vae.encode([vae_input], device=pipe.device, **tiler_kwargs)[0]
# vae_output = vae_output.to(pipe.device).to(pipe.torch_dtype)
# B, T, H, W = 1, num_frames, 480 // 8, 832 // 8
# msk = torch.zeros(B, T, H, W, device=pipe.device)
# msk[:, 0] = 1.0  
# msk = torch.cat([msk[:, :1].repeat(1, 4, 1, 1), msk[:, 1:]], dim=1)  # [1, 84, H, W]
# msk = msk.view(B, (num_frames + 3) // 4, 4, H, W)  # [1, 21, 4, H, W]
# msk = msk.transpose(1, 2)[0]  # [4, 21, H, W]
# msk = msk.to(device=pipe.device, dtype=pipe.torch_dtype)
# y_adv = torch.cat([msk, vae_output], dim=0).unsqueeze(0)  # [1, C+4, 21, H, W]