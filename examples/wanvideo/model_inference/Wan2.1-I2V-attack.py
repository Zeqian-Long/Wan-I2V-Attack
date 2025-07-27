import torch
import sys
diffsynth_path = "/workspace/DiffSynth-Studio-main"
sys.path.append(diffsynth_path)

from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline, model_fn_wan_video
from diffsynth.data.video import save_video, VideoData, LowMemoryImageFolder

from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image

from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float16, # Image Encoder is loaded with float16
)
model_manager.load_models(
    [
        [
            "/workspace/DiffSynth-Studio-main/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "/workspace/DiffSynth-Studio-main/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "/workspace/DiffSynth-Studio-main/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "/workspace/DiffSynth-Studio-main/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "/workspace/DiffSynth-Studio-main/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "/workspace/DiffSynth-Studio-main/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "/workspace/DiffSynth-Studio-main/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "/workspace/DiffSynth-Studio-main/models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
        "/workspace/DiffSynth-Studio-main/models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management()

pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
# Download example image
# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=f"data/examples/wan/input_image.jpg"
# )

image = Image.open("data/examples/wan/input_image.jpg")
image = image.resize((832, 480))

video = VideoData("/workspace/DiffSynth-Studio-main/video.mp4", height=480, width=832)

# Encode Prompt
prompt = "一艘小船正勇敢地乘风破浪前行。蔚蓝的大海波涛汹涌，白色的浪花拍打着船身，但小船毫不畏惧，坚定地驶向远方。阳光洒在水面上，闪烁着金色的光芒，为这壮丽的场景增添了一抹温暖。镜头拉近，可以看到船上的旗帜迎风飘扬，象征着不屈的精神与冒险的勇气。这段画面充满力量，激励人心，展现了面对挑战时的无畏与执着。"
pipe.load_models_to_device(["text_encoder"])
with torch.no_grad():
    prompt_emb_posi = pipe.encode_prompt(prompt=prompt, positive=True)



# Image-to-video
# video = pipe(
#     prompt="一艘小船正勇敢地乘风破浪前行。蔚蓝的大海波涛汹涌，白色的浪花拍打着船身，但小船毫不畏惧，坚定地驶向远方。阳光洒在水面上，闪烁着金色的光芒，为这壮丽的场景增添了一抹温暖。镜头拉近，可以看到船上的旗帜迎风飘扬，象征着不屈的精神与冒险的勇气。这段画面充满力量，激励人心，展现了面对挑战时的无畏与执着。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     input_image=image,
#     num_inference_steps=50,
#     seed=0, tiled=True
# )
# save_video(video, "video.mp4", fps=15, quality=5)



# Image-to-video
# video = pipe(
#     prompt="一只金毛犬正在广阔的草地上欢快地奔跑。它的毛发在阳光下闪耀着金色的光芒，随风飘动，洋溢着生命的活力。四周的野花在微风中轻轻摇曳，映衬着它矫健的身姿。镜头慢慢推进，它张着嘴巴，神情专注，充满朝气与能量。远处的天空湛蓝清澈，为这奔跑的画面增添了一份自由与希望。每一次踏地、每一次跳跃，都仿佛在述说着对生活的热爱与对自由的追寻。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     input_image=image,
#     num_inference_steps=50,
#     seed=0, tiled=True, height=256, width=256
# )
# save_video(video, "dog.mp4", fps=15, quality=5)

# import pdb; pdb.set_trace()

# Encode Image
tiler_kwargs = {"tiled": True, "tile_size": (30, 52), "tile_stride": (15, 26)}



pipe.load_models_to_device(["image_encoder", "vae"])

image_emb_src = pipe.encode_image(
    image, end_image=None, num_frames=81, height=480, width=832, **tiler_kwargs
)

print("Source Image Embedding Shape:", image_emb_src["y"].shape)
# [1, 20, 21, 60, 104]


# Black image for target
target_image = Image.new("RGB", (832, 480), color=(0, 0, 0))
image_emb_tgt = pipe.encode_image(
    target_image, end_image=None, num_frames=81, height=480, width=832, **tiler_kwargs
)
print("Target Image Embedding Shape:", image_emb_tgt["y"].shape)
# [1, 20, 21, 60, 104]


# NOTE: Huge Computation Cost!
I_adv = pipe.preprocess_image(image).to(pipe.device).detach().requires_grad_(True)
# print("Initial Image Shape:", I_adv.shape)

I_adv_before = I_adv.clone().detach()

# Encode Frames
pipe.load_models_to_device(['vae'])
video = pipe.preprocess_images(video)
video = torch.stack(video, dim=2).to(dtype=pipe.torch_dtype, device=pipe.device)
X_0 = pipe.encode_video(video, **tiler_kwargs).to(dtype=pipe.torch_dtype, device=pipe.device)
print("Source Video Embedding Shape:", X_0.shape)
# [1, 16, 21, 60, 104]

# Black Frames
frames = [target_image] * 81
save_video(frames, "video_black.mp4", fps=15)

target_video = VideoData("/workspace/DiffSynth-Studio-main/video_black.mp4", height=480, width=832)
target_video = pipe.preprocess_images(target_video)
target_video = torch.stack(target_video, dim=2).to(dtype=pipe.torch_dtype, device=pipe.device)
X_tg = pipe.encode_video(target_video, **tiler_kwargs).to(dtype=pipe.torch_dtype, device=pipe.device)
print("Target Video Embedding Shape:", X_tg.shape)


num_steps = 5
lr = 0.01      
optimizer = torch.optim.Adam([I_adv], lr=lr)
ATTN_BLOCK_ID = 3



alpha = 1.0  
beta = 1.0   
gamma = 1.0 
tao_1 = 1.0     
tao_2 = 1.0 


for step in tqdm(range(num_steps), desc="Optimizing"):
    optimizer.zero_grad()

    # use for pgd attack
    I_orig = I_adv.clone().detach()

    # Encoder attack
    pipe.load_models_to_device(["vae"])
    img_tensor = I_adv[0]  # [3, H, W]
    vae_input = torch.cat([
        img_tensor.unsqueeze(1),  # [3, 1, H, W]
        torch.zeros(3, 81 - 1, 480, 832, device=img_tensor.device)
    ], dim=1)  # [3, 81, 480, 832]
    vae_input = vae_input.to(pipe.device).to(pipe.torch_dtype)
    vae_output = pipe.vae.encode([vae_input], device=pipe.device, **tiler_kwargs)[0]
    vae_output = vae_output.to(pipe.device).to(pipe.torch_dtype)
    B, T, H, W = 1, 81, 480 // 8, 832 // 8
    msk = torch.zeros(B, T, H, W, device=pipe.device)
    msk[:, 0] = 1.0  
    msk = torch.cat([msk[:, :1].repeat(1, 4, 1, 1), msk[:, 1:]], dim=1)  # [1, 84, H, W]
    msk = msk.view(B, 84 // 4, 4, H, W)  # [1, 21, 4, H, W]
    msk = msk.transpose(1, 2)[0]  # [4, 21, H, W]
    msk = msk.to(device=pipe.device, dtype=pipe.torch_dtype)
    y_adv = torch.cat([msk, vae_output], dim=0).unsqueeze(0)  # [1, C+4, 21, H, W]
    L_enc = alpha * torch.nn.functional.mse_loss(y_adv, image_emb_tgt["y"])

    # Generate noisy frames X_t
    pipe.scheduler.set_timesteps(num_inference_steps=25, denoising_strength=1.0)
    timestep = pipe.scheduler.timesteps[0]
    timestep = timestep.to(dtype=torch.float32, device=pipe.device)
    noise = pipe.generate_noise(
        shape=X_0.shape,
        seed=0,
        device=pipe.device,
        dtype=torch.float32
    ).to(dtype=pipe.torch_dtype).to(device=pipe.device)
    X_t = pipe.scheduler.add_noise(X_0, noise, timestep)



    pipe.load_models_to_device(["vae", "image_encoder"])
    image_emb_adv = pipe.encode_image(I_orig, end_image=None, num_frames=81, height=480, width=832, **tiler_kwargs)


    # Diffusion / Flow Attack
    pipe.load_models_to_device(["dit"])

    # with torch.no_grad():
    timestep_tensor = pipe.scheduler.timesteps[0].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

    X0_adv_hat = pipe.dit(X_t, timestep=timestep_tensor, **prompt_emb_posi, **image_emb_adv).to(dtype=pipe.torch_dtype, device=pipe.device)

    # Attention hook
    attn_adv = pipe.dit.blocks[2].self_attn.last_attn.clone()

    X0_src_hat = pipe.dit(X_t, timestep=timestep_tensor, **prompt_emb_posi, **image_emb_src).to(dtype=pipe.torch_dtype, device=pipe.device)

    # Attention hook
    attn_src = pipe.dit.blocks[2].self_attn.last_attn.clone()

    L_con = beta * (
        torch.nn.functional.mse_loss(X0_adv_hat, X_tg) +
        torch.clamp(tao_1 - torch.nn.functional.mse_loss(X0_adv_hat, X_0), min=0)
    )
    L_att = gamma * (tao_2 - ((attn_adv - attn_src) ** 2).mean())

    L = L_att + L_con + L_enc + torch.nn.functional.mse_loss(I_adv, pipe.preprocess_image(image).to(pipe.device))
    print(L)

    # import pdb; pdb.set_trace()

    L.backward()
    optimizer.step()

    epsilon = 8.0 / 255 * 2
    delta = torch.clamp(I_adv - I_orig, min=-epsilon, max=epsilon)
    I_adv.data = torch.clamp(I_orig + delta, -1.0, 1.0)


I_adv_out = I_adv.detach().cpu().squeeze(0)
diff = (I_adv_out - I_adv_before).abs()
max_diff = diff.max()
mean_diff = diff.mean()
print(f"Max Diff: {max_diff.item():.6f}")
print(f"Mean Diff: {mean_diff.item():.6f}")


I_adv_out = (I_adv_out + 1.0) / 2.0         
I_adv_out = I_adv_out.clamp(0, 1)  
I_adv_pil = to_pil_image(I_adv_out)
I_adv_pil.save("I_adv_final.jpg")