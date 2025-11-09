from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

def crop_and_resize(image, mask, output_size=(224, 224), offset_ratio=0.15):
    """
    Crop and resize an image and its corresponding mask to a square region.

    Args:
        image (PIL.Image.Image): Input image.
        mask (PIL.Image.Image): Corresponding mask image.
        output_size (tuple): Target output size (width, height). Default is (224, 224).
        offset_ratio (float): Horizontal offset ratio for the crop. Default is 0.15.

    Returns:
        (PIL.Image.Image, PIL.Image.Image): Cropped and resized image and mask.
    """
    w, h = image.size
    side = min(w, h)

    left = (w - side) // 2 - side * offset_ratio
    top = (h - side) // 2
    right = left + side
    bottom = top + side

    left = max(left, 0)
    top = max(top, 0)
    right = min(right, w)
    bottom = min(bottom, h)

    image_cropped = image.crop((left, top, right, bottom))
    mask_cropped = mask.crop((left, top, right, bottom))

    image_resized = image_cropped.resize(output_size, Image.BICUBIC)
    mask_resized = mask_cropped.resize(output_size, Image.BICUBIC)

    return image_resized, mask_resized


def make_hook(saved_features, name):
    """
    Create a forward hook that stores the output of a layer in a shared dict.
    """
    def hook(module, inp, out):
        saved_features[name] = out
    return hook


def register_vae_hooks(pipe):
    """
    Register forward hooks for VAE encoder and decoder layers in a diffusion pipeline.

    Args:
        pipe: A diffusion pipeline object that contains `pipe.vae.model`.

    Returns:
        dict: A dictionary that stores the captured feature maps during forward passes.
    """
    saved_features = {}

    vae = pipe.vae.model
    encoder = vae.encoder
    decoder = vae.decoder

    # Encoder hooks
    encoder.conv1.register_forward_hook(make_hook(saved_features, "conv1"))
    for i, down in enumerate(encoder.downsamples):
        down.register_forward_hook(make_hook(saved_features, f"downsample_{i}"))
    list(encoder.middle)[-1].register_forward_hook(make_hook(saved_features, "middle"))

    # VAE latent layer hook
    vae.conv1.register_forward_hook(make_hook(saved_features, "mu_logvar"))

    # Decoder hooks
    for i, up in enumerate(decoder.upsamples):
        up.register_forward_hook(make_hook(saved_features, f"upsample_{i}"))

    return saved_features



def setup_pipe_modules(pipe, enable_vram_management=False, num_persistent_param_in_dit=None):
    """
    Move all core submodules of a diffusion pipeline to the correct device and optionally enable VRAM management.

    Args:
        pipe: The initialized diffusion pipeline (e.g., WanVideoPipeline, FluxPipeline).
        enable_vram_management (bool): Whether to enable VRAM management.
        num_persistent_param_in_dit (int, optional): Persistent parameter count for VRAM control.

    Returns:
        The same pipe, after moving modules and optional VRAM setup.
    """
    # Move major submodules to the target device if they exist
    for module_name in ["dit", "vae", "image_encoder", "text_encoder"]:
        if hasattr(pipe, module_name):
            getattr(pipe, module_name).to(pipe.device)

    # Optional VRAM management
    if enable_vram_management:
        if num_persistent_param_in_dit:
            pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)
        else:
            pipe.enable_vram_management()

    return pipe


def init_adv_image(I, epsilon=0.03, value_range=(-1.0, 1.0)):
    if not isinstance(I, torch.Tensor):
        raise TypeError("I must be a torch.Tensor")
    I_adv = I.clone()
    noise = torch.empty_like(I_adv).uniform_(-epsilon, epsilon)
    I_adv = I_adv + noise
    I_adv = torch.clamp(I_adv, value_range[0], value_range[1]).detach()
    I_adv.requires_grad_(True)
    return I_adv



def plot_loss_curve(loss_history, save_path="loss_curve.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, label="Total Loss L")
    plt.xlabel("Step")
    plt.ylabel("Loss Value")
    plt.title("Attack Loss Curve during Optimization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()




def save_adv_result(I_adv, I_adv_before, save_path="I_adv_final.jpg"):
    """
    Compare two adversarial images and save the final output.

    Steps:
      1. Compute max and mean absolute pixel differences.
      2. Convert the final adversarial tensor to [0,1] range.
      3. Save it as a PIL image.

    Args:
        I_adv (torch.Tensor): Current adversarial image tensor (usually with grad).
        I_adv_before (torch.Tensor): Previous iteration's adversarial image tensor.
        save_path (str): Output file path for saving the image.

    Returns:
        dict: Contains max_diff and mean_diff (floats).
    """
    # Detach and move to CPU for comparison
    I_adv_out = I_adv.detach().cpu().squeeze(0)
    I_adv_before = I_adv_before.detach().cpu()

    # Compute difference metrics
    diff = (I_adv_out - I_adv_before).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max Diff: {max_diff:.6f}")
    print(f"Mean Diff: {mean_diff:.6f}")

    # Convert tensor from [-1, 1] → [0, 1] and save as image
    I_adv_out = (I_adv_out + 1.0) / 2.0
    I_adv_out = I_adv_out.to(torch.float32).clamp(0, 1)

    I_adv_pil = to_pil_image(I_adv_out)
    I_adv_pil.save(save_path)

    return {"max_diff": max_diff, "mean_diff": mean_diff}
