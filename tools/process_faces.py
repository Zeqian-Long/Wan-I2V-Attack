from pathlib import Path

from PIL import Image


TARGET_WIDTH = 832
TARGET_HEIGHT = 480
TARGET_RATIO = TARGET_WIDTH / TARGET_HEIGHT


# Manual per-image framing so the face stays natural after conversion.
# Each value is an anchor point in normalized coordinates for the crop center.
ANCHORS = {
    "Daniel": (0.54, 0.43),
    "LeCun": (0.58, 0.50),
    "musk": (0.67, 0.34),
    "Hinton": (0.50, 0.46),
    "yoshua": (0.53, 0.42),
    "下载": (0.50, 0.47),
}


# Preserve the existing labels where the filenames already provide them.
# The last image does not have a usable identity label, so keep it neutral.
OUTPUT_NAMES = {
    "Daniel": "Daniel.jpg",
    "LeCun": "LeCun.jpg",
    "musk": "Musk.jpg",
    "Hinton": "Hinton.jpg",
    "下载": "Unknown.jpg",
}


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def crop_to_ratio(image: Image.Image, anchor_x: float, anchor_y: float) -> Image.Image:
    width, height = image.size
    src_ratio = width / height

    if abs(src_ratio - TARGET_RATIO) < 1e-6:
        return image

    if src_ratio > TARGET_RATIO:
        crop_height = height
        crop_width = int(round(height * TARGET_RATIO))
    else:
        crop_width = width
        crop_height = int(round(width / TARGET_RATIO))

    center_x = int(round(width * anchor_x))
    center_y = int(round(height * anchor_y))

    left = clamp(center_x - crop_width // 2, 0, width - crop_width)
    top = clamp(center_y - crop_height // 2, 0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom))


def process_image(path: Path, output_path: Path) -> None:
    with Image.open(path) as image:
        image = image.convert("RGB")
        anchor_x, anchor_y = ANCHORS.get(path.stem, (0.5, 0.5))
        cropped = crop_to_ratio(image, anchor_x, anchor_y)
        resized = cropped.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
        resized.save(output_path, quality=95)


def main() -> None:
    faces_dir = Path("faces")
    for path in sorted(faces_dir.glob("*.jpg")):
        output_name = OUTPUT_NAMES.get(path.stem, f"{path.stem}.jpg")
        output_path = faces_dir / output_name
        process_image(path, output_path)
        print(f"{path.name} -> {output_path.name}")


if __name__ == "__main__":
    main()
