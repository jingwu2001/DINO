import argparse
from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image

import datasets.transforms as T
from main import build_model_main
from util.slconfig import SLConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DINO inference on a folder of images and save predictions to a txt file"
    )
    parser.add_argument("--config-file", required=True, help="Path to DINO config (e.g., config/DINO/DINO_valve_4scale.py)")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (the .pth from training)")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing test images")
    parser.add_argument("--output", type=Path, required=True, help="Txt file to write detections")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="Drop predictions below this confidence")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
        help="Image extensions to include",
    )
    return parser.parse_args()


def get_inference_transform() -> T.Compose:
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            normalize,
        ]
    )


def load_model(config_file: str, checkpoint: str, device: str):
    args = SLConfig.fromfile(config_file)
    args.device = device
    model, _, postprocessors = build_model_main(args)
    chk = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(chk["model"], strict=False)
    model.to(device)
    model.eval()
    return model, postprocessors


def iter_images(root: Path, extensions: Iterable[str]) -> List[Path]:
    exts = set(e.lower() for e in extensions)
    paths = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return sorted(paths)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, postprocessors = load_model(args.config_file, args.checkpoint, device)
    transform = get_inference_transform()

    images = iter_images(args.image_dir, args.image_extensions)
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")

    print(f"Found {len(images)} images. Writing predictions to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    with torch.no_grad():
        for idx, img_path in enumerate(images, start=1):
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size
            img_tensor, _ = transform(image, None)
            img_tensor = img_tensor.unsqueeze(0).to(device)

            outputs = model(img_tensor)
            orig_sizes = torch.tensor([[orig_h, orig_w]], device=device)
            results = postprocessors["bbox"](outputs, orig_sizes)[0]

            scores = results["scores"].cpu()
            labels = results["labels"].cpu()
            boxes = results["boxes"].cpu()  # xyxy in original pixel space

            keep = scores >= args.score_threshold
            for score, label, box in zip(scores[keep], labels[keep], boxes[keep]):
                x0, y0, x1, y1 = box.tolist()
                # Force class id 0 as requested
                line = f"{img_path.name} 0 {score:.4f} {x0:.1f} {y0:.1f} {x1:.1f} {y1:.1f}"
                lines.append(line)

            if idx % 50 == 0 or idx == len(images):
                print(f"Processed {idx}/{len(images)}")

    args.output.write_text("\n".join(lines))
    print(f"Saved {len(lines)} predictions to {args.output}")


if __name__ == "__main__":
    main()
