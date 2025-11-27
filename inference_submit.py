import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# Import DINO modules
from util.slconfig import SLConfig
from main import build_model_main

def get_args():
    parser = argparse.ArgumentParser(description="DINO Inference for Submission")
    parser.add_argument("--config_file", type=str, default="config/DINO/DINO_valve_4scale.py", help="Path to config file")
    parser.add_argument("--checkpoint_path", type=str, default="logs/DINO/R50-MS4-valve/checkpoint.pth", help="Path to checkpoint")
    parser.add_argument("--input_dir", type=str, default="data/testing_image", help="Directory containing images")
    parser.add_argument("--output_file", type=str, default="predictions.txt", help="Output text file")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Confidence threshold")
    return parser.parse_args()

def get_all_images(root_dir):
    """Recursively finds all images in subfolders."""
    image_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                full_path = os.path.join(root, file)
                image_list.append(full_path)
    return image_list

def main():
    args_cli = get_args()

    # 1. Setup Model
    if not os.path.exists(args_cli.config_file):
        print(f"Error: Config file not found at {args_cli.config_file}")
        return

    args = SLConfig.fromfile(args_cli.config_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    
    # Build model
    model, criterion, postprocessors = build_model_main(args)
    
    # Load trained weights
    if not os.path.exists(args_cli.checkpoint_path):
        print(f"Error: Checkpoint not found at {args_cli.checkpoint_path}")
        return

    print(f"Loading weights from {args_cli.checkpoint_path}...")
    checkpoint = torch.load(args_cli.checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # 2. Transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    results_lines = []
    
    # 3. Get list of all images
    if not os.path.exists(args_cli.input_dir):
         print(f"Error: Input directory not found at {args_cli.input_dir}")
         return

    all_image_paths = get_all_images(args_cli.input_dir)
    print(f"Found {len(all_image_paths)} images in {args_cli.input_dir}.")

    with torch.no_grad():
        for img_path in tqdm(all_image_paths):
            # Filename without extension
            img_name_full = os.path.basename(img_path)
            img_name_no_ext = os.path.splitext(img_name_full)[0]
            
            try:
                image_pil = Image.open(img_path).convert("RGB")
                w, h = image_pil.size
                
                img_tensor = transform(image_pil).unsqueeze(0).to(device)
                outputs = model(img_tensor)
                
                # Get first batch item [0]
                probs = outputs['pred_logits'][0].sigmoid()
                boxes = outputs['pred_boxes'][0]

                # Get top score per query
                scores, labels = probs.max(-1)
                
                # Filter by threshold
                keep = scores > args_cli.confidence_threshold
                
                valid_scores = scores[keep]
                valid_labels = labels[keep]
                valid_boxes = boxes[keep]
                
                # Best Box Logic (Pick highest score if any exist)
                if len(valid_scores) > 0:
                    best_idx = valid_scores.argmax()
                    
                    label = valid_labels[best_idx].item()
                    score = valid_scores[best_idx].item()
                    
                    # Force label to 0 as per requirement "object class (which should be zero...)"
                    # But we should check if the model actually predicts 0 (valve). 
                    # If the model predicts something else, we might want to skip or force it.
                    # Given the user said "object class (which should be zero for all images with an object in it)",
                    # I will output 0.
                    output_label = 0 

                    # Un-normalize boxes (cx, cy, w, h) -> (x1, y1, x2, y2)
                    cx, cy, bw, bh = valid_boxes[best_idx].tolist()
                    cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                    
                    x1 = max(0, cx - bw / 2)
                    y1 = max(0, cy - bh / 2)
                    x2 = min(w, cx + bw / 2)
                    y2 = min(h, cy + bh / 2)

                    # Round to integers
                    x1_int = int(round(x1))
                    y1_int = int(round(y1))
                    x2_int = int(round(x2))
                    y2_int = int(round(y2))

                    # Format: image name, class, score, x1, y1, x2, y2
                    # Example: patient0083_0048 0 0.414 180 258 195 267
                    line = f"{img_name_no_ext} {output_label} {score:.4f} {x1_int} {y1_int} {x2_int} {y2_int}"
                    results_lines.append(line)
            except Exception as e:
                print(f"Error processing {img_name_full}: {e}")

    # Write to file
    with open(args_cli.output_file, "w") as f:
        f.write("\n".join(results_lines))
    print(f"Saved predictions to {args_cli.output_file}")

if __name__ == '__main__':
    main()
