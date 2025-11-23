import os
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# Import DINO modules
from util.slconfig import SLConfig
from main import build_model_main

# --- Settings ---
CONFIG_FILE = "config/DINO/DINO_valve_4scale.py"
# Checkpoint Path (Make sure this points to your .pth file)
CHECKPOINT_PATH = "logs/DINO/R50-MS4-valve/checkpoint.pth" 
# Root of the testing images
TEST_IMG_DIR = "data/testing_image" 
OUTPUT_FILE = "predictions.txt"
CONFIDENCE_THRESHOLD = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_all_images(root_dir):
    """Recursively finds all images in subfolders."""
    image_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Store full path to open, and filename for reporting
                full_path = os.path.join(root, file)
                image_list.append(full_path)
    return image_list

def main():
    # 1. Setup Model
    args = SLConfig.fromfile(CONFIG_FILE)
    args.device = DEVICE
    model, criterion, postprocessors = build_model_main(args)
    
    # Load trained weights
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    print(f"Loading weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)
    model.eval()

    # 2. Transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    results_lines = []
    
    # 3. Get list of all images (recursive)
    if not os.path.exists(TEST_IMG_DIR):
         print(f"Error: Test image directory not found at {TEST_IMG_DIR}")
         return

    all_image_paths = get_all_images(TEST_IMG_DIR)
    print(f"Found {len(all_image_paths)} images in patient subfolders.")

    with torch.no_grad():
        for img_path in tqdm(all_image_paths):
            # Filename for the output text (e.g., "patient0051_01.png")
            # If you need the relative path (e.g. "patient0051/scan.png"), change this to:
            # img_name = os.path.relpath(img_path, TEST_IMG_DIR)
            img_name = os.path.basename(img_path) 
            
            try:
                image_pil = Image.open(img_path).convert("RGB")
                w, h = image_pil.size
                
                img_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
                outputs = model(img_tensor)
                
                # Get first batch item [0]
                probs = outputs['pred_logits'][0].sigmoid()
                boxes = outputs['pred_boxes'][0]

                # Get top score per query
                scores, labels = probs.max(-1)
                
                # Filter
                keep = scores > CONFIDENCE_THRESHOLD
                
                valid_scores = scores[keep]
                valid_labels = labels[keep]
                valid_boxes = boxes[keep]
                
                # Best Box Logic (Pick highest score if any exist)
                if len(valid_scores) > 0:
                    best_idx = valid_scores.argmax()
                    
                    label = valid_labels[best_idx].item()
                    score = valid_scores[best_idx].item()
                    
                    if label == 0: # Ensure it is aortic valve
                        # Un-normalize boxes (cx, cy, w, h) -> (x1, y1, x2, y2)
                        cx, cy, bw, bh = valid_boxes[best_idx].tolist()
                        cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                        
                        x1 = max(0, cx - bw / 2)
                        y1 = max(0, cy - bh / 2)
                        x2 = min(w, cx + bw / 2)
                        y2 = min(h, cy + bh / 2)

                        # Format: image name, class, score, x1, y1, x2, y2
                        line = f"{img_name} {label} {score:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
                        results_lines.append(line)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    # Write to file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results_lines))
    print(f"Saved predictions to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()