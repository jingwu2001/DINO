import os
import torch
from PIL import Image
import torchvision.transforms as T

# Import DINO modules
from util.slconfig import SLConfig
from main import build_model_main

# --- Settings ---
CONFIG_FILE = "config/DINO/DINO_4scale_valve.py"
# Use the best checkpoint saved by main.py
CHECKPOINT_PATH = "logs/DINO/R50-MS4-valve/checkpoint_best_regular.pth" 
TEST_IMG_DIR = "/path/to/testing_image" # Update this path
OUTPUT_FILE = "predictions.txt"
CONFIDENCE_THRESHOLD = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # 1. Setup Model
    args = SLConfig.fromfile(CONFIG_FILE)
    args.device = DEVICE
    model, criterion, postprocessors = build_model_main(args)
    
    # Load trained weights
    print(f"Loading weights from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)
    model.eval()

    # 2. Transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    results_lines = []
    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Running inference on {len(image_files)} images...")

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(TEST_IMG_DIR, img_name)
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
            
            for i in range(len(valid_scores)):
                label = valid_labels[i].item()
                score = valid_scores[i].item()
                
                # Ensure we only output class 0
                if label != 0: continue

                # Un-normalize boxes (cx, cy, w, h) -> (x1, y1, x2, y2)
                cx, cy, bw, bh = valid_boxes[i].tolist()
                cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                
                x1 = max(0, cx - bw / 2)
                y1 = max(0, cy - bh / 2)
                x2 = min(w, cx + bw / 2)
                y2 = min(h, cy + bh / 2)

                # Format: image name, class, score, x1, y1, x2, y2
                line = f"{img_name} {label} {score:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
                results_lines.append(line)

    # Write to file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results_lines))
    print(f"Saved predictions to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()