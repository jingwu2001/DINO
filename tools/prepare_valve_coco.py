import os
import json
import glob
import random
import math
import numpy as np
from tqdm import tqdm

# --- Configuration ---
ROOT_DIR = "data"  # <--- CHANGE THIS
TRAIN_DIR = os.path.join(ROOT_DIR, "train2017")
VAL_DIR = os.path.join(ROOT_DIR, "val2017")
OUTPUT_DIR = os.path.join(ROOT_DIR, "annotations")
NEGATIVE_RATIO = 1.0 # Keep 1 negative for every 1 positive
SEED = 42

IMG_W, IMG_H = 512, 512 

def get_single_yolo_box(label_file, w, h):
    """Reads the single line from YOLO label and converts to COCO."""
    if not os.path.exists(label_file):
        return None
    
    with open(label_file, 'r') as f:
        line = f.readline().strip() # Only read the first line
        
    if not line: 
        return None

    parts = line.split()
    if len(parts) != 5: 
        return None
    
    # YOLO: class, cx, cy, bw, bh
    # Class is ignored (assumed 0/aortic_valve), saved as ID 1
    cx, cy, bw, bh = map(float, parts[1:])
    
    # Convert to absolute COCO coordinates
    x_min = (cx * w) - (bw * w / 2)
    y_min = (cy * h) - (bh * h / 2)
    abs_w = bw * w
    abs_h = bh * h
    
    return {
        "category_id": 0, 
        "bbox": [x_min, y_min, abs_w, abs_h],
        "area": abs_w * abs_h,
        "iscrowd": 0
    }

def scan_dataset(root_dir):
    """
    Scans a dataset folder (e.g., data/train2017) containing 'images' and 'labels' subfolders.
    Returns a list of image entries.
    """
    image_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "labels")
    
    print(f"Scanning dataset in {root_dir}...")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Support multiple extensions if needed, but user said .png
    images = glob.glob(os.path.join(image_dir, "*.png"))
    
    dataset_entries = []
    positive_count = 0
    
    for img_path in tqdm(images):
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Label is expected in labels/ folder with same basename
        label_path = os.path.join(label_dir, base_name + ".txt")
        
        has_label = False
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            has_label = True
            positive_count += 1
        
        dataset_entries.append({
            'path': img_path,
            'label_path': label_path,
            'has_label': has_label
        })
        
    print(f"Found {len(dataset_entries)} images, {positive_count} positives.")
    return dataset_entries, positive_count

def export_coco(image_entries, positive_count, split_name):
    coco_output = {
        "info": {"description": "Aortic Valve Dataset"},
        "categories": [{"id": 0, "name": "aortic_valve"}],
        "images": [],
        "annotations": []
    }
    
    ann_id = 1
    img_id = 1
    
    n_images = len(image_entries)
    negative_count = n_images - positive_count
    
    # --- Negative Sampling Logic ---
    if negative_count == 0:
        keep_mask = [] 
    else:
        target = math.ceil(positive_count * NEGATIVE_RATIO)
        negatives_needed = max(1, target)
        negatives_needed = min(negatives_needed, negative_count)
        # Create a boolean mask for negatives: [True, True, ... False, False]
        keep_mask = np.array([True] * negatives_needed + [False] * (negative_count - negatives_needed))
        np.random.shuffle(keep_mask)
    
    neg_idx = 0
    
    print(f"Generating {split_name}.json...")
    for img_entry in tqdm(image_entries):
        
        # Handle Negative Sampling
        if not img_entry['has_label']:
            if neg_idx >= len(keep_mask):
                continue # Should not happen if counts are correct
            
            keep_negative = keep_mask[neg_idx]
            neg_idx += 1
            
            if not keep_negative:
                continue

        # Image Info
        file_name = os.path.basename(img_entry['path'])
        # Rel path: e.g., "train2017/images/patient001.png"
        # DINO root is 'data', so we need path relative to 'data'
        rel_path = os.path.join("images", file_name)
        
        coco_output["images"].append({
            "id": img_id,
            "file_name": rel_path,
            "height": IMG_H,
            "width": IMG_W
        })
        
        # Annotations
        if img_entry['has_label']:
            box = get_single_yolo_box(img_entry['label_path'], IMG_W, IMG_H)
            if box:
                box["id"] = ann_id
                box["image_id"] = img_id
                coco_output["annotations"].append(box)
                ann_id += 1
        
        img_id += 1
            
    save_path = os.path.join(OUTPUT_DIR, f"instances_{split_name}2017.json")
    with open(save_path, 'w') as f:
        json.dump(coco_output, f)
    print(f"Saved to {save_path}")

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Process Train
    if os.path.exists(TRAIN_DIR):
        print("Processing Training Set...")
        train_entries, train_pos = scan_dataset(TRAIN_DIR)
        export_coco(train_entries, train_pos, "train")
    else:
        print(f"Warning: Train directory not found at {TRAIN_DIR}")

    # 2. Process Val
    if os.path.exists(VAL_DIR):
        print("Processing Validation Set...")
        val_entries, val_pos = scan_dataset(VAL_DIR)
        export_coco(val_entries, val_pos, "val")
    else:
        print(f"Warning: Val directory not found at {VAL_DIR}")
    
    print("Done!")

if __name__ == "__main__":
    main()