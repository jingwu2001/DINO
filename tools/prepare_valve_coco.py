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
TRAIN_LABEL_DIR = os.path.join(TRAIN_DIR, "label")
VAL_LABEL_DIR = os.path.join(VAL_DIR, "label")
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

def scan_patients(image_dir, label_dir):
    """
    Fast scan: Counts positives simply by checking if label file exists.
    """
    patient_data = {} 
    
    print(f"Scanning dataset structure in {image_dir}...")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
    patient_folders = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    
    for patient in tqdm(patient_folders):
        patient_img_dir = os.path.join(image_dir, patient)
        patient_data[patient] = {'images': [], 'positive_count': 0}
        
        images = glob.glob(os.path.join(patient_img_dir, "*.*"))
        
        for img_path in images:
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            file_name = os.path.basename(img_path)
            base_name = os.path.splitext(file_name)[0]
            label_path = os.path.join(label_dir, patient, base_name + ".txt")
            
            # OPTIMIZATION: Just check existence, don't open file
            if not os.path.exists(label_path):
                has_label = False
            
            # If the file is empty then also set has_label to False
            if os.path.getsize(label_path) == 0:
                has_label = False
            
            patient_data[patient]['images'].append({
                'path': img_path,
                'label_path': label_path,
                'has_label': has_label
            })
            
            if has_label:
                patient_data[patient]['positive_count'] += 1
            
    return patient_data



def export_coco(patient_data, patient_list, split_name):
    coco_output = {
        "info": {"description": "Aortic Valve Dataset"},
        "categories": [{"id": 0, "name": "aortic_valve"}],
        "images": [],
        "annotations": []
    }
    
    ann_id = 1
    img_id = 1
    
    print(f"Generating {split_name}.json...")
    for patient in tqdm(patient_list):
        images = patient_data[patient]['images']
        n_images = len(images)
        positive_count = patient_data[patient]['positive_count']
        negative_count = n_images - positive_count

        if negative_count == 0:
            add_to_train = [] 
        else:
            target = math.ceil(positive_count * NEGATIVE_RATIO)
            negatives_needed = max(1, target)
            negatives_needed = min(negatives_needed, negative_count)
            add_to_train = np.array([True] * negatives_needed + [False] * (negative_count - negatives_needed))
            add_to_train = np.random.permutation(add_to_train)
        
        neg_idx = 0
        for img_entry in images:
            
            # Image Info
            file_name = os.path.basename(img_entry['path'])
            # DINO expects relative path from the root of your coco_path
            rel_path = os.path.join(patient, file_name) 
            
            coco_output["images"].append({
                "id": img_id,
                "file_name": rel_path,
                "height": IMG_H,
                "width": IMG_W
            })
            
            # Annotations (Only if label file existed during scan)
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

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Process Train
    print("Processing Training Set...")
    train_data = scan_patients(TRAIN_DIR, TRAIN_LABEL_DIR)
    train_patients = list(train_data.keys())
    export_coco(train_data, train_patients, "train")

    # 2. Process Val
    print("Processing Validation Set...")
    val_data = scan_patients(VAL_DIR, VAL_LABEL_DIR)
    val_patients = list(val_data.keys())
    export_coco(val_data, val_patients, "val")
    
    print("Done!")

if __name__ == "__main__":
    main()