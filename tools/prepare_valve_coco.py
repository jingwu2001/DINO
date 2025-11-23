import os
import json
import glob
import random
from tqdm import tqdm

# --- Configuration ---
ROOT_DIR = "data"  # <--- CHANGE THIS
IMAGE_DIR = os.path.join(ROOT_DIR, "training_image")
LABEL_DIR = os.path.join(ROOT_DIR, "training_label")
OUTPUT_DIR = os.path.join(ROOT_DIR, "annotations")
VAL_SPLIT = 0.2
SEED = 42

IMG_W, IMG_H = 512, 512 

def get_single_yolo_box(label_file, w, h):
    """Reads the single line from YOLO label and converts to COCO."""
    if not os.path.exists(label_file):
        return None
    
    with open(label_file, 'r') as f:
        line = f.readline().strip() # Only read the first line
        
    if not line: return None

    parts = line.split()
    if len(parts) != 5: return None
    
    # YOLO: class, cx, cy, bw, bh
    # Class is ignored (assumed 0/aortic_valve), saved as ID 1
    cx, cy, bw, bh = map(float, parts[1:])
    
    # Convert to absolute COCO coordinates
    x_min = (cx * w) - (bw * w / 2)
    y_min = (cy * h) - (bh * h / 2)
    abs_w = bw * w
    abs_h = bh * h
    
    return {
        "category_id": 1, 
        "bbox": [x_min, y_min, abs_w, abs_h],
        "area": abs_w * abs_h,
        "iscrowd": 0
    }

def scan_patients():
    """
    Fast scan: Counts positives simply by checking if label file exists.
    """
    patient_data = {} 
    
    print("Scanning dataset structure...")
    
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
        
    patient_folders = [d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))]
    
    for patient in tqdm(patient_folders):
        patient_img_dir = os.path.join(IMAGE_DIR, patient)
        patient_data[patient] = {'images': [], 'positive_count': 0}
        
        images = glob.glob(os.path.join(patient_img_dir, "*.*"))
        
        for img_path in images:
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            file_name = os.path.basename(img_path)
            base_name = os.path.splitext(file_name)[0]
            label_path = os.path.join(LABEL_DIR, patient, base_name + ".txt")
            
            # OPTIMIZATION: Just check existence, don't open file
            has_label = os.path.exists(label_path)
            
            patient_data[patient]['images'].append({
                'path': img_path,
                'label_path': label_path,
                'has_label': has_label
            })
            
            if has_label:
                patient_data[patient]['positive_count'] += 1
            
    return patient_data

def split_patients_stratified(patient_data):
    """Splits patients ensuring Val gets ~20% of total positive samples."""
    
    patients = list(patient_data.keys())
    random.shuffle(patients) 
    
    total_positives = sum(p['positive_count'] for p in patient_data.values())
    target_val_positives = total_positives * VAL_SPLIT
    
    train_patients = []
    val_patients = []
    current_val_positives = 0
    
    for patient in patients:
        p_positives = patient_data[patient]['positive_count']
        
        # Greedy assignment to Validation until target is hit
        if current_val_positives < target_val_positives:
            val_patients.append(patient)
            current_val_positives += p_positives
        else:
            train_patients.append(patient)
            
    print("-" * 30)
    print(f"Total Positive Images: {total_positives}")
    print(f"Target Val Positives:  {int(target_val_positives)}")
    print(f"Actual Val Positives:  {current_val_positives} ({current_val_positives/total_positives:.1%})")
    print(f"Train Patients: {len(train_patients)} | Val Patients: {len(val_patients)}")
    print("-" * 30)
    
    return train_patients, val_patients

def export_coco(patient_data, patient_list, split_name):
    coco_output = {
        "info": {"description": "Aortic Valve Dataset"},
        "categories": [{"id": 1, "name": "aortic_valve"}],
        "images": [],
        "annotations": []
    }
    
    ann_id = 1
    img_id = 1
    
    print(f"Generating {split_name}.json...")
    for patient in tqdm(patient_list):
        images = patient_data[patient]['images']
        
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
            
    save_path = os.path.join(OUTPUT_DIR, f"instances_{split_name}.json")
    with open(save_path, 'w') as f:
        json.dump(coco_output, f)

def main():
    random.seed(SEED)
    
    # 1. Fast Scan
    patient_data = scan_patients()
    
    # 2. Stratified Split
    train_p, val_p = split_patients_stratified(patient_data)
    
    # 3. Export
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    export_coco(patient_data, train_p, "train")
    export_coco(patient_data, val_p, "val")
    
    print("Done!")

if __name__ == "__main__":
    main()