import os
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO labels to GT txt format.")
    parser.add_argument("--label_dir", default="data/val2017/labels", help="Directory containing .txt label files")
    parser.add_argument("--output_file", default="gt_val.txt", help="Output file path")
    parser.add_argument("--img_size", type=int, default=512, help="Image size (assumed square)")
    args = parser.parse_args()

    print(f"Reading labels from {args.label_dir}...")
    
    label_files = glob.glob(os.path.join(args.label_dir, "*.txt"))
    label_files.sort()
    
    print(f"Found {len(label_files)} label files.")

    with open(args.output_file, 'w') as f_out:
        for label_file in label_files:
            image_name = os.path.splitext(os.path.basename(label_file))[0]
            
            try:
                with open(label_file, 'r') as f_in:
                    lines = f_in.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    # YOLO format: class x_center y_center width height (normalized)
                    cls = int(float(parts[0]))
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    # Convert to x1, y1, x2, y2 (absolute)
                    # x1 = (x_c - w/2) * W
                    # x2 = (x_c + w/2) * W
                    W = args.img_size
                    H = args.img_size
                    
                    x1 = (x_c - w / 2) * W
                    y1 = (y_c - h / 2) * H
                    x2 = (x_c + w / 2) * W
                    y2 = (y_c + h / 2) * H
                    
                    # Round coordinates
                    x1 = int(round(x1))
                    y1 = int(round(y1))
                    x2 = int(round(x2))
                    y2 = int(round(y2))
                    
                    # Format: image_name class 1.0 x1 y1 x2 y2
                    f_out.write(f"{image_name} {cls} 1.0 {x1} {y1} {x2} {y2}\n")
                    
            except Exception as e:
                print(f"Error processing {label_file}: {e}")

    print(f"Saved ground truth to {args.output_file}")

if __name__ == "__main__":
    main()
