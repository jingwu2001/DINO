import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Sort predictions by confidence and round coordinates.")
    parser.add_argument('--input_file', help='Path to predictions.txt')
    parser.add_argument('--output_file', help='Path to output file (default: overwrite input)', default=None)
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file

    print(f"Reading from {args.input_file}...")
    
    data = {}
    try:
        with open(args.input_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                # Expected format: image_name class score x1 y1 x2 y2
                if len(parts) < 7:
                    print(f"Skipping invalid line: {line.strip()}")
                    continue

                image_name = parts[0]
                cls = parts[1]
                try:
                    score = float(parts[2])
                    coords = [float(x) for x in parts[3:]]
                except ValueError:
                    print(f"Skipping line with invalid numbers: {line.strip()}")
                    continue
                
                if image_name not in data:
                    data[image_name] = []
                
                data[image_name].append({
                    'cls': cls,
                    'score': score,
                    'coords': coords
                })
    except FileNotFoundError:
        print(f"Error: File {args.input_file} not found.")
        return

    print(f"Processing {len(data)} images...")

    # Sort and write
    with open(args.output_file, 'w') as f:
        # Sort image names alphabetically
        sorted_images = sorted(data.keys())
        
        for img in sorted_images:
            preds = data[img]
            # Sort by score descending
            preds.sort(key=lambda x: x['score'], reverse=True)
            
            for p in preds:
                # Round coordinates to nearest integer
                rounded_coords = [int(round(c)) for c in p['coords']]
                
                # Reconstruct line
                # image_name class confidence_score x1 y1 x2 y2
                line = f"{img} {p['cls']} {p['score']:.6f} {rounded_coords[0]} {rounded_coords[1]} {rounded_coords[2]} {rounded_coords[3]}\n"
                f.write(line)
    
    print(f"Saved sorted and rounded predictions to {args.output_file}")

if __name__ == '__main__':
    main()
