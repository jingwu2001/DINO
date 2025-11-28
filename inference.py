import os
import sys
import torch
from PIL import Image
import datasets.transforms as T
from main import build_model_main
from util.slconfig import SLConfig
import argparse
from util.box_ops import box_cxcywh_to_xyxy

def get_args_parser():
    parser = argparse.ArgumentParser(description="DINO Inference")
    parser.add_argument("--config_file", default="config/DINO/DINO_valve_4scale.py", type=str)
    parser.add_argument("--checkpoint_path", default="ckpt/checkpoint0033_4scale.pth", type=str)
    parser.add_argument("--input_folder", default="data/testing_image", type=str)
    parser.add_argument("--output_file", default="predictions.txt", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--confidence_threshold", default=0.0, type=float)
    # Add other args that might be needed by build_model_main if not in config
    # Usually config has everything, but we need to merge them.
    return parser

def main(args):
    # Load config
    cfg = SLConfig.fromfile(args.config_file)
    
    # Merge config into args
    # This mimics what main.py does
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            # If it's in args, we might want to keep the arg value (like device)
            # But main.py raises ValueError if key is in both and not handled.
            # Here we just let args override config if it exists, or config override args?
            # main.py says: raise ValueError("Key {} can used by args only".format(k))
            # This implies config keys should NOT be in args parser if they are in config.
            # But we are using a simplified parser.
            # Let's just set attributes safely.
            pass
    
    # Ensure device is set correctly in args (it's already there from parser)
    
    # Build model using build_model_main
    # This returns model, criterion, postprocessors
    model, criterion, postprocessors = build_model_main(args)
    model.to(args.device)
    model.eval()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)

    # Transform
    # Standard ImageNet normalization
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform = T.Compose([normalize])

    # Find images
    image_paths = []
    for root, dirs, files in os.walk(args.input_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    
    image_paths.sort()
    print(f"Found {len(image_paths)} images")

    with open(args.output_file, 'w') as f:
        for i, img_path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"Processing {i}/{len(image_paths)}")
            
            try:
                # Load image
                img_pil = Image.open(img_path).convert("RGB")
                w, h = img_pil.size
                
                # Transform
                img_tensor, _ = transform(img_pil, None)
                img_tensor = img_tensor.unsqueeze(0).to(args.device)
                
                # Inference
                with torch.no_grad():
                    outputs = model(img_tensor)
                
                # Post-process
                # PostProcess expects target_sizes as tensor of shape [batch_size, 2] (h, w)
                target_sizes = torch.tensor([[h, w]], device=args.device)
                results = postprocessors['bbox'](outputs, target_sizes)
                
                # Results is a list of dicts
                result = results[0]
                scores = result['scores']
                labels = result['labels']
                boxes = result['boxes'] # These are already xyxy and scaled to image size by PostProcess
                
                # Get image name without suffix
                image_name = os.path.splitext(os.path.basename(img_path))[0]
                
                for j in range(len(scores)):
                    score = scores[j].item()
                    if score < args.confidence_threshold:
                        continue
                    
                    label = labels[j].item()
                    box = boxes[j].tolist()
                    
                    # Format: image_name class confidence_score topleft_x topleft_y bottomleft_x bottomleft_y
                    # box is x1, y1, x2, y2 (top-left, bottom-right)
                    f.write(f"{image_name} {label} {score:.6f} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f}\n")
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
