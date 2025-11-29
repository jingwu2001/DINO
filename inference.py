import os
import sys
import torch
from PIL import Image
import datasets.transforms as T
from main import build_model_main
from util.slconfig import SLConfig
from util.misc import collate_fn
import argparse
from torch.utils.data import Dataset, DataLoader

class SimpleImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(root, file))
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img_pil = Image.open(img_path).convert("RGB")
            w, h = img_pil.size
            if self.transform:
                img_tensor, _ = self.transform(img_pil, None)
            else:
                img_tensor = T.ToTensor()(img_pil, None)[0]
            return img_tensor, {"original_size": (h, w), "image_path": img_path}
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy tensor and info to avoid crashing the whole batch
            # This is a bit risky but better than crashing. 
            # Ideally we should filter these out beforehand.
            return torch.zeros(3, 512, 512), {"original_size": (512, 512), "image_path": "ERROR"}

def get_args_parser():
    parser = argparse.ArgumentParser(description="DINO Inference")
    parser.add_argument("--config_file", default="config/DINO/DINO_valve_4scale.py", type=str)
    parser.add_argument("--checkpoint_path", default="ckpt/checkpoint0033_4scale.pth", type=str)
    parser.add_argument("--input_folder", default="data/testing_image", type=str)
    parser.add_argument("--output_file", default="predictions.txt", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--confidence_threshold", default=0.0, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    return parser

def main(args):
    # Load config
    cfg = SLConfig.fromfile(args.config_file)
    
    # Merge config into args
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
    
    # Build model
    model, criterion, postprocessors = build_model_main(args)
    model.to(args.device)
    model.eval()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)

    # Transform
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform = T.Compose([normalize])

    # Dataset and DataLoader
    dataset = SimpleImageDataset(args.input_folder, transform=transform)
    print(f"Found {len(dataset)} images")
    
    # Custom collate to handle the extra info dict
    def custom_collate(batch):
        # batch is list of (tensor, dict)
        tensors = [item[0] for item in batch]
        infos = [item[1] for item in batch]
        # Use util.misc.collate_fn logic for tensors (NestedTensor)
        # But collate_fn expects a list of tensors, or list of (tensor, target)
        # util.misc.collate_fn: 
        #   batch = list(zip(*batch))
        #   batch[0] = nested_tensor_from_tensor_list(batch[0])
        #   return tuple(batch)
        # So if we pass list of (tensor, info), it will return (NestedTensor, tuple(infos))
        return collate_fn(batch)

    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=custom_collate
    )

    with open(args.output_file, 'w') as f:
        for i, (samples, infos) in enumerate(data_loader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(data_loader)}")
            
            try:
                samples = samples.to(args.device)
                
                # Inference
                with torch.no_grad():
                    outputs = model(samples)
                
                # Post-process
                # target_sizes needs to be [batch_size, 2]
                target_sizes = torch.stack([torch.tensor(info["original_size"]) for info in infos]).to(args.device)
                
                results = postprocessors['bbox'](outputs, target_sizes)
                
                # Iterate over batch results
                for j, result in enumerate(results):
                    info = infos[j]
                    img_path = info["image_path"]
                    if img_path == "ERROR":
                        continue

                    scores = result['scores']
                    labels = result['labels']
                    boxes = result['boxes']
                    
                    image_name = os.path.splitext(os.path.basename(img_path))[0]
                    
                    if len(scores) > 0:
                        # Find the index of the highest score
                        max_idx = scores.argmax()
                        score = scores[max_idx].item()
                        
                        if score >= args.confidence_threshold:
                            label = labels[max_idx].item()
                            box = boxes[max_idx].tolist()
                            
                            # Round coordinates to nearest integer
                            rounded_box = [int(round(c)) for c in box]
                            
                            f.write(f"{image_name} {label} {score:.6f} {rounded_box[0]} {rounded_box[1]} {rounded_box[2]} {rounded_box[3]}\n")
            
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
