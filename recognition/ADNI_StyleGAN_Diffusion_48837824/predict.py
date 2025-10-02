"""
Generate images from trained conditional StyleGAN2
Use this after training to generate AD or NC samples on demand
"""

import torch
from torchvision.utils import save_image
import argparse
import os

from modules import Generator
from modules import ConditionalMappingNetwork as MappingNetwork
from utils import get_w, get_noise, DEVICE, W_DIM, LOG_RESOLUTION, Z_DIM, CLASS_NAMES

def load_checkpoint(checkpoint_path, device=DEVICE):
    """Load trained models from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize models
    gen = Generator(LOG_RESOLUTION, W_DIM).to(device)
    mapping = MappingNetwork(Z_DIM, W_DIM, num_classes=2).to(device)
    
    # Load weights
    gen.load_state_dict(checkpoint['gen_state'])
    mapping.load_state_dict(checkpoint['mapping_state'])
    
    gen.eval()
    mapping.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    return gen, mapping

def generate_class_samples(gen, mapping, class_idx, num_samples=16, output_dir='generated_samples', device=DEVICE):
    """
    Generate samples from a specific class.
    
    Args:
        gen: Generator model
        mapping: Mapping network
        class_idx: 0 for AD, 1 for NC
        num_samples: Number of images to generate
        output_dir: Directory to save images
        device: Device
    """
    os.makedirs(output_dir, exist_ok=True)
    class_name = CLASS_NAMES[class_idx]
    
    print(f"\nGenerating {num_samples} {class_name} images...")
    
    with torch.no_grad():
        # Create class labels
        class_labels = torch.full((num_samples,), class_idx, dtype=torch.long, device=device)
        
        # Generate images
        w = get_w(num_samples, mapping, class_labels, device)
        noise = get_noise(num_samples, device)
        fake_imgs = gen(w, noise)
        
        # Denormalize
        fake_imgs = fake_imgs * 0.5 + 0.5
        fake_imgs = torch.clamp(fake_imgs, 0, 1)
        
        # Save grid
        grid_path = f"{output_dir}/{class_name}_grid_{num_samples}samples.png"
        save_image(fake_imgs, grid_path, nrow=4, padding=2, normalize=False)
        print(f"✓ Saved grid: {grid_path}")
        
        # Save individual images
        for i in range(num_samples):
            img_path = f"{output_dir}/{class_name}_sample_{i+1:03d}.png"
            save_image(fake_imgs[i], img_path, normalize=False)
        
        print(f"✓ Saved {num_samples} individual images to {output_dir}/")
    
    return fake_imgs

def generate_mixed_batch(gen, mapping, num_per_class=8, output_dir='generated_samples', device=DEVICE):
    """Generate equal numbers of AD and NC images for comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating mixed batch ({num_per_class} AD + {num_per_class} NC)...")
    
    with torch.no_grad():
        all_imgs = []
        
        for class_idx in [0, 1]:
            class_labels = torch.full((num_per_class,), class_idx, dtype=torch.long, device=device)
            w = get_w(num_per_class, mapping, class_labels, device)
            noise = get_noise(num_per_class, device)
            imgs = gen(w, noise)
            imgs = imgs * 0.5 + 0.5
            imgs = torch.clamp(imgs, 0, 1)
            all_imgs.append(imgs)
        
        # Concatenate: AD on top, NC on bottom
        combined = torch.cat(all_imgs, dim=0)
        
        grid_path = f"{output_dir}/mixed_comparison_{num_per_class}x2.png"
        save_image(combined, grid_path, nrow=num_per_class, padding=2, normalize=False)
        print(f"✓ Saved comparison grid: {grid_path}")
        print(f"  (Top row: AD, Bottom row: NC)")

def generate_latent_walk(gen, mapping, class_idx, steps=10, output_dir='generated_samples', device=DEVICE):
    """Generate a walk through latent space for one class"""
    os.makedirs(output_dir, exist_ok=True)
    class_name = CLASS_NAMES[class_idx]
    
    print(f"\nGenerating latent walk for {class_name}...")
    
    with torch.no_grad():
        # Two random starting points
        z1 = torch.randn(1, Z_DIM, device=device)
        z2 = torch.randn(1, Z_DIM, device=device)
        
        class_labels = torch.tensor([class_idx], dtype=torch.long, device=device)
        noise = get_noise(1, device)
        
        imgs = []
        for alpha in torch.linspace(0, 1, steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            w = mapping(z_interp, class_labels)
            
            # Expand w for all layers
            num_blocks = LOG_RESOLUTION - 2
            num_layers = 1 + 2 * num_blocks
            w = w.unsqueeze(1).expand(-1, num_layers, -1)
            
            img = gen(w, noise)
            img = img * 0.5 + 0.5
            img = torch.clamp(img, 0, 1)
            imgs.append(img)
        
        imgs = torch.cat(imgs, dim=0)
        walk_path = f"{output_dir}/{class_name}_latent_walk.png"
        save_image(imgs, walk_path, nrow=steps, padding=2, normalize=False)
        print(f"✓ Saved latent walk: {walk_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images from conditional StyleGAN2')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--class_idx', type=int, choices=[0, 1], default=None, 
                       help='Class to generate: 0=AD, 1=NC (default: both)')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples per class')
    parser.add_argument('--output_dir', type=str, default='generated_samples', help='Output directory')
    parser.add_argument('--mixed', action='store_true', help='Generate mixed AD/NC comparison')
    parser.add_argument('--walk', action='store_true', help='Generate latent space walk')
    
    args = parser.parse_args()
    
    # Load model
    gen, mapping = load_checkpoint(args.checkpoint)
    
    print(f"\nClass mapping: 0=AD (Alzheimer's), 1=NC (Normal Control)")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)
    
    # Generate based on arguments
    if args.mixed:
        generate_mixed_batch(gen, mapping, args.num_samples // 2, args.output_dir)
    
    elif args.walk:
        if args.class_idx is None:
            # Generate walks for both classes
            for class_idx in [0, 1]:
                generate_latent_walk(gen, mapping, class_idx, steps=10, output_dir=args.output_dir)
        else:
            generate_latent_walk(gen, mapping, args.class_idx, steps=10, output_dir=args.output_dir)
    
    elif args.class_idx is not None:
        # Generate specific class
        generate_class_samples(gen, mapping, args.class_idx, args.num_samples, args.output_dir)
    
    else:
        # Generate both classes by default
        for class_idx in [0, 1]:
            generate_class_samples(gen, mapping, class_idx, args.num_samples, args.output_dir)
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"Check {args.output_dir}/ for generated images")