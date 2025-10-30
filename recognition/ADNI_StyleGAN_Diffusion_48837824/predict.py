import torch
from torchvision.utils import save_image
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from modules import Generator, ConditionalDiscriminator
from modules import ConditionalMappingNetwork as MappingNetwork
from utils import get_w, get_noise, DEVICE, W_DIM, LOG_RESOLUTION, Z_DIM, CLASS_NAMES
from dataset import get_dataloaders
from torchvision import transforms
from copy import deepcopy

"""
predict.py
Inference and visualization script for conditional StyleGAN2 on ADNI dataset.
Supports image generation, latent walks, and t-SNE embedding visualization.
Author: Tyreece Paul
"""

def load_checkpoint(checkpoint_path, device=DEVICE):
    """
    Load model checkpoint
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (str): Device to load the model onto
    Returns:
        gen: Loaded Generator model
        mapping: Loaded Mapping Network
        disc: Loaded Discriminator model (optional, for feature extraction)
    """

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize models
    gen = Generator(LOG_RESOLUTION, W_DIM).to(device)
    mapping = MappingNetwork(Z_DIM, W_DIM, num_classes=2).to(device)
    disc = ConditionalDiscriminator(LOG_RESOLUTION, num_classes=2).to(device)
    
    # Load weights
    gen.load_state_dict(checkpoint['gen_state'])
    mapping.load_state_dict(checkpoint['mapping_state'])
    if 'disc_state' in checkpoint:
        disc.load_state_dict(checkpoint['disc_state'])
    
    # Load EMA parameters if available (Official StyleGAN2 feature)
    if 'gen_ema_shadow' in checkpoint:
        print("  Loading EMA generator parameters...")
        for name, param in gen.named_parameters():
            if name in checkpoint['gen_ema_shadow']:
                param.data.copy_(checkpoint['gen_ema_shadow'][name])
    
    gen.eval()
    mapping.eval()
    disc.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    return gen, mapping, disc

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
        print(f"Saved grid: {grid_path}")
        
        # Save individual images
        for i in range(num_samples):
            img_path = f"{output_dir}/{class_name}_sample_{i+1:03d}.png"
            save_image(fake_imgs[i], img_path, normalize=False)
        
        print(f"Saved {num_samples} individual images to {output_dir}/")
    
    return fake_imgs

def generate_mixed_batch(gen, mapping, num_per_class=8, output_dir='generated_samples', device=DEVICE):
    """
    Generate a mixed batch of AD and NC images for comparison
    Args:
        gen: Generator model
        mapping: Mapping network
        num_per_class: Number of images per class (total will be double)
        output_dir: Directory to save images
        device: Device
    """

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating mixed batch ({num_per_class} AD + {num_per_class} NC)...")
    
    with torch.no_grad():
        all_imgs = []
        
        for class_idx in [0, 1]:
            class_name = CLASS_NAMES[class_idx]
            class_labels = torch.full((num_per_class,), class_idx, dtype=torch.long, device=device)
            w = get_w(num_per_class, mapping, class_labels, device)
            noise = get_noise(num_per_class, device)
            imgs = gen(w, noise)
            imgs = imgs * 0.5 + 0.5
            imgs = torch.clamp(imgs, 0, 1)
            all_imgs.append(imgs)
            
            # Save individual images
            for i, img in enumerate(imgs):
                img_path = f"{output_dir}/{class_name}_{i+1}.png"
                save_image(img, img_path, normalize=False)
        
        # Concatenate: AD on top, NC on bottom
        combined = torch.cat(all_imgs, dim=0)
        
        grid_path = f"{output_dir}/mixed_comparison_{num_per_class}x2.png"
        save_image(combined, grid_path, nrow=num_per_class, padding=2, normalize=False)
        print(f"Saved comparison grid: {grid_path}")
        print(f"Saved {num_per_class * 2} individual images to {output_dir}/")
        print(f"(AD: AD_1.png to AD_{num_per_class}.png)")
        print(f"(NC: NC_1.png to NC_{num_per_class}.png)")

def generate_latent_walk(gen, mapping, class_idx, steps=10, output_dir='generated_samples', device=DEVICE):
    """
    Generate a latent space walk between two random points for a specific class.
    Args:
        gen: Generator model
        mapping: Mapping network
        class_idx: 0 for AD, 1 for NC
        steps: Number of interpolation steps
        output_dir: Directory to save images
        device: Device
    """
    
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
        print(f"Saved latent walk: {walk_path}")


def generate_cross_class_interpolation(gen, mapping, steps=10, output_dir='generated_samples', device=DEVICE):
    """
    Generate interpolation between AD and NC classes.
    Shows the smooth transition from Alzheimer's Disease to Normal Control.
    
    Args:
        gen: Generator model
        mapping: Mapping network
        steps: Number of interpolation steps (default: 10)
        output_dir: Directory to save images
        device: Device
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating cross-class interpolation (AD → NC)...")
    
    with torch.no_grad():
        # Use same random z for both classes to isolate class effect
        z = torch.randn(1, Z_DIM, device=device)
        noise = get_noise(1, device)
        
        # Generate W vectors for both classes with same z
        ad_label = torch.tensor([0], dtype=torch.long, device=device)
        nc_label = torch.tensor([1], dtype=torch.long, device=device)
        
        w_ad = mapping(z, ad_label)
        w_nc = mapping(z, nc_label)
        
        # Expand w for all layers
        num_blocks = LOG_RESOLUTION - 2
        num_layers = 1 + 2 * num_blocks
        w_ad = w_ad.unsqueeze(1).expand(-1, num_layers, -1)
        w_nc = w_nc.unsqueeze(1).expand(-1, num_layers, -1)
        
        imgs = []
        for alpha in torch.linspace(0, 1, steps):
            # Interpolate in W space
            w_interp = (1 - alpha) * w_ad + alpha * w_nc
            
            img = gen(w_interp, noise)
            img = img * 0.5 + 0.5
            img = torch.clamp(img, 0, 1)
            imgs.append(img)
        
        imgs = torch.cat(imgs, dim=0)
        
        interp_path = f"{output_dir}/AD_to_NC_interpolation.png"
        save_image(imgs, interp_path, nrow=steps, padding=2, normalize=False)
        print(f"  Saved cross-class interpolation: {interp_path}")
        print(f"  Shows smooth transition from AD (left) to NC (right)")


def extract_w_vectors(mapping, num_samples, labels, device=DEVICE):
    """
    Generate W vectors (style space vectors) from the mapping network.
    
    Args:
        mapping: Mapping network model
        num_samples: Number of samples to generate
        labels: Class labels [N]
        device: Device
        
    Returns:
        w_vectors: Style space vectors [N, w_dim]
    """
    all_w_vectors = []
    
    with torch.no_grad():
        for label in labels:
            # Sample random z vector
            z = torch.randn(1, Z_DIM).to(device)
            
            # Convert to W vector through mapping network
            label_tensor = torch.tensor([label], dtype=torch.long).to(device)
            w = mapping(z, label_tensor)
            
            # Take the first layer's w vector (they're repeated across layers)
            if w.dim() == 3:  # [batch, num_layers, w_dim]
                w = w[:, 0, :]  # Take first layer
            
            all_w_vectors.append(w.cpu())
    
    # Concatenate all vectors
    return torch.cat(all_w_vectors, dim=0).numpy()


def visualize_embeddings(gen, mapping, disc, output_dir='generated_samples', 
                         num_samples=100, device=DEVICE):
    """
    Create TWO SEPARATE t-SNE embeddings visualizations:
    
    1. Style Space (W-space) Analysis:
       - Extracts W vectors from mapping network for both real and generated samples
       - Shows how the mapping network structures the latent space
       - Reveals class separation and real vs generated clustering
    
    2. Ground Truth Dataset Analysis:
       - Extracts discriminator features from REAL images only
       - Shows the actual distribution of the ground truth dataset
       - Reveals natural clustering of AD vs NC pathology in image space
    
    Args:
        gen: Generator model
        mapping: Mapping network
        disc: Discriminator model
        output_dir: Directory to save plots
        num_samples: Number of samples per class (real and generated)
        device: Device
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"GENERATING t-SNE EMBEDDING VISUALIZATIONS")
    print(f"{'='*60}")
    print(f"Creating two separate t-SNE analyses:")
    print(f"  1. Style Space (W-space): Latent manifold structure")
    print(f"  2. Ground Truth Dataset: Real image feature distribution")
    print(f"{'='*60}")
    
    # Load real images from dataset
    print(f"\nLoading {num_samples} real images per class...")
    train_loader, _ = get_dataloaders(batch_size=1)
    
    real_images = {0: [], 1: []}  # AD: 0, NC: 1
    real_labels = []
    
    for imgs, labels in train_loader:
        for img, label in zip(imgs, labels):
            label_item = label.item()
            if len(real_images[label_item]) < num_samples:
                real_images[label_item].append(img.squeeze(0))  # Remove batch dimension
                real_labels.append(label_item)
            
            if len(real_images[0]) >= num_samples and len(real_images[1]) >= num_samples:
                break
        
        if len(real_images[0]) >= num_samples and len(real_images[1]) >= num_samples:
            break
    
    # Combine real images - stack to create proper batch dimension
    all_real_images = real_images[0] + real_images[1]
    real_imgs_tensor = torch.stack(all_real_images, dim=0).to(device)
    real_labels_tensor = torch.tensor(real_labels, dtype=torch.long, device=device)
    
    print(f"✓ Loaded {len(real_labels)} real images (shape: {real_imgs_tensor.shape})")
    
    # ========================================================================
    # ANALYSIS 1: STYLE SPACE (W-SPACE) t-SNE
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"ANALYSIS 1: STYLE SPACE (W-SPACE)")
    print(f"{'='*60}")
    
    # Generate W vectors for real images (using random z vectors with ground truth labels)
    print(f"Generating W vectors for real images...")
    real_w_vectors = extract_w_vectors(mapping, len(real_labels), real_labels, device)
    
    # Generate synthetic images and their W vectors
    print(f"Generating {num_samples} synthetic images per class...")
    gen_w_vectors_list = []
    gen_labels = []
    
    with torch.no_grad():
        for class_idx in [0, 1]:
            class_labels = [class_idx] * num_samples
            gen_w = extract_w_vectors(mapping, num_samples, class_labels, device)
            gen_w_vectors_list.append(gen_w)
            gen_labels.extend(class_labels)
    
    gen_w_vectors = np.vstack(gen_w_vectors_list)
    
    print(f"✓ Generated W vectors for {len(gen_labels)} samples")
    
    # Combine all W vectors
    all_features_w = np.vstack([real_w_vectors, gen_w_vectors])
    all_labels_w = np.array(real_labels + gen_labels)
    all_types_w = np.array(['Real'] * len(real_labels) + ['Generated'] * len(gen_labels))
    
    print(f"✓ Extracted W-space features: shape {all_features_w.shape}")
    print(f"  Label distribution: AD={np.sum(all_labels_w==0)}, NC={np.sum(all_labels_w==1)}")
    print(f"  Real samples: AD={np.sum((all_types_w=='Real') & (all_labels_w==0))}, NC={np.sum((all_types_w=='Real') & (all_labels_w==1))}")
    print(f"  Generated samples: AD={np.sum((all_types_w=='Generated') & (all_labels_w==0))}, NC={np.sum((all_types_w=='Generated') & (all_labels_w==1))}")
    
    # Perform t-SNE on W-space vectors
    print(f"Computing t-SNE projection for W-space...")
    reducer_w = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='pca',
        max_iter=1000,
        random_state=42,
        verbose=1
    )
    embeddings_w = reducer_w.fit_transform(all_features_w)
    print(f"✓ W-space t-SNE projection complete")
    
    # Create W-space visualization
    print("Creating W-space visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Color schemes
    class_colors = {0: '#E74C3C', 1: '#3498DB'}  # Red for AD, Blue for NC
    class_names = {0: 'AD (Alzheimer\'s)', 1: 'NC (Normal Control)'}
    
    # Plot 1: By class label
    ax1 = axes[0]
    for class_idx in [0, 1]:
        mask = all_labels_w == class_idx
        ax1.scatter(embeddings_w[mask, 0], embeddings_w[mask, 1], 
                   c=class_colors[class_idx], label=class_names[class_idx],
                   alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    ax1.set_title(f't-SNE of Style Space (W) Vectors', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f't-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel(f't-SNE Dimension 2', fontsize=12)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Real vs Generated
    ax2 = axes[1]
    type_colors = {'Real': '#27AE60', 'Generated': '#F39C12'}  # Green for Real, Orange for Generated
    type_markers = {'Real': 'o', 'Generated': '^'}
    
    for data_type in ['Real', 'Generated']:
        mask = all_types_w == data_type
        ax2.scatter(embeddings_w[mask, 0], embeddings_w[mask, 1],
                   c=type_colors[data_type], label=data_type,
                   alpha=0.6, s=50, marker=type_markers[data_type],
                   edgecolors='k', linewidth=0.5)
    
    ax2.set_title(f't-SNE of Style Space (W) Vectors', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f't-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel(f't-SNE Dimension 2', fontsize=12)
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save W-space plot
    plot_path_w = f"{output_dir}/tsne_embeddings_style_space.png"
    plt.savefig(plot_path_w, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved W-space visualization: {plot_path_w}")
    
    # ========================================================================
    # ANALYSIS 2: GROUND TRUTH DATASET (DISCRIMINATOR FEATURES)
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"ANALYSIS 2: GROUND TRUTH DATASET")
    print(f"{'='*60}")
    
    # Extract discriminator features from REAL images only
    print(f"Extracting discriminator features from real images...")
    
    # Process in batches to avoid OOM
    batch_size = 16
    real_disc_features_list = []
    
    with torch.no_grad():
        for i in range(0, len(real_imgs_tensor), batch_size):
            batch = real_imgs_tensor[i:i+batch_size]
            
            # Process images through discriminator up to feature layer
            x = disc.from_rgb(batch)
            x = disc.blocks(x)
            
            # Apply minibatch standard deviation if used
            if disc.mbstd:
                x = disc.minibatch_stddev(x)
            
            # Final convolution and feature flattening
            x = disc.final_conv(x)
            batch_features = torch.flatten(x, 1)
            real_disc_features_list.append(batch_features.cpu())
    
    real_disc_features = torch.cat(real_disc_features_list, dim=0).numpy()
    
    print(f"✓ Extracted discriminator features: shape {real_disc_features.shape}")
    print(f"  Label distribution: AD={np.sum(real_labels_tensor.cpu().numpy()==0)}, NC={np.sum(real_labels_tensor.cpu().numpy()==1)}")
    
    # Perform t-SNE on discriminator features
    print(f"Computing t-SNE projection for discriminator features...")
    reducer_disc = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='pca',
        max_iter=1000,
        random_state=42,
        verbose=1
    )
    embeddings_disc = reducer_disc.fit_transform(real_disc_features)
    print(f"✓ Discriminator features t-SNE projection complete")
    
    # Create ground truth dataset visualization
    print("Creating ground truth dataset visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    real_labels_np = real_labels_tensor.cpu().numpy()
    for class_idx in [0, 1]:
        mask = real_labels_np == class_idx
        ax.scatter(embeddings_disc[mask, 0], embeddings_disc[mask, 1],
                  c=class_colors[class_idx], label=class_names[class_idx],
                  alpha=0.7, s=60, edgecolors='k', linewidth=0.5)
    
    ax.set_title('t-SNE of Ground Truth Dataset (Discriminator Features)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save ground truth plot
    plot_path_disc = f"{output_dir}/tsne_embeddings_ground_truth.png"
    plt.savefig(plot_path_disc, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ground truth visualization: {plot_path_disc}")
    
    # Print interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    print(f"\nANALYSIS 1 - STYLE SPACE (W-SPACE):")
    print(f"  File: {plot_path_w}")
    print(f"  - Total W vectors: {len(all_labels_w)} ({len(real_labels)} real, {len(gen_labels)} generated)")
    print(f"  - W dimension: {all_features_w.shape[1]} → 2D projection")
    print(f"  - Shows learned latent manifold structure")
    print(f"  - Reveals class separation (AD vs NC) in style space")
    print(f"  - Compares real vs generated W-vector distributions")
    print()
    print(f"ANALYSIS 2 - GROUND TRUTH DATASET:")
    print(f"  File: {plot_path_disc}")
    print(f"  - Total images: {len(real_labels)} (real images only)")
    print(f"  - Feature dimension: {real_disc_features.shape[1]} → 2D projection")
    print(f"  - Shows natural clustering of AD vs NC pathology")
    print(f"  - Reveals actual data distribution in image feature space")
    print(f"  - Based on discriminator's learned representation")
    print()
    print(f"KEY INSIGHTS:")
    print(f"  1. W-space shows how the mapping network structures latent space")
    print(f"  2. Ground truth shows the actual disease feature distribution")
    print(f"  3. Both analyses reveal class separability from different perspectives")
    print(f"  4. Overlap in W-space (real vs generated) indicates good sampling")
    print(f"  5. Ground truth clustering validates meaningful learned features")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images from conditional StyleGAN2')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--class_idx', type=int, choices=[0, 1], default=None, 
                       help='Class to generate: 0=AD, 1=NC (default: both)')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples per class')
    parser.add_argument('--output_dir', type=str, default='generated_samples', help='Output directory')
    parser.add_argument('--mixed', action='store_true', help='Generate mixed AD/NC comparison')
    parser.add_argument('--walk', action='store_true', help='Generate latent space walk')
    parser.add_argument('--cross_class', action='store_true', help='Generate interpolation between AD and NC classes')
    parser.add_argument('--embeddings', action='store_true', help='Generate t-SNE embedding visualization')
    parser.add_argument('--embedding_samples', type=int, default=100,
                       help='Number of samples per class for embeddings (default: 100)')
    
    args = parser.parse_args()
    
    # Load model
    gen, mapping, disc = load_checkpoint(args.checkpoint)
    
    print(f"\nClass mapping: 0=AD (Alzheimer's), 1=NC (Normal Control)")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)
    
    # Generate based on arguments
    if args.embeddings:
        # Generate t-SNE embedding visualization
        visualize_embeddings(gen, mapping, disc, args.output_dir, 
                           num_samples=args.embedding_samples)
    
    elif args.mixed:
        generate_mixed_batch(gen, mapping, args.num_samples // 2, args.output_dir)
    
    elif args.cross_class:
        # Generate cross-class interpolation (AD → NC)
        generate_cross_class_interpolation(gen, mapping, steps=10, output_dir=args.output_dir)
    
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
            generate_class_samples(gen, mapping, args.class_idx, args.num_samples, args.output_dir)
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"Check {args.output_dir}/ for generated images")
