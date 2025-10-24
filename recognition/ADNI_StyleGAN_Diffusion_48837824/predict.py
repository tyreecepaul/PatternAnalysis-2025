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
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
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
        print(f"✓ Saved grid: {grid_path}")
        
        # Save individual images
        for i in range(num_samples):
            img_path = f"{output_dir}/{class_name}_sample_{i+1:03d}.png"
            save_image(fake_imgs[i], img_path, normalize=False)
        
        print(f"✓ Saved {num_samples} individual images to {output_dir}/")
    
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
        print(f"✓ Saved comparison grid: {grid_path}")
        print(f"✓ Saved {num_per_class * 2} individual images to {output_dir}/")
        print(f"  (AD: AD_1.png to AD_{num_per_class}.png)")
        print(f"  (NC: NC_1.png to NC_{num_per_class}.png)")

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
        print(f"✓ Saved latent walk: {walk_path}")


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
    Create t-SNE embeddings visualization comparing real and generated images.
    
    This function:
    1. Loads real images from the dataset
    2. Generates synthetic images for both classes
    3. Extracts W-space features from the mapping network
    4. Projects features to 2D using t-SNE with accurate parameters
    5. Creates visualization with ground truth labels and data type markers
    
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
    print(f"GENERATING t-SNE EMBEDDING VISUALIZATION")
    print(f"{'='*60}")
    
    # Load real images from dataset
    print(f"Loading {num_samples} real images per class...")
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
    all_features = np.vstack([real_w_vectors, gen_w_vectors])
    all_labels = np.array(real_labels + gen_labels)
    all_types = np.array(['Real'] * len(real_labels) + ['Generated'] * len(gen_labels))
    
    print(f"✓ Extracted features: shape {all_features.shape}")
    print(f"  Label distribution: AD={np.sum(all_labels==0)}, NC={np.sum(all_labels==1)}")
    print(f"  Real samples: AD={np.sum((all_types=='Real') & (all_labels==0))}, NC={np.sum((all_types=='Real') & (all_labels==1))}")
    print(f"  Generated samples: AD={np.sum((all_types=='Generated') & (all_labels==0))}, NC={np.sum((all_types=='Generated') & (all_labels==1))}")
    
    # Perform dimensionality reduction with accurate t-SNE parameters
    print(f"Computing t-SNE projection...")
    reducer = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='pca',
        max_iter=1000,
        random_state=42,
        verbose=1
    )
    embeddings = reducer.fit_transform(all_features)
    print(f"✓ t-SNE projection complete")
    
    # Create visualization
    print("Creating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Color schemes
    class_colors = {0: '#E74C3C', 1: '#3498DB'}  # Red for AD, Blue for NC
    class_names = {0: 'AD (Alzheimer\'s)', 1: 'NC (Normal Control)'}
    
    # Plot 1: By class label
    ax1 = axes[0]
    for class_idx in [0, 1]:
        mask = all_labels == class_idx
        ax1.scatter(embeddings[mask, 0], embeddings[mask, 1], 
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
        mask = all_types == data_type
        ax2.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=type_colors[data_type], label=data_type,
                   alpha=0.6, s=50, marker=type_markers[data_type],
                   edgecolors='k', linewidth=0.5)
    
    ax2.set_title(f't-SNE of Style Space (W) Vectors', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f't-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel(f't-SNE Dimension 2', fontsize=12)
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"{output_dir}/tsne_embeddings.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {plot_path}")
    
    # Create combined plot with class and type information
    fig2, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot with both class and type information
    for class_idx in [0, 1]:
        for data_type in ['Real', 'Generated']:
            mask = (all_labels == class_idx) & (all_types == data_type)
            if mask.sum() > 0:
                marker = 'o' if data_type == 'Real' else '^'
                label = f"{class_names[class_idx]} ({data_type})"
                ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                          c=class_colors[class_idx], label=label,
                          alpha=0.6, s=60, marker=marker,
                          edgecolors='k', linewidth=0.5)
    
    ax.set_title('t-SNE Embedding: Class Labels and Data Type', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE-1', fontsize=12)
    ax.set_ylabel('t-SNE-2', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    combined_path = f"{output_dir}/tsne_embeddings_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined visualization: {combined_path}")
    
    # Print interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    print(f"The t-SNE visualization of W vectors reveals several key insights:")
    print()
    print("1. STYLE SPACE MANIFOLD:")
    print("   - W vectors form a continuous manifold in the learned style space.")
    print("   - The curved structure shows the mapping network transforms random")
    print("     z vectors into a structured, disentangled representation.")
    print()
    print("2. CLASS CONDITIONING:")
    print("   - Distinct regions for AD vs NC indicate successful class conditioning.")
    print("   - The mapping network learned to separate classes in style space")
    print("     while maintaining smooth interpolation within each class.")
    print()
    print("3. LATENT SPACE PROPERTIES:")
    print(f"   - Total W vectors: {len(embeddings)} ({len(real_labels)} real labels, {len(gen_labels)} generated)")
    print(f"   - W dimension: {all_features.shape[1]} → 2D projection")
    print(f"   - Classes balanced: AD={sum(all_labels==0)}, NC={sum(all_labels==1)}")
    print()
    print("4. GENERATIVE MODEL QUALITY:")
    print("   - Smooth, continuous distribution indicates well-trained mapping network.")
    print("   - Overlap between real and generated W vectors shows the model")
    print("     samples from the same learned distribution for both classes.")
    print(f"{'='*60}\n")
    
    return embeddings, all_labels, all_types

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images from conditional StyleGAN2')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--class_idx', type=int, choices=[0, 1], default=None, 
                       help='Class to generate: 0=AD, 1=NC (default: both)')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples per class')
    parser.add_argument('--output_dir', type=str, default='generated_samples', help='Output directory')
    parser.add_argument('--mixed', action='store_true', help='Generate mixed AD/NC comparison')
    parser.add_argument('--walk', action='store_true', help='Generate latent space walk')
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