import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def visualize_generated_file(filepath, num_images=16):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at {filepath}")

    # Load and unnormalize
    samples = torch.load(filepath)
    if samples.max() > 1 or samples.min() < -1:
        print("Warning: image values outside expected range [-1, 1]. Display may be incorrect.")

    # Normalize from [-1, 1] to [0, 1] for display
    samples = samples * 0.5 + 0.5

    # Pick num_images randomly
    indices = torch.randperm(len(samples))[:num_images]
    sample_images = samples[indices]

    # Create grid
    grid = torchvision.utils.make_grid(sample_images, nrow=int(num_images**0.5))
    npimg = grid.numpy()
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Samples from {os.path.basename(filepath)}")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize images from a GAN-generated .pt file.")
    parser.add_argument("filepath", type=str, help="Path to gen_outputs_stage_*.pt")
    parser.add_argument("--num_images", type=int, default=16, help="Number of images to display")

    args = parser.parse_args()
    visualize_generated_file(args.filepath, args.num_images)
