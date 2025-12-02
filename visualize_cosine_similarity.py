import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity

def visualize_cosine_similarity(frame_num: int, target_x: int, target_y: int, height: int, width: int):
    """
    Visualizes the cosine similarity maps for all 4 dimensions between a target patch and all other patches.

    Args:
        frame_num (int): The frame number to analyze.
        target_x (int): The x-coordinate of the target patch.
        target_y (int): The y-coordinate of the target patch.
        height (int): The spatial height of the feature map.
        width (int): The spatial width of the feature map.
    """
    base_path = "/home/zrj/project/ori_v2v/streamv2v/vid2vid/output/self_attn_feats_SD/"
    file_template = "up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor.frame{}.pt"

    file_path = os.path.join(base_path, file_template.format(frame_num))
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading hidden states for frame {frame_num}...")
    data = torch.load(file_path, map_location='cpu')
    all_hidden_states = data['hidden_states']  # Shape: [4, seq_len, feature_dim]

    if all_hidden_states.dim() != 3 or all_hidden_states.shape[0] != 4:
        print(f"Expected hidden_states to have shape [4, seq_len, feature_dim], but got {all_hidden_states.shape}. Exiting.")
        return

    seq_len, feature_dim = all_hidden_states.shape[1], all_hidden_states.shape[2]
    if seq_len != height * width:
        print(f"Sequence length {seq_len} does not match provided dimensions {height}x{width}. Exiting.")
        return

    if not (0 <= target_x < width and 0 <= target_y < height):
        print(f"Target coordinates ({target_x}, {target_y}) are out of bounds for dimensions ({width}, {height}). Exiting.")
        return

    # Create a 2x2 subplot to display the 4 similarity maps
    fig, axes = plt.subplots(2, 2, figsize=(12, 12 * (height / width) * 0.9))
    axes = axes.flatten() # Flatten for easy iteration

    dim_titles = [
        "Dim 0: Unconditional (A)",
        "Dim 1: Conditional (A)",
        "Dim 2: Unconditional (B)",
        "Dim 3: Conditional (B)",
    ]

    print(f"Calculating cosine similarity maps with the patch at ({target_x}, {target_y})...")
    
    # Find the min and max similarity scores across all maps for a consistent color scale
    all_sim_maps = []
    for i in range(4):
        hidden_states = all_hidden_states[i]
        target_patch_index = target_y * width + target_x
        target_vector = hidden_states[target_patch_index].unsqueeze(0).numpy()
        similarity_scores = cosine_similarity(target_vector, hidden_states.numpy())[0]
        similarity_map = similarity_scores.reshape((height, width))
        all_sim_maps.append(similarity_map)
    
    vmin = min(m.min() for m in all_sim_maps)
    vmax = max(m.max() for m in all_sim_maps)

    # Loop through each of the 4 dimensions to plot
    for i, (sim_map, ax) in enumerate(zip(all_sim_maps, axes)):
        im = ax.imshow(sim_map, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.scatter(target_x, target_y, s=100, c='red', marker='x')
        ax.set_title(dim_titles[i])
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')

    fig.suptitle(f'Cosine Similarity Maps for Frame {frame_num}\nReference Patch at ({target_x}, {target_y})', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    # Add a single, shared color bar for the entire figure
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.08)
    cbar.set_label('Cosine Similarity')

    output_path = f'cosine_similarity_4dims_frame{frame_num}_patch({target_x},{target_y}).png'
    plt.savefig(output_path)
    print(f"Combined similarity map saved to {output_path}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize cosine similarity of attention hidden states for all 4 dimensions.')
    parser.add_argument('--frame', type=int, default=24, help='The frame number to analyze.')
    parser.add_argument('--x', type=int, required=True, help='The x-coordinate of the target patch.')
    parser.add_argument('--y', type=int, required=True, help='The y-coordinate of the target patch.')
    parser.add_argument('--height', type=int, default=64, help='Spatial height of the feature map.')
    parser.add_argument('--width', type=int, default=114, help='Spatial width of the feature map.')
    args = parser.parse_args()
    
    visualize_cosine_similarity(args.frame, args.x, args.y, args.height, args.width)
