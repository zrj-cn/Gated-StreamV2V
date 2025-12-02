import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import argparse

def visualize_spatial_pca(height: int, width: int):
    """
    Loads attention hidden states, performs PCA on spatial features,
    and visualizes the first three principal components as an RGB image.
    Args:
        height (int): The spatial height of the feature map.
        width (int): The spatial width of the feature map.
    """
    base_path = "/home/zrj/project/ori_v2v/streamv2v/vid2vid/output/self_attn_feats_SD/"
    file_template = "up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor.frame{}.pt"
    frame_numbers = [0, 24, 48, 72]
    
    pca_results = {}

    print("Loading and processing hidden states for spatial PCA...")
    for frame_num in frame_numbers:
        file_path = os.path.join(base_path, file_template.format(frame_num))
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        data = torch.load(file_path, map_location='cpu')
        hidden_states = data['hidden_states']
        # 打印hidden_states的shape
        print(f"Frame {frame_num} hidden_states shape: {hidden_states.shape}")
        hidden_states = data['hidden_states'][0] # Shape: [seq_len, feature_dim]
        
        seq_len = hidden_states.shape[0]
        if seq_len != height * width:
            print(f"Frame {frame_num}: Sequence length {seq_len} does not match provided dimensions {height}x{width}. Skipping.")
            continue

        print(f"Frame {frame_num} hidden_states shape for PCA: {hidden_states.shape}")

        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(hidden_states.numpy()) # Shape: [seq_len, 3]
        
        pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0))
        
        pc_image = pca_features.reshape((height, width, 3))
        print(f"Frame {frame_num} PCA image shape: {pc_image.shape}")
        pca_results[frame_num] = pc_image

    if not pca_results:
        print("No frames were successfully processed for spatial PCA. Exiting.")
        return

    print("Generating spatial PCA visualization plot...")
    fig, axes = plt.subplots(1, len(pca_results), figsize=(5 * len(pca_results), 5 * (height/width)))
    if len(pca_results) == 1:
        axes = [axes]

    for i, (frame_num, pc_image) in enumerate(pca_results.items()):
        axes[i].imshow(pc_image)
        axes[i].set_title(f'Frame {frame_num} - Spatial PCA')
        axes[i].axis('off')

    plt.tight_layout()
    output_path = f'spatial_pca_visualization_{height}x{width}.png'
    plt.savefig(output_path)
    print(f"Spatial PCA visualization plot saved to {output_path}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize spatial PCA of attention hidden states.')
    # --height 64 --width 116
    parser.add_argument('--height', type=int, required=True, help='Spatial height of the feature map.')
    parser.add_argument('--width', type=int, required=True, help='Spatial width of the feature map.')
    args = parser.parse_args()
    
    visualize_spatial_pca(args.height, args.width)
