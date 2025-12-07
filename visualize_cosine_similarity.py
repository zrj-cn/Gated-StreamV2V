import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity

# ===================== 手动配置参数（可根据需要修改） =====================
FRAME_NUM = 24       # 要分析的帧号
TARGET_X = 20        # 目标patch的x坐标
TARGET_Y = 20        # 目标patch的y坐标
HEIGHT = 64          # 特征图的空间高度
WIDTH = 114          # 特征图的空间宽度
TARGET_DIM = 0       # 要显示的维度（0-3，对应原4个维度）
# =========================================================================

def visualize_cosine_similarity(frame_num: int, target_x: int, target_y: int, height: int, width: int, target_dim: int):
    """
    Visualizes the cosine similarity map for a single specified dimension (only pure image with red dot, no other elements).

    Args:
        frame_num (int): The frame number to analyze.
        target_x (int): The x-coordinate of the target patch.
        target_y (int): The y-coordinate of the target patch.
        height (int): The spatial height of the feature map.
        width (int): The spatial width of the feature map.
        target_dim (int): The specific dimension to visualize (0-3).
    """
    # 检查维度合法性
    if target_dim not in [0, 1, 2, 3]:
        print(f"Error: Target dimension {target_dim} is invalid (must be 0-3). Exiting.")
        return

    base_path = "/home/zrj/project/ori_v2v/streamv2v/vid2vid/saved_states/TTT_soft_ori/self_attn_feats_SD"
    file_template = "up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor.frame{}.pt"

    file_path = os.path.join(base_path, file_template.format(frame_num))
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading hidden states for frame {frame_num} (dimension {target_dim})...")
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

    # 仅计算指定维度的相似度图
    hidden_states = all_hidden_states[target_dim]
    target_patch_index = target_y * width + target_x
    target_vector = hidden_states[target_patch_index].unsqueeze(0).numpy()
    similarity_scores = cosine_similarity(target_vector, hidden_states.numpy())[0]
    similarity_map = similarity_scores.reshape((height, width))

    # 创建画布，设置边距为0，无坐标轴
    fig, ax = plt.subplots(1, 1, figsize=(width/10, height/10), dpi=100)  # 按特征图尺寸比例创建画布
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 移除所有边距

    # 绘制相似度图
    ax.imshow(similarity_map, cmap='viridis', vmin=similarity_map.min(), vmax=similarity_map.max())
    
    # 绘制红色圆点标记目标patch
    ax.scatter(target_x, target_y, s=180, c='red', marker='o', edgecolor='white', linewidth=1)
    
    # 完全隐藏坐标轴（刻度、标签、边框）
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')  # 关闭所有坐标轴显示

    # 输出文件名包含维度信息
    output_path = f'cosine_similarity_dim{target_dim}_frame{frame_num}_patch({target_x},{target_y}).png'
    # 保存时去掉多余空白，设置bbox_inches='tight'和pad_inches=0
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    print(f"Pure similarity map saved to {output_path}")
    plt.close(fig)  # 关闭画布释放内存

if __name__ == '__main__':
    visualize_cosine_similarity(FRAME_NUM, TARGET_X, TARGET_Y, HEIGHT, WIDTH, TARGET_DIM)