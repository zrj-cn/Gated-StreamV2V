import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import argparse

def visualize_spatial_pca(height: int, width: int, output_dir: str):
    """
    Loads attention hidden states, performs PCA on spatial features,
    and saves each frame's PCA visualization as a separate image to the specified directory.
    Args:
        height (int): The spatial height of the feature map.
        width (int): The spatial width of the feature map.
        output_dir (str): Directory to save individual PCA visualization frames.
    """
    src = "TTT_soft_ori"
    base_path = f"/home/zrj/project/ori_v2v/streamv2v/vid2vid/saved_states/{src}/self_attn_bank"
    # base_path = f"/home/zrj/project/ori_v2v/streamv2v/vid2vid/saved_states/{src}/self_attn_feats_SD/"
    file_template = "up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor.frame{}.pt"
    # file_template = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.frame{}.pt"
    frame_numbers = [20, 24, 28, 32]
    # frame_numbers = [15, 24, 36, 48]
    feature_name = "hidden_states_out"
    # feature_name = "update_delta"
    output_dir = "/home/zrj/project/ori_v2v/tools/feature/tttbank"
    # 自动创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir} (created if not exists)")

    print("Loading and processing hidden states for spatial PCA...")
    for frame_num in frame_numbers:
        file_path = os.path.join(base_path, file_template.format(frame_num))
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path} | Skip frame {frame_num}")
            continue
        
        # 加载数据并处理
        data = torch.load(file_path, map_location='cpu')
        hidden_states = data[feature_name]
        print(f"Frame {frame_num} hidden_states shape: {hidden_states.shape}")
        hidden_states = hidden_states[0]  # Shape: [seq_len, feature_dim]
        
        seq_len = hidden_states.shape[0]
        if seq_len != height * width:
            print(f"[Warning] Frame {frame_num}: Sequence length {seq_len} != {height}x{width} | Skip")
            continue

        # 执行PCA降维
        print(f"Frame {frame_num} running PCA (input shape: {hidden_states.shape})...")
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(hidden_states.numpy())  # Shape: [seq_len, 3]
        
        # 归一化到[0,1]范围（RGB像素值要求）
        pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0))
        
        # 重塑为空间维度
        pc_image = pca_features.reshape((height, width, 3))
        print(f"Frame {frame_num} PCA image shape: {pc_image.shape}")

        # 定义单帧图片命名规则：包含帧序号、尺寸、特征类型
        img_filename = f"spatial_pca_{feature_name}_frame{frame_num}_{height}x{width}.png"
        img_path = os.path.join(output_dir, img_filename)

        # 单独绘制并保存当前帧
        plt.figure(figsize=(width/10, height/10))  # 按特征图尺寸比例设置画布（10像素/单位）
        plt.imshow(pc_image)
        plt.title(f'Frame {frame_num} | PCA Components (RGB) | {height}x{width}', fontsize=10)
        plt.axis('off')  # 关闭坐标轴
        plt.tight_layout(pad=0)  # 去除边距，让图片填满画布
        plt.savefig(img_path, dpi=150, bbox_inches='tight')  # 保存图片，DPI可调整
        plt.close()  # 关闭画布释放内存

        print(f"[Success] Frame {frame_num} saved to: {img_path}")

    print("\nAll frames processed! Check output directory: {}".format(output_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize spatial PCA of attention hidden states (save individual frames).')
    # 默认参数 --height 64 --width 114
    parser.add_argument('--height', type=int, default=64, help='Spatial height of the feature map.')
    parser.add_argument('--width', type=int, default=114, help='Spatial width of the feature map.')
    # 新增：指定输出目录（默认./spatial_pca_frames）
    parser.add_argument('--output_dir', type=str, default='./spatial_pca_frames', 
                        help='Directory to save individual PCA visualization frames.')
    args = parser.parse_args()
    
    visualize_spatial_pca(args.height, args.width, args.output_dir)