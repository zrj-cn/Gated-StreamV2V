import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import cv2
import numpy as np
import csv
from torchvision.io import read_video
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.functional as F
from typing import Tuple, Dict, Optional


class VideoMotionAnalyzer:
    """视频运动强度分析工具类，用于计算视频运动强度并推荐噪声强度和缓存间隔参数"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化运动分析器
        
        参数:
            device: 默认为GPU
        """
        self.device = device
        # 加载预训练的RAFT光流模型
        local_weights_path = "/share/zrj/streamv2v/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
        self.raft_model = raft_large(weights=None, progress=False)
        state_dict = torch.load(local_weights_path, map_location="cpu")
        self.raft_model.load_state_dict(state_dict, strict=False)
        self.raft_model = self.raft_model.to(self.device)
        self.raft_model.eval()
        
        # 运动强度到参数的映射配置
        self.motion_config = {
            "low": {"noise_strength": 0.3, "cache_interval": 6},
            "medium": {"noise_strength": 0.5, "cache_interval": 4},
            "high": {"noise_strength": 0.7, "cache_interval": 2}
        }

    def _preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """预处理视频帧以适应光流模型输入"""
        # 转换为浮点型并归一化到[-1, 1]
        frames = frames.to(dtype=torch.float32) / 255.0
        frames = F.normalize(frames, mean=0.5, std=0.5)
        return frames.to(self.device)

    def calculate_motion_strength(self, video_path: str, sample_stride: int = 5) -> Tuple[float, np.ndarray]:
        """
        计算视频的整体运动强度（支持间隔采样以提速）
        
        参数:
            video_path: 视频文件路径
            sample_stride: 采样间隔，仅对相邻帧对进行子采样，例如5表示每5帧计算一次相邻帧对的光流
        
        返回:
            整体运动强度（采样帧对的平均光流幅值）
            每帧对的运动强度数组（采样后）
        """
        # 读取视频
        frames, _, _ = read_video(video_path)  # 形状: [T, H, W, C]
        frames = frames.permute(0, 3, 1, 2)   # 转换为 [T, C, H, W]
        
        if len(frames) < 2:
            raise ValueError("视频至少需要包含2帧才能计算运动强度")
        
        # 预处理
        frames = self._preprocess_frames(frames)
        T, C, H, W = frames.shape
        
        # 计算相邻帧对的光流
        motion_strengths = []
        for i in range(0, T - 1, max(1, sample_stride)):
            frame_pair = torch.stack([frames[i], frames[i+1]])  # [2, C, H, W]
            with torch.no_grad():
                flow_preds = self.raft_model(frame_pair[0:1], frame_pair[1:2])
                flow = flow_preds[-1]
            flow_magnitude = torch.norm(flow, dim=1).mean().item()
            motion_strengths.append(flow_magnitude)
        
        # 计算整体运动强度（平均所有帧对的运动强度）
        overall_strength = sum(motion_strengths) / len(motion_strengths)
        return overall_strength, np.array(motion_strengths)

    def get_recommended_params(self, motion_strength: float) -> Dict[str, float]:
        """
        根据运动强度推荐噪声强度和缓存间隔参数
        
        参数:
            motion_strength: 视频整体运动强度
        
        返回:
            包含推荐的noise_strength和cache_interval的字典
        """
        # 动态确定运动强度等级（阈值可根据实际数据调整）
        if motion_strength < 1.5:
            return self.motion_config["low"].copy()
        elif motion_strength < 3.0:
            return self.motion_config["medium"].copy()
        else:
            return self.motion_config["high"].copy()

    def analyze_video(self, video_path: str) -> Dict[str, any]:
        """
        完整分析视频运动并返回结果和推荐参数
        
        参数:
            video_path: 视频文件路径
        
        返回:
            分析结果字典，包含整体运动强度、每帧运动强度和推荐参数
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        overall_strength, frame_strengths = self.calculate_motion_strength(video_path)
        params = self.get_recommended_params(overall_strength)
        
        return {
            "video_path": video_path,
            "overall_motion_strength": overall_strength,
            "frame_pair_motion_strengths": frame_strengths,
            "recommended_params": params
        }

    def analyze_directory_to_csv(self, dir_path: str, output_csv: str, sample_stride: int = 5):
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"目录不存在: {dir_path}")
        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}
        files = [f for f in os.listdir(dir_path) if os.path.splitext(f)[1].lower() in exts]
        results = []
        for fname in files:
            path = os.path.join(dir_path, fname)
            try:
                strength, _ = self.calculate_motion_strength(path, sample_stride=sample_stride)
                results.append((fname, strength))
            except Exception:
                pass
        results.sort(key=lambda x: x[1], reverse=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            for name, strength in results:
                writer.writerow([name, f"{strength:.6f}"])
        return results


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析视频运动强度并推荐参数")
    parser.add_argument("--video_path", type=str, help="输入视频路径")
    parser.add_argument("--dir_path", type=str, help="输入视频目录")
    parser.add_argument("--output_csv", type=str, default=None, help="输出CSV路径")
    parser.add_argument("--sample_stride", type=int, default=5, help="采样间隔")
    args = parser.parse_args()
    
    analyzer = VideoMotionAnalyzer()
    
    try:
        if args.dir_path:
            out_csv = args.output_csv or os.path.join(args.dir_path, "motion_strengths.csv")
            analyzer.analyze_directory_to_csv(args.dir_path, out_csv, sample_stride=args.sample_stride)
            print(f"已保存CSV: {out_csv}")
        elif args.video_path:
            result = analyzer.analyze_video(args.video_path)
            print(f"视频路径: {result['video_path']}")
            print(f"整体运动强度: {result['overall_motion_strength']:.4f}")
            print("推荐参数:")
            print(f"  噪声强度 (noise_strength): {result['recommended_params']['noise_strength']}")
            print(f"  缓存间隔 (cache_interval): {result['recommended_params']['cache_interval']}")
        else:
            raise ValueError("请提供 --video_path 或 --dir_path")
    except Exception as e:
        print(f"分析失败: {str(e)}")