import os
import argparse
import cv2
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.io import read_video
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.functional as F
import torchvision.transforms as T


plt.rcParams["savefig.bbox"] = "tight"
# sphinx_gallery_thumbnail_number = 2
method_version = 'ori-eval'

def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)

def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = torch.nn.functional.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img

def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ

def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
        ]
    )
    batch = transforms(batch)
    return batch

def calculate_error(frame1, frame2, mask):
    frame1_norm = frame1 
    frame2_norm = frame2
    mask = mask.numpy().astype(np.uint8)
    
    pixels_to_consider = (mask == 0)  # We are interested in pixels where mask == 0
    # Calculate L1 for the selected pixels
    error = np.abs(frame1_norm - frame2_norm)[pixels_to_consider].mean()

    return error


def calculate_warp_error_video(model, ref_video_path, edit_video_path):

    ref_frames, _, _ = read_video(str(ref_video_path))
    ref_frames = ref_frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    edit_frames, _, _ = read_video(str(edit_video_path))
    edit_frames = edit_frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    ref_height, ref_width = ref_frames.shape[2], ref_frames.shape[3]
    edit_frames = torch.nn.functional.interpolate(edit_frames, size=(ref_height, ref_width), mode='bilinear', align_corners=False)

    num_frames = edit_frames.shape[0]
    ref_frames = ref_frames[:num_frames]
    error = []
    for i in range(num_frames - 1):
        fwd_batch = torch.stack([ref_frames[i], ref_frames[i+1]])
        bwd_batch = torch.stack([ref_frames[i+1], ref_frames[i]])
        fwd_batch = preprocess(fwd_batch).to(device)
        bwd_batch = preprocess(bwd_batch).to(device)
        list_of_flows = model(fwd_batch.to(device), bwd_batch.to(device))
        predicted_flows = list_of_flows[-1]
        h, w = predicted_flows.shape[2:]
        fwd_occ, bwd_occ = forward_backward_consistency_check(predicted_flows[:1], predicted_flows[1:])  # [1, H, W] float
        edit_image_1 = edit_frames[i]
        edit_image_2 = edit_frames[i+1]
        edit_image_1 = edit_image_1.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        edit_image_2 = edit_image_2.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((grid_x, grid_y), axis=2)
        flow = predicted_flows[1].permute(1,2,0).cpu().detach().numpy()
        warped_grid = (grid + flow).astype(np.float32)
        warped_image = cv2.remap(edit_image_1, warped_grid, None, cv2.INTER_LINEAR)
        # We actually only use the occlusion mask
        occlusion = fwd_occ[0].cpu().bool()
        warped_image[occlusion] = np.array([0,0,0], dtype=np.uint8)
        error.append(calculate_error(warped_image, edit_image_2, occlusion))

    avg_error = sum(error) / len(error)
    return avg_error


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate warp error for video evaluation')
    parser.add_argument('--cuda_visible_devices', type=str, default='7', help='CUDA visible devices.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--method_version', type=str, default='motion_strength_2', help='Method version to evaluate')
    parser.add_argument('--set_file_path', type=str, default='user_study_upload/eval_motion.json', help='Path to the JSON file containing evaluation data')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import torch
    
    # 使用命令行参数
    device = args.device
    method_version = args.method_version
    set_file_path = args.set_file_path
    # raft_path = "/share/zrj/streamv2v/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
    # model = raft_large(weights=raft_path, progress=False).to("cuda")

    local_weights_path = "/share/zrj/streamv2v/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
    # 初始化模型 → 加载权重 → 移到指定设备
    model = raft_large(weights=None, progress=False)
    state_dict = torch.load(local_weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    model = model.eval()

    # 使用命令行参数指定的文件路径
    with open(set_file_path, 'r') as file:
        json_data = json.load(file)

    # method_name = 'tokenflow'
    ref_video_dir = "source_video"
    edit_video_dir = f"output/{method_version}"

    video_error = []
    out_json = {}
    for item in tqdm(json_data):
        try:
            src_vid_name = item['src_vid_name']
            vid_name = item['vid_name']
            ref_video_path = os.path.join(ref_video_dir, f'{src_vid_name}.mp4')
            edit_video_path = os.path.join(edit_video_dir, f'{vid_name}.mp4')
            cur_video_error = calculate_warp_error_video(model, ref_video_path, edit_video_path)
            out_json[vid_name] = cur_video_error
            video_error.append(cur_video_error)
            print()
        except:
            print("pass")
    # 将json文件放在./warp_error_log目录下
    json.dump(out_json, open(f"warp_error_log/{method_version}.warperror", "w"), sort_keys=True, indent=4)
    print(f"Avg warp error of {method_version} is {sum(video_error)/len(video_error)}") 
    # 同时将平均误差写入json文件
    out_json['avg_error'] = sum(video_error)/len(video_error)
    json.dump(out_json, open(f"warp_error_log/{method_version}.warperror", "w"), sort_keys=True, indent=4)

