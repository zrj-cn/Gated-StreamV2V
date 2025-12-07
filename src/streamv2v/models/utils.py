from collections import deque
from typing import Tuple, Callable

from numpy import dtype

from einops import rearrange
import torch
import torch.nn.functional as F
import math



# ==========================================
# 新增：TTT3R 核心逻辑 (TTT3R Core Logic)
# ==========================================

def compute_ttt_beta_new(feature_bank, feature_in, reverse_tag=False):
    """
    修正版：计算基于位置敏感的余弦相似度门控。
    逻辑：Compare bank[i] vs input[i] (Strict Spatial Alignment).
    """
    dtype = feature_bank.dtype
    
    # 1. 强制 L2 归一化 (计算 Cosine Sim 的基础)
    # q_norm, k_norm: (B, Heads, N, D)
    feature_bank_norm = F.normalize(feature_bank.to(torch.float32), p=2, dim=-1)
    feature_in_norm = F.normalize(feature_in.to(torch.float32), p=2, dim=-1)

    # 2. 【关键修改】逐位置点积 (Element-wise Dot Product)
    # 不要使用 torch.sum(dim=2) 或 einsum 聚合！
    # 直接相乘并在特征维度(dim=-1)求和，保留空间维度 N
    # sim: (B, Heads, N) —— 每个位置都有一个独立的相似度分数
    sim = torch.sum(feature_bank_norm * feature_in_norm, dim=-1)
    
    # 3. 缩放与偏移 (关键调节参数)
    # Temperature: 放大差异，让 Sigmoid 更敏感
    # Bias: 设定“变化”的阈值。
    # 例如 bias=0.8，意味着相似度 < 0.8 就被认为是“变了”
    temperature = 10.0
    bias = 0.8
    score = (sim - bias) * temperature
    
    # 4. 聚合多头 (Average over Heads)
    # 不同 Head 可能关注不同特征，取平均作为该位置的整体一致性
    # shape: (B, Heads, N) -> (B, 1, N)
    score = torch.mean(score, dim=1, keepdim=True)
    
    # 调整维度以适配 Feature Bank: (B, 1, N) -> (B, N, 1)
    score = score.transpose(1, 2)
    
    # 5. 计算 Beta (Sigmoid + Reverse)
    # 相似度高 (sim > bias) -> score > 0 -> sigmoid > 0.5 -> beta < 0.5 (保持)
    # 相似度低 (sim < bias) -> score < 0 -> sigmoid < 0.5 -> beta > 0.5 (更新)
    beta = torch.sigmoid(score)
    
    if reverse_tag:
        beta = 1.0 - beta
        # 增加一个微小的保底更新率，防止死锁
        beta = torch.clamp(beta, min=0.01)
        
    return beta.to(dtype)

def compute_ttt_beta_cos(feature_bank, feature_in, temperature=8.0, reverse_tag=True):
    """
    计算位置敏感的余弦相似度 Beta。
    对应你的例子：
    q_bank: (B, N, C) -> [[1,2], [0,1]]
    k_in:   (B, N, C) -> [[1,2], [4,-3]]
    """
    dtype = feature_bank.dtype
    
    # 1. 强制 L2 归一化 (计算 Cosine Sim 的基础)
    # q_norm: [[1/√5, 2/√5], [0, 1]]
    # k_norm: [[1/√5, 2/√5], [4/5, -3/5]]
    feature_bank_norm = F.normalize(feature_bank.to(torch.float32), p=2, dim=-1)
    feature_in_norm = F.normalize(feature_in.to(torch.float32), p=2, dim=-1)

    # 2. 逐位置点积 (Element-wise Dot Product)
    # 这一步直接得到每个位置的 Cosine Similarity
    # dim=-1 求和，保留空间维度 N
    # sim: [1.0, -0.6]
    sim = torch.sum(feature_bank_norm * feature_in_norm, dim=-1, keepdim=True) 
    
    # 3. 缩放 (Temperature)
    # score: [4.0, -2.4]
    score = sim * temperature
    
    # 4. Sigmoid
    # sigmoid_score: [0.982, 0.083]
    beta = torch.sigmoid(score)
    
    # 5. Beta 反转
    # beta: [0.018, 0.917]
    if reverse_tag:
        beta = 1.0 - beta
        # 建议加个极小值保底，防止彻底死锁
        # beta = torch.clamp(beta, min=0.01)
        
    return beta.to(dtype)


def compute_ttt_beta(q_bank, k_in, reverse_tag = False):
    # ... (获取维度信息)
    dtype = q_bank.dtype
    head_dim = q_bank.shape[-1]
    scale_factor = 1.0 / math.sqrt(head_dim)

    # === 优化开始 ===
    q_bank_f32 = q_bank.to(torch.float32)
    k_in_f32 = k_in.to(torch.float32)
    # 1. 先对 Input Key 在序列维度 (dim=2) 求和
    # k_in shape: (B, Heads, N_in, D) -> k_sum shape: (B, Heads, D)
    k_sum = torch.mean(k_in_f32, dim=2)
    # 再给平均的这个维度的数值乘上2
    k_sum = k_sum
    
    # 2. 计算 Query 与 聚合Key 的点积
    # q_bank: (B, Heads, N_bank, D)
    # k_sum:  (B, Heads, D)
    # 结果 match_score: (B, Heads, N_bank)
    # 使用 einsum 进行批量点积
    match_score = torch.einsum('b h i d, b h d -> b h i', q_bank_f32, k_sum)
    
    # print("match_score shape:", match_score.shape)
    # === 优化结束 ===

    match_score = match_score * scale_factor
    

    # 后续处理保持不变 (Mean over heads, Transpose, Sigmoid)
    match_score = torch.mean(match_score, dim=1, keepdim=True) 
    match_score = match_score.transpose(1, 2)
    
    # print("match_score shape2:", match_score.shape)

    beta = torch.sigmoid(match_score) 
    if reverse_tag:
        beta = 1.0 - beta
    
    return beta.to(dtype)

def apply_ttt_update(bank_state, update_delta, beta, lr=0.5):
    """
    执行 TTT 状态更新。
    公式近似: S_t = S_{t-1} + lr * beta * (Target - S_{t-1})
    
    Args:
        bank_state: 当前 Bank State (B, N, C)
        update_delta: 从输入帧检索并投影回来的更新目标 (B, N, C)
        beta: 自适应学习率 (B, N, 1)
        lr: 全局学习率
    """
    effective_lr = beta * lr
    # effective_lr < 0.2的部分, 值置为0 
    # effective_lr[effective_lr < 0.2] = 0.0
    # effective_lr[effective_lr > 0.3] = 0.0
    # 使用插值更新策略，防止数值爆炸，同时保留惯性
    # 如果 beta 高，更多地采纳 update_delta；如果 beta 低，保持原样
    new_state = (1 - effective_lr) * bank_state + effective_lr * update_delta
    # print("new_state shape:", new_state.shape)
    
    return new_state

def soft_feature_injection(x, bank_output, threshold=0.9, alpha=2.0):
    """
    改进点：在 Soft Retrieval 阶段加入 Hard Mask，防止低相似度的背景噪声污染特征。
    只有相似度 > threshold 的 Token 才有资格参与特征重构。
    """
    dtype = x.dtype
    # ... (前置代码不变)
    head_dim = x.shape[-1]
    
    # === Step 1: 归一化与相似度 ===
    x_norm = F.normalize(x, p=2, dim=-1)
    bank_norm = F.normalize(bank_output, p=2, dim=-1)
    similarity = torch.matmul(x_norm, bank_norm.transpose(1, 2))
    
    # === Step 2: 软检索 (带阈值过滤) ===
    scale_factor = 1.0 / math.sqrt(head_dim)
    scaled_sim = similarity / scale_factor
    
    # 【关键改进】: 过滤掉不达标的 Token
    # 创建一个 Mask，把相似度低于 threshold 的位置设为 -inf
    # 这样 Softmax 后它们的权重就是 0，完全不会污染结果
    # 注意：为了防止某一行全都被 mask 掉导致 NaN，需要处理全 -inf 的情况
    mask = similarity < threshold
    scaled_sim = scaled_sim.masked_fill(mask, -float('inf'))
    
    # 计算 Attention 权重
    attn = F.softmax(scaled_sim, dim=-1) 
    
    # 处理全被 Mask 的情况（即没有一个 Token 达标）
    # 如果某行全是 -inf，Softmax 结果是 NaN。
    # 我们将 NaN 替换为 0 (或者均匀分布，但这里 0 更安全，意味着不检索任何东西)
    # 实际上 Step 3 的 Gate 会拦截这种情况，所以 retrieved_feat 是多少无所谓，只要不是 NaN 即可。
    attn = torch.nan_to_num(attn, nan=0.0)
    
    # 重构特征
    # 现在 retrieved_feat 只包含相似度 > 0.9 的那些 Token 的混合
    retrieved_feat = torch.matmul(attn, bank_output)
    
    # === Step 3: 生成软掩码 (决策) ===
    # 逻辑保持不变，这部分负责“平滑过渡”
    max_sim, _ = torch.max(similarity, dim=-1, keepdim=True)

    invalid_mask = (max_sim < threshold).to(dtype) # 1 表示无效(没找到)，0 表示有效
    retrieved_feat = invalid_mask * x + (1 - invalid_mask) * retrieved_feat

    diff = threshold - max_sim
    std = torch.std(diff, dim=1, keepdim=True) + 1e-6
    adaptive_scale = alpha / std
    soft_mask = torch.sigmoid(diff * adaptive_scale)
    
    # === Step 4: 特征融合 ===
    out = soft_mask * x + (1 - soft_mask) * retrieved_feat
    
    return out

def get_nn_feats(x, y, threshold=0.9):
    if type(x) is deque:
        x = torch.cat(list(x), dim=1)
    if type(y) is deque:
        y = torch.cat(list(y), dim=1)

    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)

    cosine_similarity = torch.matmul(x_norm, y_norm.transpose(1, 2))

    max_cosine_values, nearest_neighbors_indices = torch.max(cosine_similarity, dim=-1)
    mask = max_cosine_values < threshold
    # 输出被threshold过滤的token数量
    # print(f"mask ratio: {torch.sum(mask)/x.shape[0]/x.shape[1]:.4f}")
    indices_expanded = nearest_neighbors_indices.unsqueeze(-1).expand(-1, -1, x_norm.size(-1))
    nearest_neighbor_tensor = torch.gather(y, 1, indices_expanded)
    selected_tensor = torch.where(mask.unsqueeze(-1), x, nearest_neighbor_tensor)

    return selected_tensor

def get_nn_latent(x, y, threshold=0.9):
    assert len(x.shape) == 4
    _, c, h, w = x.shape
    x_ = rearrange(x, 'n c h w -> n (h w) c')
    y_ = []
    for i in range(len(y)):
        y_.append(rearrange(y[i], 'n c h w -> n (h w) c'))
    y_ = torch.cat(y_, dim=1)
    x_norm = F.normalize(x_, p=2, dim=-1)
    y_norm = F.normalize(y_, p=2, dim=-1)

    cosine_similarity = torch.matmul(x_norm, y_norm.transpose(1, 2))

    max_cosine_values, nearest_neighbors_indices = torch.max(cosine_similarity, dim=-1)
    mask = max_cosine_values < threshold
    indices_expanded = nearest_neighbors_indices.unsqueeze(-1).expand(-1, -1, x_norm.size(-1))
    nearest_neighbor_tensor = torch.gather(y_, 1, indices_expanded)

    # Use values from x where the cosine similarity is below the threshold
    x_expanded = x_.expand_as(nearest_neighbor_tensor)
    selected_tensor = torch.where(mask.unsqueeze(-1), x_expanded, nearest_neighbor_tensor)

    selected_tensor = rearrange(selected_tensor, 'n (h w) c -> n c h w', h=h, w=w, c=c)

    return selected_tensor

def random_bipartite_soft_matching(
    metric: torch.Tensor, use_grid: bool = False, ratio: float = 0.5
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].
    This will reduce the number of tokens by a ratio of ratio/2.
    """
    with torch.no_grad():
        B, N, _ = metric.shape
        if use_grid:
            assert ratio == 0.5
            sample = torch.randint(2, size=(B, N//2, 1), device=metric.device)
            sample_alternate = 1 - sample
            grid = torch.arange(0, N, 2).view(1, N//2, 1).to(device=metric.device)
            grid = grid.repeat(4, 1, 1)
            rand_idx = torch.cat([sample + grid, sample_alternate + grid], dim = 1)
        else:
            rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)
        r = int(ratio * N)
        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]
        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge_kv_out(keys: torch.Tensor, values: torch.Tensor, outputs: torch.Tensor, mode="mean") -> torch.Tensor:
        src_keys, dst_keys = split(keys)
        C_keys = src_keys.shape[-1]
        dst_keys = dst_keys.scatter_reduce(-2, dst_idx.expand(B, r, C_keys), src_keys, reduce=mode)
        src_values, dst_values = split(values)
        C_values = src_values.shape[-1]
        dst_values = dst_values.scatter_reduce(-2, dst_idx.expand(B, r, C_values), src_values, reduce=mode)
        src_outputs, dst_outputs = split(outputs)
        C_outputs = src_outputs.shape[-1]
        dst_outputs = dst_outputs.scatter_reduce(-2, dst_idx.expand(B, r, C_outputs), src_outputs, reduce=mode)
        return dst_keys, dst_values, dst_outputs

    def merge_kv(keys: torch.Tensor, values: torch.Tensor, mode="mean") -> torch.Tensor:
        src_keys, dst_keys = split(keys)
        C_keys = src_keys.shape[-1]
        dst_keys = dst_keys.scatter_reduce(-2, dst_idx.expand(B, r, C_keys), src_keys, reduce=mode)
        src_values, dst_values = split(values)
        C_values = src_values.shape[-1]
        dst_values = dst_values.scatter_reduce(-2, dst_idx.expand(B, r, C_values), src_values, reduce=mode)
        return dst_keys, dst_values

    def merge_out(outputs: torch.Tensor, mode="mean") -> torch.Tensor:
        src_outputs, dst_outputs = split(outputs)
        C_outputs = src_outputs.shape[-1]
        dst_outputs = dst_outputs.scatter_reduce(-2, dst_idx.expand(B, r, C_outputs), src_outputs, reduce=mode)
        return dst_outputs
        
    return merge_kv_out, merge_kv, merge_out
