from collections import deque
from typing import Tuple, Callable

from einops import rearrange
import torch
import torch.nn.functional as F
import math



# ==========================================
# 新增：TTT3R 核心逻辑 (TTT3R Core Logic)
# ==========================================
def compute_ttt_beta_optimized(q_bank, k_in, scale=1.0, reverse_tag=False):
    """
    计算更新门控 Beta。
    逻辑：如果 Bank 中的某个 Token 在 Current Frame (k_in) 中找到了非常相似的匹配，
    说明这个 Token 是显著的/被激活的，我们应该更新它（或者保持它，取决于策略）。
    
    Args:
        q_bank: (B, Heads, N_bank, D)
        k_in:   (B, Heads, N_in, D)
    """
    dtype = q_bank.dtype
    # 1. 计算完整的 Attention Score (不先求和)
    # (B, Heads, N_bank, D) @ (B, Heads, D, N_in) -> (B, Heads, N_bank, N_in)
    sim_matrix = torch.matmul(q_bank.to(torch.float32), k_in.transpose(-1, -2).to(torch.float32)) * scale
    
    # 2. 获取每个 Bank Token 在当前帧中的最大匹配度
    # 含义：Bank Token i 在当前帧里找到的最相似物体的相似度是多少？
    # shape: (B, Heads, N_bank)
    max_sim, _ = torch.max(sim_matrix, dim=-1)
    
    # 3. 计算 Beta (Sigmoid 归一化)
    # 这里的 temperature 可以控制门控的锐度，建议可调
    beta = torch.sigmoid(max_sim) 
    
    # 4. 聚合 Heads (取平均) -> (B, N_bank, 1)
    beta = torch.mean(beta, dim=1, keepdim=True).transpose(1, 2)
    
    if reverse_tag:
        beta = 1.0 - beta
        
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
    改进的特征注入函数 (Scheme A + Scaled Dot-Product)。
    
    1. Retrieval: 使用 Scaled Cosine Similarity (Sim / (1/sqrt(d))) 替代固定温度，
       让检索清晰度自适应特征维度。
    2. Gating: 使用基于标准差的自适应缩放 (Adaptive Scaling via Std)，
       让融合边界的锐度自适应当前画面的匹配分布。
    
    Args:
        x: 当前帧特征 (B, N_x, C)
        bank_output: Feature Bank 特征 (B, N_bank, C)
        threshold: 相似度阈值 (默认 0.9)
        alpha: 自适应缩放的基准系数 (建议 1.0 ~ 3.0，控制 Mask 的整体锐度)
    """
    # 1. 兼容 deque 输入
    if isinstance(bank_output, deque):
        bank_output = torch.cat(list(bank_output), dim=1)
        
    # 获取维度信息
    head_dim = x.shape[-1]
    
    # === Step 1: 归一化与相似度 (Cosine Similarity) ===
    x_norm = F.normalize(x, p=2, dim=-1)
    bank_norm = F.normalize(bank_output, p=2, dim=-1)

    # similarity range: [-1, 1]
    similarity = torch.matmul(x_norm, bank_norm.transpose(1, 2))
    
    # === Step 2: 软检索 (Soft Retrieval) - "回忆" ===
    # 使用标准的 Attention Scaling 思想
    # 缩放因子: scale = 1 / sqrt(d)
    # 对于归一化后的点积(余弦)，我们需要 '除以' scale (即乘以 sqrt(d)) 来放大数值，
    # 防止 Softmax 过于平坦导致特征变糊。
    # 例如：d=64, sqrt(d)=8. Sim范围[-1,1] -> Logits范围[-8,8]。
    scale_factor = 1.0 / math.sqrt(head_dim)
    
    # 注意这里是除法，相当于 logits = sim * sqrt(d)
    attn = F.softmax(similarity / scale_factor, dim=-1) 
    
    # 重构特征
    retrieved_feat = torch.matmul(attn, bank_output)
    
    # === Step 3: 生成软掩码 (Soft Mask) - "决策" (方案 A) ===
    # 3.1 获取最佳匹配分数
    max_sim, _ = torch.max(similarity, dim=-1, keepdim=True) # (B, N_x, 1)
    
    # 3.2 计算差异 (diff > 0 表示不匹配/新内容)
    diff = threshold - max_sim
    
    # 3.3 计算当前帧内差异的标准差 (反映了匹配的不确定性范围)
    # 加上 eps 防止除以 0（例如纯色背景导致 std=0）
    std = torch.std(diff, dim=1, keepdim=True) + 1e-6
    
    # 3.4 动态计算 Sigmoid 的缩放系数
    # 如果分布很散(std大)，adaptive_scale 变小，过渡带变宽(平滑)
    # 如果分布很挤(std小)，adaptive_scale 变大，过渡带变窄(锐利)
    adaptive_scale = alpha / std
    
    # 3.5 计算掩码
    # 当 diff > 0 (不匹配) -> sigmoid > 0.5 -> mask 趋向 1 -> 保留 x
    # 当 diff < 0 (匹配)   -> sigmoid < 0.5 -> mask 趋向 0 -> 使用 retrieved_feat
    soft_mask = torch.sigmoid(diff * adaptive_scale)
    
    # === Step 4: 特征融合 (Blending) ===
    out = soft_mask * x + (1 - soft_mask) * retrieved_feat
    
    return out

# ==========================================
# 原有函数 (保持兼容性)
# ==========================================

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
    # print('mask ratio', torch.sum(mask)/x.shape[0]/x.shape[1])
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
