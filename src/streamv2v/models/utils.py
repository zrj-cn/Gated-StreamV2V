from collections import deque
from typing import Tuple, Callable

from numpy import dtype

from einops import rearrange
import torch
import torch.nn.functional as F
import math


def compute_beta_similarity(feature_bank, feature_in, reverse_tag=True):
    """
    Compute position-sensitive cosine-similarity gating.
    """
    dtype = feature_bank.dtype
    
    # 1. Enforce L2 normalization (basis for cosine similarity)
    # (B, Heads, N, D)
    feature_bank_norm = F.normalize(feature_bank.to(torch.float32), p=2, dim=-1)
    feature_in_norm = F.normalize(feature_in.to(torch.float32), p=2, dim=-1)

    # 2. Compute element-wise dot product (cosine similarity)
    # sim: (B, Heads, N) —— each position has an independent similarity score
    sim = torch.sum(feature_bank_norm * feature_in_norm, dim=-1)
    
    # 3. Scaling and shifting (key tuning parameters)
    # Temperature: amplifies differences to make Sigmoid more sensitive
    # Bias: sets the threshold for "change".
    # For example, bias=0.8 means similarity < 0.8 is considered "changed"
    temperature = 10.0
    bias = 0.8
    score = (sim - bias) * temperature
    
    # 4. Average over Heads
    # Different Heads may focus on different features, so we take the mean
    # shape: (B, Heads, N) -> (B, 1, N)
    score = torch.mean(score, dim=1, keepdim=True)
    
    # 调整维度以适配 Feature Bank: (B, 1, N) -> (B, N, 1)
    score = score.transpose(1, 2)
    
    # 5. Compute Beta (Sigmoid + Reverse)
    # High similarity (sim > bias) -> score > 0 -> sigmoid > 0.5 -> beta < 0.5 (keep)
    # Low similarity (sim < bias) -> score < 0 -> sigmoid < 0.5 -> beta > 0.5 (update)
    beta = torch.sigmoid(score)
    
    if reverse_tag:
        beta = 1.0 - beta
        # Add a small guarantee update rate to prevent deadlock
        beta = torch.clamp(beta, min=0.01)
        
    return beta.to(dtype)

def compute_beta_confidence(q_bank, k_in):

    dtype = q_bank.dtype
    head_dim = q_bank.shape[-1]
    scale_factor = 1.0 / math.sqrt(head_dim)

    q_bank_f32 = q_bank.to(torch.float32)
    k_in_f32 = k_in.to(torch.float32)
    # Use mean to avoid gradient explosion
    # k_in shape: (B, Heads, N_in, D) -> k_mean shape: (B, Heads, D)
    k_mean = torch.mean(k_in_f32, dim=2)
    
    # 2. Compute dot product between Query and aggregated Key
    # q_bank: (B, Heads, N_bank, D)
    # k_mean:  (B, Heads, D)
    # Result match_score: (B, Heads, N_bank)
    # Use einsum for batch dot product
    match_score = torch.einsum('b h i d, b h d -> b h i', q_bank_f32, k_mean)
    

    match_score = match_score * scale_factor
    
    match_score = torch.mean(match_score, dim=1, keepdim=True) 
    match_score = match_score.transpose(1, 2)

    beta = torch.sigmoid(match_score) 
    
    return beta.to(dtype)

def soft_feature_injection(x, bank_output, threshold=0.9, alpha=2.0):
    """
    Improvement: introduce a hard mask during soft retrieval to prevent low-similarity background noise from corrupting features.
    Only tokens whose similarity exceeds the threshold are allowed to participate in feature reconstruction.
    """
    dtype = x.dtype
    head_dim = x.shape[-1]
    
    # Normalization & Similarity 
    x_norm = F.normalize(x, p=2, dim=-1)
    bank_norm = F.normalize(bank_output, p=2, dim=-1)
    similarity = torch.matmul(x_norm, bank_norm.transpose(1, 2))
    
    # Soft retrieval (with threshold filtering) 
    scale_factor = 1.0 / math.sqrt(head_dim)
    scaled_sim = similarity / scale_factor
    
    # Filter out dissimilar tokens
    mask = similarity < threshold
    scaled_sim = scaled_sim.masked_fill(mask, -float('inf'))
    
    # Compute Attention Weights 
    attn = F.softmax(scaled_sim, dim=-1) 
    
    # Handle cases where no token exceeds the threshold
    # If all sims are -inf, softmax results in NaN, we replace NaN with 0.0 to ensure valid feature reconstruction.
    attn = torch.nan_to_num(attn, nan=0.0)
    
    # Now retrieved_feat only contains features of tokens with similarity > threshold
    retrieved_feat = torch.matmul(attn, bank_output)
    
    # Soft Mask 
    max_sim, _ = torch.max(similarity, dim=-1, keepdim=True)

    invalid_mask = (max_sim < threshold).to(dtype)
    retrieved_feat = invalid_mask * x + (1 - invalid_mask) * retrieved_feat

    diff = threshold - max_sim
    std = torch.std(diff, dim=1, keepdim=True) + 1e-6
    adaptive_scale = alpha / std
    soft_mask = torch.sigmoid(diff * adaptive_scale)
    
    # Feature Fusion 
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
