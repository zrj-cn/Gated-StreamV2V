from importlib import import_module
from typing import Callable, Optional, Union
from collections import deque
import os
import time
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer

from .utils import soft_feature_injection, compute_ttt_beta, apply_ttt_update, compute_ttt_beta_cos


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class TTTCachedSTXFormersAttnProcessor:
    r"""
    加入门控实现利用TTT特性

    """

    def __init__(self, attention_op: Optional[Callable] = None, name=None, 
                 use_feature_injection=True, 
                 feature_similarity_threshold=0.98,
                 interval=1,  # 每 interval 帧更新一次state
                 ttt_lr=0.5, # 更新state时对beta的缩放系数
                 ttt_temperature=10.0, # 不一定有用
                 save_attn_map=False, # 是否保存attn map
                 reverse_tag=False, # 是否翻转beta
                 vis=False, # 是否进行可视化
                 use_concat=True): # 计算自注意力时是否进行concat
        '''
        参数介绍
        '''
        self.attention_op = attention_op
        self.name = name
        self.use_feature_injection = use_feature_injection
        self.threshold = feature_similarity_threshold
        self.frame_id = 0
        self.interval = interval
        
        self.ttt_lr = ttt_lr
        self.ttt_temperature = ttt_temperature
        
        self.reverse_tag = reverse_tag
        # 只维护一个 hidden state，存储原始特征 (B, N, C)
        self.bank_state = None
        self.save_attn_map = save_attn_map
    
        self.use_attn_concat = use_concat
        self.vis = vis

    def _ttt_update_step(self, attn, input_hidden_states, cached_key, cached_value, scale=1.0):
        """
        TTT Update Step: S_t = S_{t-1} - beta * grad(S_{t-1}, X_t)
        这里利用 Attention(Q=Bank, K=In, V=In) 作为检索到的目标信息。
        """
        beta = torch.tensor([])
        update_delta = torch.tensor([])
        args = () if USE_PEFT_BACKEND else (scale,)
        if self.bank_state is not None:
            if self.vis:
                print("bank is not none:", self.bank_state.shape)
            # start_time = time.time()
            # 1. 准备 Q, K, V
            # Q 来自 Bank State (Historical Memory)，作为查询去 Input 中找对应信息
            # K, V 来自 Input Frame (New Observation)
            # 注意：input_hidden_states 应该是 Norm 后的特征
            q_bank = attn.to_q(self.bank_state, *args)
            k_in = cached_key.clone()
            v_in = cached_value.clone()

            q_bank = attn.head_to_batch_dim(q_bank)
            k_in = attn.head_to_batch_dim(k_in)
            v_in = attn.head_to_batch_dim(v_in)

            # 2. 计算 Update Delta (检索新信息)
            # Attention(Q=Bank, K=In, V=In)
            # 这一步计算出 Bank 希望从当前帧吸收什么信息
            update_attn_out = xformers.ops.memory_efficient_attention(
                q_bank, k_in, v_in, op=self.attention_op, scale=attn.scale
            )
            update_attn_out = update_attn_out.to(q_bank.dtype)
            update_attn_out = attn.batch_to_head_dim(update_attn_out)
            if self.vis:
                print("update_attn_out shape:", update_attn_out.shape)

            
            # Output Projection (回到 Hidden State 空间)
            update_delta = attn.to_out[0](update_attn_out, *args)
            update_delta = attn.to_out[1](update_delta)

            # 3. 计算 Beta (更新置信度)
            # 需要计算 Q_bank 和 K_in 的相似度
            # reshape 回 (Batch, Heads, SeqLen, Dim) 以供 utils 计算
            batch_size = input_hidden_states.shape[0]
            q_bank_reshaped = q_bank.view(batch_size, attn.heads, -1, q_bank.shape[-1])
            k_in_reshaped = k_in.view(batch_size, attn.heads, -1, k_in.shape[-1])
            
            if self.vis:
                print("q_bank_reshaped shape:", q_bank_reshaped.shape)
                print("k_in_reshaped shape:", k_in_reshaped.shape)


            # print("q_bank_reshaped shape:", q_bank_reshaped.shape)
            # print("k_in_reshaped shape:", k_in_reshaped.shape)
            
            beta = compute_ttt_beta(
                q_bank_reshaped, 
                k_in_reshaped,
                reverse_tag=self.reverse_tag
            )

            # beta = compute_ttt_beta_cos(
            #     self.bank_state, 
            #     input_hidden_states,
            #     reverse_tag=self.reverse_tag
            # )
            
            # 4. 执行更新
            self.bank_state = apply_ttt_update(
                self.bank_state,
                update_delta, 
                beta,
                lr=self.ttt_lr
            )
            
        else:
            # 初始化 Bank State
            if self.vis:
                print("init bank state:", input_hidden_states.shape)
            self.bank_state = input_hidden_states.clone()
            # self.bank_state = self.bank_state[:,::4,:]
        
        if self.save_attn_map and self.frame_id % 4 == 0 and self.frame_id <= 48:
            key = attn.to_k(self.bank_state, *args)
            value = attn.to_v(self.bank_state, *args)
            query = attn.to_q(self.bank_state, *args)
            os.makedirs(f'./output/TTT/self_attn_bank/', exist_ok=True)
            feats = {
                        "hidden_states": self.bank_state.clone().cpu(),
                        # "key": key.clone().cpu(),
                        # "value": value.clone().cpu(),
                        # "query": query.clone().cpu(),
                        "beta": beta.clone().cpu(),
                        "update_delta": update_delta.clone().cpu(),
                    }
            torch.save(feats, f'./output/TTT/self_attn_bank/{self.name}.frame{self.frame_id}.pt')
        
    def _ttt_update_step_opt(self, attn, input_hidden_states, cached_key, cached_value, scale=1.0):
        if self.bank_state is None:
            # 初始化：通常第一帧直接作为 Bank
            self.bank_state = input_hidden_states.clone()
            return

        # === 准备数据 ===
        # Q 来自 Bank (Old Memory)
        q_bank = attn.to_q(self.bank_state) # 注意：Diffusers通常to_q有bias吗？检查一下，这里假设没有或不影响
        q_bank = attn.head_to_batch_dim(q_bank) # (B*H, N, D)
        
        # K 来自 Current Frame (New Observation)
        # cached_key 已经是 head_to_batch_dim 过的吗？
        # 在你的 __call__ 里：cached_key = key.clone() -> 还没 head_to_batch_dim
        # 所以这里需要处理维度
        args = () 
        
        # 重新生成当前帧的 K (为了确保维度对齐，建议重新计算或正确reshape)
        # 你的 cached_key 是 (B, N, C*H)，需要 reshape 成 (B, H, N, D)
        batch_size = input_hidden_states.shape[0]
        head_dim = q_bank.shape[-1]
        heads = attn.heads
        
        # Reshape Q_bank: (B, Heads, N, D)
        q_bank_reshaped = q_bank.view(batch_size, heads, -1, head_dim)
        
        # 重新计算当前帧的 K (直接用 cached_hidden_states 计算最稳妥)
        k_in = attn.to_k(input_hidden_states)
        k_in_reshaped = k_in.view(batch_size, heads, -1, head_dim)

        # === 1. 计算自适应学习率 Beta ===
        # 使用优化后的计算方式
        beta = compute_ttt_beta_cos(
            q_bank_reshaped, 
            k_in_reshaped, 
            scale=attn.scale,
            reverse_tag=self.reverse_tag
        ) # Output: (B, N, 1)

        # beta = compute_ttt_beta_cos(
        #     self.bank_state, 
        #     input_hidden_states, 
        #     reverse_tag=self.reverse_tag
        # ) # Output: (B, N, 1)

        # === 2. 执行更新 (Fix Space Mismatch) ===
        # 目标：将 input_hidden_states (当前帧特征) 融合进 bank_state
        # Update Target 是 Current Frame Features
        update_target = input_hidden_states
        
        effective_lr = beta
        
        # Update Rule: S_new = (1 - lr) * S_old + lr * S_new
        # 这样保证了空间一致性，因为 bank_state 和 update_target 都在 Pre-Attention 空间
        self.bank_state = (1 - effective_lr) * self.bank_state + effective_lr * update_target
        
        # Debug / Vis
        if self.vis and self.frame_id % 4 == 0:
            print(f"Frame {self.frame_id}: Beta Mean {beta.mean().item():.4f}, Max {beta.max().item():.4f}")

        if self.save_attn_map and self.frame_id % 4 == 0 and self.frame_id <= 48:
            print(f"Frame {self.frame_id}: Beta Mean {beta.mean().item():.4f}, Max {beta.max().item():.4f},Min {beta.min().item():.4f}, Name {self.name}")
            key = attn.to_k(self.bank_state, *args)
            value = attn.to_v(self.bank_state, *args)
            query = attn.to_q(self.bank_state, *args)
            os.makedirs(f'./output/TTT/self_attn_bank/', exist_ok=True)
            feats = {
                        "hidden_states": self.bank_state.clone().cpu(),
                        # "key": key.clone().cpu(),
                        # "value": value.clone().cpu(),
                        # "query": query.clone().cpu(),
                        "beta": beta.clone().cpu(),
                    }
            torch.save(feats, f'./output/TTT/self_attn_bank/{self.name}.frame{self.frame_id}.pt')
             
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        # 开始计时
        # start_time = time.time()

        if attn.residual_connection:
            residual = hidden_states.clone()

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        # === Norm 处理 ===
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # [Crucial]: 暂存 Norm 后的 hidden_states (Before Projection)
        # 用于后续的 TTT 更新步，因为更新步需要用它生成 K, V
        cached_hidden_states = hidden_states.clone()

        if self.vis:
            print("cached_hidden_states shape:", cached_hidden_states.shape)
        # === 1. 生成 Query (当前帧) ===
        query = attn.to_q(hidden_states, *args)
        
        is_selfattn = False
        
        if encoder_hidden_states is None:
            is_selfattn = True
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        cached_key = key.clone()
        cached_value = value.clone()

        if is_selfattn:
            if self.vis:
                print("自注意力self-attention")
            cached_key = key.clone()
            cached_value = value.clone()

            if self.bank_state is None:
                # 对于第一帧
                # self.bank_state = cached_hidden_states.clone()
                key = key
                value = value
            else:
                # self.bank_state = self.bank_state[:,::4,:]
                # 对于后续帧
                # self.bank_state = uniform_sample_sequence_dim(self.bank_state, 1574)
                if self.use_attn_concat == True:
                    # 对于concat方案
                    key = torch.cat([key, attn.to_k(self.bank_state, *args)], dim=1)
                    value = torch.cat([value, attn.to_v(self.bank_state, *args)], dim=1)
                else:
                    # 直接与历史特征cross attention的方案
                    key = attn.to_k(self.bank_state, *args)
                    value = attn.to_v(self.bank_state, *args)

            
            if self.save_attn_map and self.frame_id % 4 == 0:
                # 如果需要保存self-attention特征
                os.makedirs(f'./output/TTT/self_attn_feats_SD/', exist_ok=True)
                feats = {
                            "hidden_states": hidden_states.clone().cpu(),
                            # "query": query.clone().cpu(),
                            # "key": key.clone().cpu(),
                            # "value": value.clone().cpu(),
                        }
                torch.save(feats, f'./output/TTT/self_attn_feats_SD/{self.name}.frame{self.frame_id}.pt')
        
        # === 3. 计算 Attention ===
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        if self.vis:
            print("hidden_states shape:", hidden_states.shape)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        if self.vis:
            print("hidden_states shape after batch_to_head_dim:", hidden_states.shape)
        # Output Projection
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout层
        hidden_states = attn.to_out[1](hidden_states)
        
        if self.vis:
            print("hidden_states shape after dropout:", hidden_states.shape)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if self.vis:
            print("hidden_states shape after residual connection:", hidden_states.shape)
            print("use_feature_injection:", self.use_feature_injection)

        if is_selfattn and self.use_attn_concat == False and self.bank_state is not None:
            print("NO use_attn_concat:", self.use_attn_concat)
            hidden_states = hidden_states + cached_hidden_states.clone()

        # === 4. Feature Injection (从 Bank 注入) ===
        if is_selfattn and self.use_feature_injection and self.bank_state is not None:
            if "up_blocks.0" in self.name or "up_blocks.1" in self.name or 'mid_block' in self.name:
                b_state_reshaped = self.bank_state.clone()
                # 准备 Bank State 的形状以匹配 Hidden States
                if input_ndim == 4:
                    b_state_reshaped = b_state_reshaped.transpose(1, 2).reshape(batch_size, channel, height, width)
                # else:
                #     b_state_reshaped = b_state_reshaped.transpose(1, 2)
                
                # 使用 Soft Injection
                hidden_states = soft_feature_injection(
                    hidden_states, 
                    b_state_reshaped, 
                    threshold=self.threshold,
                    # temperature=self.ttt_temperature
                )

        # === 5. 更新 Bank State (Update Step) ===
        # 使用暂存的 cached_hidden_states
        if is_selfattn and (self.frame_id % self.interval == 0):
            self._ttt_update_step(attn, cached_hidden_states, cached_key, cached_value)
            # self._ttt_update_step_opt(attn, cached_hidden_states, None, None)
        self.frame_id += 1
        if self.vis:
            print("frame_id:", self.frame_id)
            exit(-1)
        return hidden_states