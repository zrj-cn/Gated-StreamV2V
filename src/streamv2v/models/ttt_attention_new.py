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

from .utils import soft_feature_injection, compute_ttt_beta_new


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class TTTCachedSTXFormersAttnProcessor_cos:
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

    def _ttt_update_step(self, attn, input_hidden_states, scale=1.0):
            """
            TTT Update Step (Spatial / Pixel-wise Mode)
            
            Logic:
            1. Beta: Calculated via spatial cosine similarity between Bank and Input Features.
            (Low Sim -> High Beta -> Update)
            2. Target: Direct Input Features (Overwrite old content with new).
            """
            args = () if USE_PEFT_BACKEND else (scale,)
            
            if self.bank_state is not None:
                if self.vis:
                    print("bank is not none:", self.bank_state.shape)

                feature_bank = self.bank_state.unsqueeze(1)        # (B, 1, N, C)
                feature_in   = input_hidden_states.unsqueeze(1)    # (B, 1, N, C)
                
                if self.vis:
                    print("feature_bank shape:", feature_bank.shape)
                
                # 调用 utils 中的函数 (确保它是我们讨论过的逐像素 Cosine 版本)
                beta = compute_ttt_beta_new(
                    feature_bank, 
                    feature_in,
                    reverse_tag=self.reverse_tag # True: 不相似 -> Beta高 -> 更新
                )

                # === 2. 设定 Update Delta (更新目标) ===
                # 我们的目标是：检测到变化后，直接用当前帧的新内容覆盖旧内容。
                # 所以 Delta 就是当前帧的特征本身。
                update_delta = input_hidden_states.clone()

                if self.vis:
                    print("update_delta shape (from input):", update_delta.shape)

                # === 3. 执行更新 ===
                # S_new = (1 - lr*beta) * S_old + lr*beta * Input
                beta = beta * self.ttt_lr
                self.bank_state = (1 - beta) * self.bank_state + beta * update_delta
                
                # === 4. 生成可视化用的 Output ===
                hidden_states_out = attn.to_out[0](self.bank_state.clone(), *args)
                hidden_states_out = attn.to_out[1](hidden_states_out)

            else:
                # 初始化 Bank State
                if self.vis:
                    print("init bank state:", input_hidden_states.shape)
                self.bank_state = input_hidden_states.clone()
                return 
            
            # === 保存可视化数据 ===
            if self.save_attn_map and self.frame_id % 4 == 0:

                
                os.makedirs(f'./saved_states/TTT/self_attn_bank/', exist_ok=True)
                feats = {
                            "hidden_states": self.bank_state.clone().cpu(),     # 原始 Bank (Pre-proj)
                            "hidden_states_out": hidden_states_out.clone().cpu(), # 投影后 Bank (Post-proj)
                            "beta": beta.clone().cpu(),                         # 更新热力图
                            "update_delta": update_delta.clone().cpu(),         # 更新目标 (即 Input)
                        }
                torch.save(feats, f'./saved_states/TTT/self_attn_bank/{self.name}.frame{self.frame_id}.pt')
             
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

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # 暂存hidden states 用于后续的 states 更新
        cached_hidden_states = hidden_states.clone()

        if self.vis:
            print("cached_hidden_states shape:", cached_hidden_states.shape)
        
        # === 1. 生成 Query (当前帧) ===
        query = attn.to_q(hidden_states, *args)
        
        is_selfattn = False
        
        # 判断是否为自注意力层
        if encoder_hidden_states is None:
            is_selfattn = True
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        # 计算K和V
        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        if is_selfattn:
            if self.vis:
                print("自注意力self-attention")

            if self.bank_state is not None:

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
                os.makedirs(f'./saved_states/TTT/self_attn_feats_SD/', exist_ok=True)
                feats = {
                            "hidden_states": hidden_states.clone().cpu(),
                            # "query": query.clone().cpu(),
                            # "key": key.clone().cpu(),
                            # "value": value.clone().cpu(),
                        }
                torch.save(feats, f'./saved_states/TTT/self_attn_feats_SD/{self.name}.frame{self.frame_id}.pt')
        
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

        if is_selfattn and not self.use_attn_concat:
            hidden_states = hidden_states + cached_hidden_states

        if self.vis:
            print("hidden_states shape after batch_to_head_dim:", hidden_states.shape)
        # Output Projection
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
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

        # === 4. Feature Injection (从 Bank 注入) ===
        if is_selfattn and self.use_feature_injection and self.bank_state is not None:
            if "up_blocks.0" in self.name or "up_blocks.1" in self.name or 'mid_block' in self.name:
                b_state_reshaped = self.bank_state.clone()

                b_state_reshaped = attn.to_out[0](b_state_reshaped, *args)
                b_state_reshaped = attn.to_out[1](b_state_reshaped)
                # 准备 Bank State 的形状以匹配 Hidden States
                if input_ndim == 4:
                    b_state_reshaped = b_state_reshaped.transpose(1, 2).reshape(batch_size, channel, height, width)
                
                # 使用 Soft Injection
                hidden_states = soft_feature_injection(
                    hidden_states, 
                    b_state_reshaped, 
                    threshold=self.threshold,
                    # temperature=self.ttt_temperature
                )

        # === 5. 更新 Bank State (Update Step) ===
        if is_selfattn and (self.frame_id % self.interval == 0):
            self._ttt_update_step(attn, cached_hidden_states)
            # self._ttt_update_step_opt(attn, cached_hidden_states, None, None)
        self.frame_id += 1
        if self.vis:
            print("frame_id:", self.frame_id)
            exit(-1)
        return hidden_states