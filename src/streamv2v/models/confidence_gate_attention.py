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

from .utils import soft_feature_injection, compute_beta_confidence


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class ConfidenceGateCachedSTXFormersAttnProcessor:

    def __init__(self, attention_op: Optional[Callable] = None, name=None, 
                 use_feature_injection=True, 
                 feature_similarity_threshold=0.98,
                 interval=1,  
                 ttt_lr=1.0, 
                 save_attn_map=False, 
                 vis=False, 
                 use_concat=True): 
        """
        Args:
            attention_op: Optional xFormers attention operation; None defaults to auto-selection.
            name: Identifier for this processor, used in logging / saving.
            use_feature_injection: Whether to use feature fusion.
            threshold: threshold for soft injection (only active when injection is enabled).
            interval: Update the bank state every N frames.
            ttt_lr: Scaling factor for the confidence-weighted update (beta).
            save_attn_map: If True, periodically save attention maps and intermediate tensors to disk.
            vis: Enable verbose printing and visualization hooks. Only one round of visualization output will be produced when this is used.
            use_concat: In self-attention, concatenate current k/v with bank k/v (True) or replace them (False).
        """
        self.attention_op = attention_op
        self.name = name
        self.use_feature_injection = use_feature_injection
        self.threshold = feature_similarity_threshold
        self.frame_id = 0
        self.interval = interval
        
        self.ttt_lr = ttt_lr
        
        self.bank_state = None
        self.save_attn_map = save_attn_map
    
        self.use_attn_concat = use_concat
        self.vis = vis

    def _cached_update_step(self, attn, input_hidden_states, cached_key, cached_value, scale=1.0):
        """
        Update Step
        """
        beta = torch.tensor([])
        update_delta = torch.tensor([])
        args = () if USE_PEFT_BACKEND else (scale,)
        if self.bank_state is not None:
            if self.vis:
                print("bank is not none:", self.bank_state.shape)

            q_bank = attn.to_q(self.bank_state, *args)
            k_in = cached_key.clone()
            v_in = cached_value.clone()

            q_bank = attn.head_to_batch_dim(q_bank)
            k_in = attn.head_to_batch_dim(k_in)
            v_in = attn.head_to_batch_dim(v_in)

            # 2. 计算 Update Delta (检索新信息)

            update_attn_out = xformers.ops.memory_efficient_attention(
                q_bank, k_in, v_in, op=self.attention_op, scale=attn.scale
            )

            update_attn_out = update_attn_out.to(q_bank.dtype)
            update_attn_out = attn.batch_to_head_dim(update_attn_out)

            if self.vis:
                print("update_attn_out shape:", update_attn_out.shape)

            update_delta = update_attn_out

            # 3. 计算 Beta (更新置信度)
            batch_size = input_hidden_states.shape[0]
            q_bank_reshaped = q_bank.view(batch_size, attn.heads, -1, q_bank.shape[-1])
            k_in_reshaped = k_in.view(batch_size, attn.heads, -1, k_in.shape[-1])
            
            if self.vis:
                print("q_bank_reshaped shape:", q_bank_reshaped.shape)
                print("k_in_reshaped shape:", k_in_reshaped.shape)
            
            beta = compute_beta_confidence(
                q_bank_reshaped, 
                k_in_reshaped
            )

            beta = beta * self.ttt_lr
            self.bank_state = (1 - beta) * self.bank_state + beta * update_delta
            
            hidden_states_out = attn.to_out[0](self.bank_state.clone(), *args)
            hidden_states_out = attn.to_out[1](hidden_states_out)

        else:
            # 初始化 Bank State
            if self.vis:
                print("init bank state:", input_hidden_states.shape)
            self.bank_state = input_hidden_states.clone()
            return 
        
        if self.save_attn_map and self.frame_id % 4 == 0:
            os.makedirs(f'./saved_states/confidence_gate/self_attn_bank/', exist_ok=True)
            feats = {
                        "hidden_states": self.bank_state.clone().cpu(),
                        "hidden_states_out": hidden_states_out.clone().cpu(),
                        "beta": beta.clone().cpu(),
                        "update_delta": update_delta.clone().cpu(),
                    }
            torch.save(feats, f'./saved_states/confidence_gate/self_attn_bank/{self.name}.frame{self.frame_id}.pt')
        

             
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:

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
        
        cached_hidden_states = hidden_states.clone()

        if self.vis:
            print("cached_hidden_states shape:", cached_hidden_states.shape)
        
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
                print("self-attention")
            cached_key = key.clone()
            cached_value = value.clone()

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
                os.makedirs(f'./saved_states/confidence_gate/self_attn_feats_SD/', exist_ok=True)
                feats = {
                            "hidden_states": hidden_states.clone().cpu(),
                            # "query": query.clone().cpu(),
                            # "key": key.clone().cpu(),
                            # "value": value.clone().cpu(),
                        }
                torch.save(feats, f'./saved_states/confidence_gate/self_attn_feats_SD/{self.name}.frame{self.frame_id}.pt')
        
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

        # === Feature Injection  ===
        if is_selfattn and self.use_feature_injection and self.bank_state is not None:
            if "up_blocks.0" in self.name or "up_blocks.1" in self.name or 'mid_block' in self.name:
                b_state_reshaped = self.bank_state.clone()

                b_state_reshaped = attn.to_out[0](b_state_reshaped, *args)
                b_state_reshaped = attn.to_out[1](b_state_reshaped)

                if input_ndim == 4:
                    b_state_reshaped = b_state_reshaped.transpose(1, 2).reshape(batch_size, channel, height, width)
                
                # Soft Injection
                hidden_states = soft_feature_injection(
                    hidden_states, 
                    b_state_reshaped, 
                    threshold=self.threshold,
                )

        # === Update Bank State ===
        if is_selfattn and (self.frame_id % self.interval == 0):
            self._cached_update_step(attn, cached_hidden_states, cached_key, cached_value)

        self.frame_id += 1
        if self.vis:
            print("frame_id:", self.frame_id)
            exit(-1)
        return hidden_states