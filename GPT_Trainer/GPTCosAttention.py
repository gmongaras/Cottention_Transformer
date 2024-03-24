from typing import Optional, Tuple
import math
import torch
from torch import nn
import os
import wandb
from transformers.models.gptj.modeling_gptj import create_sinusoidal_positions, get_embed_positions, apply_rotary_pos_emb
from transformers.file_utils import is_torch_fx_proxy
from typing import Union





import torch
import sys
sys.path.append("Cuda_Kernel")
from Custom_Kernel import CustomAttention



class Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        ctx.Q = Q
        ctx.K = K
        ctx.V = V
        return Q @ (K.transpose(-1, -2) @ V)

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V = ctx.Q, ctx.K, ctx.V
        grad_output = grad_output.to(Q.dtype)
        grad_Q = grad_output @ (V.transpose(-1, -2) @ K)
        grad_K = V @ (grad_output.transpose(-1, -2) @ Q)
        grad_V = Q @ (grad_output.transpose(-1, -2) @ K)
        return grad_Q, grad_K, grad_V


class GPTCosAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = config.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)
        
        
        # Learnable constant for each head for norm
        self.norm_const = nn.Parameter(0.5*torch.ones(1, self.num_attention_heads, 1, 1, dtype=self.q_proj.weight.dtype)).to(self.q_proj.weight.device)

    def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        if rotary:
            return tensor
        if len(tensor.shape) == 5:
            return tensor.permute(0, 1, 3, 2, 4)  # (batch, blocks, head, block_length, head_features)
        elif len(tensor.shape) == 4:
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
    
    def _cos_attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        # # Keep the attention weights computation in fp32 to avoid overflow issues
        # query = query.to(torch.float32)
        # key = key.to(torch.float32)

        
        # Normalize query, and keys
        query = torch.nn.functional.normalize(query, dim=-1, p=2)
        key = torch.nn.functional.normalize(key, dim=-1, p=2)
        
        attention_mask = (attention_mask == 0)
        
        # Scale the values by the length of the sequence
        value = value / (((causal_mask * attention_mask)).sum(-1, keepdims=True)**self.norm_const.sigmoid()).clamp(min=1)
        
        
        # Mask query, key, and value layers
        if attention_mask is not None:
            query = query * attention_mask.transpose(-1, -2)
            key = key * attention_mask.transpose(-1, -2)
            value = value * attention_mask.transpose(-1, -2)
        
        
        
        # if query.shape[-2] > self.head_dim:
        if True:
            #### Custom Attention ####
            # attn_output = query @ (key.transpose(-1, -2) @ value)
            attn_output = CustomAttention.apply(query, key, value)
            # attn_output = Multiply.apply(query, key, value)
            #### Custom Attention ####
            
            # attn_output = query @ (key.transpose(-1, -2) @ value)
        else:
            #### Normal Attention ####
            attn_weights = torch.matmul(query, key.transpose(-1, -2))

            # mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            # mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            # Mask with zeros for the causal mask
            attn_weights = torch.where(causal_mask, attn_weights, 0)

            # attn_weights = attn_weights / self.scale_attn

            # if attention_mask is not None:
            #     # Apply the attention mask
            #     attn_weights = attn_weights * (attention_mask==0)

            # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = attn_weights.to(value.dtype)
            # attn_weights = self.attn_dropout(attn_weights)

            # Mask heads if we want to
            if head_mask is not None:
                attn_weights = attn_weights * head_mask

            attn_output = torch.matmul(attn_weights, value)
            #### Normal Attention ####
            
        

        return attn_output, attn_output

    def _get_embed_positions(self, position_ids):
        embed_positions = self.embed_positions
        if embed_positions.device != position_ids.device:
            embed_positions = embed_positions.to(position_ids.device)
            self.embed_positions = embed_positions
        return embed_positions.repeat(position_ids.shape[0], 1, 1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)

        if is_torch_fx_proxy(position_ids) or torch.jit.is_tracing():
            # The logic to conditionally copy to GPU could not be traced, so we do this
            # every time in the torch.fx case
            embed_positions = get_embed_positions(self.embed_positions, position_ids)
        else:
            embed_positions = self._get_embed_positions(position_ids)

        repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        sincos = torch.gather(embed_positions, 1, repeated_position_ids)
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            key = apply_rotary_pos_emb(key, sin, cos)
            query = apply_rotary_pos_emb(query, sin, cos)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            # Note that this cast is quite ugly, but is not implemented before ROPE as the original codebase keeps the key in float32 all along the computation.
            # Reference: https://github.com/kingoflolz/mesh-transformer-jax/blob/f8315e3003033b23f21d78361b288953064e0e76/mesh_transformer/layers.py#L128
            present = (key.to(hidden_states.dtype), value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        # attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        
        # compute cosine self-attention: V x Sim(QK^T)
        attn_output, attn_weights = self._cos_attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


