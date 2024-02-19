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
        
        # Scale the values by the length of the sequence
        value = value / (((causal_mask * (attention_mask==0))).sum(-1).unsqueeze(-1)**self.norm_const.sigmoid()).clamp(min=1)
        
        
        # Mask query, key, and value layers
        if attention_mask is not None:
            query = query * (attention_mask == 0).transpose(-1, -2)
            key = key * (attention_mask == 0).transpose(-1, -2)
            value = value * (attention_mask == 0).transpose(-1, -2)
        
        
        
        # #### Custom Attention ####
        # attn_output = CustomAttention.apply(query, key, value)
        # #### Custom Attention ####
        
        
        
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
        
        
        
        """tests - don't delete
        #### Custom Attention ####
        Q1 = query.detach().clone().requires_grad_(True)
        K1 = key.detach().clone().requires_grad_(True)
        V1 = value.detach().clone().requires_grad_(True)
        attn_output = CustomAttention.apply(Q1, K1, V1)
        #### Custom Attention ####
        
        b = torch.autograd.gradcheck(CustomAttention.apply, (Q1, K1, V1))
        
        
        
        #### Normal Attention ####
        Q2 = query.detach().clone().requires_grad_(True)
        K2 = key.detach().clone().requires_grad_(True)
        V2 = value.detach().clone().requires_grad_(True)
        attn_weights = torch.matmul(Q2, K2.transpose(-1, -2))
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
        attn_output_ = torch.matmul(attn_weights, V2)
        #### Normal Attention ####
        
        
        # Gradient of attn_output wrt to Q, K, V
        grad_Q = torch.autograd.grad(attn_output, Q1, attn_output, retain_graph=True)[0]
        grad_K = torch.autograd.grad(attn_output, K1, attn_output, retain_graph=True)[0]
        grad_V = torch.autograd.grad(attn_output, V1, attn_output, retain_graph=True)[0]
        
        # Gradient of attn_output_ wrt to Q, K, V
        grad_Q_ = torch.autograd.grad(attn_output_, Q2, attn_output_, retain_graph=True)[0]
        grad_K_ = torch.autograd.grad(attn_output_, K2, attn_output_, retain_graph=True)[0]
        grad_V_ = torch.autograd.grad(attn_output_, V2, attn_output_, retain_graph=True)[0]
        """
        
        

        return attn_output, attn_weights

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






class GPTCosAttention_(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Learnable constant for each head
        # self.relu_constant = nn.Parameter(torch.zeros(1, self.num_attention_heads, 1, 1, dtype=self.query.weight.dtype, device=self.query.weight.device))
        
        # Learnable constant for each head for norm
        self.norm_const = nn.Parameter(0.5*torch.ones(1, self.num_attention_heads, 1, 1, dtype=self.query.weight.dtype)).to(self.query.weight.device)
        # init between 0.1 and 0.9
        # self.norm_const = nn.Parameter(torch.rand(1, self.num_attention_heads, 1, 1, dtype=self.query.weight.dtype, device=self.query.weight.device)*0.8+0.1)
        # Between -1 and 1
        # self.norm_const = nn.Parameter(torch.rand(1, self.num_attention_heads, 1, 1, dtype=self.query.weight.dtype, device=self.query.weight.device)*2-1)
        
        # self.norm = nn.LayerNorm(self.attention_head_size)
        
        # self.ff = nn.Sequential(nn.Linear(self.all_head_size, self.num_attention_heads), nn.Sigmoid())
        # self.ff = nn.Sequential(nn.Linear(self.attention_head_size, 1), nn.Sigmoid())
        
        # self.relu = SoftenedReLU()
        
        # self.token_dropout = Token_Dropout(p=config.attention_probs_dropout_prob)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        ### Cosine Similarity
        
        # Mask query, key, and value layers
        if attention_mask is not None:
            attention_mask = attention_mask == 0
            query_layer = query_layer * attention_mask.transpose(-1, -2)
            key_layer = key_layer * attention_mask.transpose(-1, -2)
            value_layer = value_layer * attention_mask.transpose(-1, -2)
        
        
        # Normalize query, and keys
        query_layer = torch.nn.functional.normalize(query_layer, dim=-1, p=2)
        key_layer = torch.nn.functional.normalize(key_layer, dim=-1, p=2)
        
        # Scale the values
        # value_layer = value_layer / (attention_mask.sum(-1).unsqueeze(-1)**(value_layer.sigmoid())).clamp(min=1)
        # value_layer = value_layer / (attention_mask.sum(-1)**self.ff(hidden_states)).clamp(min=1).transpose(-1, -2).unsqueeze(-1)
        # value_layer = value_layer / (attention_mask.sum(-1)**self.ff(hidden_states)).clamp(min=1).transpose(-1, -2).unsqueeze(-1)
        # value_layer = value_layer / (attention_mask.sum(-1).unsqueeze(-1)**self.ff(value_layer)).clamp(min=1)
        value_layer = value_layer / (attention_mask.sum(-1).unsqueeze(-1)**self.norm_const.sigmoid()).clamp(min=1)
        # value_layer = value_layer / self.all_head_size**0.5
        
        # # Apply dropout to the queries and keys
        # query_layer = self.token_dropout(query_layer)
        # key_layer = self.token_dropout(key_layer)
        
        
        # # Project the query, key, and value layers
        # query_layer = (self.q_proj_global(query_layer) * attention_mask.transpose(-1, -2)) + torch.nn.functional.normalize(query_layer, dim=-1, p=2)
        # key_layer = (self.k_proj_global(key_layer) * attention_mask.transpose(-1, -2)) + torch.nn.functional.normalize(key_layer, dim=-1, p=2)
        # value_layer = (self.v_proj_global(value_layer) * attention_mask.transpose(-1, -2))
        
        # query_layer = torch.nn.functional.normalize(query_layer, dim=-1, p=2)
        # key_layer = torch.nn.functional.normalize(key_layer, dim=-1, p=2)
        
        
        # If dimensionality is larger than sequence length, then we are doing
        # S^2 by (QK^T)V
        if query_layer.shape[-1] > query_layer.shape[-2]:
            # # attention_probs = ((torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer))) / (attention_mask.sum(-1).unsqueeze(-1))**self.norm_const.sigmoid()
            # attention_probs = ((torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer)))
            
            # # Matplotlib attention heatmap
            # import matplotlib.pyplot as plt
            # probs = attention_probs[0].detach().cpu().numpy()
            # for head in range(probs.shape[0]):
            #     # Shape is (num_heads, seq_len, seq_len)
            #     plt.imshow(probs[head], vmin=-1, vmax=1)
            #     plt.show()
            #     if not os.path.exists("imgs"):
            #         os.makedirs("imgs")
            #     plt.savefig(f"imgs/attention{head}.png")
            # context_layer = torch.matmul(attention_probs, value_layer)
            
            
            # # Implemented as einsum:
            # context_layer = torch.einsum(
            #     "nhse,nhqe,nhqw->nhsw", 
            #     query_layer, key_layer, value_layer
            # )
            
            
            # More effiicent implementation:
            context_layer = torch.einsum("nhsq,nhqw->nhsw", torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer), value_layer)
            # attn = torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer)
            # for blk in self.a_proj_global:
            #     attn = blk(attn) * attention_mask.transpose(-1, -2)
            # context_layer = torch.einsum("nhsq,nhqw->nhsw", 
            #                              attn,
            #                              value_layer)
            
        # Otherwise, we are doing d^2 Q(K^TV)
        else:
            # attention_probs = torch.matmul(
            #     key_layer.transpose(-1, -2),
            #     value_layer
            # )
            
            # context_layer = torch.matmul(
            #     query_layer, 
            #     attention_probs
            # )
            
            # # Implemented as einsum:
            # context_layer = torch.einsum(
            #     "nhse,nhqe,nhqw->nhsw", 
            #     query_layer, key_layer, value_layer
            # )
            
            # More effiicent implementation:
            context_layer = torch.einsum("nhsw,nhwe->nhse", query_layer, torch.einsum("nhse,nhsw->nhew", key_layer, value_layer))
        
        # Scale outputs then normalize
        # context_layer = self.norm(context_layer / attention_mask.sum(-1).unsqueeze(-1))
        # context_layer = (context_layer * self.ff(hidden_states).transpose(-1, -2).unsqueeze(-1))
        
        # context_layer = torch.einsum("nhsq,nhqw->nhsw", self.dropout(torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer).relu()), value_layer)
        # context_layer = torch.einsum("nhsq,nhqw->nhsw", torch.nn.functional.gelu(torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer)), value_layer)
        # context_layer = torch.einsum("nhsq,nhqw->nhsw", torch.nn.functional.silu(torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer).relu()), value_layer)
        # self.relu_constant.data = torch.clamp(self.relu_constant.data, -1, 1)
        # context_layer = torch.einsum("nhsq,nhqw->nhsw", torch.nn.functional.relu(torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer)+self.relu_constant), value_layer)
        
        
        # # Divide by sequence length
        # context_layer = torch.einsum("nhsq,nhqw->nhsw", 
        #                              ((torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer))
        #                                 / attention_mask.sum(-1).unsqueeze(-1)**0.5), 
        #                 value_layer)
        # attention_probs = ((torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer)) / (attention_mask.sum(-1).unsqueeze(-1))**0.5)
        # context_layer = torch.einsum("nhsq,nhqw->nhsw", attention_probs, value_layer)
        
        # # Divide by total sequence length
        # context_layer = torch.einsum("nhsq,nhqw->nhsw", 
        #                              (torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer))
        #                                 / 512, 
        #                 value_layer)
        
        # # Divide by learnable constant
        # context_layer = torch.einsum("nhsq,nhqw->nhsw", 
        #                              (torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer))
        #                                 / self.norm_const, 
        #                 value_layer)
        
        # # Exp by learnable constant
        # context_layer = torch.einsum("nhsq,nhqw->nhsw", 
        #                              ((torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer))
        #                                 / attention_mask.sum(-1).unsqueeze(-1)**self.norm_const), 
        #                 value_layer)
        
        # # Divide by sum
        # attention_probs = torch.einsum("nhse,nhqe->nhsq", query_layer, key_layer)
        # context_layer = torch.einsum("nhsq,nhqw->nhsw", 
        #                                 (attention_probs
        #                                     / (attention_probs.sum(-1).unsqueeze(-1)+1e-7)) * attention_mask.transpose(-1, -2),
        #                 value_layer)
         
        attention_probs = None
            
        # attention_probs = (((((value_layer**2).sum(-1)**0.5)-((context_layer**2).sum(-1)**0.5))**2).sum(-1)**0.5).mean()
        
        # penalize that the sequence magnitudes are different
        #attention_probs = ((((value_layer**2).sum(-1)**0.5).mean(-1)-((context_layer**2).sum(-1)**0.5).mean(-1))**2).mean()
        
        # penalize that the token magnitudes are different
        # attention_probs = ((((value_layer**2).sum(-1)**0.5)-((context_layer**2).sum(-1)**0.5))**2).mean()

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            raise NotImplementedError

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            raise NotImplementedError
            attention_probs = attention_probs * head_mask

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        


        # context_layer = self.norm(context_layer)
        # attention_probs = (hidden_states*attention_mask.transpose(-1, -2).squeeze(1), context_layer)





        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
