from typing import Optional, Tuple
import math
import torch
from torch import nn
import os
import wandb





import torch
from torch.autograd import Function

class SoftenedReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, leak_slope=1e-7):
        ctx.save_for_backward(input)
        ctx.leak_slope = leak_slope
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] *= ctx.leak_slope
        return grad_input, None
    
    
class SoftenedReLU(torch.nn.Module):
    def __init__(self, leak_slope=1e-7):
        super(SoftenedReLU, self).__init__()
        self.leak_slope = leak_slope

    def forward(self, input):
        return SoftenedReLUFunction.apply(input, self.leak_slope)







class BertCosAttention(nn.Module):
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
        # self.norm_const = nn.Parameter(torch.ones(1, self.num_attention_heads, 1, 1, dtype=self.query.weight.dtype, device=self.query.weight.device))
        # init between 0.1 and 0.9
        # self.norm_const = nn.Parameter(torch.rand(1, self.num_attention_heads, 1, 1, dtype=self.query.weight.dtype, device=self.query.weight.device)*0.8+0.1)
        # Between -1 and 1
        # self.norm_const = nn.Parameter(torch.rand(1, self.num_attention_heads, 1, 1, dtype=self.query.weight.dtype, device=self.query.weight.device)*2-1)
        
        # self.norm = nn.LayerNorm(self.all_head_size)
        
        # self.relu = SoftenedReLU()
        
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
        value_layer = value_layer / attention_mask.sum(-1).unsqueeze(-1)#**self.norm_const.sigmoid()
        # value_layer = torch.nn.functional.normalize(value_layer, dim=-1, p=2)
        
        
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
