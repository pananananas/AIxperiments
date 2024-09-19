import math
import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x: (Batch_size, Seq_len, Dim)
        input_shape = x.shape
        batch_size, seq_length, dim = input_shape

        intermediate_shape = (batch_size, seq_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)      # (Batch_size, Seq_len, Dim)  ->  (Batch_size, Seq_length, Dim*3)  ->  3*(Batch_size, Seq_len, Dim)
        q = q.view(intermediate_shape).transpose(1, 2)  # (Batch_size, Seq_len, Dim)  ->  (Batch_size, Seq_len, H, Dim/H)  ->  (Batch_size, H, Seq_len, Dim/H)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)        # (Batch_size, H, Seq_len, Dim/H)  @  (Batch_size, H, Dim/H, Seq_len)  ->  (Batch_size, H, Seq_len, Seq_len)

        if causal_mask:
            # Mask where upper triangle is made up of 1s
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)       
        weight = F.softmax(weight, dim=-1)      
        output = weight @ v                     # (Batch_size, H, Seq_len, Seq_len)  @  (Batch_size, H, Seq_len, Dim/H)  ->  (Batch_size, H, Seq_len, Dim/H)
        output = output.transpose(1, 2)         # (Batch_size, H, Seq_len, Dim/H)  ->  (Batch_size, Seq_len, H, Dim/H)
        output = output.reshape(input_shape)    # (Batch_size, Seq_len, H, Dim/H)  ->  (Batch_size, Seq_len, Dim)
        output = self.out_proj(output)

        return output                           # (Batch_size, Seq_len, Dim)
    


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Seq_len_Q, Dim_Q)
        # y: (Batch_size, Seq_len_KV, Dim_KV)
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x).view(interim_shape).transpose(1, 2)  
        k = self.k_proj(y).view(interim_shape).transpose(1, 2)  
        v = self.v_proj(y).view(interim_shape).transpose(1, 2)  

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2).continuous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        
        return output