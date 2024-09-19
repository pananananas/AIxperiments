import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class  CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_token: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_token, n_embed))


    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        x = self.token_embedding(tokens)         # (Batch_Size, Seq_Len)  ->  (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding
        
        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embed: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        
        self.layernorm_2 = nn.LayerNorm(n_embed)

        self.linear_1 = nn.Linear(n_embed, n_embed * 4),
        self.linear_2 = nn.Linear(n_embed * 4, n_embed)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Self attention
        residual = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)     # (Batch_Size, Seq_Len, Dim)
        x += residual

        # Feed forward layer
        residual = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x) # Quick GELU activation
        x = self.linear_2(x)
        x += residual

        return x


class CLIP(nn.Module):

    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module({
            CLIPLayer(12, 768) for i in range(12)
        })

        self.layernorm = nn.LayerNorm(768)


    def forward(self, tokrens: torch.LongTensor) -> torch.Tensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)          # (Batch_Size, Seq_Len)  ->  (Batch_Size, Seq_Len, Dim)
        
        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)          # (Batch_Size, Seq_Len, Dim)  ->  (Batch_Size, Seq_Len, Dim)
    
        return output