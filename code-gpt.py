import torch
import torch.nn as nn
from torchtyping import TensorType

# 1. Remember to include an additional LayerNorm after the block sequence and before the final linear layer
# 2. Instantiate in the following order: Word embeddings, position embeddings, transformer blocks, final layer norm, and vocabulary projection.
class GPT(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, model_dim)
        # Position embeddings
        self.pos_emb = nn.Embedding(context_length, model_dim)
        # Transformer blocks sequence
        self.blocks = nn.Sequential(*[
            self.TransformerBlock(model_dim, num_heads)
            for _ in range(num_blocks)
        ])
        # Final layer norm
        self.final_norm = nn.LayerNorm(model_dim)
        # Vocabulary projection
        self.vocab_proj = nn.Linear(model_dim, vocab_size)
        # Store context length for position IDs
        self.context_length = context_length

    def forward(self, context: TensorType[int]) -> TensorType[float]:
        # context is a tensor of shape (batch_size, seq_len) containing token IDs
        torch.manual_seed(0)
        batch_size, seq_len = context.shape
        # token + position embeddings
        tok = self.token_emb(context) # one model_dim-vector per token
        positions = torch.arange(seq_len) # 1D tensor [0,1,2,...,seq_len-1]
        pos = self.pos_emb(positions)         # → [seq_len, model_dim]
        pos = pos.unsqueeze(0)                # → [1, seq_len, model_dim]
        pos = pos.expand(batch_size, -1, -1)  # → [batch_size, seq_len, model_dim]
        x = tok + pos # offset each token embedding by its position embedding
        
        # transformer blocks
        x = self.blocks(x)
        # final norm and projection
        x = self.final_norm(x)
        # [batch_size, seq_len, model_dim] -> [batch_size, seq_len, vocab_size]
        # each entry is the raw score for vocab token v being the next token at t in seq b
        logits = self.vocab_proj(x)
        probs = nn.functional.softmax(logits, dim=-1)
        return torch.round(probs * 10000) / 10000

    # Do NOT modify the code below this line
    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim: int, head_size: int):
                    super().__init__()
                    torch.manual_seed(0)
                    self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.value_gen = nn.Linear(model_dim, head_size, bias=False)
                
                def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                    k = self.key_gen(embedded)
                    q = self.query_gen(embedded)
                    v = self.value_gen(embedded)

                    scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
                    context_length, attention_dim = k.shape[1], k.shape[2]
                    scores = scores / (attention_dim ** 0.5)

                    lower_triangular = torch.tril(torch.ones(context_length, context_length))
                    mask = lower_triangular == 0
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim = 2)

                    return scores @ v
                
            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                torch.manual_seed(0)
                self.att_heads = nn.ModuleList()
                for i in range(num_heads):
                    self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))

            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                head_outputs = []
                for head in self.att_heads:
                    head_outputs.append(head(embedded))
                concatenated = torch.cat(head_outputs, dim = 2)
                return concatenated
        
        class VanillaNeuralNetwork(nn.Module):

            def __init__(self, model_dim: int):
                super().__init__()
                torch.manual_seed(0)
                self.up_projection = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2) # using p = 0.2
            
            def forward(self, x: TensorType[float]) -> TensorType[float]:
                torch.manual_seed(0)
                return self.dropout(self.down_projection(self.relu(self.up_projection(x))))

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.linear_network = self.VanillaNeuralNetwork(model_dim)
            self.first_norm = nn.LayerNorm(model_dim)
            self.second_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            embedded = embedded + self.attention(self.first_norm(embedded)) # skip connection
            embedded = embedded + self.linear_network(self.second_norm(embedded)) # another skip connection
            return embedded
