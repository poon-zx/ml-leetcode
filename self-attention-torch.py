
import torch
import math
import torch.nn.functional as F

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    result = None
    d_k = query.size(dim=1)
    attention_values = torch.nn.functional.softmax(torch.matmul(query, torch.t(key)) / math.sqrt(d_k), dim=-1)
    result = torch.matmul(attention_values, value)
    return result