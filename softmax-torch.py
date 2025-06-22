import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    # Step 1: Subtract the max value from each row for numerical stability
    # Step 2: Calculate the exponential of every element of the adjusted tensor
    # Step 3: Sum the exponential values along each row
    # Step 4: Divide each row by its corresponding sum

    maxx, _ = logits.max(dim=1, keepdims=True)
    logits -= maxx
    res = torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdims=True)

    return res
