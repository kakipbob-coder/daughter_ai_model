import torch

def causal_mask(seq_len):
    """
    Create a causal (lower-triangular) mask.
    Shape: (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)
