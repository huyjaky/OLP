# PyTorch implementations for token-level mixup / cutmix
import torch
import numpy as np

def token_cutmix_swap_torch(token_ids, labels, swap_prob=0.1, device=None):
    """
    Discrete token-level CutMix using PyTorch tensors. For each sample, each token position
    is swapped with probability `swap_prob` from a paired (shuffled) sample.

    Args:
        token_ids: torch.LongTensor shape (N, L)
        labels: torch.Tensor shape (N, C) (one-hot/soft labels)
        swap_prob: float probability of swapping each token position
        device: torch.device or None (defaults to token_ids.device)

    Returns:
        mixed_ids: torch.LongTensor (N, L)
        mixed_labels: torch.Tensor (N, C)
        swap_fracs: numpy array (N,) fraction swapped per sample
        index: torch.LongTensor permutation used
    """
    if device is None:
        device = token_ids.device

    N, L = token_ids.size()
    index = torch.randperm(N, device=device)
    ids_shuf = token_ids[index]
    labels_shuf = labels[index]

    mask = torch.rand(N, L, device=device) < float(swap_prob)
    mixed_ids = token_ids.clone()
    mixed_ids[mask] = ids_shuf[mask]

    swap_fracs = mask.float().sum(dim=1) / float(L)
    lam = (1.0 - swap_fracs).view(N, 1)
    mixed_labels = lam * labels + (1.0 - lam) * labels_shuf

    return mixed_ids, mixed_labels, swap_fracs.cpu().numpy(), index

