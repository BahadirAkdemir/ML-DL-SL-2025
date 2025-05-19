import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

LARGE_NUM = 1e9


def add_supervised_loss(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    # If labels are one-hot, convert to class indices
    return F.cross_entropy(logits, labels)



def add_contrastive_loss(hidden: torch.Tensor,
                         hidden_norm: bool = True,
                         temperature: float = 1.0,
                         strategy: Optional[torch.nn.parallel.DistributedDataParallel] = None) -> Tuple[

    torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SimCLR contrastive loss."""
    if hidden_norm:
        hidden = F.normalize(hidden, dim=-1)

    hidden1, hidden2 = torch.chunk(hidden, 2, dim=0)
    batch_size = hidden1.shape[0]

    if strategy is not None and torch.distributed.is_initialized():
        hidden1_large = gather_from_all(hidden1)
        hidden2_large = gather_from_all(hidden2)
        rank = torch.distributed.get_rank()
        enlarged_batch_size = hidden1_large.shape[0]
        labels_idx = torch.arange(batch_size, device=hidden.device) + rank * batch_size
        labels = F.one_hot(labels_idx, num_classes=enlarged_batch_size*2)
        masks = F.one_hot(labels_idx, num_classes=enlarged_batch_size).to(hidden1_large.device)
    else:
        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(torch.arange(batch_size), batch_size*2)
        masks = F.one_hot(torch.arange(batch_size), batch_size).to(hidden1_large.device)

    # Compute logits
    logits_aa = torch.matmul(hidden1, hidden1_large.T) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, hidden2_large.T) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, hidden2_large.T) / temperature
    logits_ba = torch.matmul(hidden2, hidden1_large.T) / temperature

    # Compute loss
    logits_a = torch.cat([logits_ab, logits_aa], dim=1)
    logits_b = torch.cat([logits_ba, logits_bb], dim=1)
    labels = labels.to(logits_a.device).float()

    loss_a = F.cross_entropy(logits_a, labels, reduction='none')
    loss_b = F.cross_entropy(logits_b, labels, reduction='none')
    loss = (loss_a + loss_b).mean()

    return loss, logits_ab, labels


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all distributed processes."""
    if not torch.distributed.is_initialized():
        return tensor

    world_size = torch.distributed.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)