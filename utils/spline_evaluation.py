import torch
import torch.nn.functional as F

from torch import Tensor

def BSpline(
    knots: Tensor,
    degree: int,
    eval_eigs: Tensor,)->Tensor:
    """
    Taken directly from https://pypi.org/project/splinetorch/.
    """
    eps = 1e-10
    dtype = eval_eigs.dtype
    device = eval_eigs.device

    n_knots = len(knots)
    n_bases = n_knots - degree - 1

    # degree 0 init
    degree0 = []
    for j in range(n_knots - 1):
        if j < n_bases - 1:
            mask = (knots[j] <= eval_eigs) & (eval_eigs < knots[j + 1])
        else:
            # include right endpoint for last base
            mask = (knots[j] <= eval_eigs) & (eval_eigs <= knots[j + 1])
        degree0.append(mask.to(dtype))

    basis_prev = torch.stack(degree0, dim=-1)

    for p in range(1, degree + 1):
        current_list  = []
        for j in range(n_knots - p - 1):

            denom1 = knots[j + p] - knots[j]
            left = torch.zeros_like(eval_eigs, dtype=dtype, device=device)
            if denom1 > eps:
                left = (eval_eigs - knots[j]) / denom1 * basis_prev[..., j]

            denom2 = knots[j + p + 1] - knots[j + 1]
            right = torch.zeros_like(eval_eigs, dtype=dtype, device=device)
            if denom2 > eps:
                right = (knots[j + p + 1] - eval_eigs) / denom2 * basis_prev[..., j+1]

            current_list.append(left + right)
        basis_prev = torch.stack(current_list, dim=-1)  # shape: (batch, n_points, n_knots - p - 1)

    return basis_prev[..., :n_bases]


def pad_columns(
        tensors: list[Tensor],
        value: float
    ) -> list[Tensor]:

    lengths = [tensor.shape[0] for tensor in tensors]
    max_length = max(lengths)
    padded_tensors = [F.pad(
        tensor, (0, 0, 0, max_length - lengths[i]), value=value)
     for i, tensor in enumerate(tensors)]

    return padded_tensors


def pad_rows(
        tensors: list[Tensor],
        value: float
    ) -> list[Tensor]:

    lengths = [tensor.shape[1] for tensor in tensors]
    max_length = max(lengths)
    padded_tensors = [F.pad(
        tensor, (0, max_length - lengths[i], 0, 0), value=value)
     for i, tensor in enumerate(tensors)]

    return padded_tensors
