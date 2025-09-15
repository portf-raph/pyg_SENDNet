import numpy as np
from math import sqrt
from typing import Callable

import torch
from torch import Tensor
import torch_scatter as S
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter, to_dense_adj, get_laplacian


def get_eigs(adj: Tensor):
    if adj.device == 'cuda':  # TODO: return type hint
        eigs, V = cp.linalg.eigh(cp.asarray(adj.to('cuda')))
        eigs, V = cp.squeeze(eigs), cp.squeeze(V)
        eigs = torch.Tensor(eigs)
        V = torch.Tensor(V)
    else:
        eigs, V = np.linalg.eigh(adj.detach().cpu().numpy())
        eigs, V = np.squeeze(eigs), np.squeeze(V)
        eigs = torch.from_numpy(eigs)
        V = torch.from_numpy(V)
    return eigs, V


def unroll_eigs(eigs: Tensor,
                atol: float,
                spacing: float) -> Tensor:

    repeats = torch.isclose(eigs, F.pad(eigs[:-1], (1,0)), atol=1e-12)
    starts = repeats & (~F.pad(repeats[:-1], (1,0), value=False))
    ends = repeats & (~F.pad(repeats[1:], (0,1), value=False))

    group_ids = (torch.cumsum(starts.int(), dim=0)-1) * repeats.int()
    group_cumsum = (torch.cumsum(repeats.int(), dim=0)) * repeats.int()
    group_increments = torch.unique(group_cumsum * ends.int(),sorted=False)

    counts = group_cumsum - group_increments[group_ids]
    eigs += counts * spacing
    return eigs


def unique(eigs: Tensor, 
           atol: float=1e-12):
    repeats_l = torch.isclose(eigs, F.pad(eigs[:-1], (1,0), value=-1), atol=atol)
    uniques = ~repeats_l
    group_ids = torch.cumsum(uniques.int(), dim=0)-1

    return eigs[uniques], group_ids


def spectral_energy(group_ids: Tensor,
                    fourier_coeffs: Tensor,
                    ) -> Tensor:
    
    energy = torch.square(fourier_coeffs)
    energy_out = S.scatter(src=energy,
                           index=group_ids,
                           dim=-2)  # B x N x F or N x F

    return energy_out   # val


def pyg_batch(data_dicts: list[dict]) -> Batch:
    batch = [Data(
        x=data_dict['x'],
        edge_index=data_dict['edge_index'],
    ) for data_dict in data_dicts]

    return Batch.from_data_list(batch, follow_batch=['x'])


def serial_routine(
                data: Data,
                count: int,
                upper: float,
                lower: float,
                transform: Callable,
                logger,
                ):
    data_dict = {}
    data_dict['x'] = data.x
    data_dict['y'] = data.y
    data_dict['edge_index'] = data.edge_index
    data_dict['edge_attr'] = data.edge_attr
    edge_index, edge_attr = get_laplacian(edge_index=data.edge_index,
                                          edge_weight=data.edge_attr,
                                          normalization='sym')
    dense_adj = to_dense_adj(edge_index=edge_index,
                             edge_attr=edge_attr)
    eigs, V = get_eigs(dense_adj)

    data_dict['eigs'] = transform(eigs, data.edge_index)
    if eigs is None:
        logger.info('eigs is None @ count {}'.format(count))

    eigs_max = torch.max(eigs).item()
    eigs_min = torch.min(eigs).item()
    if eigs_max > upper:
        upper = eigs_max
    if eigs_min < lower:
        lower = eigs_min

    data_dict['V'] = V
    if V is None:
        logger.info('V is None @ count {}'.format(count))

    return data_dict, upper, lower
