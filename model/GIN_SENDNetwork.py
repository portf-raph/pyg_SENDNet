import torch
import torch.nn.functional as F

from torch import Tensor

from utils.spline_evaluation import pad_columns, pad_rows
from utils.data_helper import pyg_batch, spectral_energy
from .SplineFilter import SplineFilter
from .MLP import MLP
from .GIN import GIN


class GIN_SENDNetwork(torch.nn.Module):
    def __init__(self,
                 GIN_cfg: dict,
                 MLP_cfg: dict,
                 filter_channels: int,
                 filter_scale: float,
                 filter_knots: Tensor,
                 filter_degree: int,
                 filter_coeffs: Tensor=None,
                 device = 'cpu'):
        super().__init__()
        self.SplineFilter = SplineFilter(
            filter_channels=filter_channels,
            filter_scale=filter_scale,
            filter_knots=filter_knots,
            filter_degree=filter_degree,
            filter_coeffs=filter_coeffs,
            device=device,
        )
        self.GIN = GIN(**GIN_cfg,
                       device=device)
        self.MLP = MLP(**MLP_cfg,
                       device=device)

    def forward(self,
                data_dicts: list[dict]):
        # GIN
        batch = pyg_batch(data_dicts)
        batch.x = self.GIN(batch.x, batch.edge_index)    # post 1x1
        batch_vec = batch.x_batch
        
        # GFT & SEND
        batch_x = torch.stack(
                pad_columns([batch.x[batch_vec==i] for i in range(len(data_dicts))],
                value=0),
                dim=0
                )
        batch_V = pad_rows([d['V'] for d in data_dicts], value=0)
        batch_V = pad_columns([V for V in batch_V], value=0)
        batch_V = torch.stack(batch_V, dim=0)
        batch_Vx = torch.bmm(batch_V, batch_x)  # (B x repN x untrN) x (B x untrN x F)

        batch_group_ids = torch.stack(
                pad_columns([d['group_ids'].unsqueeze(-1).expand(-1, batch_Vx.shape[-1]) for d in data_dicts], 
                value=0),
                dim=0
            )   # B x repN x F
        eval_x = spectral_energy(group_ids=batch_group_ids,
                                 fourier_coeffs=batch_Vx)  # B x repN x F -> B x N x F

        # SplineFilter
        eval_eigs = torch.stack(
            pad_columns([d['eigs'].unsqueeze(-1) for d in data_dicts], value=float('-1.0e2')),
            dim=0
        ).squeeze()     # B x N

        fourier_mod = self.SplineFilter(eval_x=eval_x, eval_eigs=eval_eigs)

        PSReg = torch.stack(
            pad_rows([d['PSReg'] for d in data_dicts], value=0),
            dim=0
        )

        N = fourier_mod.shape[1]
        F = fourier_mod.shape[2]
        spline_coeffs = torch.bmm(PSReg, fourier_mod)   # (B x n_bases x N) x (B x N x F) = (B x n_bases x F)
        spline_coeffs = torch.split(spline_coeffs, 1, dim=-1)   # F, B x n_bases
        spline_coeffs = [SC.squeeze() for SC in spline_coeffs]  # NEED FIX
        spline_coeffs = torch.cat(spline_coeffs, dim=-1)   # B x F*n_bases

        # MLP
        out = self.MLP(spline_coeffs)
        return out
