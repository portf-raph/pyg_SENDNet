import torch
import torch.nn.functional as F

from torch import Tensor

from utils.spline_evaluation import pad_columns, pad_rows
from .SplineFilter import SplineFilter
from .MLP import MLP


class SENDNetwork(torch.nn.Module):
    def __init__(self,
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
        self.MLP = MLP(**MLP_cfg,
                       device=device)

    def forward(self,
                data_dicts: list[dict]):
        eval_x = torch.stack(
            pad_columns([d['x'] for d in data_dicts], value=0),
            dim=0
        )
        eval_eigs = torch.stack(
            pad_columns([d['eigs'].unsqueeze(-1) for d in data_dicts], value=float('-1.0e2')),
            dim=0
        ).squeeze()

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

        out = self.MLP(spline_coeffs)
        return out
