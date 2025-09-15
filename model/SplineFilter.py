import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn.parameter import Parameter

from utils.spline_evaluation import pad_rows, pad_columns, BSpline


class SplineFilter(torch.nn.Module):
    def __init__(
            self,
            filter_channels: int,
            filter_scale: int,
            filter_knots: Tensor,
            filter_degree: int,
            filter_coeffs: Tensor=None,
            device = 'cpu'
            ):

        super().__init__()
        self.n_knots = len(filter_knots)
        self.n_bases = self.n_knots - filter_degree - 1
        self.filter_knots = filter_knots
        self.filter_degree = filter_degree

        self.filter_channels = filter_channels

        self.filter_coeffs = filter_coeffs
        if self.filter_coeffs is None:
            self.filter_coeffs = Parameter(
                torch.randn((self.n_bases, filter_channels), device=device)*filter_scale
                )
        self.device = device

    def forward(
        self,
        eval_x: Tensor,
        eval_eigs: Tensor,
    ):
        # need sparse implementation
        eval_eigs = BSpline(knots=self.filter_knots,
                            degree=self.filter_degree,
                            eval_eigs=eval_eigs)          # B x N x n_bases;

        F = self.filter_channels
        B = eval_eigs.shape[0]
        N = eval_eigs.shape[1]

        eval_eigs = eval_eigs.unsqueeze(0).expand(
            F, -1, -1, -1
            ).reshape(F * B, N, self.n_bases)

        filter_coeffs = self.filter_coeffs.unsqueeze(0).expand(
            B, -1, -1
            )
        filter_coeffs = torch.split(filter_coeffs, 1, dim=-1)
        filter_coeffs = torch.cat(filter_coeffs, dim=0)   # F * B x n_bases x 1

        spectral_filter = torch.bmm(eval_eigs, filter_coeffs).squeeze()   # F*B x N
        spectral_filter = torch.split(spectral_filter, B)   # B contiguous: F, B x N
        spectral_filter = torch.stack(spectral_filter).transpose(0,1)   # B x F x N
        spectral_filter = spectral_filter.transpose(-1, -2)   # B x N x F
        spectral_filter = torch.square(spectral_filter)   # d

        fourier_mod = eval_x * spectral_filter
        return fourier_mod
