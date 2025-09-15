import os
import logging
import argparse
import numpy as np
import torch.nn.functional as F

import torch
from torch import Tensor
from torch_geometric.data import Dataset

from .pth_dataset import PthDataset
from utils.spline_evaluation import BSpline


def PSpline_processing(
    save_dir: str,
    dataset_name: str,
    dataset_PS: PthDataset,
    knots: Tensor,
    _lambda: float,
    ):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # de Boor's
    degree = 3
    n_bases = (len(knots) - degree - 1) - 4   # remove first and last two columns

    # 2nd order, Neumann boundaries
    ldiag = torch.diag(torch.ones(n_bases-1), diagonal=-1)
    udiag = torch.diag(torch.ones(n_bases-1), diagonal=1)
    mdiag = -2*torch.diag(torch.ones(n_bases), diagonal=0)
    D_2 = ldiag + udiag + mdiag
    D_2[0,0] = D_2[n_bases-1, n_bases-1] = -1

    count = 0
    for data_dict in dataset_PS:
        B = BSpline(knots=knots,
                    degree=degree,
                    eval_eigs=data_dict['eigs'])[:, 2:-2]   # CPU, val
        PSReg = torch.linalg.inv(B.T @ B + _lambda*D_2.T @ D_2) @ B.T
        data_dict['PSReg'] = PSReg

        torch.save(
            data_dict,
            os.path.join(save_dir, '{}_PS_{:05d}.pth'.format(dataset_name, count))
            )
        count += 1


def main():
    # 1.  Parser
    parser = argparse.ArgumentParser(
        description="P-Spline regression matrices"
    )
    parser.add_argument('--save_root', type=str, default='../data',
                        help='Script config file path')
    parser.add_argument('--dataset_name', type=str, default='PROTEINS',
                        help='Name of TUDataset')
    parser.add_argument('--knots', type=str, default='../data/PROTEINS/pth_STATS/PROTEINS_knots.pth',
                        help='Torch tensor of knots')
    parser.add_argument('--_lambda', type=float, default=0.0,
                        help='Finite difference penalty parameter')
    parser.add_argument('--auto_clear', type=bool, default=False,
                        help='Remove directory from which old data is loaded, recommended: False')
    args = parser.parse_args()

    # 2. pickel dump
    save_dir = os.path.join(args.save_root, args.dataset_name, 'pth_PSReg')
    load_dir = os.path.join(args.save_root, args.dataset_name, 'pth_truncate')
    dataset_PS = PthDataset(load_dir=load_dir)
    knots = torch.load(
        args.knots
    )

    PSpline_processing(
        save_dir=save_dir,
        dataset_name=args.dataset_name,
        dataset_PS=dataset_PS,
        knots=knots,
        _lambda=args._lambda,
    )

    if args.auto_clear:
        os.system("rm -rf {}".format(load_dir))

if __name__ == '__main__':
    main()
