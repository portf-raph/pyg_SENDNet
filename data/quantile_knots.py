import os
import logging
import argparse
import numpy as np

import torch
from torch import Tensor


def quantile_knots(
    save_dir: str,
    dataset_name: str,
    list_eigs: Tensor,
    n_knots: int,
    threshold: float,
    ):
    quantiles = torch.linspace(start=0,
                               end=1,
                               steps=n_knots)
    knots = torch.quantile(
                torch.cat((list_eigs, torch.Tensor([threshold])), dim=-1),
                quantiles)
    knots = torch.unique(knots)

    print("Number of knots, unique: {}".format(len(knots)))
    repeat_begin = torch.Tensor([knots[0], knots[0]])
    repeat_end = torch.Tensor([knots[-1], knots[-1]])
    augmented_knots = torch.cat((repeat_begin, knots), dim=-1)
    augmented_knots = torch.cat((augmented_knots, repeat_end),dim=-1)

    torch.save(
        augmented_knots,
        os.path.join(save_dir, '{}_knots.pth'.format(dataset_name))
        )


def main():
    # 1.  Parser
    parser = argparse.ArgumentParser(
        description="Preprocessing data, getting pth file"
    )
    parser.add_argument('--save_root', type=str, default='../data',
                        help='Script config file path')
    parser.add_argument('--dataset_name', type=str, default='PROTEINS',
                        help='Name of TUDataset')
    parser.add_argument('--n_knots', type=int, default=50,
                        help='Number of knots originally (before removal of repeats and addition of start & end repeats).')
    parser.add_argument('--threshold', type=float, default=2.0,
                        help='Earlier threshold used to truncate eigenvalues')
    args = parser.parse_args()

    # 2. Pickel dump
    save_dir = os.path.join(args.save_root, args.dataset_name, 'pth_STATS')
    load_dir = os.path.join(args.save_root, args.dataset_name, 'pth_STATS')

    list_eigs = torch.load(
        os.path.join(load_dir, '{}_eigs.pth'.format(args.dataset_name))
    )
    quantile_knots(
        save_dir=save_dir,
        dataset_name=args.dataset_name,
        list_eigs=list_eigs,
        n_knots=args.n_knots,
        threshold=args.threshold
    )

if __name__ == '__main__':
    main()
