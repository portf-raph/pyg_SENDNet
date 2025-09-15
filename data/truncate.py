import os
import logging
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset

from utils.data_helper import unique, spectral_energy
from .pth_dataset import PthDataset


def GFT_truncate(
    save_dir: str,
    dataset_name: str,
    dataset: PthDataset,
    threshold: float,
    ):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    count = 0
    num_eigs = 0
    for data_dict in dataset:

        # truncate
        eigs = data_dict['eigs']
        V = data_dict['V']

        mask = eigs <= threshold
        eigs = eigs[mask]
        V = V[mask]

        # unique eigs, group_ids
        eigs, group_ids = unique(eigs, atol=1e-12)

        data_dict['eigs'] = eigs
        data_dict['group_ids'] = group_ids
        data_dict['V'] = V
        torch.save(
            data_dict,
            os.path.join(save_dir, '{}_truncate_{:05d}.pth'.format(dataset_name, count))
            )
        num_eigs += len(data_dict['eigs'])
        count += 1

    print('Num eigs: {}'.format(num_eigs))


def main():
    # 1.  Parser
    parser = argparse.ArgumentParser(
        description="Graph Fourier transform and truncation"
    )
    parser.add_argument('--save_root', type=str, default='../data',
                        help='Script config file path')
    parser.add_argument('--dataset_name', type=str, default='PROTEINS',
                        help='Name of TUDataset')
    parser.add_argument('--threshold', type=float, default=0,
                        help='Cut-off value for eigenvalues')
    parser.add_argument('--auto_clear', type=bool, default=True,
                        help='Remove directory from which old data is loaded, recommended: True')
    args = parser.parse_args()

    # 2. pickel dump
    save_dir = os.path.join(args.save_root, args.dataset_name, 'pth_truncate')
    load_dir = os.path.join(args.save_root, args.dataset_name, 'pth')
    dataset = PthDataset(load_dir=load_dir)

    GFT_truncate(
        save_dir=save_dir,
        dataset_name=args.dataset_name,
        dataset=dataset,
        threshold=args.threshold
    )

    if args.auto_clear:
        os.system("rm -rf {}".format(load_dir))

if __name__ == '__main__':
    main()
