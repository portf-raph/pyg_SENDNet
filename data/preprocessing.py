import os
import math
import logging
import argparse
import numpy as np

import torch
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj

from utils.data_helper import get_eigs, serial_routine


def dump_eigs_data(
              save_root: str,
              dataset_name: str,
              dataset: Dataset,
              ):
    save_root_dir = os.path.join(save_root, dataset_name)
    save_dir = os.path.join(save_root, dataset_name, 'pth')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    log_file = os.path.join(save_root_dir, 'log_{}.txt'.format(dataset_name))
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger.info('Dumping eigs data from {}'.format(dataset_name))

    count = 0
    upper = 0
    lower = 0
    for data in dataset:
        data_dict, upper, lower = serial_routine(
            data=data,
            count=count,
            upper=upper,
            lower=lower,
            transform=lambda eigs, edge_index: eigs,
            logger=logger,
        )
        torch.save(
            data_dict,
            os.path.join(save_dir, '{}_{:05d}.pth'.format(dataset_name, count))
        )
        if count == 0:
            logger.info('STORAGE: {}'.format(data.edge_index.device))
        count += 1

    extreme_eigs = {
        'upper': upper,
        'lower': lower,
    }
    print(extreme_eigs)

    torch.save(
        extreme_eigs,
        os.path.join(save_root_dir, '{}_extreme_eigs.pth'.format(dataset_name))
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
    args = parser.parse_args()

    # 2. Pickel dump
    dataset = TUDataset(root=args.save_root, name=args.dataset_name)
    dump_eigs_data(
        save_root=args.save_root,
        dataset_name=args.dataset_name,
        dataset=dataset,
        )

if __name__ == '__main__':
    main()
