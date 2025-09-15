import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Dataset

from .pth_dataset import PthDataset


def run_stats(
    save_dir: str,
    dataset_name: str,
    dataset: PthDataset,
    extreme_eigs: dict,
    save_eigs: bool,
    num_bounds: int=100,
    ):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    log_file = os.path.join(save_dir, 'log_{}.txt'.format(dataset_name))
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger.info('Running statistics on training set')

    num_eigs = 0
    list_eigs = []

    upper = extreme_eigs['upper']
    lower = extreme_eigs['lower']
    bounds = np.linspace(upper, lower, num_bounds)
    bin_length = bounds[1] - bounds[0]
    bin_centers = bounds + bin_length/2

    eigs_bins = [0] * (num_bounds)
    lengths = []
    for data_dict in dataset:
        eigs = data_dict['eigs'].numpy()
        num_eigs += len(eigs)
        lengths += [len(eigs)]
        list_eigs += eigs.tolist()

        unique_bins, bin_counts = np.unique(np.digitize(eigs, bounds), return_counts=True)
        for bin_idx, bin_count in zip(unique_bins, bin_counts):
            eigs_bins[bin_idx-1] += bin_count

    print('Num eigs: {}'.format(num_eigs))

    # Plot eigs of class 0
    plt.figure(figsize=(10,1))
    plt.scatter(list_eigs, np.zeros(len(list_eigs)), marker='x', alpha=0.7, linewidths=0.01)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.yticks([])
    plt.xlabel('Spectral interval')
    plt.title('Spectral density of dataset')
    plt.show(block=True)

    eigs_bins = np.array(eigs_bins) / num_eigs
    plt.bar(bin_centers, eigs_bins, label='Dataset spectral density', width=bin_length)
    plt.title('Spectral density of dataset (bars)')
    plt.show(block=True)

    max_length = np.amax(lengths)
    lengths_bins = np.bincount(lengths, minlength=max_length+1)
    plt.bar(np.arange(0, max_length+1, 1), lengths_bins, label='Dataset graph sizes')
    plt.title('Graph sizes of dataset')
    plt.show(block=True)

    logger.info('Done displaying statistics')

    print('Length of list_eigs: {}'.format(len(list_eigs)))

    if save_eigs:
        list_eigs = torch.Tensor(list_eigs)
        torch.save(list_eigs,
                  os.path.join(save_dir, '{}_eigs.pth'.format(dataset_name))
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
    parser.add_argument('--load_subdir', type=str, default='pth',
                        help='pth or pth_truncate')
    parser.add_argument('--num_bounds', type=int, default=100,
                        help='Number of bins, plus 1')
    parser.add_argument('--save_eigs', type=bool, default=False,
                        help='Recommended: set save_eigs=True after truncation, False otherwise')
    args = parser.parse_args()

    # 2. Stats
    save_dir = os.path.join(args.save_root, args.dataset_name, 'pth_STATS')
    load_dir = os.path.join(args.save_root, args.dataset_name, args.load_subdir)

    dataset = PthDataset(load_dir=load_dir)
    extreme_eigs = torch.load(
        os.path.join(args.save_root, args.dataset_name, '{}_extreme_eigs.pth'.format(args.dataset_name)),
        weights_only=False,
    )
    run_stats(
        save_dir=save_dir,
        dataset_name=args.dataset_name,
        dataset=dataset,
        extreme_eigs=extreme_eigs,
        save_eigs=args.save_eigs,
        num_bounds=args.num_bounds,
        )


if __name__ == '__main__':
    main()
