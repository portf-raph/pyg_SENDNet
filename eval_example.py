import os
import sys
import torch
import pprint
import logging
import datetime
import argparse
import traceback

import torch
from torch.utils.data import random_split

from model.GIN_SENDNetwork import GIN_SENDNetwork
from model.SENDNetwork import SENDNetwork
from runner.pth_runner import PTHRunner
from data.pth_dataset import PthDataset
from utils.train_helper import load_model, get_config


def main():
    #1. Parser
    parser = argparse.ArgumentParser(
          description="Running evaluation experiment"
    )
    parser.add_argument('--GIN_experiment', type=bool, default=True, 
            help="True if using GIN_SENDNetwork, False otherwise")
    parser.add_argument('--script_cfg', type=str, default='./config/DEFAULT/DEF_config.json')
    parser.add_argument('--MLP_cfg', type=str, default='./config/DEFAULT/DEF_MLP_cfg.json')
    parser.add_argument('--GIN_cfg', type=str, default='./config/DEFAULT/DEF_GIN_cfg.json')
    parser.add_argument('--filter_cfg', type=str, default='./config/DEFAULT/DEF_filter_cfg.json')
    parser.add_argument('--dataset_load_dir', type=str, default='../data/PROTEINS/pth_PSReg/')
    parser.add_argument('--dataset_split_seed', type=int, default='0')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help="Logging Level, \
                          DEBUG, \
                          INFO, \
                          WARNING, \
                          ERROR, \
                          CRITICAL")
    parser.add_argument('--comment', type=str, help="Experiment comment")
    parser.add_argument('--test', type=str, default='False')
    args = parser.parse_args()


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    
    #3. Dataset
    dataset = PthDataset(load_dir=args.dataset_load_dir)
    data_dict = next(iter(dataset))
    in_channels = data_dict["x"].shape[-1]
    n_bases = data_dict["PSReg"].shape[0]
    print("Dataset in_channels: {}\n Dataset n_bases: {}".format(in_channels, n_bases))

    generator = torch.Generator().manual_seed(args.dataset_split_seed)
    train_dataset, dev_dataset, test_dataset = random_split(
            dataset=dataset, 
            lengths=[0.81, 0.09, 0.1], 
            generator=generator
        )

    #2. Load model
    GIN_experiment = args.GIN_experiment
    script_cfg = get_config(args.script_cfg)
    seed = script_cfg["seed"]
    torch.manual_seed(seed)

    MLP_cfg = get_config(args.MLP_cfg)
    GIN_cfg = get_config(args.GIN_cfg)
    filter_cfg = get_config(args.filter_cfg)
    
    if not GIN_experiment:
        MLP_cfg["in_dim"] = n_bases * in_channels
        model = SENDNetwork(
                    MLP_cfg=MLP_cfg,
                    filter_channels=in_channels,
                    filter_scale=filter_cfg["filter_scale"],
                    filter_knots=torch.load(filter_cfg["filter_knots"]),
                    filter_degree=filter_cfg["filter_degree"],
                    filter_coeffs=None,
                    device=device,
                )
    else:
        in_channels = GIN_cfg["ChannelMLP_cfg"]["num_ChannelMLPs"]
        MLP_cfg["in_dim"] = n_bases * in_channels
        model = GIN_SENDNetwork(
                    GIN_cfg=GIN_cfg,
                    MLP_cfg=MLP_cfg,
                    filter_channels=in_channels,
                    filter_scale=filter_cfg["filter_scale"],
                    filter_knots=torch.load(filter_cfg["filter_knots"]),
                    filter_degree=filter_cfg["filter_degree"],
                    filter_coeffs=None,
                    device=device,
                )
    load_model(model=model, file_name=script_cfg["test"]["test_model"], optimizer=None)
    model.eval()

    # 4. logger
    log_file = '../exp/pyg_SRC/log_{}.txt'.format(datetime.datetime.now())
    logging.basicConfig(level=args.log_level,
                        filename=log_file,
                        filemode='a',)
    logger = logging.getLogger(__name__)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(script_cfg["run_id"]))

    # 5. Runner
    script_cfg["use_gpu"] = script_cfg["use_gpu"] and torch.cuda.is_available()
    
    runner = PTHRunner(model_object=model, script_cfg=script_cfg, logger=logger,
                       train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset)     # R
    runner.test()    


if __name__ == '__main__':
    main()
