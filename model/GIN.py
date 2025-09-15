import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv

from .MLP import MLP


class GIN(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 MLP_cfg: dict,
                 ChannelMLP_cfg: dict,
                 skip_first_features: bool=False,
                 device='cpu'):
        super().__init__()
        self.skip_first_features = skip_first_features
        self.convs = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.ChannelMLPs = None
        self.out_channels = 0 if skip_first_features else in_channels
        self.out_channels += num_layers * MLP_cfg['out_dim']

        for layer in range(num_layers):
            if layer == 0:
                local_in_channels = in_channels
            else:
                local_in_channels = MLP_cfg['out_dim']    # TODO: validate

            MLP_ = MLP(in_dim=local_in_channels,
                      out_dim=MLP_cfg['out_dim'],
                      hid_dim=MLP_cfg['hid_dim'],
                      num_hid=MLP_cfg['num_hid'],
                      dp_cfg=MLP_cfg['dp_cfg'],
                      bn_cfg=MLP_cfg['bn_cfg'],
                      output_activation='relu', device=device)  # mod call
            GIN_layer = GINConv(MLP_, eps=0., train_eps=False).to(device)
            self.convs.append(GIN_layer)

        if ChannelMLP_cfg is not None:
            self.ChannelMLPs = torch.nn.ModuleList()
            self.num_ChannelMLPs = ChannelMLP_cfg['num_ChannelMLPs']

            for _ in range(self.num_ChannelMLPs):
                MLP_ = MLP(in_dim=self.out_channels,
                           out_dim=1,
                           hid_dim=ChannelMLP_cfg['hid_dim'],
                           num_hid=ChannelMLP_cfg['num_hid'],
                           dp_cfg=ChannelMLP_cfg['dp_cfg'],
                           bn_cfg=ChannelMLP_cfg['bn_cfg'],
                           output_activation='relu', device=device)
                self.ChannelMLPs.append(MLP_)

    def forward(self, x, edge_index):

        x_cat = []
        if not self.skip_first_features:
            x_cat.append(x)

        x = self.convs[0](x, edge_index)
        x_cat.append(x)

        for layer in range(1, self.num_layers):
            x = self.convs[layer](x, edge_index)
            x_cat.append(x)
        x = torch.cat(x_cat, dim=1)     # N_batch x C

        if self.ChannelMLPs is not None:
            x_cat = []

            for i in range(self.num_ChannelMLPs):
                x_i = self.ChannelMLPs[i](x)    # N_batch x 1
                x_cat.append(x_i)
            x = torch.cat(x_cat, dim=1)

        return x
