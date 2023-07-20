import math

import torch
import torch.nn as nn

from ..edsr.modeling_edsr import EdsrModel
from ..mdsr.modeling_mdsr import MdsrModel
from ..msrn.modeling_msrn import MsrnModel
from .configuration_eensemble import EensembleConfig
from ...modeling_utils import (
    default_conv,
    BamBlock,
    MeanShift,
    PreTrainedModel
)


class EensembleModel(PreTrainedModel):
    config_class = EensembleConfig

    def __init__(self, args, conv=default_conv):
        super(EensembleModel, self).__init__(args)
        self.scale = args.scale
        n_feats = 3
        n_colors = 3
        kernel_size = 3

        backbone_msrn = MsrnModel.from_pretrained('eugenesiow/msrn', scale=self.scale)
        for param in backbone_msrn.parameters():
          param.requires_grad = False
        
        backbone_mdrs = MdsrModel.from_pretrained('eugenesiow/mdsr', scale=self.scale)
        for param in backbone_mdrs.parameters():
          param.requires_grad = False
        
        backbone_edrs_base = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=self.scale)
        for param in backbone_edrs_base.parameters():
          param.requires_grad = False
        
        #self.conv = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=3, padding='same')
        #self.conv1 = conv(n_feats, n_colors, kernel_size)
        #self.conv2 = conv(n_feats, n_colors, kernel_size)
        #self.conv3 = conv(n_feats, n_colors, kernel_size)
        
        self.net1 = nn.Sequential(list(backbone_msrn.children()))
        self.net1.add_module('conv11', conv(n_feats, n_colors, kernel_size))
        self.net1.add_module('conv12', conv(n_feats, n_colors, kernel_size))

        self.net2 = nn.Sequential(list(backbone_mdrs.children()))
        self.net1.add_module('conv21', conv(n_feats, n_colors, kernel_size))
        self.net1.add_module('conv22', conv(n_feats, n_colors, kernel_size))

        self.net3 = nn.Sequential(list(backbone_edrs_base.children()))
        self.net1.add_module('conv31', conv(n_feats, n_colors, kernel_size))
        self.net1.add_module('conv32', conv(n_feats, n_colors, kernel_size))

    def forward(self, x):
        #x1 = self.conv1(self.model_msrn(x))
        #x2 = self.conv2(self.model_mdrs(x))
        #x3 = self.conv3(self.model_edrs_base(x))
        x1 = self.net1(x)
        x2 = self.net2(x)
        x3 = self.net3(x)
        out = torch.add(x1, x2)
        out = torch.add(out, x3)

        return out