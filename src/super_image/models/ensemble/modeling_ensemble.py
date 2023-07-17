import math

import torch
import torch.nn as nn

from ..edsr.modeling_edsr import EdsrModel
from ..mdsr.modeling_mdsr import MdsrModel
from ..msrn.modeling_msrn import MsrnModel
from .configuration_ensemble import EnsembleConfig
from ...modeling_utils import (
    BamBlock,
    MeanShift,
    PreTrainedModel
)


class EnsembleModel(PreTrainedModel):
    config_class = EnsembleConfig

    def __init__(self, args):
        super(EnsembleModel, self).__init__(args)
        self.scale = args.scale
        self.model_msrn = MsrnModel.from_pretrained('eugenesiow/msrn', scale=self.scale)
        for param in self.model_msrn.parameters():
          param.requires_grad = False
        self.model_mdrs = MdsrModel.from_pretrained('eugenesiow/mdsr', scale=self.scale)
        for param in self.model_mdrs.parameters():
          param.requires_grad = False
        self.model_edrs_base = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=self.scale)
        for param in self.model_edrs_base.parameters():
          param.requires_grad = False
        self.conv = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=3, padding='same')


    def forward(self, x):
        x1 = self.model_msrn(x)
        x2 = self.model_mdrs(x)
        x3 = self.model_edrs_base(x)
        xall = torch.cat((x1, x2, x3), dim=1)
        out = self.conv(xall)

        return out