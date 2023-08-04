import torch
import torch.nn as nn

from .configuration_eedsr import EedsrConfig
from ...modeling_utils import (
    default_conv,
    BamBlock,
    MeanShift,
    Upsampler,
    PreTrainedModel
)

class DenseResidualBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
            channels (int): The number of channels in the input image.
            growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growths: int, res_scale=0.2, bias=True):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale
        '''
        def block(in_features, filters, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            return nn.Sequential(*layers)

        self.b1 = block(channels + growths * 0, growths)
        self.b2 = block(channels + growths * 1, growths)
        self.b3 = block(channels + growths * 2, growths)
        self.b4 = block(channels + growths * 3, growths)
        self.b5 = block(channels + growths * 4, channels, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
        '''
        self.conv1 = nn.Conv2d(channels, growths, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channels + growths, growths, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channels + 2 * growths, growths, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channels + 3 * growths, growths, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channels + 4 * growths, channels, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        

    def forward(self, x):
        #inputs = x
        #for block in self.blocks:
        #    out = block(inputs)
        #    inputs = torch.cat([inputs, out], 1)
        #return out.mul(self.res_scale) + x
        x1  = self.lrelu(self.conv1(x))
        x2  = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3  = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4  = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growths (int): The number of channels that increase in each layer of convolution.
    """
    def __init__(self, channels: int, growths: int, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(channels, growths, res_scale=res_scale), 
            DenseResidualBlock(channels, growths, res_scale=res_scale), 
            DenseResidualBlock(channels, growths, res_scale=res_scale)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x
    

class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size, bam=False,
            bias=True, bn=None, act=nn.ReLU(True), res_scale=0.2):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn == "batch":
                m.append(nn.BatchNorm2d(n_feats))
            elif bn == "instance":
                m.append(nn.InstanceNorm2d(n_feats, affine=True))
            
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.bam = bam
        if bam:
            self.attention = BamBlock(n_feats)
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EedsrModel(PreTrainedModel):
    config_class = EedsrConfig

    def __init__(self, args, conv=default_conv):
        super(EedsrModel, self).__init__(args)

        self.args = args
        n_resblocks = args.n_resblocks
        bam = args.bam
        n_feats = args.n_feats
        n_growths = args.n_growths
        n_colors = args.n_colors
        kernel_size = 3
        scale = args.scale
        rgb_range = args.rgb_range
        if args.act is None:
            act =  nn.ReLU(True)
        elif args.act == 'ReLU':
            act = nn.ReLU(True)
        elif args.act == 'GeLU':
            act = nn.GeLU()
        elif args.act == 'LeakyReLU':
            act = nn.LeakyReLU()
        else: 
            act =  nn.ReLU(True)

        bn = args.bn
        #act = nn.ReLU(True)
        #act = nn.GeLU()
        #act = nn.LeakyReLU()
        self.sub_mean = MeanShift(rgb_range, rgb_mean=args.rgb_mean, rgb_std=args.rgb_std)  # standardize input
        self.add_mean = MeanShift(rgb_range, sign=1, rgb_mean=args.rgb_mean, rgb_std=args.rgb_std)  # restore output

        # define head module, channels: 3->64
        #m_head = [conv(n_colors, n_feats, kernel_size)]
        self.head = nn.Sequential(*[conv(n_colors, n_feats, kernel_size)])

        # define body module, channels: 64->64
        
        self.rdbbody = nn.Sequential(*[
                ResidualInResidualDenseBlock(
                     n_feats, n_growths, res_scale=0.2
                ) for _ in range(8)
        ])


        self.resbody = nn.Sequential(*[
            ResBlock(
                conv, n_feats, kernel_size, bn=bn, bam=bam, act=act, res_scale=args.res_scale
            #ResidualInResidualDenseBlock(
            #     n_feats, n_growths, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ])
        #m_body.append(conv(n_feats, n_feats, kernel_size))
        #self.body.add_module(str(n_resblocks), conv(n_feats, n_feats, kernel_size))

        if self.args.no_add:
            print("CAT")
            self.conv = conv(n_feats*2, n_feats, kernel_size)
        else:
            print("ADD")
            self.conv = conv(n_feats, n_feats, kernel_size)



        #self.head = nn.Sequential(*m_head)
        #self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        #res = self.body(x)
        rdb = self.rdbbody(x)
        res = self.resbody(x)
        if self.args.no_add:
            res = torch.cat((rdb, res), dim=1)
        else:
            res = rdb + res
        res = self.conv(res)
        res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError(f'While copying the parameter named {name}, '
                                           f'whose dimensions in the model are {own_state[name].size()} and '
                                           f'whose dimensions in the checkpoint are {param.size()}.')
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(f'unexpected key "{name}" in state_dict')
