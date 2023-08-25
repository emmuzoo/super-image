import torch
import torch.nn as nn
import torch.fft as fft

from .configuration_edsrffc import EdsrffcConfig
from ...modeling_utils import (
    default_conv,
    MeanShift,
    Upsampler,
    PreTrainedModel
)



class SpatialTransform(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: int = 3, 
                 bias: bool = False):
        super(SpatialTransform, self).__init__()

        self.body = nn.Sequential(*[
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=kernel_size, 
                      padding=(kernel_size - 1) // 2, 
                      bias=bias),  # 3x3 convolution
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Conv2d(out_channels, 
                      out_channels, 
                      kernel_size=kernel_size, 
                      padding=(kernel_size - 1) // 2, 
                      bias=bias),  # 3x3 convolution
        ])

    def forward(self, x):
        res = self.body(x)
        res += x

        return res
    
class FourierUnit(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 groups: int=1,
                 fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        
        self.conv_layer = nn.Conv2d(in_channels=in_channels * 2, 
                      out_channels=out_channels * 2, 
                      kernel_size=1, 
                      groups=groups,
                      padding=0, 
                      bias=False)  # 3x3 convolution
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        #ffted = fft.rfft(x, signal_ndim=2, normalized=True)
        fft_dim = (-2, -1)
        ffted = fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        #ffted = self.relu(self.bn(ffted))
        ffted = self.relu(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        #output = fft.irfft(ffted, signal_ndim=2,
        #                   signal_sizes=r_size[2:], normalized=True)
        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output

class FrequencyTrasform(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 groups: int = 1,
                 bias: bool = False):
        super(FrequencyTrasform, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels // 2, 
                      kernel_size=1, 
                      groups=groups, 
                      bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True) 
        )

        self.fu = FourierUnit(out_channels // 2, 
                              out_channels // 2,
                              groups = groups)
        
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=bias)  # 1x1 convolution nf->nf

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)

        output = self.conv2(x + output)
        
        return output


class SFB(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: int = 3, 
                 bias: bool = False):
        super(SFB, self).__init__()
        
        self.spatial = SpatialTransform(in_channels, 
                                        out_channels, 
                                        kernel_size=kernel_size,
                                        bias=bias)
        
        self.frequency = FrequencyTrasform(in_channels, 
                                           out_channels, 
                                           bias=bias)

        self.conv1 = nn.Conv2d(out_channels * 2, 
                               out_channels, 
                               kernel_size = 1, 
                               bias = bias)  # 1x1 convolution nf->nf

    def forward(self, x):
        x_spatial = self.spatial(x)
        x_transform = self.spatial(x)

        # Conv 1x1: x_spatial || x_transform
        output = self.conv1(torch.cat((x_spatial, x_transform), dim=1))

        return output 
    



class FFCResBlock(nn.Module):
    def __init__(
            self, 
            n_feats, 
            kernel_size,
            bias=True, 
            bn=False, 
            act=nn.ReLU(True), 
            res_scale=1):

        super(FFCResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(SFB(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EdsrffcModel(PreTrainedModel):
    config_class = EdsrffcConfig

    def __init__(self, args, conv=default_conv):
        super(EdsrffcModel, self).__init__(args)

        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        n_colors = args.n_colors
        kernel_size = 3
        scale = args.scale
        rgb_range = args.rgb_range
        act = nn.ReLU(True)
        self.sub_mean = MeanShift(rgb_range, rgb_mean=args.rgb_mean, rgb_std=args.rgb_std)  # standardize input
        self.add_mean = MeanShift(rgb_range, sign=1, rgb_mean=args.rgb_mean, rgb_std=args.rgb_std)  # restore output

        # define head module, channels: 3->64
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module, channels: 64->64
        m_body = [
            FFCResBlock(
                n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(SFB(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

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

        res = self.body(x)
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
