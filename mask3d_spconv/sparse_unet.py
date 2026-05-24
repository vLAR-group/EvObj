import torch
import torch.nn as nn
import spconv.pytorch as spconv

# from timm.models.layers import trunc_normal_
from collections import OrderedDict
from functools import partial
import os


SPCONV_ALGO = 'auto'    # 'auto', 'implicit_gemm', 'native'
env_spconv_algo = os.environ.get('SPCONV_ALGO')
if env_spconv_algo is not None and env_spconv_algo in ['auto', 'implicit_gemm', 'native']:
    SPCONV_ALGO = env_spconv_algo

algo = None
if SPCONV_ALGO == 'native':
    algo = spconv.ConvAlgo.Native
elif SPCONV_ALGO == 'implicit_gemm':
    algo = spconv.ConvAlgo.MaskImplicitGemm



class BasicBlock(spconv.SparseModule):
    expansion = 1
    def __init__(self, in_channels, embed_channels, stride=1, norm_fn=None, indice_key=None, bias=False):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, embed_channels, kernel_size=1, bias=False),
                norm_fn(embed_channels, momentum=0.02))

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key, algo=algo)

        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key, algo=algo)

        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out



class SpUNetBase(nn.Module):
    channels = (32, 64, 128, 256, 256, 256, 256, 256)
    layers = (2, 2, 2, 2, 2, 2, 2, 2)
    bn_momentum = 0.02

    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        assert len(self.layers) % 2 == 0
        assert len(self.layers) == len(self.channels)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = len(self.layers) // 2

        norm_fn = partial(nn.BatchNorm1d, eps=1e-5, momentum=self.bn_momentum)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem"),
            norm_fn(base_channels, momentum=0.02),
            nn.ReLU())

        enc_channels = base_channels
        dec_channels = self.channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(spconv.SparseSequential(spconv.SparseConv3d(
                        enc_channels,
                        self.channels[s],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}", algo=algo),
                    norm_fn(self.channels[s], momentum=self.bn_momentum),
                    nn.ReLU()))

            self.enc.append(spconv.SparseSequential(OrderedDict([(f"block{i}",
                                block(self.channels[s],
                                    self.channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}"))
                            for i in range(self.layers[s])])))

            # decode num_stages
            self.up.append(spconv.SparseSequential(spconv.SparseInverseConv3d(
                        self.channels[len(self.channels) - s - 2],
                        dec_channels,
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}", algo=algo),
                    norm_fn(dec_channels, momentum=self.bn_momentum),
                    nn.ReLU()))

            self.dec.append(spconv.SparseSequential(OrderedDict([((f"block{i}",
                                    block(dec_channels + enc_channels,
                                        dec_channels,
                                        norm_fn=norm_fn,
                                        indice_key=f"subm{s}")) if i == 0
                                else (f"block{i}", block(dec_channels,
                                        dec_channels,
                                        norm_fn=norm_fn,
                                        indice_key=f"subm{s}")))
                            for i in range(self.layers[len(self.channels) - s - 1])])))

            enc_channels = self.channels[s]
            dec_channels = self.channels[len(self.channels) - s - 2]

        # self.final = spconv.SubMConv3d(96, 1, kernel_size=1, padding=1, bias=True, algo=algo)
        # self.final = nn.Linear(96, 1, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=0.02)
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            # trunc_normal_(m.weight, std=0.02)
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, y):
        feature_maps = []
        x = self.conv_input(y)
        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)
        # dec forward
        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            skip = skips.pop(-1)
            x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
            x = self.dec[s](x)
            feature_maps.append(x)

        return x, feature_maps



class Res16UNet14(SpUNetBase):
    layers = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16UNet18(SpUNetBase):
    layers = (2, 2, 2, 2, 2, 2, 2, 2)


class Res16UNet34(SpUNetBase):
    layers = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet14A(Res16UNet14):
    channels = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet14A2(Res16UNet14A):
    layers = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B(Res16UNet14):
    channels = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet14B2(Res16UNet14B):
    layers = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B3(Res16UNet14B):
    layers = (2, 2, 2, 2, 1, 1, 1, 1)


class Res16UNet14C(Res16UNet14):
    channels = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16UNet14D(Res16UNet14):
    channels = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18A(Res16UNet18):
    channels = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet18B(Res16UNet18):
    channels = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet18D(Res16UNet18):
    channels = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet34A(Res16UNet34):
    channels = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet34B(Res16UNet34):
    channels = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34C(Res16UNet34):
    channels = (32, 64, 128, 256, 256, 128, 96, 96)


class Custom30M(Res16UNet34):
    channels = (32, 64, 128, 256, 128, 64, 64, 32)


class Res16UNet34D(Res16UNet34):
    channels = (32, 64, 128, 256, 256, 128, 96, 128)