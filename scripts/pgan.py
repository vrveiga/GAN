"""Gerador da Progressive GAN (Karras et al., 2018) pré-treinada no CelebA-HQ 512x512.

Reimplementação mínima, só para inferência, compatível com os pesos publicados pelo
pytorch_GAN_zoo (Facebook Research). Carregamos apenas os tensores (weights_only=True),
sem executar código de terceiros.

Pesos: https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaHQ16_december_s7_i96000-9c72988c.pth
"""
import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = "data/pretrained/pgan_celebahq512.pth"
URL = "https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaHQ16_december_s7_i96000-9c72988c.pth"
NZ = 512


def pixel_norm(x):
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class EqConv(nn.Conv2d):
    """Conv com 'equalized learning rate': saída escalada por sqrt(2/fan_in)."""
    def forward(self, x):
        fan_in = self.weight[0].numel()
        return super().forward(x) * math.sqrt(2.0 / fan_in)


class EqLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x) * math.sqrt(2.0 / self.in_features)


class PGANGenerator(nn.Module):
    def __init__(self, depth0=512, depths=(512, 512, 512, 256, 128, 64, 32)):
        super().__init__()
        self.formatLayer = EqLinear(NZ, 16 * depth0)
        self.groupScale0 = nn.ModuleList([EqConv(depth0, depth0, 3, padding=1)])
        self.scaleLayers = nn.ModuleList()
        self.toRGBLayers = nn.ModuleList([EqConv(depth0, 3, 1)])
        prev = depth0
        for d in depths:
            self.scaleLayers.append(nn.ModuleList([EqConv(prev, d, 3, padding=1), EqConv(d, d, 3, padding=1)]))
            self.toRGBLayers.append(EqConv(d, 3, 1))
            prev = d

    def forward(self, z):
        act = lambda t: F.leaky_relu(t, 0.2)
        x = pixel_norm(z)
        x = act(self.formatLayer(x)).view(z.size(0), -1, 4, 4)
        x = pixel_norm(x)
        for conv in self.groupScale0:
            x = pixel_norm(act(conv(x)))
        for group in self.scaleLayers:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            for conv in group:
                x = pixel_norm(act(conv(x)))
        return self.toRGBLayers[-1](x)  # aprox. em [-1, 1]


def load_pgan(path=WEIGHTS, use_avg=True):
    # O checkpoint guarda um objeto de configuração do repositório original; criamos uma
    # classe vazia com o mesmo nome só para o carregador seguro conseguir lê-lo.
    name = "models.utils.config"
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.BaseConfig = type("BaseConfig", (), {"__module__": name})
        sys.modules[name] = mod
    BaseConfig = sys.modules[name].BaseConfig
    with torch.serialization.safe_globals([BaseConfig]):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    G = PGANGenerator()
    G.load_state_dict(ck["avgG" if use_avg else "netG"])
    return G.eval()
