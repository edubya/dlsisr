import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
from IPython.core.debugger import set_trace

# vgg19_model = vgg19(pretrained=True)
# VGG/i.j = features from the relu following the jth Conv2d before the ith Maxpool2d

# The 5th max pool layer is layer number #37 (index 36). The 4th conv2d layer preceding it is layer #29 (index 28). So the activation layer is layer #30 (index 29)
# layers = [l for l in vgg19_model.features.children()]
# is_conv = [int(isinstance(l, torch.nn.Conv2d)) for l in layers]
# is_relu = [int(isinstance(l, torch.nn.ReLU)) for l in layers]
# is_maxpool = [int(isinstance(l, torch.nn.MaxPool2d)) for l in layers]
# i = 5
# j = 4
# idx = np.arange(37)
# idx_maxpool = min(idx[np.cumsum(is_maxpool) == i])
# min(idx[(np.cumsum(is_conv) - np.cumsum(is_conv)[idx_maxpool]) == -j+1])+1

class VGG19Grayscale(nn.Module):
    def __init__(self, layer_num):
        super(VGG19Grayscale, self).__init__()
        self.vgg19 = vgg19(pretrained=True)
        self.select_layer(layer_num)

    def select_layer(self, layer_num):
        # layer_num counts from 1, not 0.
        self.layer_num = layer_num
        vgg19_layers = list(self.vgg19.features.children())[:layer_num+1]
        self.layers = nn.Sequential(*vgg19_layers)

    def forward(self, img):
        return self.layers(torch.cat((img, img, img), 1))


class ConvolutionalResidualBlock(nn.Module):
    def __init__(self, k=3, n=64, s=1, p=1):
        super(ConvolutionalResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(n, n, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(n),
            nn.PReLU(),
            nn.Conv2d(n, n, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(n),
        )

    def forward(self, x):
        return x + self.layers(x)

class Upsampler(nn.Module):
    def __init__(self, in_channels, k=3, n=256, s=1, p=1, upscale=2):
        super(Upsampler, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, n, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(n),
            nn.PixelShuffle(upscale),
            nn.PReLU()
            )

    def forward(self, x):
        return self.layers(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_crb=16, n_features=64):
        super(Generator, self).__init__()

        # Pre-convolutional residual blocks
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, n_features, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
            )

        # Residual blocks
        generate_crb = (ConvolutionalResidualBlock(n=n_features) for _ in range(n_crb))
        self.crbs = nn.Sequential(*list(generate_crb))

        # Post-convolutional residual blocks
        self.post = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_features)
            )

        # Trainable upsampling layers
        self.upsample = nn.Sequential(
            Upsampler(n_features),
            Upsampler(n_features)
            )

        # Final Conv2d plus activation
        self.final = nn.Sequential(
            nn.Conv2d(n_features, out_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
            )

    def forward(self, x):
        y = self.pre(x)
        z = self.post(self.crbs(y))
        return self.final(self.upsample(y+z))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, alpha=0.2):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            # (1) Produce 64 features
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            # (2) Reduce resolution by stride=2
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(alpha, inplace=True),
            # (3) Double features
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(alpha, inplace=True),
            # (4) Reduce
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(alpha, inplace=True),
            # (5) Double
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(alpha, inplace=True),
            # (6) Reduce
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(alpha, inplace=True),
            # (7) Double
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(alpha, inplace=True),
            # (8) Reduce
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(alpha, inplace=True)
            )

        # Classifier without sigmoid activation (sigmoid goes in BCE with logits loss function)
        self.final = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        out = self.layers(x)
        out = torch.flatten(out, 1)
        out = self.final(out)
        return out
