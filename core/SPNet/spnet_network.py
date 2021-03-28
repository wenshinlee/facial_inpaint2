from abc import ABC
import core.utils as utils
import torch.nn as nn
import torch
from core.networks import define_MultiscaleDis


class ResnetBlock(nn.Module, ABC):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            norm_layer(dim),
            activation,
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            norm_layer(dim),
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


class SPNet(nn.Module, ABC):
    def __init__(self, input_class, output_class, block=ResnetBlock):
        super().__init__()
        self.resnet_initial_kernel_size = 7
        self.resnet_n_blocks = 9

        ngf = 64
        activation = nn.ReLU(False)

        self.down = nn.Sequential(
            nn.ReflectionPad2d(self.resnet_initial_kernel_size // 2),
            nn.Conv2d(input_class, ngf, kernel_size=self.resnet_initial_kernel_size, stride=2, padding=0),
            nn.BatchNorm2d(ngf),
            activation,

            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            activation,

            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            activation,

            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            activation,
        )

        # resnet blocks
        resnet_blocks = []
        for i in range(self.resnet_n_blocks):
            resnet_blocks += [block(ngf * 8, norm_layer=nn.BatchNorm2d, kernel_size=3)]
        self.bottle_neck = nn.Sequential(*resnet_blocks)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf * 8),
            activation,

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf * 4),
            activation,

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf * 2),
            activation,

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf),
            activation,
        )

        self.out = nn.Sequential(
            nn.ReflectionPad2d(self.resnet_initial_kernel_size // 2),
            nn.Conv2d(ngf, output_class, kernel_size=7, padding=0),
            nn.Softmax2d()
        )

    def forward(self, x):
        print("")
        print("x", torch.sum(torch.isnan(x)))
        x = self.down(x)
        x = self.bottle_neck(x)
        x = self.up(x)
        out = self.out(x)
        print("out", torch.sum(torch.isnan(out)))
        return out


def define_SPNet(hy):
    gpu_ids = hy['gpu_ids']
    init_type = hy['init_type']
    init_gain = hy['init_gain']
    n_label = hy['num_segmap_label']
    n_image = hy['input_dim']
    # + n_label
    gen = SPNet(n_image, n_label)
    return utils.init_net(gen, init_type, init_gain, gpu_ids)


def define_SPNet_Dis(input_nc, output_nc, hy):
    return define_MultiscaleDis(input_nc, output_nc, hy=hy)
