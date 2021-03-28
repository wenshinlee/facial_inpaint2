import torch
import math
from abc import ABC

import torch.nn as nn
import torch.nn.functional as F
import core.utils as utils
import functools
import numpy as np


#################################
#           Function
#################################
def define_inpaint_gen(hy, gpu_ids, init_type, init_gain):
    inpaint_gen = Gen(hy)
    return utils.init_net(inpaint_gen, init_type, init_gain, gpu_ids)


def define_semantic_gan(hy, gpu_ids, init_type, init_gain):
    semantic_gan = SemanticExtractNet(hy['input_dim'] + 1, hy['num_semantic_label'])
    return utils.init_net(semantic_gan, init_type, init_gain, gpu_ids)


def define_MultiscaleDis(hy, input_nc, output_nc, gpu_ids, init_type, init_gain):
    multi_scale_dis = MultiscaleDiscriminator(input_nc, output_nc, hy)
    return utils.init_net(multi_scale_dis, init_type, init_gain, gpu_ids)


#################################
#           Block
#################################
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


class DownBlock(nn.Module, ABC):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(2e-1, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.avg_pool2d(self.sc(x), 2)
        out = self.conv2(self.activ(F.avg_pool2d(self.conv1(self.activ(x.clone())), 2)))
        out = residual + out
        return out / math.sqrt(2)


class InstanceNorm2d(nn.Module, ABC):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class DownBlockIN(nn.Module, ABC):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        # use nn.InstanceNorm2d(in_dim, affine=True) if you want.
        self.in1 = nn.InstanceNorm2d(in_dim, affine=True)
        self.in2 = nn.InstanceNorm2d(in_dim, affine=True)

        self.activ = nn.LeakyReLU(2e-1, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.avg_pool2d(self.sc(x), 2)
        out = self.conv2(self.activ(self.in2(F.avg_pool2d(self.conv1(self.activ(self.in1(x.clone()))), 2))))
        out = residual + out
        return out / math.sqrt(2)


class LinearBlock(nn.Module, ABC):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.linear(self.activ(x))


class UpBlock(nn.Module, ABC):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(2e-1, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.interpolate(self.sc(x), scale_factor=2, mode='nearest')
        out = self.conv2(self.activ(self.conv1(F.interpolate(self.activ(x.clone()), scale_factor=2, mode='nearest'))))
        out = residual + out
        return out / math.sqrt(2)


class UpBlockIN(nn.Module, ABC):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.in1 = nn.InstanceNorm2d(in_dim, affine=True)
        self.in2 = nn.InstanceNorm2d(in_dim // 2, affine=True)

        self.activ = nn.LeakyReLU(2e-1, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.interpolate(self.sc(x), scale_factor=2, mode='nearest')
        y = F.interpolate(self.activ(self.in1(x.clone())), scale_factor=2, mode='nearest')
        z = self.conv1(y)
        out = self.conv2(self.activ(self.in2(z)))
        out = residual + out
        return out / math.sqrt(2)


class FixBlockIN(nn.Module, ABC):
    def __init__(self, in_dim, out_dim):
        super(FixBlockIN, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.in1 = nn.InstanceNorm2d(in_dim, affine=True)
        self.in2 = nn.InstanceNorm2d(in_dim, affine=True)

        self.activ = nn.LeakyReLU(2e-1, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = self.sc(x)
        out = self.conv2(self.activ(self.in2(self.conv1(self.activ(self.in1(x.clone()))))))
        out = residual + out
        return out / math.sqrt(2)


#################################
#           Module
#################################
class StyleAttentionExtractor(nn.Module, ABC):
    def __init__(self, hyperparameters, num_segmap_attentions):
        super(StyleAttentionExtractor, self).__init__()
        self.num_segmap_attentions = num_segmap_attentions
        self.linear_dim = hyperparameters['extractors_gen']['channels'][-1]
        self.linear_block = nn.Sequential(
            *[LinearBlock(self.linear_dim, self.linear_dim) for _ in range(num_segmap_attentions)]
        )

    def forward(self, x, segmap_attentions):
        batch_size_codes = x.shape[0]
        channel_codes = x.shape[1]
        segmap_attentions = F.interpolate(segmap_attentions, size=x.size()[2:], mode='nearest')
        codes_vector = torch.zeros((batch_size_codes, self.num_segmap_attentions, channel_codes),
                                   dtype=x.dtype, device=x.device)

        for i in range(batch_size_codes):
            for j in range(self.num_segmap_attentions):
                component_mask_area = torch.sum(segmap_attentions.bool()[i, j])
                if component_mask_area > 0:
                    codes_component_feature = x[i].masked_select(segmap_attentions.bool()[i, j]) \
                        .reshape(channel_codes, component_mask_area).mean(1)
                    codes_vector[i][j] = self.linear_block[j](codes_component_feature)

        return codes_vector


class GatedConv2dSpade(nn.Module, ABC):
    def __init__(self, out_dim, in_dim, n_hidden=128):
        super(GatedConv2dSpade, self).__init__()

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(in_dim, n_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(2e-1, inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(n_hidden, out_dim, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(n_hidden, out_dim, kernel_size=3, padding=1)

    def forward(self, mask):
        out_mask = self.mlp_shared(mask)
        gamma_gated = self.mlp_gamma(out_mask)
        beta_gated = self.mlp_beta(out_mask)

        return gamma_gated, beta_gated


class SemanticRegionMaskGuideNorm2d(nn.Module, ABC):
    def __init__(self, style_dim, in_dim):
        super(SemanticRegionMaskGuideNorm2d, self).__init__()
        self.style_dim = style_dim
        self.noise_var = nn.Parameter(torch.zeros(in_dim), requires_grad=True)

        # use nn.InstanceNorm2d(in_dim, affine=True) if you want.
        self.in_norm2d = nn.InstanceNorm2d(in_dim, affine=True)

        self.gated_conv2d = GatedConv2dSpade(out_dim=in_dim, in_dim=3)

        self.conv_gamma = nn.Conv2d(self.style_dim, in_dim, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(self.style_dim, in_dim, kernel_size=3, padding=1)
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, segmap_attentions, codes_vector, mask):
        # Part 1. generate parameter-free normalized activations
        added_noise = (torch.randn(x.shape[0], x.shape[3], x.shape[2], 1, dtype=x.dtype, device=x.device)
                       * self.noise_var).transpose(1, 3)
        x_norm = self.in_norm2d(x + added_noise)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap_attentions = F.interpolate(segmap_attentions, size=x.size()[2:], mode='nearest')
        mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')

        # Part 2.1 generate StyleMatrix
        [b_size, _, h_size, w_size] = x_norm.shape
        style_matrix = torch.zeros((b_size, self.style_dim, h_size, w_size), device=x.device)
        for i in range(b_size):
            for j in range(codes_vector.shape[1]):
                component_mask_area = torch.sum(segmap_attentions.bool()[i, j])
                if component_mask_area > 0:
                    component_mu = codes_vector[i][j].reshape(self.style_dim, 1) \
                        .expand(self.style_dim, component_mask_area)
                    style_matrix[i].masked_scatter_(segmap_attentions.bool()[i, j], component_mu)

        gamma_style = self.conv_gamma(style_matrix)
        beta_style = self.conv_beta(style_matrix)

        gamma_gated, beta_gated = self.gated_conv2d(mask)
        gamma_alpha = torch.sigmoid(self.blending_gamma)
        beta_alpha = torch.sigmoid(self.blending_beta)

        gamma_final = gamma_alpha * gamma_style + (1 - gamma_alpha) * gamma_gated
        beta_final = beta_alpha * beta_style + (1 - beta_alpha) * beta_gated
        out = x_norm * (1 + gamma_final) + beta_final

        return out


class SemanticRegionMaskGuideResBlock(nn.Module, ABC):
    def __init__(self, hyperparameters, idx_channels):
        super(SemanticRegionMaskGuideResBlock, self).__init__()
        is_spectral_norm = hyperparameters['spectral_norm']
        style_dim = hyperparameters['style_dim']

        channels = hyperparameters['decoder']['channels']
        in_dim, out_dim = channels[idx_channels], channels[idx_channels + 1]

        self.learned_skip = (in_dim != out_dim)
        middle_dim = min(in_dim, out_dim)

        self.conv_0 = nn.Conv2d(in_dim, middle_dim, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_dim, out_dim, kernel_size=3, padding=1)
        self.srmg_norm2d_0 = SemanticRegionMaskGuideNorm2d(style_dim, in_dim)
        self.srmg_norm2d_1 = SemanticRegionMaskGuideNorm2d(style_dim, middle_dim)
        self.in_norm2d_0 = nn.InstanceNorm2d(in_dim, affine=True)
        self.in_norm2d_1 = nn.InstanceNorm2d(middle_dim, affine=True)

        if self.learned_skip:
            self.conv_s = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
            self.srmg_norm2d_s = SemanticRegionMaskGuideNorm2d(style_dim, in_dim)

        if is_spectral_norm:
            self.conv_0 = nn.utils.spectral_norm(self.conv_0)
            self.conv_1 = nn.utils.spectral_norm(self.conv_1)
            if self.learned_skip:
                self.conv_s = nn.utils.spectral_norm(self.conv_s)

        self.activ = nn.LeakyReLU(2e-1, inplace=True)

    def forward(self, x, semantic_region, codes_vector, mask):
        if self.learned_skip:
            x_s = F.interpolate(self.conv_s(x), scale_factor=2, mode='nearest')
        else:
            x_s = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.srmg_norm2d_0(x, semantic_region, codes_vector, mask)
        x = self.conv_0(self.activ(self.in_norm2d_0(x)))

        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.srmg_norm2d_1(x, semantic_region, codes_vector, mask)
        x = self.conv_1(self.activ(self.in_norm2d_1(x)))
        out = x_s + x

        return out


class SemanticExtractNet(nn.Module, ABC):
    def __init__(self, input_class, output_class, block=ResnetBlock):
        super().__init__()
        self.resnet_initial_kernel_size = 7
        self.resnet_n_blocks = 3

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
        x = self.down(x)
        x = self.bottle_neck(x)
        x = self.up(x)
        out = self.out(x)
        return out


#################################
#           Generator
#################################
class Gen(nn.Module, ABC):
    def __init__(self, hy):
        super(Gen, self).__init__()
        num_semantic_regions, _, _ = utils.get_num_semantic(hy['tags'])

        channels = hy['encoder']['share_channels']
        self.share_encoder = nn.Sequential(
            nn.Conv2d(hy['input_dim'] + 1, channels[0], 1, 1, 0),
            *[DownBlockIN(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        channels = hy['encoder']['middle_channels']
        self.middle_encode = nn.Sequential(
            *[DownBlockIN(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        channels = hy['extractors_gen']['channels']
        extractors_gen_list = []
        for i in range(len(channels) - 1):
            if i < 512:
                extractors_gen_list.append(DownBlockIN(channels[i], channels[i + 1]))
            else:
                extractors_gen_list.append(UpBlockIN(channels[i], channels[i + 1]))

        self.extractors_gen = nn.Sequential(*extractors_gen_list)

        self.extractors = StyleAttentionExtractor(hy, num_semantic_regions)

        channels = hy['decoder']['channels']
        self.decode = nn.ModuleList([SemanticRegionMaskGuideResBlock(hy, idx_channels)
                                     for idx_channels in range(len(channels) - 1)])

        self.conv_img = nn.Conv2d(channels[-1], hy['input_dim'], 3, stride=1, padding=1)

    def forward(self, x, semantic_regions, mask):
        x_share = self.share_encoder(x)
        x_middle = self.middle_encode(x_share)
        codes_vector = self.extractors(self.extractors_gen(x_share), semantic_regions)
        for i in range(len(self.decode)):
            x_middle = self.decode[i](x_middle, semantic_regions, codes_vector, mask)
        out = torch.tanh(self.conv_img(x_middle))
        return out


#################################
#           Discriminator
#################################
def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class DiscriminatorBlock(nn.Module, ABC):
    def __init__(self, in_dim, out_dim, kw, padw, stride=2, use_bias=False, use_spectral_norm=True):
        super().__init__()

        self.sequence = nn.Sequential(*[
            spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size=kw, stride=stride, padding=padw, bias=use_bias),
                          use_spectral_norm),
            nn.LeakyReLU(2e-1, inplace=True)
        ])

    def forward(self, x):
        return self.sequence(x)


class NLayerDiscriminator(nn.Module, ABC):
    def __init__(self, input_nc, output_nc, hyperparameters, get_inter_feat=False):
        super(NLayerDiscriminator, self).__init__()

        use_sigmoid = hyperparameters['NLayerDis']['use_sigmoid']
        use_spectral_norm = hyperparameters['NLayerDis']['use_spectral_norm']
        norm_layer = hyperparameters['NLayerDis']['norm_layer']
        channels = hyperparameters['NLayerDis']['channels']

        self.get_inter_feat = get_inter_feat

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        stride = 2
        self.dis_block = [
            DiscriminatorBlock(input_nc, channels[0], kw, padw, stride, use_bias, use_spectral_norm)
        ]
        self.dis_block += [
            DiscriminatorBlock(channels[i], channels[i + 1], kw, padw, stride, use_bias, use_spectral_norm)
            for i in range(len(channels) - 1)
        ]
        self.dis_block += [
            nn.Conv2d(channels[-1], output_nc, kernel_size=kw, stride=stride, padding=padw, bias=False)
        ]

        if use_sigmoid:
            self.dis_block += [
                nn.Sigmoid()
            ]
        if self.get_inter_feat:
            for i in range(len(self.dis_block)):
                setattr(self, 'model_' + str(i), nn.Sequential(self.dis_block[i]))
        else:
            self.model = nn.Sequential(*self.dis_block)

    def forward(self, x):
        if self.get_inter_feat:
            res = [x]
            for i in range(len(self.dis_block)):
                model = getattr(self, 'model_' + str(i))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(x)


class MultiscaleDiscriminator(nn.Module, ABC):
    def __init__(self, input_nc, output_nc, hyperparameters):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_dis = hyperparameters['MultiscaleDis']['multiscale']
        self.get_inter_feat = hyperparameters['MultiscaleDis']['get_inter_feat']

        for i in range(self.num_dis):
            net_dis = NLayerDiscriminator(input_nc, output_nc, hyperparameters, self.get_inter_feat)
            setattr(self, 'dis_scale_{}'.format(str(i)), net_dis)

        self.down_sample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        result = []
        for i in range(self.num_dis):
            dis_scale_model = getattr(self, 'dis_scale_{}'.format(str(i)))
            dis_res = dis_scale_model(x)
            result.append(dis_res)

        return result


if __name__ == '__main__':
    hy = utils.get_config('../configs/celeba-hq.yaml')
    g = Gen(hy)
    x = torch.randn(2, 3, 256, 256)
    mask = torch.randn(2, 3, 256, 256)
    semantic_regions = torch.randn(2, 19, 256, 256)
    g(x, semantic_regions, mask)
