from abc import ABC

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class L1ReconLoss(torch.nn.Module, ABC):
    """
    L1 Reconstruction loss for two image
    """

    def __init__(self, weight=1):
        super(L1ReconLoss, self).__init__()
        self.weight = weight

    def forward(self, imgs, recon_imgs, masks=None):
        if masks is None:
            return self.weight * torch.mean(torch.abs(imgs - recon_imgs))
        else:
            return self.weight * torch.mean(
                torch.abs(imgs - recon_imgs) / masks.view(masks.size(0), -1).mean(1).view(-1, 1, 1, 1))


class VGG16(torch.nn.Module, ABC):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.max3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])
        for x in range(16, 17):
            self.max3.add_module(str(x), features[x])

        for x in range(17, 19):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(19, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        max_3 = self.max3(relu3_3)

        relu4_1 = self.relu4_1(max_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_1(relu5_1)
        relu5_3 = self.relu5_1(relu5_2)
        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'max_3': max_3,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        return out


class PerceptualLoss(nn.Module, ABC):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=None):
        super(PerceptualLoss, self).__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.add_module('vgg', VGG16().cuda())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


class StyleLoss(nn.Module, ABC):
    """
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weight=1):
        super(StyleLoss, self).__init__()
        self.weight = weight
        self.add_module('vgg', VGG16().cuda())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_3']), self.compute_gram(y_vgg['relu3_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_3']), self.compute_gram(y_vgg['relu4_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return self.weight * style_loss


class TVLoss(nn.Module, ABC):
    """
    TV loss
    """

    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


#################################
#           GAN LOSS
#################################
class RelativisticLoss(nn.Module, ABC):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, real, fake, is_disc):
        ones = torch.ones_like(fake)
        if is_disc:
            return self.weight * (self.loss((real - fake), ones))
        else:
            return self.weight * (self.loss((fake - real), ones))


class RelativisticAverageLoss(nn.Module, ABC):
    def __init__(self, weight=1):
        super(RelativisticAverageLoss, self).__init__()
        self.weight = weight
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, real, fake, is_disc):
        avg_real = torch.mean(real)
        avg_fake = torch.mean(fake)
        ones = torch.ones_like(fake)
        zeros = torch.zeros_like(fake)
        if is_disc:
            return self.weight * (self.loss((real - avg_fake), ones) + self.loss((fake - avg_real), zeros))
        else:
            return self.weight * (self.loss((real - avg_fake), zeros) + self.loss((fake - avg_real), ones))


class MultiScaleRelativisticAverageLoss(nn.Module, ABC):
    def __init__(self, get_inter_feat=False):
        super(MultiScaleRelativisticAverageLoss, self).__init__()
        self.get_inter_feat = get_inter_feat
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, real, fake, is_disc):
        if isinstance(real, list) and isinstance(fake, list):  # MultiScaleDis
            mult_loss = 0
            num_dis = len(real)
            if self.get_inter_feat:
                pass
            else:
                for i in range(num_dis):
                    avg_real = torch.mean(real[i])
                    avg_fake = torch.mean(fake[i])
                    ones = torch.ones_like(fake[i])
                    zeros = torch.zeros_like(fake[i])
                    if is_disc:
                        mult_loss += (self.loss((real[i] - avg_fake), ones) + self.loss((fake[i] - avg_real), zeros))
                    else:
                        mult_loss += (self.loss((real[i] - avg_fake), zeros) + self.loss((fake[i] - avg_real), ones))
            return mult_loss
        else:  # standard dis
            avg_real = torch.mean(real)
            avg_fake = torch.mean(fake)
            ones = torch.ones_like(fake)
            zeros = torch.zeros_like(fake)
            if is_disc:
                return self.loss((real - avg_fake), ones) + self.loss((fake - avg_real), zeros)
            else:
                return self.loss((real - avg_fake), zeros) + self.loss((fake - avg_real), ones)


class GANFeatMatchingLoss(nn.Module, ABC):
    def __init__(self, criterionFeat=nn.L1Loss):
        super(GANFeatMatchingLoss, self).__init__()
        self.criterionFeat = criterionFeat()

    def forward(self, pred_real, pred_fake):
        num_dis = len(pred_real)
        num_match_layers = len(pred_real[0])
        loss_FeatMatch = 0
        for i in range(num_dis):
            for j in range(num_match_layers - 1):
                loss_FeatMatch += self.criterionFeat(pred_fake[i][j], pred_real[i][j])
        return loss_FeatMatch


#################################
#           SEGMAP_ATTENTION
#################################
class SegmapGuideAttentionLoss(nn.Module, ABC):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.CE = nn.CrossEntropyLoss()

    def forward(self, pred_group_attention, segmap):
        n_segmap = segmap.shape[1]
        n_pred_group_attention = pred_group_attention.shape[1]
        num_group = n_pred_group_attention // n_segmap
        ce_loss = 0
        for i in range(0, n_pred_group_attention, num_group):
            ce_loss += self.CE(pred_group_attention[:, i:i + num_group, :, :], segmap[:, i // num_group, :, :].long())

        return self.weight * (ce_loss / n_segmap)


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode

        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255, is_softmax=False):
        super(CrossEntropyLoss2d, self).__init__()
        self.is_softmax = is_softmax
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        if torch.sum(torch.isnan(inputs)) > 0:
            print("NAN!!!!!!!")
            import sys
            sys.exit(0)
        if self.is_softmax:
            input_log = torch.log(inputs)
            out = self.nll_loss(input_log, targets)
            return out
        else:
            return self.nll_loss(F.log_softmax(inputs), targets)
