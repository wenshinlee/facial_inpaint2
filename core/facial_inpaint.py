import os
import cv2
import torch
import core.networks as networks
import core.loss as loss
import core.utils as utils
import numpy as np

from abc import ABC
from core.base_inpaint import BaseInpaint


class FacialInpaint(BaseInpaint, ABC):
    def __init__(self, hy):
        super(FacialInpaint, self).__init__(hy)
        # define gen net
        self.inpaint_gen = networks.define_inpaint_gen(hy, self.gpu_ids, self.init_type, self.init_gain)
        self.semantic_gan = networks.define_semantic_gan(hy, self.gpu_ids, self.init_type, self.init_gain)
        self.model_names.append('inpaint_gen')
        self.model_names.append('semantic_gan')

        self.inpaint_dis = networks.define_MultiscaleDis(hy, self.input_dim, self.mult_dis_para['output_nc'],
                                                         self.gpu_ids, self.init_type, self.init_gain)
        self.semantic_dis = networks.define_MultiscaleDis(hy, self.num_semantic_label, self.mult_dis_para['output_nc'],
                                                          self.gpu_ids, self.init_type, self.init_gain)
        self.model_names.append('inpaint_dis')
        self.model_names.append('semantic_dis')

        print('---------- Networks initialized -------------')
        self.print_network()

        if self.is_train:
            self.init_optimizer()

        if self.is_train:
            self.criterionGAN = loss.MultiScaleRelativisticAverageLoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.PerceptualLoss = loss.PerceptualLoss()
            self.StyleLoss = loss.StyleLoss()
            # segmap
            self.SegmapLoss = loss.MultiScaleRelativisticAverageLoss()

    def set_input(self, inputs):
        file_name, image_gt, mask, semantic_label, segmap_one_hot, edge = inputs

        self.file_name = file_name
        self.image_gt = image_gt.to(self.device)
        self.mask = mask.to(self.device)
        self.semantic_label = semantic_label.to(self.device)
        self.semantic_one_hot = segmap_one_hot.to(self.device)
        self.edge = edge.to(self.device)

        # target(just for Cross entropy Loss)
        self.target_semantic = self.semantic_label[:, 0, :, :]

        # mask 0 is hole
        self.inv_ex_mask = torch.add(torch.neg(self.mask.float()), 1).float()

        # input
        self.input_edge = self.edge * self.mask[:, 0:1, :, :]
        self.input_segmap = self.semantic_one_hot * self.inv_ex_mask[:, 0:1, :, :]
        # Do not set the mask regions as 0
        self.input_image = image_gt.to(self.device)
        self.input_image.narrow(1, 0, 1).masked_fill_(self.mask.narrow(1, 0, 1).bool(), 2 * 123.0 / 255.0 - 1.0)
        self.input_image.narrow(1, 1, 1).masked_fill_(self.mask.narrow(1, 0, 1).bool(), 2 * 104.0 / 255.0 - 1.0)
        self.input_image.narrow(1, 2, 1).masked_fill_(self.mask.narrow(1, 0, 1).bool(), 2 * 117.0 / 255.0 - 1.0)

    def forward(self, mode='semantic_net'):
        net_input = torch.cat([self.input_image, self.input_edge], dim=1)
        if mode == 'semantic_net':
            self.fake_semantic = self.semantic_gan(net_input)
        elif mode == 'inpaint_net':
            self.fake_image = self.inpaint_gen(net_input, self.fake_semantic.detach(), self.mask)
        else:
            raise NotImplementedError

    def backward_semantic_dis(self):
        real = self.semantic_one_hot
        fake = self.fake_semantic
        # gan loss
        pred_fake = self.semantic_dis(fake.detach())
        pred_real = self.semantic_dis(real)
        self.loss_sd_semantic = self.SegmapLoss(pred_real, pred_fake, True)
        # backward
        self.loss_sd_semantic.backward()

    def backward_semantic_gen(self):
        real = self.semantic_one_hot
        fake = self.fake_semantic
        # gan loss
        pred_fake = self.semantic_dis(fake)
        pred_real = self.semantic_dis(real)
        self.loss_sg_semantic = self.SegmapLoss(pred_real, pred_fake, False)
        # L1 loss
        self.loss_sg_semantic_L1 = self.criterionL1(fake, real)
        # total loss
        self.loss_sg = self.loss_sg_semantic * 0.1 + self.loss_sg_semantic_L1 * 100
        # backward
        self.loss_sg.backward()

    def backward_inpaint_dis(self):
        real = self.image_gt
        fake = self.fake_image
        # # gan loss
        pred_fake = self.inpaint_dis(fake.detach())
        pred_real = self.inpaint_dis(real)
        self.loss_id_gan = self.criterionGAN(pred_real, pred_fake, True)
        # backward
        self.loss_id_gan.backward()

    def backward_inpaint_gen(self):
        real = self.image_gt
        fake = self.fake_image

        # Reconstruction loss, style loss, L1 loss
        self.loss_ig_L1 = self.criterionL1(fake, real)
        self.loss_ig_perceptual = self.PerceptualLoss(fake, real)
        self.loss_ig_style = self.StyleLoss(fake, real)
        # gan loss
        pred_fake = self.inpaint_dis(fake)
        pred_real = self.inpaint_dis(real)
        self.loss_ig_gan = self.criterionGAN(pred_real, pred_fake, False)
        # total loss
        self.loss_ig = self.loss_ig_L1 * 1 + self.loss_ig_perceptual * 0.2 + self.loss_ig_style * 250 \
                       + self.loss_ig_gan
        # backward
        self.loss_ig.backward()

    def optimize_parameters(self, epoch):
        if self.is_train:
            self.forward(mode='semantic_net')
            self.optimize_semantic_net()
            if epoch > self.num_train_semantic_net:
                self.forward(mode='inpaint_net')
                self.optimize_inpaint_net()
        else:
            self.test()

    def optimize_semantic_net(self):
        # Optimize the SemanticExtractNet Dis
        self.set_requires_grad(self.semantic_gan, False)
        self.set_requires_grad(self.semantic_dis, True)
        self.optimizer_semantic_dis.zero_grad()
        self.backward_semantic_dis()
        self.optimizer_semantic_dis.step()

        # Optimize the SemanticExtractNet G
        self.set_requires_grad(self.semantic_gan, True)
        self.set_requires_grad(self.semantic_dis, False)
        self.optimizer_semantic_gan.zero_grad()
        self.backward_semantic_gen()
        self.optimizer_semantic_gan.step()

    def optimize_inpaint_net(self):
        # Optimize the Gen Dis
        self.set_requires_grad(self.inpaint_gen, False)
        self.set_requires_grad(self.inpaint_dis, True)
        self.optimizer_inpaint_dis.zero_grad()
        self.backward_inpaint_dis()
        self.optimizer_inpaint_dis.step()

        # Optimize the Gen G
        self.set_requires_grad(self.inpaint_gen, True)
        self.set_requires_grad(self.inpaint_dis, False)
        self.optimizer_inpaint_gen.zero_grad()
        self.backward_inpaint_gen()
        self.optimizer_inpaint_gen.step()

    def get_current_errors(self, epoch):
        loss_dict = {
            # discriminator
            'loss_sd_semantic': self.loss_sd_semantic.data,
            # Generator
            'loss_sg': self.loss_sg.data,
            'loss_sg_semantic': self.loss_sg_semantic.data,
            'loss_sg_semantic_L1': self.loss_sg_semantic_L1.data,
        }
        if epoch > self.num_train_semantic_net:
            # discriminator
            loss_dict['loss_id_gan'] = self.loss_id_gan.data
            # Generator
            loss_dict['loss_ig_L1'] = self.loss_ig_L1.data
            loss_dict['loss_ig_perceptual'] = self.loss_ig_perceptual.data
            loss_dict['loss_ig_style'] = self.loss_ig_style.data
            loss_dict['loss_ig_gan'] = self.loss_ig_gan.data
            loss_dict['loss_ig'] = self.loss_ig.data

        return loss_dict

    def get_current_visuals(self, epoch):
        input_image = (self.input_image.data.cpu() + 1) / 2.0
        gt_image = (self.image_gt.data.cpu() + 1) / 2.0
        mask = self.mask.data.cpu()

        pred_semantic = utils.tensor2label(self.fake_semantic, self.num_semantic_label)
        pred_semantic = pred_semantic.transpose(0, 3, 1, 2) / 255
        pred_semantic = torch.from_numpy(pred_semantic)

        semantic = utils.tensor2label(self.semantic_one_hot, self.num_semantic_label)
        semantic = semantic.transpose(0, 3, 1, 2) / 255
        semantic = torch.from_numpy(semantic)

        if epoch > self.num_train_semantic_net:
            fake_image = (self.fake_image.data.cpu() + 1) / 2.0
            return semantic, pred_semantic, input_image, gt_image, mask, fake_image
        else:
            return semantic, pred_semantic, input_image, gt_image, mask

    def test(self):
        self.forward(mode='semantic_net')
        self.forward(mode='inpaint_net')
        # save
        self.save_image('mask', self.mask.cpu().numpy())
        self.save_image('image_gt', self.image_gt.cpu().numpy())
        self.save_image('input', self.input_image.cpu().numpy())
        # semantic
        real_semantic = utils.tensor2label(self.semantic_one_hot.cpu(), self.num_semantic_label)
        fake_semantic = utils.tensor2label(self.fake_semantic.cpu(), self.num_semantic_label)
        self.save_image('real_semantic', real_semantic, is_hwc=True)
        self.save_image('fake_semantic', fake_semantic, is_hwc=True)

    def save_image(self, image_title, image_tensor, is_hwc=False, is_rgb=True):
        image_np = image_tensor.astype(np.uint8)
        if not is_hwc:
            image_np = image_np.transpose((0, 2, 3, 1)) * 255

        save_path = os.path.join(self.output_dir, image_title)
        if not os.path.exists(save_path):
            utils.mkdirs(save_path)

        if image_np.ndim == 4:
            for b in range(image_np.shape[0]):
                file_name = self.file_name[b]
                if is_rgb:
                    image_np = image_np[b][..., ::-1]
                file_path = os.path.join(save_path, file_name)
                cv2.imwrite(file_path, image_np)
        # Gray image
        elif image_tensor.ndim == 3:
            for b in range(image_np.shape[0]):
                file_name = self.file_name[b]
                if is_rgb:
                    image_np = image_np[..., ::-1]
                file_path = os.path.join(save_path, file_name)
                cv2.imwrite(file_path, image_np)
