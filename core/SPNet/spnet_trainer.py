import torch
import core.SPNet.spnet_network as spnet_network
from datasets.dataset import get_dataloader
import core.utils as utils
import core.loss as loss
import os


class SegInpaintModel(torch.nn.Module):
    def __init__(self, hy):
        self.hy = hy
        super(SegInpaintModel, self).__init__()
        self.ByteTensor = torch.cuda.ByteTensor if len(hy['gpu_ids']) > 0 \
            else torch.ByteTensor
        self.FloatTensor = torch.cuda.FloatTensor if len(hy['gpu_ids']) > 0 \
            else torch.FloatTensor

        # GPU
        self.device = torch.device('cuda:{}'.format(hy['gpu_ids'][0])) \
            if len(hy['gpu_ids']) > 0 else torch.device('cpu')
        # model
        self.sp_net = spnet_network.define_SPNet(hy)
        self.model_names = ['sp_net']
        self.mult_dis = spnet_network.define_MultiscaleDis(input_nc=hy['n_label'], output_nc=1,hy=hy)
        self.model_names.append('mult_dis')

        is_train = hy['is_train']
        self.is_train = is_train
        continue_train = hy['continue_train']
        which_epoch = hy['which_epoch']

        if is_train:
            self.criterion_GAN = loss.GANLoss(gan_mode='hinge', tensor=self.FloatTensor)
            self.criterion_Feat = loss.GANFeatMatchingLoss()

        if is_train:
            self.optimizers = []
            self.schedulers = []
            self.optimizer_sp_net = torch.optim.Adam(self.sp_net.parameters(),
                                                     lr=0.0002 / 2,
                                                     betas=(0, 0.9))
            self.optimizer_mult_dis = torch.optim.Adam(self.mult_dis.parameters(),
                                                       lr=0.0002 * 2,
                                                       betas=(0, 0.9))
            self.optimizers.append(self.optimizer_sp_net)
            self.optimizers.append(self.optimizer_mult_dis)

            for optimizer in self.optimizers:
                self.schedulers.append(utils.get_scheduler(optimizer, hy))

            print('---------- Networks initialized -------------')
            utils.print_network(self.sp_net)
            # utils.print_network(self.dis_img)
            utils.print_network(self.mult_dis)

        if continue_train:
            assert which_epoch >= 0, "load epoch must > 0 !!"
            print('Loading pre-trained network for train!')
            self.load_networks(hy['checkpoints_dir'], hy['model_name'], which_epoch)

        if not is_train:
            assert which_epoch >= 0, "load epoch must > 0 !!"
            print('Loading pre-trained network for test!')
            self.load_networks(hy['checkpoints_dir'], hy['model_name'], which_epoch)
            for model in self.model_names:
                getattr(self, model).eval()
                for para in getattr(self, model).parameters():
                    para.requires_grad = False

    def set_input(self, inputs):
        file_name, image_gt, mask, segmap = inputs

        self.file_name = file_name
        self.image_gt = image_gt.to(self.device)
        self.mask = mask.to(self.device)

        self.segmap = segmap.to(self.device)

        bs, _, h, w = self.segmap.size()
        nc = self.hy['n_label']
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        self.input_semantics = input_label.scatter_(1, self.segmap[:, 0:1, :, :], 1.0)

        # mask 0 is hole
        self.inv_ex_mask = torch.add(torch.neg(self.mask.float()), 1).float()
        self.corrupted_seg = self.input_semantics * self.inv_ex_mask[:, 0:1, :, :]

        # Do not set the mask regions as 0
        self.input = image_gt.to(self.device)
        self.input.narrow(1, 0, 1).masked_fill_(self.mask.narrow(1, 0, 1).bool(), 0)
        self.input.narrow(1, 1, 1).masked_fill_(self.mask.narrow(1, 0, 1).bool(), 0)
        self.input.narrow(1, 2, 1).masked_fill_(self.mask.narrow(1, 0, 1).bool(), 0)

        # self.input_spn = torch.cat([self.input, self.corrupted_seg], dim=1)

    def forward(self):
        self.fake_out = self.sp_net(self.input)

    def backward_d(self):
        self.d_pred_real_seg = self.mult_dis(self.input_semantics)
        self.d_pred_fake_seg = self.mult_dis(self.fake_out.detach())

        self.d_real_loss = self.criterion_GAN(self.d_pred_real_seg, target_is_real=True, for_discriminator=True)
        self.d_fake_loss = self.criterion_GAN(self.d_pred_fake_seg, target_is_real=False, for_discriminator=True)
        self.d_loss = self.d_real_loss + self.d_fake_loss
        self.d_loss.backward()

    def backward_g(self):
        g_pred_fake_seg = self.mult_dis(self.fake_out)
        self.g_pred_fake_seg_loss = 0.1 * self.criterion_GAN(g_pred_fake_seg, target_is_real=True,
                                                             for_discriminator=False)
        self.g_pred_fake_seg_loss.backward()

    def optimize_parameters(self):
        self.forward()
        # Optimize the D
        self.set_requires_grad(self.sp_net, False)
        self.set_requires_grad(self.mult_dis, True)
        self.optimizer_mult_dis.zero_grad()
        self.backward_d()
        self.optimizer_mult_dis.step()

        # Optimize G
        self.set_requires_grad(self.sp_net, True)
        self.set_requires_grad(self.mult_dis, False)
        self.optimizer_sp_net.zero_grad()
        self.backward_g()
        self.optimizer_sp_net.step()

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_errors(self):
        loss_dict = {
            # discriminator
            'loss_D': self.d_loss.data,
            'loss_D_real': self.d_real_loss.data,
            'loss_D_fake': self.d_fake_loss.data,
            # Generator
            'loss_G': self.g_pred_fake_seg_loss,
        }
        return loss_dict

    def get_current_visuals(self):
        pred_attention = utils.tensor2label(self.fake_out, self.hy['n_label'], tile=False)
        pred_attention = pred_attention.transpose(0, 3, 1, 2) / 255
        pred_attention = torch.from_numpy(pred_attention)

        segmap = utils.tensor2label(self.input_semantics, self.hy['n_label'], tile=False)
        segmap = segmap.transpose(0, 3, 1, 2) / 255
        segmap = torch.from_numpy(segmap)

        return pred_attention, segmap

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_networks(self, checkpoints_dir, model, gpu_ids, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s_%s.pth' % (model, which_epoch, name)
                save_path = os.path.join(checkpoints_dir, save_filename)
                net = getattr(self, name)
                optimize = getattr(self, 'optimizer_' + name)

                if len(gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save({'net': net.module.cpu().state_dict(), 'optimize': optimize.state_dict()}, save_path)
                    net.cuda(gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, checkpoints_dir, model, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                # load pretrain model
                load_filename = '%s_%s_%s.pth' % (model, which_epoch, name)
                load_path = os.path.join(checkpoints_dir, load_filename)
                state_dict = torch.load(load_path, map_location=str(self.device))

                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict['net'])

                if self.is_train:
                    optimize = getattr(self, 'optimizer_' + name)
                    optimize.load_state_dict(state_dict['optimize'])

            print("load [%s] successful!" % name)
