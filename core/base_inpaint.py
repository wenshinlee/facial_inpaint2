import os
from abc import ABC

import torch
import core.utils as utils
from torch.optim import lr_scheduler


class BaseInpaint(torch.nn.Module, ABC):
    def __init__(self, hy):
        super(BaseInpaint, self).__init__()
        ######################
        #    init parameter
        ######################
        self.hy = hy
        self.is_train = hy['is_train']
        self.num_train_semantic_net = hy['num_train_semantic_net']
        self.which_epoch = hy['which_epoch']
        self.experiment_name = hy['model_name']
        self.checkpoints_dir = hy['checkpoints_dir']
        if not os.path.exists(self.checkpoints_dir):
            utils.mkdirs(self.checkpoints_dir)
        # init net
        self.gpu_ids = hy['gpu_ids']
        self.init_type = hy['init_type']
        self.init_gain = hy['init_gain']
        # init lr
        self.lr_gan = hy['lr_gen']
        self.lr_dis = hy['lr_dis']
        self.betas = (hy['beta1'], hy['beta2'])
        # init GPU
        self.gpu_ids = hy['gpu_ids']
        self.num_semantic_label = self.hy['num_semantic_label']
        self.input_dim = hy['input_dim']
        self.mult_dis_para = hy['MultiscaleDis']
        # scheduler
        self.lr_policy = hy['lr_policy']
        self.lr_decay_iters = hy['lr_decay_iters']
        # for train
        self.continue_train = hy['continue_train']
        self.epoch_count = hy['epoch_count']
        self.niter = hy['niter']
        self.niter_decay = hy['niter_decay']
        # print, display
        self.print_freq = hy['print_freq']
        self.display_freq = hy['display_freq']
        self.save_epoch_freq = hy['save_epoch_freq']
        self.output_dir = hy['output_dir']

        ######################
        #    init GPU
        ######################
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) \
            if len(self.gpu_ids) > 0 else torch.device('cpu')

        ######################
        #    init list
        ######################
        self.model_names = []
        self.optimizers = []
        self.schedulers = []

        ######################
        #    init variable
        ######################
        self.file_name = None
        self.image_gt = None
        self.mask = None
        self.inv_ex_mask = None
        self.semantic_label = None
        self.semantic_one_hot = None
        # just for Cross entropy Loss
        self.target_semantic = None
        self.edge = None
        # inputs
        self.input_edge = None
        self.input_segmap = None
        self.input_image = None
        # output
        self.fake_semantic = None
        self.fake_image = None

        ######################
        #    init loss
        ######################
        # inpaint_gen
        self.loss_ig_L1 = None
        self.loss_ig_perceptual = None
        self.loss_ig_style = None
        self.loss_ig_gan = None
        self.loss_ig = None
        # inpaint_dis
        self.loss_id_gan = None
        # semantic_gan
        self.loss_sg_semantic = None
        self.loss_sg_semantic_L1 = None
        self.loss_sg = None
        # semantic_dis
        self.loss_sd_semantic = None

    def print_network(self):
        for model_name in self.model_names:
            model = getattr(self, model_name)
            utils.print_network(model)

    def init_optimizer(self):
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if 'gan' in model_name:
                lr = self.lr_gan
            else:
                lr = self.lr_dis
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=self.betas)
            setattr(self, 'optimizer_' + model_name, optimizer)

            self.optimizers.append(getattr(self, 'optimizer_' + model_name))
            self.schedulers.append(self.get_scheduler(optimizer=getattr(self, 'optimizer_' + model_name)))

    def load_networks(self):
        assert self.which_epoch >= 0, 'load epoch must > 0 !!'
        for model_name in self.model_names:
            if isinstance(model_name, str):
                load_filename = '%s_%s_%s.pth' % (self.experiment_name, self.which_epoch, model_name)
                load_path = os.path.join(self.checkpoints_dir, load_filename)
                if os.path.exists(load_path):
                    state_dict = torch.load(load_path, map_location=str(self.device))

                    net = getattr(self, model_name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    net.load_state_dict(state_dict['net'])

                    if self.is_train:
                        optimize = getattr(self, 'optimizer_' + model_name)
                        optimize.load_state_dict(state_dict['optimize'])
                else:
                    print("{} not exists, will adapt default init net parameter!".format(model_name))

            print("load [%s] successful!" % model_name)

    def save_networks(self, which_epoch):
        for model_name in self.model_names:
            if isinstance(model_name, str):
                save_filename = '%s_%s_%s.pth' % (model_name, which_epoch, model_name)
                save_path = os.path.join(self.checkpoints_dir, save_filename)
                net = getattr(self, model_name)
                optimize = getattr(self, 'optimizer_' + model_name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save({'net': net.module.cpu().state_dict(), 'optimize': optimize.state_dict()}, save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

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

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        for i, optimizers in enumerate(self.optimizers):
            lr = optimizers.param_groups[0]['lr']
            print('optimizers_{} learning rate = {}'.format(str(i), str(lr)))

    def get_scheduler(self, optimizer):
        if self.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + self.epoch_count - self.niter) / float(self.niter_decay + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif self.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_iters, gamma=0.1)
        elif self.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01,
                                                       patience=5)
        elif self.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.niter, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', self.lr_policy)
        return scheduler
