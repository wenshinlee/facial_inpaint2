import os
import yaml
import torch
import socket
import torchvision
import functools
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import init
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


################################
#   load yaml
################################
def get_config(config_path):
    with open(config_path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


################################
#   semantic
################################
def get_num_semantic(tags):
    total_num_semantic = len(tags)
    ignore_tag_names = []
    ignore_tag_names_idx = []
    for idx, tag in enumerate(tags):
        if not tag['status']:
            total_num_semantic -= 1
            ignore_tag_names.append(tags['name'])
            ignore_tag_names_idx.append(idx)

    return total_num_semantic, ignore_tag_names, ignore_tag_names_idx


def get_support_semantic_names(tags):
    support_semantic_names = []
    for tag in tags:
        if tag['status']:
            support_semantic_names.append(tag['name'])
    return support_semantic_names


################################
#   net init, print and norm layer
################################
def get_norm_layer(norm_type='BatchNorm2d'):
    if norm_type == 'BatchNorm2d':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'InstanceNorm2d':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=None):
    if gpu_ids is None:
        gpu_ids = []
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


################################
#   creat dirs
################################
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


################################
#   Tensorboard
################################
class TensorboardVisuals(object):
    def __init__(self, checkpoints_dir, experiment_name):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(checkpoints_dir,
                               'runs', experiment_name + '-' + current_time + '-' + socket.gethostname())
        if not os.path.exists(log_dir):
            mkdirs(log_dir)

        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def add_image(self, tag, value, step):
        self.writer.add_image(tag, value, step)


################################
#   dataloader images visual
################################
def imgshow(img, unnormalize=True):
    """
    显示图片
    data_iter = iter(data_loader)
    file_name, image, mask, segmap = next(data_iter)
    imgshow(image)
    imgshow(mask, unnormalize=False)
    imgshow(segmap[:, 0:1, ...], unnormalize=False),segmap 每一个通道代表一个面部特征的mask
    :param img:输入图片格式的 B,C,H,W.其中C可以为 1
    :param unnormalize:去归一化
    :return:None
    """
    img = torchvision.utils.make_grid(img)
    if unnormalize:
        img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


################################
#   semantic label visual
################################
# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, img_type=np.uint8):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, img_type)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    if n_label == 0:
        return tensor2img(label_tensor, img_type)

    label_tensor = label_tensor.cpu().float()

    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]

    label_tensor = Colorize(n_label)(label_tensor)

    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(img_type)
    return result


# Converts a Tensor into a Numpy array
# |img_type|: the desired type of the converted numpy array
def tensor2img(image_tensor, img_type=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2img(image_tensor[i], img_type, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2img(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)

        return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(img_type)


class Colorize(object):
    def __init__(self, n_labels=35):
        self.color_map = label2colormap(n_labels)
        self.color_map = torch.from_numpy(self.color_map[:n_labels])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.color_map)):
            mask = (label == gray_image[0])
            color_image[0][mask] = self.color_map[label][0]
            color_image[1][mask] = self.color_map[label][1]
            color_image[2][mask] = self.color_map[label][2]

        return color_image


def label2colormap(n_labels):
    color_map = np.zeros((n_labels, 3), dtype=np.uint8)
    for i in range(n_labels):
        r, g, b = 0, 0, 0
        id = i + 1  # let's give 0 a color
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        color_map[i, 0] = r
        color_map[i, 1] = g
        color_map[i, 2] = b

    return color_map


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])
