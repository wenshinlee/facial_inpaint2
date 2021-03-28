import os
import random
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms

import cv2
from PIL import Image


class CelebA_Dataset(data.Dataset):
    def __init__(self, hy, image_transform, mask_transform):
        super(CelebA_Dataset, self).__init__()
        """ transform options"""
        self.mask_transform = mask_transform
        self.image_transform = image_transform

        """ hyperparameters options"""
        # number segmap label
        self.num_semantic_label = hy['num_semantic_label']
        # edge
        self.edge_mode = hy['edge_mode']
        # status options
        self.is_train = hy['is_train']
        # file path options
        self.image_dir = hy['image_dir']
        self.segmap_dir = hy['segmap_dir']
        self.pconv_mask_dir = hy['pconv_mask_dir']
        self.facial_region_mask_dir = hy['facial_region_mask_dir']
        # data processing options
        self.image_size = hy['image_size']
        self.dilate_iter = hy['dilate_iter']
        self.shuffle_seed = hy['shuffle_seed']
        self.p_generate_miss = hy['p_generate_miss']
        self.num_max_miss_facial_names = hy['num_max_miss_facial_names']
        # support facial region names only for facial region miss
        self.support_facial_region_names = hy['facial_semantic_region_names']

        """ data file list options"""
        # get data file name list
        # image file list
        dataset = self.get_dataset(self.image_dir)
        self.image_dataset = dataset[2000:] if self.is_train else dataset[:2000]
        # pconv mask file list
        self.pconv_mask_dataset = self.get_dataset(self.pconv_mask_dir)
        # segmap file list
        dataset = self.get_dataset(self.segmap_dir)
        self.segmap_dataset = dataset[2000:] if self.is_train else dataset[:2000]
        # facial segmap region mask dir
        # facial_region_mask_dir + support_facial_region_names[?] + image file name

        assert len(self.support_facial_region_names) >= self.num_max_miss_facial_names, \
            "max_miss_facial_names must <= support_facial_names!"

    def get_dataset(self, file_path):
        np.random.seed(self.shuffle_seed)
        dataset_list = sorted(os.listdir(file_path))
        np.random.shuffle(dataset_list)
        return dataset_list

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, item):
        file_name = self.image_dataset[item]
        image = self.load_images(file_name=file_name)

        mask_zeros = np.zeros_like(image)
        mask = self.load_mask(file_name=file_name, mask_zeros=mask_zeros, item=item,
                              miss_area_names=self.get_miss_area_names())

        segmap_name = self.segmap_dataset[item]
        segmap, segmap_one_hot = self.load_segmap(file_name=segmap_name)

        edge = self.load_edge(image, mode=self.edge_mode)

        return file_name, self.image_transform(self.convert_pil(image)), \
               self.mask_transform(self.convert_pil(mask)), \
               torch.FloatTensor(segmap), torch.FloatTensor(segmap_one_hot), torch.FloatTensor(edge)

    def load_images(self, file_name):
        image = cv2.imread(os.path.join(self.image_dir, file_name))
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return image

    def load_mask(self, file_name, mask_zeros, item, miss_area_names=None):
        if self.is_train:
            np.random.seed(None)
            which_mask_type_p = np.random.random()
            if which_mask_type_p > -1:
                # Missing based on Pconv mask dataset
                mask_path = os.path.join(self.pconv_mask_dir, np.random.choice(self.pconv_mask_dataset))
                if os.path.exists(mask_path):
                    mask = cv2.resize(cv2.imread(mask_path), (self.image_size, self.image_size),
                                      interpolation=cv2.INTER_NEAREST)
                    # miss area dilate
                    mask = self.dilate_mask(mask)
                    # add to mask_zeros
                    mask_zeros += mask
                else:
                    raise FileNotFoundError

            else:
                # Missing based on semantic segmentation map
                if miss_area_names is not None and isinstance(miss_area_names, list):
                    # not None, just for specified facial fea miss
                    for miss_type in miss_area_names:
                        mask_path = os.path.join(self.facial_region_mask_dir, miss_type.lower(), file_name)
                        if os.path.exists(mask_path):
                            mask = cv2.resize(cv2.imread(mask_path), (self.image_size, self.image_size),
                                              interpolation=cv2.INTER_NEAREST)
                            # miss area dilate
                            mask = self.dilate_mask(mask)
                            # add to mask_zeros
                            mask_zeros += mask
                        else:
                            print("[{}] file not found!, image will not miss anything!!".format(mask_path))

                else:  # Random facial fea area miss
                    mask_zeros = generate_stroke_mask(mask_zeros)
                    mask_zeros = (mask_zeros > 0).astype(np.uint8) * 255

        else:
            # test, load in order
            mask_path = os.path.join(self.pconv_mask_dir, self.pconv_mask_dataset[item])
            if os.path.exists(mask_path):
                mask = cv2.resize(cv2.imread(mask_path), (self.image_size, self.image_size),
                                  interpolation=cv2.INTER_NEAREST)
                # miss area dilate
                mask = self.dilate_mask(mask)
                # add to mask_zeros
                mask_zeros += mask
            else:
                raise FileNotFoundError

        mask_zeros = mask_zeros.astype(np.uint8)
        return mask_zeros

    def dilate_mask(self, mask):
        # miss area dilate
        kernel_size = np.random.randint(2, 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=self.dilate_iter)
        return mask

    def load_segmap(self, file_name):
        # The old version used the known segmap for region selection
        segmap = np.zeros((len(self.support_facial_region_names), self.image_size, self.image_size))
        for idx, facial_fea_name in enumerate(self.support_facial_region_names):
            segmap_path = os.path.join(self.support_facial_region_names, facial_fea_name.lower(), file_name)
            if os.path.exists(segmap_path):
                facial_segmap = cv2.resize(cv2.imread(segmap_path, cv2.IMREAD_GRAYSCALE),
                                           (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                segmap[idx, :, :] = facial_segmap / 255.0
            else:
                print("[{}] file not found!, segmap will use default zero segmap!!".format(segmap_path))

        return segmap, segmap

    @staticmethod
    def load_edge(image, mode=0):
        if mode == 0:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Gaussian blur, noise reduction
            Gauss_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
            threshold1 = random.randint(30, 50)
            canny_image = cv2.Canny(Gauss_image, threshold1, 150) / 255.0
            canny_image = canny_image[np.newaxis, :, :]
        elif mode == 1:
            canny_image = np.zeros((1, image.shape[0], image.shape[1])).astype(np.float)
        else:
            raise NotImplementedError

        return canny_image

    def get_miss_area_names(self, p=None):
        if self.p_generate_miss > np.random.random(1):  # [0.0, 1.0)
            return None
        else:
            # Equal probability missing
            if not p:
                p_factor = 1.0 / len(self.support_facial_region_names)
                p = [p_factor for _ in self.support_facial_region_names]
            else:
                assert len(p) == len(self.support_facial_region_names), \
                    "p length must equal the number of support facial names!"

        np.random.seed(None)
        num_miss_area = np.random.randint(1, self.num_max_miss_facial_names + 1)
        miss_area_names = np.random.choice(self.support_facial_region_names, num_miss_area,
                                           replace=False, p=p).tolist()
        return miss_area_names

    @staticmethod
    def convert_pil(np_image):
        pil_image = Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB))
        return pil_image


class CelebA_SPNet(CelebA_Dataset):
    def __init__(self, hy, image_transform, mask_transform):
        super(CelebA_SPNet, self).__init__(hy, image_transform, mask_transform)

    def load_segmap(self, file_name):
        segmap_path = os.path.join(self.segmap_dir, file_name)
        if os.path.exists(segmap_path):
            facial_segmap = cv2.resize(cv2.imread(segmap_path, cv2.IMREAD_COLOR),
                                       (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST) \
                .transpose((2, 0, 1))
        else:
            raise FileNotFoundError

        facial_segmap = torch.from_numpy(facial_segmap).float()
        # convert one-hot (num_segmap_label,h,w)
        segmap_one_hot = torch.FloatTensor(self.num_semantic_label, facial_segmap.shape[1],
                                           facial_segmap.shape[2]).zero_()

        # segmap_one_hot[index[i][j][k]][j][k] = facial_segmap[i][j][k]  # if dim == 0
        segmap_one_hot = segmap_one_hot.scatter_(0, facial_segmap[0:1, :, :].long(), 1.0)

        return facial_segmap, segmap_one_hot


def generate_stroke_mask(mask_zeros, max_parts=9, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360):
    np.random.seed(None)
    parts = random.randint(1, max_parts)
    for i in range(parts):
        mask_zeros = mask_zeros + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle,
                                                    mask_zeros.shape[0], mask_zeros.shape[1])
    mask_zeros = np.minimum(mask_zeros, 1.0)
    return mask_zeros


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def get_dataloader(hyperparameters):
    """return a data loader"""
    # transform
    image_transform = transforms.Compose([
        transforms.Resize(hyperparameters['image_size'], interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(hyperparameters['image_size'], interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    dataset_name = hyperparameters['dataset_name']
    num_workers = hyperparameters['num_workers']
    batch_size = hyperparameters['batch_size']
    is_train = hyperparameters['is_train']

    if dataset_name == 'CelebA_Dataset':
        # The old version used the known segmap for region selection
        dataset = CelebA_Dataset(hyperparameters, image_transform, mask_transform)
    elif dataset_name == 'CelebA_SPNet':
        dataset = CelebA_SPNet(hyperparameters, image_transform, mask_transform)
    else:
        raise ValueError('dataset {} do not support!'.format(dataset_name))

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=is_train, num_workers=num_workers)

    return data_loader
