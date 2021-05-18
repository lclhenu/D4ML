from torch.utils.data import Dataset
from PIL import Image
import torch as t
import os
import scipy.io as sio
from skimage import io
import torchvision.transforms as transforms
import cv2
import os
import numpy as np

# rgb2gray = transforms.Compose([transforms.ToPILImage(), transforms.Resize([64, 64]), transforms.RandomHorizontalFlip(),
#                                transforms.Grayscale(3), transforms.ToTensor()])
rgb2gray = transforms.Compose([transforms.ToPILImage(), transforms.Resize([64, 64]), transforms.Grayscale(3),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
lab_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])
hf = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
vf = transforms.Compose([transforms.ToPILImage(), transforms.RandomVerticalFlip(), transforms.ToTensor()])

rgb2graypoints = transforms.Compose(
    [transforms.ToTensor(), transforms.ToPILImage(), transforms.Grayscale(3), transforms.ToTensor()])
hfpoints = transforms.Compose(
    [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
vfpoints = transforms.Compose(
    [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomVerticalFlip(), transforms.ToTensor()])

r2g = transforms.Compose([transforms.ToPILImage(),
                          transforms.TenCrop(39),
                          transforms.Lambda(lambda crops: torch.stack([rgb2graypoints(crop) for crop in crops]))])
horizontal_flip = transforms.Compose([transforms.ToPILImage(),
                                      transforms.TenCrop(39),
                                      transforms.Lambda(lambda crops: torch.stack([hfpoints(crop) for crop in crops]))])
vertical_flip = transforms.Compose([transforms.ToPILImage(),
                                    transforms.TenCrop(39),  # this is a list of PIL Images,一张图变成10张图
                                    transforms.Lambda(lambda crops: torch.stack([vfpoints(crop) for crop in crops]))
                                    # returns a 4D tensor
                                    ])
normal = transforms.Compose([transforms.ToPILImage(),
                             transforms.TenCrop(39),
                             transforms.Lambda(
                                 lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])

import torch
import torch.nn as nn
import numpy as np

from random import shuffle


def fliplr(img):
    """
    flip horizontal
    """

    inv_idx = torch.arange(img.size(2) - 1, -1, -1).long()  # C x H x W

    img_flip = img.index_select(2, inv_idx)
    return img_flip


class KinShipDataSetZL(Dataset):
    rel_lookup = {'fd': 'father-dau', 'fs': 'father-son', 'md': 'mother-dau', 'ms': 'mother-son', 'all': 'all'}

    def __init__(self, relation, mode, image_path, meta_data_path, train_index, valid_index, test_index, fold,
                 transform=None, aug=False):
        self.meta_data = sio.loadmat(os.path.join(meta_data_path, relation + '_pairs.mat'))
        self.relation = relation
        self.transform = transform
        self.dim = 64
        self.image_path = image_path
        self.meta_data_path = meta_data_path
        self.aug = False
        self.mode = mode
        self.fold = fold

        if self.mode == 'train':
            self.tvtid = train_index
        elif self.mode == 'valid':
            self.tvtid = valid_index
        else:
            self.tvtid = test_index
        self.tvtlen = len(self.tvtid)

    def __len__(self):
        return self.tvtlen

    def __getitem__(self, i):
        strlist = self.meta_data_path.split('/')
        data_name = strlist[1]
        if data_name == 'UBKinFace':
            s1 = self.meta_data['pairs'][self.tvtid[i], 2][0]
            s2 = self.meta_data['pairs'][self.tvtid[i], 3][0]
            s1 = s1.replace('_new', '')
            s2 = s2.replace('_new', '')
            if self.relation == 'set1':
                image_file1 = os.path.join(self.image_path, '03' + '/' + s1)
                image_file2 = os.path.join(self.image_path, '01' + '/' + s2)
            elif self.relation == 'set2':
                image_file1 = os.path.join(self.image_path, '02' + '/' + s1)
                image_file2 = os.path.join(self.image_path, '01' + '/' + s2)
        else:
            if data_name == 'TSKinFace' or data_name == 'CornellKinFace':
                folder = KinShipDataSetZL.rel_lookup[self.relation]
            else:
                folder = KinShipDataSetZL.rel_lookup[self.meta_data['pairs'][self.tvtid[i], 2][0][:2]]
            image_file1 = os.path.join(self.image_path,
                                       folder + '/' + self.meta_data['pairs'][self.tvtid[i], 2][0])
            image_file2 = os.path.join(self.image_path,
                                       folder + '/' + self.meta_data['pairs'][self.tvtid[i], 3][0])


        image1 = io.imread(image_file1)
        image2 = io.imread(image_file2)
        if self.transform:
            image1 = self.transform(Image.fromarray(image1))
            image2 = self.transform(Image.fromarray(image2))
            gray1 = rgb2gray(image1)
            gray2 = rgb2gray(image1)


        else:
            image1 = torch.from_numpy(image1)
            image2 = torch.from_numpy(image2)

        if self.mode == "test":
            if self.aug:
                vf1 = vf(image1)
                vf2 = vf(image2)
                hf1 = hf(image1)
                hf2 = hf(image2)

                pair_normal = torch.cat((image1, image2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_gray = torch.cat((gray1, gray2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_hf = torch.cat((hf1, hf2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_vf = torch.cat((vf1, vf2), dim=0).view(-1, 6, self.dim, self.dim)

                pair = torch.cat((pair_normal, pair_gray, pair_hf, pair_vf), dim=0)

                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                sample = {'pair1': image1, 'pair2': image2, 'label': label}
            else:

                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                sample = {'input1': image1, 'input2': image2, 'label': label}
        else:
            if self.aug:
                vf1 = vf(image1)
                vf2 = vf(image2)
                hf1 = hf(image1)
                hf2 = hf(image2)
                pair_normal = torch.cat((image1, image2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_gray = torch.cat((gray1, gray2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_hf = torch.cat((hf1, hf2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_vf = torch.cat((vf1, vf2), dim=0).view(-1, 6, self.dim, self.dim)

                pair = torch.cat((pair_normal, pair_gray, pair_hf, pair_vf), dim=0)


                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                sample = {'pair1': image1, 'pair2': image2, 'label': label}
            else:
                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                # UB
                if self.meta_data_path.split('/')[1] == 'UBKinFace':
                    image1_label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 2][0].split('_')[0])-1])
                    image2_label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 3][0].split('_')[0])-1])

                # CK
                if self.meta_data_path.split('/')[1] == 'CornellKinFace':
                    image1_label = torch.LongTensor(
                        [int(self.meta_data['pairs'][self.tvtid[i], 2][0].split('_')[1].split('.')[0]) - 1])
                    image2_label = torch.LongTensor(
                        [int(self.meta_data['pairs'][self.tvtid[i], 3][0].split('_')[1].split('.')[0]) - 1])

                # KinFace
                if self.meta_data_path.split('/')[1] == 'KinFaceW-I' or self.meta_data_path.split('/')[1] == 'KinFaceW-II':
                    image1_label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 2][0].split('_')[1])-1])
                    image2_label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 3][0].split('_')[1])-1])

                sample = {'input1': image1, 'input2': image2, 'label': label,
                          'image1_label': image1_label, 'image2_label': image2_label}
        return sample


class KinShipDataSetZLAug(Dataset):
    rel_lookup = {'fd': 'father-dau', 'fs': 'father-son', 'md': 'mother-dau', 'ms': 'mother-son', 'all': 'all'}

    def __init__(self, relation, mode, image_path, meta_data_path, train_index, valid_index, test_index, fold,
                 transform=None, aug=False):
        self.meta_data = sio.loadmat(os.path.join(meta_data_path, relation + '_pairs.mat'))
        self.relation = relation
        self.transform = transform
        self.dim = 64
        self.image_path = image_path
        self.meta_data_path = meta_data_path
        self.aug = aug
        self.mode = mode
        self.fold = fold

        if self.mode == 'train':
            self.tvtid = train_index
        elif self.mode == 'valid':
            self.tvtid = valid_index
        else:
            self.tvtid = test_index
        self.tvtlen = len(self.tvtid)

    def __len__(self):
        return self.tvtlen

    def __getitem__(self, i):

        strlist = self.meta_data_path.split('/')
        data_name = strlist[2]
        if data_name == 'UBKinFace':
            s1 = self.meta_data['pairs'][self.tvtid[i], 2][0]
            s2 = self.meta_data['pairs'][self.tvtid[i], 3][0]
            s1 = s1.replace('_new', '')
            s2 = s2.replace('_new', '')
            if self.relation == 'set1':
                image_file1 = os.path.join(self.image_path, '03' + '/' + s1)
                image_file2 = os.path.join(self.image_path, '01' + '/' + s2)
            elif self.relation == 'set2':
                image_file1 = os.path.join(self.image_path, '02' + '/' + s1)
                image_file2 = os.path.join(self.image_path, '01' + '/' + s1)
        else:
            if data_name == 'TSKinFace' or data_name == 'CornellKinFace':
                folder = KinShipDataSetZLAug.rel_lookup[self.relation]
            else:
                folder = KinShipDataSetZLAug.rel_lookup[self.meta_data['pairs'][self.tvtid[i], 2][0][:2]]
            image_file1 = os.path.join(self.image_path,
                                       folder + '/' + self.meta_data['pairs'][self.tvtid[i], 2][0])
            image_file2 = os.path.join(self.image_path,
                                       folder + '/' + self.meta_data['pairs'][self.tvtid[i], 3][0])
            filename1 = self.meta_data['pairs'][self.tvtid[i], 2][0]
            filename2 = self.meta_data['pairs'][self.tvtid[i], 3][0]
            filena1 = filename1.split('.')[0]
            filena2 = filename2.split('.')[0]

            image_file10 = os.path.join(self.image_path,
                                        folder + '/after/' + filena1 + "_0.jpg")
            image_file20 = os.path.join(self.image_path,
                                        folder + '/after/' + filena2 + "_0.jpg")
            image_file11 = os.path.join(self.image_path,
                                        folder + '/after/' + filena1 + "_1.jpg")
            image_file21 = os.path.join(self.image_path,
                                        folder + '/after/' + filena2 + "_1.jpg")
            image_file12 = os.path.join(self.image_path,
                                        folder + '/after/' + filena1 + "_2.jpg")
            image_file22 = os.path.join(self.image_path,
                                        folder + '/after/' + filena2 + "_2.jpg")
            image_file13 = os.path.join(self.image_path,
                                        folder + '/after/' + filena1 + "_3.jpg")
            image_file23 = os.path.join(self.image_path,
                                        folder + '/after/' + filena2 + "_3.jpg")
            image_file14 = os.path.join(self.image_path,
                                        folder + '/after/' + filena1 + "_4.jpg")
            image_file24 = os.path.join(self.image_path,
                                        folder + '/after/' + filena2 + "_4.jpg")

        image1 = io.imread(image_file1)
        image2 = io.imread(image_file2)

        image10 = io.imread(image_file10)
        image20 = io.imread(image_file20)
        image11 = io.imread(image_file11)
        image21 = io.imread(image_file21)
        image12 = io.imread(image_file12)
        image22 = io.imread(image_file22)
        image13 = io.imread(image_file13)
        image23 = io.imread(image_file23)
        image14 = io.imread(image_file14)
        image24 = io.imread(image_file24)

        gray1 = rgb2gray(image1)
        gray2 = rgb2gray(image2)

        if self.transform:

            image1 = self.transform(Image.fromarray(image1))  #
            image2 = self.transform(Image.fromarray(image2))  #
            image10 = self.transform(Image.fromarray(image10))  #
            image20 = self.transform(Image.fromarray(image20))  #
            image11 = self.transform(Image.fromarray(image11))  #
            image21 = self.transform(Image.fromarray(image21))  #
            image12 = self.transform(Image.fromarray(image12))  #
            image22 = self.transform(Image.fromarray(image22))  #
            image13 = self.transform(Image.fromarray(image13))  #
            image23 = self.transform(Image.fromarray(image23))  #
            image14 = self.transform(Image.fromarray(image14))  #
            image24 = self.transform(Image.fromarray(image24))  #


        else:
            image1 = torch.from_numpy(image1)
            image2 = torch.from_numpy(image2)
            image10 = torch.from_numpy(image10)
            image20 = torch.from_numpy(image20)
            image11 = torch.from_numpy(image11)
            image21 = torch.from_numpy(image21)
            image12 = torch.from_numpy(image12)
            image22 = torch.from_numpy(image22)
            image13 = torch.from_numpy(image13)
            image23 = torch.from_numpy(image23)
            image14 = torch.from_numpy(image14)
            image24 = torch.from_numpy(image24)

        if self.mode == "test":
            if self.aug:
                vf1 = vf(image1)
                vf2 = vf(image2)
                hf1 = hf(image1)
                hf2 = hf(image2)

                pair_normal = torch.cat((image1, image2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_gray = torch.cat((gray1, gray2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_hf = torch.cat((hf1, hf2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_vf = torch.cat((vf1, vf2), dim=0).view(-1, 6, self.dim, self.dim)

                pair = torch.cat((pair_normal, pair_gray, pair_hf, pair_vf), dim=0)

                label = torch.LongTensor(
                    np.full((4), int(self.meta_data['pairs'][self.tvtid[i], 1]), dtype=int).tolist())
                sample = {'pair': pair, 'label': label}
            else:
                pair = torch.cat((image1, image2), dim=0)
                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                sample = {'pair': pair, 'label': label}
        else:
            if self.aug:
                vf1 = vf(image1)
                vf2 = vf(image2)
                hf1 = hf(image1)
                hf2 = hf(image2)


                pair_normal = torch.cat((image1, image2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_gray = torch.cat((gray1, gray2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_hf = torch.cat((hf1, hf2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_vf = torch.cat((vf1, vf2), dim=0).view(-1, 6, self.dim, self.dim)

                pair_normal0 = torch.cat((image10, image20), dim=0).view(-1, 6, self.dim, self.dim)
                pair_normal1 = torch.cat((image11, image21), dim=0).view(-1, 6, self.dim, self.dim)
                pair_normal2 = torch.cat((image12, image22), dim=0).view(-1, 6, self.dim, self.dim)
                pair_normal3 = torch.cat((image13, image23), dim=0).view(-1, 6, self.dim, self.dim)
                pair_normal4 = torch.cat((image14, image24), dim=0).view(-1, 6, self.dim, self.dim)

                pair = torch.cat((pair_normal, pair_normal0, pair_normal1, pair_normal2, pair_normal3, pair_normal4,
                                  pair_gray, pair_hf, pair_vf), dim=0)


                label = torch.LongTensor(
                    np.full((9), int(self.meta_data['pairs'][self.tvtid[i], 1]),
                            dtype=int).tolist())  # if not self.test else None
                sample = {'pair': pair, 'label': label}
            else:
                pair = torch.cat((image1, image2), dim=0).view(-1, 6, self.dim, self.dim)
                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                sample = {'pair': pair, 'label': label}
        return sample


class KinShipDataSetPoints(Dataset):
    rel_lookup = {'fd': 'father-dau', 'fs': 'father-son', 'md': 'mother-dau', 'ms': 'mother-son', 'all': 'all'}

    def __init__(self, relation, mode, image_path, meta_data_path, train_index, valid_index, test_index, fold,
                 transform=None, aug=False):
        self.meta_data = sio.loadmat(os.path.join(meta_data_path, relation + '_pairs.mat'))
        self.relation = relation
        self.transform = transform
        self.dim = 39
        self.image_path = image_path
        self.meta_data_path = meta_data_path
        self.aug = aug
        self.mode = mode
        self.fold = fold

        if self.mode == 'train':
            self.tvtid = train_index
        elif self.mode == 'valid':
            self.tvtid = valid_index
        else:
            self.tvtid = test_index
        self.tvtlen = len(self.tvtid)

    def __len__(self):
        return self.tvtlen

    def __getitem__(self, i):

        strlist = self.meta_data_path.split('/')
        data_name = strlist[2]
        if data_name == 'UBKinFace':
            s1 = self.meta_data['pairs'][self.tvtid[i], 2][0]
            s2 = self.meta_data['pairs'][self.tvtid[i], 3][0]
            s1 = s1.replace('_new', '')
            s2 = s2.replace('_new', '')
            if self.relation == 'set1':
                image_file1 = os.path.join(self.image_path, '03' + '/' + s1)
                image_file2 = os.path.join(self.image_path, '01' + '/' + s2)
            elif self.relation == 'set2':
                image_file1 = os.path.join(self.image_path, '02' + '/' + s1)
                image_file2 = os.path.join(self.image_path, '01' + '/' + s1)
        else:
            if data_name == 'TSKinFace' or data_name == 'CornellKinFace':
                folder = KinShipDataSetPoints.rel_lookup[self.relation]
            else:
                folder = KinShipDataSetPoints.rel_lookup[self.meta_data['pairs'][self.tvtid[i], 2][0][:2]]
            image_file1 = os.path.join(self.image_path,
                                       folder + '/' + self.meta_data['pairs'][self.tvtid[i], 2][0])
            image_file2 = os.path.join(self.image_path,
                                       folder + '/' + self.meta_data['pairs'][self.tvtid[i], 3][0])


        image1 = io.imread(image_file1)
        image2 = io.imread(image_file2)
        gray1 = r2g(image1)
        gray2 = r2g(image2)

        if self.transform:
            image1 = self.transform(image1)

            image2 = self.transform(image2)


        else:
            image1 = torch.from_numpy(image1)
            image2 = torch.from_numpy(image2)

        if self.mode == "test":
            if self.aug:
                vf1 = vertical_flip(image1)
                vf2 = vertical_flip(image2)
                hf1 = horizontal_flip(image1)
                hf2 = horizontal_flip(image2)

                pair_normal = torch.cat((normal(image1), normal(image2)), dim=1).view(-1, 10, 6, self.dim, self.dim)
                pair_gray = torch.cat((gray1, gray2), dim=1).view(-1, 10, 6, self.dim, self.dim)
                pair_hf = torch.cat((hf1, hf2), dim=1).view(-1, 10, 6, self.dim, self.dim)
                pair_vf = torch.cat((vf1, vf2), dim=1).view(-1, 10, 6, self.dim, self.dim)
                pair = torch.cat((pair_normal, pair_gray, pair_hf, pair_vf), dim=0)
                label = torch.LongTensor(
                    np.full((4), int(self.meta_data['pairs'][self.tvtid[i], 1]), dtype=int).tolist())
                sample = {'pair': pair, 'label': label}
            else:
                pair = torch.cat((normal(image1), normal(image2)), dim=1).view(-1, 10, 6, self.dim, self.dim)
                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                sample = {'pair': pair, 'label': label}
        else:
            if self.aug:
                vf1 = vertical_flip(image1)
                vf2 = vertical_flip(image2)
                hf1 = horizontal_flip(image1)
                hf2 = horizontal_flip(image2)

                temp = (normal(image1), normal(image2))

                pair_normal = torch.cat((normal(image1), normal(image2)), dim=1).view(-1, 10, 6, self.dim, self.dim)
                pair_gray = torch.cat((gray1, gray2), dim=1).view(-1, 10, 6, self.dim, self.dim)  # [1,10,6,39,39]
                pair_hf = torch.cat((hf1, hf2), dim=1).view(-1, 10, 6, self.dim, self.dim)
                pair_vf = torch.cat((vf1, vf2), dim=1).view(-1, 10, 6, self.dim, self.dim)
                pair = torch.cat((pair_normal, pair_gray, pair_hf, pair_vf), dim=0)
                label = torch.LongTensor(
                    np.full((4), int(self.meta_data['pairs'][self.tvtid[i], 1]),
                            dtype=int).tolist())  # if not self.test else None
                sample = {'pair': pair, 'label': label}
            else:
                pair = torch.cat((normal(image1), normal(image2)), dim=1).view(-1, 10, 6, self.dim, self.dim)
                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                sample = {'pair': pair, 'label': label}
        return sample


class KinShipDataSetZL1(Dataset):
    rel_lookup = {'fd': 'father-dau', 'fs': 'father-son', 'md': 'mother-dau', 'ms': 'mother-son', 'all': 'all'}

    def __init__(self, relation, mode, image_path, meta_data_path, train_index, valid_index, test_index, fold,
                 transform=None, aug=False):
        self.meta_data = sio.loadmat(os.path.join(meta_data_path, relation + '_pairs.mat'))
        self.relation = relation
        self.transform = transform
        self.dim = 64
        self.image_path = image_path
        self.meta_data_path = meta_data_path
        self.aug = aug
        self.mode = mode
        self.fold = fold

        if self.mode == 'train':
            self.tvtid = train_index
        elif self.mode == 'valid':
            self.tvtid = valid_index
        else:
            self.tvtid = test_index
        self.tvtlen = len(self.tvtid)

    def __len__(self):
        return self.tvtlen

    def __getitem__(self, i):
        strlist = self.meta_data_path.split('/')
        data_name = strlist[2]
        if data_name == 'UBKinFace':
            s1 = self.meta_data['pairs'][self.tvtid[i], 2][0]
            s2 = self.meta_data['pairs'][self.tvtid[i], 3][0]
            s1 = s1.replace('_new', '')
            s2 = s2.replace('_new', '')
            if self.relation == 'set1':
                image_file1 = os.path.join(self.image_path, '03' + '/' + s1)
                image_file2 = os.path.join(self.image_path, '01' + '/' + s2)
            elif self.relation == 'set2':
                image_file1 = os.path.join(self.image_path, '02' + '/' + s1)
                image_file2 = os.path.join(self.image_path, '01' + '/' + s1)
        else:
            if data_name == 'TSKinFace' or data_name == 'CornellKinFace':
                folder = KinShipDataSetZL.rel_lookup[self.relation]
            else:
                folder = KinShipDataSetZL.rel_lookup[self.meta_data['pairs'][self.tvtid[i], 2][0][:2]]
            image_file1 = os.path.join(self.image_path,
                                       folder + '/' + self.meta_data['pairs'][self.tvtid[i], 2][0])
            image_file2 = os.path.join(self.image_path,
                                       folder + '/' + self.meta_data['pairs'][self.tvtid[i], 3][0])


        image1 = io.imread(image_file1)
        image2 = io.imread(image_file2)
        gray1 = rgb2gray(image1)
        gray2 = rgb2gray(image2)

        if self.transform:

            image1 = self.transform(Image.fromarray(image1))  #
            image2 = self.transform(Image.fromarray(image2))  #


        else:
            image1 = torch.from_numpy(image1)
            image2 = torch.from_numpy(image2)

        if self.mode == "test":
            if self.aug:
                vf1 = vf(image1)
                vf2 = vf(image2)
                hf1 = hf(image1)
                hf2 = hf(image2)

                pair_normal = torch.cat((image1, image2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_gray = torch.cat((gray1, gray2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_hf = torch.cat((hf1, hf2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_vf = torch.cat((vf1, vf2), dim=0).view(-1, 6, self.dim, self.dim)

                pair = torch.cat((pair_normal, pair_gray, pair_hf, pair_vf), dim=0)

                label = torch.LongTensor(
                    np.full((4), int(self.meta_data['pairs'][self.tvtid[i], 1]), dtype=int).tolist())
                sample = {'pair': pair, 'label': label}
            else:
                pair = torch.cat((image1, image2), dim=0)
                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                sample = {'pair': pair, 'label': label}
        else:
            if self.aug:
                vf1 = vf(image1)
                vf2 = vf(image2)
                hf1 = hf(image1)
                hf2 = hf(image2)

                pair_normal = torch.cat((image1, image2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_gray = torch.cat((gray1, gray2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_hf = torch.cat((hf1, hf2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_vf = torch.cat((vf1, vf2), dim=0).view(-1, 6, self.dim, self.dim)

                pair = torch.cat((pair_normal, pair_gray, pair_hf, pair_vf), dim=0)
                print(pair.shape)
                label = torch.LongTensor(
                    np.full((4), int(self.meta_data['pairs'][self.tvtid[i], 1]),
                            dtype=int).tolist())  # if not self.test else None
                sample = {'pair': pair, 'label': label}
            else:
                pair = torch.cat((image1, image2), dim=0).view(-1, 6, self.dim, self.dim)
                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                sample = {'pair': pair, 'label': label}
        return sample


class KinShipDataSetAll(Dataset):
    rel_lookup = {'fd': 'father-dau', 'fs': 'father-son', 'md': 'mother-dau', 'ms': 'mother-son', 'all': 'all'}

    def __init__(self, relation, mode, image_path, meta_data_path, u_index, transform=None, aug=False):
        self.meta_data = sio.loadmat(os.path.join(meta_data_path, relation + '_pairs.mat'))
        self.relation = relation
        self.transform = transform
        self.dim = 64
        self.image_path = image_path
        self.meta_data_path = meta_data_path
        self.aug = aug
        self.mode = mode

        self.tvtid = u_index
        self.tvtlen = len(self.tvtid)

    def __len__(self):
        return self.tvtlen

    def __getitem__(self, i):

        strlist = self.meta_data_path.split('/')
        data_name = strlist[2]
        if data_name == 'UBKinFace':
            s1 = self.meta_data['pairs'][self.tvtid[i], 2][0]
            s2 = self.meta_data['pairs'][self.tvtid[i], 3][0]
            s1 = s1.replace('_new', '')
            s2 = s2.replace('_new', '')
            if self.relation == 'set1':
                image_file1 = os.path.join(self.image_path, '03' + '/' + s1)
                image_file2 = os.path.join(self.image_path, '01' + '/' + s2)
            elif self.relation == 'set2':
                image_file1 = os.path.join(self.image_path, '02' + '/' + s1)
                image_file2 = os.path.join(self.image_path, '01' + '/' + s1)
        else:
            if data_name == 'TSKinFace' or data_name == 'CornellKinFace':
                folder = KinShipDataSetAll.rel_lookup[self.relation]
            else:
                folder = KinShipDataSetAll.rel_lookup[self.meta_data['pairs'][self.tvtid[i], 2][0][:2]]
            image_file1 = os.path.join(self.image_path,
                                       folder + '/' + self.meta_data['pairs'][self.tvtid[i], 2][0])
            image_file2 = os.path.join(self.image_path,
                                       folder + '/' + self.meta_data['pairs'][self.tvtid[i], 3][0])

        image1 = io.imread(image_file1)
        image2 = io.imread(image_file2)

        gray1 = rgb2gray(image1)
        gray2 = rgb2gray(image2)

        if self.transform:

            image1 = self.transform(Image.fromarray(image1))  #
            image2 = self.transform(Image.fromarray(image2))  #


        else:
            image1 = torch.from_numpy(image1)
            image2 = torch.from_numpy(image2)

        if self.mode == "test":
            if self.aug:
                vf1 = vf(image1)
                vf2 = vf(image2)
                hf1 = hf(image1)
                hf2 = hf(image2)

                pair_normal = torch.cat((image1, image2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_gray = torch.cat((gray1, gray2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_hf = torch.cat((hf1, hf2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_vf = torch.cat((vf1, vf2), dim=0).view(-1, 6, self.dim, self.dim)

                pair = torch.cat((pair_normal, pair_gray, pair_hf, pair_vf), dim=0)

                label = torch.LongTensor(
                    np.full((4), int(self.meta_data['pairs'][self.tvtid[i], 1]), dtype=int).tolist())
                sample = {'pair': pair, 'label': label}
            else:
                pair = torch.cat((image1, image2), dim=0)
                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                sample = {'pair': pair, 'label': label}
        else:
            if self.aug:
                vf1 = vf(image1)
                vf2 = vf(image2)
                hf1 = hf(image1)
                hf2 = hf(image2)

                c = torch.cat((image1, image2), dim=0)
                pair_normal = c.view(-1, 6, self.dim, self.dim)
                pair_gray = torch.cat((gray1, gray2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_hf = torch.cat((hf1, hf2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_vf = torch.cat((vf1, vf2), dim=0).view(-1, 6, self.dim, self.dim)

                pair = torch.cat((pair_normal, pair_gray, pair_hf, pair_vf), dim=0)

                label = torch.LongTensor(
                    np.full((4), int(self.meta_data['pairs'][self.tvtid[i], 1]),
                            dtype=int).tolist())  # if not self.test else None
                sample = {'pair': pair, 'label': label}
            else:
                pair = torch.cat((image1, image2), dim=0).view(-1, 6, self.dim, self.dim)
                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                sample = {'pair': pair, 'label': label}
        return sample


class KinShipDataSetAllAug(Dataset):
    rel_lookup = {'fd': 'father-dau', 'fs': 'father-son', 'md': 'mother-dau', 'ms': 'mother-son', 'all': 'all'}

    def __init__(self, relation, mode, image_path, meta_data_path, u_index, transform=None, aug=False):
        self.meta_data = sio.loadmat(os.path.join(meta_data_path, relation + '_pairs.mat'))
        self.relation = relation
        self.transform = transform
        self.dim = 64
        self.image_path = image_path
        self.meta_data_path = meta_data_path
        self.aug = aug
        self.mode = mode


        self.tvtid = u_index
        self.tvtlen = len(self.tvtid)

    def __len__(self):
        return self.tvtlen

    def __getitem__(self, i):

        strlist = self.meta_data_path.split('/')
        data_name = strlist[2]
        if data_name == 'UBKinFace':
            s1 = self.meta_data['pairs'][self.tvtid[i], 2][0]
            s2 = self.meta_data['pairs'][self.tvtid[i], 3][0]
            s1 = s1.replace('_new', '')
            s2 = s2.replace('_new', '')
            if self.relation == 'set1':
                image_file1 = os.path.join(self.image_path, '03' + '/' + s1)
                image_file2 = os.path.join(self.image_path, '01' + '/' + s2)
            elif self.relation == 'set2':
                image_file1 = os.path.join(self.image_path, '02' + '/' + s1)
                image_file2 = os.path.join(self.image_path, '01' + '/' + s1)
        else:
            if data_name == 'TSKinFace' or data_name == 'CornellKinFace':
                folder = KinShipDataSetAllAug.rel_lookup[self.relation]
            else:
                folder = KinShipDataSetAllAug.rel_lookup[self.meta_data['pairs'][self.tvtid[i], 2][0][:2]]
            image_file1 = os.path.join(self.image_path,
                                       folder + '/' + self.meta_data['pairs'][self.tvtid[i], 2][0])
            image_file2 = os.path.join(self.image_path,
                                       folder + '/' + self.meta_data['pairs'][self.tvtid[i], 3][0])
            filename1 = self.meta_data['pairs'][self.tvtid[i], 2][0]
            filename2 = self.meta_data['pairs'][self.tvtid[i], 3][0]
            filena1 = filename1.split('.')[0]
            filena2 = filename2.split('.')[0]

            image_file10 = os.path.join(self.image_path,
                                        folder + '/after/' + filena1 + "_0.jpg")
            image_file20 = os.path.join(self.image_path,
                                        folder + '/after/' + filena2 + "_0.jpg")
            image_file11 = os.path.join(self.image_path,
                                        folder + '/after/' + filena1 + "_1.jpg")
            image_file21 = os.path.join(self.image_path,
                                        folder + '/after/' + filena2 + "_1.jpg")
            image_file12 = os.path.join(self.image_path,
                                        folder + '/after/' + filena1 + "_2.jpg")
            image_file22 = os.path.join(self.image_path,
                                        folder + '/after/' + filena2 + "_2.jpg")
            image_file13 = os.path.join(self.image_path,
                                        folder + '/after/' + filena1 + "_3.jpg")
            image_file23 = os.path.join(self.image_path,
                                        folder + '/after/' + filena2 + "_3.jpg")
            image_file14 = os.path.join(self.image_path,
                                        folder + '/after/' + filena1 + "_4.jpg")
            image_file24 = os.path.join(self.image_path,
                                        folder + '/after/' + filena2 + "_4.jpg")

        image1 = io.imread(image_file1)
        image2 = io.imread(image_file2)

        image10 = io.imread(image_file10)
        image20 = io.imread(image_file20)
        image11 = io.imread(image_file11)
        image21 = io.imread(image_file21)
        image12 = io.imread(image_file12)
        image22 = io.imread(image_file22)
        image13 = io.imread(image_file13)
        image23 = io.imread(image_file23)
        image14 = io.imread(image_file14)
        image24 = io.imread(image_file24)
        gray1 = rgb2gray(image1)
        gray2 = rgb2gray(image2)

        if self.transform:

            image1 = self.transform(Image.fromarray(image1))  #
            image2 = self.transform(Image.fromarray(image2))  #
            image10 = self.transform(Image.fromarray(image10))  #
            image20 = self.transform(Image.fromarray(image20))  #
            image11 = self.transform(Image.fromarray(image11))  #
            image21 = self.transform(Image.fromarray(image21))  #
            image12 = self.transform(Image.fromarray(image12))  #
            image22 = self.transform(Image.fromarray(image22))  #
            image13 = self.transform(Image.fromarray(image13))  #
            image23 = self.transform(Image.fromarray(image23))  #
            image14 = self.transform(Image.fromarray(image14))  #
            image24 = self.transform(Image.fromarray(image24))  #


        else:
            image1 = torch.from_numpy(image1)
            image2 = torch.from_numpy(image2)
            image10 = torch.from_numpy(image10)
            image20 = torch.from_numpy(image20)
            image11 = torch.from_numpy(image11)
            image21 = torch.from_numpy(image21)
            image12 = torch.from_numpy(image12)
            image22 = torch.from_numpy(image22)
            image13 = torch.from_numpy(image13)
            image23 = torch.from_numpy(image23)
            image14 = torch.from_numpy(image14)
            image24 = torch.from_numpy(image24)
        if self.mode == "test":
            if self.aug:
                vf1 = vf(image1)
                vf2 = vf(image2)
                hf1 = hf(image1)
                hf2 = hf(image2)

                pair_normal = torch.cat((image1, image2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_gray = torch.cat((gray1, gray2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_hf = torch.cat((hf1, hf2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_vf = torch.cat((vf1, vf2), dim=0).view(-1, 6, self.dim, self.dim)

                pair = torch.cat((pair_normal, pair_gray, pair_hf, pair_vf), dim=0)

                label = torch.LongTensor(
                    np.full((4), int(self.meta_data['pairs'][self.tvtid[i], 1]), dtype=int).tolist())
                sample = {'pair': pair, 'label': label}
            else:
                pair = torch.cat((image1, image2), dim=0)
                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                sample = {'pair': pair, 'label': label}
        else:
            if self.aug:
                vf1 = vf(image1)
                vf2 = vf(image2)
                hf1 = hf(image1)
                hf2 = hf(image2)

                c = torch.cat((image1, image2), dim=0)
                pair_normal = c.view(-1, 6, self.dim, self.dim)
                pair_gray = torch.cat((gray1, gray2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_hf = torch.cat((hf1, hf2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_vf = torch.cat((vf1, vf2), dim=0).view(-1, 6, self.dim, self.dim)
                pair_normal0 = torch.cat((image10, image20), dim=0).view(-1, 6, self.dim, self.dim)
                pair_normal1 = torch.cat((image11, image21), dim=0).view(-1, 6, self.dim, self.dim)
                pair_normal2 = torch.cat((image12, image22), dim=0).view(-1, 6, self.dim, self.dim)
                pair_normal3 = torch.cat((image13, image23), dim=0).view(-1, 6, self.dim, self.dim)
                pair_normal4 = torch.cat((image14, image24), dim=0).view(-1, 6, self.dim, self.dim)

                pair = torch.cat((pair_normal, pair_normal0, pair_normal1, pair_normal2, pair_normal3, pair_normal4,
                                  pair_gray, pair_hf, pair_vf), dim=0)

                label = torch.LongTensor(
                    np.full((9), int(self.meta_data['pairs'][self.tvtid[i], 1]),
                            dtype=int).tolist())  # if not self.test else None
                sample = {'pair': pair, 'label': label}
            else:
                pair = torch.cat((image1, image2), dim=0).view(-1, 6, self.dim, self.dim)
                label = torch.LongTensor([int(self.meta_data['pairs'][self.tvtid[i], 1])])
                sample = {'pair': pair, 'label': label}
        return sample
