from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import pickle

import  numpy as np
import scipy.ndimage
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]

        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        # num_views = len(tracks)
        
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")



class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None,stransform=None, transform2=None):
        self.dataset = dataset
        self.transform = transform
        self.stransform = stransform
        self.transform2 = transform2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            cimg = self.transform(img)
            # if 'testB' in img_path.split('/'):
            #     simg = self.stransform(img)
            #     return cimg, simg, pid, camid, trackid
            # else:
            return cimg, pid, camid

import json, os
class ImageDataset_QMUL(Dataset):
    def __init__(self, dataset, transform=None, stransform=None, transform2=None, cfg=None):
        self.dataset = dataset
        self.transform = transform
        self.transform2 = transform2
        # self.view = cfg.MODEL.VIEW
        # self.aux = cfg.MODEL.AUX
        self.stransform = stransform
        # if cfg.DATASETS.NAMES == 'chairv2': ## initial a-sketch 
        #     self.pyfile = np.load('./datasets/chair_g.npy', allow_pickle=True).item()
        # else:
        #     self.pyfile = np.load('./datasets/shoe_g.npy', allow_pickle=True).item()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        c_img_path, c_pid, c_camid, c_trackid = self.dataset[index][0]
        c_img = read_image(c_img_path)
        # sid = self.pyfile[os.path.join(c_img_path.split('/')[-2],c_img_path.split('/')[-1])]
        # sid.requires_grad = False

        if len(self.dataset[index][0:]):
            n = random.randint(1,len(self.dataset[index][0:])-1)
            s_img_path, s_pid, s_camid, s_trackid = self.dataset[index][n]
            assert s_trackid==1
            s_img = read_image(s_img_path)

        if self.transform is not None:
            c_imgs = self.transform(c_img)
            cc_img = self.stransform(c_img)
            s_img = self.transform(s_img)

        return  cc_img, torch.stack([c_imgs,s_img],dim=0), torch.from_numpy(np.array([c_pid,s_pid])), torch.from_numpy(np.array([c_camid,s_camid]))


class ImageDataset_PKU(Dataset):
    def __init__(self, dataset, transform=None, strain_transforms=None, cfg=None):
        self.dataset = dataset
        self.transform = transform
        self.view = cfg.MODEL.VIEW
        self.aux = cfg.MODEL.AUX
        self.stransform = strain_transforms
        # self.pyfile = np.load('./datasets/pku_g.npy', allow_pickle=True).item()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        c_img_path_1, c_pid_1, c_camid_1 = self.dataset[index][0]
        c_img_path_2, c_pid_2, c_camid_2 = self.dataset[index][1]
        s_img_path, s_pid, s_camid = self.dataset[index][2]

        s_img = read_image(s_img_path)

        if bool(random.getrandbits(1)):
            c_img_path, c_pid, c_camid = c_img_path_1, c_pid_1, c_camid_1 
            c_img = read_image(c_img_path)
            # sid = self.pyfile[os.path.join(c_img_path.split('/')[-2],c_img_path.split('/')[-1])]
            # sid.requires_grad = False

        else:
            c_img_path, c_pid, c_camid = c_img_path_2, c_pid_2, c_camid_2
            c_img = read_image(c_img_path)
            # sid = self.pyfile[os.path.join(c_img_path.split('/')[-2],c_img_path.split('/')[-1])]
            # sid.requires_grad = False
      
        if self.transform is not None:
            c_imgs = self.transform(c_img)
            s_img = self.transform(s_img)
            cc_img = self.stransform(c_img)

        return  cc_img, torch.stack([c_imgs,s_img],dim=0), torch.from_numpy(np.array([c_pid,s_pid])), torch.from_numpy(np.array([c_camid,s_camid]))

