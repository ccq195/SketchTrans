import os
from PIL import Image
import os.path as osp
from collections import  defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
import pickle

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

class SketchyDataset(Dataset):
    def __init__(self, mode='train', transform=None,  stransform=None):
        assert mode in ['train', 'gallery', 'query']

        if mode == 'train':
            file_ls_p = '../Sketchy/zeroshot1/all_photo_filelist_train.txt'
            file_ls_s = '../Sketchy/zeroshot1/sketch_tx_000000000000_ready_filelist_train.txt'
            file_ls_s2 = '../Sketchy/zeroshot1/sketch_tx_000000000000_ready_filelist_test.txt'
            with open(file_ls_p, 'r') as fh:
                file_contentp = fh.readlines()
            with open(file_ls_s, 'r') as fh:
                file_contents = fh.readlines()
            with open(file_ls_s2, 'r') as fh:
                file_contents2 = fh.readlines()
        
            img_paths= np.array([' '.join(ff.strip().split()[:-1]) for ff in file_contentp] + [' '.join(ff.strip().split()[:-1]) for ff in file_contents] + [' '.join(ff.strip().split()[:-1]) for ff in file_contents2])
            img_paths = [os.path.join('../Sketchy', f) for f in img_paths]
            selected_ids = np.array([int(ff.strip().split()[-1]) for ff in file_contentp] + [int(ff.strip().split()[-1]) for ff in file_contents] + [int(ff.strip().split()[-1]) for ff in file_contents2])
            num_ids = len(set(selected_ids))
            cam_ids =  np.array([0 for ff in file_contentp] + [1 for ff in file_contents] + [1 for ff in file_contents2])
            print('sketchy training: {}'.format(len(img_paths)))
        else:
            if mode == "gallery":
                file_ls = '../Sketchy/zeroshot1/all_photo_filelist_zero.txt'
                with open(file_ls, 'r') as fh:
                    file_content = fh.readlines()
                img_paths= np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
                img_paths = [os.path.join('../Sketchy', f) for f in img_paths]
                selected_ids = np.array([int(ff.strip().split()[-1]) for ff in file_content])
                cam_ids =  np.array([0 for ff in file_content])
                num_ids = len(selected_ids)
                print('sketchy query: {}'.format(len(img_paths)))
            elif mode == 'query':
                file_ls = '../Sketchy/zeroshot1/sketch_tx_000000000000_ready_filelist_zero.txt'
                with open(file_ls, 'r') as fh:
                    file_content = fh.readlines()
                img_paths= np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
                img_paths = [os.path.join('../Sketchy', f) for f in img_paths]
                selected_ids = np.array([int(ff.strip().split()[-1]) for ff in file_content])
                cam_ids =  np.array([1 for ff in file_content])
                num_ids = len(selected_ids)
                print('sketchy query: {}'.format(len(img_paths)))

        self.img_paths = img_paths
        # self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths] # 0 photo 1 sketch
        self.cam_ids = cam_ids
        self.num_ids = num_ids
        self.transform = transform
        self.stransform = stransform
        self.ids = selected_ids
        self.mode = mode
        print('finished')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):

        path = self.img_paths[item]
        img = read_image(path)

        if self.mode == 'train':
            label =self.ids[item]
            cam = self.cam_ids[item]

            img2 = self.stransform(img)
            imgg = self.transform(img)

            return img2, imgg, torch.from_numpy(np.array(label)), torch.from_numpy(np.array(cam)), torch.from_numpy(np.array(cam))

        else:
            label = self.ids[item]
            cam = self.cam_ids[item]
            img = self.transform(img)
            return img, label, torch.from_numpy(np.array(cam))

