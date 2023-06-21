from PIL import Image
import os.path as osp
from collections import  defaultdict
from .bases import BaseImageDataset
import numpy as np

class PKU(BaseImageDataset):
    def __init__(self,root,trial):
        super(PKU,self).__init__()
        self.train = {}
        self.train = defaultdict(list)
        self.train2 = []
        self.gallery = []
        self.query = []
        train_visible_path = root+'chen_idx2/train_visible_{}.txt'.format(trial)
        train_sketch_path = root+'chen_idx2/train_sketch_{}.txt'.format(trial)
        test_visible_path = root+'chen_idx2/test_visible_{}.txt'.format(trial)
        test_sketch_path = root+'chen_idx2/test_sketch_{}.txt'.format(trial)

        #trainset RGB-0 sketch-1
        data_file_list = open(train_visible_path, 'rt').read().splitlines()
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        pid2label_img = {pid: label for label, pid in enumerate(np.unique(file_label))}
        file_cam = [int(s.split(' ')[2]) for s in data_file_list]
        pid2cam_img = {pid: label for label, pid in enumerate(np.unique(file_cam))}

        for j in range(len(data_file_list)):
            img = root+data_file_list[j].split(' ')[0]
            id = int(data_file_list[j].split(' ')[1])
            cam = int(data_file_list[j].split(' ')[2])
            sty = int(data_file_list[j].split(' ')[3])
            self.train2.append((img, pid2label_img[id], 0))
            self.train[pid2label_img[id]].append((img, pid2label_img[id], 0))

        data_file_list = open(train_sketch_path, 'rt').read().splitlines()
        for j in range(len(data_file_list)):
            img = root+data_file_list[j].split(' ')[0]
            id = int(data_file_list[j].split(' ')[1])
            cam = int(data_file_list[j].split(' ')[2])
            sty = int(data_file_list[j].split(' ')[3])
            self.train2.append((img, pid2label_img[id], 1))
            self.train[pid2label_img[id]].append((img, pid2label_img[id], 1))

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train2)

        #query set
        data_file_list = open(test_sketch_path, 'rt').read().splitlines()
        for j in range(len(data_file_list)):
            img = root+data_file_list[j].split(' ')[0]
            id = int(data_file_list[j].split(' ')[1])
            cam = int(data_file_list[j].split(' ')[2])
            sty = int(data_file_list[j].split(' ')[3])
            self.query.append((img, id, 1))
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)

        #gallery set
        data_file_list = open(test_visible_path, 'rt').read().splitlines()
        file_cam = [int(s.split(' ')[2]) for s in data_file_list]
        pid2cam_img = {pid: label for label, pid in enumerate(np.unique(file_cam))}
        for j in range(len(data_file_list)):
            img = root+data_file_list[j].split(' ')[0]
            id = int(data_file_list[j].split(' ')[1])
            cam = int(data_file_list[j].split(' ')[2])
            sty = int(data_file_list[j].split(' ')[3])
            self.gallery.append((img, id, 0))
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams  = self.get_imagedata_info(self.gallery)

