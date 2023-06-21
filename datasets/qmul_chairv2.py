from PIL import Image
import os.path as osp
from collections import  defaultdict
from .bases import BaseImageDataset
import numpy as np

class CHAIR(BaseImageDataset):
    def __init__(self,root,trial):
        super(CHAIR,self).__init__()
        self.train = {}
        self.train = defaultdict(list)
        self.train2 = []
        self.gallery = []
        self.query = []
        train_visible_path = root+'ChairV2/train_visible.txt'
        train_sketch_path = root+'ChairV2/train_sketch.txt'
        test_visible_path = root+'ChairV2/test_visible.txt'
        test_sketch_path = root+'ChairV2/test_sketch.txt'

        #trainset RGB-0 sketch-1
        data_file_list = open(train_visible_path, 'rt').read().splitlines()
        file_label = [(s.split(' ')[1]) for s in data_file_list]
        pid2label_img = {pid: label for label, pid in enumerate(np.unique(file_label))}
        file_cam = [int(0) for s in data_file_list]

        for j in range(len(data_file_list)):
            img = root+data_file_list[j].split(' ')[0]
            id = (data_file_list[j].split(' ')[1])
            cam = int(0)
            self.train2.append((img, pid2label_img[id], file_cam[cam]))
            self.train[pid2label_img[id]].append((img, pid2label_img[id], file_cam[cam],0))

        data_file_list = open(train_sketch_path, 'rt').read().splitlines()
        for j in range(len(data_file_list)):
            img = root+data_file_list[j].split(' ')[0]
            id = (data_file_list[j].split(' ')[1])
            cam = int(1)
            self.train2.append((img, pid2label_img[id], cam))
            self.train[pid2label_img[id]].append((img, pid2label_img[id], cam,1))

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train2)

        #query set
        data_file_list = open(test_sketch_path, 'rt').read().splitlines()
        for j in range(len(data_file_list)):
            img = root+data_file_list[j].split(' ')[0]
            id = (data_file_list[j].split(' ')[1])
            cam = int(1)
            self.query.append((img, id, cam))
        self.num_query_pids, self.num_query_imgs, self.num_query_cams  = self.get_imagedata_info(self.query)

        #gallery set
        data_file_list = open(test_visible_path, 'rt').read().splitlines()
        for j in range(len(data_file_list)):
            img = root+data_file_list[j].split(' ')[0]
            id = (data_file_list[j].split(' ')[1])
            cam = int(0)
            self.gallery.append((img, id, cam))
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams  = self.get_imagedata_info(self.gallery)

