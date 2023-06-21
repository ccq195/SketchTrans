from PIL import Image
import os.path as osp

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


class MultiDateset(object):
    def __init__(self,train_path=None,query_path=None,gallery_path=None,regdb_v=False,regdb_r=False):
        super(MultiDateset,self).__init__()
        self.train = []
        self.gallery = []
        self.query = []
        if train_path!=None:
            data_file_list = open(train_path, 'rt').read().splitlines()
            for j in range(len(data_file_list)):
                img = data_file_list[j].split(' ')[0]
                id = int(data_file_list[j].split(' ')[1])
                cam = int(data_file_list[j].split(' ')[2])
                self.train.append((img,id,cam,1))
            self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids  = self.get_imagedata_info(self.train)

        if query_path!=None:
            data_file_list = open(query_path, 'rt').read().splitlines()
            if regdb_r:
                for j in range(len(data_file_list)):
                    img = data_file_list[j].split(' ')[0]
                    id = int(data_file_list[j].split(' ')[1])
                    cam = 2
                    self.query.append((img, id, cam,1))
                self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
            else:
                for j in range(len(data_file_list)):
                    img = data_file_list[j].split(' ')[0]
                    id = int(data_file_list[j].split(' ')[1])
                    cam = int(data_file_list[j].split(' ')[2])
                    self.query.append((img, id, cam,1))
                self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)


        if gallery_path!=None:
            data_file_list = open(gallery_path, 'rt').read().splitlines()
            if regdb_v:
                for j in range(len(data_file_list)):
                    img = data_file_list[j].split(' ')[0]
                    id = int(data_file_list[j].split(' ')[1])
                    cam = 1
                    self.gallery.append((img, id, cam,1))
                self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids  = self.get_imagedata_info(self.gallery)
            else:
                for j in range(len(data_file_list)):
                    img = data_file_list[j].split(' ')[0]
                    id = int(data_file_list[j].split(' ')[1])
                    cam = int(data_file_list[j].split(' ')[2])
                    self.gallery.append((img, id, cam,1))
                self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids  = self.get_imagedata_info(self.gallery)

    def get_imagedata_info(self, data):
        pids, cams, tracks= [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]

        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)

        return num_pids, num_imgs, num_cams, num_views

class ImageDateManager(object):
    def __init__(self,dataset,transform=None):
        super(ImageDateManager,self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


