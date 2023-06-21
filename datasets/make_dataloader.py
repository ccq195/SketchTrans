import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler

from .bases import ImageDataset, ImageDataset_PKU, ImageDataset_QMUL
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .pku import PKU
from .qmul_shoev2 import SHOE
from .qmul_chairv2 import CHAIR
from .sketchy import SketchyDataset
from .tuberlin import TuberlinDataset
from .sampler2 import RandomIdentitySampler, CrossModalityIdentitySampler, NormTripletSampler

__factory = {
    'pku':PKU,
    'shoev2': SHOE,
    'chairv2':CHAIR,
    'sketchy': SketchyDataset,
     'tuberlin': TuberlinDataset,
}


def train_collate_fn_pku(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    c_imgs1, c_imgs2, s_imgs, c_pids1,c_pids2,s_pids, c_camids1, c_camids2, s_camids, c_viewids1, c_viewids2, s_viewids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn_pku(batch):
    imgs, pids, camids = zip(*batch)
    # if not isinstance(camids[0], int):
    imgs = torch.stack(imgs, dim=0)
    # b, t, c, h, w = imgs.size()
    # imgs = imgs.view(-1, c, h, w)
        # viewids = torch.stack(viewids,dim=0).view(-1)
        # viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)

    return imgs, pids, camids_batch
    # else:
    #     # viewids = torch.tensor(viewids, dtype=torch.int64)
    # camids_batch = torch.tensor(camids, dtype=torch.int64)

    # return torch.stack(imgs, dim=0), pids, camids, camids_batch
    
def val_collate_fn(batch):
    imgs, pids, camids = zip(*batch)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids_batch

def collate_fn(batch):  # img, label, cam_id, img_path, img_id
    samples = list(zip(*batch))
    data = [torch.stack(x, 0) for i, x in enumerate(samples)]
    return data

def collate_fn_test(batch):  # img, label, cam_id, img_path, img_id
    samples = list(zip(*batch))
    data = [torch.stack(x, 0) for i, x in enumerate(samples) if i !=1]
    data.insert(1, samples[1])
    return data

def make_dataloader(cfg,t):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.Grayscale(num_output_channels=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),

        ])
    strain_transforms = T.Compose([
            T.Resize([512,512], interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])
    
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DATASETS.NAMES=='pku':
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR,trial=t)
        train_set = ImageDataset_PKU(dataset.train, train_transforms, strain_transforms, cfg)
        num_classes = dataset.num_train_pids
        cam_num = dataset.num_train_cams
        if cfg.MODEL.VIEW==2:
            view_num = 3
        else:
            view_num = 2 #0,1,2

        if cfg.MODEL.DIST_TRAIN:
            train_loader = DataLoader(
                train_set, sampler=DistributedSampler(train_set), batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=num_workers)
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers)

        query_set = ImageDataset(dataset.query, val_transforms, strain_transforms)
        gallery_set = ImageDataset(dataset.gallery, val_transforms, strain_transforms)

        query_loader = DataLoader(
            query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn_pku
        )
        gallery_loader = DataLoader(
            gallery_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn_pku
        )
        num_query = len(dataset.query)
        
    elif cfg.DATASETS.NAMES in ['tuberlin', 'sketchy']: #zero-shot datasets
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        p_size = 16
        k_size = 8
        train_dataset = __factory[cfg.DATASETS.NAMES](mode='train', transform=train_transforms, stransform=strain_transforms)
        # sampler
        batch_size = p_size * k_size
        sampler = RandomIdentitySampler(train_dataset, p_size * k_size, k_size)
        # loader
        train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, drop_last=True, pin_memory=True,
                                collate_fn=collate_fn, num_workers=num_workers)

        gallery_dataset = __factory[cfg.DATASETS.NAMES](mode='gallery', transform=val_transforms)
        query_dataset = __factory[cfg.DATASETS.NAMES](mode='query', transform=val_transforms)
        # dataloader
        query_loader = DataLoader(dataset=query_dataset,
                                batch_size=cfg.TEST.IMS_PER_BATCH,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn_test,
                                num_workers=num_workers)

        gallery_loader = DataLoader(dataset=gallery_dataset,
                                    batch_size=cfg.TEST.IMS_PER_BATCH,
                                    shuffle=False,
                                    pin_memory=True,
                                    drop_last=False,
                                    collate_fn=collate_fn_test,
                                    num_workers=num_workers)


        num_classes = train_dataset.num_ids
        cam_num = len(set(train_dataset.cam_ids))
        num_query = len(query_dataset.img_paths)
        
    else: #'chairv2, shoev2'
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR,trial=t)
        train_set = ImageDataset_QMUL(dataset.train, train_transforms, strain_transforms, cfg)
        num_classes = dataset.num_train_pids
        cam_num = dataset.num_train_cams

        if cfg.MODEL.VIEW==2:
            view_num = 3
        else:
            view_num = 2 #0,1,2

        if cfg.MODEL.DIST_TRAIN:
            train_loader = DataLoader(
                    train_set, sampler=DistributedSampler(train_set), batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=num_workers)
        else:
            train_loader = DataLoader(
                    train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers)


        query_set = ImageDataset(dataset.query, val_transforms, strain_transforms)
        gallery_set = ImageDataset(dataset.gallery, val_transforms, strain_transforms)

        query_loader = DataLoader(
            query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
                collate_fn=val_collate_fn
            )
        gallery_loader = DataLoader(
                gallery_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
                collate_fn=val_collate_fn
            )
        num_query = len(dataset.query)
        
    return train_loader, query_loader, gallery_loader, num_query, num_classes, cam_num
