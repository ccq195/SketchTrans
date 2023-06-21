import imp
import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from processor.data import tensor_to_img
import numpy as np

def do_train(cfg,
             model,
             train_loader,
             query_loader,
             gallery_loader,
             optimizer,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)


    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(cfg, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    best_rank = 0
    best_rank2 = 0
    best_rank3 = 0
    best_map = 0
    best_map3 = 0
    wg = 0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
   
        scheduler.step(epoch)
        model.train()
        for n_iter, data in enumerate(train_loader):
            # img, vid, target_cam, target_view = data
            # sid, simg, img, target, target_cam = data
            simg, img, target, target_cam = data

            if cfg.DATASETS.NAMES in ['tuberlin', 'sketchy']:
                simg = simg[target_cam==0].to(device)
                img = img.to(device)
            else:
                b,t,c,h,w = img.size()
                img = img.view(-1,c,h,w)
                img = img.to(device)
                simg = simg.to(device) # pku chairv2 dataset
            
            target = target.view(-1).to(device)
            target_cam = target_cam.view(-1).to(device)
            
            optimizer.zero_grad()

            with amp.autocast(enabled=True):
                score, feat, sdiff, reconloss = model(simg, img, target, target_cam, cfg=cfg)
                target = torch.cat((target,target[target_cam==0]),dim=0)
                loss = loss_fn(score, feat, sdiff, reconloss, target)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()


            # if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
            #     for param in center_criterion.parameters():
            #         param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                # optimizer.step()

                # scaler.step(optimizer_center)
                # scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        # wg = 1. / (1. + loss_meter.avg)
        # print(wg)

        end_time = time.time()
        # time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done.  loss:{:.3f} Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch,loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            gfeat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((gfeat, vid, camid))  # camid no tensor

                        cmc, mAP, _, _, _, _, _ = evaluator.compute()

                        logger.info("Validation Results - Epoch: {}".format(epoch))
                        logger.info("mAP: {:.1%}".format(mAP))
                        for r in [1, 5, 10]:
                            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                        logger.info("mAP2: {:.1%}".format(mAP2))
                        for r in [1, 5, 10]:
                            logger.info("CMC2 curve gfeat, Rank-{:<3}:{:.1%}".format(r, cmc2[r - 1]))

                        logger.info("mAP3: {:.1%}".format(mAP3))
                        for r in [1, 5, 10]:
                            logger.info("CMC3 curve lfeat, Rank-{:<3}:{:.1%}".format(r, cmc3[r - 1]))

                        if cmc[0] > best_rank:
                            best_rank = cmc[0]
                            torch.save(model.state_dict(),
                                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))

                        if cmc2[0] > best_rank2:
                            best_rank2 = cmc2[0]
                            torch.save(model.state_dict(),
                                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_global_best.pth'))

                        if cmc3[0] > best_rank3:
                            best_rank3 = cmc3[0]
                            torch.save(model.state_dict(),
                                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_local_best.pth'))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, data in enumerate(query_loader):
                    img, vid, camid = data
                    with torch.no_grad():
                        img = img.to(device)
                        camid = camid.to(device)
                        gfeat = model(img, img, cam_label=camid)
                        evaluator.update((gfeat, vid, camid))#camid no tensor
                       
                for n_iter, data in enumerate(gallery_loader):
                    img, vid, camid = data
                    with torch.no_grad():
                        img = img.to(device)
                        camid = camid.to(device)
                        gfeat = model(img, img, cam_label=camid)
                        evaluator.update((gfeat, vid, camid))#camid no tensor
                       

                if cfg.DATASETS.NAMES in ['tuberlin', 'sketchy']:
                    m_m, m_s, p_m, p_s = evaluator.compute()

                    logger.info("Validation Results - Epoch: {}".format(epoch))

                    logger.info('gfeat Average mAP: {} {}'.format(str(m_m)[:5], str(m_s)[:5]))
                    logger.info('gfeat Average precision: {} {}'.format(str(p_m)[:5], str(p_s)[:5]))

                    if m_m > best_map:
                        best_map = m_m
                        torch.save(model.state_dict(),
                                os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_map_global_best.pth'))
                else:
                    cmc, mAP = evaluator.compute()

                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                    if cmc[0] > best_rank:
                        best_rank = cmc[0]
                        torch.save(model.state_dict(),
                                os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_r1_global_best.pth'))

                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 query_loader,
                 gallery_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(cfg, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, data in enumerate(query_loader):
        img, vid, camid = data
        with torch.no_grad():
            img = img.to(device)
            camid = camid.to(device)
            gfeat = model(img, img, cam_label=camid)
            evaluator.update((gfeat, vid, camid))#camid no tensor

    for n_iter, data in enumerate(gallery_loader):
        img, vid, camid = data
        with torch.no_grad():
            img = img.to(device)
            camid = camid.to(device)
            gfeat = model(img, img, cam_label=camid)
            evaluator.update((gfeat, vid, camid))#camid no tensor
    
    
    if cfg.DATASETS.NAMES in ['tuberlin', 'sketchy']:
        m_m, m_s, p_m, p_s = evaluator.compute()
        # logger.info('gfeat Average mAP: {} {}'.format(str(m_m)[:5], str(m_s)[:5]))
        # logger.info('gfeat Average precision: {} {}'.format(str(p_m)[:5], str(p_s)[:5]))
        return  m_m, m_s, p_m, p_s
    else:
        cmc, mAP = evaluator.compute()
        # logger.info("mAP: {:.1%}".format(mAP))
        # for r in [1, 5, 10]:
        #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))     
    
        return mAP, cmc[0], cmc[4], cmc[9], cmc[19]


