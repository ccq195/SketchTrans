import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/PKU/vit_base_aux_chair.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, query_loader, gallery_loader, num_query, num_classes, camera_num = make_dataloader(cfg,cfg.DATASETS.TRIAL)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num)

    # WEIGHT = cfg.TEST.RESUME + '/' + cfg.TEST.WEIGHT
    # model.load_param(WEIGHT)

    if cfg.DATASETS.NAMES in ['pku', 'sketchy', 'tuberlin']:
        
        if cfg.DATASETS.NAMES == 'pku':
            test_num = 10
            for trial in range(test_num):
                trial = trial+1
                WEIGHT = cfg.TEST.RESUME + '/'+'{}'.format(trial) + '/' + cfg.TEST.WEIGHT
                model.load_param(WEIGHT)
                print('test: ', trial)
                print('load weights: ',  WEIGHT)
                # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg,trial)
                train_loader, query_loader, gallery_loader, num_query, num_classes, camera_num = make_dataloader(
                    cfg,trial)

                map, rank1, rank5, rank10,rank20 = do_inference(cfg,
                    model,
                    query_loader, gallery_loader,
                    num_query)
                if trial == 1:
                    all_rank_1 = rank1
                    all_rank_5 = rank5
                    all_rank_10 = rank10
                    all_rank_20 = rank20
                    all_map = map
                else:
                    all_rank_1 = all_rank_1 + rank1
                    all_rank_5 = all_rank_5 + rank5
                    all_rank_10 = all_rank_10 + rank10
                    all_rank_20 = all_rank_20 + rank20
                    all_map = all_map + map

                logger.info("mAP:{}, rank_1:{}, rank_5 {}, rank10:{}, rank20:{}, trial : {}".format(map, rank1, rank5, rank10, rank20, trial))

            logger.info("sum_map:{:.1%}, sum_rank_1:{:.1%}, sum_rank_5 {:.1%},sum_rank_10 {:.1%},sum_rank_20 {:.1%}".format(all_map.sum()/10.0, all_rank_1.sum()/10.0, all_rank_5.sum()/10.0, all_rank_10.sum()/10.0, all_rank_20.sum()/10.0))
        else:
            test_num = 5
            for trial in range(test_num):
                trial = trial+1

                WEIGHT = cfg.TEST.RESUME + '/'+'{}'.format(trial) + '/' + cfg.TEST.WEIGHT
                model.load_param(WEIGHT)
                print('test: ', trial)
                print('load weights: ',  WEIGHT)
                # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg,trial)
                train_loader, query_loader, gallery_loader, num_query, num_classes, camera_num = make_dataloader(
                    cfg,trial)

                m_m, m_s, p_m, p_s = do_inference(cfg, model, query_loader, gallery_loader, num_query)
                if trial == 1:
                    all_rank_1 = m_m
                    all_rank_5 = m_s
                    all_rank_10 = p_m
                    all_rank_20 = p_s
                else:
                    all_rank_1 = all_rank_1 + m_m
                    all_rank_5 = all_rank_5 + m_s
                    all_rank_10 = all_rank_10 + p_m
                    all_rank_20 = all_rank_20 + p_s
                    
                logger.info('gfeat mAP: {} {}'.format(str(m_m)[:5], str(m_s)[:5]))
                logger.info('gfeat precision: {} {}'.format(str(p_m)[:5], str(p_s)[:5]))

            logger.info('gfeat Average mAP: {} {}'.format(str(all_rank_1.sum()/5.0)[:5], str(all_rank_5.sum()/5.0)[:5]))
            logger.info('gfeat Average precision: {} {}'.format(str(all_rank_10.sum()/5.0)[:5], str(all_rank_20.sum()/5.0)[:5]))
            

    else:
       do_inference(cfg,
                 model,
                 query_loader, gallery_loader,
                 num_query)

