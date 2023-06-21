

## Pku-Sketch
CUDA_VISIBLE_DEVICES='7' python3 train.py --config_file configs/vit_base_aux_pku.yml DATASETS.TRIAL 1 MODEL.G 0.01 OUTPUT_DIR '/data1/ccq/pku_sketchtrans_test/1' 
CUDA_VISIBLE_DEVICES='0' python3 train.py --config_file configs/vit_base_aux_pku.yml DATASETS.TRIAL 2 MODEL.G 0.01 OUTPUT_DIR '/data1/ccq/pku/sketchtrans_classprototype_average_001contrastiveloss/2' 
CUDA_VISIBLE_DEVICES='0' python3 train.py --config_file configs/vit_base_aux_pku.yml DATASETS.TRIAL 3 MODEL.G 0.01 OUTPUT_DIR '/data1/ccq/pku/sketchtrans_classprototype_average_001contrastiveloss/3' 
CUDA_VISIBLE_DEVICES='0' python3 train.py --config_file configs/vit_base_aux_pku.yml DATASETS.TRIAL 4 MODEL.G 0.01 OUTPUT_DIR '/data1/ccq/pku/sketchtrans_classprototype_average_001contrastiveloss/4' 
CUDA_VISIBLE_DEVICES='0' python3 train.py --config_file configs/vit_base_aux_pku.yml DATASETS.TRIAL 5 MODEL.G 0.01 OUTPUT_DIR '/data1/ccq/pku/sketchtrans_classprototype_average_001contrastiveloss/5' 
CUDA_VISIBLE_DEVICES='0' python3 train.py --config_file configs/vit_base_aux_pku.yml DATASETS.TRIAL 6 MODEL.G 0.01 OUTPUT_DIR '/data1/ccq/pku/sketchtrans_classprototype_average_001contrastiveloss/6'
CUDA_VISIBLE_DEVICES='0' python3 train.py --config_file configs/vit_base_aux_pku.yml DATASETS.TRIAL 7 MODEL.G 0.01 OUTPUT_DIR '/data1/ccq/pku/sketchtrans_classprototype_average_001contrastiveloss/7'
CUDA_VISIBLE_DEVICES='0' python3 train.py --config_file configs/vit_base_aux_pku.yml DATASETS.TRIAL 8 MODEL.G 0.01 OUTPUT_DIR '/data1/ccq/pku/sketchtrans_classprototype_average_001contrastiveloss/8' 
CUDA_VISIBLE_DEVICES='0' python3 train.py --config_file configs/vit_base_aux_pku.yml DATASETS.TRIAL 9 MODEL.G 0.01 OUTPUT_DIR '/data1/ccq/pku/sketchtrans_classprototype_average_001contrastiveloss/9' 
CUDA_VISIBLE_DEVICES='0' python3 train.py --config_file configs/vit_base_aux_pku.yml DATASETS.TRIAL 10 MODEL.G 0.01 OUTPUT_DIR '/data1/ccq/pku/sketchtrans_classprototype_average_001contrastiveloss/10' 

CUDA_VISIBLE_DEVICES='0' python3 test.py --config_file configs/vit_base_aux_pku.yml MODEL.G 0.01 TEST.RESUME '/data1/ccq/pku/sketchtrans_classprototype_average_001contrastiveloss' OUTPUT_DIR '/data1/ccq/pku/sketchtrans_classprototype_average_001contrastiveloss' TEST.WEIGHT 'transformer_best.pth'
CUDA_VISIBLE_DEVICES='0' python3 test.py --config_file configs/vit_base_aux_pku.yml MODEL.G 0.01 TEST.RESUME '/data1/ccq/pku/sketchtrans_classprototype_average_001contrastiveloss' OUTPUT_DIR '/data1/ccq/pku/sketchtrans_classprototype_average_001contrastiveloss' TEST.WEIGHT 'transformer_local_best.pth'



### ChairV2
python train.py --config_file configs/vit_base_aux_chair.yml MODEL.DEVICE_ID "'2'" OUTPUT_DIR '/data1/ccq/chair/B_tri'
python test.py --config_file configs/vit_base_aux_chair.yml MODEL.DEVICE_ID "'2'" TEST.RESUME '/data1/ccq/chair/B_tri' OUTPUT_DIR '/data1/ccq/chair/B_tri' TEST.WEIGHT 'transformer_best.pth'
python test.py --config_file configs/vit_base_aux_chair.yml MODEL.DEVICE_ID "'2'" TEST.RESUME '/data1/ccq/chair/B_tri' OUTPUT_DIR '/data1/ccq/chair/B_tri' TEST.WEIGHT 'transformer_local_best.pth'


