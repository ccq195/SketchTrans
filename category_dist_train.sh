


### instacne-level datasets
#0
CUDA_VISIBLE_DEVICES='7' python3 train.py --config_file configs/vit_base_aux_tuberlin.yml  MODEL.R 1.0 MODEL.W 0.001 MODEL.G 0.5 OUTPUT_DIR '/data1/ccq/zeroshot_tublerin/sketchtrans_LD_R1W0001G05/0' 
# # #1
CUDA_VISIBLE_DEVICES='7' python3 train.py --config_file configs/vit_base_aux_tuberlin.yml  MODEL.R 1.0 MODEL.W 0.0 MODEL.G 0.5 OUTPUT_DIR '/data1/ccq/zeroshot_tublerin/sketchtrans_LD_R1W0001G05/1' 
#2
CUDA_VISIBLE_DEVICES='7' python3 train.py --config_file configs/vit_base_aux_tuberlin.yml  MODEL.R 1.0 MODEL.W 0.0 MODEL.G 0.5 OUTPUT_DIR '/data1/ccq/zeroshot_tublerin/sketchtrans_LD_R1W0001G05/2' 
#3
CUDA_VISIBLE_DEVICES='7' python3 train.py --config_file configs/vit_base_aux_tuberlin.yml  MODEL.R 1.0 MODEL.W 0.00 MODEL.G 0.5 OUTPUT_DIR '/data1/ccq/zeroshot_tublerin/sketchtrans_LD_R1W0001G05/3' 
#4
CUDA_VISIBLE_DEVICES='7' python3 train.py --config_file configs/vit_base_aux_tuberlin.yml  MODEL.R 1.0 MODEL.W 0.00 MODEL.G 0.5 OUTPUT_DIR '/data1/ccq/zeroshot_tublerin/sketchtrans_LD_R1W0001G05/4' 
#test

CUDA_VISIBLE_DEVICES='7' python3 test.py --config_file configs/vit_base_aux_tuberlin.yml MODEL.R 1.0 MODEL.W 0.0 MODEL.G 0.5 TEST.RESUME '/data1/ccq/zeroshot_tublerin/sketchtrans_LD_R1W0001G05'  OUTPUT_DIR '/data1/ccq/zeroshot_tublerin/sketchtrans_LD_R1W0001G05'  TEST.WEIGHT 'transformer_map_global_best.pth'
