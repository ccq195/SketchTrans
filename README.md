This project is our implementation of "Sketch Transformer: Asymmetrical Disentanglement Learning from Dynamic Synthesis" (ACM MM22) and the extension that is submitted to PAMI.
 
 # 1. Prepare Datasets
 (1) Category-level datasets: [link](https://github.com/huangzongheng/MATHM) 

 (2)Instance-level Datasets: PKU-Sketch dataset: https://www.pkuml.org/resources/pkusketchreid-dataset.html;
 QMUL: https://sketchx.eecs.qmul.ac.uk/downloads/

 # 2. Running Train and Test
  (1) Train and Test for category-level sketch-photo recognition: sh ./category_dist_train.sh;
  
  (2)Train and Test for instance-level sketch-photo recognition: sh ./instance_dist_train.sh.

# 3. Trained Models

  Our trained models and the pretrained weights of the generator G can be downloaded from [baidu netdisk](https://pan.baidu.com/s/1ZCdvq5xBA2hEBG_UfxuJSg) (the extraction code is d28h).

# 4. Citation
@ARTICLE{chen_sketchtrans_tpami,
  author={Chen, Cuiqun and Ye, Mang and Qi, Meibin and Du, Bo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={SketchTrans: Disentangled Prototype Learning With Transformer for Sketch-Photo Recognition}, 
  year={2024},
  volume={46},
  number={5},
  pages={2950-2964},
  keywords={Prototypes;Transformers;Image retrieval;Learning systems;Task analysis;Semantics;Optimization;Asymmetric disentanglement;dynamic synthesis;prototype learning;sketch-photo;recognition},
  doi={10.1109/TPAMI.2023.3337005}}
  
@inproceedings{chen2022sketch,
  title={Sketch transformer: Asymmetrical disentanglement learning from dynamic synthesis},
  author={Chen, Cuiqun and Ye, Mang and Qi, Meibin and Du, Bo},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={4012--4020},
  year={2022}
}

# 5. License
The code is distributed under the MIT License. See LICENSE for more information.
