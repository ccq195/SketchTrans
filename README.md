This project is our implementation of "Sketch Transformer: Asymmetrical Disentanglement Learning from Dynamic Synthesis" (ACM MM22) and the extension that is submitted to PAMI.
 # 1. Prepare Datasets
 (1) Category-level datasets: ![link](https://github.com/huangzongheng/MATHM) 

 (2)Instance-level Datasets: PKU-Sketch dataset: https://www.pkuml.org/resources/pkusketchreid-dataset.html;
 QMUL: https://sketchx.eecs.qmul.ac.uk/downloads/

 # 2. Train and Test Instruction
  (1) Train and Test for category-level sketch-photo recognition: sh ./category_dist_train.sh;
  
  (2)Train and Test for instance-level sketch-photo recognition: sh ./instance_dist_train.sh.

# 3. Trained Models

  Our trained models can be downloaded from ![baidu netdisk]() (the extraction code is go78).

# 4. Citation
@inproceedings{chen2022sketch,
  title={Sketch Transformer: Asymmetrical Disentanglement Learning from Dynamic Synthesis},
  author={Chen, Cuiqun and Ye, Mang and Qi, Meibin and Du, Bo},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={4012--4020},
  year={2022}
}

# 5. License
The code is distributed under the MIT License. See LICENSE for more information.
