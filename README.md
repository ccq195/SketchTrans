This project is our implementation of Sketch Transformer: Asymmetrical Disentanglement Learning from Dynamic Synthesis (ACM MM2022) and Disentangled Prototype Learning for Sketch-Photo
Recognition (Submitted to PAMI).


# 1. Prepare Datasets
(1) Category-level datasets: ![Link](https://github.com/huangzongheng/MATHM)

(2) Instance-level datasets: PKU-sketch dataset from https://www.pkuml.org/resources/pkusketchreid-dataset.html; 
                             QMUL dataset from https://sketchx.eecs.qmul.ac.uk/downloads/.

# 2. Train and Test
(1) For zero-shot category-level sketch-photo recognition: Run category_train.sh (The final result is the average of 5 experiments)
(2) For instance-level sketch-photo recognition: Run instance_train.sh (for PKU-Sketch dataset, the final result is the average of 10 experiments)

# 3. Trained Models
We provide a set of trained models available for download in ![Link](https://github.com/huangzongheng/MATHM)

# 3. License
The code is distributed under the MIT License. See LICENSE for more information.

# 4. Citation
@inproceedings{chen2022sketch,
  title={Sketch Transformer: Asymmetrical Disentanglement Learning from Dynamic Synthesis},
  author={Chen, Cuiqun and Ye, Mang and Qi, Meibin and Du, Bo},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={4012--4020},
  year={2022}
}
