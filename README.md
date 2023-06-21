This project is our implementation of Sketch Transformer: Asymmetrical Disentanglement Learning from Dynamic Synthesis (ACM MM2022) and Disentangled Prototype Learning for Sketch-Photo
Recognition (Submitted to PAMI).


# 1. Prepare Datasets
(1) Category-level datasets: ![Link](https://github.com/huangzongheng/MATHM)

(2) Instance-level datasets: PKU-sketch dataset from https://www.pkuml.org/resources/pkusketchreid-dataset.html; 
                             QMUL dataset from https://sketchx.eecs.qmul.ac.uk/downloads/.

# 2. Train
(1) For zero-shot category-level sketch-photo recognition: Run category_train.sh (The final result is the average of 5 experiments)
(2) For instance-level sketch-photo recognition: Run instance_train.sh (for PKU-Sketch dataset, the final result is the average of 10 experiments)
