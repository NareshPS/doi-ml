# doi-ml
This repo contains my ML experiments. They are organized as follows:
* image
* text
* common: Techniques like convolution and attention which are used across dataset categories.
* logistics: Related to setting up ML experiments on various platforms.
* evolution: Evolution programs.
* hpo: Hyperparameter Optimization.
* rl: Reinforcement Learning.
* projects: DL projects.

## Notable Projects
* Training the lunar lander in the Open AI Gym using Reinforcement Learning (RL). The project is located at [Lunar Lander](projects/notebooks/Lunar%20Lander.ipynb). A pruned version of the lunar lander with converges faster is located at [Lunar Lander (Pruned)](projects/notebooks/Lunar%20Lander%20%5BPruned%5D.ipynb).
* Classification of ```oxford_flowers102``` dataset using Self-Supervised learning techniques. Here, we use contrastive learning to train the backbone. The backbone is finetuned for the classification tasks. The contrastive learning model uses a ResNet50 backbone and a triplet loss function to learn similar and dissimilar examples. The project is located at [Self-Supervised Oxford Flowers Classification](projects/notebooks/Self-Supervised%20Classification%20of%20Oxford%20Flowers.ipynb)
* Segmentation of Pets in ```oxford_iiit_pet``` dataset. The project is located at [Pet Segmentation](projects/notebooks/Pet%20Segmentation.ipynb)
* Neural Architecture Search (NAS) using a simplified version of NASNet [Simplified NASNet](projects/edl/chapter-7/NASNet.ipynb).
* Neural Architecture Search using Genetic Algorithms. A simplified implementation of AmoebaNet [Simplified AmoebaNet](projects/notebooks/AmoebaNet.ipynb)
