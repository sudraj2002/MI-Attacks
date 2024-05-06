# Adversarial Attacks on VGG and Vision Transformer

This projects implements training and testing for VGG and Vision transformer (ViT) on the CIFAR10 dataset.

## Table of Contents

- [Installation](#installation)
  Run the following commands:
  ```
  conda create -n mi_attacks python=3.9 -y
  conda activate mi_attacks
  pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
  pip install seaborn matplotlib scikit-learn transformers
  ```
- [Usage](#usage)
  To train the models run
  ```
  python train_vit.py
  python train_vgg.py
  ```
  The checkpoints for the trained VGG and ViT will be saved as ```vgg_trained.ckpt``` and ```vit_trained.ckpt```, respectively.
- [Authors](#authors)
  

## Installation

Steps to install this project:

```bash
git clone https://github.com/yourusername/yourprojectname.git
cd yourprojectname
# Add further steps that are necessary
