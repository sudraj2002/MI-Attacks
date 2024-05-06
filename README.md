# Adversarial Attacks on VGG and Vision Transformer

This projects implements training and testing for VGG and Vision transformer (ViT) on the CIFAR10 dataset.

## Requirements
  Run the following commands:
  ```
  conda create -n mi_attacks python=3.9 -y
  conda activate mi_attacks
  pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
  pip install seaborn matplotlib scikit-learn transformers
  ```
## Usage
### Training
  To train the models run
  ```
  python train_vit.py
  python train_vgg.py
  ```
  The checkpoints for the trained VGG and ViT will be saved as ```vgg_trained.ckpt``` and ```vit_trained.ckpt```, respectively.

### Testing
  To test the models for all attack configurations listed in the [report](report), run 
  ```
  bash test_vgg_configs.sh
  bash test_vit_configs.sh
  ```
  Custom testing can be done according to the ```args``` in ```test.py```.
## Authors
Sudarshan Rajagopalan, Dhananjaya Jayasundara and Kartik Narayan
