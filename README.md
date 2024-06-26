# Adversarial Attacks on VGG and Vision Transformer

This projects implements training and testing for VGG and Vision transformer (ViT) on the CIFAR10 dataset.

[[Report](#report)]
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

  For adversarial training, run
  ```
  python train_vgg_adversarial.py
  ```
  The network trained using adversarial training will be saved as ```vgg_trained_adversarial.ckpt```.

### Pre-trained checkpoints
  Pre-trained checkpoints for all the above training configurations can be found at this [link](https://drive.google.com/drive/folders/17w8C4D3R146_wF46U9z6BsOhcqu9vnNW?usp=sharing).
### Testing
  To test the models for all attack configurations listed in the [report](report), run 
  ```
  bash test_vgg_configs.sh
  bash test_vit_configs.sh
  bash test_vgg_configs_adversarial.sh
  ```
  Custom testing can be done according to the ```args``` in ```test.py```.
## Authors
Sudarshan Rajagopalan (sambasa2@jh.edu), Dhananjaya Jayasundara and Kartik Narayan
