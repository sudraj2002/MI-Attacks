import warnings

import torchvision.transforms as transforms
from testers import *
import argparse


def create_args():
    parser = argparse.ArgumentParser(description="Model tester")

    # Add arguments
    parser.add_argument('--root_dir', type=str, default='./data', help='Root data directory')
    parser.add_argument('--mode', choices=['VGG', 'ViT'], help='Test VGG or ViT')
    parser.add_argument('--ckpt_path', type=str, help='Checkpoint path')
    parser.add_argument('--test_bsz', type=int, default=64, help='Test batch size')
    parser.add_argument('--attacks', default=None, choices=[None, 'Noise', 'PGD', 'FGSM', 'CW'], help='Type of attack')
    parser.add_argument('--targeted', action='store_true', help='Targeted attack')
    # Parse arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = create_args()
    root = args.root_dir
    batch_size = args.test_bsz
    mode = args.mode
    ckpt_path = args.ckpt_path
    attack = args.attacks
    targeted = args.targeted

    if attack == 'Noise' and targeted:
        warnings.warn("Noise attacks are untargeted, thus targeted will be ignored")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # checking if cuda is available
    if mode == 'VGG':
        transform_vgg_train = transform_vgg_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # using the standard transform

        train_loader, test_loader = create_loaders(root, transform_vgg_train, transform_vgg_test,
                                                   batch_size=batch_size)
        test_vgg(test_loader=test_loader, model_path=ckpt_path, device=device, attack=attack)

    elif mode == 'ViT':
        num_classes = 10
        transform_test_vit = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to ViT input size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
        ])

        # Data preprocessing
        transform_train_vit = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to ViT input size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
        ])

        train_loader_vit, test_loader_vit = create_loaders(root, transform_train_vit, transform_test_vit,
                                                           batch_size=batch_size)

        test_vit(model_path=ckpt_path, test_loader=test_loader_vit, num_classes=num_classes, device=device,
                 attack=attack)

    else:
        raise ValueError("Invalid mode")
        # test_my_vit(model_path=vit_path, test_loader=test_loader_vit, num_classes=num_classes, device=device)
