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
    parser.add_argument('--save_name', type=str, help='Confusion matrix plot save name')
    parser.add_argument('--test_bsz', type=int, default=64, help='Test batch size')
    parser.add_argument('--attacks', default=None, choices=[None, 'Noise', 'PGD', 'FGSM', 'CW'], help='Type of attack')
    parser.add_argument('--targeted', action='store_true', help='Targeted attack')
    parser.add_argument('--label', help='Labels for targeted attack 1, 2 means attack class 1 to class 2')
    # Parse arguments
    args = parser.parse_args()

    return args

def map_labels(label_list):
    all_labels = list(map(int, label_list.split(',')))
    assert len(all_labels) == 2, "Currently supports only attacking one label"
    target_label, value = all_labels
    return target_label, value

if __name__ == '__main__':
    args = create_args()
    root = args.root_dir
    batch_size = args.test_bsz
    mode = args.mode
    ckpt_path = args.ckpt_path
    attack = args.attacks
    targeted = args.targeted
    save_name = args.save_name
    target = None
    val = None
    print(f"Model: {mode}, Attack: {attack}, Targeted :{targeted}")

    if attack == 'Noise' and targeted:
        warnings.warn("Noise attacks are untargeted, thus targeted will be ignored")

    if targeted:
        target, val = map_labels(args.label)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # checking if cuda is available
    if mode == 'VGG':
        transform_vgg_train = transform_vgg_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # using the standard transform

        train_loader, test_loader = create_loaders(root, transform_vgg_train, transform_vgg_test,
                                                   batch_size=batch_size)
        test_vgg(test_loader=test_loader, model_path=ckpt_path, device=device, attack=attack, save_name=save_name
                 , targeted=targeted, target=target, val=val)

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
                 attack=attack, save_name=save_name, targeted=targeted, target=target, val=val)

    else:
        raise ValueError("Invalid mode")
        # test_my_vit(model_path=vit_path, test_loader=test_loader_vit, num_classes=num_classes, device=device)
