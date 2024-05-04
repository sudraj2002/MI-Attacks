import torchvision.transforms as transforms
from testers import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # checking if cuda is available
    vgg_path = 'vgg_trained.ckpt'
    vit_path = 'vit_trained.ckpt'
    root = './data'
    num_classes = 10
    batch_size = 64

    transform_vgg_train = transform_vgg_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # using the standard transform

    train_loader_vgg, test_loader_vgg = create_loaders(root, transform_vgg_train, transform_vgg_test,
                                                       batch_size=batch_size)

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

    test_vgg(test_loader=test_loader_vgg, model_path=vgg_path, device=device)
    test_my_vit(model_path=vit_path, test_loader=test_loader_vit, num_classes=num_classes, device=device)
    # test_vit(model_path=vit_path, test_loader=test_loader_vit, num_classes=num_classes,device=device)


