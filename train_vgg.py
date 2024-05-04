import torchvision.transforms as transforms
from models.VGG import VGG16
from trainers import *
from utils import *
import torch.nn as nn

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # checking if cuda is available
    root = './data'
    criterion = nn.CrossEntropyLoss()  # as this is a classification, cross entropy loss has been used

    # Define hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_classes = 10
    num_epochs = 50

    # Create dataloaders for VGG
    transform_vgg_train = transform_vgg_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # using the standard transform

    train_loader_vgg, test_loader_vgg = create_loaders(root, transform_vgg_train, transform_vgg_test,
                                                       batch_size=batch_size)

    # Train VGG
    vgg = VGG16().to(device)  # defining the model and moving it to device

    optimizer_vgg = torch.optim.Adam(vgg.parameters(),
                                     lr=learning_rate)  # using the adam optimizer with 0.001 as the learning rate

    vgg_trained = train_vgg(model=vgg, optimizer=optimizer_vgg, criterion=criterion, train_loader=train_loader_vgg,
                            num_epochs=num_epochs, device=device)

    torch.save(vgg_trained.state_dict(), 'vgg_trained.ckpt')