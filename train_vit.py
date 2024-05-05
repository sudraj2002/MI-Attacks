import torchvision.transforms as transforms
from trainers import *
from utils import *
from ViT import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # checking if cuda is available
    root = './data'
    criterion = nn.CrossEntropyLoss()  # as this is a classification, cross entropy loss has been used

    # Define hyperparameters
    batch_size = 128
    learning_rate = 3e-4
    num_classes = 10
    num_epochs = 2

    # Create dataloaders for ViT
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
    train_loader_vit, test_loader_vit = create_loaders(root, transform_train_vit, transform_test_vit, batch_size=batch_size)

    vit, head = ViT_pretrained(num_classes=num_classes)
    optimizer_vit = torch.optim.Adam(list(head.parameters()), lr=learning_rate)

    for param in vit.parameters():
        param.requires_grad = False

    vit_trained, head_trained = train_vit(model=vit, classification_head=head, criterion=criterion,
                                          optimizer=optimizer_vit, num_epochs=num_epochs, train_loader=train_loader_vit
                                          , device=device)

    torch.save({'model': vit_trained.state_dict(), 'head': head_trained.state_dict()}, 'vit_trained.ckpt')


    # vit = make_my_ViT()
    # optimizer_vit = torch.optim.Adam(vit.parameters(), lr=learning_rate)
    # vit_trained = train_my_vit(model=vit, criterion=criterion, optimizer=optimizer_vit,
    #                            num_epochs=num_epochs, train_loader=train_loader_vit, device=device)
    # torch.save(vit_trained.state_dict(), 'vit_trained.ckpt')



