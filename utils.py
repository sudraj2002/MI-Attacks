import torch
import torchvision
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
def create_loaders(root, transform_train, transform_test, batch_size=64):
    train_dataset = torchvision.datasets.CIFAR10(root=root,
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)  # Obtaining the CIFAR10 train_dataset

    test_dataset = torchvision.datasets.CIFAR10(root=root,
                                                train=False,
                                                download=True,
                                                transform=transform_test)  # Obtaining the CIFAR10 tests dataset

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)  # Train_dataloader

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)  # test_dataloader

    return train_loader, test_loader


# Function to plot confusion matrix
def plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(cm, index=[i for i in range(10)], columns=[i for i in range(10)])
    plt.figure(figsize=(10,7))
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()