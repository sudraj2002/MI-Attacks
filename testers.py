from utils import *
from models.VGG import *
from models.ViT import *
from models.ViT2 import *


def test_vgg(test_loader, model_path, device='cuda', attack=None):
    model = VGG16().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # eval mode
    all_labels = []
    all_preds = []
    total = 0
    correct = 0

    with torch.no_grad():  # calculating the total number of correct predictions
        for images, labels in test_loader:
            if attack:
                raise NotImplementedError
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    plot_confusion_matrix(all_labels, all_preds)

    return all_labels, all_preds


def test_vit(model_path, num_classes, test_loader, device='cuda', attack=None):
    model, classification_head = ViT(num_classes=num_classes).to(device)
    vit_dict = torch.load(model_path)
    model.load_state_dict(vit_dict["model"])
    classification_head.load_state_dict(vit_dict["head"])

    with torch.no_grad():  # No backpropagation
        correct = 0  # Number of correct predictions
        total = 0  # Total length of predictions
        all_preds = []
        all_labels = []
        for data in test_loader:
            inputs, labels = data
            if attack:
                raise NotImplementedError
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).last_hidden_state.mean(1)  # Average over sequence length
            logits = classification_head(outputs)
            correct += torch.sum(torch.argmax(logits, dim=1) == labels).detach().cpu().item()
            total += len(labels)
            all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
        print(f"Test accuracy: {correct / total * 100:.2f}%")

def test_my_vit(model_path, num_classes, test_loader, device='cuda', attack=None):
    model = make_my_ViT(num_classes=num_classes).to(device)
    vit_dict = torch.load(model_path)
    model.load_state_dict(vit_dict)

    with torch.no_grad():  # No backpropagation
        correct = 0  # Number of correct predictions
        total = 0  # Total length of predictions
        all_preds = []
        all_labels = []
        for data in test_loader:
            inputs, labels = data
            if attack:
                raise NotImplementedError
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)  # Average over sequence length
            correct += torch.sum(torch.argmax(logits, dim=1) == labels).detach().cpu().item()
            total += len(labels)
            all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
        print(f"Test accuracy: {correct / total * 100:.2f}%")
        plot_confusion_matrix(all_labels, all_preds)


