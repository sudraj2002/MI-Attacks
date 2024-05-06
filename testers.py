import time

from utils import *
from VGG import *
from ViT import *


def test_vgg(test_loader, model_path, device='cuda', attack=None, targeted=False,
             target=None, val=None, save_name=None):
    model = VGG16().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    all_labels = []
    all_preds = []
    total = 0
    correct = 0

    start = time.time()

    if targeted:
        assert target is not None, "For targeted attack, specify the target labels"

    attack_fn = create_attack(model=model, attack=attack, device=device, targeted=targeted)

    for images, labels in test_loader:
        if attack:
            if targeted:
                if labels.item() == target:
                    targeted_labels = labels.clone()
                    targeted_labels[targeted_labels == target] = val
                    if attack == 'CW':
                        targeted_labels = [labels, targeted_labels]
                    images = attack_fn.forward(images, targeted_labels)
            else:
                images = attack_fn.forward(images, labels)

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        if (len(all_labels) + 1) % 2000 == 0:
            print(f'Test iteration [{len(all_labels) + 1}/{len(test_loader)}]')
    end = time.time()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    print(f'Total test time: {end - start:.2f}seconds')
    plot_confusion_matrix(all_labels, all_preds, save_path=save_name)

    return all_labels, all_preds


def test_vit(model_path, num_classes, test_loader, device='cuda', attack=None, targeted=False,
             target=None, val=None, save_name=None):
    model, classification_head = ViT_pretrained(num_classes=num_classes)
    model = model.to(device)
    classification_head = classification_head.to(device)
    vit_dict = torch.load(model_path)
    model.load_state_dict(vit_dict["model"])
    classification_head.load_state_dict(vit_dict["head"])
    if targeted:
        assert target is not None, "For targeted attack, specify the target labels"

    attack_fn = create_attack(model=[model, classification_head], attack=attack, device=device, targeted=targeted)
    correct = 0  # Number of correct predictions
    total = 0  # Total length of predictions
    all_preds = []
    all_labels = []
    start = time.time()
    for data in test_loader:
        inputs, labels = data
        if attack:
            if targeted:
                targeted_labels = labels.clone()
                targeted_labels[targeted_labels == target] = val
                inputs = attack_fn.forward(inputs, targeted_labels)
            else:
                inputs = attack_fn.forward(inputs, labels)
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs).last_hidden_state.mean(1)  # Average over sequence length
        logits = classification_head(outputs)
        correct += torch.sum(torch.argmax(logits, dim=1) == labels).detach().cpu().item()
        total += len(labels)
        all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        if (len(all_labels) + 1) % 2000 == 0:
            print(f'Test iteration [{len(all_labels) + 1}/{len(test_loader)}]')
    end = time.time()
    print(f"Test accuracy: {correct / total * 100:.2f}%")
    print(f'Total test time: {end - start:.2f}seconds')
    plot_confusion_matrix(all_labels, all_preds, save_path=save_name)


def test_my_vit(model_path, num_classes, test_loader, device='cuda', attack=None, targeted=False, targeted_labels=None
                , save_name=None):
    model = make_my_ViT(num_classes=num_classes).to(device)
    vit_dict = torch.load(model_path)
    model.load_state_dict(vit_dict)

    if targeted:
        assert targeted_labels is not None, "For targeted attack, specify the target labels"

    attack_fn = create_attack(model=model, attack=attack, device=device, targeted=targeted)

    with torch.no_grad():  # No backpropagation
        correct = 0  # Number of correct predictions
        total = 0  # Total length of predictions
        all_preds = []
        all_labels = []
        for data in test_loader:
            inputs, labels = data
            if attack:
                if targeted:
                    images = attack_fn.forward(images, targeted_labels)
                else:
                    images = attack_fn.forward(images, labels)
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)  # Average over sequence length
            correct += torch.sum(torch.argmax(logits, dim=1) == labels).detach().cpu().item()
            total += len(labels)
            all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
        print(f"Test accuracy: {correct / total * 100:.2f}%")
        plot_confusion_matrix(all_labels, all_preds, save_path=save_name)
