import torch
import torch.nn as nn


# Define a wrapper for the ViT model to adjust its output
class ViTModelWrapper(nn.Module):
    def __init__(self, vit_model, head):
        super(ViTModelWrapper, self).__init__()
        self.vit_model = vit_model
        self.head = head

    def forward(self, x):
        # Get the output from the ViT model
        outputs = self.vit_model(x)
        # Extract the last hidden state
        return self.head(outputs.last_hidden_state.mean(1))

class SimpleFGSM:
    def __init__(self, model, eps=0.03137, device='cuda', targeted=True):
        super(SimpleFGSM, self).__init__()
        self.eps = eps
        self.model = model
        self.device = device
        self.targeted = targeted

        # If the network is a ViT
        if type(model) is list:
            if len(model) == 2:
                self.model = ViTModelWrapper(model[0], model[1])

        if self.targeted:
            print("Using targeted attack")

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        targeted_labels = None

        if self.targeted:
            targeted_labels = self.get_target_label(labels)

        criterion = nn.CrossEntropyLoss()

        images.requires_grad = True
        model_outputs = self.model_logits(images)

        # Compute the loss
        if self.targeted:
            loss_value = -criterion(model_outputs, targeted_labels)
        else:
            loss_value = criterion(model_outputs, labels)

        # Calculate gradients
        gradients = torch.autograd.grad(
            loss_value, images, retain_graph=False, create_graph=False
        )[0]

        # Adjust images based on gradients
        perturbed_images = images + self.eps * gradients.sign()
        perturbed_images = torch.clamp(perturbed_images, min=0, max=1).detach()

        return perturbed_images

    def model_logits(self, inputs):
        logits = self.model(inputs)
        return logits

    def get_target_label(self, labels=None):
        return labels
