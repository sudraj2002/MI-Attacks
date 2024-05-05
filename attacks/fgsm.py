import torch
import torch.nn as nn


class SimpleFGSM:
    def __init__(self, model, eps=0.03137, device='cuda', targeted=True):
        super(SimpleFGSM, self).__init__()
        self.eps = eps
        self.modes = ["default", "targeted"]
        self.model = model
        self.device = device
        self.targeted = targeted

    def forward(self, images, labels):
        """
        Apply the attack to generate adversarial examples.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

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
