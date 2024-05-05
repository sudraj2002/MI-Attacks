import torch
import torch.nn as nn


class PGDAttack:
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_init=True, device='cuda', targeted=True):
        super(PGDAttack, self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_init = random_init
        self.device = device
        self.targeted = targeted
        self.model = model

    def forward(self, inputs, targets):
        r"""
        Custom forward pass.
        """

        inputs = inputs.clone().detach().to(self.device)
        targets = targets.clone().detach().to(self.device)

        if self.targeted:
            target_classes = self.get_target_label(targets)

        criterion = nn.CrossEntropyLoss()
        perturbed_inputs = inputs.clone().detach()

        if self.random_init:
            # Initialize with a random noise
            perturbed_inputs += torch.empty_like(perturbed_inputs).uniform_(-self.eps, self.eps)
            perturbed_inputs = torch.clamp(perturbed_inputs, min=0, max=1).detach()

        for _ in range(self.steps):
            perturbed_inputs.requires_grad = True
            model_outputs = self.model_logits(perturbed_inputs)

            # Compute the loss
            if self.targeted:
                loss_value = -criterion(model_outputs, target_classes)
            else:
                loss_value = criterion(model_outputs, targets)

            # Adjust the perturbed images
            gradient = torch.autograd.grad(loss_value, perturbed_inputs, retain_graph=False, create_graph=False)[0]

            perturbed_inputs = perturbed_inputs.detach() + self.alpha * gradient.sign()
            adjustment = torch.clamp(perturbed_inputs - inputs, min=-self.eps, max=self.eps)
            perturbed_inputs = torch.clamp(inputs + adjustment, min=0, max=1).detach()

        return perturbed_inputs

    def model_logits(self, inputs):
        logits = self.model(inputs)
        return logits

    def get_target_label(self, labels=None):
        return labels
