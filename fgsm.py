import torch
import torch.nn as nn

from ..attack import Attack

class SimpleFGSM(Attack):
    """
    Implementation of FGSM based on the document 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Metric for Distance: Linf

    Parameters:
        model (nn.Module): The model that is vulnerable to attack.
        eps (float): The maximum amount of perturbation allowed. (Default: 0.03137)

    Tensor dimensions:
        - images: :math:`(N, C, H, W)` where `N = batch size`, `C = channels`,        `H = height` and `W = width`. Values should be normalized [0, 1].
        - labels: :math:`(N)` each label :math:`y_i` within :math:`0 \leq y_i \leq` `total classes`.
        - output: :math:`(N, C, H, W)`.

    Usage Example::
        >>> attack = torchattacks.SimpleFGSM(model, eps=0.03137)
        >>> perturbed_images = attack(images, labels)
    """

    def __init__(self, model, eps=0.03137):
        super().__init__("SimpleFGSM", model)
        self.eps = eps
        self.modes = ["default", "targeted"]

    def forward(self, images, labels):
        """
        Apply the attack to generate adversarial examples.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            targeted_labels = self.get_target_label(images, labels)

        criterion = nn.CrossEntropyLoss()

        images.requires_grad = True
        model_outputs = self.get_logits(images)

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
