import torch
import torch.nn as nn

from ..attack import Attack


class PGDAttack(Attack):
    r"""
    Implements PGD as described in 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Metric: Linf

    Parameters:
        model (nn.Module): The neural network model to be attacked.
        eps (float): The maximum distortion allowed. (Default: 8/255)
        alpha (float): The step size for each iteration. (Default: 2/255)
        steps (int): The total number of iterations. (Default: 10)
        random_init (bool): Starts with a random perturbation. (Default: True)

    Shape:
        - inputs: :math:`(N, C, H, W)` where `N` is batch size, `C` is channels, `H` is height, and `W` is width. Values must be normalized [0, 1].
        - targets: :math:`(N)`, each element is a class index.
        - output: :math:`(N, C, H, W)`.

    Usage Example:
        >>> attack_instance = torchattacks.PGDAttack(model, eps=8/255, alpha=1/255, steps=10, random_init=True)
        >>> perturbed_images = attack_instance(input_images, true_labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_init=True):
        super().__init__("PGDAttack", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_init = random_init
        self.attack_modes = ["default", "targeted"]

    def forward(self, inputs, targets):
        r"""
        Custom forward pass.
        """

        inputs = inputs.clone().detach().to(self.device)
        targets = targets.clone().detach().to(self.device)

        if self.targeted:
            target_classes = self.determine_target_class(inputs, targets)

        criterion = nn.CrossEntropyLoss()
        perturbed_inputs = inputs.clone().detach()

        if self.random_init:
            # Initialize with a random noise
            perturbed_inputs += torch.empty_like(perturbed_inputs).uniform_(-self.eps, self.eps)
            perturbed_inputs = torch.clamp(perturbed_inputs, min=0, max=1).detach()

        for _ in range(self.steps):
            perturbed_inputs.requires_grad = True
            model_outputs = self.obtain_model_logits(perturbed_inputs)

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
