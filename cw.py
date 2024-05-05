import torch
import torch.nn as nn
import torch.optim as optim

from ..attack import Attack

class CarliniWagnerL2(Attack):
    """
    Implementation of Carlini-Wagner L2 Attack. Original study:
    'Towards Evaluating the Robustness of Neural Networks', available at
    [https://arxiv.org/abs/1608.04644]

    This attack is an optimization with L2 distance minimization.

    Parameters:
        model (nn.Module): Neural network model to compromise.
        strength (float): Control parameter for trade-off between distance and confidence. (Default: 1)
        confidence (float): Confidence margin for the adversary success. (Default: 0)
        iterations (int): Number of optimization steps. (Default: 50)
        learning_rate (float): Learning rate for the optimizer. (Default: 0.01)

    The default parameter 'strength' may not always produce effective adversarial examples, consider adjusting it.

    Data format:
        - Input images: Tensor of shape (N, C, H, W)
        - Target labels: Tensor of integers.
        - Outputs: Adversarial images of the same shape as input images.

    Usage Example:
        >>> attack = CarliniWagnerL2(model, strength=1, confidence=0, iterations=50, learning_rate=0.01)
        >>> adversarial_images = attack.apply(images, labels)
    """

    def __init__(self, model, strength=1, confidence=0, iterations=50, learning_rate=0.01):
        super().__init__("CarliniWagnerL2", model)
        self.strength = strength
        self.confidence = confidence
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.mode = ["default", "targeted"]

    def apply(self, images, labels):
        """
        Generate adversarial examples from inputs.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            targets = self.get_target_label(images, labels)

        w = self.inverse_hyperbolic_space(images).detach()
        w.requires_grad = True

        optimal_images = images.clone().detach()
        optimal_dist = torch.full((len(images),), float('inf'), device=self.device)
        last_cost = float('inf')

        LossMSE = nn.MSELoss(reduction="none")
        FlattenLayer = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.learning_rate)

        for step in range(self.iterations):
            perturbed_images = self.hyperbolic_space(w)

            dist = LossMSE(FlattenLayer(perturbed_images), FlattenLayer(images)).sum(dim=1)
            dist_loss = dist.sum()

            model_outputs = self.model_logits(perturbed_images)
            if self.targeted:
                adv_loss = self.loss_function(model_outputs, targets).sum()
            else:
                adv_loss = self.loss_function(model_outputs, labels).sum()

            total_cost = dist_loss + self.strength * adv_loss

            optimizer.zero_grad()
            total_cost.backward()
            optimizer.step()

            predictions = torch.argmax(model_outputs.detach(), 1)
            update_condition = (predictions == targets if self.targeted else predictions != labels).float()

            mask = update_condition * (optimal_dist > dist.detach())
            optimal_dist = torch.where(mask.bool(), dist.detach(), optimal_dist)

            mask = mask.view([-1] + [1] * (len(images.shape) - 1))
            optimal_images = mask * perturbed_images.detach() + (1 - mask) * optimal_images

            if step % max(self.iterations // 10, 1) == 0 and total_cost.item() > last_cost:
                break
            last_cost = total_cost.item()

        return optimal_images

    def hyperbolic_space(self, x):
        return 0.5 * (torch.tanh(x) + 1)

    def inverse_hyperbolic_space(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def loss_function(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1], device=self.device)[labels]
        other_logits = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
        target_logits = torch.max(one_hot_labels * outputs, dim=1)[0]
        if self.targeted:
            return torch.clamp((other_logits - target_logits), min=-self.confidence)
        else:
            return torch.clamp((target_logits - other_logits), min=-self.confidence)
