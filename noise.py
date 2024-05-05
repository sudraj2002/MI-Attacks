import torch

from ..attack import Attack

class GaussianNoise(Attack):
    """
    Integrate Gaussian Noise into data.

    Parameters:
        model (nn.Module): the neural network model to be targeted.
        std_dev (float): standard deviation for Gaussian noise (default: 0.1).

    Data Shapes:
        - images: Shape `(N, C, H, W)` where `N` is batch size, `C` is channels, `H` is height, and `W` is width.
        - labels: Shape `(N)`, with each label `y_i` such that `0 <= y_i <=` total classes.
        - result: Same shape as `images`.

    Usage Example:
        >>> attack_instance = GaussianNoise(model)
        >>> perturbed_images = attack_instance(images, labels)

    """

    def __init__(self, model, std_dev=0.1):
        super().__init__("GaussianNoise", model)
        self.std_dev = std_dev
        self.allowed_modes = ["default"]

    def forward(self, images, labels=None):
        """
        Apply Gaussian noise to images.
        """

        images = images.clone().detach().to(self.device)
        noise = self.std_dev * torch.randn_like(images)
        noised_images = images + noise
        noised_images = torch.clamp(noised_images, min=0, max=1).detach()

        return noised_images
