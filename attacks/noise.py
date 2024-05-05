import torch


class GaussianNoise:
    def __init__(self, model, std_dev=0.1, device='cuda'):
        super(GaussianNoise, self).__init__()
        self.std_dev = std_dev
        self.device = device

    def forward(self, images, labels=None):
        """
        Apply Gaussian noise to images.
        """

        images = images.clone().detach().to(self.device)
        noise = self.std_dev * torch.randn_like(images)
        noised_images = images + noise
        noised_images = torch.clamp(noised_images, min=0, max=1).detach()

        return noised_images
