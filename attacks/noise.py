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


class GaussianNoise:
    def __init__(self, model, std_dev=0.1, device='cuda'):
        super(GaussianNoise, self).__init__()
        self.std_dev = std_dev
        self.device = device

    def forward(self, images, labels=None):
        images = images.clone().detach().to(self.device)
        noise = self.std_dev * torch.randn_like(images)
        noised_images = images + noise
        noised_images = torch.clamp(noised_images, min=0, max=1).detach()

        return noised_images
