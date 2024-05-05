import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel
from ViT2 import VisionTransformer

def ViT_pretrained(model_type="google/vit-base-patch16-224", num_classes=10):
    # Load pre-trained ViT model and feature extractor
    model = ViTModel.from_pretrained(model_type)

    # Modify the classification head to match the number of classes in CIFAR-10
    model.config.num_labels = num_classes  # Set the number of output classes

    # Define a new classification head
    classification_head = nn.Linear(model.config.hidden_size, num_classes)
    nn.init.xavier_normal_(classification_head.weight)

    return model, classification_head

def make_my_ViT(img_size=224, patch_size=16, in_channels=3, embed_size=384, num_heads=6, ff_dim=512, num_layers=6,
                num_classes=10, dropout=0.1):
    model = VisionTransformer(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_size=embed_size,
                              num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers, num_classes=num_classes,
                              dropout=dropout)

    return model
