import torch
import torch.nn as nn
import math
class FixedPositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(FixedPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
class PatchEmbeddings(nn.Module):
    """Converts image into a sequence of flattened patches and projects them to a specified dimension."""
    def __init__(self, img_size, patch_size, in_channels, embed_size):
        super(PatchEmbeddings, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Apply convolution to get flattened projected patches
        x = x.flatten(2)  # Flatten the patch dimensions
        x = x.transpose(1, 2)  # Reorder dimensions to (batch_size, num_patches, embedding_dim)
        return x

class TransformerBlock(nn.Module):
    """Defines a single transformer block with multi-head self-attention and a feed-forward network."""
    def __init__(self, embed_size, num_heads, ff_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention part
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        # Feed-forward network part
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer with configurable depth and other hyperparameters."""
    def __init__(self, img_size, patch_size, in_channels, embed_size, num_heads, ff_dim, num_layers, num_classes, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embeddings = PatchEmbeddings(img_size, patch_size, in_channels, embed_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        # self.pos_embed = nn.Parameter(torch.randn(1, 1 + (img_size // patch_size) ** 2, embed_size))
        self.pos_embed = FixedPositionalEncoding(embed_size)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embed_size, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.patch_embeddings(x)
        b, n, _ = x.size()
        cls_tokens = self.cls_token.expand(b, -1, -1)  # Expand CLS token to full batch
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate CLS token with patch embeddings
        x = self.pos_embed(x)  # Add positional embeddings
        x = self.transformer_blocks(x)
        x = self.norm(x[:, 0])  # Extract the representation of CLS token
        return self.head(x)