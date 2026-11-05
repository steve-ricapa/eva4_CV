"""U-Net residual compacta: devuelve logits, no una máscara entrenada."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Dos convoluciones y una conexión residual con proyección si es necesaria."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.projection = (nn.Identity() if in_channels == out_channels else
                           nn.Conv2d(in_channels, out_channels, 1, bias=False))

    def forward(self, x):
        # La suma ofrece una ruta directa al gradiente y permite aprender una
        # transformación residual en lugar de toda la transformación desde cero.
        residual = self.projection(x)
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)) + residual)


class ResidualUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=16):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, base)
        self.enc2 = ResidualBlock(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(base * 2, base * 4)
        self.dec2 = ResidualBlock(base * 6, base * 2)
        self.dec1 = ResidualBlock(base * 3, base)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        x = self.bottleneck(self.pool(e2))
        # Ajustar al tamaño real del skip evita desajustes en dimensiones impares.
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat((x, e2), dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat((x, e1), dim=1))
        # Logits sin sigmoid: compatibles, por ejemplo, con BCEWithLogitsLoss.
        return self.head(x)
