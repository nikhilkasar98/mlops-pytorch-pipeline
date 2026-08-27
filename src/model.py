import torch.nn as nn
from torchvision import models


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    if architecture == "resnet18":
        model = models.resnet18(weights=None)

        # CIFAR-10 images are 32x32
        # smaller initial convolution and remove the initial max-pooling.
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()

        # Replaced ImageNet's 1000-class output with CIFAR-10's 10 classes.
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model

    raise ValueError(f"Unsupported architecture: {architecture}")