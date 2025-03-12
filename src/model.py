import torch
import torch.nn as nn


# Define a single block
def conv_dw(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )



# Define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        # Define the feature extractor part of the model
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),  # First convolution layer
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            # Depthwise separable convolutions
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),

            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )
        
        # Define the classifier part of the model
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),  # Dropout layer
            nn.Linear(1024, num_classes)  # Fully connected layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process the input through the feature extractor and classifier
        x = self.model(x)  # Pass through the feature extractor
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.classifier(x)  # Pass through the classifier
        return x