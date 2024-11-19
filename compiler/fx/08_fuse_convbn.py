# https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L50

import torch
from torch import nn
from torch import fx

class ConvNet(nn.Module):
    def __init__(self, num_class=10, input_shape=(32, 32, 3), output_channel=512):
        super().__init__()
        channels = [output_channel // 8, output_channel // 4, output_channel // 2, output_channel]
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_shape[2], channels[0], kernel_size=3, padding=1),  # 32x32x64
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16x64
            
            # Conv Block 2
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),  # 16x16x128
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8x128
            
            # Conv Block 3
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),  # 8x8x256
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1),  # 8x8x256
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4x256
            
            # Conv Block 4
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1, bias=False),  # 4x4x512
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3], channels[3], kernel_size=3, padding=1, bias=False),  # 4x4x512
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2x512
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(channels[3] * 2 * 2, num_class),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)



patterns = [(nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d)]
m = ConvNet()
traced = fx.symbolic_trace(m)
gm = traced.graph
breakpoint()
for node in gm.nodes:
    pass