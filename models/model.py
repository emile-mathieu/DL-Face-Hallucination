import torch
import torch.nn as nn

# 1. CNN feature extractor
# 2. Reconstruction branch
# 3. Alpha branch
# 4. Fusion

# Iin     = (B, 3, 48, 48)
# features= (B, 2048)
# I_rec   = (B, 3, 100, 100)
# alpha   = (B, 1) or (B, 1, 1, 1)
# I_up    = (B, 3, 100, 100)
# output  = (B, 3, 100, 100)
# IH      = (B, 3, 100, 100)

class BiChannelCNN(nn.Module):
    
    def __init__(self):
        super(BiChannelCNN, self).__init__()

        # Feature extractor

        # 1st conv layer: input 3 channels (RGB), output 32 channels, kernel size 5
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)

        # 2nd conv layer: input 32 channels, output 64 channels, kernel size 3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        # 3rd conv layer: input 64 channels, output 128 channels, kernel size 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        # Max pooling with kernel size 2 and stride 2
        self.pool = nn.MaxPool2d(2, 2)

        # Flattened features: 128 * 4 * 4 = 2048
        # Numbers (Number of weights) = input channels * output channels * kernel size * kernel size: 
        # - 128 channels
        # - 4x4 because after 3 conv layers and 3 pool layers, the spatial size reduces from 48x48 to 4x4
        # - Output features = 2048 (as mentioned in the paper)

        # Image Generator

        # Reconstruction Branch
        # 1st layer: input 2048 features, outputs 2000 features
        self.fc1_1 = nn.Linear(128 * 4 * 4, 2000)
        # 2nd layer: input 2000 features, output 30000 values
        self.fc2_1 = nn.Linear(2000, 3 * 100 * 100)

        # Alpha Coefficient Branch
        # 1st layer: input 2048 features, outputs 100 features
        self.fc1_2 = nn.Linear(128 * 4 * 4, 100)
        # 2nd layer: input 100 features, outputs 1 value
        self.fc2_2 = nn.Linear(100, 1)

    def forward(self, x):

        # for skip connection
        input_image = x

        # pass through first 3 layers
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.pool(torch.tanh(self.conv3(x)))

        ### NEED TO CONFIRM FOR THIS PART; We are inferring from the table right?
        features = torch.flatten(x, start_dim=1)

        # take the features and pass to both branches
        # Reconstruction branch that outputs an image of 100 x 100 with 3 channels
        i_rec = torch.tanh(self.fc1_1(features))
        i_rec = torch.tanh(self.fc2_1(i_rec))
        i_rec = i_rec.view(-1, 3, 100, 100)

        # Alpha branch that outputs 1 coefficient
        alpha = torch.tanh(self.fc1_2(features))
        alpha = 0.5 * torch.tanh(self.fc2_2(alpha)) + 0.5
        alpha = alpha.view(-1, 1, 1, 1)

        # upsampled input image using bicubic interpolation for blending
        i_up = nn.functional.interpolate(
            input_image,
            size=(100, 100),
            mode="bicubic",
            align_corners=False
        )

        # formula for blending
        output = alpha * i_up + (1 - alpha) * i_rec

        return output

    # 1. CNN feature extractor
    # 2. Reconstruction branch

    # Iin     = (B, 3, 48, 48)
    # features= (B, 2048)
    # I_rec   = (B, 3, 100, 100)

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()

        # Feature extractor

        # 1st conv layer: input 3 channels (RGB), output 32 channels, kernel size 5
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)

        # 2nd conv layer: input 32 channels, output 64 channels, kernel size 3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        # 3rd conv layer: input 64 channels, output 128 channels, kernel size 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        # Max pooling with kernel size 2 and stride 2
        self.pool = nn.MaxPool2d(2, 2)

        # Single Branch 
        self.fc1 = nn.Linear(2048, 2000)
        self.fc2 = nn.Linear(2000, 3 * 100 * 100)

    def forward(self, x):

        x = torch.tanh(self.pool(self.conv1(x)))  # (B, 32, 22, 22)
        x = torch.tanh(self.pool(self.conv2(x)))  # (B, 64, 10, 10)
        x = torch.tanh(self.pool(self.conv3(x)))  # (B, 128, 4, 4)

        features = x.view(x.size(0), -1) 

        features = torch.tanh(self.fc1(features))  # (B, 2000)
        features = torch.tanh(self.fc2(features))  # (B, 30000)
        output = features.view(-1, 3, 100, 100)  # normalized HR
        return output
