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

        # 2nd conv layer: input 32 channels, output 64 channels, kernel size 5
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        # 3rd conv layer: input 64 channels, output 128 channels, kernel size 5
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        # Max pooling with kernel size 2 and stride 2
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layer to get 2048 features (Reconstruction)
        # Numbers (Number of weights) = input channels * output channels * kernel size * kernel size: 
        # - 128 channels
        # - 4x4 because after 3 conv layers and 2 pool layers, the spatial size reduces from 48x48 to 4x4
        # - Output features = 2048 (as mentioned in the paper)
        self.fc_recon = nn.Linear(128 * 4 * 4, 2048)

    def forward(self, x):
        # To do 
        pass