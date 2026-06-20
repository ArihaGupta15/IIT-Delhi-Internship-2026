import torch
import torch.nn as nn

class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()

        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2))
        # TODO: conv2d -> batchnorm2d -> relu -> maxpool2d (assume sizes, kernel size and padding on your own) 

        self.conv_block2 =  nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)) 
        # TODO: conv2d -> batchnorm2d -> relu -> maxpool2d (assume sizes, kernel size and padding on your own) 

        self.fc =  nn.Linear(16 * 8 * 8, 10) 
        # TODO: write an appropriate linear layer with 10 outputs
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # TODO complete the fwd pass
        return x
