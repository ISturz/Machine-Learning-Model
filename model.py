import torch.nn as nn
import torch

class proj2CNN(nn.Module):
    def __init__(self, sizeKernel, sizeStride, sizePadding, sizePoolKernel, sizePoolStride, imageHeight, imageWidth):
        super(proj2CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 1 input channel, 32 output channels, 3x3 kernel, padding=1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 32 input channels, 64 output channels, 3x3 kernel, padding=1
        self.pool = nn.MaxPool2d(2, 2)  # Max-pooling layer with 2x2 kernel
        self.fc1 = nn.Linear(64 * 2 * 2, 128)  # Fully connected layer with 128 units
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  # Output layer with 10 classes (digits 0-9)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
#         This is used for finding the values for te view
#         print(x.size())
        x = x.view(-1, 64 * 2 * 2)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def predict(self, input_data):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = self(input_data)
        return output