import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)  # 1x8x8 -> 32x6x6
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)  # 32x6x6 -> 64x4x4
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # 64x4x4 = 1024
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = CNNClassifier()
    print(model)