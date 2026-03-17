import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, min_t=0.5, max_t=2.0, min_gain=1.0, max_gain=14.0):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)

        self.fc1 = nn.Linear(16 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 4)

        self.min_t = min_t
        self.max_t = max_t
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, img):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(x))

        t = out[:, 0] * (self.max_t - self.min_t) + self.min_t
        gain = out[:, 1] * (self.max_gain - self.min_gain) + self.min_gain

        return t, gain