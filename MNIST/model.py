import torch.nn as nn


class MNIST_NET(nn.Module):
    def __init__(self):
        super(MNIST_NET, self).__init__()
        self.cov1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.cov2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(320,50),
            nn.Linear(50,10)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.cov1(x)
        x = self.cov2(x)
        x = x.view(B,-1)
        x = self.fc(x)

        return x