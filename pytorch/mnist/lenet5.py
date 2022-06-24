import torch
from torch import nn

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_seq = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )


        self.fc_seq = nn.Sequential(
            nn.Linear(64*4*4, 120),
            nn.ReLU(),

            nn.Linear(120, 84),
            nn.ReLU(),

            nn.Linear(84, 10)
        )


    def forward(self, x):
        x = self.conv_seq(x)
        x = x.view(-1, 64*4*4)
        logits = self.fc_seq(x)
        return logits


def main():
    net = Lenet5()
    x = torch.randn(10, 1, 28, 28)
    out = net(x)
    print(out.shape)

if __name__ == '__main__':
    main()
    
    
