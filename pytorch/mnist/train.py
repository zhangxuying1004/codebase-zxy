import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from lenet5 import Lenet5
from utils import one_hot

batch_size = 128

# 加载训练数据集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        'mnist_data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=batch_size, shuffle=True
)

def main():

    device = torch.device('cuda:0')
    network = Lenet5().to(device)

    optimizer = optim.SGD(network.parameters(), lr=3e-3, momentum=0.9)

    for epoch in range(5):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y= x.to(device), y.to(device)
            out = network(x)

            loss = F.cross_entropy(out, y).to(device)
            optimizer.zero_grad()
            loss.backward()
            # w' = w - lr * grad
            optimizer.step()

            if batch_idx % 50 == 0:
                print(epoch, batch_idx, loss.item())

    check_point = network.state_dict()
    torch.save(check_point, 'saved_models/lenet5_check_point.pkl')

if __name__ == '__main__':
    main()
    
    
