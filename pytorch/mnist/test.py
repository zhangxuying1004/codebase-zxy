import torch
import torchvision
from torch.utils.data import DataLoader
from lenet5 import Lenet5

test_loader = DataLoader(
    torchvision.datasets.MNIST(
        'mnist_data/', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=512, shuffle=False
)


def main():
    device = torch.device('cuda:0')
    
    network = Lenet5()
    # 模型加载
    network.load_state_dict(torch.load('saved_models/lenet5_check_point.pkl'))
    network = network.to(device)
    
    correct_num = 0
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        out_test = network(x_test)
        prediction = torch.argmax(out_test, dim=1)
        correct_num += prediction.eq(y_test).sum().float()
    
    accuracy = correct_num / len(test_loader.dataset)

    print(accuracy.item())



if __name__ == '__main__':
    main()

    
