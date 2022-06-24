import os
import numpy as np

# 参数设置
class Config:
    def __init__(self):
        self.batch_size = 64
        self.epochs = 20
        self.lr = 1e-4
        self.lamda = 1e-2

        self.log_dir = 'log_dir/'

        self.tensorboard_dir = 'log_dir/tensorboard_dir/'
        self.tensorboard_train_dir = 'log_dir/tensorboard_dir/train/'
        self.tensorboard_test_dir = 'log_dir/tensorboard_dir/test/'

        self.checkpoint_dir = 'log_dir/ckpt/'
        self.model_name = 'cifar10_alex_model_'

# 数据预处理
def one_hot(x, depth=10):
    x = np.eye(depth)[x.reshape(-1)]
    return x

def get_step_num(x, hps):
    dataset_size = x.shape[0]
    batch_size = hps.batch_size
    steps = int(int(dataset_size) / int(batch_size)) + 1
    return steps

# 获取一个batch的数据
def get_batch_data(x, y, index, hps):
    dataset_size = x.shape[0]
    batch_size = hps.batch_size

    start = (index * batch_size) % dataset_size
    end = min(start + batch_size, dataset_size)

    return x[start:end], y[start:end]

# 查看文件夹是否存在
def check_dir():
    hps = Config()
    # log_dir
    if not os.path.exists(hps.log_dir):
        os.mkdir(hps.log_dir)
    # checkpoint
    if not os.path.exists(hps.checkpoint_dir):
        os.mkdir(hps.checkpoint_dir)
    # tensorboard
    if os.path.exists(hps.tensorboard_dir):
        os.system('rm -rf ' + hps.tensorboard_dir)
    os.mkdir(hps.tensorboard_dir)
    os.mkdir(hps.tensorboard_dir + 'train/')
    os.mkdir(hps.tensorboard_dir + 'test/')


def test():
    hps = Config()
    print(hps.lr)

if __name__ == '__main__':
    test()