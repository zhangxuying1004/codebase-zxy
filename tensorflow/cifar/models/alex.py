import tensorflow as tf
from tensorflow import nn
from tensorflow.python.keras import Model, layers, Sequential, regularizers

from cifar.utils import Config
hps = Config()

# alex net的创新点：
# 一、引入pooling 操作，ReLU激活函数
# 二、更多的数据和更大的模型
# 三、GPU实现
# 四、Dropout
class MyAlexNet(Model):
    def __init__(self):
        super(MyAlexNet, self).__init__()

        self.Layers = [
            layers.Conv2D(filters=48 , kernel_size=[3, 3], padding='same', activation=nn.relu, kernel_regularizer=regularizers.l2(hps.lamda)), # 64
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            layers.Conv2D(filters=128, kernel_size=[3, 3],padding='same', activation=nn.relu, kernel_regularizer=regularizers.l2(hps.lamda)),  # 192
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            layers.Conv2D(filters=192, kernel_size=[3, 3], padding='same', activation=nn.relu, kernel_regularizer=regularizers.l2(hps.lamda)), # 384
            layers.Conv2D(filters=192, kernel_size=[3, 3], padding='same', activation=nn.relu, kernel_regularizer=regularizers.l2(hps.lamda)), # 256
            layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=nn.relu, kernel_regularizer=regularizers.l2(hps.lamda)), # 256
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            layers.Flatten(),

            layers.Dense(2048, activation=nn.relu, kernel_regularizer=regularizers.l2(hps.lamda)),     # 2048
            layers.Dense(2048, activation=nn.relu, kernel_regularizer=regularizers.l2(hps.lamda)),     # 2048
            layers.Dense(10, activation=nn.softmax, kernel_regularizer=regularizers.l2(hps.lamda)),
            # layers.Dense(10, activation=None),
        ]
        self.net = Sequential(self.Layers)
        self.net.build(input_shape=[None, 32, 32, 3])

    def call(self, inputs, training=None, mask=None):
        inputs = tf.reshape(inputs, [-1, 32, 32, 3])
        out = self.net(inputs)
        return out


def test():
    x = tf.random_normal([8, 32, 32, 3])
    network = MyAlexNet()
    out = network(x)
    print(out.shape)


if __name__ == '__main__':
    test()

