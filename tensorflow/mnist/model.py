import tensorflow as tf
from tensorflow import nn
from tensorflow.python.keras import Model, layers, Sequential

# tensorflow框架自定义层(继承layers.layer)或网络(继承keras.Model)，需要的组合是__init__()和call()两个函数

# [b, 28, 28, 1] =>[b, 10]
class Network(Model):
    def __init__(self):
        super(Network, self).__init__()

        self.mylayers = [
            # unit1
            layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            # unit2
            layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            # flatten the tensor
            layers.Flatten(),
            # 2 full-connected layers
            layers.Dense(512, activation=nn.relu),
            layers.Dense(10, activation=nn.softmax)
            # layers.Dense(10, activation=None)
        ]
        # 根据tensorflow的版本确定网络的最后一层要不要加activation=nn.softmax
        # 如果tf版本低于1.13:
        #     则需要添加,
        #     训练时在model.compile中设置loss=keras.losses.categorical_crossentropy
        # 如果tf版本等于或者高于1.13:
        #     则不需要添加,
        #     训练时在model.compile中设置loss=keras.losses.CategoricalCrossentropy(from_logits=True)

        self.net = Sequential(self.mylayers)

    def call(self, inputs, training=None, mask=None):
        x = tf.reshape(inputs, [-1, 28, 28, 1])
        x = self.net(x)
        return x


def test():
    x = tf.random_normal([4, 28, 28, 1])
    network = Network()
    y = network(x)
    print(y.shape)

if __name__ == '__main__':
    test()