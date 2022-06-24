import tensorflow as tf
from tensorflow import nn
from tensorflow.python.keras import Model, layers, Sequential

# VGG-16 network
class MyVGGNet(Model):
    def __init__(self):
        super(MyVGGNet, self).__init__()

        self.Layers = [
            layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=nn.relu),
            layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2),

            layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=nn.relu),
            layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2),

            layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=nn.relu),
            layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=nn.relu),
            layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2),

            layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=nn.relu),
            layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=nn.relu),
            layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2),

            layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=nn.relu),
            layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=nn.relu),
            layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2),

            layers.Flatten(),

            layers.Dense(units=4096, activation=nn.relu),
            layers.Dense(units=4096, activation=nn.relu),
            layers.Dense(units=10, activation=nn.softmax),
            # layers.Dense(units=10),

        ]

        self.net = Sequential(self.Layers)
        self.net.build(input_shape=[None, 32, 32, 3])
        # self.net.build(input_shape=[None, 224, 224, 3])

    def call(self, inputs, training=None, mask=None):
        inputs = tf.reshape(inputs, [-1, 32, 32, 3])
        out = self.net(inputs)
        return out

def test():

    x = tf.random_normal([8, 32, 32, 3])
    network = MyVGGNet()
    out = network(x)
    print(out.shape)


if __name__ == '__main__':
    test()



