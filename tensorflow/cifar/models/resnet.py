import tensorflow as tf
# from tensorflow.python import keras
from tensorflow.python.keras import Model, layers, Sequential

# ResNet 2~3个卷积层 + 一个short-cut比较合适，不能有维度的衰减
# 每个unit，输入和输出的维度相等，1x1, 3x3, 1x1卷积

# cv research unit: Conv-BN-ReLU

class Basic_Block(layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(Basic_Block, self).__init__()
        # path 1
        self.conv1 = layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        # path2
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda  x: x

    def call(self, inputs, training=None):
        # convolution path    
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        
        # short-cut path
        identity = self.downsample(inputs)
        
        # merge two path
        output = layers.add([x, identity])
        
        output = tf.nn.relu(output)
        return output


class ResNet(Model):

    def __init__(self, layer_dims, num_classes):
        super(ResNet, self).__init__()
        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])

        self.layer1 = self.build_resblock(filter_num=64, blocks=layer_dims[0])
        self.layer2 = self.build_resblock(filter_num=128, blocks=layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(filter_num=256, blocks=layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(filter_num=512, blocks=layer_dims[3], stride=2)

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(units=num_classes)
        # self.fc = layers.Dense(units=num_classes, activation=tf.nn.relu)

    def call(self, inputs, training=None, mask=None):

        x = inputs
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # print(x.shape)
        # [b, chanel]
        x = self.avg_pool(x)
        # print(x.shape)

        # [b, num_classes]
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, stride=1):

        res_blocks = Sequential()
        # may down sample
        res_blocks.add(Basic_Block(filter_num, stride))
        # do not down sample
        for _ in range(1, blocks):
            res_blocks.add(Basic_Block(filter_num, stride=1))
        return res_blocks

def resnet18():
    return ResNet([2, 2, 2, 2], 10)

def resnet34():
    return ResNet([3, 4, 6, 3], 10)


def test():
    model = resnet18()
    x = tf.random_uniform([8, 32, 32, 3])
    out = model(x)
    print(out.shape)

    model.summary()
    
    
if __name__ == '__main__':
    test()

