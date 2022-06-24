import tensorflow as tf
from tensorflow.python.keras import Model, Sequential, layers

class GoogLeNet(Model):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()

        self.conv1 = Sequential([
            layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same'),
            layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same'),
            layers.Activation('relu')
        ])

        self.conv2 = Sequential([
            layers.Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='same'),
            layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same'),
            layers.Activation('relu')
        ])

        self.inc3a = Inception_Block(out_channels_1_1=64, reduce_channels_3_3=96, out_channels_3_3=128, reduce_channels_5_5=16, out_channels_5_5=32, pool_proj=32, check=256)
        self.inc3b = Inception_Block(out_channels_1_1=128, reduce_channels_3_3=128, out_channels_3_3=192, reduce_channels_5_5=32, out_channels_5_5=96, pool_proj=64, check=480)

        self.inc4a = Inception_Block(out_channels_1_1=192, reduce_channels_3_3=96, out_channels_3_3=208, reduce_channels_5_5=16, out_channels_5_5=48, pool_proj=64, check=512)
        self.inc4b = Inception_Block(out_channels_1_1=160, reduce_channels_3_3=112, out_channels_3_3=224, reduce_channels_5_5=24, out_channels_5_5=64, pool_proj=64, check=512)
        self.inc4c = Inception_Block(out_channels_1_1=128, reduce_channels_3_3=128, out_channels_3_3=256, reduce_channels_5_5=24, out_channels_5_5=64, pool_proj=64, check=512)
        self.inc4d = Inception_Block(out_channels_1_1=112, reduce_channels_3_3=144, out_channels_3_3=288, reduce_channels_5_5=32, out_channels_5_5=64, pool_proj=64, check=528)
        self.inc4e = Inception_Block(out_channels_1_1=256, reduce_channels_3_3=160, out_channels_3_3=320, reduce_channels_5_5=32, out_channels_5_5=128, pool_proj=128, check=832)

        self.inc5a = Inception_Block(out_channels_1_1=256, reduce_channels_3_3=160, out_channels_3_3=320, reduce_channels_5_5=32, out_channels_5_5=128, pool_proj=128, check=832)
        self.inc5b = Inception_Block(out_channels_1_1=384, reduce_channels_3_3=192, out_channels_3_3=384, reduce_channels_5_5=48, out_channels_5_5=128, pool_proj=128, check=1024)

        self.last_seq = Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(num_classes),
            # layers.Dense(num_classes, activation=tf.nn.softmax)
        ])

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.inc3a(x)
        x = self.inc3b(x)

        x = self.inc4a(x)
        x = self.inc4b(x)
        x = self.inc4c(x)
        x = self.inc4d(x)
        x = self.inc4e(x)

        x = self.inc5a(x)
        x = self.inc5b(x)

        out = self.last_seq(x)
        return out



class Inception_Block(Model):
    def __init__(self, out_channels_1_1, reduce_channels_3_3, out_channels_3_3, reduce_channels_5_5, out_channels_5_5, pool_proj, check):
        super(Inception_Block, self).__init__()
        assert out_channels_1_1 + out_channels_3_3 + out_channels_5_5 + pool_proj == check

        self.inception_conv_1 = Sequential([
            layers.Conv2D(filters=out_channels_1_1, kernel_size=(1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])


        self.inception_conv_3 = Sequential([
            layers.Conv2D(filters=reduce_channels_3_3, kernel_size=(1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2D(filters=out_channels_3_3, kernel_size=(3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])


        self.inception_conv_5 = Sequential([
            layers.Conv2D(filters=reduce_channels_5_5, kernel_size=(1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2D(filters=out_channels_5_5, kernel_size=(3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])

        self.inception_conv_pool = Sequential([
            layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same'),
            layers.Conv2D(filters=pool_proj, kernel_size=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])

    def call(self, inputs, training=None, mask=None):
        x = inputs

        inception_1 = self.inception_conv_1(x)
        inception_3 = self.inception_conv_3(x)
        inception_5 = self.inception_conv_5(x)
        inception_pool = self.inception_conv_pool(x)

        inception = tf.concat([inception_1, inception_3, inception_5, inception_pool], axis=3)

        return inception

def test():
    # x = tf.random_uniform([8, 32, 32, 3])

    x = tf.random_normal([8, 224, 224, 3])
    model = GoogLeNet(num_classes=10)
    out = model(x)
    print(out.shape)

    model.summary()
if __name__ == '__main__':
    test()
