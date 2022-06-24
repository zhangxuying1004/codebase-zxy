import tensorflow as tf
from tensorflow.python.keras import Model, Sequential, layers, applications


class Transfer_Model(Model):
    def __init__(self):
        super(Transfer_Model, self).__init__()
        # load the pre_trained model
        self.transfer_net = applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='max')
        self.transfer_net.trainable = False

        # add the pre_trained model into my model
        self.net = Sequential([
            self.transfer_net,
            layers.Dense(5, activation=tf.nn.softmax)
        ])

    def call(self, inputs, training=None, mask=None):
        x = inputs
        out = self.net(x)
        return out
