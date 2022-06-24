import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python import keras
from lstm.utils import Config

# hyperparamers config
hps = Config()

# model define
class MyLSTMCell(keras.Model):
    def __init__(self, hidden_len):
        super(MyLSTMCell, self).__init__()
        # three gate
        self.gate_i = layers.Dense(1, activation=tf.nn.sigmoid)
        self.gate_f = layers.Dense(1, activation=tf.nn.sigmoid)
        self.gate_o = layers.Dense(1, activation=tf.nn.sigmoid)

        # linear transform
        self.Cell = layers.Dense(hidden_len, activation=tf.nn.tanh)

    def call(self, inputs, training=None, mask=None):
        # variable process
        x, state = inputs
        h, c_pre = state[0], state[1]
        x = tf.concat([x, h], axis=1)
        # gate process
        i_value = self.gate_i(x)
        f_value = self.gate_f(x)
        o_value = self.gate_o(x)
        # current memory cell
        c_cur = self.Cell(x)
        # final value
        c = i_value * c_cur + f_value * c_pre
        h = o_value * tf.nn.tanh(c)
        out = h

        return [out, h, c]


class MyLSTM(keras.Model):
    def __init__(self, hidden_len=64):
        super(MyLSTM, self).__init__()

        self.embedding = layers.Embedding(hps.total_words, hps.embedding_len,
                                          input_length=hps.max_sentence_len)

        self.state = [tf.zeros(shape=[hps.batch_size, hidden_len]), tf.zeros(shape=[hps.batch_size, hidden_len])]
        self.lstm_cell = MyLSTMCell(hidden_len=hidden_len)

        self.drop = layers.Dropout(0.5)
        self.fc = layers.Dense(1, activation=tf.nn.sigmoid)


    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.embedding(x)

        state = self.state
        out = self.state[0]
        for word in tf.unstack(x, axis=1):
            [out, h, c] = self.lstm_cell((word, state))
            state = [h, c]

        x = self.drop(out)
        prob = self.fc(x)
        return prob

# model test #
def test():

    x = tf.random_normal([128, 80])

    # # lstmcell model test
    # state = [tf.zeros(shape=[128, 64]), tf.zeros(shape=[128, 64])]
    # inputs = x, state
    # model = MyLSTMCell(hidden_len=64)
    # [out, h, c] = model(inputs)
    # print(out.shape)


    # lstm model test
    model = MyLSTM(hidden_len=64)
    out = model(x)
    print(out.shape)


if __name__ == '__main__':
    test()
