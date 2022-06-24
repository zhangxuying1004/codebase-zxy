import tensorflow as tf
from tensorflow.python.keras import Model, Sequential, layers

from sentiment_analysis.utils import HyperParameter

# hyperparameter set
hp = HyperParameter()

class MyRNN(Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()

        # transform text to embedding representation
        self.embedding = layers.Embedding(hp.total_words, hp.embedding_len,
                                          input_length=hp.max_sentence_len)

        # two layer rnn
        # normal rnn
        self.rnn = Sequential([
            layers.SimpleRNN(units=units, dropout=0.5, return_sequences=True, unroll=True),
            layers.SimpleRNN(units=units, dropout=0.5, unroll=True)
        ])

        # # lstm rnn
        # self.rnn = Sequential([
        #     layers.LSTM(units=units, dropout=0.5, return_sequences=True, unroll=True),
        #     layers.LSTM(units=units, dropout=0.5, unroll=True)
        # ])

        # # gru rnn
        # self.rnn = Sequential([
        #     layers.GRU(units=units, dropout=0.5, return_sequences=True, unroll=True),
        #     layers.GRU(units=units, dropout=0.5, unroll=True)
        # ])

        self.fc = layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        # print('embedding shape:', x.shape)
        x = self.rnn(x)
        out = self.fc(x)

        return out



def test():
    inputs = tf.random_normal([2500, 80])
    print('input shape:', inputs.shape)
    model = MyRNN(units=64)
    out = model(inputs)
    print('output shape:', out.shape)

if __name__ == '__main__':
    test()