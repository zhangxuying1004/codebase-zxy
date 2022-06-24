
from tensorflow.python import keras

# hyperparamer set and preprocess the dataset
class Config:
    def __init__(self):
        self.epochs = 10
        self.batch_size = 128
        self.learning_rate = 1e-3

        self.total_words = 10000
        self.max_sentence_len = 80
        self.embedding_len = 100


def preprocess(x, y):
    hps = Config()

    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=hps.max_sentence_len)

    total_num = x.shape[0]
    extra_num = total_num % hps.batch_size
    return x[:-extra_num, :], y[:-extra_num]

