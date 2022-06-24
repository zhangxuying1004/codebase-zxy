from tensorflow.python.keras import datasets, preprocessing, optimizers, losses
from sentiment_analysis.utils import HyperParameter
from sentiment_analysis.model import MyRNN


def run():
    hp = HyperParameter()

    # data load and process
    (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=hp.total_words)

    # x_train:[b, 80]
    # x_test:[b, 80]
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=hp.max_sentence_len)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=hp.max_sentence_len)

    hidden_units = 64

    model = MyRNN(hidden_units)
    model.compile(
        optimizer=optimizers.Adam(lr=hp.learning_rate),
        loss=losses.binary_crossentropy,
        metrics=['accuracy']
    )
    model.fit(x_train, y_train,
              batch_size=hp.batch_sz,epochs=hp.epochs)
    score = model.evaluate(x_test, y_test)
    print('test loss: ', score[0])
    print('test accuracy: ', score[1])


if __name__ == '__main__':
    run()