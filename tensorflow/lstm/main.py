from tensorflow.python import keras
from lstm.model import MyLSTM
from lstm.utils import Config, preprocess

# data load and process
hps = Config()
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=hps.total_words)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_train.shape)

x_train, y_train = preprocess(x_train, y_train)
x_test, y_test = preprocess(x_test, y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_train.shape)

def run():
    model = MyLSTM(hidden_len=64)

    model.compile(
        optimizer=keras.optimizers.Adam(hps.learning_rate),
        loss=keras.losses.binary_crossentropy,
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train,
        epochs=hps.epochs, batch_size=hps.batch_size
    )

    score = model.evaluate(x_test, y_test, batch_size=hps.batch_size)
    print('loss:', score[0])
    print('accuracy:', score[1])

if __name__ == '__main__':
    run()

