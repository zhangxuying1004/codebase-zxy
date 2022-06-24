from tensorflow.python.keras import optimizers, losses
from transfer_learning.utils import Config, load_pokemon, preprocess
from transfer_learning.model import Transfer_Model

# dataset
train_images, train_labels, _ = load_pokemon(mode='train')
test_images, test_labels, _ = load_pokemon(mode='test')
train_images, train_labels = preprocess(train_images, train_labels)
test_images, test_labels = preprocess(test_images, test_labels)

print('train:', train_images.shape, train_labels.shape)
print('test:', test_images.shape, test_labels.shape)

hps = Config()

# build model
net = Transfer_Model()
net.summary()

# train
net.compile(
    optimizer=optimizers.Adam(hps.learning_rate),
    loss=losses.categorical_crossentropy,
    metrics=['accuracy']
)
net.fit(
    train_images, train_labels,
    batch_size=hps.batch_size, epochs=hps.epochs
)

# test
score = net.evaluate(test_images, test_labels)
print('loss:', score[0])
print('accuracy:', score[1])

