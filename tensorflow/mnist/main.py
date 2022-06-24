import tensorflow as tf
from mnist.model import Network
from mnist.utils import MNIST, preprocess
from tensorflow.python.keras import optimizers, losses
# tf.enable_eager_execution() # 保证可以使用dataset.__iter__()
# 参数设置
batch_size = 64
learning_rate = 1e-3
epochs = 20
# 占位符
x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28*1], name='x-input')
y_gt = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y-output')
# 计算accuracy
def calculate_accuracy(y_gt, y_predict):
    total_num = y_gt.shape[0]
    correct_num = tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(y_gt, 1), tf.argmax(y_predict, 1)), dtype=tf.float32)
    )
    accuracy = correct_num / total_num
    return accuracy

def train(mnist_dataset, y):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_gt, 1))
    loss = tf.reduce_mean(cross_entropy)

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    batch_num = int(len(mnist_dataset.train_op.labels) / batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0

        for epoch in range(epochs):
            for bt in range(batch_num):
                xs, ys = mnist_dataset.train_op.next_batch(batch_size)
                sess.run(train_op, feed_dict={x:xs, y_gt:ys})

                if bt % 100 == 0:
                    x_test, y_test = mnist_dataset.test_op.images, mnist_dataset.test_op.labels
                    y_predict = sess.run(y, feed_dict={x: x_test})

                    accuracy = calculate_accuracy(y_test, y_predict)
                    print("step:{}, accuracy:{}".format(step, sess.run(accuracy)))
                    step += 1

def run1(mnist_dataset):

    network = Network()
    y = network(x)
    train(mnist_dataset, y)


def run2(mnist_dataset):

    x_train, y_train = mnist_dataset.train_data()
    x_val, y_val = mnist_dataset.validate_data()

    x_train = x_train.reshape([-1, 28, 28, 1])
    x_val = x_val.reshape([-1, 28, 28, 1])

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)

    network = Network()
    network.compile(
        optimizer=optimizers.Adam(lr=learning_rate),
        loss=losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    network.fit(
        x_train, y_train,
        epochs=epochs, batch_size=batch_size
    )

    score = network.evaluate(x_val, y_val)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])



if __name__ == '__main__':
    mnist_dataset = MNIST()
    # run1(mnist_dataset)
    run2(mnist_dataset)
