from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 参数是mnist数据集保存的路径
class MNIST:
    def __init__(self, MNIST_dir = '/home/zhangxuying/Resource/DataSet/MNIST/'):
        self.dataset = input_data.read_data_sets(MNIST_dir, one_hot=True)

        # op includes images, labels, next_batch(batch_size)
        self.train_op = self.dataset.train
        self.validate_op = self.dataset.validation
        self.test_op = self.dataset.test

    # total data
    def train_data(self):
        return (self.train_op.images, self.train_op.labels)
    def validate_data(self):
        return (self.validate_op.images, self.validate_op.labels)
    def test_data(self):
        return (self.test_op.images, self.test_op.labels)

    # batch data
    def batch_train_data(self, batch_size):
        return self.train_op.next_batch(batch_size)
    def batch_validate_data(self, batch_size):
        return self.validate_op.next_batch(batch_size)
    def batch_test_data(self, batch_size):
        return self.test_op.next_batch(batch_size)

def preprocess(x, y):
    x = tf.cast(x, tf.float32) - 0.5
    y = tf.cast(y, tf.int32)
    return x, y


def test():
    mnist_dataset = MNIST()
    (train_images, train_labels) = mnist_dataset.train_data()
    (val_images, val_labels) = mnist_dataset.validate_data()
    (test_images, test_labels) = mnist_dataset.test_data()

    print('train files:({0}, {1})'.format(train_images.shape, train_labels.shape))
    print('validation files:({0}, {1})'.format(val_images.shape, val_labels.shape))
    print('test files:({0}, {1})'.format(test_images.shape, test_labels.shape))


if __name__ == '__main__':
    test()