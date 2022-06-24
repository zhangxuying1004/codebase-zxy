import tensorflow as tf
from tensorflow.python.keras import datasets
import os
from cifar.utils import Config, one_hot
from cifar.models.alex import MyAlexNet
from cifar.models.vgg import MyVGGNet
from cifar.models.resnet import resnet18
from cifar.models.inception import GoogLeNet

def main():
    # dataset load and process
    _, (x_test, y_test) = datasets.cifar10.load_data()
    # print(x_test.shape, y_test.shape)
    y_test = one_hot(y_test)

    # 参数
    hps = Config()

    # 模型及评价指标的定义
    x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
    y_gt = tf.placeholder(dtype=tf.int32, shape=[None, 10])
    network = MyAlexNet()
    # network = MyVGGNet()
    # network = resnet18()
    # network = GoogLeNet()

    out = network(x)
    correct_prediction = tf.equal(
        tf.argmax(out, axis=1), tf.argmax(y_gt, axis=1)
    )
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32)
    )

    # 保存的模型的路径
    files = os.listdir(hps.checkpoint_dir)
    index = int((len(files) - 1) / 3) - 1
    latest_model = hps.model_name + str(index)
    latest_model_dir = hps.checkpoint_dir + latest_model
    # print(latest_model_dir)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, latest_model_dir)
        accuracy_value = sess.run(accuracy, feed_dict={x:x_test, y_gt:y_test})

        print('test dataset accuracy: ', accuracy_value)

if __name__ == '__main__':
    main()