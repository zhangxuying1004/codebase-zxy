import tensorflow as tf
from tensorflow.python.keras import datasets

from cifar.utils import Config, one_hot, get_step_num, get_batch_data, check_dir
from cifar.models.alex import MyAlexNet
from cifar.models.vgg import MyVGGNet
from cifar.models.resnet import resnet18
from cifar.models.inception import GoogLeNet

# 占位符
x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
y_gt = tf.placeholder(dtype=tf.int32, shape=[None, 10])


def main():
    # dataset
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    y_train = one_hot(y_train, depth=10)
    y_test = one_hot(y_test, depth=10)

    # 参数
    hps = Config()
    saved_model_dir = hps.checkpoint_dir
    check_dir()

    # 模型及评价指标的定义
    network = MyAlexNet()
    # network = MyVGGNet()
    # network = resnet18()
    # network = GoogLeNet()

    out = network(x)
    with tf.name_scope('loss'):
        cross_entropy = tf.losses.softmax_cross_entropy(y_gt, out)
        loss = tf.reduce_mean(cross_entropy)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(
            tf.argmax(out, axis=1), tf.argmax(y_gt, axis=1)
        )
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32)
        )
    with tf.name_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate=hps.lr).minimize(loss, global_step=global_step)

    # tensorboad 记录
    with tf.name_scope('summary'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)

        summary_op = tf.summary.merge_all()

    # 一个epoch内的batch数目
    step_num = get_step_num(x_train, hps)

    with tf.Session() as sess:
        # 初始化参数！
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(hps.tensorboard_train_dir, sess.graph)
        test_writer = tf.summary.FileWriter(hps.tensorboard_test_dir)  # 注：此处没有sess.graph

        for epoch in range(hps.epochs):
            for step in range(step_num):
                # 获取一个batch的数据
                x_train_batch, y_train_batch = get_batch_data(x_train, y_train, step, hps)
                x_test_batch, y_test_batch = get_batch_data(x_test, y_test, step, hps)

                # 训练模型
                train_summary_value, train_loss_value, train_accuracy_value, _ = \
                    sess.run([summary_op, loss, accuracy, train_op], feed_dict={x: x_train_batch, y_gt: y_train_batch})
                train_loss_value, train_accuracy_value, _ = sess.run([loss, accuracy, train_op],
                                                                     feed_dict={x: x_train_batch, y_gt: y_train_batch})

                # 计算测试信息，观察是否出现过拟合
                test_summary_value = sess.run(summary_op, feed_dict={x: x_test_batch, y_gt: y_test_batch})

                if global_step.eval() % 500 == 0:
                    print(global_step.eval(), 'loss:', float(train_loss_value), ' accuracy:', train_accuracy_value)

                    train_writer.add_summary(train_summary_value, global_step.eval())
                    test_writer.add_summary(test_summary_value, global_step.eval())

            # save model
            saver = tf.train.Saver()
            saver.save(sess, saved_model_dir + hps.model_name + str(epoch))
        train_writer.close()
        test_writer.close()


if __name__ == '__main__':
    main()
