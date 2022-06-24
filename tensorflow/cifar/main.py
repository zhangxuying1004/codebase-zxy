from tensorflow.python.keras import datasets, optimizers, losses
from cifar.utils import Config, one_hot

from cifar.models.alex import MyAlexNet
from cifar.models.vgg import MyVGGNet
from cifar.models.resnet import resnet18
from cifar.models.inception import GoogLeNet


# dataset load and process
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
y_train = one_hot(y_train, depth=10)
y_test = one_hot(y_test, depth=10)

# 参数
hps = Config()

# 网络定义
network = MyAlexNet()
# network = MyVGGNet()
# network = resnet18()
# network = GoogLeNet()

# 模型训练
network.compile(
    optimizer=optimizers.Adam(lr=hps.lr),
    loss=losses.categorical_crossentropy,
    metrics=['accuracy']
)
network.fit(
    x_train, y_train,
    epochs=hps.epochs, batch_size=hps.batch_size
)
# 模型评价
score = network.evaluate(x_test, y_test)
print('test loss: ', score[0])
print('test accuracy: ', score[1])

