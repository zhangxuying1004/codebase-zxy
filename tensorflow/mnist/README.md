**（1）内容简介**    
>这一部分主要是MNIST数据集和kears框架的使用。   
utils.py涉及了mnist数据的获取和批处理。  
model.py设计了使用keras框架自定义网络模型,使用的是LeNet5模型。  
main.py中涉及了两种训练模型的方法，其中run1()是使用自定义的方法训练模型，run2()是使用keras框架封装的compile和fit函数训练模型。  
注:使用keras封装的函数训练模型时需要考虑到使用的tensorflow的版本，  
如果tf版本低于1.13，在定义模型时，最后一层的激活函数设置为softmax，训练模型时，compile中设置loss=keras.losses.categorical_crossentropy；  
如果tf版本高于或者等于1.13，在定义模型时，最后一层的激活函数设置为None，训练模型时，compile中设置loss=keras.losses.CategoricalCrossentropy(from_logits=True)。  

**(2) MNIST数据集简介**    
>MNIST数据集由深度学习三驾马车之一的Yann LeCun创造，用于书写数字识别，下载网址：http://yann.lecun.com/exdb/mnist/  
图片中包含一个0~9的数字，每个数字有7k张，这个数据集共有70k张图片，每张图片都有相对应的label。   
其中，train_data: 60k，test_data: 10k或者train_data: 55k, validation_data: 5k, test_data: 10k。   
下面是使用tensorflow处理下载到本地的MNSIT数据集的代码：
```python
from tensorflow.examples.tutorials.mnist import input_data
MNIST_dir = 'xxx/'
dataset = input_data.read_data_sets(MNIST_dir, one_hot=True)
# 获取全部数据
train_images, train_labels = dataset.train.images, dataset.train.labels
validation_images, validation_labels = dataset.validation.images, dataset.validation.labels
test_images, test_labels = dataset.test.images, dataset.test.labels 
# 获取一个batch的数据
train_batch_images, train_batch_labels = dataset.train.next_batch(batch_size), dataset.train.next_batch(batch_size)
validation_batch_images, validation_batch_labels = dataset.validation.next_batch(batch_size), dataset.validation.next_batch(batch_size)
test_batch_images, test_batch_labels = dataset.test.next_batch(batch_size), dataset.test.next_batch(batch_size)
```

**（3）遇到的坑**    
>定义模型时，最后一层没有设置activation，即模型的输出是logits。  
使用keras框架封装的函数训练模型，因为tensorflow版本较低，keras.losses中没有CategoricalCrossentropy类，发现只有keras.losses.categorical_crossentropy，感觉二者非常接近，就索性将compile中的loss设置成了keras.losses.categorical_crossentropy，但是categorical_crossentropy的参数中只有y_true和y_predict，无法设置from_logits=True，没有想那么多，以为keras内部封装的函数会自动处理，就没有设置，直接训练。造成的结果是loss基本不变或者上下波动，accuracy也是上下波动并且非常低（接近10%）。  
检查了半天也没发现错在哪，于是乎陷入了焦虑。。。  
接着，自己又写了一个训练模型的函数，一般的训练方式，所有batch_size，epoch，lr等超参数都设置成一样。  
惊奇地发现，同样的模型，同样的数据集，同样的超参数设置，自定义的训练，最后的accuracy接近98%或者99%。  
于是乎，更加焦虑。。。  
最后，发现模型的输出是logits，而keras框架中的compile设置时没有体现，觉得可能是这个问题，便在模型的最后一层Dence中设置activation=nn.sotfmax，重新使用框架封装的函数进行训练，发现loss和accuracy有了惊人的改变，最后的accuracy接近99%。  
查阅官方文档才发现，原来是由于tensorflow版本造成的keras.losses中categorical_crossentropy与CategoricalCrossentropy的不同。  
