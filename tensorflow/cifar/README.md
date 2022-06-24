**(1) cifar数据集简介**   
>cifar数据集的下载地址: https://www.cs.toronto.edu/~kriz/cifar.html  
**cifar10数据集**  
cifar-10数据集由10个类的60k个32x32彩色图像组成，每个类有6k个图像，这些类完全相互排斥，没有重叠。  
数据集分为5个训练批次和1个测试批次，每个批次有10k个图像，即数据集有50k个训练图像和10k个测试图像。  
其中，测试批次包含来自每个类别的恰好1k个随机选择的图像；而训练批次以随机顺序包含剩余图像，但一些训练批次可能包含来自一个类别的图像比另一个更多，总体来说，5个训练集之和包含来自每个类的正好5k张图像。  
**cifar100数据集**  
cifar100中的图片与cifar10中的图片完全一样，只是类别的划分更加精细。  
对于下载到本地的cifar10/cifar100，官方给出批次文件的加载方式如下：   
```python   
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict   
```
>以这种方式加载每个批次文件得到的dict都包含**data**和**labels**两种元素。其中，  
**data**是一个类型为uint8、大小为10000x3072的numpy数组。阵列的每一行存储32x32彩色图像即每一行存储32*32*3=3072个数字信息。前1024个条目包含红色通道值，中间1024个包含绿色通道值，最后1024个包含蓝色通道值。图像以行优先顺序存储，以便数组的前32个条目是图像第一行的红色通道值。  
**labels**是10000个由0~9数字组成的列表，索引i处的数字表示阵列数据中第i个图像的标签。  
此外，cifar10/100数据集中还有一个**batches.meta**文件，它也包含一个字典文件，用label_names[i]来指明类别i的名称。  

**（2）代码说明**   
>model文件夹中，是使用keras框架自定义的AlexNet、VGGNet、ResNet和InceptionNet。   
utils.py中，Config类用于模型中超参数的设置，其他的method用于预处理。   
train.py中，加载数据集，使用一般的方法对模型进行训练，并使用Tensorboard监测几个重要的scalar的变化，最后保存训练得到模型。   
eval.py中，加载数据集，加载保存的模型，对模型进行测试。   
main.py中，加载数据集，使用keras框架封装的函数对模型进行训练和测试。  
注：代码中使用的是keras框架中封装的方法来加载数据集，代码如下：   
```python 
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
```

