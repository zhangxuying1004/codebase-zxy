**（1）简介**
>这一部分是使用keras框架，用rnncell级别的模块实现lstm。   
utils.py主要包含两部分的内容：  
一个Config类，用于设置模型中用到的超参数。   
二是对数据集进行预处理，因为序列模型每次处理一个batch的数据，batch这个条件必须保证，但是数据集的数目未必是batch_size的整数倍，于是在加载数据集的时候需要舍弃最后一个不足一个batch的数据。   
model.py中主要是模型的定义，包含两部分的内容，    
一是lstmcell的定义，二是lstm的定义。    
main.py是加载数据集，并对模型进行训练和测试。   

**（2）遇到的坑**
>至今没有解决！！！    
**坑一：**  
在keras框架中已经封装好了lstmcell的实现，众所周知，lstmcell的初始状态通常设置为全0，但是具体实现时，发现这个state却不知道该怎么设置！！
在较高版本的tensorflow中，直接设置为
```python
 self.state = [tf.zeros([batch_size, hidden_units]),tf.zeros([batch_size, hidden_units])]
```
>我使用的tensorflow版本是1.8，发现这么设置不行，查阅官方文档，发现state就是list，不知道究竟怎么回事。  
最后无奈之下，只好自己动手实现mylstmcell类。  
此外，在keras封装的simplernncell、grucell中也存在这个问题！   
**坑二：**  
还是与state相关的问题。。      
对于自定义的mylstmcell，调用其instance中的call方法，输入是inputs，在__main__中测试时，发现使用      
```python
inputs=x, state
instance_name(inputs)
```
>即可调用，但是，在lstmcell类中，发现这么使用，会报错：TypeError：Tensor object are not iterable when eager execution is not enableed. To iterate over this tensor use tf.map_fn. 这个问题纠结了好久。  
最后发现把x, state加括号即(x, state)作为instance_name的输入就不报错了。    
感觉前者也是一个元组啊，而且测试时没错，使用时却出错了，没弄清是怎么回事。
