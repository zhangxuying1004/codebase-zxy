**（1）简介**  
>这一部分主要是imdb数据集的加载和几种RNN的基本使用。其中，  
utils.py中是主要超参数的设置；  
model.py中是SimpleRNN, LSTM和GRU三种rnn级别的模型。  
main.py是imdb数据集的加载，使用keras封装的函数训练模型、测试模型。     

**（2）知识点**   
>keras框架使用keras.layers.SimpleRNN(params), keras.layers.LSTM(params), keras.layers.GRU(params)调用三种rnn模型，下面说明一下params（参数列表）的设置。      
units表示输出空间的维度，也是隐含单元个数，是必须设置的参数。    
return_sequences=True/False，表示返回的是所有时间步输出组成的整个输出序列还是最后一个时间步的输出，默认是False，返回最后一个时间步的输出。   
return_state=True/False，表示是否返回最后一个时间步输出的隐含状态，默认是False，不输出，若设置为True，则rnn返回的是一个长度为2的列表，列表中的第一个元素是模型的输出（输出序列或者最后一个时间步的输出），第二个元素是最后一个时间步输出的隐含状态。   
unroll=True/Fales，表示是否将网络展开，默认为False，不展开，若设置为True，可以加速RNN，但是会占用大量内存，因此只适用于短序列。      
