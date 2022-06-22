
## Image captioning task （训练集+验证集）
两个标注文件：captions_train2014.json，captions_train2014.json。  
这两个文件中存储的数据均为字典，包含['info', 'images', 'licenses', 'annotations']四类信息（四个键），对应的四类值按照列表的形式组织，每个列表元素为一个字典数据。  
其中，最重要的是'images'和'annotations'。依据'images'键可以获得每张图像的信息，如图像尺寸、图像id等；依据'annotations'可以获得每个文本标注的信息，如标注内容，标注id和对应的图像id。  
 

## 测试集
只有一个记录文件：image_info_test2014.json。  
该文件存储的数据也是字典，包含['info', 'images', 'licenses', 'categories']四类信息，对应的四类值的组织形式与上相同。   
其中，最重要的是'images'和'categories'。依据'images'键可以获得每张图像的信息；依据'categories'可以获得图像中视觉对象的类别信息，如类别名，类别id和子类名。  

