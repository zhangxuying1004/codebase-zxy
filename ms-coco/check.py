"""
查看MS COCO标注数据的组织形式
"""

import os
import argparse


def image_captioning(datasplit='train'):
    path_to_train_ann_file = './datasets/annotations/captions_train2014.json'
    path_to_val_ann_file = './datasets/annotations/captions_val2014.json'
    path_to_test_ann_file = './datasets/annotations/image_info_test2014.json.json'

    if datasplit == 'train':
        path_to_ann_file = path_to_train_ann_file
    elif datasplit == 'val':
        path_to_ann_file = path_to_val_ann_file
    else:
        path_to_ann_file = path_to_test_ann_file
    
    assert os.path.exists(path_to_ann_file)
    with open(path_to_ann_file, 'r') as ann_f:
        ann = json.load(ann_f)
    print(type(ann))
    
    images_info = ann['images']
    
    print(type(images_info))
    print(images_info[0])
    
    if datasplit != 'test':
        anns_info = ann['annotations']
        print(type(anns_info))
        print(anns_info[0])
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data process') 
    parser.add_argument('--datasplit', type=str, default='test') # 'train', 'val', 'test'
    
    args = parser.parse_args()
    
    image_captioning(args.datasplit)
    
 
