import os
import h5py
from tqdm import tqdm
import argparse

def read_image_ids(path_to_ann_files):
    image_ids = []
    for path_to_ann_file in path_to_ann_files:
        with open(path_to_ann_file, 'r') as ann_f:
            ann_info = json.load(ann_f)
        images_info = ann_info['images']
        image_ids += [item['id'] for item in images_info]
    return image_ids

 def dataset_process(path_to_before_feats, path_to_after_feats, image_ids=None): 
    assert image_ids is not None
    with h5py.File(path_to_after_feats, 'w') as f_after:    # 生成新的hdf5文件
        f_before = h5py.File(path_to_before_feats, 'r')     # 读取旧的hdf5文件
        for i in tqdm(range(len(image_ids))):
            image_id = int(image_ids[i])
            try:
                feats = f_before['%d_features' % image_id][()]
            except KeyError:
                warnings.warn('Could not find {} feats for {}'.format(backbone, image_id))
                return None
            f_after.create_dataset('%d_regions' % image_id, data=feats)   # 将数据插入新文件中
    f_before.close()
    f_after.close()
    print('Done!')
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data process') 
    parser.add_argument('--backbone', type=str, default='X101') # 'X101', 'resnet50'
    parser.add_argument('--datatype', type=str, default='test') # 'trainval', 'test'
    
    # trainval
    parser.add_argument('--path_to_trainval_ann_files', type=list, default=['./datasets/annotations/captions_train2014.json', './datasets/annotations/captions_val2014.json'])
    parser.add_argument('--path_to_before_X101feats_trainval', type=str, default='./datasets0/feats/X101_coco_test.hdf5')
    parser.add_argument('--path_to_before_res50feats_trainval', type=str, default='./datasets0/feats/res50_coco_test.hdf5')
    parser.add_argument('--path_to_after_X101feats_trainval', type=str, default='./dataset/feats/X101_coco_test.hdf5')
    parser.add_argument('--path_to_after_res50feats_trainval', type=str, default='./datasets/feats/res50_coco_test.hdf5')
    
    # test ann file
    parser.add_argument('--path_to_test_ann_files', type=str, default='test')  # test
    parser.add_argument('--path_to_before_X101feats_test', type=str, default='./datasets0/feats/X101_coco_test.hdf5')
    parser.add_argument('--path_to_before_res50feats_test', type=str, default='./datasets0/feats/res50_coco_test.hdf5')
    parser.add_argument('--path_to_after_X101feats_test', type=str, default='./dataset/feats/X101_coco_test.hdf5')
    parser.add_argument('--path_to_after_res50feats_test', type=str, default='./datasets/feats/res50_coco_test.hdf5')
    
    args = parser.parse_args()
    print(args)
    
    if args.datatype == 'trainval':
        image_ids = read_image_ids(args.path_to_trainval_ann_files)
    else:
        image_ids = read_image_ids(args.path_to_test_ann_files)
    
    if args.backbone == 'X101':
        if args.datatype == 'trainval':
            dataset_process(args.path_to_before_X101feats_trainval, args.path_to_after_X101feats_trainval, image_ids)
        else:
            dataset_process(args.path_to_before_X101feats_test, args.path_to_after_X101feats_test, image_ids)
    else:
        if args.datatype == 'trainval':
            dataset_process(args.path_to_before_res50feats_trainval, args.path_to_after_res50feats_trainval, image_ids)
        else:
            dataset_process(args.path_to_before_res50feats_test, args.path_to_after_res50feats_test, image_ids)
    
    
    
