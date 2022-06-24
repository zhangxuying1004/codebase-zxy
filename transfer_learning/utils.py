import tensorflow as tf
import os
import glob, csv, random
import cv2

# hyperparametrs config
class Config:
    def __init__(self):
        self.batch_size = 128
        self.epochs = 30
        self.learning_rate = 3e-3


# check csv file exit or not, if not, create
def check_csv(root, filename, name2label):
    if not os.path.exists(os.path.join(root, filename)):
        images = []
        for name in name2label.keys():
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))

        random.shuffle(images)

        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:
                name = img.split(os.sep)[-2]
                label = name2label[name]
                writer.writerow([img, label])
            print('write into csv file:', filename)

# load data from csv file
def load_from_csv(root, filename, name2label):
    check_csv(root, filename, name2label)

    images = []
    labels = []

    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            label = int(label)

            images.append(img)
            labels.append(label)

    assert len(images) == len(labels)
    if len(images) != 0:
        print('read from csv file')
    return images, labels

# root is path dir where the dataset in
def load_pokemon(root='pokemon', mode='train'):
    # create index for each category
    name2label = {}
    for name in sorted(os.listdir(os.path.join(root))):
        if not os.path.isdir(os.path.join(root, name)):
            continue

        name2label[name] = len(name2label.keys())

    # images here is the images dir, labels is category index
    images, labels = load_from_csv(root, 'images.csv', name2label)

    # return specific dataset according to mode
    data_num = len(images)
    if mode == 'train':  # 0->60%
        images = images[:int(0.6 * data_num)]
        labels = labels[:int(0.6 * data_num)]
    elif mode == 'val':  # 60%->80%
        images = images[int(0.6 * data_num):int(0.8 * data_num)]
        labels = labels[int(0.6 * data_num):int(0.8 * data_num)]
    else:  # 20% = 80%->100%
        images = images[int(0.8 * data_num):]
        labels = labels[int(0.8 * data_num):]

    return images, labels, name2label

# load image according to its dir
def transform_from_path(x):
    x = cv2.imread(x)
    x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_CUBIC)
    x = tf.cast(x, dtype=tf.float32) / 255.
    return x

def preprocess(images, labels):
    images_list = []
    for image in images:
        images_list.append(transform_from_path(image))
    real_images = tf.stack(images_list)
    labels_one_hot = tf.one_hot(labels, depth=5)

    with tf.Session() as _:
        return real_images.eval(), labels_one_hot.eval()


def test():

    images, labels, table = load_pokemon('train')
    # images, labels, table = load_pokemon('val')
    # images, labels, table = load_pokemon('test')

    images, labels = preprocess(images, labels)

    print(type(images), type(labels))
    print(images.shape, labels.shape)


if __name__ == '__main__':
    test()


