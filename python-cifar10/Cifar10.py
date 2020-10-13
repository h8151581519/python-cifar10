import numpy as np

def unpickle(path):
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def load_batch_data(path):
    data = unpickle(path)
    imgs = np.array(data[b'data'])
    labs = np.array(data[b'labels'])
    return imgs, labs

def cifar10_load_train_data(dir_path, is_to_gray=False):
    num_of_files = 5
    num_of_batches = 10000
    num_of_train = num_of_files * num_of_batches
    img_w, img_h, img_dim= 32, 32, 3
    imgs = np.zeros(shape=[num_of_train, img_w, img_h, img_dim], dtype=np.float32)
    labs = np.zeros(shape=[num_of_train, 1], dtype=np.int8)
    for _i in range(num_of_files):
        path = dir_path + '/data_batch_{}'.format(_i+1)
        batch_imgs, batch_labs = load_batch_data(path)
        imgs[(_i*num_of_batches):((_i+1)*num_of_batches)] = batch_imgs.reshape([-1, img_dim, img_w, img_h]).transpose([0, 2, 3, 1])
        labs[(_i*num_of_batches):((_i+1)*num_of_batches)] = batch_labs.reshape(num_of_batches, 1)
    imgs = imgs if not is_to_gray else np.mean(imgs, axis=-1, keepdims=True)
    print('images shape:', imgs.shape)
    print('labels shape:', labs.shape)
    return imgs/255, labs

def cifar10_load_test_data(dir_path, is_to_gray=False):
    num_of_files = 1
    num_of_batches = 10000
    num_of_train = num_of_files * num_of_batches
    img_w, img_h, img_dim= 32, 32, 3
    imgs = np.zeros(shape=[num_of_train, img_w, img_h, img_dim], dtype=np.float32)
    labs = np.zeros(shape=[num_of_train, 1], dtype=np.int8)
    path = dir_path + '/test_batch'
    batch_imgs, batch_labs = load_batch_data(path)
    imgs[:] = batch_imgs.reshape([-1, img_dim, img_w, img_h]).transpose([0, 2, 3, 1])
    labs[:] = batch_labs.reshape(num_of_batches, 1)
    imgs = imgs if not is_to_gray else np.mean(imgs, axis=-1, keepdims=True)
    print('images shape:', imgs.shape)
    print('labels shape:', labs.shape)
    return imgs/255, labs
