import mxnet as mx
from mxnet import gluon
from mxnet import nd
import numpy as np

class DataLoader(object):
    """similiar to gluon.data.DataLoader, but faster"""
    def __init__(self, X, y, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle
        if not isinstance(y, nd.NDArray):
            y = nd.array(y)
        self.X = X
        self.y = y

    def __iter__(self):
        n = self.X.shape[0]
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            self.X = nd.array(self.X.asnumpy()[idx])
            self.y = nd.array(self.y.asnumpy()[idx])

        for i in range(n//self.batch_size):
            yield (self.X[i*self.batch_size:(i+1)*self.batch_size],
                   self.y[i*self.batch_size:(i+1)*self.batch_size])

    def __len__(self):
        return self.X.shape[0]//self.batch_size

def load_mnist(batch_size, resize=None):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        if resize:
            # resize to resize x resize
            n = data.shape[0]
            new_data = nd.zeros((n, resize, resize, data.shape[3]))
            for i in range(n):
                new_data[i] = image.imresize(data[i], resize, resize)
            data = new_data
        # change data from batch x height x weight x channel to batch x channel x height x weight
        return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')
    mnist_train = gluon.data.vision.MNIST(
        train=True, transform=transform_mnist)[:]
    mnist_test = gluon.data.vision.MNIST(
        train=False, transform=transform_mnist)[:]
    train_data = DataLoader(mnist_train[0], nd.array(mnist_train[1]), batch_size, shuffle=True)
    test_data = DataLoader(mnist_test[0], nd.array(mnist_test[1]), batch_size, shuffle=False)
    return (train_data, test_data)

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
