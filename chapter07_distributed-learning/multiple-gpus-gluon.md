#  Training on multiple GPUs with `gluon`

Gluon makes it easy to implement data parallel training.
In this notebook, we'll implement data parallel training for a convolutional neural network.
If you'd like a finer grained view of the concepts, 
you might want to first read the previous notebook,
[multi gpu from scratch](./multiple-gpus-scratch.ipynb) with `gluon`.

To get started, let's first define a simple convolutional neural network and loss function.

```{.python .input  n=1}
from mxnet import gluon, gpu
net = gluon.nn.Sequential(prefix='cnn_')
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=3, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(128, activation="relu"))
    net.add(gluon.nn.Dense(10))
    
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Initialize on multiple devices

Gluon supports initialization of network parameters over multiple devices. We accomplish this by passing in an array of device contexts, instead of the single contexts we've used in earlier notebooks.
When we pass in an array of contexts, the parameters are initialized 
to be identical across all of our devices.

```{.python .input  n=2}
GPU_COUNT = 2 # increase if you have more
ctx = [gpu(i) for i in range(GPU_COUNT)]
net.collect_params().initialize(ctx=ctx)
```

Given a batch of input data,
we can split it into parts (equal to the number of contexts) 
by calling `gluon.utils.split_and_load(batch, ctx)`.
The `split_and_load` function doesn't just split the data,
it also loads each part onto the appropriate device context. 

So now when we call the forward pass on two separate parts,
each one is computed on the appropriate corresponding device and using the version of the parameters stored there.

```{.python .input  n=3}
from mxnet.test_utils import get_mnist
mnist = get_mnist()
batch = mnist['train_data'][0:GPU_COUNT*2, :]
data = gluon.utils.split_and_load(batch, ctx)
print(net(data[0]))
print(net(data[1]))
```

At any time, we can access the version of the parameters stored on each device. 
Recall from the first Chapter that our weights may not actually be initialized
when we call `initialize` because the parameter shapes may not yet be known. 
In these cases, initialization is deferred pending shape inference.

```{.python .input  n=4}
weight = net.collect_params()['cnn_conv0_weight']

for c in ctx:
    print('=== channel 0 of the first conv on {} ==={}'.format(
        c, weight.data(ctx=c)[0]))
    
```

Similarly, we can access the gradients on each of the GPUs. Because each GPU gets a different part of the batch (a different subset of examples), the gradients on each GPU vary.

```{.python .input  n=5}
from mxnet import autograd

def forward_backward(net, data, label):
    with autograd.record():
        losses = [loss(net(X), Y) for X, Y in zip(data, label)]
    for l in losses:
        l.backward()
        
label = gluon.utils.split_and_load(mnist['train_label'][0:4], ctx)
forward_backward(net, data, label)
for c in ctx:
    print('=== grad of channel 0 of the first conv2d on {} ==={}'.format(
        c, weight.grad(ctx=c)[0]))
```

## Put all things together

Now we can implement the remaining functions. Most of them are the same as [when we did everything by hand](./chapter07_distributed-learning/multiple-gpus-scratch.ipynb); one notable difference is that if a `gluon` trainer recognizes multi-devices, it will automatically aggregate the gradients and synchronize the parameters.

```{.python .input  n=6}
from mxnet import nd
from mxnet.io import NDArrayIter
from time import time

def train_batch(batch, ctx, net, trainer):
    # split the data batch and load them on GPUs
    data = gluon.utils.split_and_load(batch.data[0], ctx)
    label = gluon.utils.split_and_load(batch.label[0], ctx)
    # compute gradient
    forward_backward(net, data, label)
    # update parameters
    trainer.step(batch.data[0].shape[0])
    
def valid_batch(batch, ctx, net):
    data = batch.data[0].as_in_context(ctx[0])
    pred = nd.argmax(net(data), axis=1)
    return nd.sum(pred == batch.label[0].as_in_context(ctx[0])).asscalar()    

def run(num_gpus, batch_size, lr):    
    # the list of GPUs will be used
    ctx = [gpu(i) for i in range(num_gpus)]
    print('Running on {}'.format(ctx))
    
    # data iterator
    mnist = get_mnist()
    train_data = NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size)
    valid_data = NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size)
    print('Batch size is {}'.format(batch_size))
    
    net.collect_params().initialize(force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    for epoch in range(5):
        # train
        start = time()
        train_data.reset()
        for batch in train_data:
            train_batch(batch, ctx, net, trainer)
        nd.waitall()  # wait until all computations are finished to benchmark the time
        print('Epoch %d, training time = %.1f sec'%(epoch, time()-start))
        
        # validating
        valid_data.reset()
        correct, num = 0.0, 0.0
        for batch in valid_data:
            correct += valid_batch(batch, ctx, net)
            num += batch.data[0].shape[0]                
        print('         validation accuracy = %.4f'%(correct/num))
        
run(1, 64, .3)        
run(GPU_COUNT, 64*GPU_COUNT, .3*GPU_COUNT)            
```

## Conclusion

Both parameters and trainers in `gluon` support multi-devices. Moving from one device to multi-devices is straightforward. 

## Next
[Distributed training with multiple machines](../chapter07_distributed-learning/training-with-multiple-machines.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
