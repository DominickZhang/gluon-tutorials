# Multiclass logistic regression with ``gluon``

Now that we've built a [logistic regression model from scratch](./softmax-regression-scratch.ipynb), let's make this more efficient with ``gluon``. If you completed the corresponding chapters on linear regression, you might be tempted rest your eyes a little in this one. We'll be using ``gluon`` in a rather similar way and since the interface is reasonably well designed, you won't have to do much work. To keep you awake we'll introduce a few subtle tricks.

Let's start by importing the standard packages.

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
import sys
sys.path.append('..')
import utils
```

## Set the context

Now, let's set the context. In the linear regression tutorial we did all of our computation on the cpu (`mx.cpu()`) just to keep things simple. When you've got 2-dimensional data and scalar labels, a smartwatch can probably handle the job. Already, in this tutorial we'll be working with a considerably larger dataset. If you happen to be running this code on a server with a GPU and installed the GPU-enabled version of MXNet (or remembered to build MXNet with ``CUDA=1``), you might want to substitute the following line for its commented-out counterpart.

```{.python .input  n=2}
# Set the context to CPU
ctx = mx.cpu()

# To set the context to GPU use this
# ctx = mx.gpu()
```

## The MNIST Dataset

We won't suck up too much wind describing the MNIST dataset for a second time. If you're unfamiliar with the dataset and are reading these chapters out of sequence, take a look at the data section in the previous chapter on [softmax regression from scratch](./P02-C03-softmax-regression-scratch.ipynb).


We'll load up data iterators corresponding to the training and test splits of MNIST dataset.

```{.python .input  n=3}
batch_size = 64
num_inputs = 784
num_outputs = 10
train_data, test_data = utils.load_mnist(batch_size)
```

We're also going to want to load up an iterator with *test* data. After we train on the training dataset we're going to want to test our model on the test data. Otherwise, for all we know, our model could be doing something stupid (or treacherous?) like memorizing the training examples and regurgitating the labels on command.

## Multiclass Logistic Regression

Now we're going to define our model.
Remember from [our tutorial on linear regression with ``gluon``](./P02-C02-linear-regression-gluon)
that we add ``Dense`` layers by calling ``net.add(gluon.nn.Dense(num_outputs))``.
This leaves the parameter shapes under-specified,
but ``gluon`` will infer the desired shapes
the first time we pass real data through the network.

```{.python .input  n=4}
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_outputs))
```

## Parameter initialization

As before, we're going to register an initializer for our parameters. Remember that ``gluon`` doesn't even know what shape the parameters have because we never specified the input dimension. The parameters will get initialized during the first call to the forward method.

```{.python .input  n=5}
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=ctx)
```

## Softmax Cross Entropy Loss

Note, we didn't have to include the softmax layer because MXNet's has an efficient function that simultaneously computes the softmax activation and cross-entropy loss. However, if ever need to get the output probabilities,

```{.python .input  n=6}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Optimizer

And let's instantiate an optimizer to make our updates

```{.python .input  n=7}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## Evaluation Metric

This time, let's simplify the evaluation code by relying on MXNet's built-in ``metric`` package.

```{.python .input  n=8}
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]
```

Because we initialized our model randomly, and because roughly one tenth of all examples belong to each of the ten classes, we should have an accuracy in the ball park of .10.

```{.python .input  n=9}
evaluate_accuracy(test_data, net)
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "0.064803685897435903"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Execute training loop

```{.python .input  n=10}
epochs = 4
moving_loss = 0.
smoothing_constant = .01
niter = 0

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        niter +=1
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * nd.mean(loss).asscalar()
        est_loss = moving_loss/(1-(1-smoothing_constant)**niter)

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, est_loss, train_accuracy, test_accuracy))

```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 1.03587294828, Train_acc 0.7957410619, Test_acc 0.798177083333\nEpoch 1. Loss: 0.764642039905, Train_acc 0.838880736393, Test_acc 0.841145833333\nEpoch 2. Loss: 0.674115589454, Train_acc 0.856556830309, Test_acc 0.858373397436\n"
 },
 {
  "ename": "KeyboardInterrupt",
  "evalue": "",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-10-9ae3cf6dc426>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     18\u001b[0m         \u001b[0;31m##########################\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     19\u001b[0m         \u001b[0mniter\u001b[0m \u001b[0;34m+=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 20\u001b[0;31m         \u001b[0mmoving_loss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m \u001b[0;34m-\u001b[0m \u001b[0msmoothing_constant\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mmoving_loss\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0msmoothing_constant\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mnd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mloss\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masscalar\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     21\u001b[0m         \u001b[0mest_loss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmoving_loss\u001b[0m\u001b[0;34m/\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0msmoothing_constant\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m**\u001b[0m\u001b[0mniter\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     22\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/miniconda3/envs/gluon_zh_docs/lib/python3.6/site-packages/mxnet/ndarray/ndarray.py\u001b[0m in \u001b[0;36masscalar\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   1496\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1497\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"The current array is not a scalar\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1498\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masnumpy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1499\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1500\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mastype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/miniconda3/envs/gluon_zh_docs/lib/python3.6/site-packages/mxnet/ndarray/ndarray.py\u001b[0m in \u001b[0;36masnumpy\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   1478\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhandle\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1479\u001b[0m             \u001b[0mdata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mctypes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata_as\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mctypes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mc_void_p\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1480\u001b[0;31m             ctypes.c_size_t(data.size)))\n\u001b[0m\u001b[1;32m   1481\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1482\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
  ]
 }
]
```

```{.python .input  n=31}
import matplotlib.pyplot as plt

def model_predict(net,data):
    output = net(data)
    return nd.argmax(output, axis=1)

# let's sample 10 random data points from the test set
for i, (data, _) in enumerate(test_data):
    data = data[0:10]
    data = data.as_in_context(ctx)
    im = nd.transpose(data,(1,2,0,3))
    im = nd.reshape(im,(28,-1,1))
    im = nd.tile(im, (1,1,3))
    plt.imshow(im.asnumpy())
    plt.show()
    pred=model_predict(net,data.reshape((-1,784)))
    print('model predictions are:', pred)
    break
```

```{.json .output n=31}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(10, 1, 28, 28)\n"
 },
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7fedc3cf0828>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "model predictions are: \n[ 7.  2.  1.  0.  4.  1.  4.  9.  6.  9.]\n<NDArray 10 @cpu(0)>\n"
 }
]
```

## Next
[Overfitting and regularization from scratch](../chapter02_supervised-learning/regularization-scratch.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
