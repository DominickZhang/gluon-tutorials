# Multilayer perceptrons in ``gluon``

Using gluon, we only need two additional lines of code to transform our logistic regression model into a multilayer perceptron.

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
from mxnet import gluon
import sys
sys.path.append('..')
import utils
```

We'll also want to set the compute context for our modeling. Feel free to go ahead and change this to mx.gpu(0) if you're running on an appropriately endowed machine.

```{.python .input  n=2}
ctx = mx.cpu()
```

## The MNIST dataset

```{.python .input  n=3}
num_outputs = 10
batch_size = 64
train_data, test_data = utils.load_mnist(batch_size)
```

## Define the model

*Here's the only real difference. We add two lines!*

```{.python .input  n=4}
num_hidden = 256
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))
```

## Parameter initialization

```{.python .input  n=5}
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

## Softmax cross-entropy loss

```{.python .input  n=6}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Optimizer

```{.python .input  n=7}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
```

## Evaluation metric

```{.python .input  n=8}
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]
```

## Training loop

```{.python .input  n=9}
epochs = 10
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
            loss.backward()
        trainer.step(data.shape[0])

        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) 
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, moving_loss, train_accuracy, test_accuracy))
```

## Conclusion

We showed the much simpler way to define a multilayer perceptrons in ``gluon``. Now let's take a look at how to build convolutional neural networks.

## Next
[Dropout regularization from scratch](../chapter03_deep-neural-networks/mlp-dropout-scratch.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
