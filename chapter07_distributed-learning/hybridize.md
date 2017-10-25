# Fast, portable neural networks with Gluon HybridBlocks


The tutorials we saw so far adopt the *imperative*, or define-by-run, programming paradigm. 
It might not even occur to you to give a name to this style of programming 
because it's how we always write Python programs. 

Take for example a prototypical program written below in pseudo-Python.
We grab some input arrays, we compute upon them to produce some intermediate values,
and finally we produce the result that we actually care about.

```{.python .input  n=1}
def add(A, B):
    return A + B

def fancy_func(A, B, C, D):
    E = add(A, B)
    F = add(C, D)
    G = add(E, F)
    return G

fancy_func(1,2,3,4)
```

As you might expect when we compute `E`, we're actually performing some numerical operation, like multiplication, and returning an array that we assign to the variable `E`. Same for `F`. And if we want to do a similar computation many times by putting these lines in a function, each time our program *will have to step through these three lines of Python*. 

The advantage of this approach is it's so natural that it might not even occur to some people that there is another way. But the disadvantage is that it's slow. That's because we are constantly engaging the Python execution environment (which is slow) even though our entire function performs the same three low-level operations in the same sequence every time. It's also holding on to all the intermediate values `D` and `E` until the function returns even though we can see that they're not needed. We might have made this program more efficient by re-using memory from either `E` or `F` to store the result `G`. 


There actually is a different way to do things. It's called *symbolic* programming 
and most of the early deep learning libraries, including Theano and Tensorflow, 
embraced this approach exclusively. 
You might have also heard this approach referred to as *declarative* programming or *define-then-run* programming.
These all mean the exact same thing.
The approach consists of three basic steps:

* Define a computation workflow, like a pass through a neural network, using placeholder data
* Compile the program into a front-end language, e.g. Python, independent format
* Invoke the compiled function, feeding it real data

Revisiting our previous pseudo-Python example, a symbolic version of the same program might look something like this:

```{.python .input  n=2}
def add_str():
    return '''
def add(A, B):
    return A + B
'''

def fancy_func_str():
    return '''
def fancy_func(A, B, C, D):
    E = add(A, B)
    F = add(C, D)
    G = add(E, F)
    return G
'''

def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1,2,3,4))
'''

prog = evoke_str()
y = compile(prog, '', 'exec')
exec(y)
```

Here, when we run the line ``fancy_func_str()``, *no numerical computation actually happens*.
Instead, the symbolic library notes the way that `E` is related to `A` and `B` and records this information.
We don't do actual computation, we just make *a roadmap* for how to go from inputs to outputs.
Because we can draw all of the variables and operations (both inputs and intermediate values) a nodes, and the relationships between nodes with edges, we call the resulting roadmap a computational graph. 
In the symbolic approach, we first define the entire graph, and then compile it.


### Imperative Programs Tend to be More Flexible

When you’re using an imperative-style library from Python, you are writing in Python. Nearly anything that would be intuitive to write in Python, you could accelerate by calling down in the appropriate places to the imperative deep learning library. On the other hand, when you write a symbolic program, you may not have access to all the familiar Python constructs, like iteration. It's also easy to debug an imperative program. For one, because all the intermediate values hang around, it's easy to introspect the program later. Imperative programs are also much easier to debug because we can just stick print statements in between operations.

In short, from a developer's standpoint, imperative programs are just better. 
They're a joy to work with.
You don't have the tricky indirection of working with placeholders.
You can do anything that you can do with native Python.
And faster debugging, means you get to try out more ideas.
But the catch is that imperative programs are *comparatively* slow.

### Symbolic Programs Tend to be More Efficient

The main reason is efficiency, both in terms of memory and speed. Let’s revisit our toy example from before. Consider the following program:

```{.python .input  n=3}
import numpy as np
a = np.ones(10)
b = np.ones(10) * 2
c = b * a
d = c + 1
# ...
```

Assume that each cell in the array occupies 8 bytes of memory. How much memory do we need to execute this program in the Python console? As an imperative program we need to allocate memory at each line. That leaves us allocating 4 arrays of size 10. So we’ll need $4 * 10 * 8 = 320$ bytes. On the other hand, if we built a computation graph, and knew in advance that we only needed d, we could reuse the memory originally allocated for intermediate values. For example, by performing computations in-place, we might recycle the bits allocated for b to store c. And we might recycle the bits allocated for c to store d. In the end we could cut our memory requirement in half, requiring just $2 * 10 * 8$ = 160 bytes.

Symbolic programs can also perform another kind of optimization, called operation folding. Returning to our toy example, the multiplication and addition operations can be folded into one operation. If the computation runs on a GPU processor, one GPU kernel will be executed, instead of two. In fact, this is one way we hand-craft operations in optimized libraries, such as CXXNet and Caffe. Operation folding improves computation efficiency. Note, you can’t perform operation folding in imperative programs, because the intermediate values might be referenced in the future. Operation folding is possible in symbolic programs because we get the entire computation graph in advance, before actually doing any calculation, giving us a clear specification of which values will be needed and which will not.

## Getting the best of both worlds with MXNet Gluon's HybridBlocks

Most libraries deal with the imperative / symbolic design problem by simply choosing a side.
Theano and those frameworks it inspired, like TensorFlow, run with the symbolic way.
And because the first versions of MXNet optimized performance, they also went symbolic.
Chainer and its descendants like PyTorch are fully imperative way.
In designing MXNet Gluon, we asked the following question. 
Is it possible to get *all* of the benefits of imperative programming
but to still exploit, whenever possible, the speed and memory efficiency of symbolic programming.
In other words, a user should be able to use Gluon fully imperatively.
And if they never want their lives to be more complicated then they can get on just fine imagining that the story ends there. 
But when a user needs production-level performance, it should be easy to compile the entire compute graph, 
or at least to compile large subsets of it. 


MXNet accomplishes this through the use of HybridBlocks. Each ``HybridBlock`` can run fully imperatively defining their computation with real functions acting on real inputs. But they're also capable of running symbolically, acting on placeholders. Gluon hides most of this under the hood so you'll only need to know how it works when you want to write your own layers. Given a HybridBlock whose forward computation consists of going through other HybridBlocks, you can compile that section of the network by calling the HybridBlocks ``.hybridize()`` method.

**All of MXNet's predefined layers are HybridBlocks.** This means that any network consisting entirely of predefined MXNet layers can be compiled and run at much faster speeds by calling ``.hybridize()``.




## HybridSequential

We already learned how to use `Sequential` to stack the layers. 
The regular `Sequential` can be built from regular Blocks and so it too has to be a regular Block.
However, when you want to build a network using sequential and run it at crazy speeds,
you can construct your network  using `HybridSequential` instead. 
The functionality is the same `Sequential`:

```{.python .input  n=4}
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd

def get_net():
    # construct a MLP
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dense(128, activation="relu"))
        net.add(nn.Dense(2))
    # initialize the parameters
    net.collect_params().initialize()
    return net

# forward
x = nd.random_normal(shape=(1, 512))
net = get_net()
print('=== net(x) ==={}'.format(net(x)))
```

To compile and optimize the `HybridSequential`, we can then call its `hybridize` method. Only `HybridBlock`s, e.g. `HybridSequential`, can be compiled. But you can still call `hybridize` on normal `Block` and its `HybridBlock` children will be compiled instead. We will talk more about ``HybridBlock``s later.

```{.python .input  n=5}
net.hybridize()
print('=== net(x) ==={}'.format(net(x)))
```

## Performance

To get a sense of the speedup from hybridizing, 
we can compare the performance before and after hybridizing 
by measuring in either case the time it takes to make 1000 forward passes through the network.

```{.python .input  n=6}
from time import time
def bench(net, x):
    mx.nd.waitall()
    start = time()
    for i in range(1000):
        y = net(x)
    mx.nd.waitall()
    return time() - start
        
net = get_net()
print('Before hybridizing: %.4f sec'%(bench(net, x)))
net.hybridize()
print('After hybridizing: %.4f sec'%(bench(net, x)))
```

As you can see, hybridizing gives a significant performance boost, almost 2x the speed.

## Get the symbolic program

Previously, we feed `net` with `NDArray` data `x`, and then `net(x)` returned the forward results. Now if we feed it with a `Symbol` placeholder, then the corresponding symbolic program will be returned.

```{.python .input  n=7}
from mxnet import sym
x = sym.var('data')
print('=== input data holder ===')
print(x)

y = net(x)
print('\n=== the symbolic program of net===')
print(y)

y_json = y.tojson()
print('\n=== the according json definition===')
print(y_json)
```

Now we can save both the program and parameters onto disk, so that it can be loaded later not only in Python, but in all other supported languages, such as C++, R, and Scala, as well.

```{.python .input  n=8}
y.save('model.json')
net.save_params('model.params')
```

## HybridBlock

Now let's dive deeper into how `hybridize` works. Remember that gluon networks are composed of Blocks each of which subclass `gluon.Block`. With normal Blocks, we just need to define a forward function that takes an input `x` and computes the result of the forward pass through the network. MXNet can figure out the backward pass for us automatically with autograd.

To define a `HybridBlock`, we instead have a `hybrid_forward` function:

```{.python .input  n=9}
from mxnet import gluon

class Net(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.fc1 = nn.Dense(256)
            self.fc2 = nn.Dense(128)
            self.fc3 = nn.Dense(2)

    def hybrid_forward(self, F, x):
        # F is a function space that depends on the type of x
        # If x's type is NDArray, then F will be mxnet.nd
        # If x's type is Symbol, then F will be mxnet.sym
        print('type(x): {}, F: {}'.format(
                type(x).__name__, F.__name__))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

The `hybrid_forward` function takes an additional input, `F`, which stands for a backend. 
This exploits one awesome feature of MXNet. 
MXNet has both a symbolic API (``mxnet.symbol``) and an imperative API (``mxnet.ndarray``).
In this book, so far, we've only focused on the latter.
Owing to fortuitous historical reasons, the imperative and symbolic interfaces both support roughly the same API.
They have many of same functions (currently about 90% overlap) and when they do, they support the same arguments in the same order.
When we define ``hybrid_forward``, we pass in `F`. 
When running in imperative mode, ``hybrid_forward`` is called with `F` as `mxnet.ndarray` and `x` as some ndarray input.
When we compile with ``hybridize``, `F` will be `mxnet.symbol` and `x` will be some placeholder or intermediate symbolic value. Once we call hybridize, the net is compiled, so we'll never need to call ``hybrid_forward`` again.

Let's demonstrate how this all works by feeding some data through the network twice. We'll do this for both a regular network and a hybridized net. You'll see that in the first case, ``hybrid_forward`` is actually called twice.

```{.python .input  n=10}
net = Net()
net.collect_params().initialize()
x = nd.random_normal(shape=(1, 512))
print('=== 1st forward ===')
y = net(x)
print('=== 2nd forward ===')
y = net(x)
```

Now run it again after hybridizing.

```{.python .input  n=11}
net.hybridize()
print('=== 1st forward ===')
y = net(x)
print('=== 2nd forward ===')
y = net(x)
```

It differs from the previous execution in two aspects:

1. the input data type now is `Symbol` even when we fed an `NDArray` into `net`, because `gluon` implicitly constructed a symbolic data placeholder.
2. `hybrid_forward` is called once at the first time we run `net(x)`. It is because `gluon` will construct the symbolic program on the first forward, and then keep it for reuse later.

One main reason that the network is faster after hybridizing is because we don't need to repeatedly invoke the Python forward function, while keeping all computations within the highly efficient C++ backend engine.

But the potential drawback is the loss of flexibility to write the forward function. In other ways, inserting `print` for debugging or control logic such as `if` and `for` into the forward function is not possible now.

## Conclusion

Through `HybridSequental` and `HybridBlock`, we can convert an imperative program into a symbolic program by calling `hybridize`. 

## Next
[Training MXNet models with multiple GPUs](../chapter07_distributed-learning/multiple-gpus-scratch.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
