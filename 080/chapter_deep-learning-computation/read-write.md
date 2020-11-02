# File I/O
# 파일 I/O

0.15.0

So far we discussed how to process data and how
to build, train, and test deep learning models.
However, at some point, we will hopefully be happy enough
with the learned models that we will want
to save the results for later use in various contexts
(perhaps even to make predictions in deployment).
Additionally, when running a long training process,
the best practice is to periodically save intermediate results (checkpointing)
to ensure that we do not lose several days worth of computation
if we trip over the power cord of our server.
Thus it is time to learn how to load and store
both individual weight vectors and entire models.
This section addresses both issues.

지금까지 우리는 데이터를 어떻게 처리하고 어떻게 딥러닝 모델을 만들고, 학습 시키고, 테스트를 하는지를 논의했습니다. 하지만, 어느 시점에서 우리는 학습된 모델의 성능에 만족하게 되면, 다양한 상황(아마도 모델을 배포한 후 예측 수행하기)에서 모델을 사용하기 위해서 결과를 저장해야 합니다. 만약 오래 수행되는 학습의 경우에는 중간의 결과들(체크보인트)를 주기적으로 저장하는 것이 모범 사례입니다. 학습이 수행되는 중에 서버의 전원선에 걸려 넘어져서 몇 일 동안의 계산이 날라가는 것을 막아야하니까요. 따라서, 이제는 각 가중치 벡터들과 전체 모델을 저장하고 로드하는 방법을 살펴볼 차례입니다.

## Loading and Saving Tensors
## 텐서를 읽어오고 저장하기

For individual tensors, we can directly
invoke the `load` and `save` functions
to read and write them respectively.
Both functions require that we supply a name,
and `save` requires as input the variable to be saved.

각 텐서들에 대해서  `load` 와 `save`  함수를 사용해 읽거나 쓸기를 수행할 수 있습니다. 두 함수는 파일 이름이 필요하고,  `save` 함수는 저장할 변수가 입력으로 필요합니다.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import numpy as np

x = tf.range(4)
np.save("x-file.npy", x)
```

We can now read the data from the stored file back into memory.

이제 우리는 저장된 파일에서 데이터를 읽어서 메모리 가져올 수 있습니다.

```{.python .input}
x2 = npx.load('x-file')
x2
```

```{.python .input}
#@tab pytorch
x2 = torch.load("x-file")
x2
```

```{.python .input}
#@tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

We can store a list of tensors and read them back into memory.

또한 텐서들의 리스트를 저장하고, 메모리로 다시 읽어올 수 있습니다.

```{.python .input}
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

We can even write and read a dictionary that maps
from strings to tensors.
This is convenient when we want
to read or write all the weights in a model.

문자열과 텐서를 매핑하는 사전을 저장하고 읽는 것도 가능합니다. 이 방법은 모델의 모든 가중치를 저장하고 읽어올 때 유용합니다.

```{.python .input}
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
#@tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
#@tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

## Loading and Saving Model Parameters
## 모델 파라미터 읽어오기와 저장하기

Saving individual weight vectors (or other tensors) is useful,
but it gets very tedious if we want to save
(and later load) an entire model.
After all, we might have hundreds of
parameter groups sprinkled throughout.
For this reason the deep learning framework provides built-in functionalities
to load and save entire networks.
An important detail to note is that this
saves model *parameters* and not the entire model.
For example, if we have a 3-layer MLP,
we need to specify the architecture separately.
The reason for this is that the models themselves can contain arbitrary code,
hence they cannot be serialized as naturally.
Thus, in order to reinstate a model, we need
to generate the architecture in code
and then load the parameters from disk.
Let us start with our familiar MLP.

가중치 벡터 또는 다른 텐서들을 따로 저장하는 것이 유용하지만, 전체 모델을 저장하고 이 후에 읽어오기에는 매우 번거롭습니다. 결국에 우리는 모델 전체에 걸친 수백 개의 파라미터 그룹들을 갖을 수도 있습니다. 이런 이유로 딥러닝 프레임워크는 전체 모델을 읽어오고 저장하는 빌트인 함수를 제공합니다. 여기서 중요하게 알아두어야 할 점은 이 함수는 전체 모델을 저장하는 것이 아니라 모델 *파라미터들* 을 저장한다는 것입니다. 예를 들어, 3 층으로 구성된 MLP를 만들었다면, 모델 아키텍처는 별도로 지정해야 합니다. 이렇게 하는 이유는 모델은 임의 코드를 갖을 수 있고 따라서, 코드를 직렬화 할 수 없기 때문입니다. 즉, 모델을 다시 생성하기 위해서는 아키텍처를 코드로서 생성하고, 디스크에서 파라미터들을 읽어와야 합니다. 그럼 우리가 친숙한 MLP를 사용해서 알아보겠습니다.

```{.python .input}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

Next, we store the parameters of the model as a file with the name "mlp.params".

다음으로 우리는 모델의 파라미터들을 "mlp.params"라는 이름의 파일에 저장합니다.

```{.python .input}
net.save_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
#@tab tensorflow
net.save_weights('mlp.params')
```

To recover the model, we instantiate a clone
of the original MLP model.
Instead of randomly initializing the model parameters,
we read the parameters stored in the file directly.

모델을 복구하기 위해서 우리는 원래의 MLP 모델의 복사본 인스턴스를 만듭니다. 모델 파라미터들을 임의의 수로 초기화하는 대신, 우리는 파일에 저장되어 있는 파라미터들을 직접 읽어옵니다.

```{.python .input}
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
clone = MLP()
clone.load_state_dict(torch.load("mlp.params"))
clone.eval()
```

```{.python .input}
#@tab tensorflow
clone = MLP()
clone.load_weights("mlp.params")
```

Since both instances have the same model parameters,
the computational result of the same input `X` should be the same.
Let us verify this.

두 모델 인스턴스들이 같은 모델 파라미터들을 갖고 있기 때문에, 같은 입력  `X`  에 대한 계산 결과는 일치합니다. 확인해 봅시다.

```{.python .input}
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab pytorch
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab tensorflow
Y_clone = clone(X)
Y_clone == Y
```

## Summary
## 요약

* The `save` and `load` functions can be used to perform file I/O for tensor objects.
* We can save and load the entire sets of parameters for a network via a parameter dictionary.
* Saving the architecture has to be done in code rather than in parameters.



-  `save` 와 `load` 함수는 텐서 객체들에 대한 파일 I/O를 수행하는데 사용됩니다.
- 파라미터 사전을 통해서 네트워크의 파라미터 전체 셋을 저장하고 읽어올 수 있습니다.
- 아키텍처 저장은 파라미터 형태가 아니라 코드 형도로 저장되야 합니다.

## Exercises



1. Even if there is no need to deploy trained models to a different device, what are the practical benefits of storing model parameters?
1. Assume that we want to reuse only parts of a network to be incorporated into a network of a different architecture. How would you go about using, say the first two layers from a previous network in a new network?
1. How would you go about saving the network architecture and parameters? What restrictions would you impose on the architecture?



1. 학습된 모델을 다른 장치에 배포하지 않을 지라도 모델 파라미터를 저장하는 것은 어떤 실용적인 이점이 있나요?
2. 네트워크의 일부만 다른 아키텍처의 네트워크로 재사용하고 싶다고 가정합니다. 예를 들어 이전 네트워크는 처음 두 개의 층만 새로운 네트워크에서 사용하고자 한다면, 어떻게 하겠습니까?
3. 네트워크 아키텍처와 파라미터를 어떻게 저장하겠습니까? 아키텍처에 어떤 제약이 생길까요?



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/327)
:end_tab:
