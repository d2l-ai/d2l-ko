# 파일 입출력

지금까지 데이터를 처리하는 방법과 딥 러닝 모델을 구축, 학습 및 테스트하는 방법에 대해 논의했습니다.그러나 어느 시점에서는 나중에 다양한 컨텍스트에서 사용할 수 있도록 결과를 저장하려는 학습 된 모델에 충분히 만족할 것입니다 (아마도 배포 중에 예측을 수행하기 위해).또한 긴 교육 프로세스를 실행할 때 가장 좋은 방법은 서버의 전원 코드를 넘어갈 때 며칠 분량의 계산을 잃지 않도록 중간 결과 (체크 포인트) 를 주기적으로 저장하는 것입니다.따라서 개별 가중치 벡터와 전체 모델을 로드하고 저장하는 방법을 배워야 할 때입니다.이 섹션에서는 두 가지 문제를 모두 다룹니다.

## (**텐서 로드 및 저장**)

개별 텐서의 경우 `load` 및 `save` 함수를 직접 호출하여 각각 읽고 쓸 수 있습니다.두 함수 모두 이름을 제공해야 하며 `save`는 저장할 변수를 입력으로 요구합니다.

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
np.save('x-file.npy', x)
```

이제 저장된 파일의 데이터를 메모리로 다시 읽을 수 있습니다.

```{.python .input}
x2 = npx.load('x-file')
x2
```

```{.python .input}
#@tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
#@tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

[**텐서 목록을 저장하고 메모리로 다시 읽을 수 있습니다.**]

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

[**문자열에서 텐서로 매핑되는 딕셔너리를 작성하고 읽기**] 도 가능합니다. 모델의 모든 가중치를 읽거나 쓰고 싶을 때 편리합니다.

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

## [**모델 파라미터 불러오기 및 저장**]

개별 가중치 벡터 (또는 다른 텐서) 를 저장하는 것이 유용하지만 전체 모델을 저장 (나중에 로드) 하려는 경우 매우 지루합니다.결국 수백 개의 매개 변수 그룹이 전체적으로 뿌려질 수 있습니다.이러한 이유로 딥러닝 프레임워크는 전체 네트워크를 로드하고 저장하는 기본 제공 기능을 제공합니다.주목해야 할 중요한 세부 사항은 전체 모델이 아니라 모델*매개변수*를 저장한다는 것입니다.예를 들어 3계층 MLP가 있는 경우 아키텍처를 별도로 지정해야 합니다.그 이유는 모델 자체가 임의의 코드를 포함할 수 있기 때문에 자연스럽게 직렬화할 수 없기 때문입니다.따라서 모델을 복원하려면 코드에서 아키텍처를 생성한 다음 디스크에서 매개 변수를 로드해야 합니다.(**익숙한 MLP부터 시작하겠습니다. **)

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

다음으로, 이름이 “mlp.params”인 [**모델의 매개 변수를 파일로 저장**] 합니다.

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

모델을 복구하기 위해 원래 MLP 모델의 복제본을 인스턴스화합니다.모델 매개 변수를 무작위로 초기화하는 대신 [**파일에 저장된 매개 변수를 직접 읽습니다**].

```{.python .input}
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
#@tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

두 인스턴스 모두 동일한 모델 매개변수를 가지므로 동일한 입력 `X`의 계산 결과는 동일해야 합니다.이것을 확인해 보겠습니다.

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

## 요약

* `save` 및 `load` 함수는 텐서 객체에 대한 파일 I/O를 수행하는 데 사용할 수 있습니다.
* 파라미터 딕셔너리를 통해 네트워크에 대한 전체 파라미터 세트를 저장하고 로드할 수 있습니다.
* 아키텍처 저장은 매개 변수가 아닌 코드로 수행해야 합니다.

## 연습문제

1. 학습된 모델을 다른 기기에 배포할 필요가 없더라도 모델 파라미터를 저장하면 실질적인 이점은 무엇입니까?
1. 네트워크의 일부만 재사용하여 다른 아키텍처의 네트워크에 통합하려고 한다고 가정합니다.새 네트워크에서 이전 네트워크의 처음 두 계층을 어떻게 사용하시겠습니까?
1. 네트워크 아키텍처와 파라미터를 어떻게 저장하시겠습니까?아키텍처에 어떤 제한을 두겠는가?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/327)
:end_tab:
