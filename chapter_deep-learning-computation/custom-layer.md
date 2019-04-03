# 커스텀 레이어Layers

딥러닝의 성공 요인 중에 하나는 딥 네트워크에서 사용할 수 있는 다양한 종류의 레이어가 있다는 점에서 찾아볼 수 있습니다. 즉, 다양한 형태의 레이어를 사용해서 많은 종류의 커스터마이징와 다양한 문제에 적용이 가능하게 되었습니다. 예를 들면, 과학자들이 이미지, 텍스트, 풀링, loop, 동적 프로그램밍, 그리고 심지어는 컴퓨터 프로그램을 위한 레이어를 발명해왔습니다. 앞으로도 Gluon에 현재 존재하지 않은 새로운 레이어를 만나게될 것이고, 어쩌면 여러분이 만난 문제를 해결하기 위해서 새로운 레이어를 직접 발명 할지도 모릅니다. 자 그럼 커스텀 레이어를 만들어 보는 것을 이 절에서 배워보겠습니다.

## 파라메터가 없는 레이어들

커스텀 레이어를 만드는 것은 다소 복잡할 수 있기 때문에, 파라메터를 계승 받지 않는 커스텀 레이어 (또는 Block)를 만드는 것부터 시작해보겠습니다. 첫번째 시작은 이전에 [introduced blocks](model-construction.md) 에서 소개했던 것과 비슷합니다. 아래 `CenteredLayer` 클래스는 입력에서 평균을 빼는 것을 계산하는 레이어를 정의합니다. 우리는 이것을 Block 클래스를 상속하고, `forward` 메소드를 구현해서 만듭니다.

```{.python .input  n=1}
from mxnet import gluon, nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()
```

어떻게 동작하는지 보기 위해서, 데이터를 레이어에 입력해봅니다.

```{.python .input  n=2}
layer = CenteredLayer()
layer(nd.array([1, 2, 3, 4, 5]))
```

우리는 이를 사용해서 더 복잡한 모델을 만들 수도 있습니다.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

그럼 이 가운데로 만들어주는 레이어가 잘 작동하는지 보겠습니다. 이를 위해서 난수 데이터를 생성하고, 네트워크에 입력한 후 평균만큼 값이 조정되는지 확입합니다. 우리가 다루는 변수가 실수형이기 때문에, 아죽 작지만 0이 아닌 숫자를 보게될 것임을 염두하세요.

```{.python .input  n=4}
y = net(nd.random.uniform(shape=(4, 8)))
y.mean().asscalar()
```

## 파라메터가 있는 레이어들

레이어를 어떻게 정의하는지 원리를 알게 되었으니, 파라메터를 갖는 레이어를 정의해보겠습니다. 이 파라메터들은 학습을 통해서 조정될 값들입니다. 딥러닝 연구자들의 일을 편하게 만들어 주기 위해서, `Parameter`  클래스와 `ParameterDict` dictionary는 많이 사용하는 기능을 제공하고 있습니다. 이 클래스들은 접근을 관리하고, 초기화를 하고, 공유를 하고, 모델 파라메터를 저장하고 로딩하는 기능을 관리해줍니다. 예를 들면, 새로운 커스텀 레이어를 만든 때 매번 직렬화(serialization) 루틴을 작성할 필요가 없습니다. 

다른 예로는, Block 클래스와 함께 제공되는  `ParameterDict` 타입인 `params` 를 사용할 수도 있습니다. 이 dictionary는 문자 타입의 파라메터 이름을 `Parameter` 타입의 모델 파라메터로 매핑하는 기능을 제공합니다.  `ParameterDict` 의 `get` 함수를 사용해서 `Parameter` 인스턴스를 생성하는 것도 가능합니다.

```{.python .input  n=7}
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
params
```

Dense 레이어를 직접 구현해보겠습니다. 이 레이어는 두 파라메터, weight와 bais, 를 갖습니다. 약간 특별하게 만들기 위해서, ReLU activation 함수를 기본으로 적용하도록 만들어봅니다. weight와 bias 파라메터를 갖는 fully connected 레이어를 구현하고, ReLU를 activation 함수로 추가합니다. `in_units`와 `units` 는 각각 입력과 출력의 개수입니다.

```{.python .input  n=19}
class MyDense(nn.Block):
    # units: the number of outputs in this layer; in_units: the number of
    # inputs in this layer
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
```

파라메터에 이름을 부여하는 것은 이후에 dictionary 조회를 통해서 원하는 파라메터를 직접 접근할 수 있도록 해줍니다. 그렇기 때문에, 잘 설명하는 이름을 정하는 것이 좋은 생각입니다. 자 이제 `MyDense` 클래스의 인스턴스를 만들고 모델 파라메터들을 직접 확인해봅니다.

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

커스텀 레이어의 forward 연산을 수행합니다.

```{.python .input  n=20}
dense.initialize()
dense(nd.random.uniform(shape=(2, 5)))
```

커스텀 레이어를 이용해서 모델은 만들어 보겠습니다. 만들어진 모델은 기본으로 제공되는 dense 레이어처럼 사용할 수 있습니다. 하나 다른 점은 입력, 출력의 크기를 자동으로 계산하는 것이 없다는 점입니다. 어떻게 이 기능을 구현할 수 있는지는 [MXNet documentation](http://www.mxnet.io) 를 참고하세요.

```{.python .input  n=19}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(nd.random.uniform(shape=(2, 64)))
```

## 요약

* Block 클래스를 이용해서 커스텀 레이어를 만들 수 있습니다. 이 방법은 블럭 팩토리를 정의하는 것보다 더 강력한 방법인데, 그 이유는 다양한 context들에서 불려질 수 있기 때문입니다.
* 블럭들은 로컬 파라매터를 갖을 수 있습니다.


## 문제

1. 데이터에 대해서 affine 변환을 학습하는 레이어를 디자인하세요. 예를 들면, 평균 값을 빼고, 대신 더할 파라메터를 학습합니다.
1. 입력을 받아서 텐서 축소를 하는 레이어를 만들어 보세요. 즉, $y_k = \sum_{i,j} W_{ijk} x_i x_j​$ 를 반환합니다.
1. 데이터에 대한 퓨리에 계수의 앞에서 반을 리턴하는 레이어를 만들어보세요. 힌트 - MXNet의 `fft` 함수를 참고하세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2328)

![](../img/qr_custom-layer.svg)
