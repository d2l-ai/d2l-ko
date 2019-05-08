# 파일 입/출력

지금까지 우리는 데이터를 처리하고, 딥러닝 모델을 만들고, 학습시키고, 테스트하는 방법들을 알아봤습니다. 실험을 진행하다가 어느 시점에는 얻은 결과가 만족스러워서, 나중에도 활용하고 배포하기 위해서 결과를 저장할 필요가 생깁니다. 또한 긴 학습을 수행할 때 서버의 전원에 문제가 생겼을 때 며칠 동안 수행한 내용을 잃지 않기 위해서 중간 결과들을 저장하는 것이 가장 최선의 방법이기도 합니다. 예를 들면, 영어 단어 임베딩을 가지고, 멋진 스팸 분류기를 만들고자 하는 경우, 미리 학습된 (pretrained) 모델을 읽어야 하는 경우도 있습니다. 이 모든 경우를 수행하기 위해서는,  개별 가중치 벡터들 또는 전체 모델을 저장하고 읽어야 합니다. 이번 절에서는 이 두 가지에 대해서 알아보겠습니다.

## NDArray

가장 간단한 방법은 `save` 와 `load` 함수를 직접 호출해서 NDArray를 하나씩 저장하고 읽을 수 있습니다. 이는 다음과 같이 저장하는 것을 간단히 구현할 수 있습니다.

```{.python .input}
from mxnet import nd
from mxnet.gluon import nn

x = nd.arange(4)
nd.save('x-file', x)
```

그리고,우리는 이 파일을 메모리로 다시 읽습니다.

```{.python .input}
x2 = nd.load('x-file')
x2
```

하나의 NDArray 객체 뿐만 아니라, NDArray들의 리스트도 저장하고 다시 메모리로 읽기도 가능합니다.

```{.python .input  n=2}
y = nd.zeros(4)
nd.save('x-files', [x, y])
x2, y2 = nd.load('x-files')
(x2, y2)
```

문자를 NDArray로 매핑하는 사전(dictionary)를 저장하고 읽는 것도 가능합니다. 이 방법은 모델의 전체 가중치들을 한꺼번에 저장하고 읽을 때 유용합니다.

```{.python .input  n=4}
mydict = {'x': x, 'y': y}
nd.save('mydict', mydict)
mydict2 = nd.load('mydict')
mydict2
```

## Gluon 모델 파라미터들

가중치 벡터를 하나씩 (또는 NDArray 텐서들) 저장하는 것이 유용하지만, 모델 전체를 저장하고 이 후에 읽는데는 매우 불편한 방법입니다. 왜냐하면 모델 전체에 걸쳐서 수백개의 파라미터 그룹들이 있을 수 있기 때문입니다. 만약 모든 값을 모아서 아키텍처에 매핑시키는 스크립트를 작성한다면 매우 많은 일을 해야합니다. 이런 이유로 Gluon은 개별 가중치 벡터를 저장하는 것보다는 전체 네트워크를 저장하고 읽을 수 있는 기능을 제공합니다. 유의해야할 점은 전체 모델을 저장하는 것이 아니라, 모델의 *파라미터들* 을 저장한다는 것입니다. 만약에 3개 층을 갖는 MLP가 있다면, 네트워크의 *아키텍처* 는 별도록 명시해줘야 합니다. 이렇게 한 이유는 모델들 자체는 임의의 코드를 담고 있을 수 있는데 이 경우에는 코드가 쉽게 직렬화(serialization)되지 않을 수 있기 때문입니다. (단, 컴파일된 모델의 경우에는 방법이 있는데, 기술적인 자세한 내용은  [MXNet documentation](http://www.mxnet.io) 을 참고하세요.) 결국, 모델을 다시 만들려면, 아키텍처를 코드 형태로 만들고, 디스크로부터 파라미터를 로딩해야합니다. [지연된 초기화](deferred-init.md) 는 실제 값을 할당할 필요 없이 모델을 정의할 수 있기 때문에 이런 방식에 아주 도움이 됩니다. 역시 우리의 MLP를 사용해서 설명하겠습니다.

```{.python .input  n=6}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
x = nd.random.uniform(shape=(2, 20))
y = net(x)
```

모델 파라미터들을 'mlp.params'라는 이름의 파일에 저장합니다.

```{.python .input}
net.save_parameters('mlp.params')
```

모델을 복원할 수 있는지 확인하기 위해서, 원본 MLP 모델의 복사본을 만듭니다. 모델 파라미터를 난수로 초기화하는 것이 아니라, 파일에 저장했던 파라미터들을 직접 읽습니다.

```{.python .input  n=8}
clone = MLP()
clone.load_parameters('mlp.params')
```

두 모델의 인스턴스가 같은 모델 파라미터를 갖고 있기 때문에, 같은 입력 `x` 를 가지고 계산한 결과는 같아야합니다. 확인해보겠습니다.

```{.python .input}
yclone = clone(x)
yclone == y
```

## 요약

* `save` 와 `load` 함수를 이용해서 NDArray 객체들을 파일에 저장하고 읽을 수 있습니다.
* `load_parameters` 와 `save_parameters` 함수는 Gluon의 네트워크 전체 파라미터를 저장하고 읽는데 사용됩니다.
* 아키텍처를 저장하는 것은 파라미터와는 별도로 코드로 해야합니다.

## 문제

1. 다른 디바이스에 학습된 모델을 배포할 필요가 없을 경우라도, 모델 파라미터를 저장할 수 있을 때 얻을 수 있는 실용적인 이점은 무엇인가요?
1. 네트워크의 일부를 다른 아키텍처의 네트워크에 포함해야 한다고 가정합니다. 예를 들어 이전 네트워크의 처음 두 개층을 새로운 네트워크에서 어떻게 사용할 수 있을까요?
1. 네트워크 아키텍처와 파라미터를 저장하는 방법이 무엇이 있을까요? 네트워크 아키텍처에 어떤 제약을 둬야할까요?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2329)

![](../img/qr_read-write.svg)
