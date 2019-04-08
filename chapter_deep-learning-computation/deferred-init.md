# 초기화 지연(deferred Initialization)

앞의 예제들에서는 네트워크들을 빠르게 그리고 조금은 느슨하게 만들어왔습니다. 특히 다음과 같이 동작하지 않을 것처럼 보일 수 있는 것들을 했습니다.

* 입력의 차원을 고려하지 않고 네트워크 아키텍처를 정의했습니다.
* 이전 층(layer)의 출력 차원을 고려하지 않고 다음 층에 추가했습니다.
* 얼마나 많은 파라미터들이 있을지 모르는 상태에서 이 파라미터들을 초기화까지 했습니다.

이 모든 것이 불가능하게 들리고, 실제로도 불가능합니다. 사실 MXNet 이나 다른 프레임워크들이 네트워크에 들어올 입력값의 차원을 예측할 수 있는 방법은 없습니다. 이후에 살펴 볼, 컨볼루션(convolutional) 네트워크나 이미지를 다룰 때 이 문제는 더욱 그렇게 보일 것입니다. 그 이유는 이미지의 해상도 같은 입력의 차원은 네트워크의 연속된 층들의 차원에 영향을 미치기 때문입니다. 따라서,  코드를 작성할 때, 차원이 무엇인지 미리 알 필요없이 파라미터를 설정할 수 있는 능력은 통계적인 모델링을 아주 간단하게 해줄 수 있습니다. 지금부터, 초기화를 예로 어떻게 동작하는지 살펴보겠습니다. 결국에는 존재하는지 모르는 변수를 초기화하는 것은 불가능합니다.

## 네트워크 생성하기

네트워크에 대한 인스턴스를 만들면 일어나는 일을 살펴보겠습니다. 앞에서와 같이 MLP를 사용합니다.

```{.python .input}
from mxnet import init, nd
from mxnet.gluon import nn

def getnet():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = getnet()
```

이 시점에서 네트워크는 여러 파라미터들의 차원이 어떻게 되는지를 알 수 있는 방법이 없습니다. 이 상태에서 말할 수 있는 사실은 각 층의 차원이 무엇이 되던 가중치들과 편향들이 필요하다는 것입니다. 파라미터들 읽어보면 이것을 알 수 있습니다.

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

`net[0].weight.data(0)` 을 수행하면 무언가를 하기 위해서는 네트워크가 초기화 되어야한다는 런타임 에러를 만나게 됩니다.  파라미터를 초기화한 후, 무엇인 바뀌는지를 확인해보겠습니다. 

```{.python .input}
net.initialize()
net.collect_params()
```

결과에서 볼 수 있듯이, 아무것도 바뀐 것이 없습니다. 네트워크에 데이터를 입력하는 경우에 비로서 변화가 생기게 됩니다. 한번 해보겠습니다.

```{.python .input}
x = nd.random.uniform(shape=(2, 20))
net(x)  # Forward computation

net.collect_params()
```

이전에 대비해서 주요 차이점은 입력에 대한 차원, $\mathbf{x} \in \mathbb{R}^{20}$ 을 알게 되면, 첫번째 층의 가중치 행렬을 정의할 수 있다는 것입니다. 이렇게 되면, 우리는 두번째 층에 대한 차원을  $10 \times 256$ 으로 결정할 수 있게 됩니다. 계속 연산 그래프를 따라서 차원이 결정되게 됩니다. 이것이 끝나면, 파라미터를 초기화하는 것을 진행할 수 있습니다. 이 방법이 위 세가지 문제의 해결책입니다.

## 초기화 지연의 적용

이론적으로 어떻게 동작하는지를 배웠으니, 초기화가 실제로 언제 일어나는지 보겠습니다. 이를 확인하기 위해서 어떤 파라미터를 초기화할지를 지정하면서 호출되면 아무것도 하지 않지만 디버그 메시지를 출력하는 초기화 클래스를 하나 정의합니다.

```{.python .input  n=22}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # The actual initialization logic is omitted here

net = getnet()
net.initialize(init=MyInit())
```

`MyInit` 은 호출되면 모델 파라미터에 대한 정보를 출력하게 만들어졌는데, `initialize` 함수를 호출해도 아무런 정보가 출력되지 않고 있습니다. 즉, `initialize` 함수가 호출이 되어도 실제 파라미터 초기화가 일어나지 않습니다. 이제 입력값을 정의하고 forward 연산을 수행해봅니다.

```{.python .input  n=25}
x = nd.random.uniform(shape=(2, 20))
y = net(x)
```

이제야 모델 파라미터들에 대한 정보가 화면에 출력됩니다. 주어진 입력 `x`에 대한 forward 연산을 수행할 때, 시스템은 입력의 모양(shape)을 기반으로 모든 층의 가중치 파라미터의 모양(shapr)을 추론해냅니다. 시스템이 이 파라미터들을 생성하고 나면, `MyInit` 인스턴스를 호출해서 파라미터들을 초기화한 후, forward 연산을 수행하게 됩니다.

물론 이 초기화는 최초 forward 연산을 수행할 때만 일어납니다. 즉, 이후에 `net(x)` 을 호출해서 forward 연산이 수행되면 재초기화가 수행되지 않고, 따라서 `MyInit` 인스턴스의 결과는 다시 출력되지 않습니다.

```{.python .input}
y = net(x)
```

이절을 시작하면서 언급했듯이, 지연된 초기화는 혼동을 가져올 수도 있습니다. 예를 들면, 첫번째 forward 연산 전에는, 모델 파라미터를 직접 바꾸는 것이 불가능합니다. 즉, `data` 나 `set_data` 함수를 호출해서 모델 파라미터의 값을 얻거나 바꾸는 것이 불가능합니다. 따라서, 필요한 경우에는 샘플 입력을 이용해서 네트워크를 강제 초기화하기도 합니다.

## 강제 초기화

 `initialize` 함수가 수행되는 시점에 시스템에 모든 파라미터의 모양(shape)을 하는 경우에는 지연된 초기화가 일어나지 않습니다. 아래 두 가지가 그런 경우입니다.

* 이미 어떤 데이터를 봤고, 파라미터를 재설정하고 싶은 경우
* 네트워크를 정의할 때, 모든 입력과 출력의 차원을 알고 있는 경우

첫번째 경우는 아래 예제 코드 처럼 간단히 할 수 있습니다.

```{.python .input}
net.initialize(init=MyInit(), force_reinit=True)
```

두번째 경우는 레이어를 생성할 때 파라미터에 대한 정보를 명시 해줘야합니다. 예를 들어, 덴스층(dense layer)의 경우에는 `in_units` 에 대한 값을 명시하면 `initialize` 가 호출되면 파라미터가 바로 초기화됩니다.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))

net.initialize(init=MyInit())
```

## 요약

* 지연된 초기화는 좋습니다. 지연된 초기화는 Gluon이 많은 것들을 자동으로 설정할 수 있게 해주고, 새로운 네트워크 아키텍처를 정의할 때 발생할 수 있는 많은 오류 요소를 제거해줍니다.
* 간접적으로 정의된 변수의 값을 할당 하면 이 기능은 우회할 수 있습니다.
* `force_reinit=True`  플래그를 사용해서 초기화를 강제하거나 다시 수행하게 할 수 있습니다.


## 문제

1. 입력들의 일부만 차원을 명시하면 어떤 일이 일어날까요? 이 경우에도 즉시 초기화가 수행되나요?
1. 잘못된 자원을 명시하면 어떻게 될까요?
1. 차원이 변하는 입력이 있을 때 어떻게 해야할까요? 힌트 - 파라미터 묶기를 참조하세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2327)

![](../img/qr_deferred-init.svg)
