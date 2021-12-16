# 지연된 초기화
:label:`sec_deferred_init`

지금까지는 네트워크 설정이 엉성한 것처럼 보일 수 있습니다.특히 다음과 같은 직관적이지 않은 작업을 수행했는데 작동하지 않는 것처럼 보일 수 있습니다. 

* 입력 차원을 지정하지 않고 네트워크 아키텍처를 정의했습니다.
* 이전 레이어의 출력 차원을 지정하지 않고 레이어를 추가했습니다.
* 모델에 포함해야 할 매개 변수 수를 결정하기에 충분한 정보를 제공하기 전에 이러한 매개 변수를 “초기화”했습니다.

코드가 전혀 실행되지 않는다는 사실에 놀라실 것입니다.결국 딥러닝 프레임워크가 네트워크의 입력 차원이 무엇인지 알 수 있는 방법은 없습니다.여기서 비결은 프레임워크가 초기화를 연기하고 모델을 통해 데이터를 처음 전달할 때까지 대기하여 각 레이어의 크기를 즉시 추론하는 것입니다. 

나중에 컨벌루션 신경망으로 작업할 때 입력 차원 (즉, 이미지의 해상도) 이 각 후속 계층의 차원에 영향을 미치기 때문에 이 기법이 훨씬 더 편리해질 것입니다.따라서 코드를 작성할 때 차원이 무엇인지 알 필요없이 매개 변수를 설정하는 기능은 모델을 지정하고 이후에 수정하는 작업을 크게 단순화 할 수 있습니다.다음으로 초기화 메커니즘에 대해 자세히 설명합니다. 

## 네트워크 인스턴스화

시작하기 위해 MLP를 인스턴스화 해 보겠습니다.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = get_net()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

이 시점에서는 입력 차원이 알려지지 않았기 때문에 네트워크는 입력 계층의 가중치 차원을 알 수 없습니다.따라서 프레임워크는 아직 매개변수를 초기화하지 않았습니다.아래 매개 변수에 액세스하려고 시도하여 확인합니다.

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
#@tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
파라미터 개체가 존재하는 동안 각 레이어의 입력 치수는 -1로 나열됩니다.MXNet은 매개변수 차원이 알 수 없는 상태로 남아 있음을 나타내기 위해 특수 값 -1을 사용합니다.이 시점에서 `net[0].weight.data()`에 액세스하려고 하면 매개 변수에 액세스하기 전에 네트워크를 초기화해야 한다는 런타임 오류가 트리거됩니다.이제 `initialize` 함수를 통해 매개 변수를 초기화하려고 할 때 어떤 일이 발생하는지 살펴 보겠습니다.
:end_tab:

:begin_tab:`tensorflow`
각 레이어 객체는 존재하지만 가중치는 비어 있습니다.`net.get_weights()`를 사용하면 가중치가 아직 초기화되지 않았으므로 오류가 발생합니다.
:end_tab:

```{.python .input}
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
보시다시피 아무것도 변하지 않았습니다.입력 차원을 알 수 없는 경우 initialize를 호출해도 매개 변수가 실제로 초기화되지 않습니다.대신 이 호출은 매개 변수를 초기화하기를 원하는 (그리고 선택적으로 어떤 배포에 따라) MXNet에 등록됩니다.
:end_tab:

다음으로 네트워크를 통해 데이터를 전달하여 프레임 워크가 최종적으로 매개 변수를 초기화하도록합니다.

```{.python .input}
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

입력 차원 (20) 을 알게 되자마자 프레임워크는 값 20을 연결하여 첫 번째 레이어의 가중치 행렬의 모양을 식별할 수 있습니다.첫 번째 레이어의 모양을 인식하면 프레임워크는 모든 모양이 알려질 때까지 계산 그래프를 통해 두 번째 레이어로 진행됩니다.이 경우 첫 번째 계층에만 지연된 초기화가 필요하지만 프레임워크는 순차적으로 초기화됩니다.모든 매개 변수 모양이 알려지면 프레임워크가 최종적으로 매개 변수를 초기화할 수 있습니다. 

## 요약

* 초기화를 지연하면 프레임워크가 매개 변수 셰이프를 자동으로 추론할 수 있으므로 아키텍처를 쉽게 수정할 수 있으며 일반적인 오류 원인 하나를 제거할 수 있습니다.
* 모델을 통해 데이터를 전달하여 프레임 워크가 최종적으로 매개 변수를 초기화하도록 할 수 있습니다.

## 연습문제

1. 첫 번째 레이어에는 입력 치수를 지정하고 후속 레이어에는 지정하지 않으면 어떻게 됩니까?즉시 초기화가 가능한가요?
1. 일치하지 않는 차원을 지정하면 어떻게 됩니까?
1. 다양한 차원에 대한 입력이 있는 경우 어떻게 해야 합니까?힌트: 매개 변수 묶기를 살펴보십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/281)
:end_tab:
