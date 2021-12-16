# 컴파일러 및 인터프리터
:label:`sec_hybridize`

지금까지 이 책은 `print`, `+` 및 `if`와 같은 문장을 사용하여 프로그램의 상태를 변경하는 명령형 프로그래밍에 중점을 두었습니다.간단한 명령형 프로그램의 다음 예를 살펴보겠습니다.

```{.python .input}
#@tab all
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

파이썬은*해석된 언어*입니다.위의 `fancy_func` 함수를 평가할 때 함수의 본문을 구성하는 연산을*순서대로 수행합니다*.즉, `e = add(a, b)`를 평가하고 결과를 변수 `e`으로 저장하여 프로그램의 상태를 변경합니다.다음 두 문 `f = add(c, d)` 및 `g = add(e, f)`도 유사하게 실행되어 추가를 수행하고 결과를 변수로 저장합니다. :numref:`fig_compute_graph`는 데이터 흐름을 보여줍니다. 

![Data flow in an imperative program.](../img/computegraph.svg)
:label:`fig_compute_graph`

명령형 프로그래밍은 편리하지만 비효율적일 수 있습니다.한편으로 `add` 함수가 `fancy_func` 전체에서 반복적으로 호출되더라도, 파이썬은 세 가지 함수 호출을 개별적으로 실행합니다.예를 들어 GPU (또는 여러 GPU) 에서 실행되면 파이썬 인터프리터에서 발생하는 오버헤드가 압도될 수 있습니다.또한 `fancy_func`의 모든 문이 실행될 때까지 `e` 및 `f`의 변수 값을 저장해야 합니다.이는 `e = add(a, b)` 및 `f = add(c, d)` 문이 실행 된 후 프로그램의 다른 부분에서 변수 `e` 및 `f`를 사용할지 여부를 알 수 없기 때문입니다. 

## 기호 프로그래밍

일반적으로 공정이 완전히 정의된 후에만 계산이 수행되는 대안*기호 프로그래밍*을 생각해 보십시오.이 전략은 Theano와 TensorFlow를 포함한 여러 딥 러닝 프레임워크에서 사용됩니다 (후자는 명령형 확장을 획득했습니다).일반적으로 다음 단계가 포함됩니다. 

1. 실행할 작업을 정의합니다.
1. 작업을 실행 가능한 프로그램으로 컴파일합니다.
1. 필요한 입력을 제공하고 컴파일된 프로그램을 호출하여 실행합니다.

이를 통해 상당한 양의 최적화가 가능합니다.첫째, 많은 경우 파이썬 인터프리터를 건너 뛸 수 있으므로 CPU의 단일 Python 스레드와 쌍을 이루는 여러 개의 고속 GPU에서 중요 할 수있는 성능 병목 현상을 제거 할 수 있습니다.둘째, 컴파일러는 위의 코드를 최적화하고 `print((1 + 2) + (3 + 4))` 또는 `print(10)`로 다시 작성할 수 있습니다.컴파일러가 기계 명령어로 변환하기 전에 전체 코드를 볼 수 있기 때문에 가능합니다.예를 들어 변수가 더 이상 필요하지 않을 때마다 메모리를 해제하거나 할당하지 않을 수 있습니다.또는 코드를 완전히 동등한 부분으로 변환할 수 있습니다.더 나은 아이디어를 얻으려면 아래에서 명령형 프로그래밍의 다음 시뮬레이션 (결국 Python입니다) 을 고려하십시오.

```{.python .input}
#@tab all
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

명령형 (해석된) 프로그래밍과 기호 프로그래밍의 차이점은 다음과 같습니다. 

* 명령형 프로그래밍이 더 쉽습니다.Python에서 명령형 프로그래밍을 사용하는 경우 대부분의 코드는 간단하고 작성하기 쉽습니다.명령형 프로그래밍 코드를 디버깅하는 것도 더 쉽습니다.관련된 모든 중간 변수 값을 얻고 인쇄하거나 파이썬의 내장 디버깅 도구를 사용하는 것이 더 쉽기 때문입니다.
* 기호 프로그래밍이 더 효율적이고 이식하기가 더 쉽습니다.기호 프로그래밍을 사용하면 컴파일 중에 코드를 더 쉽게 최적화 할 수 있으며 Python과 독립적 인 형식으로 프로그램을 이식 할 수도 있습니다.이를 통해 프로그램을 비 파이썬 환경에서 실행할 수 있으므로 파이썬 인터프리터와 관련된 잠재적 성능 문제를 피할 수 있습니다.

## 하이브리드 프로그래밍

역사적으로 대부분의 딥 러닝 프레임워크는 명령적 접근 또는 상징적 접근 방식 중에서 선택합니다.예를 들어 테아노, 텐서플로우 (전자에서 영감을 얻음), 케라스 및 CNTK는 모델을 상징적으로 공식화합니다.반대로 체인너와 파이토치는 필수적인 접근 방식을 취합니다.이후 개정판에서 텐서플로우 2.0과 케라스에 명령형 모드가 추가되었습니다.

:begin_tab:`mxnet`
Gluon을 설계 할 때 개발자는 두 프로그래밍 패러다임의 이점을 결합 할 수 있는지 여부를 고려했습니다.이로 인해 사용자는 순수한 명령형 프로그래밍으로 개발 및 디버깅할 수 있는 하이브리드 모델이 탄생했으며, 제품 수준의 컴퓨팅 성능 및 배포가 필요할 때 대부분의 프로그램을 기호 프로그램으로 변환하여 실행할 수 있습니다. 

실제로 이것은 `HybridBlock` 또는 `HybridSequential` 클래스를 사용하여 모델을 빌드한다는 것을 의미합니다.기본적으로 둘 중 하나는 명령형 프로그래밍에서 `Block` 또는 `Sequential` 클래스가 실행되는 것과 같은 방식으로 실행됩니다.`HybridSequential` 클래스는 `HybridBlock`의 하위 클래스입니다 (`Sequential` 서브클래스 `Block`과 마찬가지로).`hybridize` 함수가 호출되면 Gluon은 모델을 기호 프로그래밍에 사용되는 형식으로 컴파일합니다.이를 통해 모델 구현 방식을 희생하지 않고 계산 집약적 구성 요소를 최적화 할 수 있습니다.순차 모델 및 블록에 초점을 맞추어 다음과 같은 이점을 설명하겠습니다.
:end_tab:

:begin_tab:`pytorch`
위에서 언급했듯이 PyTorch는 명령형 프로그래밍을 기반으로하며 동적 계산 그래프를 사용합니다.기호 프로그래밍의 이식성과 효율성을 활용하기 위해 개발자는 두 프로그래밍 모델의 이점을 결합 할 수 있는지 여부를 고려했습니다.이로 인해 사용자는 순수한 명령형 프로그래밍을 사용하여 개발 및 디버깅할 수 있는 토치 스크립트가 생성되었으며, 제품 수준의 컴퓨팅 성능 및 배포가 필요할 때 대부분의 프로그램을 기호 프로그램으로 변환할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
명령형 프로그래밍 패러다임은 이제 Tensorflow 2의 기본값이며, 이는 언어를 처음 접하는 사람들에게 환영받는 변화입니다.그러나 동일한 기호 프로그래밍 기법과 후속 계산 그래프가 여전히 TensorFlow에 존재하며 사용하기 쉬운 `tf.function` 데코레이터를 통해 액세스할 수 있습니다.이로 인해 필수 프로그래밍 패러다임이 TensorFlow에 도입되어 사용자가 보다 직관적인 함수를 정의한 다음 TensorFlow 팀에서 [autograph](https://www.tensorflow.org/api_docs/python/tf/autograph)라고 부르는 기능을 사용하여 자동으로 래핑하고 계산 그래프로 컴파일할 수 있었습니다.
:end_tab:

## `Sequential` 클래스를 하이브리드화

하이브리드화의 작동 방식을 이해하는 가장 쉬운 방법은 여러 계층으로 구성된 심층 네트워크를 고려하는 것입니다.일반적으로 파이썬 인터프리터는 CPU 또는 GPU로 전달할 수 있는 명령을 생성하기 위해 모든 계층에 대해 코드를 실행해야 합니다.단일 (고속) 컴퓨팅 장치의 경우 이로 인해 큰 문제가 발생하지 않습니다.반면에 AWS P3DN.24xlarge 인스턴스와 같은 고급 8 GPU 서버를 사용하는 경우 파이썬은 모든 GPU를 바쁘게 유지하는 데 어려움을 겪을 것입니다.단일 스레드 파이썬 인터프리터는 여기서 병목 현상이 됩니다.`Sequential`를 `HybridSequential`로 대체하여 코드의 중요한 부분에 대해 이 문제를 어떻게 해결할 수 있는지 살펴보겠습니다.간단한 MLP를 정의하는 것부터 시작합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# Factory for networks
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Factory for networks
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Factory for networks
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)
```

:begin_tab:`mxnet`
`hybridize` 함수를 호출하면 MLP에서 계산을 컴파일하고 최적화할 수 있습니다.모델의 계산 결과는 변경되지 않고 그대로 유지됩니다.
:end_tab:

:begin_tab:`pytorch`
`torch.jit.script` 함수를 사용하여 모델을 변환하면 MLP에서 계산을 컴파일하고 최적화 할 수 있습니다.모델의 계산 결과는 변경되지 않고 그대로 유지됩니다.
:end_tab:

:begin_tab:`tensorflow`
이전에는 TensorFlow에 빌드된 모든 함수가 계산 그래프로 빌드되었으므로 기본적으로 JIT가 컴파일되었습니다.그러나 텐서플로우 2.X 및 EagerTensor가 출시되면서 더 이상 기본 동작이 아닙니다.tf.function을 사용하여이 기능을 다시 활성화 할 수 있습니다. tf.function은 함수 데코레이터로 더 일반적으로 사용되지만 아래와 같이 일반 파이썬 함수로 직접 호출 할 수 있습니다.모델의 계산 결과는 변경되지 않고 그대로 유지됩니다.
:end_tab:

```{.python .input}
net.hybridize()
net(x)
```

```{.python .input}
#@tab pytorch
net = torch.jit.script(net)
net(x)
```

```{.python .input}
#@tab tensorflow
net = tf.function(net)
net(x)
```

:begin_tab:`mxnet`
이것은 사실이 되기에는 너무 좋은 것 같습니다. 블록을 `HybridSequential`으로 지정하고 이전과 동일한 코드를 작성한 다음 `hybridize`를 호출하면 됩니다.이 경우 네트워크가 최적화됩니다 (아래에서 성능을 벤치마킹합니다).안타깝게도 모든 레이어에서 마법처럼 작동하지는 않습니다.즉, 레이어가 `HybridBlock` 클래스 대신 `Block` 클래스에서 상속되는 경우 레이어가 최적화되지 않습니다.
:end_tab:

:begin_tab:`pytorch`
이것은 사실이 되기에는 너무 좋은 것 같습니다. 이전과 동일한 코드를 작성하고 `torch.jit.script`를 사용하여 모델을 변환하기만 하면 됩니다.이 경우 네트워크가 최적화됩니다 (아래에서 성능을 벤치마킹합니다).
:end_tab:

:begin_tab:`tensorflow`
이것은 사실이 되기에는 너무 좋은 것 같습니다. 이전과 동일한 코드를 작성하고 `tf.function`를 사용하여 모델을 변환하기만 하면 됩니다.이런 일이 발생하면 네트워크는 TensorFlow의 MLIR 중간 표현에서 계산 그래프로 구축되며 빠른 실행을 위해 컴파일러 수준에서 크게 최적화됩니다 (아래 성능을 벤치마킹합니다).`tf.function()` 호출에 `jit_compile = True` 플래그를 명시적으로 추가하면 텐서플로에서 XLA (가속 선형 대수) 기능이 활성화됩니다.XLA는 특정 인스턴스에서 JIT 컴파일 코드를 추가로 최적화할 수 있습니다.그래프 모드 실행은 이러한 명시적 정의 없이 활성화되지만 XLA는 특히 GPU 환경에서 특정 대형 선형 대수 연산 (딥 러닝 애플리케이션에서 볼 수 있는 것과 같은 맥락에서) 을 훨씬 빠르게 만들 수 있습니다.
:end_tab:

### 하이브리드화를 통한 가속

컴파일을 통해 얻은 성능 향상을 입증하기 위해 하이브리드화 전후에 `net(x)`를 평가하는 데 필요한 시간을 비교합니다.이번에는 먼저 측정할 클래스를 정의해 보겠습니다.성능을 측정 (및 개선) 하기 시작하면서 챕터 전체에서 유용하게 사용될 것입니다.

```{.python .input}
#@tab all
#@save
class Benchmark:
    """For measuring running time."""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```

:begin_tab:`mxnet`
이제 하이브리드화 없이 네트워크를 두 번, 한 번, 한 번 호출할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
이제 횃불 없이 네트워크를 두 번, 한 번, 한 번 호출할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
이제 네트워크를 세 번 호출할 수 있습니다. 한 번은 열심히 실행되고, 한 번은 그래프 모드 실행으로 한 번, 그리고 JIT 컴파일된 XLA를 사용하여 다시 호출할 수 있습니다.
:end_tab:

```{.python .input}
net = get_net()
with Benchmark('Without hybridization'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('With hybridization'):
    for i in range(1000): net(x)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
net = get_net()
with Benchmark('Without torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('With torchscript'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab tensorflow
net = get_net()
with Benchmark('Eager Mode'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph Mode'):
    for i in range(1000): net(x)
```

:begin_tab:`mxnet`
위의 결과에서 볼 수 있듯이 `HybridSequential` 인스턴스가 `hybridize` 함수를 호출한 후 기호 프로그래밍을 사용하여 컴퓨팅 성능이 향상됩니다.
:end_tab:

:begin_tab:`pytorch`
위의 결과에서 볼 수 있듯이 `nn.Sequential` 인스턴스를 `torch.jit.script` 함수를 사용하여 스크립팅한 후 기호 프로그래밍을 사용하여 컴퓨팅 성능이 향상됩니다.
:end_tab:

:begin_tab:`tensorflow`
위의 결과에서 볼 수 있듯이 `tf.keras.Sequential` 인스턴스를 `tf.function` 함수를 사용하여 스크립팅한 후 tensorflow에서 그래프 모드 실행을 통한 기호 프로그래밍을 사용하여 컴퓨팅 성능이 향상됩니다.
:end_tab:

### 직렬화

:begin_tab:`mxnet`
모델 컴파일의 이점 중 하나는 모델과 매개 변수를 디스크에 직렬화 (저장) 할 수 있다는 것입니다.이를 통해 선택한 프런트 엔드 언어와 독립적인 방식으로 모델을 저장할 수 있습니다.이를 통해 학습된 모델을 다른 장치에 배포하고 다른 프런트 엔드 프로그래밍 언어를 쉽게 사용할 수 있습니다.동시에 코드는 명령형 프로그래밍에서 얻을 수 있는 것보다 더 빠른 경우가 많습니다.`export` 함수가 실제로 작동하는지 살펴 보겠습니다.
:end_tab:

:begin_tab:`pytorch`
모델 컴파일의 이점 중 하나는 모델과 매개 변수를 디스크에 직렬화 (저장) 할 수 있다는 것입니다.이를 통해 선택한 프런트 엔드 언어와 독립적인 방식으로 모델을 저장할 수 있습니다.이를 통해 학습된 모델을 다른 장치에 배포하고 다른 프런트 엔드 프로그래밍 언어를 쉽게 사용할 수 있습니다.동시에 코드는 명령형 프로그래밍에서 얻을 수 있는 것보다 더 빠른 경우가 많습니다.`save` 함수가 실제로 작동하는지 살펴 보겠습니다.
:end_tab:

:begin_tab:`tensorflow`
모델 컴파일의 이점 중 하나는 모델과 매개 변수를 디스크에 직렬화 (저장) 할 수 있다는 것입니다.이를 통해 선택한 프런트 엔드 언어와 독립적인 방식으로 모델을 저장할 수 있습니다.이를 통해 학습된 모델을 다른 장치에 배포하고 다른 프런트 엔드 프로그래밍 언어를 쉽게 사용하거나 서버에서 학습된 모델을 실행할 수 있습니다.동시에 코드는 명령형 프로그래밍에서 얻을 수 있는 것보다 더 빠른 경우가 많습니다.텐서플로를 절약할 수 있는 저수준 API는 `tf.saved_model`입니다.`saved_model` 인스턴스가 실제로 작동하는지 살펴보겠습니다.
:end_tab:

```{.python .input}
net.export('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab pytorch
net.save('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab tensorflow
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```

:begin_tab:`mxnet`
모델은 모델 계산을 실행하는 데 필요한 프로그램의 JSON 설명과 (큰 바이너리) 매개 변수 파일로 분해됩니다.이 파일은 C++, R, 스칼라 및 펄과 같이 파이썬이나 MXNet에서 지원하는 다른 프런트 엔드 언어에서 읽을 수 있습니다.모델 설명의 처음 몇 줄을 살펴 보겠습니다.
:end_tab:

```{.python .input}
!head my_mlp-symbol.json
```

:begin_tab:`mxnet`
앞서 `hybridize` 함수를 호출한 후 모델이 뛰어난 컴퓨팅 성능과 이식성을 달성할 수 있음을 입증했습니다.하이브리드화는 특히 제어 흐름 측면에서 모델 유연성에 영향을 미칠 수 있습니다.  

또한 `forward` 함수를 사용해야 하는 `Block` 인스턴스와 달리 `HybridBlock` 인스턴스의 경우 `hybrid_forward` 함수를 사용해야 합니다.
:end_tab:

```{.python .input}
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```

:begin_tab:`mxnet`
위 코드는 4개의 은닉 유닛과 2개의 출력으로 구성된 간단한 네트워크를 구현합니다.`hybrid_forward` 함수는 추가 인수 `F`를 사용합니다.이는 코드가 하이브리드화되었는지 여부에 따라 처리를 위해 약간 다른 라이브러리 (`ndarray` 또는 `symbol`) 를 사용하기 때문에 필요합니다.두 클래스 모두 매우 유사한 기능을 수행하며 MXNet은 자동으로 인수를 결정합니다.무슨 일이 일어나고 있는지 이해하기 위해 함수 호출의 일부로 인수를 인쇄합니다.
:end_tab:

```{.python .input}
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```

:begin_tab:`mxnet`
순방향 계산을 반복하면 동일한 출력이 생성됩니다 (세부 사항은 생략).이제 `hybridize` 함수를 호출하면 어떤 일이 발생하는지 살펴보겠습니다.
:end_tab:

```{.python .input}
net.hybridize()
net(x)
```

:begin_tab:`mxnet`
`ndarray`를 사용하는 대신 `F`에 `symbol` 모듈을 사용합니다.또한 입력이 `ndarray` 유형이지만 네트워크를 통해 흐르는 데이터는 이제 컴파일 프로세스의 일부로 `symbol` 유형으로 변환됩니다.함수 호출을 반복하면 놀라운 결과를 얻을 수 있습니다.
:end_tab:

```{.python .input}
net(x)
```

:begin_tab:`mxnet`
이것은 이전에 본 것과는 상당히 다릅니다.`hybrid_forward`에 정의된 모든 인쇄 문은 생략됩니다.실제로 하이브리드화 후 `net(x)`의 실행에는 더 이상 파이썬 인터프리터가 포함되지 않습니다.즉, 훨씬 더 간소화된 실행과 더 나은 성능을 위해 가짜 파이썬 코드 (예: print 문) 가 생략됩니다.대신 MXNet은 C++ 백엔드를 직접 호출합니다.또한 일부 함수는 `symbol` 모듈 (예: `asnumpy`) 에서 지원되지 않으며 `a += b` 및 `a[:] = a + b`과 같은 현재 위치에서 수행되는 작업은 `a = a + b`으로 다시 작성해야 합니다.그럼에도 불구하고 속도가 중요할 때마다 모델을 컴파일하는 것은 그만한 가치가 있습니다.이점은 모델의 복잡성, CPU 속도, GPU 속도 및 수에 따라 작은 백분율 포인트에서 두 배 이상의 속도까지 다양합니다.
:end_tab:

## 요약

* 명령형 프로그래밍을 사용하면 제어 흐름과 많은 양의 Python 소프트웨어 에코시스템을 사용할 수 있는 코드를 작성할 수 있으므로 새 모델을 쉽게 설계할 수 있습니다.
* 기호 프로그래밍을 수행하려면 프로그램을 지정하고 실행하기 전에 컴파일해야 합니다.이점은 성능 향상입니다.

:begin_tab:`mxnet`
* MXNet은 필요에 따라 두 가지 접근 방식의 장점을 결합할 수 있습니다.
* `HybridSequential` 및 `HybridBlock` 클래스로 구성된 모델은 `hybridize` 함수를 호출하여 명령형 프로그램을 기호 프로그램으로 변환할 수 있습니다.
:end_tab:

## 연습문제

:begin_tab:`mxnet`
1. 이 섹션에 있는 `HybridNet` 클래스의 `hybrid_forward` 함수의 첫 번째 줄에 `x.asnumpy()`를 추가합니다.코드를 실행하고 발생한 오류를 관찰합니다.왜 그런 일이 일어날까요?
1. 제어 흐름, 즉 `hybrid_forward` 함수에서 파이썬 문장 `if` 및 `for`를 추가하면 어떻게 될까요?
1. 이전 장에서 관심 있는 모델을 검토합니다.다시 구현하여 컴퓨팅 성능을 향상시킬 수 있습니까?
:end_tab:

:begin_tab:`pytorch,tensorflow`
1. 이전 장에서 관심 있는 모델을 검토합니다.다시 구현하여 컴퓨팅 성능을 향상시킬 수 있습니까?
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/360)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2490)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2492)
:end_tab:
