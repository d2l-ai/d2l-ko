# 비동기 계산
:label:`sec_async`

오늘날의 컴퓨터는 다중 CPU 코어 (종종 코어당 다중 스레드), GPU당 다중 처리 요소 및 장치당 여러 GPU로 구성된 고도의 병렬 시스템입니다.요컨대, 우리는 종종 다른 장치에서 여러 가지 다른 것들을 동시에 처리 할 수 있습니다.안타깝게도 파이썬은 병렬 및 비동기 코드를 작성하는 좋은 방법이 아닙니다. 적어도 추가 도움 없이는 그렇지 않습니다.결국 파이썬은 단일 스레드이며 향후 변경되지 않을 것입니다.MXNet 및 TensorFlow와 같은 딥 러닝 프레임워크는 성능을 개선하기 위해*비동기 프로그래밍* 모델을 채택하고, PyTorch는 Python의 자체 스케줄러를 사용하여 다른 성능 절충을 초래합니다.파이토치의 경우 기본적으로 GPU 작업은 비동기식입니다.GPU를 사용하는 함수를 호출하면 작업이 특정 장치에 큐에 들어가지만 나중에야 실행할 필요는 없습니다.이를 통해 CPU 또는 기타 GPU에서의 작업을 포함하여 더 많은 계산을 병렬로 실행할 수 있습니다. 

따라서 비동기 프로그래밍의 작동 방식을 이해하면 계산 요구 사항과 상호 종속성을 사전에 줄임으로써 보다 효율적인 프로그램을 개발할 수 있습니다.이를 통해 메모리 오버헤드를 줄이고 프로세서 사용률을 높일 수 있습니다.

```{.python .input}
from d2l import mxnet as d2l
import numpy, os, subprocess
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy, os, subprocess
import torch
from torch import nn
```

## 백엔드를 통한 비동기

:begin_tab:`mxnet`
워밍업을 위해 다음과 같은 장난감 문제를 고려하십시오. 랜덤 행렬을 생성하고 곱하려고 합니다.차이점을 확인하기 위해 NumPy와 `mxnet.np`에서 모두 수행해 보겠습니다.
:end_tab:

:begin_tab:`pytorch`
워밍업을 위해 다음과 같은 장난감 문제를 고려하십시오. 랜덤 행렬을 생성하고 곱하려고 합니다.차이점을 확인하기 위해 NumPy와 PyTorch 텐서 모두에서 수행해 보겠습니다.파이토치 `tensor`는 GPU에 정의되어 있습니다.
:end_tab:

```{.python .input}
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```

```{.python .input}
#@tab pytorch
# Warmup for GPU computation
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
```

:begin_tab:`mxnet`
MXNet을 통한 벤치마크 출력은 훨씬 더 빠릅니다.둘 다 동일한 프로세서에서 실행되므로 다른 작업을 수행해야 합니다.반환하기 전에 MXNet이 모든 백엔드 계산을 끝내도록 강제하면 이전에 무슨 일이 있었는지 알 수 있습니다. 계산은 백엔드에 의해 실행되고 프런트 엔드는 Python에 제어를 반환합니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch를 통한 벤치마크 출력은 훨씬 더 빠릅니다.NumPy 내적은 CPU 프로세서에서 실행되는 반면 PyTorch 행렬 곱셈은 GPU에서 실행되므로 후자는 훨씬 더 빠를 것으로 예상됩니다.그러나 엄청난 시차는 다른 일이 벌어지고 있음을 암시합니다.기본적으로 GPU 작업은 파이토치에서 비동기적입니다.반환하기 전에 PyTorch가 모든 계산을 끝내도록 강제하면 이전에 무슨 일이 있었는지 알 수 있습니다: 계산은 백엔드에 의해 실행되고 프론트엔드는 파이썬에 제어를 반환합니다.
:end_tab:

```{.python .input}
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)
```

:begin_tab:`mxnet`
일반적으로 MXNet에는 Python을 통한 사용자와의 직접적인 상호 작용을위한 프론트 엔드와 시스템에서 계산을 수행하는 데 사용하는 백엔드가 있습니다.:numref:`fig_frontends`에서 볼 수 있듯이 사용자는 파이썬, R, 스칼라 및 C++와 같은 다양한 프런트 엔드 언어로 MXNet 프로그램을 작성할 수 있습니다.사용되는 프런트 엔드 프로그래밍 언어에 관계없이 MXNet 프로그램의 실행은 주로 C++ 구현의 백엔드에서 발생합니다.프런트엔드 언어로 실행된 작업은 실행을 위해 백엔드로 전달됩니다.백엔드는 대기 중인 작업을 지속적으로 수집하고 실행하는 자체 스레드를 관리합니다.이 기능이 작동하려면 백엔드에서 계산 그래프의 다양한 단계 간의 종속성을 추적할 수 있어야 합니다.따라서 서로 종속된 연산은 병렬화할 수 없습니다.
:end_tab:

:begin_tab:`pytorch`
일반적으로 PyTorch에는 Python을 통해 사용자와 직접 상호 작용할 수있는 프론트 엔드와 시스템에서 계산을 수행하는 데 사용하는 백엔드가 있습니다.:numref:`fig_frontends`에서 볼 수 있듯이 사용자는 파이썬과 C++와 같은 다양한 프론트엔드 언어로 파이토치 프로그램을 작성할 수 있습니다.사용되는 프론트엔드 프로그래밍 언어에 관계없이 PyTorch 프로그램의 실행은 주로 C++ 구현의 백엔드에서 발생합니다.프런트엔드 언어로 실행된 작업은 실행을 위해 백엔드로 전달됩니다.백엔드는 대기 중인 작업을 지속적으로 수집하고 실행하는 자체 스레드를 관리합니다.이 기능이 작동하려면 백엔드에서 계산 그래프의 다양한 단계 간의 종속성을 추적할 수 있어야 합니다.따라서 서로 종속된 연산은 병렬화할 수 없습니다.
:end_tab:

![Programming language frontends and deep learning framework backends.](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`

종속성 그래프를 좀 더 잘 이해하기 위해 또 다른 장난감 예제를 살펴보겠습니다.

```{.python .input}
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

```{.python .input}
#@tab pytorch
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z
```

![The backend tracks dependencies between various steps in the computational graph.](../img/asyncgraph.svg)
:label:`fig_asyncgraph`

위의 코드 스니펫은 :numref:`fig_asyncgraph`에도 설명되어 있습니다.파이썬 프론트엔드 스레드가 처음 세 문 중 하나를 실행할 때마다, 단순히 작업을 백엔드 큐에 반환합니다.마지막 명령문의 결과를*인쇄*해야 하는 경우, 파이썬 프론트엔드 스레드는 C++ 백엔드 스레드가 변수 `z`의 결과 계산을 마칠 때까지 기다립니다.이 설계의 한 가지 이점은 파이썬 프론트엔드 스레드가 실제 계산을 수행할 필요가 없다는 것입니다.따라서 파이썬의 성능에 관계없이 프로그램의 전반적인 성능에는 거의 영향을 미치지 않습니다. :numref:`fig_threading`는 프론트엔드와 백엔드가 상호 작용하는 방식을 보여줍니다. 

![Interactions of the frontend and backend.](../img/threading.svg)
:label:`fig_threading`

## 배리어 및 블로커

:begin_tab:`mxnet`
파이썬이 완료를 기다리도록 강제하는 많은 연산이 있습니다: 

* 가장 명백하게 `npx.waitall()`는 계산 명령이 언제 발행되었는지에 관계없이 모든 계산이 완료될 때까지 기다립니다.실제로 이 연산자는 성능 저하로 이어질 수 있으므로 반드시 필요한 경우가 아니면 사용하는 것은 좋지 않습니다.
* 특정 변수를 사용할 수 있을 때까지 기다리려면 `z.wait_to_read()`를 호출할 수 있습니다.이 경우 MXNet 블록은 변수 `z`가 계산될 때까지 파이썬으로 돌아갑니다.나중에 다른 계산이 계속 될 수 있습니다.

이것이 실제로 어떻게 작동하는지 봅시다.
:end_tab:

```{.python .input}
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```

:begin_tab:`mxnet`
두 작업을 모두 완료하는 데 거의 같은 시간이 걸립니다.명백한 차단 작업 외에도*암시적* 차단기를 알고 있는 것이 좋습니다.변수를 명확하게 인쇄하려면 변수를 사용할 수 있어야 하므로 차단기가 됩니다.마지막으로, NumPy에는 비동기 개념이 없기 때문에 `z.asnumpy()`을 통한 NumPy로의 변환과 `z.item()`를 통한 스칼라로의 변환이 차단됩니다.`print` 함수와 마찬가지로 값에 액세스해야 합니다.  

소량의 데이터를 MXNet의 범위에서 NumPy로 자주 복사하면 효율적인 코드의 성능이 저하 될 수 있습니다. 이러한 각 작업은 계산 그래프가 다른 작업을 수행하기 전에* 관련 용어를 얻는 데 필요한 모든 중간 결과를 평가해야하기 때문입니다.
:end_tab:

```{.python .input}
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

## 계산 개선

:begin_tab:`mxnet`
다중 스레드가 많은 시스템 (일반 랩톱에서도 스레드가 4개 이상이고 다중 소켓 서버에서는 이 수가 256개를 초과할 수 있음) 에서는 예약 작업의 오버헤드가 상당히 커질 수 있습니다.따라서 계산과 스케줄링을 비동기적으로 병렬로 수행하는 것이 매우 바람직합니다.이렇게 하는 것의 이점을 설명하기 위해 변수를 순서대로 또는 비동기적으로 1씩 여러 번 증가시키면 어떤 일이 발생하는지 살펴보겠습니다.각 추가 사이에 `wait_to_read` 장벽을 삽입하여 동기 실행을 시뮬레이션합니다.
:end_tab:

```{.python .input}
with d2l.Benchmark('synchronous'):
    for _ in range(10000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(10000):
        y = x + 1
    npx.waitall()
```

:begin_tab:`mxnet`
파이썬 프론트엔드 스레드와 C++ 백엔드 스레드 간의 약간 단순화된 상호 작용은 다음과 같이 요약할 수 있습니다.
1. 프런트 엔드는 백엔드에 계산 작업 `y = x + 1`를 대기열에 삽입하도록 명령합니다.
1. 그런 다음 백엔드는 대기열에서 계산 작업을 수신하고 실제 계산을 수행합니다.
1. 그런 다음 백엔드는 계산 결과를 프런트엔드로 반환합니다.
이 세 단계의 지속 시간이 각각 $t_1, t_2$과 $t_3$이라고 가정합니다.비동기 프로그래밍을 사용하지 않는 경우 10,000회의 계산을 수행하는 데 걸리는 총 시간은 약 $10000 (t_1+ t_2 + t_3)$입니다.비동기 프로그래밍을 사용하는 경우 프런트엔드가 백엔드가 각 루프에 대한 계산 결과를 반환할 때까지 기다릴 필요가 없으므로 10000개의 계산을 수행하는 데 걸리는 총 시간을 $t_1 + 10000 t_2 + t_3$ ($10000 t_2 > 9999t_1$으로 가정) 로 줄일 수 있습니다.
:end_tab:

## 요약

* 딥러닝 프레임워크는 파이썬 프론트엔드를 실행 백엔드에서 분리할 수 있습니다.이를 통해 백엔드에 명령을 비동기식으로 빠르게 삽입하고 병렬화할 수 있습니다.
* 비동기는 다소 반응이 빠른 프론트엔드로 이어집니다.그러나 작업 큐를 너무 많이 채우면 메모리가 과도하게 소모될 수 있으므로 주의해야 합니다.프런트엔드와 백엔드가 거의 동기화되도록 각 미니배치를 동기화하는 것이 좋습니다.
* 칩 공급업체는 딥 러닝의 효율성에 대해 훨씬 더 세분화된 통찰력을 얻을 수 있는 정교한 성능 분석 도구를 제공합니다.

:begin_tab:`mxnet`
* MXNet의 메모리 관리에서 Python으로 변환하면 특정 변수가 준비될 때까지 백엔드가 대기하게 된다는 사실에 유의하십시오.`print`, `asnumpy` 및 `item`와 같은 함수는 모두 이 효과가 있습니다.이렇게 하는 것이 바람직할 수 있지만 부주의하게 동기화를 사용하면 성능이 저하될 수 있습니다.
:end_tab:

## 연습문제

:begin_tab:`mxnet`
1. 위에서 언급한 바와 같이 비동기 계산을 사용하면 10,000회의 계산을 수행하는 데 필요한 총 시간을 $t_1 + 10000 t_2 + t_3$로 줄일 수 있습니다.여기서 $10000 t_2 > 9999 t_1$를 가정해야 하는 이유는 무엇입니까?
1. `waitall`와 `wait_to_read` 사이의 차이를 측정합니다.힌트: 여러 명령을 수행하고 중간 결과를 위해 동기화합니다.
:end_tab:

:begin_tab:`pytorch`
1. CPU에서 이 섹션의 동일한 행렬 곱셈 연산을 벤치마킹합니다.여전히 백엔드를 통해 비동기를 관찰할 수 있습니까?
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/361)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2564)
:end_tab:
