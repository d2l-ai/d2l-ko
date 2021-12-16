# 자동 병렬 처리
:label:`sec_auto_para`

딥 러닝 프레임워크 (예: MXNet 및 PyTorch) 는 백엔드에서 계산 그래프를 자동으로 구성합니다.계산 그래프를 사용하여 시스템은 모든 종속성을 인식하고 상호 의존적이지 않은 여러 작업을 병렬로 선택적으로 실행하여 속도를 향상시킬 수 있습니다.예를 들어, :numref:`sec_async`의 :numref:`fig_asyncgraph`는 두 변수를 독립적으로 초기화합니다.따라서 시스템은 병렬로 실행하도록 선택할 수 있습니다. 

일반적으로 단일 연산자는 모든 CPU 또는 단일 GPU의 모든 계산 리소스를 사용합니다.예를 들어 `dot` 연산자는 단일 시스템에 여러 CPU 프로세서가 있는 경우에도 모든 CPU에서 모든 코어 (및 스레드) 를 사용합니다.단일 GPU에도 동일하게 적용됩니다.따라서 단일 장치 컴퓨터에서는 병렬화가 그다지 유용하지 않습니다.장치가 여러 개인 경우 상황이 더 중요합니다.병렬화는 일반적으로 여러 GPU 간에 가장 관련이 있지만 로컬 CPU를 추가하면 성능이 약간 향상됩니다.예를 들어 GPU와 CPU를 결합한 컴퓨터 비전 모델을 학습하는 데 중점을 둔 :cite:`Hadjis.Zhang.Mitliagkas.ea.2016`를 참조하십시오.자동 병렬화 프레임워크의 편리함으로 몇 줄의 파이썬 코드로 동일한 목표를 달성할 수 있습니다.보다 광범위하게 자동 병렬 계산에 대한 논의는 CPU와 GPU를 모두 사용한 병렬 계산과 계산 및 통신의 병렬화에 중점을 둡니다. 

이 섹션의 실험을 실행하려면 최소 두 개의 GPU가 필요합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

## GPU에서의 병렬 계산

테스트할 참조 워크로드를 정의하는 것부터 시작하겠습니다. 아래 `run` 함수는 `x_gpu1` 및 `x_gpu2`의 두 변수에 할당된 데이터를 사용하여 선택한 장치에서 10개의 행렬-행렬 곱셈을 수행합니다.

```{.python .input}
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

:begin_tab:`mxnet`
이제 함수를 데이터에 적용합니다.캐싱이 결과에서 중요한 역할을 하지 않도록 측정하기 전에 장치 중 하나에서 단일 패스를 수행하여 장치를 워밍업합니다.
:end_tab:

:begin_tab:`pytorch`
이제 함수를 데이터에 적용합니다.캐싱이 결과에서 중요한 역할을 하지 않도록 측정하기 전에 장치 중 하나에서 단일 패스를 수행하여 장치를 워밍업합니다. `torch.cuda.synchronize()`는 CUDA 장치의 모든 스트림에 있는 모든 커널이 완료될 때까지 기다립니다.동기화가 필요한 장치인 `device` 인수를 사용합니다.장치 인수가 `None` (기본값) 인 경우 `current_device()`에서 지정한 현재 장치를 사용합니다.
:end_tab:

```{.python .input}
run(x_gpu1)  # Warm-up both devices
run(x_gpu2)
npx.waitall()  

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # Warm-up all devices
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
두 작업 사이에서 `waitall` 문을 제거하면 시스템은 두 장치의 계산을 자동으로 병렬화할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
두 작업 사이에서 `synchronize` 문을 제거하면 시스템은 두 장치의 계산을 자동으로 병렬화할 수 있습니다.
:end_tab:

```{.python .input}
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

위의 경우 딥 러닝 프레임워크는 사용자를 대신하여 정교한 코드 없이도 두 GPU 디바이스에서 자동으로 계산을 예약하므로 총 실행 시간은 파트의 합보다 짧습니다. 

## 병렬 계산 및 통신

대부분의 경우 CPU와 GPU 간 또는 서로 다른 GPU 간에 서로 다른 장치 간에 데이터를 이동해야 합니다.예를 들어, 여러 액셀러레이터 카드에서 그라디언트를 집계해야 하는 분산 최적화를 수행하려는 경우에 발생합니다.GPU에서 계산 한 다음 결과를 CPU로 다시 복사하여 시뮬레이션해 보겠습니다.

```{.python .input}
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
이는 다소 비효율적입니다.목록의 나머지 부분이 아직 계산되는 동안 이미 `y`의 일부를 CPU에 복사하기 시작할 수 있습니다.이러한 상황은 예를 들어 미니 배치에서 그래디언트를 계산할 때 발생합니다.일부 파라미터의 그라디언트는 다른 파라미터보다 먼저 사용할 수 있습니다.따라서 GPU가 계속 실행되는 동안 PCI-Express 버스 대역폭을 사용하는 것이 유리합니다.두 부분 사이에 `waitall`를 제거하면 이 시나리오를 시뮬레이션할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
이는 다소 비효율적입니다.목록의 나머지 부분이 아직 계산되는 동안 이미 `y`의 일부를 CPU에 복사하기 시작할 수 있습니다.이러한 상황은 예를 들어 미니 배치에서 (backprop) 그라디언트를 계산할 때 발생합니다.일부 파라미터의 그라디언트는 다른 파라미터보다 먼저 사용할 수 있습니다.따라서 GPU가 계속 실행되는 동안 PCI-Express 버스 대역폭을 사용하는 것이 유리합니다.파이토치에서 `to()` 및 `copy_()`와 같은 여러 함수는 명시적인 `non_blocking` 인수를 허용하므로 호출자가 필요하지 않을 때 동기화를 우회할 수 있습니다.`non_blocking=True`를 설정하면 이 시나리오를 시뮬레이션할 수 있습니다.
:end_tab:

```{.python .input}
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

두 작업에 필요한 총 시간은 (예상대로) 부품의 합계보다 짧습니다.이 작업은 CPU와 GPU 간의 버스라는 다른 리소스를 사용하므로 병렬 계산과는 다릅니다.실제로 두 장치에서 컴퓨팅하고 통신할 수 있습니다. 이 모든 작업을 동시에 수행할 수 있습니다.위에서 설명한 것처럼 계산과 통신 간에는 종속성이 있습니다. `y[i]`를 계산해야 CPU에 복사할 수 있습니다.다행히도 시스템은 `y[i]`를 계산하는 동안 `y[i-1]`를 복사하여 총 실행 시간을 줄일 수 있습니다. 

:numref:`fig_twogpu`에 설명 된대로 CPU와 두 개의 GPU에서 훈련 할 때 간단한 2 계층 MLP에 대한 계산 그래프와 종속성을 그림으로 마무리합니다.이로 인해 병렬 프로그램을 수동으로 예약하는 것은 매우 고통 스러울 것입니다.최적화를 위해 그래프 기반 컴퓨팅 백엔드를 사용하는 것이 유리합니다. 

![The computational graph and its dependencies of a two-layer MLP on a CPU and two GPUs.](../img/twogpu.svg)
:label:`fig_twogpu`

## 요약

* 최신 시스템에는 여러 GPU 및 CPU와 같은 다양한 장치가 있습니다.비동기적으로 병렬로 사용할 수 있습니다. 
* 최신 시스템에는 PCI Express, 스토리지 (일반적으로 솔리드 스테이트 드라이브 또는 네트워크를 통한) 및 네트워크 대역폭과 같은 다양한 통신 리소스가 있습니다.최대 효율을 위해 병렬로 사용할 수 있습니다. 
* 백엔드는 자동 병렬 계산 및 통신을 통해 성능을 향상시킬 수 있습니다. 

## 연습문제

1. 이 섹션에 정의된 `run` 함수에서 8개의 작업이 수행되었습니다.둘 사이에는 종속성이 없습니다.딥러닝 프레임워크가 자동으로 병렬로 실행되는지 확인하는 실험을 설계합니다.
1. 개별 오퍼레이터의 워크로드가 충분히 작은 경우 단일 CPU 또는 GPU에서도 병렬화가 도움이 될 수 있습니다.실험을 설계하여 이를 검증합니다. 
1. CPU, GPU 및 두 장치 간의 통신에 대한 병렬 계산을 사용하는 실험을 설계합니다.
1. NVIDIA의 [Nsight](https://developer.nvidia.com/nsight-compute-2019_5)와 같은 디버거를 사용하여 코드가 효율적인지 확인합니다. 
1. 더 복잡한 데이터 종속성을 포함하는 계산 작업을 설계하고 실험을 실행하여 성능을 향상시키면서 올바른 결과를 얻을 수 있는지 확인합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1681)
:end_tab:
