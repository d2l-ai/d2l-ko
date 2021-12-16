# GPU
:label:`sec_use_gpu`

:numref:`tab_intro_decade`에서 우리는 지난 20년 동안 계산의 급속한 성장에 대해 논의했습니다.간단히 말해서 GPU 성능은 2000년 이후 10년마다 1000배 증가했습니다.이는 좋은 기회를 제공하지만 이러한 성능을 제공해야 할 중요한 필요성을 시사합니다. 

이 섹션에서는 연구에 이러한 계산 성능을 활용하는 방법에 대해 논의하기 시작합니다.먼저 단일 GPU를 사용하고 나중에는 여러 GPU와 여러 서버 (여러 GPU 포함) 를 사용하는 방법을 설명합니다. 

특히 계산에 단일 NVIDIA GPU를 사용하는 방법에 대해 설명합니다.먼저 NVIDIA GPU가 하나 이상 설치되어 있는지 확인합니다.그런 다음 [NVIDIA driver and CUDA](https://developer.nvidia.com/cuda-downloads)를 다운로드하고 프롬프트에 따라 적절한 경로를 설정합니다.이러한 준비가 완료되면 `nvidia-smi` 명령을 사용하여 (**그래픽 카드 정보 보기**) 할 수 있습니다.

```{.python .input}
#@tab all
!nvidia-smi
```

:begin_tab:`mxnet`
MXNet 텐서가 NumPy `ndarray`와 거의 동일하게 보인다는 것을 눈치 챘을 것입니다.하지만 몇 가지 중요한 차이점이 있습니다.MXNet과 NumPy를 구별하는 주요 기능 중 하나는 다양한 하드웨어 장치를 지원한다는 것입니다. 

MXNet에서 모든 배열에는 컨텍스트가 있습니다.지금까지 기본적으로 모든 변수와 관련 계산이 CPU에 할당되었습니다.일반적으로 다른 컨텍스트는 다양한 GPU가 될 수 있습니다.여러 서버에 작업을 배포하면 상황이 더욱 까다로워질 수 있습니다.컨텍스트에 어레이를 지능적으로 할당함으로써 장치 간 데이터 전송에 소요되는 시간을 최소화할 수 있습니다.예를 들어 GPU가 있는 서버에서 신경망을 훈련시키는 경우 일반적으로 모델의 매개 변수가 GPU에 상주하는 것을 선호합니다. 

다음으로 GPU 버전의 MXNet이 설치되어 있는지 확인해야합니다.CPU 버전의 MXNet이 이미 설치되어 있는 경우 먼저 제거해야 합니다.예를 들어 `pip uninstall mxnet` 명령을 사용한 다음 CUDA 버전에 따라 해당 MXNet 버전을 설치합니다.CUDA 10.0이 설치되어 있다고 가정하면 `pip install mxnet-cu100`를 통해 CUDA 10.0을 지원하는 MXNet 버전을 설치할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch에서는 모든 배열에 장치가 있으며, 종종 이를 컨텍스트라고 부릅니다.지금까지 기본적으로 모든 변수와 관련 계산이 CPU에 할당되었습니다.일반적으로 다른 컨텍스트는 다양한 GPU가 될 수 있습니다.여러 서버에 작업을 배포하면 상황이 더욱 까다로워질 수 있습니다.컨텍스트에 어레이를 지능적으로 할당함으로써 장치 간 데이터 전송에 소요되는 시간을 최소화할 수 있습니다.예를 들어 GPU가 있는 서버에서 신경망을 훈련시키는 경우 일반적으로 모델의 매개 변수가 GPU에 상주하는 것을 선호합니다. 

다음으로 파이토치의 GPU 버전이 설치되어 있는지 확인해야 합니다.파이토치의 CPU 버전이 이미 설치되어 있는 경우, 먼저 제거해야 합니다.예를 들어 `pip uninstall torch` 명령을 사용한 다음 CUDA 버전에 따라 해당 파이토치 버전을 설치합니다.CUDA 10.0이 설치되어 있다고 가정하면 `pip install torch-cu100`를 통해 CUDA 10.0을 지원하는 파이토치 버전을 설치할 수 있습니다.
:end_tab:

이 섹션의 프로그램을 실행하려면 GPU가 두 개 이상 필요합니다.이는 대부분의 데스크톱 컴퓨터에서는 매우 유용할 수 있지만 클라우드에서 쉽게 사용할 수 있습니다 (예: AWS EC2 다중 GPU 인스턴스 사용).거의 모든 다른 섹션에는 여러 개의 GPU가 필요한*아닙니다*.대신, 이는 단순히 서로 다른 장치 간에 데이터가 어떻게 흐르는지를 설명하기 위한 것입니다. 

## [**컴퓨팅 장치**]

저장 및 계산을 위해 CPU 및 GPU와 같은 장치를 지정할 수 있습니다.기본적으로 텐서는 메인 메모리에 생성 된 다음 CPU를 사용하여 텐서를 계산합니다.

:begin_tab:`mxnet`
MXNet에서 CPU와 GPU는 `cpu()` 및 `gpu()`로 표시할 수 있습니다.`cpu()` (또는 괄호 안의 정수) 는 모든 물리적 CPU와 메모리를 의미합니다.즉, MXNet의 계산은 모든 CPU 코어를 사용하려고 시도합니다.그러나 `gpu()`는 하나의 카드와 해당 메모리만 나타냅니다.GPU가 여러 개인 경우 `gpu(i)`을 사용하여 $i^\mathrm{th}$ GPU를 나타냅니다 ($i$은 0에서 시작).또한 `gpu(0)`과 `gpu()`는 동일합니다.
:end_tab:

:begin_tab:`pytorch`
파이토치에서 CPU와 GPU는 `torch.device('cpu')` 및 `torch.device('cuda')`로 표시할 수 있습니다.`cpu` 장치는 모든 물리적 CPU와 메모리를 의미합니다.즉, 파이토치의 계산은 모든 CPU 코어를 사용하려고 시도합니다.그러나 `gpu` 장치는 하나의 카드와 해당 메모리만 나타냅니다.GPU가 여러 개인 경우 `torch.device(f'cuda:{i}')`를 사용하여 $i^\mathrm{th}$ GPU를 나타냅니다 ($i$은 0에서 시작).또한 `gpu:0`과 `gpu`은 동일합니다.
:end_tab:

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

npx.cpu(), npx.gpu(), npx.gpu(1)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

tf.device('/CPU:0'), tf.device('/GPU:0'), tf.device('/GPU:1')
```

(**사용 가능한 GPU 수를 쿼리**) 할 수 있습니다.

```{.python .input}
npx.num_gpus()
```

```{.python .input}
#@tab pytorch
torch.cuda.device_count()
```

```{.python .input}
#@tab tensorflow
len(tf.config.experimental.list_physical_devices('GPU'))
```

이제 [**요청한 GPU가 없어도 코드를 실행할 수 있는 편리한 함수 두 개를 정의합니다.**]

```{.python .input}
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    return npx.gpu(i) if npx.num_gpus() >= i + 1 else npx.cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu()] if no GPU exists."""
    devices = [npx.gpu(i) for i in range(npx.num_gpus())]
    return devices if devices else [npx.cpu()]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab pytorch
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab tensorflow
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    devices = [tf.device(f'/GPU:{i}') for i in range(num_gpus)]
    return devices if devices else [tf.device('/CPU:0')]

try_gpu(), try_gpu(10), try_all_gpus()
```

## 텐서 및 GPU

기본적으로 텐서는 CPU에 생성됩니다.[**텐서가 있는 기기를 쿼리할 수 있습니다.**]

```{.python .input}
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

여러 용어로 작동하려면 항상 동일한 장치에 있어야 한다는 점에 유의해야 합니다.예를 들어, 두 개의 텐서를 합하면 두 인수가 모두 동일한 기기에 있는지 확인해야합니다. 그렇지 않으면 프레임 워크가 결과를 저장할 위치 또는 계산을 수행 할 위치를 결정하는 방법을 알지 못합니다. 

### GPU의 스토리지

[**GPU에 텐서를 저장**] 하는 방법에는 여러 가지가 있습니다. 예를 들어 텐서를 만들 때 저장 장치를 지정할 수 있습니다.다음으로 첫 번째 `gpu`에서 텐서 변수 `X`를 만듭니다.GPU에서 생성된 텐서는 이 GPU의 메모리만 사용합니다.`nvidia-smi` 명령을 사용하여 GPU 메모리 사용량을 볼 수 있습니다.일반적으로 GPU 메모리 제한을 초과하는 데이터를 생성하지 않도록해야합니다.

```{.python .input}
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
#@tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
#@tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

GPU가 두 개 이상이라고 가정하면 다음 코드는 (**두 번째 GPU에 임의의 텐서를 생성합니다.**)

```{.python .input}
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
#@tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

### 복사

[**`X + Y`를 계산하려면 이 작업을 수행할 위치를 결정해야 합니다.**] 예를 들어 :numref:`fig_copyto`에서 볼 수 있듯이 `X`을 두 번째 GPU로 전송하여 작업을 수행할 수 있습니다.
** 단순히 `X` 및 `Y`를 추가하지 마십시오.
이 경우 예외가 발생합니다.런타임 엔진은 무엇을 해야 할지 모릅니다. 동일한 기기에서 데이터를 찾을 수 없고 실패합니다.`Y`는 두 번째 GPU에 존재하므로 `X`를 그곳으로 옮겨야 두 GPU를 추가할 수 있습니다. 

![Copy data to perform an operation on the same device.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
#@tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

이제 [**데이터가 동일한 GPU (`Z`와 `Y`가 모두 있음) 에 있으므로 추가할 수 있습니다.**]

```{.python .input}
#@tab all
Y + Z
```

:begin_tab:`mxnet`
변수 `Z`이 이미 두 번째 GPU에 있다고 가정해 보십시오.여전히 `Z.copyto(gpu(1))`로 전화하면 어떻게 될까요?해당 변수가 이미 원하는 장치에 있더라도 복사본을 만들고 새 메모리를 할당합니다.코드가 실행되는 환경에 따라 두 개의 변수가 이미 같은 기기에 존재할 수 있습니다.따라서 변수가 현재 다른 장치에 있는 경우에만 복사본을 만들고 싶습니다.이 경우 `as_in_ctx`로 전화할 수 있습니다.변수가 이미 지정된 장치에 있는 경우 이는 작동하지 않습니다.특별히 사본을 만들고 싶지 않다면 `as_in_ctx`를 선택하는 방법입니다.
:end_tab:

:begin_tab:`pytorch`
변수 `Z`가 이미 두 번째 GPU에 있다고 가정해 보십시오.여전히 `Z.cuda(1)`로 전화하면 어떻게 될까요?복사본을 만들고 새 메모리를 할당하는 대신 `Z`를 반환합니다.
:end_tab:

:begin_tab:`tensorflow`
변수 `Z`가 이미 두 번째 GPU에 있다고 가정해 보십시오.동일한 장치 범위에서 `Z2 = Z`를 계속 호출하면 어떻게 되나요?복사본을 만들고 새 메모리를 할당하는 대신 `Z`를 반환합니다.
:end_tab:

```{.python .input}
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
#@tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

### 사이드 노트

사람들은 GPU가 빠르기를 기대하기 때문에 GPU를 사용하여 기계 학습을 수행합니다.그러나 장치 간 변수 전송은 느립니다.그래서 우리는 당신이 그것을 할 수 있도록 하기 전에 느린 일을 하고 싶다는 것을 100% 확신하기를 바랍니다.딥러닝 프레임워크가 충돌 없이 자동으로 복사를 수행했다면 느린 코드를 작성했다는 사실을 깨닫지 못할 수도 있습니다. 

또한 장치 (CPU, GPU 및 기타 시스템) 간에 데이터를 전송하는 것은 계산보다 훨씬 느립니다.또한 더 많은 작업을 진행하기 전에 데이터가 전송 (또는 수신 될 때까지) 기다려야하기 때문에 병렬화가 훨씬 더 어려워집니다.따라서 복사 작업을 신중하게 수행해야 합니다.일반적으로 많은 소규모 작업은 하나의 큰 작업보다 훨씬 나쁩니다.또한, 여러분이 무엇을 하고 있는지 알지 못한다면 한 번에 여러 작업이 코드에 산재되어 있는 많은 단일 작업보다 훨씬 낫습니다.한 장치가 다른 작업을 수행하기 전에 다른 장치를 기다려야 하는 경우 이러한 작업이 차단될 수 있기 때문입니다.전화로 사전 주문하고 준비가되었는지 확인하는 대신 대기열에서 커피를 주문하는 것과 비슷합니다. 

마지막으로 텐서를 인쇄하거나 텐서를 NumPy 형식으로 변환 할 때 데이터가 주 메모리에 없으면 프레임 워크가 먼저 주 메모리에 복사하여 추가 전송 오버 헤드가 발생합니다.더 나쁜 것은 이제 모든 것이 파이썬이 완료될 때까지 기다리게 하는 두려운 전역 인터프리터 록에 종속된다는 것입니다. 

## [**신경망 및 GPU**]

마찬가지로 신경망 모델은 기기를 지정할 수 있습니다.다음 코드에서는 모델 파라미터를 GPU에 배치합니다.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

```{.python .input}
#@tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

다음 장에서는 GPU에서 모델을 실행하는 방법에 대한 더 많은 예를 볼 수 있습니다. 단순히 계산 집약적이 될 것이기 때문입니다. 

입력이 GPU의 텐서인 경우 모델은 동일한 GPU에서 결과를 계산합니다.

```{.python .input}
#@tab all
net(X)
```

(**모델 파라미터가 동일한 GPU에 저장되어 있는지 확인**)

```{.python .input}
net[0].weight.data().ctx
```

```{.python .input}
#@tab pytorch
net[0].weight.data.device
```

```{.python .input}
#@tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

즉, 모든 데이터와 매개 변수가 동일한 장치에 있으면 모델을 효율적으로 학습 할 수 있습니다.다음 장에서는 이러한 몇 가지 예를 볼 수 있습니다. 

## 요약

* CPU 또는 GPU와 같은 저장 및 계산을 위해 장치를 지정할 수 있습니다.기본적으로 데이터는 주 메모리에 생성된 다음 계산에 CPU를 사용합니다.
* 딥러닝 프레임워크에서는 계산을 위한 모든 입력 데이터가 동일한 기기 (CPU 또는 동일한 GPU) 에 있어야 합니다.
* 데이터를 신경 쓰지 않고 이동하면 성능이 크게 저하될 수 있습니다.일반적인 실수는 다음과 같습니다. GPU의 모든 미니 배치의 손실을 계산하고 명령 줄에서 사용자에게 다시 보고하거나 NumPy `ndarray`에 로깅하면 모든 GPU를 중지하는 전역 인터프리터 잠금이 트리거됩니다.GPU 내부 로깅을 위해 메모리를 할당하고 더 큰 로그만 이동하는 것이 훨씬 좋습니다.

## 연습문제

1. 큰 행렬의 곱셈과 같은 더 큰 계산 작업을 시도하고 CPU와 GPU 간의 속도 차이를 확인합니다.계산량이 적은 작업은 어떻습니까?
1. GPU에서 모델 파라미터를 읽고 쓰려면 어떻게 해야 할까요?
1. GPU에 로그를 유지하고 최종 결과만 전송하는 것과 비교하여 $100 \times 100$ 행렬의 1000개의 행렬-행렬 곱셈을 계산하고 출력 행렬의 Frobenius 노름을 한 번에 하나씩 기록하는 데 걸리는 시간을 측정합니다.
1. 하나의 GPU에서 순차적으로 두 개의 GPU에서 두 개의 행렬-행렬 곱셈을 동시에 수행하는 데 걸리는 시간을 측정합니다.힌트: 거의 선형 스케일링이 보일 것입니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/270)
:end_tab:
