# 다중 GPU에 대한 교육
:label:`sec_multi_gpu`

지금까지 CPU와 GPU에서 모델을 효율적으로 학습하는 방법에 대해 논의했습니다.우리는 :numref:`sec_auto_para`에서 딥 러닝 프레임워크가 어떻게 계산과 통신을 자동으로 병렬화할 수 있는지를 보여주었습니다.또한 :numref:`sec_use_gpu`에서 `nvidia-smi` 명령을 사용하여 컴퓨터에서 사용 가능한 모든 GPU를 나열하는 방법을 보여주었습니다.우리가 논의하지 않은 것은 딥 러닝 트레이닝을 실제로 병렬화하는 방법에 관한 것입니다.대신, 우리는 어떻게 든 여러 장치로 데이터를 분할하여 작동시킬 것이라는 것을 암시했습니다.본 섹션에서는 세부 사항을 채우고 처음부터 시작할 때 네트워크를 병렬로 훈련시키는 방법을 보여줍니다.상위 수준 API에서 기능을 활용하는 방법에 대한 자세한 내용은 :numref:`sec_multi_gpu_concise`로 강등됩니다.여기서는 사용자가 :numref:`sec_minibatch_sgd`에 설명된 것과 같은 미니배치 확률적 경사하강법 알고리즘에 익숙하다고 가정합니다. 

## 문제 분할

간단한 컴퓨터 비전 문제와 약간 오래된 네트워크 (예: 여러 계층의 컨볼 루션, 풀링 및 결국 몇 개의 완전히 연결된 계층) 로 시작하겠습니다.즉, 르넷 :cite:`LeCun.Bottou.Bengio.ea.1998` 또는 알렉스넷 :cite:`Krizhevsky.Sutskever.Hinton.2012`와 매우 유사한 네트워크로부터 시작하겠습니다.여러 개의 GPU (데스크톱 서버인 경우 2개, AWS g4dn.12xlarge 인스턴스의 경우 4개, p3.16xlarge의 경우 8개, p2.16xlarge의 경우 16개) 를 고려할 때, 간단하고 재현 가능한 설계 선택의 이점을 얻는 동시에 우수한 속도 향상을 달성하는 방식으로 교육을 분할하려고 합니다.결국 다중 GPU는*메모리*와*계산* 능력을 모두 향상시킵니다.간단히 말해서 분류하려는 훈련 데이터의 미니 배치가 주어지면 다음과 같은 선택 사항이 있습니다. 

먼저 네트워크를 여러 GPU로 분할할 수 있습니다.즉, 각 GPU는 특정 계층으로 흐르는 데이터를 입력으로 받아 여러 후속 계층에서 데이터를 처리 한 다음 데이터를 다음 GPU로 보냅니다.이를 통해 단일 GPU가 처리할 수 있는 것과 비교할 때 더 큰 네트워크로 데이터를 처리할 수 있습니다.또한 GPU당 메모리 풋프린트를 잘 제어할 수 있습니다 (전체 네트워크 풋프린트의 일부). 

그러나 계층 (및 GPU) 간의 인터페이스에는 긴밀한 동기화가 필요합니다.이는 특히 컴퓨팅 워크로드가 계층 간에 적절하게 일치하지 않는 경우 까다로울 수 있습니다.이 문제는 많은 수의 GPU에서 더욱 악화됩니다.또한 레이어 간의 인터페이스에는 활성화 및 그래디언트와 같은 많은 양의 데이터 전송이 필요합니다.이로 인해 GPU 버스의 대역폭이 압도될 수 있습니다.또한 컴퓨팅 집약적이면서도 순차적 작업은 분할하기가 쉽지 않습니다.이와 관련하여 최선의 노력을 기울이려면 예를 들어 :cite:`Mirhoseini.Pham.Le.ea.2017`를 참조하십시오.여전히 어려운 문제이며 중요하지 않은 문제에 대해 좋은 (선형) 스케일링을 달성 할 수 있는지 여부는 확실하지 않습니다.여러 GPU를 함께 연결하기 위한 우수한 프레임워크 또는 운영 체제 지원이 없는 한 권장하지 않습니다. 

둘째, 작업을 레이어별로 분할할 수 있습니다.예를 들어 단일 GPU에서 64개의 채널을 계산하는 대신 4개의 GPU로 문제를 분할할 수 있습니다. 각 GPU는 16개 채널에 대한 데이터를 생성합니다.마찬가지로 완전히 연결된 계층의 경우 출력 단위 수를 나눌 수 있습니다. :numref:`fig_alexnet_original` (:cite:`Krizhevsky.Sutskever.Hinton.2012`에서 가져옴) 는 이 설계를 보여 주며, 이 전략은 메모리 공간이 매우 작은 GPU (당시 2GB) 를 처리하는 데 사용되었습니다.이렇게 하면 채널 (또는 단위) 수가 너무 작지 않은 경우 계산 측면에서 좋은 스케일링이 가능합니다.또한 사용 가능한 메모리가 선형으로 확장되므로 여러 GPU가 점점 더 큰 네트워크를 처리 할 수 있습니다. 

![Model parallelism in the original AlexNet design due to limited GPU memory.](../img/alexnet-original.svg)
:label:`fig_alexnet_original`

그러나 각 계층은 다른 모든 계층의 결과에 의존하므로*매우 많은* 수의 동기화 또는 장벽 작업이 필요합니다.또한 전송해야 하는 데이터의 양은 GPU에 레이어를 배포할 때보다 훨씬 더 많을 수 있습니다.따라서 대역폭 비용과 복잡성으로 인해 이 방법을 사용하지 않는 것이 좋습니다. 

마지막으로 여러 GPU에 데이터를 분할할 수 있었습니다.이렇게 하면 모든 GPU가 서로 다른 관측치에도 불구하고 동일한 유형의 작업을 수행합니다.그래디언트는 훈련 데이터의 각 미니배치 후에 GPU에서 집계됩니다.가장 간단한 방법이며 어떤 상황에서도 적용 할 수 있습니다.각 미니배치 이후에만 동기화하면 됩니다.즉, 다른 매개 변수가 아직 계산되는 동안 이미 그라디언트 매개 변수를 교환하는 것이 매우 바람직합니다.또한 GPU 수가 많을수록 미니 배치 크기가 커져 훈련 효율성이 향상됩니다.하지만 GPU를 더 추가한다고 해서 더 큰 모델을 훈련시킬 수는 없습니다. 

![Parallelization on multiple GPUs. From left to right: original problem, network partitioning, layerwise partitioning, data parallelism.](../img/splitting.svg)
:label:`fig_splitting`

여러 GPU에서 서로 다른 병렬화 방법을 비교한 내용은 :numref:`fig_splitting`에 나와 있습니다.대체로 메모리가 충분히 큰 GPU에 액세스 할 수 있다면 데이터 병렬 처리가 가장 편리한 방법입니다.분산 교육을 위한 파티셔닝에 대한 자세한 설명은 :cite:`Li.Andersen.Park.ea.2014`를 참조하십시오.GPU 메모리는 딥 러닝 초창기에 문제였습니다.지금까지 이 문제는 가장 흔하지 않은 경우를 제외한 모든 경우에 해결되었습니다.우리는 다음 내용에서 데이터 병렬화에 중점을 둡니다. 

## 데이터 병렬성

컴퓨터에 $k$개의 GPU가 있다고 가정합니다.훈련할 모델이 주어지면 GPU의 파라미터 값이 동일하고 동기화되지만 각 GPU는 모델 파라미터의 완전한 세트를 독립적으로 유지합니다.예를 들어, :numref:`fig_data_parallel`는 $k=2$일 때 데이터 병렬성을 사용한 훈련을 보여줍니다. 

![Calculation of minibatch stochastic gradient descent using data parallelism on two GPUs.](../img/data-parallel.svg)
:label:`fig_data_parallel`

일반적으로 교육은 다음과 같이 진행됩니다. 

* 무작위 미니 배치가 주어지면 훈련 반복에서 배치의 예제를 $k$ 부분으로 분할하고 GPU에 고르게 분배합니다.
* 각 GPU는 할당된 미니배치 서브셋을 기반으로 모델 파라미터의 손실과 기울기를 계산합니다.
* $k$ GPU 각각의 로컬 그래디언트는 현재 미니배치 확률적 그래디언트를 얻기 위해 집계된다.
* 집계 그래디언트가 각 GPU에 다시 분산됩니다.
* 각 GPU는 이 미니배치 확률적 그래디언트를 사용하여 유지 관리하는 모델 파라미터의 전체 세트를 업데이트합니다.

실제로 우리는 $k$ GPU에서 훈련할 때 미니배치 크기를 $k$배*증가*하여 각 GPU가 단일 GPU에서만 훈련하는 것과 동일한 양의 작업을 하도록 합니다.16GPU 서버에서는 미니 배치 크기가 상당히 커질 수 있으므로 그에 따라 학습 속도를 높여야 할 수도 있습니다.또한 :numref:`sec_batch_norm`의 배치 정규화는 예를 들어 GPU당 별도의 배치 정규화 계수를 유지하여 조정해야 합니다.다음에서는 장난감 네트워크를 사용하여 다중 GPU 훈련을 설명합니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

## [**장난감 네트워크**]

우리는 :numref:`sec_lenet`에 도입된 대로 LeNet을 사용합니다 (약간의 수정 포함).매개 변수 교환 및 동기화를 자세히 설명하기 위해 처음부터 정의합니다.

```{.python .input}
# Initialize model parameters
scale = 0.01
W1 = np.random.normal(scale=scale, size=(20, 1, 3, 3))
b1 = np.zeros(20)
W2 = np.random.normal(scale=scale, size=(50, 20, 5, 5))
b2 = np.zeros(50)
W3 = np.random.normal(scale=scale, size=(800, 128))
b3 = np.zeros(128)
W4 = np.random.normal(scale=scale, size=(128, 10))
b4 = np.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = npx.convolution(data=X, weight=params[0], bias=params[1],
                              kernel=(3, 3), num_filter=20)
    h1_activation = npx.relu(h1_conv)
    h1 = npx.pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2_conv = npx.convolution(data=h1, weight=params[2], bias=params[3],
                              kernel=(5, 5), num_filter=50)
    h2_activation = npx.relu(h2_conv)
    h2 = npx.pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = np.dot(h2, params[4]) + params[5]
    h3 = npx.relu(h3_linear)
    y_hat = np.dot(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
# Initialize model parameters
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = nn.CrossEntropyLoss(reduction='none')
```

## 데이터 동기화

효율적인 다중 GPU 트레이닝을 위해서는 두 가지 기본 연산이 필요합니다.먼저 [**여러 기기에 파라미터 목록을 배포**] 하고 그래디언트를 첨부할 수 있어야 합니다 (`get_params`).매개 변수가 없으면 GPU에서 네트워크를 평가할 수 없습니다.둘째, 여러 장치에서 매개 변수를 합산하는 기능이 필요합니다. 즉, `allreduce` 함수가 필요합니다.

```{.python .input}
def get_params(params, device):
    new_params = [p.copyto(device) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params
```

```{.python .input}
#@tab pytorch
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
```

모델 매개 변수를 하나의 GPU에 복사하여 사용해 보겠습니다.

```{.python .input}
#@tab all
new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)
```

아직 계산을 수행하지 않았으므로 바이어스 매개 변수에 대한 기울기는 여전히 0입니다.이제 벡터가 여러 GPU에 분산되어 있다고 가정해 보겠습니다.다음 [**`allreduce` 함수는 모든 벡터를 더하고 결과를 모든 GPU에 다시 브로드캐스트**] 합니다.이 기능이 작동하려면 데이터를 기기에 복사하여 결과를 누적해야 합니다.

```{.python .input}
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].ctx)
    for i in range(1, len(data)):
        data[0].copyto(data[i])
```

```{.python .input}
#@tab pytorch
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)
```

서로 다른 장치에서 서로 다른 값을 가진 벡터를 만들고 집계하여 테스트해 보겠습니다.

```{.python .input}
data = [np.ones((1, 2), ctx=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

```{.python .input}
#@tab pytorch
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

## 데이터 배포

[**여러 GPU에 걸쳐 미니배치를 고르게 분배**] 하려면 간단한 유틸리티 함수가 필요합니다.예를 들어 두 개의 GPU에서 데이터의 절반을 GPU 중 하나에 복사하려고 합니다.더 편리하고 간결하기 때문에 딥 러닝 프레임워크의 내장 함수를 사용하여 $4 \times 5$ 행렬에 대해 시험해 봅니다.

```{.python .input}
data = np.arange(20).reshape(4, 5)
devices = [npx.gpu(0), npx.gpu(1)]
split = gluon.utils.split_and_load(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

```{.python .input}
#@tab pytorch
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

나중에 재사용하기 위해 데이터와 레이블을 모두 분할하는 `split_batch` 함수를 정의합니다.

```{.python .input}
#@save
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, devices),
            gluon.utils.split_and_load(y, devices))
```

```{.python .input}
#@tab pytorch
#@save
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

## 트레이닝

이제 [**단일 미니배치에서 다중 GPU 훈련**] 을 구현할 수 있습니다.구현은 주로 이 섹션에서 설명하는 데이터 병렬화 접근 방식을 기반으로 합니다.방금 논의한 보조 기능인 `allreduce` 및 `split_and_load`를 사용하여 여러 GPU 간에 데이터를 동기화합니다.병렬 처리를 위해 특정 코드를 작성할 필요는 없습니다.계산 그래프는 미니배치 내의 장치 간에 종속성이 없으므로 병렬로*자동으로* 실행됩니다.

```{.python .input}
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    with autograd.record():  # Loss is calculated separately on each GPU
        ls = [loss(lenet(X_shard, device_W), y_shard)
              for X_shard, y_shard, device_W in zip(
                  X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    for i in range(len(device_params[0])):
        allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])  # Here, we use a full-size batch
```

```{.python .input}
#@tab pytorch
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # Loss is calculated separately on each GPU
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # Here, we use a full-size batch
```

이제 [**훈련 함수**] 를 정의할 수 있습니다.이전 장에서 사용한 것과 약간 다릅니다. GPU를 할당하고 모든 모델 매개 변수를 모든 장치에 복사해야합니다.분명히 각 배치는 `train_batch` 함수를 사용하여 처리되어 여러 GPU를 처리합니다.편의성 (및 코드의 간결성) 을 위해 단일 GPU에서 정확도를 계산하지만 다른 GPU는 유휴 상태이기 때문에*비효율적입니다*.

```{.python .input}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            npx.waitall()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

이것이 [**단일 GPU에서**] 얼마나 잘 작동하는지 살펴보겠습니다.먼저 배치 크기 256과 학습률 0.2를 사용합니다.

```{.python .input}
#@tab all
train(num_gpus=1, batch_size=256, lr=0.2)
```

배치 크기와 학습률을 변경하지 않고 [**GPU 수를 2**로 늘림**] 하면 테스트 정확도가 이전 실험과 거의 동일하게 유지된다는 것을 알 수 있습니다.최적화 알고리즘의 관점에서 보면 동일합니다.안타깝게도 여기서 얻을 수 있는 의미 있는 속도 향상은 없습니다: 모델이 너무 작습니다; 게다가 우리는 작은 데이터 세트만 가지고 있으며, 다중 GPU 트레이닝을 구현하는 데 약간 정교하지 않은 접근 방식은 상당한 파이썬 오버헤드로 어려움을 겪었습니다.앞으로 더 복잡한 모델과 더 정교한 병렬화 방법에 직면하게 될 것입니다.그럼에도 불구하고 패션-MNIST에게 어떤 일이 일어나는지 봅시다.

```{.python .input}
#@tab all
train(num_gpus=2, batch_size=256, lr=0.2)
```

## 요약

* 심층 네트워크 훈련을 여러 GPU로 분할하는 방법에는 여러 가지가 있습니다.계층 간, 계층 간 또는 데이터 간에 분할할 수 있습니다.이전 두 개는 긴밀하게 안무되는 데이터 전송이 필요합니다.데이터 병렬화가 가장 간단한 전략입니다.
* 데이터 병렬 훈련은 간단합니다.하지만 효율적인 미니배치 크기가 늘어납니다.
* 데이터 병렬 처리에서 데이터는 여러 GPU로 분할되며, 각 GPU는 자체 정방향 및 역방향 작업을 실행하고 그 후 그래디언트가 집계되고 결과가 GPU로 다시 브로드캐스트됩니다.
* 더 큰 미니 배치에는 약간 더 높은 학습률을 사용할 수 있습니다.

## 연습문제

1. $k$ GPU에서 훈련할 때는 미니배치 크기를 $b$에서 $k \cdot b$로 변경합니다. 즉, GPU 수를 기준으로 크기를 늘립니다.
1. 다양한 학습률에 대한 정확도를 비교합니다.GPU 수에 따라 어떻게 확장됩니까?
1. 서로 다른 GPU에서 서로 다른 매개 변수를 집계하는 보다 효율적인 `allreduce` 함수를 구현하시겠습니까?왜 더 효율적일까요?
1. 다중 GPU 테스트 정확도 계산을 구현합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/364)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1669)
:end_tab:
