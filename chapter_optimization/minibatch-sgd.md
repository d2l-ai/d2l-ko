# 미니배치 확률적 경사하강법
:label:`sec_minibatch_sgd`

지금까지 기울기 기반 학습에 대한 접근 방식에서 두 가지 극단이 발생했습니다. : :numref:`sec_gd`는 전체 데이터 세트를 사용하여 그라디언트를 계산하고 매개 변수를 한 번에 하나씩 업데이트합니다.반대로 :numref:`sec_sgd`는 한 번에 하나의 관측치를 처리하여 진전을 이룹니다.각각에는 고유 한 단점이 있습니다.경사 하강법은 데이터가 매우 유사할 때마다 특히*데이터 효율적*이 아닙니다.CPU 및 GPU는 벡터화의 모든 기능을 활용할 수 없기 때문에 확률적 경사하강법은 특히*계산적으로 효율적*이 아닙니다.이것은 행복한 매체가 있을지도 모른다는 것을 암시하며, 실제로 우리가 논의한 예제에서 지금까지 사용해온 것입니다. 

## 벡터화 및 캐시

미니배치 사용 결정의 핵심은 계산 효율성입니다.이는 여러 GPU 및 여러 서버에 대한 병렬화를 고려할 때 가장 쉽게 이해할 수 있습니다.이 경우 각 GPU에 적어도 하나의 이미지를 전송해야 합니다.서버당 8개의 GPU와 16개의 서버를 통해 우리는 이미 128개의 미니배치 크기에 도달했습니다. 

단일 GPU 또는 CPU의 경우 상황이 좀 더 미묘합니다.이러한 장치에는 여러 유형의 메모리가 있으며, 종종 여러 유형의 컴퓨팅 유닛과 서로 다른 대역폭 제약이 있습니다.예를 들어 CPU에는 레지스터 수가 적고 L1, L2가 있으며 경우에 따라 L3 캐시 (다른 프로세서 코어간에 공유됨) 도 있습니다.이러한 캐시는 크기와 지연 시간이 늘어나고 동시에 대역폭이 감소합니다.프로세서가 메인 메모리 인터페이스가 제공 할 수있는 것보다 더 많은 작업을 수행 할 수 있다고 말하면 충분합니다. 

* 16개의 코어와 AVX-512 벡터화를 갖춘 2GHz CPU는 초당 최대 $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$바이트를 처리할 수 있습니다.GPU의 성능은 이 수치를 100배까지 쉽게 초과합니다.반면 미드레인지 서버 프로세서는 100Gb/s 대역폭을 훨씬 넘지 않을 수 있습니다. 즉, 프로세서 공급을 유지하는 데 필요한 대역폭의 1/10보다 작을 수 있습니다.설상가상으로 모든 메모리 액세스가 동일하게 생성되는 것은 아닙니다. 첫째, 메모리 인터페이스는 일반적으로 64비트 너비 또는 더 넓습니다 (예: GPU에서 최대 384비트). 따라서 단일 바이트를 읽으면 훨씬 더 넓은 액세스 비용이 발생합니다.
* 첫 번째 액세스에는 상당한 오버헤드가 있지만 순차 액세스는 상대적으로 저렴합니다 (버스트 읽기라고도 함).여러 소켓, 칩렛 및 기타 구조가있을 때 캐싱하는 것과 같이 명심해야 할 사항이 더 많습니다.이에 대한 자세한 내용은 이 섹션의 범위를 벗어납니다.자세한 내용은 예를 들어 이 [Wikipedia article](https://en.wikipedia.org/wiki/Cache_hierarchy)를 참조하십시오.

이러한 제약 조건을 완화하는 방법은 실제로 프로세서에 데이터를 제공할 수 있을 만큼 빠른 CPU 캐시 계층 구조를 사용하는 것입니다.이것이 딥 러닝에서 일괄 처리의 원동력입니다.문제를 단순하게 유지하려면 행렬-행렬 곱셈 (예: $\mathbf{A} = \mathbf{B}\mathbf{C}$) 을 고려하십시오.$\mathbf{A}$를 계산하는 데는 여러 가지 옵션이 있습니다.예를 들어 다음을 시도해 볼 수 있습니다. 

1. $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}^\top$를 계산할 수 있습니다. 즉, 내적을 사용하여 요소별로 계산할 수 있습니다.
1. $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}^\top$을 계산할 수 있습니다. 즉, 한 번에 한 열씩 계산할 수 있습니다.마찬가지로 한 번에 한 행 $\mathbf{A}$를 계산할 수 있습니다.
1. 간단히 $\mathbf{A} = \mathbf{B} \mathbf{C}$를 계산할 수 있습니다.
1. $\mathbf{B}$과 $\mathbf{C}$를 더 작은 블록 행렬로 나누고 $\mathbf{A}$를 한 번에 한 블록씩 계산할 수 있습니다.

첫 번째 옵션을 따르는 경우 요소 $\mathbf{A}_{ij}$를 계산할 때마다 행 벡터와 열 벡터 하나를 CPU에 복사해야합니다.더 나쁜 것은 행렬 요소가 순차적으로 정렬되기 때문에 메모리에서 읽을 때 두 벡터 중 하나에 대해 많은 분리된 위치에 액세스해야 한다는 것입니다.두 번째 옵션은 훨씬 유리합니다.여기서는 $B$을 계속 통과하는 동안 열 벡터 $\mathbf{C}_{:,j}$를 CPU 캐시에 유지할 수 있습니다.이렇게 하면 메모리 대역폭 요구 사항이 절반으로 줄어들어 액세스 속도가 빨라집니다.물론 옵션 3이 가장 바람직합니다.안타깝게도 대부분의 행렬은 캐시에 완전히 맞지 않을 수 있습니다 (결국 논의 중입니다).그러나 옵션 4는 실제로 유용한 대안을 제공합니다. 매트릭스 블록을 캐시로 이동하고 로컬로 곱할 수 있습니다.최적화된 라이브러리가 이를 처리합니다.이러한 작업이 실제로 얼마나 효율적인지 살펴 보겠습니다. 

계산 효율성 외에도 Python과 딥 러닝 프레임워크 자체에서 발생하는 오버헤드는 상당합니다.명령을 실행할 때마다 파이썬 인터프리터는 MXNet 엔진에 명령을 보냅니다. MXNet 엔진은 명령을 계산 그래프에 삽입하고 스케줄링 중에 처리해야 합니다.이러한 오버 헤드는 상당히 해로울 수 있습니다.요컨대, 가능하면 벡터화 (및 행렬) 를 사용하는 것이 좋습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

timer = d2l.Timer()
A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np

timer = d2l.Timer()
A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

timer = d2l.Timer()
A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

요소별 할당은 $\mathbf{B}$과 $\mathbf{C}$의 모든 행과 열을 각각 반복하여 $\mathbf{A}$에 값을 할당합니다.

```{.python .input}
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

더 빠른 전략은 열 단위 할당을 수행하는 것입니다.

```{.python .input}
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

마지막으로 가장 효과적인 방법은 전체 작업을 한 블록에서 수행하는 것입니다.각 작업 속도가 무엇인지 살펴 보겠습니다.

```{.python .input}
# Compute A = BC in one go
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Compute A = BC in one go
timer.start()
A = torch.mm(B, C)
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## 미니 배치

:label:`sec_minibatches` 

과거에는 매개 변수를 업데이트하기 위해 단일 관측치가 아니라*미니 배치*의 데이터를 읽는 것이 당연했습니다.이제 이에 대한 간단한 정당성을 제시합니다.단일 관측값을 처리하려면 많은 단일 행렬-벡터 (또는 벡터-벡터) 곱셈을 수행해야 하는데, 이는 비용이 많이 들고 기본 딥러닝 프레임워크를 대신하여 상당한 오버헤드를 발생시킵니다.이는 데이터에 적용할 때 (추론이라고도 함) 네트워크를 평가하는 경우와 파라미터를 업데이트하기 위해 기울기를 계산할 때 모두 적용됩니다.즉, 이는 $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$를 수행 할 때마다 적용됩니다. 

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

이 연산을 한 번에 관측값의 미니 배치에 적용하여 이 연산의*계산적* 효율성을 높일 수 있습니다.즉, 단일 관측치에 대한 기울기 $\mathbf{g}_t$를 작은 배치에 대해 하나씩 바꿉니다. 

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

이것이 $\mathbf{g}_t$의 통계적 속성에 어떤 영향을 미치는지 살펴 보겠습니다. $\mathbf{x}_t$과 미니 배치 $\mathcal{B}_t$의 모든 요소가 훈련 세트에서 무작위로 균일하게 그려지기 때문에 기울기의 기대치는 변경되지 않습니다.반면에 분산은 크게 감소합니다.미니배치 기울기는 평균을 구하는 $b := |\mathcal{B}_t|$개의 독립 기울기로 구성되므로 표준 편차는 $b^{-\frac{1}{2}}$의 계수만큼 감소합니다.업데이트가 전체 그래디언트와 더 안정적으로 정렬된다는 것을 의미하므로 이는 그 자체로 좋은 일입니다. 

순진하게도 이것은 대형 미니 배치 $\mathcal{B}_t$를 선택하는 것이 보편적으로 바람직하다는 것을 나타냅니다.아아, 어느 시점 이후에는 계산 비용의 선형 증가와 비교할 때 표준 편차의 추가 감소가 최소화됩니다.실제로 우리는 GPU의 메모리에 적합하면서도 우수한 계산 효율성을 제공 할 수있을만큼 큰 미니 배치를 선택합니다.절감 효과를 설명하기 위해 몇 가지 코드를 살펴 보겠습니다.여기서 우리는 동일한 행렬-행렬 곱셈을 수행하지만 이번에는 한 번에 64 열의 “미니 배치”로 나뉩니다.

```{.python .input}
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

보시다시피, 미니 배치에서의 계산은 본질적으로 전체 행렬과 마찬가지로 효율적입니다.주의 사항이 정돈되어 있습니다.:numref:`sec_batch_norm`에서는 미니배치의 분산 양에 크게 의존하는 정규화 유형을 사용했습니다.후자를 늘리면 분산이 감소하고 배치 정규화로 인한 노이즈 주입의 이점이 있습니다.적절한 용어를 다시 스케일링하고 계산하는 방법에 대한 자세한 내용은 예를 들어 :cite:`Ioffe.2017`를 참조하십시오. 

## 데이터세트 읽기

데이터에서 미니 배치가 어떻게 효율적으로 생성되는지 살펴 보겠습니다.다음에서는 NASA에서 개발한 데이터 세트를 사용하여 날개 [noise from different aircraft](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise)을 테스트하여 이러한 최적화 알고리즘을 비교합니다.편의를 위해 첫 번째 $1,500$ 예제만 사용합니다.데이터는 전처리를 위해 희게 처리됩니다. 즉, 평균을 제거하고 분산을 좌표당 $1$로 다시 스케일링합니다.

```{.python .input}
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## 처음부터 구현

:numref:`sec_linear_scratch`의 미니배치 확률적 경사 하강 구현을 상기하십시오.다음에서는 좀 더 일반적인 구현을 제공합니다.편의를 위해 이 장의 뒷부분에 소개된 다른 최적화 알고리즘과 동일한 호출 시그니처를 갖습니다.특히 상태 입력 `states`을 추가하고 하이퍼매개 변수를 사전 `hyperparams`에 배치합니다.또한 훈련 함수에서 각 미니 배치 예제의 손실을 평균화하므로 최적화 알고리즘의 기울기를 배치 크기로 나눌 필요가 없습니다.

```{.python .input}
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

다음으로, 이 장의 뒷부분에서 소개하는 다른 최적화 알고리즘의 사용을 용이하게 하기 위해 일반 훈련 함수를 구현합니다.선형 회귀 모델을 초기화하고 미니배치 확률적 경사하강법 및 이후에 도입된 기타 알고리즘을 사용하여 모델을 훈련시키는 데 사용할 수 있습니다.

```{.python .input}
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Train
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

배치 기울기 하강법에 대한 최적화가 어떻게 진행되는지 살펴 보겠습니다.이는 미니배치 크기를 1500으로 설정하여 수행할 수 있습니다 (예: 총 예제 수).결과적으로 모델 매개변수는 epoch당 한 번만 업데이트됩니다.진전이 거의 없습니다.사실, 6 단계 후에 진행이 멈춘다.

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

배치 크기가 1이면 최적화를 위해 확률적 경사하강법을 사용합니다.구현을 단순화하기 위해 우리는 일정한 (작지만) 학습 속도를 선택했습니다.확률적 경사하강법에서는 예제가 처리될 때마다 모델 매개변수가 업데이트됩니다.이 경우 에포크당 1500회의 업데이트에 해당합니다.보시다시피, 목적 함수의 가치 하락은 한 시대 후에 느려집니다.두 절차 모두 한 시대 내에서 1500개의 예제를 처리했지만 확률적 경사하강법은 실험에서 기울기 하강보다 더 많은 시간을 소비합니다.확률적 경사하강법은 모수를 더 자주 업데이트하고 단일 관측값을 한 번에 하나씩 처리하는 것이 덜 효율적이기 때문입니다.

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

마지막으로 배치 크기가 100이면 최적화를 위해 미니배치 확률적 경사하강법을 사용합니다.Epoch당 필요한 시간은 확률적 경사하강법에 필요한 시간과 배치 경사하강법 시간보다 짧습니다.

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

배치 크기를 10으로 줄이면 각 배치의 작업 부하가 실행 효율성이 떨어지기 때문에 각 Epoch의 시간이 늘어납니다.

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

이제 이전 네 번의 실험에서 시간과 손실을 비교할 수 있습니다.보시다시피, 확률적 경사하강법은 처리된 예제의 수 측면에서 GD보다 빠르게 수렴하지만, 예제로 기울기 예제를 계산하는 것이 효율적이지 않기 때문에 GD보다 동일한 손실에 도달하는 데 더 많은 시간을 사용합니다.미니배치 확률적 경사하강법은 수렴 속도와 계산 효율성을 절충할 수 있습니다.미니배치 크기가 10이면 확률적 경사하강법보다 더 효율적입니다. 100인 미니배치 크기는 런타임 측면에서 GD를 훨씬 능가합니다.

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## 간결한 구현

글루온에서는 `Trainer` 클래스를 사용하여 최적화 알고리즘을 호출할 수 있습니다.이는 일반 훈련 함수를 구현하는 데 사용됩니다.현재 챕터 전체에서 이 기능을 사용할 것입니다.

```{.python .input}
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)

    # Note: `MSELoss` computes squared error without the 1/2 factor
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    # Note: `MeanSquaredError` computes squared error without the 1/2 factor
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                r = (d2l.evaluate_loss(net, data_iter, loss),)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

Gluon을 사용하여 마지막 실험을 반복하면 동일한 동작이 나타납니다.

```{.python .input}
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.05}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## 요약

* 벡터화는 딥 러닝 프레임워크에서 발생하는 오버헤드가 줄어들고 CPU 및 GPU의 메모리 위치 및 캐싱이 향상되어 코드의 효율성이 향상됩니다.
* 확률적 기울기 하강법에서 발생하는 통계적 효율성과 한 번에 대량의 데이터를 처리하여 발생하는 계산 효율성 사이에는 절충점이 있습니다.
* 미니배치 확률적 경사하강법은 계산 효율과 통계적 효율성이라는 두 가지 장점을 모두 제공합니다.
* 미니배치 확률적 기울기 하강에서는 훈련 데이터의 무작위 순열에 의해 얻어진 데이터의 배치를 처리합니다 (즉, 각 관측치는 무작위 순서임에도 불구하고 에포크당 한 번만 처리됩니다).
* 훈련 중에 학습률을 낮추는 것이 좋습니다.
* 일반적으로 미니배치 확률적 경사하강법은 클록 시간으로 측정할 때 더 작은 위험으로 수렴하는 확률적 경사하강법 및 경사하강법보다 빠릅니다.

## 연습문제

1. 배치 크기 및 학습률을 수정하고 목적 함수의 값에 대한 감소율과 각 Epoch에서 소비되는 시간을 관찰합니다.
1. MXNet 설명서를 읽고 `Trainer` 클래스 `set_learning_rate` 함수를 사용하여 미니배치 확률적 기울기 하강법의 학습 속도를 각 에포크 이후 이전 값의 1/10으로 줄입니다.
1. 미니배치 확률적 경사하강법 (확률적 경사하강법) 을 훈련 세트에서 실제로*대체 샘플링*하는 변형과 비교합니다.어떻게 되나요?
1. 사악한 요정은 말하지 않고 데이터 세트를 복제합니다 (예: 각 관측치가 두 번 발생하고 데이터 세트가 원래 크기의 두 배로 증가하지만 아무도 알려주지 않음).확률적 경사하강법, 미니배치 확률적 경사하강법 및 경사하강법의 동작은 어떻게 변하는가?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1069)
:end_tab:
