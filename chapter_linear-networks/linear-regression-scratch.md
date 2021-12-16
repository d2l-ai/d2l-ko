# 처음부터 선형 회귀 구현
:label:`sec_linear_scratch`

이제 선형 회귀의 핵심 아이디어를 이해했으므로 코드에서 직접 구현하는 작업을 시작할 수 있습니다.이 섹션에서는 (**데이터 파이프라인, 모델, 손실 함수, 미니배치 확률적 경사하강법 옵티마이저 등 전체 방법을 처음부터 구현하겠습니다.**) 최신 딥러닝 프레임워크가 거의 모든 작업을 자동화할 수 있지만, 처음부터 구현하는 것이 유일한 방법입니다.여러분이 무엇을 하고 있는지 확실히 알 수 있도록 하기 위해서입니다.또한 모델을 사용자 정의하고 자체 레이어 또는 손실 함수를 정의해야 할 때 내부에서 작동하는 방식을 이해하는 것이 편리할 것입니다.이 섹션에서는 텐서와 자동 차별화에만 의존합니다.그 후 딥 러닝 프레임워크의 종소리와 휘파람을 활용하여 보다 간결한 구현을 소개합니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## 데이터세트 생성

단순하게 유지하기 위해 [**가산 잡음이 있는 선형 모델에 따라 인공 데이터 세트를 구성합니다.**] 우리의 임무는 데이터 세트에 포함된 유한 예제 세트를 사용하여 이 모델의 매개 변수를 복구하는 것입니다.데이터를 쉽게 시각화할 수 있도록 저차원으로 유지할 것입니다.다음 코드 스니펫에서는 표준 정규분포에서 샘플링된 2개의 피처로 구성된 1000개의 예제가 포함된 데이터세트를 생성합니다.따라서 우리의 합성 데이터셋은 행렬 $\mathbf{X}\in \mathbb{R}^{1000 \times 2}$가 될 것입니다. 

(**데이터 세트를 생성하는 실제 매개 변수는 $\mathbf{w} = [2, -3.4]^\top$ 및 $b = 4.2$이며, **) 합성 레이블은 노이즈 용어 $\epsilon$을 사용하는 다음 선형 모델에 따라 할당됩니다. 

(**$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$달러**) 

$\epsilon$는 피처와 라벨에 잠재적인 측정 오류를 캡처하는 것으로 생각할 수 있습니다.표준 가정이 유지되고 따라서 $\epsilon$가 평균이 0인 정규 분포를 따른다고 가정합니다.문제를 쉽게 해결하기 위해 표준 편차를 0.01로 설정합니다.다음 코드는 합성 데이터세트를 생성합니다.

```{.python .input}
#@tab mxnet, pytorch
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))
```

```{.python .input}
#@tab tensorflow
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = d2l.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = d2l.reshape(y, (-1, 1))
    return X, y
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

[**`features`의 각 행은 2차원 데이터 예제로 구성되며 `labels`의 각 행은 1차원 레이블 값 (스칼라) 으로 구성됩니다.**]

```{.python .input}
#@tab all
print('features:', features[0],'\nlabel:', labels[0])
```

두 번째 특징 `features[:, 1]` 및 `labels`를 사용하여 산점도를 생성하면 둘 사이의 선형 상관 관계를 명확하게 관찰 할 수 있습니다.

```{.python .input}
#@tab all
d2l.set_figsize()
# The semicolon is for displaying the plot only
d2l.plt.scatter(d2l.numpy(features[:, 1]), d2l.numpy(labels), 1);
```

## 데이터세트 읽기

학습 모델은 데이터 세트를 여러 번 통과하고 한 번에 하나의 미니 배치를 가져 와서 모델을 업데이트하는 데 사용하는 것으로 구성됩니다.이 과정은 기계 학습 알고리즘을 훈련시키는 데 매우 중요하므로 데이터 세트를 섞고 미니 배치로 액세스하는 유틸리티 함수를 정의하는 것이 좋습니다. 

다음 코드에서는 [**`data_iter` 함수**](~~that~~) 를 정의하여 이 기능의 구현 가능한 한 가지를 보여줍니다.함수 (**배치 크기, 특징 행렬, 레이블 벡터를 취하여 크기가 `batch_size`.**인 미니배치를 산출함) 각 미니배치는 특징과 레이블의 튜플로 구성됩니다.

```{.python .input}
#@tab mxnet, pytorch
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = d2l.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

```{.python .input}
#@tab tensorflow
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)
```

일반적으로 우리는 연산을 병렬화하는 데 탁월한 GPU 하드웨어를 활용하기 위해 적당한 크기의 미니 배치를 사용하고 싶습니다.각 예제는 모델을 통해 병렬로 공급할 수 있고 각 예제에 대한 손실 함수의 기울기도 병렬로 취할 수 있기 때문에 GPU를 사용하면 단일 예제를 처리하는 데 걸리는 시간보다 훨씬 더 많은 시간 내에 수백 개의 예제를 처리 할 수 있습니다. 

직관을 구축하기 위해 데이터 예제의 첫 번째 작은 배치를 읽고 인쇄해 보겠습니다.각 미니배치의 피처 모양은 미니배치 크기와 입력 피처 수를 모두 알려줍니다.마찬가지로 라벨의 미니 배치는 `batch_size`에 의해 주어진 모양을 갖습니다.

```{.python .input}
#@tab all
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

반복을 실행할 때 전체 데이터셋이 모두 소진될 때까지 고유한 미니배치를 연속적으로 얻습니다 (이렇게 해 보십시오).위에서 구현한 반복은 교훈적인 목적으로는 좋지만 실제 문제에 어려움을 겪을 수 있는 방식에서는 비효율적입니다.예를 들어 메모리에 모든 데이터를 로드하고 많은 랜덤 메모리 액세스를 수행해야 합니다.딥러닝 프레임워크에 구현된 내장 반복기는 훨씬 더 효율적이며 파일에 저장된 데이터와 데이터 스트림을 통해 공급되는 데이터를 모두 처리할 수 있습니다. 

## 모델 매개변수 초기화

[**모델의 파라미터 최적화를 시작하기 전에**] 미니배치 확률적 경사하강법, (**우선 몇 가지 파라미터가 필요합니다.**) 다음 코드에서는 평균이 0이고 표준편차가 0.01인 정규분포에서 난수를 샘플링하여 가중치를 초기화합니다.바이어스를 0으로 설정합니다.

```{.python .input}
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)
```

매개 변수를 초기화한 후 다음 작업은 데이터가 충분히 잘 맞을 때까지 매개 변수를 업데이트하는 것입니다.업데이트 할 때마다 매개 변수와 관련하여 손실 함수의 기울기를 취해야합니다.이 기울기가 주어지면 손실을 줄일 수 있는 방향으로 각 파라미터를 업데이트할 수 있습니다. 

아무도 그라디언트를 명시적으로 계산하기를 원하지 않기 때문에 (지루하고 오류가 발생하기 쉽습니다) :numref:`sec_autograd`에 소개된 자동 미분을 사용하여 그래디언트를 계산합니다. 

## 모델 정의

다음으로, 입력과 파라미터를 출력에 연결하여 모델을 정의해야 합니다.**] 선형 모델의 출력을 계산하려면 입력 피처 $\mathbf{X}$와 모델 가중치 $\mathbf{w}$의 행렬-벡터 내적을 가져와 각 예제에 오프셋 $b$을 추가하기만 하면 됩니다.$\mathbf{Xw}$ 아래는 벡터이고 $b$은 스칼라입니다.:numref:`subsec_broadcasting`에 설명된 대로 방송 메커니즘을 호출합니다.벡터와 스칼라를 추가하면 벡터의 각 구성 요소에 스칼라가 추가됩니다.

```{.python .input}
#@tab all
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return d2l.matmul(X, w) + b
```

## 손실 함수 정의하기

[**모델을 업데이트하려면 손실 함수의 기울기를 가져와야하므로**](**손실 함수를 먼저 정의하십시오.**) 여기서는 :numref:`sec_linear_regression`에 설명된 대로 제곱 손실 함수를 사용합니다.구현에서는 실제 값 `y`을 예측된 값의 셰이프 `y_hat`로 변환해야 합니다.다음 함수에서 반환된 결과도 `y_hat`와 동일한 형태를 갖습니다.

```{.python .input}
#@tab all
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
```

## 최적화 알고리즘 정의

:numref:`sec_linear_regression`에서 논의한 것처럼 선형 회귀에는 폐쇄 형식 해가 있습니다.그러나 이 책은 선형 회귀에 관한 책이 아니라 딥 러닝에 관한 책입니다.이 책에서 소개하는 다른 모델은 분석적으로 해결할 수 없으므로 이번 기회에 미니 배치 확률 적 경사 하강의 첫 번째 작업 예를 소개하겠습니다.[~~선형 회귀 분석에는 폐쇄 형식 해가 있지만, 이 책의 다른 모델은 그렇지 않습니다.여기에서는 미니배치 스토캐스틱 그래디언트 하강을 소개합니다. ~~] 

각 단계에서 데이터 세트에서 무작위로 추출한 하나의 미니 배치를 사용하여 매개 변수에 대한 손실의 기울기를 추정합니다.다음으로 손실을 줄일 수 있는 방향으로 매개변수를 업데이트합니다.다음 코드에서는 파라미터 집합, 학습률 및 배치 크기가 지정된 경우 미니배치 확률적 경사하강법 업데이트를 적용합니다.업데이트 단계의 크기는 학습 속도 `lr`에 의해 결정됩니다.손실은 예제의 미니 배치에 대한 합계로 계산되므로, 일반적인 스텝 크기의 크기가 배치 크기 선택에 크게 좌우되지 않도록 단계 크기를 배치 크기 (`batch_size`) 로 정규화합니다.

```{.python .input}
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

```{.python .input}
#@tab pytorch
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)
```

## 트레이닝

이제 모든 부분이 준비되었으므로 [**기본 교육 루프를 구현할 준비가 되었습니다.**] 딥 러닝 경력 전반에 걸쳐 거의 동일한 훈련 루프를 반복해서 볼 수 있으므로 이 코드를 이해하는 것이 중요합니다. 

각 반복에서 훈련 예제의 미니 배치를 가져와 모델에 전달하여 일련의 예측을 얻습니다.손실을 계산 한 후 각 매개 변수에 대한 기울기를 저장하여 네트워크를 통과하는 역방향 통과를 시작합니다.마지막으로 최적화 알고리즘 `sgd`를 호출하여 모델 파라미터를 업데이트합니다. 

요약하면 다음 루프를 실행합니다. 

* 매개 변수를 초기화합니다. $(\mathbf{w}, b)$
* 완료될 때까지 반복
    * 컴퓨트 그래디언트 $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * 업데이트 매개 변수 $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

각*epoch*에서 훈련 데이터 세트의 모든 예제를 통과하면 전체 데이터 세트 (`data_iter` 함수 사용) 를 반복합니다 (예제 수를 배치 크기로 나눌 수 있다고 가정).시대의 수 `num_epochs`과 학습률 `lr`은 모두 하이퍼파라미터이며, 여기서는 각각 3과 0.03으로 설정합니다.안타깝게도 하이퍼파라미터 설정은 까다롭고 시행착오를 통해 약간의 조정이 필요합니다.현재로서는 이러한 세부 사항을 생략하지만 나중에 :numref:`chap_optimization`에서 수정합니다.

```{.python .input}
#@tab all
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
```

```{.python .input}
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Because `l` has a shape (`batch_size`, 1) and is not a scalar
        # variable, the elements in `l` are added together to obtain a new
        # variable, on which gradients with respect to [`w`, `b`] are computed
        l.backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab pytorch
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab tensorflow
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on l with respect to [`w`, `b`]
        dw, db = g.gradient(l, [w, b])
        # Update parameters using their gradient
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')
```

이 경우 데이터 세트를 직접 합성했기 때문에 실제 매개 변수가 무엇인지 정확하게 알 수 있습니다.따라서 훈련 루프를 통해 [**실제 매개 변수와 배운 매개 변수를 비교하여**] 훈련 성공 여부를 평가할 수 있습니다.실제로 그들은 서로 매우 가깝다는 것이 밝혀졌습니다.

```{.python .input}
#@tab all
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
print(f'error in estimating b: {true_b - b}')
```

매개 변수를 완벽하게 복구 할 수 있다는 것을 당연하게 생각해서는 안됩니다.그러나 기계 학습에서는 일반적으로 실제 기본 매개 변수를 복구하는 데 관심이 적고 매우 정확한 예측으로 이어지는 매개 변수에 더 관심이 있습니다.다행스럽게도 어려운 최적화 문제에서도 확률적 경사 하강법은 심층 네트워크의 경우 매우 정확한 예측으로 이어지는 많은 파라미터 구성이 존재하기 때문에 매우 좋은 해를 찾을 수 있습니다. 

## 요약

* 레이어나 고급 옵티마이저를 정의할 필요 없이 텐서와 자동 차별화를 사용하여 처음부터 심층 네트워크를 구현하고 최적화할 수 있는 방법을 살펴보았습니다.
* 이 섹션은 가능한 것의 표면만 긁습니다.다음 섹션에서는 방금 소개한 개념을 기반으로 추가 모델을 설명하고 이를 보다 간결하게 구현하는 방법을 학습합니다.

## 연습문제

1. 가중치를 0으로 초기화하면 어떻게 될까요?알고리즘이 여전히 작동할까요?
1. 전압과 전류 사이의 모델을 생각해 내려고 [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm)라고 가정합니다.자동 미분을 사용하여 모델의 파라미터를 학습할 수 있습니까?
1. [플랑크의 법칙](https://en.wikipedia.org/wiki/Planck%27s_law) 을 사용하여 스펙트럼 에너지 밀도를 사용하여 물체의 온도를 결정할 수 있습니까?
1. 2차 도함수를 계산하려는 경우 발생할 수 있는 문제는 무엇입니까?어떻게 고치겠어요?
1.  `squared_loss` 함수에서 `reshape` 함수가 필요한 이유는 무엇입니까?
1. 다양한 학습률을 사용하여 실험하여 손실 함수 값이 얼마나 빨리 떨어지는지 알아보십시오.
1. 예제 수를 배치 크기로 나눌 수 없는 경우 `data_iter` 함수의 동작은 어떻게 됩니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:
