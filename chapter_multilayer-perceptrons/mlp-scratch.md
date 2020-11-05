# 처음부터 다층 퍼셉트론의 구현
:label:`sec_mlp_scratch`

이제 우리는 다층 퍼셉트론 (MLP) 을 수학적으로 특징 지었으므로 우리 자신을 구현하려고 노력합시다.소프트맥스 회귀 (:numref:`sec_softmax_scratch`) 로 달성한 이전 결과와 비교하기 위해 패션 MNIST 이미지 분류 데이터 집합 (:numref:`sec_fashion_mnist`) 을 계속 사용할 것입니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 모델 매개변수 초기화

패션 MNIST에는 10 개의 클래스가 포함되어 있으며 각 이미지는 회색 음영 픽셀 값의 $28 \times 28 = 784$ 그리드로 구성됩니다.다시 말하지만, 우리는 지금 픽셀 사이의 공간 구조를 무시합니다, 그래서 우리는 784 입력 기능과 10 클래스와 단순히 분류 데이터 집합으로 생각할 수 있습니다.먼저 숨겨진 레이어와 256 개의 숨겨진 단위로 MLP를 구현합니다.우리는 하이퍼 매개 변수로 이러한 양을 모두 간주 할 수 있습니다.일반적으로 레이어 너비를 2의 거듭 제곱으로 선택합니다. 이 폭은 하드웨어에서 메모리가 할당되고 처리되는 방식 때문에 계산 효율적입니다.

다시 말하지만, 우리는 여러 텐서로 매개 변수를 나타낼 것입니다.*모든 레이어에 대해*, 우리는 하나의 가중치 행렬과 하나의 바이어스 벡터를 추적해야합니다.언제나처럼, 우리는이 매개 변수와 관련하여 손실의 그라디언트에 메모리를 할당합니다.

```{.python .input}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

```{.python .input}
#@tab tensorflow
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(
    shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(
    shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=.01))

params = [W1, b1, W2, b2]
```

## 활성화 기능

모든 것이 어떻게 작동하는지 알기 위해 내장 `relu` 함수를 직접 호출하지 않고 최대 함수를 사용하여 RelU 활성화를 구현할 것입니다.

```{.python .input}
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
#@tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
#@tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

## 모델

우리는 공간 구조를 무시하고 있기 때문에, 우리는 `reshape` 길이의 평면 벡터로 각 2 차원 이미지 `num_inputs`.마지막으로 몇 줄의 코드로 모델을 구현합니다.

```{.python .input}
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

```{.python .input}
#@tab pytorch
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(X@W1 + b1)  # Here '@' stands for matrix multiplication
    return (H@W2 + b2)
```

```{.python .input}
#@tab tensorflow
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2
```

## 손실 함수

수치 안정성을 보장하기 위해 이미 softmax 함수 (:numref:`sec_softmax_scratch`) 를 처음부터 구현했기 때문에 고급 API의 통합 함수를 사용하여 softmax 및 교차 엔트로피 손실을 계산합니다.:numref:`subsec_softmax-implementation-revisited`에서 이러한 복잡성에 대한 우리의 이전 토론을 상기하십시오.관심있는 독자는 손실 함수의 소스 코드를 검토하여 구현 세부 사항에 대한 지식을 심화하도록 권장합니다.

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)
```

## 교육

다행히 MLP에 대한 교육 루프는 softmax 회귀 분석과 정확히 동일합니다.`d2l` 패키지를 다시 활용하여 `train_ch3` 함수 (:numref:`sec_softmax_scratch` 참조) 를 호출하여 에포크 수를 10으로 설정하고 학습 속도를 0.5로 설정합니다.

```{.python .input}
num_epochs, lr = 10, 0.1
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 10, 0.1
updater = d2l.Updater([W1, W2, b1, b2], lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

학습 된 모델을 평가하기 위해 일부 테스트 데이터에 적용합니다.

```{.python .input}
#@tab all
d2l.predict_ch3(net, test_iter)
```

## 요약

* 우리는 간단한 MLP를 구현하는 것이 수동으로 수행 할 때에도 쉽다는 것을 보았습니다.
* 그러나 많은 수의 레이어가 있으면 처음부터 MLP를 구현하는 것이 여전히 지저분해질 수 있습니다 (예: 모델의 매개 변수 이름 지정 및 추적).

## 연습 문제

1. 하이퍼파라미터 `num_hiddens`의 값을 변경하고 이 하이퍼매개변수가 결과에 어떤 영향을 미치는지 확인합니다.이 하이퍼 매개 변수의 최상의 값을 결정하고 다른 모든 매개 변수를 일정하게 유지하십시오.
1. 숨겨진 레이어를 추가하여 결과에 어떤 영향을 미치는지 확인합니다.
1. 학습 속도를 변경하면 결과가 어떻게 달라지나요?모델 아키텍처 및 기타 하이퍼 매개 변수 (신기원 수 포함) 를 수정하면 어떤 학습 속도가 최상의 결과를 제공합니까?
1. 모든 하이퍼 매개 변수 (학습 속도, 신기원 수, 숨겨진 레이어 수, 레이어 당 숨겨진 단위 수) 를 공동으로 최적화하여 얻을 수있는 최상의 결과는 무엇입니까?
1. 여러 하이퍼 매개 변수를 처리하는 것이 훨씬 더 어려운 이유를 설명하십시오.
1. 여러 하이퍼 매개 변수에 대한 검색을 구조화하기위한 가장 현명한 전략은 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/227)
:end_tab:
