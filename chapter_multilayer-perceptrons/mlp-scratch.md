# 처음부터 멀티레이어 퍼셉트론 구현
:label:`sec_mlp_scratch`

이제 다층 퍼셉트론 (MLP) 을 수학적으로 특성화했으므로 직접 구현해 보겠습니다.소프트맥스 회귀 (:numref:`sec_softmax_scratch`) 로 달성한 이전 결과와 비교하기 위해 패션-MNIST 이미지 분류 데이터셋 (:numref:`sec_fashion_mnist`) 을 계속 사용할 것입니다.

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

패션-MNIST에는 10개의 클래스가 포함되어 있으며 각 이미지는 그레이스케일 픽셀 값으로 구성된 $28 \times 28 = 784$ 그리드로 구성되어 있습니다.다시 말하지만, 지금은 픽셀 간의 공간 구조를 무시할 것입니다. 따라서 784개의 입력 피처와 10개의 클래스가 있는 분류 데이터셋이라고 생각할 수 있습니다.우선, [**은닉 레이어 1개와 히든 유닛 256개가 있는 MLP를 구현하겠습니다.**] 이 두 수량을 모두 하이퍼파라미터로 간주할 수 있습니다.일반적으로 계층 너비를 2의 거듭제곱으로 선택하는데, 이는 하드웨어에서 메모리가 할당되고 처리되는 방식 때문에 계산적으로 효율적인 경향이 있습니다. 

다시 말하지만 매개 변수를 여러 텐서로 나타냅니다.*모든 레이어*에 대해 가중치 행렬 하나와 바이어스 벡터 하나를 추적해야 합니다.항상 그렇듯이 이러한 매개 변수와 관련하여 손실의 기울기에 대한 메모리를 할당합니다.

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

모든 것이 어떻게 작동하는지 확실히 알기 위해 내장 `relu` 함수를 직접 호출하는 대신 최대 함수를 사용하여 [**ReLU 활성화를 구현**] 할 것입니다.

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

공간 구조를 무시하고 있기 때문에 각 2 차원 이미지를 길이 `num_inputs`의 평면 벡터로 `reshape`합니다.마지막으로 몇 줄의 코드만으로 (**모델을 구현**) 합니다.

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

수치적 안정성을 보장하고 소프트맥스 함수를 처음부터 구현했기 때문에 (:numref:`sec_softmax_scratch`), 상위 수준 API의 통합 함수를 활용하여 소프트맥스 및 교차 엔트로피 손실을 계산합니다.:numref:`subsec_softmax-implementation-revisited`에서 이러한 복잡성에 대한 이전 논의를 상기하십시오.관심있는 독자는 구현 세부 사항에 대한 지식을 심화하기 위해 손실 함수의 소스 코드를 검토하는 것이 좋습니다.

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

## 트레이닝

다행스럽게도 [**MLP에 대한 훈련 루프는 소프트맥스 회귀와 정확히 동일합니다.**] `d2l` 패키지를 다시 활용하여 `train_ch3` 함수 (:numref:`sec_softmax_scratch` 참조) 를 호출하여 에포크 수를 10으로 설정하고 학습률을 0.1로 설정합니다.

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

학습된 모델을 평가하기 위해 [**일부 테스트 데이터에 적용**] 합니다.

```{.python .input}
#@tab all
d2l.predict_ch3(net, test_iter)
```

## 요약

* 간단한 MLP를 수동으로 구현하더라도 구현이 쉽다는 것을 알았습니다.
* 그러나 레이어가 많으면 MLP를 처음부터 구현하는 것이 여전히 복잡해질 수 있습니다 (예: 모델의 매개 변수 이름 지정 및 추적).

## 연습문제

1. 하이퍼파라미터 `num_hiddens`의 값을 변경하고 이 하이퍼파라미터가 결과에 어떤 영향을 미치는지 확인합니다.다른 모든 하이퍼파라미터는 일정하게 유지하면서 이 하이퍼파라미터의 최적 값을 결정합니다.
1. 숨겨진 레이어를 더 추가하여 결과에 어떤 영향을 미치는지 확인합니다.
1. 학습률을 변경하면 결과가 어떻게 달라지나요?모델 아키텍처 및 기타 하이퍼 파라미터 (epoch 수 포함) 를 수정하면 어떤 학습률이 최상의 결과를 얻을 수 있습니까?
1. 모든 하이퍼파라미터 (학습률, 에포크 수, 은닉 레이어 수, 레이어당 은닉 유닛 수) 를 공동으로 최적화하여 얻을 수 있는 최상의 결과는 무엇일까요?
1. 여러 하이퍼파라미터를 처리하는 것이 훨씬 더 어려운 이유를 설명하십시오.
1. 여러 초모수에 대한 검색을 구조화하기 위해 생각할 수 있는 가장 현명한 전략은 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/227)
:end_tab:
