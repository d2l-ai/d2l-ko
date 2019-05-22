# 다층 페셉트론(multilayer perceptron)을 처음부터 구현하기

다층 페셉트론(multilayer perceptron, MLP)가 이론적으로 어떻게 작동하는지 알게되었으니 직접 구현해보겠습니다. 우선 관련 패키지와 모듈을 import 합니다.

```{.python .input  n=9}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet import nd
from mxnet.gluon import loss as gloss
```

이전에 기본 softmax 회귀에서 얻은 결과와 비교하기 위해서 Fashion-MNIST 이미지 분류 데이터셋을 계속 사용합니다.

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 모델 파라미터 초기화하기

이 데이터셋은 10개의 클래스들을 갖으며 각 이미지는  $28 \times 28 = 784$  픽셀 값의 그리드로 구성되어 있다는 것을 기억해봅니다. (지금은) 공간적인 구조를 무시할 것이기 때문에, 이것을 $784$ 개의 입력 특성들과 $10$ 개의 클래스로 구성된 분류 데이터셋으로 간주하겠습니다. 특히 우리는 $256$개의 은닉 유닛들을 갖는 한 개의 은닉층으로 구성된 MLP를 구현 할 것입니다. 이 선택들은 검증 데이터의 결과에 따라서 다시 설정될 수 있는 *하이퍼파라미터* 로 간주할 수 있습니다. 일반적으로 우리는 메모리에 아주 잘 정렬되도록 층의 폭을 2의 제곱 수로 선택합니다.

다시 우리는 몇 개의 NDArray들을 할당해서 파라미터들을 표현합니다. 여기서 우리는 *층 별로* 한 개의 가중치 행렬과 한 개의 편향 벡터를 갖는다는 것을 주목하세요. 항상 그렇듯이 이들 파라미터들에 대한 그레이언트를 위한 메모리를 할당하기 위해서 `attach_grad` 를 호출해야 합니다.

```{.python .input  n=3}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

## 활성화 함수(activation function)

모든 것이 작동하는 것을 확인하기 위해서  `nd.relu` 를 직접 호출하는 대신 ReLU를 직접 구현하기 위해서 `maximum` 함수를 사용하겠습니다.

```{.python .input  n=4}
def relu(X):
    return nd.maximum(X, 0)
```

## 모델

Softmax 회귀에서 처럼, `reshape`를 호출해서 각 2차원 이미지를 `num_inputs` 길이의 평평한 벡터로 변환합니다. 마지막으로 우리는 단 몇 줄의 코드로 모델을 구현할 수 있습니다.

```{.python .input  n=5}
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2
```

## 손실 함수(loss function)

더 나은 수치 안정성을 위해서 그리고 [softmax 회귀를 처음부터 구현하는 방법](../chapter_linear-networks/softmax-regression-scratch) 을 이미 알기 때문에, Gluon의 내장 함수를 사용해서 softmax와 cross-entropy 손실을 계산하겠습니다.  [이전 절](mlp.md) 에서 이 복잡한 것들 중 일부를 논의했던 것을 기억해보세요. 흥미있는 독자는  `mxnet.gluon.loss.nnSoftmaxCrossEntropyLoss` 의 소스 코드를 살펴보면 더 자세한 내용을 알 수 있읍니다.

```{.python .input  n=6}
loss = gloss.SoftmaxCrossEntropyLoss()
```

## 학습

MLP를 학습시키는 단계는 softmax 회귀(regression) 학습과 같습니다. `g2l` 패키지에서 제공하는 `train_ch3` 함수를 직접 호출합니다. 이 함수의 구현은 [여기](softmax-regression-scratch.md) 를 참고하세요. 총 에포크(epoch) 수는 $10$으로 학습 속도(learning rate)는 $0.5$로 설정합니다.

```{.python .input  n=7}
num_epochs, lr = 10, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)
```

학습이 잘 되었는지 확인하기 위해서, 모델을 테스트 데이터에 적용해 보겠습니다. 이 모델의 성능이 궁금하다면, 동일한 분류를 수행하는 [linear 모델](softmax-regression-scratch.md) 의 결과와 비교해보세요.

```{.python .input}
for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [truelabel + '\n' + predlabel
          for truelabel, predlabel in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
```

이것은 이전 결과보다는 조금 더 좋아 보이니, 우리가 올바른 방향으로 가고 있다는 좋은 신호입니다.

## 요약

간단한 MLP를 직접 구현하는 것 조차도 쉽다는 것을 살펴봤습니다. 즉, 더 많은 층들을 사용하면, 이는 복잡해질 수 있습니다. (예를 들면, 모델 파라이터들의 이름을 정하는 것과 추적하는 것 등)

## 연습문제

1. `num_hiddens` 하이퍼파라미터를 변경해서 하이퍼파라미터가 결과에 어떤 영향을 주는지 확인해보세요.
1. 새로운 은닉층(hidden layer)를 추가해서 어떤 영향을 미치는지 확인해보세요.
1. 학습 속도(learning rate)를 변경하면 결과가 어떻게 되나요?
1. 모든 하이퍼파라미터(학습 속도(learing rate), 에포크(epoch) 회수, 은닉층(hidden layer) 개수, 각 층의 은닉 유닛(hidden unit) 개수)의 조합을 통해서 얻을 수 있는 가장 좋은 결과는 무엇인가요?

## QR 코드를 스캔해서 [논의하기](https://discuss.mxnet.io/t/2339)

![](../img/qr_mlp-scratch.svg)
