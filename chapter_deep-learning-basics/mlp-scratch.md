# 다층 퍼셉트론(multilayer perceptron)을 처음부터 구현하기

다층 퍼셉트론(multilayer perceptron, MLP)가 어떻게 작동하는지 이론적으로 배웠으니, 직접 구현해보겠습니다. 우선 관련 패키지와 모듈을 import 합니다.

```{.python .input  n=9}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet import nd
from mxnet.gluon import loss as gloss
```

이 예제에서도 Fashion-MNIST 데이터셋을 사용해서, 이미지를 분류하는데 다층 퍼셉트론(multilayer perceptron)을 사용하겠습니다.

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 모델 파라미터 초기화하기

이 데이터셋은 10개의 클래스로 구분되어 있고, 각 이미지는  $28 \times 28 = 784$ 픽셀의 해상도를 가지고 있습니다. 따라서, 입력은 784개이고, 출력은 10개가 됩니다. 우리는 한 개의 은닉층(hidden layer)을 갖는 MLP를 만들어보겠는데, 이 은닉층(hidden layer)은 256개의 은닉 유닛(hidden unit)을 갖도록 하겠습니다. 만약 원한다면 하이퍼파라미터인 은닉 유닛(hidden unit) 개수를 다르게 설정할 수도 있습니다. 일반적으로, 유닛(unit) 의 개수는 메모리에 잘 배치될 수 있도록 2의 지수승의 숫자로 선택합니다.

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

여기서 `ReLU` 를 직접 호출하는 대신, ReLU를 `maximum` 함수를 이용해서 정의합니다.

```{.python .input  n=4}
def relu(X):
    return nd.maximum(X, 0)
```

## 모델

Softmax 회귀(regression)에서 그랬던 것처럼, `reshape` 함수를 이용해서 원래 이미지를 `num_inputs` 크기의 벡터로 변환한 다음에 앞에서 설명한 대로 MLP를 구현합니다.

```{.python .input  n=5}
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2
```

## 손실 함수(loss function)

더 나은 계산의 안정성을 위해서, softmax 계산과 크로스-엔트로피 손실(cross-entropy loss) 계산은 Gluon 함수를 이용하겠습니다. 왜 그런지는 [앞 절](mlp.md)에서 이 함수의 구현에 대한 복잡성을 이야기했으니 참고 바랍니다. Gluon 함수를 이용하면 코드를 안전하게 구현하기 위해서 신경을 써야하는 많은 세밀한 것들을 간단하게 피할 수 있습니다. (자세한 내용이 궁금하다면 소스 코드를 살펴보기 바랍니다. 소스 코드을 보면 다른 관련 함수를 구현하는데 유용한 것들을 배울 수도 있습니다.)

```{.python .input  n=6}
loss = gloss.SoftmaxCrossEntropyLoss()
```

## 학습

다층 퍼셉트론(multilayer perceptron)을 학습시키는 단계는 softmax 회귀(regression) 학습과 같습니다. `g2l` 패키지에서 제공하는 `train_ch3` 함수를 직접 호출합니다. 이 함수의 구현은 [여기](softmax-regression-scratch.md) 를 참고하세요. 총 에포크(epoch) 수는 10으로 학습 속도(learning rate)는 0.5로 설정합니다.

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

이전보다 조금 성능이 좋아 보이는 것으로 보아 MLP를 사용하는 것이 좋은 것임을 알 수 있습니다.

## 요약

간단한 MLP는 직접 구현하는 것이 아주 쉽다는 것을 확인했습니다. 하지만, 많은 수의 층을 갖는 경우에는 굉장히 복잡해질 수 있습니다. (예를 들면 모델 파라미터 이름을 정하는 것 등)

## 문제

1. `num_hiddens` 하이퍼파라미터를 변경해서 결과가 어떻게 영향을 받는지 확인해보세요.
1. 새로운 은닉층(hidden layer)를 추가해서 어떤 영향을 미치는지 확인해보세요.
1. 학습 속도(learning rate)를 변경하면 결과가 어떻게 되나요?
1. 모든 하이퍼파라미터(학습 속도(learing rate), 에포크(epoch) 수, 은닉층(hidden layer) 개수, 각 층의 은닉 유닛(hidden unit) 개수)의 조합을 통해서 얻을 수 있는 가장 좋은 결과는 무엇인가요?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2339)

![](../img/qr_mlp-scratch.svg)
