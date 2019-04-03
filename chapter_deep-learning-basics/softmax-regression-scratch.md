# Softmax 회귀를 처음부터 구현하기

선형 회귀를 직접 구현해본 것처럼, softmax regression도 직접 구현해보는 것이 도움이 될 것입니다. 이 후에, 같은 내용을 Gluon을 사용해서 구현하면서 비교를 해보겠습니다. 필요한 패키지와 모듈을 import 하는 것으로 시작합니다.

```{.python .input}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet import autograd, nd
```

Fashion-MNIST 데이터 셋을 사용하고, 배치 크기는 256으로 하겠습니다.

```{.python .input}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 모델 파라메터 초기화하기

선형 회귀처럼 샘플들을 벡터로 표현합니다. 각 예제가 $28 \times 28$ 픽셀의 이미지이기 때문에 784 차원의 벡터에 저장합니다. 그리고 10개의 카테고리가 있으니 단일 레이어를 갖는 네트워크의 output 차원은 10으로 정의합니다. 이렇게 하면, softmax regression의 weight와 bias 파라매터들은 각각 크기가  $784 \times 10$,  $1 \times 10$ 인 행렬이 됩니다.  $W$ 를 가우시안 노이즈를 이용해서 초기화합니다.

```{.python .input  n=9}
num_inputs = 784
num_outputs = 10

W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)
```

이전처럼 모델 파라메터에 gradient를 붙이겠습니다.

```{.python .input  n=10}
W.attach_grad()
b.attach_grad()
```

## The Softmax

softmax regression을 정의하기에 앞서, `sum` 과 같은 연산이 NDArray의 특정 차원에서 어떻게 동작하는지를 보도록 하겠습니다. 행렬 `x` 의 같은 열 (`asix=0`) 또는 같은 행 (`axis=1`)의 값들을 모두 더할 수 있습니다. 합을 수행한 후 결과의 차원수를 줄이지 않고 그대로 유지하는 것도 가능합니다. 이는 위해서 `keepdims=True` 파라메터 값을 설정하면 됩니다.

```{.python .input  n=11}
X = nd.array([[1, 2, 3], [4, 5, 6]])
X.sum(axis=0, keepdims=True), X.sum(axis=1, keepdims=True)
```

자 이제 우리는 softmax 함수를 정의할 준비가 되었습니다. 우선 각 항에 `exp` 를 적용해서 지수값을 구하고, 정규화 상수(normalization constant)를 구하기 위해서 각 행의 값들을 모두 더합니다. 각 행을 정규화 상수(normalization contatnt)로 나누고 그 결과를 리턴합니다. 코드를 보기 전에 수식을 먼저 보겠습니다.
$$
\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(X_{ij})}{\sum_k \exp(X_{ik})}
$$

분모는 partition 함수라고 불리기도 합니다. 이 이름은 파티클의 앙상블에 대한 분포를 모델링하는 [통계 물리](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))에서 기원합니다.  [Naive Bayes](../chapter_crashcourse/naive-bayes.md)에서 그랬던 것처럼 행렬의 항목들이 너무 크거나 작아서 생기는,  숫자가 너무 커지는 overflow나 너무 작아지는 underflow를 고려하지 않고 함수를 구현하겠습니다.

```{.python .input  n=12}
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition  # The broadcast mechanism is applied here
```

보는 것처럼, 임의의 난수 입력에 대해서, 각 항목을 0 또는 양의 숫자로 변환합니다. 또한, 확률에서 요구하는 것처럼 각 행의 합은 1이 됩니다.

```{.python .input  n=13}
X = nd.random.normal(shape=(2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(axis=1)
```

## 모델

Softmax 연산을 이용해서 softmax regresssion 모델을 정의하겠습니다. `reshape` 함수를 이용해서 원본 이미지를 길이가 `num inputs` 인 벡터로 변환합니다.

```{.python .input  n=14}
def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)
```

## Loss 함수

앞 절에서 softmax regression에서 사용하는 cross-entropy loss 함수를 소개했습니다. 이는 모든 딥러닝에서 등장하는 loss 함수들 중에 가장 일반적인 loss 함수입니다. 이유는 regression 문제보다는 분류 문제가 더 많기 때문입니다.

cross-entropy의 계산은 label의 예측된 확률값을 얻고, 이 값에 logirithm  $-\log p(y|x)$ 을 적용하는 것임을 기억해두세요. Python의 `for` loop을 사용하지 않고 (비효율적임), softmax를 적용한 행렬에서 적당한 항목을 뽑아주는 `pick` 함수를 이용하겠습니다. 3개의 카테고리와 2개의 샘플의 경우 아래와 같이 구할 수 있습니다.

```{.python .input  n=15}
y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = nd.array([0, 2], dtype='int32')
nd.pick(y_hat, y)
```

이를 이용해서 cross-entropy loss 함수를 다음과 같이 정의합니다.

```{.python .input  n=16}
def cross_entropy(y_hat, y):
    return - nd.pick(y_hat, y).log()
```

## 분류 정확도

예측된 확률 분포들  `y_hat`  주어졌을 때, 가장 높은 예측 확률을 갖는 것을 결과 카테고리로 사용합니다. 실제 카테고리 `y` 과 일치하는 경우, 이 예측은 '정확하다'라고 합니다. 분류 정확도는 정확한 예측들과 전체 예측 수의 비율로 정의됩니다.

정확도를 계산하기 위해서 `accuracy` 함수를 다음과 같이 정의합니다. `y_hat.argmax(axis=1)` 는 행렬 `y_hat` 에서 가장 큰 원소의 인덱스를 리턴하고, 그 결과의 shape은  `y` 변수의 shape과 동일합니다. 이제 해야 할 일은 두개가 일치하는지 확인하는 것입니다. 동등 연산자 `==` 는 데이터 타입에 민감하기 때문에, 두개를 동일한 타입으로 바꿔야합니다. `float32` 로 하겠습니다. 결과는 각 항목이 거짓일 경우 0, 참일 경우 1의 값을 갖는 NDArray가 됩니다. 이에 대한 평균을 계산하면 원하는 결과를 얻을 수 있습니다.

```{.python .input  n=17}
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
```

예측된 확률 분표와 label에 대한 변수로 `pick` 함수에서 정의했던  `y_hat` 과 `y` 를 계속 사용하겠습니다. 첫번째 샘플의 예측 카테고리는 2 (첫번째 행에서 가장 큰 값은 0.6이고 이 값의 인덱스는 2)임을 확인할 수 있고, 이는 실제 label 0과 일치하지 않습니다. 두번째 샘플의 예측 카테고리는 2 (두번째 행에서 가장 큰 값이 0.5이고 이값의 인덱스는 2)이고, 이는 실제 label 2와 일치합니다. 따라서, 이 두 예들에 대한 분류 정확도 0.5 입니다.

```{.python .input  n=18}
accuracy(y_hat, y)
```

마찬가지로, `data_iter` 로 주어지는 데이터셋에 대한 모델 `net` 결과에 대한 정확도를 평가해볼 수 있습니다.

```{.python .input  n=19}
# The function will be gradually improved: the complete implementation will be
# discussed in the "Image Augmentation" section
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n
```

이 모델 `net` 은 난수 값으로 weight 값들이 초기화되어 있기 때문에, 정확도는 임의로 추측하는 것과 유사한 0.1 (10개의 클래스)로 나올 것입니다.

```{.python .input  n=20}
evaluate_accuracy(test_iter, net)
```

## 모델 학습

softmax regression 학습은 선형 회귀 학습과 아주 유사합니다. 모델의 loss 함수를 최적화하기 위해서 미니 배치 stochastic gradient descent를 이용합니다. 모델 학습에서 `num_epochs` epoch 횟수와 `lr` learning rate는 모두 바꿀 수 있는 hyper-parameter 입니다. 이 값을 바꾸면서, 모델의 분류 정확도를 높일 수 있습니다.

```{.python .input  n=21}
num_epochs, lr = 5, 0.1

# This function has been saved in the d2l package for future use
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                # This will be illustrated in the next section
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs,
          batch_size, [W, b], lr)
```

## 예측

학습이 완료되었으면, 모델을 이용해서 이미지를 분류 해보겠습니다. 이미지들이 주어졌을 때, 실제 label들 (텍스트 결과의 첫번째 줄)과 모델 예측 (텍스트 결과의 두번째 줄)를 비교 해보세요.

```{.python .input}
for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [truelabel + '\n' + predlabel
          for truelabel, predlabel in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
```

## 요약

softmax regression을 이용해서 다중 카테고리 분류를 할 수 있습니다. 학습은 선형 회귀와 비슷하게 수행됩니다: 데이터를 획득하고, 읽고, 모델과 loss 함수를 정의한 후, 최적화 알고리즘을 이용해서 모델을 학습시킵니다. 사실은 거의 모든 딥러닝 모델의 학습 절차는 이와 비슷합니다.

## 문제

1. 이 절에서 softmax 연산의 수학적인 정의에 따라 softmax 함수를 직접 정의해봤습니다. 이 경우 어떤 문제가 발생할 수 있을까요? (힌트 - exp(50)의 크기를 계산해보세요)
1. 이 절의 `cross_entropy` 함수 cross-entropy loss 함수의 정의를 따라서 구현되었습니다. 이 구현에 어떤 문제가 있을까요? (힌트 - logarithm의 도메인을 고려해보세요)
1. 위 두가지 문제를 어떻게 해결할 수 있는지 생각해보세요
1. 가장 유사한 label을 리턴하는 것이 항상 좋은 아이디어일까요? 예를 들면, 의료 진단에서 그렇게 하겠나요?
1. 어떤 feature들을 기반으로 다음 단어를 예측하기 위해서 softmax regression을 사용하기를 원한다고 가정하겠습니다. 단어 수가 많은 경우 어떤 문제가 있을까요?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2336)

![](../img/qr_softmax-regression-scratch.svg)
