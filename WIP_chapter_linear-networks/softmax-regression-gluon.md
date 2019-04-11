# Softmax 회귀(regression)의 간결한 구현

우리는 이미 [선형 회귀 구현](linear-regression-gluon.md)에서 Gluon을 이용하는 것이 아주 편리하다는 것을 확인했습니다. 이제 Gluon이 분류에 어떻게 적용되는지 보도록 하겠습니다. 역시 몇 가지 패키지와 모듈을 import하는 것으로 시작합니다.

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
```

앞 절과 동일하게 Fashion-MNIST 데이터셋과 같은 배치 크기를 사용합니다.

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 모델 파라미터 초기화하기

[앞 절](softmax-regression.md)에서 언급했듯이 softmax regression의 output 레이어는 fully connected 레이어입니다. 따라서, 10개의 output을 갖는 fully connected 레이어를 추가하고, weight을 평균이 0이고 표준 편차가 0.01인 분포에서 난수를 뽑아서 초기화를 합니다.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

## The Softmax

이전 예제에서는 모델의 결과를 계산하고, 이 결과에 크로스-엔트로피 손실(cross-entropy loss)을 적용 했었습니다. 이를 위해서  `-nd.pick(y_hat, y).log()` 를 이용했습니다. 수학적으로는 이렇게 하는 것이 매우 논리적입니다. 하지만, 이미 수차례 언급하였듯이 연산의 관점에서 보면 어려운 문제가 될 수 있습니다.  (예를 들면,  [Naive Bayes](../chapter_crashcourse/naive-bayes.md) 의 예 또는 이전 장의 문제들 처럼).  ``yhat`` 의 j 번째 원소를  $\hat y_j$ 라고 하고, 입력인 `y_linear` 변수의 j 번째 원소를  $z_j$ 라고 할때, softmax 함수는  $\hat y_j = \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}​$ 를 계산합니다.

만약 몇개의  $z_i$ 가 매우 큰 값을 갖는 다면,  $e^{z_i}$ 값이 `float` 변수가 표현할 수 있는 값보다 훨씬 커질 수 있습니다(overflow). 따라서, 분모 (또는 분자)가 `inf` 가 돼서 결과  $\hat y_j$가 0, `inf` 또는 `nan` 가 될 수 있습니다. 어떤 경우에든지 `cross_entropy` 는 잘 정의된 값을 리턴하지 못할 것입니다. 이런 문제 때문에, `softmax` 함수에서는 모든 $z_i$ 에서  $\text{max}(z_i)$ 뺍니다. 이렇게  $z_i$ 를 이동시키는 것이 `softmax` 의 리턴값을 변화시키지 않는다는 사실을 확인해볼 수도 있습니다.

위에서 설명한 빼기와 정규화(normalization) 단계를 거친 후에도 $z_j$ 가 매우 작은 음수값이 될 가능성이 여전히 있습니다. 따라서,  $e^{z_j}$ 가 0과 매우 근접해지거나 유한한 프리시전(finite precision)(즉, underflow) 때문에 0으로 반올림될 수도 있습니다. 이렇게 되면,  $\hat y_j$ 는 0이 되고, $\text{log}(\hat y_j)$ 는 `-inf` 가 됩니다. 역전파(backpropagation)를 몇 번 거치면, 화면에 not-a-number (`nan`) 결과가 출력되는 것을 보게 될 것입니다.

이 문제에 대한 해결책은 지수 함수를 계산함에도 불구하고, 크로스-엔트로피(cross-entropy) 함수에서 이 값에 대한 로그(log) 값을 취하도록 합니다. 이 두 연산 `softmax`  와 `cross_entropy` 를 함께 사용해서 수치 안정성 문제를 해결하고, 역전파(backpropagation) 과정에서 위 문제를 만나지 않게 할 수 있습니다. 아래 공식에서 보이는 것처럼, $\log(\exp(\cdot))$ 를 사용해서  $e^{z_j}$ 를 계산하는 것을 피하고,  $z_j$ 를 직접 사용할 수 있습니다.
$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}\right) \\
& = \log{(e^{z_j})}-\text{log}{\left( \sum_{i=1}^{n} e^{z_i} \right)} \\
& = z_j -\log{\left( \sum_{i=1}^{n} e^{z_i} \right)}
\end{aligned}
$$

모델의 확률 결과에 대한 평가를 해야하는 경우에 사용되는 이런 전형적인 softmax 함수를 간편하게 만들고 싶습니다. 하지만, softmax 확률들을 새로운 손실 함수(loss function)에 대입하는 것 보다는,  $\hat{y}$ 만 전달하면, softmax와 log 값들이 모두 softmax_cross_entropy 손실 함수(loss function)에서 계산되도록 하겠습니다. 이는 log-sum-exp 트릭  ([see on Wikipedia](https://en.wikipedia.org/wiki/LogSumExp)) 과 같이 스마트한 것을 하는 것과 같다고 볼 수 있습니다.

```{.python .input  n=4}
loss = gloss.SoftmaxCrossEntropyLoss()
```

## 최적화 알고리즘

최적화 알고리즘으로 학습 속도(learning rate)를 0.1로 하는 미니 배치 확률적 경사 하강법(stochastic gradient descent)를 사용하겠습니다. 이는 선형 회귀에서도 동일하게 사용했는데, 옵티마이져(optimizer)들의 이식성 보여줍니다.

```{.python .input  n=5}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## 학습

다음으로, 앞 절에서 정의된 학습 함수를 이용해서 모델을 학습 시킵니다.

```{.python .input  n=6}
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
```

이전과 같이 이 알고리즘은 매우 쓸만한 정확도인 83.7%로 수렴하는데, 이전보다 훨씬 더 적은 코드를 가지고 가능합니다. Gluon은 단순하게 구현할 경우 만날 수 있는 수치 안정성을 넘어서 특별한 예방을 포함하고 있기에, 모델을 직접 구현할 때 만날 수 있는 많은 일반적인 위험을 피할 수 있게 해줍니다.

## 문제

1. 배치 크기(batch size), 에포크(epoch), 학습 속도(learning rate)와 같은 하이퍼파라미터(hyper-parameter)를 변경하면서 어떤 결과가 나오는지 보세요.
1. 학습이 진행될 때 어느 정도 지나면 테스트 정확도가 감소할까요? 어떻게 고칠 수 있나요?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2337)

![](../img/qr_softmax-regression-gluon.svg)
