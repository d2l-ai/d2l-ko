# Softmax 회귀 처음부터 구현하기

선형 회귀를 직접 구현한 것처럼 다중 클래스 로지스틱 (또는 softmax) 회귀도 비슷하게 기본적이고, 여러분이 이것을 처음부터 어떻게 구현하는지를 상세하게 알아야 한다고 믿습니다. 선형 회귀처럼, 이것들을 직접 만들어 본 후에는 Gluon을 이용한 구현을 쉽게 해보면서 비교하겠습니다. 시작에 앞서, 우리가 사용할 패키지를 import 합니다. (모든 것을 직접 구현할 것이기 때문에, `autograd` 와 `nd` 만 import 합니다.) 

```{.python .input}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet import autograd, nd
```

이전 절에서 소개한 Fashion-MNIST 데이터셋을 사용하겠고, 배치 크기 256로 이터레이터(iterator)를 구성합니다.

```{.python .input}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 모델 파라미터 초기화하기

선형 회귀처럼 샘플들을 벡터로 표현합니다. 각 샘플들이 $28 \times 28$ 이미지이기 때문에, 우리는 각 샘플을 펼쳐서(flatten), $784$ 차원의 벡터로 다루겠습니다. 나중에는 이미지들의 공간적 구조를 이용하는 더 복잡한 전략들에 대해서 이야기 하겠지만, 지금은 각 픽셀 위치를 단지 하나의 특성으로 취급하겠습니다.

softmax 회귀의 경우 카테고리 개수만큼의 결과들을 갖는다는 것을 기억해 보세요. 이 데이터셋은 $10$개의 카테고리를 갖기 때문에, 우리의 네트워크는 차원이 $10$ 인 결과를 갖습니다. 따라서, 가중치는 $784 \times 10$ 행렬로, 그리고 편향은  a $1 \times 10$ 벡터로 구성됩니다. 선형 회귀에서 처럼 가중치 $W$ 는 가우시안 노이즈를 사용해서 초기화하고 편향은 모두 $0$ 으로 설정합니다.

```{.python .input  n=9}
num_inputs = 784
num_outputs = 10

W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)
```

이전처럼 모델 파라미터에 그래디언트(gradient)를 붙이겠습니다.

```{.python .input  n=10}
W.attach_grad()
b.attach_grad()
```

## Softmax

softmax 회귀(regression)을 구현하기에 앞서, `sum` 과 같은 연산이 NDArray의 특정 차원에서 어떻게 동작하는지를 보도록 하겠습니다. 어떤 행렬 `X` 에 대해서 모든 원소(기본 사용법)들을 더하거나, 같은 열(`axis=0`)의 원소들만을 더하거나, 또는 같은 행(`axis=1`)의 원소들만을 더할 수 있습니다.  `X` 가 모양이 `(2, 3)` 인 배열이라면, 열끼리의 합 (`X.sum(axis=0)`)의 결과는 모양이 `(3,)` 인 (1D) 벡터가 되는 것을 주의하세요. 만약 더한 결과 차원이 줄어들지 않고 원래 배열의 축의 개수를 유지하고 싶다면 (즉, 결과의 모양이  `(1,3)` 인 2D 배열), `sum` 을 호출할 때 `keepdims=True` 를 설정하면 됩니다.

```{.python .input  n=11}
X = nd.array([[1, 2, 3], [4, 5, 6]])
X.sum(axis=0, keepdims=True), X.sum(axis=1, keepdims=True)
```

*자 이제 우리는 softmax 함수를 정의할 준비가 되었습니다. 우선 각 항에 `exp` 를 적용해서 지수값을 구하고, 정규화 상수(normalization constant)를 구하기 위해서 각 행의 값들을 모두 더합니다. 각 행을 정규화 상수(normalization contatnt)로 나누고 그 결과를 리턴합니다. 코드를 보기 전에 수식을 먼저 보겠습니다.*

We are now ready to implement the softmax function. 
Recall that softmax consists of two steps:
First, we exponentiate each term (using `exp`). 
Then, we sum over each row (we have one row per example in the batch) 
to get the normalization constants for each example. 
Finally, we divide each row by its normalization constant,
ensuring that the result sums to $1$. 
Before looking at the code, let's recall 
what this looks expressed as an equation:

자 이제 우리는 softmax 함수를 구현할 준비가 되었습니다. softmax는 두 단계로 이뤄짐을 상기해보세요. 첫번째로 각 항을 `exp` 함수를 이용해서 거듭제곱을 계산합니다. 그리고 각 행의 원소를 더해서 (배치에는 한 행이 한 예제입니다.) 각 샘들에 대한 정규화 상수를 얻습니다. 마지막으로, 우리는 각 행을 그 것의 정규화 상수로 나눠서, 결과의 합이 1이 되도록 합니다. 코드를 보기 전에, 이 것이 방정식으로 어떻게 표현되는지 보겠습니다.
$$
\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(X_{ij})}{\sum_k \exp(X_{ik})}
$$

*분모는 파티션(partition) 함수라고 불리기도 합니다. 이 이름은 파티클의 앙상블에 대한 분포를 모델링하는 [통계 물리](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))에서 기원합니다.  [Naive Bayes](../chapter_crashcourse/naive-bayes.md)에서 그랬던 것처럼 행렬의 항목들이 너무 크거나 작아서 생기는,  숫자가 너무 커지는 오버플로우(overflow)나 너무 작아지는 언더플로우(underflow)를 고려하지 않고 함수를 구현하겠습니다.*

The denominator, or normalization constant,
is also sometimes called the partition function
(and its logarithm the log-partition function). 
The origins of that name are in [statistical physics](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)) 
where a related equation models the distribution 
over an ensemble of particles). 

정규화 상수인 분모는 파티션 함수(partition function)라고 불리기도 합니다 (그리고 이 값에 로그를 취한 값은 로그-파티션 함수(log-partition function)이라고 합니다.) 이 이름은 관련 방정식이 파티클들의 앙상블에 대한 분포를 모델링하는 [통계역학](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)) 으로부터 왔습니다.

```{.python .input  n=12}
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition  # The broadcast mechanism is applied here
```

As you can see, for any random input, we turn each element into a non-negative number. Moreover, each row sums up to 1, as is required for a probability.
Note that while this looks correct mathematically,
we were a bit sloppy in our implementation
because failed to take precautions against numerical overflow or underflow 
due to large (or very small) elements of the matrix, 
as we did in [Naive Bayes](../chapter_crashcourse/naive-bayes.md).

보는 것처럼, 임의의 난수 입력에 대해서, 각 항목을 0 또는 양의 숫자로 변환합니다. 또한, 확률에서 요구하는 것처럼 각 행의 합은 1이 됩니다. 이것이 수학적으로는 맞게 보일지 몰라도, 이 구현은 다소 엉성합니다. 그 이유는 [나이브 베이즈](../chapter_crashcourse/naive-bayes.md)에서 그랬던 것처럼 행렬의 크거나 또는 아주 작은 원소들로 인해서 일어날 수 있는 수치적인 오버플로우 또는 언더플로우를 고려하지 않았기 때문입니다.

```{.python .input  n=13}
X = nd.random.normal(shape=(2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(axis=1)
```

## 모델

softmax 연산을 구현했으니, 이제 softmax 회귀 모델을 구현할 수 있습니다. 아래 코드는 네트워크에 대한 순전파(forward pass)를 정의합니다. 배치의 각 원본 이미지를 `reshape` 함수를 이용해서 길이가 `num_inputs` 인 벡터로 바꾼 후, 모델에 데이터로 입력하는 것을 주의하세요.

```{.python .input  n=14}
def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)
```

## 손실 함수(loss function)

*앞 절에서 softmax 회귀(regression)에서 사용하는 크로스-엔트로피 손실 함수(cross-entropy loss function)를 소개했습니다. 이는 모든 딥러닝에서 등장하는 손실 함수(loss function)들 중에 가장 일반적인 손실 함수(loss function)입니다. 이유는 회귀(regression) 문제보다는 분류 문제가 더 많기 때문입니다.*

*크로스-엔트로피(cross-entropy)의 계산은 레이블(label)의 예측된 확률값을 얻고, 이 값에 로그(logirithm)  $-\log p(y|x)$ 을 적용하는 것임을 기억해두세요. Python의 `for` loop을 사용하지 않고 (비효율적임), softmax를 적용한 행렬에서 적당한 항목을 뽑아주는 `pick` 함수를 이용하겠습니다. 3개의 카테고리와 2개의 샘플의 경우 아래와 같이 구할 수 있습니다.*

Next, we need to implement the cross entropy loss function,
introduced in the [last section](softmax-regression.md).
This may be the most common loss function 
in all of deep learning because, at the moment, 
classification problems far outnumber regression problems.

다음으로는 [앞 절](softmax-regression.md) 에서 소개한 크로스 엔트로피 손실 함수(cross entropy loss function)을 정의해야 합니다. 현재는 분류의 문제가 회귀 문제보다 훨씬 많기 때문에, 이 함수는 모든 딥러닝에서 가장 흔하게 사용되는 손실 함수입니다. 

Recall that cross entropy takes the negative log likelihood 
of the predicted probability assigned to the true label $-\log p(y|x)$. 
Rather than iterating over the predictions with a Python `for` loop 
(which tends to be inefficient), we can use the `pick` function
which allows us to select the appropriate terms 
from the matrix of softmax entries easily. 
Below, we illustrate the `pick` function on a toy example,
with 3 categories and 2 examples.

크로스 엔트로피는 진짜 레이블에 할당된 예측된 확률에 대한 음수 로그 가능성 (l $-\log p(y|x)$)이라는 것을 기억해보세요. Python의 `for` loop을 이용해서 예측을 하나씩 반복하는 것이 아니라 (효율적이지 않음), 우리는 `pick` 함수를 이용하겠습니다. 이를 이용하면 softmax 요소들로 구성된 행렬에서 원하는 항들을 쉽게 선택할 수 있습니다. 아래 코드는 3개의 카테고리와 2개의 샘플로 구성된 토이 예제에 `pick` 함수를 사용하는 것을 보여줍니다.

```{.python .input  n=15}
y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = nd.array([0, 2], dtype='int32')
nd.pick(y_hat, y)
```

이제 우리는 단 한 줄의 코드로 크로스-엔트로피 손실 함수를 효율적으로 구현할 수 있습니다.

```{.python .input  n=16}
def cross_entropy(y_hat, y):
    return - nd.pick(y_hat, y).log()
```

## 분류 정확도

예측된 확률 분포 `y_hat` 이 주어지면, 결과가 *hard* 예측이어야 하는 경우에 우리는 보통은 예측된 확률이 가장 큰 클래스를 선택합니다. 실제로 많은 애플리케이션은 선택을 요구합니다. Gmail은 이메일을 중요, 소셜, 업데이트 또는 포럼 등으로 분류해야 합니다. 아마 내부적으로는 확률을 추정하고, 결국은 카테고리들 중에서 하나를 선택할 것입니다.

예측이 실제 카테고리 `y` 과 일치한다면, 예측이 정확하다고 합니다. 분류 정확도(classification accuracy)는 모든 예측 중에서 올은 것의 비율입니다. 정확도는 (미분이 불가능하기 때문에) 직접 최적화할 수 없지만, 우리가 신경쓰는 성능 지표이고, 분류기들을 학습시키는 경우 항상 정확도를 보고합니다.

정확도를 계산하는 방법은 다음과 같습니다: 우선, (각 행에서 가장 큰 값에 대한 인덱스를 뽑아서) 예측 클래스를 얻기 위해서  `y_hat.argmax(axis=1)` 를 수행합니다. 그 결과는 변수 `y` 와 같은 모양을 갖습니다. 같음 연산자 `==`  는 데이터형을 고려하기 때문에, (즉, `int` 와 `float32` 는 절대로 같을 수 없습니다.) 두 변수를 같은 타입 (우리는 `float32` 를 사용하겠습니다.)으로 변환하는 것이 필요합니다. 그 결과 거짓이면 0, 참이면 1의 값을 원소로 갖는 NDArray를 얻습니다. 이것의 평균을 구하면 원하는 결과를 얻습니다.

```{.python .input  n=17}
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
```

예측된 확률 분표와 레이블(label)에 대한 변수로 `pick` 함수에서 정의했던  `y_hat` 과 `y` 를 계속 사용하겠습니다. 첫번째 샘플의 예측 카테고리는 2 (첫번째 행에서 가장 큰 값은 0.6이고 이 값의 인덱스는 2)임을 확인할 수 있고, 이는 실제 레이블 0과 일치하지 않습니다. 두번째 샘플의 예측 카테고리는 2 (두번째 행에서 가장 큰 값이 0.5이고 이값의 인덱스는 2)이고, 이는 실제 레이블 2와 일치합니다. 따라서, 이 두 예들에 대한 분류 정확도 0.5 입니다.

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

이 모델 `net` 은 난수 값으로 가중치 값들이 초기화되어 있기 때문에, 정확도는 임의로 추측하는 것과 유사한 0.1 (10개의 클래스)로 나올 것입니다.

```{.python .input  n=20}
evaluate_accuracy(test_iter, net)
```

## 모델 학습

만약 이 장의 초반에 나온 신형 회귀 구현을 잘 읽었다면, softmax 회귀를 학습시키는 loop이 굉장히 친숙하게 보일 것입니다. 여기서도 모델의 손실 함수를 최적화하기 위해서 미니 배치 확률적 경사 하강법(stochastic gradient descdent)를 사용합니다. 에포크 횟수(`num_epochs`)와 학습 속도(`lr`)은 모두 조정이 가능한 하이퍼파라미터임을 기억해 주세요. 이 값들을 바꿔서 모델의 분류 정확도를 높일 수도 있습니다. 실제로 우리는 데이터를 3개 - 학습, 검증, 테스트 데이터 - 로 나눌 것이고, 검증 데이터를 사용해서 가장 좋은 하이퍼파라미터들을 찾습니다.

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

학습이 완료되었으니 모델은 이미지들을 분류할 준비가 되었습니다. 이미지들이 주어졌을 때, 실제 레이블들 (텍스트 결과의 첫번째 줄)과 모델 예측 (텍스트 결과의 두번째 줄)를 비교 해보세요.

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

softmax 회귀를 사용해서 다중 카테고리 분류를 위한 모델을 학습시킬 수 있습니다. 학습 loop은 선형 모델과 매우 유사합니다: 데이터를 조회해서 읽고, 모델과 손실 함수를 정의한 후, 최적화 알고리즘을 사용해서 모델을 학습시킵니다. 앞으로 알게되겠지만, 가장 일반적인 딥러닝 모델들은 이와 유사한 학습 절차를 사용합니다.

## 연습문제

1. 이 절에서 softmax 연산의 수학적인 정의에 따라 softmax 함수를 직접 정의해봤습니다. 이 경우 어떤 문제가 발생할 수 있을까요? (힌트 - exp(50)의 크기를 계산해보세요)
1. 이 절의 `cross_entropy` 함수 크로스-엔트로피 손실 함수(cross-entropy loss function)의 정의를 따라서 구현되었습니다. 이 구현에 어떤 문제가 있을까요? (힌트 - logarithm의 도메인을 고려해보세요)
1. 위 두가지 문제를 어떻게 해결할 수 있는지 생각해보세요
1. 가장 유사한 레이블을 리턴하는 것이 항상 좋은 아이디어일까요? 예를 들면, 의료 진단에서 그렇게 하겠나요?
1. 어떤 특성(feature)들을 기반으로 다음 단어를 예측하기 위해서 softmax 회귀(regression)을 사용하기를 원한다고 가정하겠습니다. 단어 수가 많은 경우 어떤 문제가 있을까요?

## QR 코드를 스캔해서 [논의하기](https://discuss.mxnet.io/t/2336)

![](../img/qr_softmax-regression-scratch.svg)
