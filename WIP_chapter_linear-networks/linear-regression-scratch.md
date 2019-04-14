# 선형 회귀를 처음부터 구현하기

자 이제 선형 회귀의 *아이디어* 에 대한 배경을 알아봤으니, 실제 구현을 하나씩 해볼 수 있는 준비가 되었습니다. 이 절에서는 (앞으로 나올 유사한 절에서도 같이) 선형 회귀의 모든 부분들을 구현합니다: 데이터 파이프라인, 모델, 손실 함수(loss function), 경사 하강 옵티마이저(gradient descent optimizer)들을 직접 구현합니다. 당연하지만 현재 딥러닝 프레임워크들은 이 일들을 거의 자동으로 해줄 수 있습니다만, 직접 구현하는 것을 배우지 않으면, 모델이 어떻게 동작하는지 잘 이해하지 못할 것입니다. 더구나,  층과 손실 함수를 정의하면서 커스텀 모델을 만들어야 할 때, 이것들이 실제로 어떻게 동작하는지 아는 것이 도움이 됩니다. 따라서, NDArray와 `autograd` 패키지의 기본적인 것들만 사용해서 선형 회귀를 어떻게 구현하는지를 시작해보겠습니다. 바로 다음 절에서는 Gluon의 모든 기능들을 이용해서 간결한 구현을 보여주겠습니다.

시작하기 위해서 이 절의 실험들을 수행하는데 필요한 패키지들을 import 합니다: 도표를 위한 `matplotlib` 을 사용하고, GUI에 임베딩하는 설정을 합니다.

```{.python .input  n=1}
%matplotlib inline
from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
```

## 데이터셋 생성하기

이 데모를 위해서 우리는 간단한 인위적인 데이터셋을 만들 것입니다. 이를 통해서 우리는 데이터를 쉽게 시각화하고, 학습된 파라미터들에 대한 실제 패턴을 비교해볼 수 있습니다. 학습 셋의 예제 개수를 1000개로 특징(또는 공변량(covariates)) 개수는 2개로 설정합니다. 임의로 생성된 데이터셋은 $\mathbf{X}\in \mathbb{R}^{1000 \times 2}$ 인 객체가 됩니다. 이 예제에서 우리는 가우시안 분포로 부터 데이터 포인트  $\mathbf{x}_i$ 를 샘플링해서 데이터를 만들겠습니다.

또한 우리의 알고리즘이 동작하는 것을 보장하기 위해서 실제 파라미터  $\mathbf{w} = [2, -3.4]^\top$ 와  $b = 4.2$ 를 따르는 선형성을 갖는다고 가정합니다. 즉, 만들어진 레이블은 특정과 레이블의 오차를 더하기 위해서 노이즈 항목 $\epsilon$ 을 포함한 아래 선형 모델에 따라서 주어집니다.

$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon​$$

일반적인 가정을 따라서, 노이즈 항목 $\epsilon$ 은 평균이 $0$인 표준 분포를 따르도록 선택하고, 표준 편차는 $0.01$ 로 설정합니다. 아래 코드는 임의의 데이터셋을 생성합니다.

```{.python .input  n=2}
num_inputs = 2
num_examples = 1000
true_w = nd.array([2, -3.4])
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
```

 `features` 의 각 행은 2차원 데이터 포인트로 구성되고,  `labels` 의 각 행은 1차원 타겟 값으로 구성됩니다.

```{.python .input  n=3}
features[0], labels[0]
```

 `features[:, 1]` 과 `labels` 를 이용해서 scatter plot을 생성해보면, 둘 사이의 선형 관계를 명확하게 관찰할 수 있습니다.

```{.python .input  n=4}
def use_svg_display():
    # Display in vector graphics
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # Set the size of the graph to be plotted
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.figure(figsize=(10, 6))
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1);
```

plot을 그려주는 함수 `plt`,  `use_svg_display` 함수와 `set_figsize` 함수는 `g2l` 패키지에 정의되어 있습니다. 이제 plot을 어떻게 만드는지 알았으니, 앞으로는 plot을 그리기 위해서  `g2l.plt` 를 직접 호출하겠습니다.  `plt` 은  `g2l` 패키지의 전역 변수로 정의되어 있기 때문에, 벡터 다이어그램과 크기를 정하기 위해서는 plot을 그리기 전에  `g2l.set_figsize()` 를 호출하면 됩니다. 

## 데이터 읽기

데이터셋을 여러번 반복해서 사용하고, 매번 미니 배치를 얻고 이를 사용해서 모델을 업데이트하는 모델 학습을 떠올려보세요. 이 절차는 머신 러닝 알고리즘 학습의 아주 근본적인 것이기 때문에, 우리는 데이터를 섞고 미니-배치를 얻는 유틸리티가 필요합니다.

아래 코드에서 이 기능의 가능한 구현으로  `data_iter` 함수를 정의합니다. 이 함수는 배치 크기, 특성을 담고 있는 디자인 행렬, 레이블 벡터를 인자로 받습니다. 미니 배치 크기는 `batch_size`  이고, 특성과 레이블의 쌍(tuple) 형태입니다.

```{.python .input  n=5}
# This function has been saved in the d2l package for future use
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)
        # The “take” function will then return the corresponding element based
        # on the indices
```

일반적으로 우리는 미니-배치 크기를 적당한 크기로 선택해서 병평 연산에 띄어난 GPU 하드웨어의 이점을 살리기를 원합니다. 각 예제는 모델에 병렬로 입력되고, 손실 함수(loss function)의 그래디언트 역시 병렬로 계산될 수 있기 때문에, GPU는 하나의 샘플을 처리하는 것과 한번에 수 백개의 예제를 처리하는 것은 거의 같은 시간이 걸립니다.

직관적인 이해를 위해서, 데이터 예제들의 첫 번째 작은 배치를 읽어서 출력해봅니다. 각 미니-배치의 특성들의 모양(shape)은 미니-배치 크기와 입력 특성의 개수를 의미합니다. 비슷하게 미니-배치 레이블은 `batch_size` 를 모양(shape)로 갖습니다.

```{.python .input  n=6}
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break
```

당연하겠지만, 이터레이터를 수행하면 데이터가 모두 소진될 때까지 매번 다른 미니배치를 얻습니다. 위에서 구현한 이터레이터는 교육적인 목적으로는 좋지만, 비효율적인 구현이기 때문에 실제 문제에 적용하면 여러분을 문제에 빠뜨릴 것입니다. 예를 들면, 모든 데이터를 메모리에 로드하도록 되어 있어고, 따라서 랜덤 메모리 접근을 아주 많이 해야합니다. Apache MXNet에 기본으로 구현되어 있는 이터레이터들은 상당히 효율적으로 구현되어 있고, 파일에 저장되어 있는 데이터 뿐만 아니라, 데이터 스트림을 통해서도 데이터를 읽어올 수 있습니다.

## 모델 파라미터들 초기화하기

그래디언트 하강법으로 모델 파라미터 최적화를 시작할 수 있기 위해서는 우선 몇 개의 파라미터들을 필요합니다. 아래 코드에서는 평균이 0이고 표준 편차가 0.01인 표준 편차에서 난수를 샘플링해서 가중치를 초기화하고, 편향 $b$ 는 0으로 설정합니다.

```{.python .input  n=7}
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
```

자 이제 파라미터 초기화를 했으니, 다음 해야 할 일은 데이터에 충분히 잘 들어맞을 때까지 파라미터들을 업데이트하는 것입니다. 매 업데이트를 하기 위해서는 파라미터들에 대해서 손실 함수의 미분(또는 다차원 미분)을 구하는 것이 필요합니다. 이 미분이 구해지면, 우리는 각 파라미터를 손실이 줄어드는 방향으로 업데이트 합니다.

미분을 직접 계산하고 싶은 사람이 없기 때문에 (이것은 지루하고 오류가 발생하기 쉽습니다), 우리는 미분을 계산해주는 자동 미분을 사용하겠습니다. 자세한 내용은  ["Automatic Gradient"](../chapter_prerequisite/autograd.md) 를 참고하세요. 'autograd' 장에서 `autograd` 가 파라미터들의 미분값을 저장해야 한다는 것을 알기 위해서, `attach_grad` 함수를 호출해서 우리가 구하고자 하는 미분값을 저장할 메모리를 할당하는 것이 필요하다는 것을 기억해보세요.

```{.python .input  n=8}
w.attach_grad()
b.attach_grad()
```

## 모델 정의하기

다음으로 우리는 입력과 파라미터를 출력으로 연결시키는 모델을 정의해야 합니다. 선형 모델의 결과를 계산하기 위해서 간단하게 예제  $\mathbf{X}$ 와 모델 가중치 $w$ 를 이용한 행렬-벡터 점곱을 수행하고, 오프셋 $b$ 를 각 예제에 더합니다. 아래 코드에서 `nd.dot(X. w)` 는 벡터이고 `b` 는 스칼라입니다. 벡터와 스칼라를 더할 때, 스칼라가 벡터의 각 원소에 더합니다.

```{.python .input  n=9}
# This function has been saved in the d2l package for future use
def linreg(X, w, b):
    return nd.dot(X, w) + b
```

## 손실 함수(loss function) 정의하기

모델 업데이트를 위해서 손실 함수의 미분을 구해야하기 때문에, 우리는 우선 손실 함수를 정의해야 합니다. 여기서는 이전 절에서 소개한 제곱 손실 함수를 이용하겠습니다. 구현할 때, 진짜 값 `y` 를 예측된 값, `y_hat`, 의 모양으로 변환시켜야 합니다. 아래 함수가 리턴하는 결과는 `y_hat` 모양과 동일한 것이 됩니다.

```{.python .input  n=10}
# This function has been saved in the d2l package for future use
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

## 최적화 알고리즘 정의하기

앞 절에서 논의했듯이 선형 회귀는 닫힌 형태의 답(closed-form solution)을 갖습니다. 하지만, 이 책은 선형 회귀에 대한 책이 아니라, 딥러닝에 대한 책입니다. 이 책이 소개하는 다른 모델들 중에 어떤 것도 분석적으로 풀리지 않기 때문에, 우리는 확률적 경사 하강법(stochastic gradient descent, SGD)의 첫번째 동작하는 예제를 소개하는 기회를 갖겠습니다.

매 스텝마다, 데이터셋에서 임의로 뽑은 하나의 배치를 사용해서 손실의 파라미터에 대한 미분을 추정합니다. 그리고는 손실을 줄이는 방향으로 파라미터를 조금 업데이트를 합니다. 미분이 이미 계산되어 있다고 가정하면, 각 파라미터 (`param`)은 그것의 미분값이 `param.grad`에 저장되어 있습니다. 아래 코드는 주어진 파라미터들의 집합, 학습 속도(learning rate), 배치 크기에 대해서 SGD 업데이트를 적용합니다. 업데이트 스텝의 크기는 학습 속도 `lr` 로 결정됩니다. 예제의 배치에 걸친 합으로 손실을 계산하기 때문에, 전형적인 스텝 크기의 정도가 배치 크기의 선택에 크게 의존되지 않도록 각 스텝의 크기를 배치 크기(`batch_size`)로 표준화합니다. 

```{.python .input  n=11}
# This function has been saved in the d2l package for future use
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

## 학습

자 이제 모든 파트들이 준비되었으니, 메인 학습 룹을 구현할 수 있습니다. 여러분의 딥러닝 경력 내내 이것과 동일한 학습 룹을 계속 볼 것이기 때문에, 이 코드를 이해하는 것은 아주 중요합니다.

매 반복(iteration)에서 모델의 미니배치를 얻고, 이를 모델에 입력해서 예측값들의 집합을 얻습니다. 손실을 계산한 후, 네트워트를 따라서 역전파를 수행하면서 각 파라미터에 대한 미분을 해당하는 `.grad` 속성에 저장하는  `backward` 함수를 호출할 것입니다. 마지막으로 모델 파라미터들을 업데이트하기 위해서 최적화 알고리즘 `sgd` 를 호출합니다. 앞에서 배치 크기 `batch_size` 를 10으로 설정했놓았기 때문에, 각 작은 배치에 대한 손실의 모양 `l` 은 (10,1)이 됩니다.

요약하면, 아래 룹을 수행합니다.

* 파라미터 $(\mathbf{w}, b)$ 를 초기화합니다.
* 끝날 때까지 다음을 반복합니다.
    * 그래디언트(gradient) 계산하기 $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{\mathcal{B}} \sum_{i \in \mathcal{B}} l(\mathbf{x}^i, y^i, \mathbf{w}, b)$
    * 파라미터 업데이트하기 $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

아래 코드에서 `l` 은 미니배치이 각 예제에 대한 손실들의 벡터입니다. `l` 이 스칼라 변수가 아니기 때문에, `l.backward()` 실행은 `l` 원소 모두를 더해서 새로운 변수를 얻고 미분을 계산합니다.

매 에포크(데이터 전체를 한번 지나감)에서 학습 데이터셋의 모든 예제들을 한번씩 지나면서 전체 데이터셋을 방문할 것입니다. (이를 위해서 `data_iter` 함수가 사용됩니다.) 여기서 우리는 전체 셈플 개수가 배치 크기로 나눠 떨어진다고 가정합니다. 에포크 횟수 `num_epochs` 와 학습 속도 `lr` 모두 하이퍼파라미터입니다. 우리는 각각 $3$과 $0.03​$으로 설정합니다. 불행하게도 하이퍼파라미터를 설정하는 것은 까라로운 일이고, 시도와 오류를 통해서 조정하는 것이 필요합니다. 우리는 세부적인 내용은 지금은 접어두겠고, 이후 ["Optimization Algorithms"](../chapter_optimization/index.md) 장에서 알아보겠습니다.

```{.python .input  n=12}
lr = 0.03  # Learning rate
num_epochs = 3  # Number of iterations
net = linreg  # Our fancy linear model
loss = squared_loss  # 0.5 (y-y')^2

for epoch in range(num_epochs):
    # Assuming the number of examples can be divided by the batch size, all
    # the examples in the training data set are used once in one epoch
    # iteration. The features and tags of mini-batch examples are given by X
    # and y respectively
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Minibatch loss in X and y
        l.backward()  # Compute gradient on l with respect to [w,b]
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))
```

이 경우에 합성 데이터 (실제로 우리가 직접 만들었습니다.) 사용했기 때문에, 실제 파라미터들이 무엇인지를 정확하게 알고 있습니다. 따라서, 실제 파라미터와 학습을 통해서 배운 것을 비교하면 학습의 성공 여부를 평가할 수 있을 것입니다. 실제로 이 값들이 매우 가깝게 있다는 것이 밝혀졌습니다.

```{.python .input  n=13}
print('Error in estimating w', true_w - w.reshape(true_w.shape))
print('Error in estimating b', true_b - b)
```

우리는 파라미터를 정확하게 되돌릴 수 있다는 것을 당연하게 받아들이지 마세요. 이것은 특별한 카테고리 문제에서만 야기됩니다: 노이즈 샘플이 깔려있는 의존도를 발견하게 할 만큼 '충분한' 데이터가 있는 강한 볼록 최적화 문제. 대부분의 경우는 이런 케이스가 *아닙니다*. 사실, 데이터를 반대 방향으로 탐색하는 것을 포함해서 모든 조건이 동일하지 않을 경우 두 학습에 대한 딥 네트워크의 파라미터들은 같거나 비슷한 경우가 드뭅니다. 하지만, 머신 러닝에서 우리는 실제 숨어 있는 파라미터를 복구하는 것에 덜 관심이 있고, 정확한 예측을 하는 파라미터들에 더 관심이 있습니다. 운이 좋계도, 어려운 최적화 문제에도 확률적 경사 하강법은 놀랍게도 좋은 해법을 종종 만들어 납니다. 일부 이유는 우리가 다루는 모델에 대해서 잘 예측하는 파라미터들의 집합이 여러개 있다는 사실 때문입니다.

## 요약

우리는 NDArray와 `autograd` 만을 사용하고, 층이나 멋진 옵티마이저 등을 정의한 필요 없이 딥 네트워크를 어떻게 구현하고, 최적화시킬 수 있는지를 봤습니다. 이것은 가능한 것들의 표면만 긁어본 것입니다. 다음 절들에서 우리는 우리가 막 소개한 개념들을 기반으로 한 추가적인 모델을 설명하고, 더 간결하게 구현하는 방법을 배워보겠 습니다.

## 연습문제

1. 가중치(weight)들을 0으로 ($\mathbf{w} = 0$) 초기화를 하면 어떤 일이 발생할까요? 알고리즘이 여전히 동작할까요?
1. 여러분이 전압과 전류간의 모델을 만들고자 하는 [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) 이라고 가정해 보세요. 여러분의 모델 파라미터를 학습시키기 위해서 `autograd` 를 사용할 수 있을까요?
1. 스펙트럼 에너지 밀도를 사용해서 물체의 온도를 결정하는 데 [Planck's Law](https://en.wikipedia.org/wiki/Planck%27s_law) 를 사용할 수 있나요?
1. `autograd` 를 이차 미분으로 확장한다면 어떤 문제를 만날 수 있을까요?
1. `squared_loss` 함수에서 `reshape` 함수가 왜 필요한가요?
1. 다양한 학습 속도(learning rate)들을 사용해서 실험하고, 그 결과 손실 함수(loss function) 값이 얼마나 빠르게 감소하는지 알아보세요.
1. 예제들의 개수가 배치 크기로 나누어 떨어지지 않을 경우에, `data_iter` 함수는 어떻게 동작할까요?

## QR 코드를 스캔해서 [논의하기](https://discuss.mxnet.io/t/2332)

![](../img/qr_linear-regression-scratch.svg)
