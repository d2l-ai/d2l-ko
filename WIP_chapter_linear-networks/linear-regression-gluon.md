# 선형 회귀의 간결한 구현

딥러닝에 대한 큰 관심이 딥러닝 모델을 구현할 때 필요한 반복적인 일들을 자동화 해주는 다양한 성숙한 소프트웨어 프레임워크들이 개발을 이끌어 왔습니다. 앞에서는 데이터 저장과 선형 대수 연산을 위해서 NDArray를, 자동 미분 기능을 제공하는 `autograd` 패키지만을 사용했었습니다. 실제 상황에서는 데이터 이터레이터, 손실 함수(loss function), 모델 아키텍처, 옵티아이저(optimizer)와 같은 더 많은 추상적인 연산들이 너무 많이 사용되기며, 딥러닝 라이브러리들이 이런 기능들을 라이브러리 함소로 제공하고 있습니다.

이 절에서는 뉴럴 네트워크를 구현하는데 사용하는 MXNet의 고차원 인터페이스인 Gluon을 소개하고, 앞 절에서 소개한 선형 회귀 모델을 매우 간결하게 구현하는 방법을 보여드리겠습니다.

## 데이터 셋 만들기

시작을 위해서, 이전 절에서 사용한 것처럼 동일한 데이터셋을 생성합니다.

```{.python .input  n=2}
from mxnet import autograd, nd

num_inputs = 2
num_examples = 1000
true_w = nd.array([2, -3.4])
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
```

## 데이터 읽기

Gluon은 데이터를 읽는데 사용할 수 있는  `data` 모듈을 제공합니다.  `data` 라는 이름은 변수 이름으로 많이 사용하기 때문에,  `gdata` 라고 별명을 붙여서 사용하겠습니다. 매 반복(iteration) 마다, 10개 데이터 인스턴스를 갖는 미니 배치를 읽어보겠습니다.

직접 이터레이터를 작성하는 대신, Gluon의 `data` 모듈을 사용해서 데이터를 읽겠습니다.  `data` 라는 이름은 변수 이름으로 많이 사용하기 때문에, 우리가 정의하는 변수들과 `data` 모듈에서 제공하는 것을 구별하기 위해서  `gdata` 라고 별명을 붙여서 사용하겠습니다 (Gluon의 첫 글짜를 붙였습니다). 여기서 우리는 `features` 와 `labels` 를 함수의 인자로 전달합니다. 그 다음에는 ArrayDataset를 사용해서 DataLoader를 초기화하는데, `batch_size` 를 통해서 배치 크기와 `shuffle` 에 boolean 값을 지정해서 `DataLoader` 가 매 에포크(epoch)마다 데이터를 섞을지를 지정해야 합니다.

```{.python .input  n=3}
from mxnet.gluon import data as gdata

batch_size = 10
# Combine the features and labels of the training data
dataset = gdata.ArrayDataset(features, labels)
# Randomly reading mini-batches
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
```

자 이제 우리는 이전 절에서 `data_iter` 함수를 사용한 것과 동일한 방법으로 `data_iter` 를 사용할 수 있습니다. 동작하는지 확인하기 위해서, 첫번째 미니 배치를 읽어서 내용을 출력해 보겠습니다.

```{.python .input  n=5}
for X, y in data_iter:
    print(X, y)
    break
```

## 모델 정의하기

앞 절에서 선형 회귀를 직접 구현했을 때는 모델 파라미터들을 정의하고 기본적인 선형 대수 연산을 이용해서 결과를 계산하는 것을 직접 작성해야 했습니다. 그렇기에 어떻게 하는지를 알겄습니다. 하지만, 여러분의 모델이 복잡해짐에 따라서, 모델을 질적으로 아주 간단하게 변경을 하기 위해서 많은 저수준의 변경이 필요할 수 있습니다. 

표준 연산들에 대해서는 Gluon의 사전에 정의된 층들을 이용해서, 구현에 집중하는 것보다는 모델을 구성하는 층들에 더 집중을 할 수 있습니다.

선형 모델을 구현하기 위해서 먼저 다양한 뉴럴 네트워크 층들을 정의하고 있는 `nn` 모듈을 import 합니다. `nn`  은 neural network의 약자입니다. 우선 모델 변수 `net` 을  `Sequencial` 의 인스턴스로 정의합니다. Gluon에서 `Sequential` 인스턴스는 다양한 레이어를 순차적으로 담는 컨테이너로 사용됩니다. 입력값이 주어지면, 컨테이너의 각 층 순서대로 계산이 이뤄지고, 각 층의 결과값은 다음 층의 입력으로 사용됩니다. 이 예제에서 우리의 모델은 단 하나의 층을 갖기 때문에 `Sequential` 이 사실은 필요 없습니다. 하지만, 앞으로 거의 모든 모델들은 여러 층을 갖을 것이기 때문에, 사용하는 습관을 갖도록 하겠습니다.

```{.python .input  n=5}
from mxnet.gluon import nn
net = nn.Sequential()
```

단일층 네트워크를 다시 생각해봅시다. 층은 모든 입력들과 모든 출력들을 연결하는 완전 연결(fully connected)로 구성되어 있고,  이는 행렬-벡터 곱으로 표현했습니다. Gluon에서는 완전 연결층(fully connected layer)이 `Dense` 클래스로 정의됩니다. 한개의 스칼라 출력을 생성하기 때문에 갯수를 1로 정의합니다.

![Linear regression is a single-layer neural network. ](../img/singleneuron.svg)

```{.python .input  n=6}
net.add(nn.Dense(1))
```

Gluon의 특별한 점은, 편리함을 위해서 각 층의 입력 모양(shape)을 별도로 지정할 필요가 없다는 것입니다. 즉 이 단계에서 Gluon에게 몇개의 입력이 이 선형층으로 들어올지를 알려줄 필요가 없습니다. 모델에 첫번째 데이터를 입력하는 순간 (즉, `net(X)` 를 수행하면), Gluon은 자동으로 각 층에 들어오는 입력의 개수를 추정합니다. 이것이 어떻게 동작하는지는 ""딥러닝 계산" 장에서 자세히 설명하겠습니다.

## 모델 파라미터들 초기화하기

`net` 을 사용하기전에 선형 회귀 모델(linear regression model)의 가중치와 편향(bias) 같은 모델 파라미터들을 초기화를 해야합니다. 이를 위해서 MXNet의 `initializer`모듈을 import 합니다. 이 모듈을 이용하면 다양한 방법으로 모델 파라미터를 초기화할 수 있습니다. Gluon은 `initializer` 패키지에 바로가기 위한 약어로 `init` 를 제공합니다. `init.Normal(sigma=0.01)` 를 호출하면, 평균이 0이고 표준편차가 0.01인 정규 분포를 따르는 난수값들로 각 *가중치* 파라미터들을 초기화합니다. *편향(bias)*는 0으로 기본 설정되게 둡니다.

```{.python .input  n=7}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

위 코드는 매우 직관적으로 보이지만, 실제로는 상당히 이상한 일이 일어납니다. 이 단계에서 우리는 아직 Gluon에게 입력이 어떤 차원을 갖는지를 알려주지 않았음에도 불구하고, 파라미터를 초기화하고 있습니다. 네트워크를 어떻게 정의하는지에 따라서 따라서 2가 될 수도, 2,000이 될 수도 있기 때문에, 이 시점에서 메모리를 미리 할당할 수도 없습니다.

내부에서는 최초로 네트워크에 데이터를 전달할 때까지 초기화를 지연시키기 때문에, Gluon이 이것을 하지 않아도 되도록 해줍니다. 단, 파라미터들이 아직 초기화되지 않았기 때문에 파라미터들을 아직은 조작하지 못한다는 것을 조심하세요.

##  손실 함수(loss function) 정의하기

Gluon의 `loss` 모듈은 다양한 손실 함수(loss function)를 정의하고 있습니다. `loss` 모듈을 import 할 때 이름을 `gloss` 바꾸고, 제곱 손실(squared loss)(`L2Loss`)의 구현을 직접 사용합니다.

```{.python .input  n=8}
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()  # The squared loss is also known as the L2 norm loss
```

## 최적화 알고리즘 정의하기

당연하게도 우리가 미니 배치 확률적 경사 하강법(stochastic gradient descent, SGD)를 구현하는 첫번째 사람이 아닙니다. 따라서, `Gluon` 은 SGD 와 이 알고리즘의 다양한 변종을 `Trainer` 클래스를 통해서 제공하고 있습니다. `Trainer` 를 초기화할 때, 우리는 최적화를 할 파라미터들(`net.collect_params()` 를 통해 얻습니다)을 지정하고, 사용할 최적화 알고리즘 (`sgd`)와 해당 최적화 알고리즘에 필요한 하이퍼파라미터들의 사전을 명시합니다. SGD의 경우 `학습 속도(learning rate)` 만 설정하면 됩니다. (여기서는 0.03으로 설정합니다.)

```{.python .input  n=9}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

## 학습

Gluon을 이용해서 모델을 표현하는 것이 상대적으로 몇 줄의 코드만으로 된다는 것을 눈치챘을 것입니다. 파라미터들을 일일이 할당하지 않았고, 손실 함수(loss function)를 직접 정의하지 않았고, 확률적 경사 하강법(stocahstic gradient descent)을 직접 구현할 필요가 없었습니다. 더욱 복잡한 형태의 모델을 다루기 시작하면, Gluon의 추상화를 사용함으로 얻는 이득은 아주 커질 것입니다. 필요한 기본적인 것들을 모두 만들었다면, 학습 loop 자체는 모든 것을 직접 구현하면서 했던 것과 굉장히 유사합니다.

앞서 구현했던 모델 학습 단계를 다시 짚어보면, 전체 학습 데이터(train_data)를 정해진 에포크(epoch)만큼 반복해서 학습을 수행합니다. 하나의 에포크(epoch)는 다시 미니 배치로 나눠지는데 미니 배치는 입력과 입력에 해당하는 진실 값(ground-truth label)들로 구성됩니다. 각 미니 배치에서는 다음 단계들이 수행됩니다.

* `net(x)` 를 호출해서 예측값을 생성하고, 손실(loss) `l` 을 계산합니다. (순전파)
* `l.backward()` 수행해서 그래티언트(gradient)들을 계산합니다. (역전파)
* SGD 옵티마이저(optimizer)를 수행해서 모델 파라미터들을 업데이트합니다. (`trainer` 는 이미 어떤 파라미터들을 최적화해야 하는지를 알고 있으니, 우리는 배치 크기만 전달하면 됩니다.)

학습 상태를 관찰하기 위해서, 매 에포크(epoch) 마다 전체 특성(feature)들에 대해서 손실(loss() 값을 계산해서 출력합니다.

```{.python .input  n=10}
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```

학습된 모델 파라미터들과 실제 모델 파라미터를 비교해봅니다. 모델의 파라미터들의 값을 확인하는 방법은, 우선  `net` 인스턴스로부터 층을 얻어내고, 그 층의 가중치(`weight`) 변수와 편향 (`bias`) 변수를 접근하는 것입니다. 학습된 파라미터들과 실제 파라미터들이 매우 비슷한 것을 볼 수 있습니다.

```{.python .input  n=12}
w = net[0].weight.data()
print('Error in estimating w', true_w.reshape(w.shape) - w)
b = net[0].bias.data()
print('Error in estimating b', true_b - b)
```

## 요약

* Gluon을 이용해서 모델을 매우 간단하게 구현할 수 있습니다.
* Gluon에서 `data` 모듈은 데이터 프로세싱하는 도구를, `nn` 모듈을 많은 종류의 뉴럴 네트워크 레이어들의 정의를 그리고 `loss` 모듈은 다양한 손실 함수(loss function)를 제공합니다.
* MXNet의 `initializer` 모듈은 모델 파라미터를 초기화하는 다양한 방법을 제공합니다.
* 모델 파라미터의 차원(dimensionality)과 저장 공간은 할당은 실제 사용될 때까지 미뤄집니다. (따라서, 초기화되기 전에 파라미터를 접근하는 경우에 주의해야 합니다.)

## 연습문제

1. `l = loss(output, y)` 를  `l = loss(output, y).mean()` 로 바꿀 경우, `trainer.step(batch_size)` 를 `trainer.step(1)` 바꿔야합니다. 왜 일까요?

1.  `gluon.loss` 와 `init` 모듈에 어떤 손실 함수(loss function)와 초기화 방법들이 포함되어 있는지 MXNet 문서를 읽어보세요. 손실 함수(loss function)를 Huber's loss로도 바꿔보세요.
2.  `dense.weight` 의 그래디언트(gradient)를 어떻게 접근할 수 있을까요?

## 

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2333)

![](../img/qr_linear-regression-gluon.svg)
