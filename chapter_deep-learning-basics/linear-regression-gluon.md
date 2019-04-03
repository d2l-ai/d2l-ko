# 선형 회귀의 간결한 구현

딥러닝 프레임워크들의 개발로 인해서, 딥러닝 어플리케이션을 개발하는 것이 나날이 쉬워지고 있습니다. 이번에는 앞 절에서 구현한 모델을 보다 간결하게 구현하고 학습시키는 방법을 MXNet에서 제공하는 Gluon 인터페이스를 이용해서 구현해보도록 하겠습니다.

## 데이터 셋 만들기

이전 절에서 사용한 것처럼 동일한 데이터셋을 생성합니다.

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

Gluon은 데이터를 읽는데 사용할 수 있는  `data` 모듈을 제공합니다.  `data` 라는 이름은 변수 이름으로 많이 사용하기 때문에,  `gdata` 라고 별명을 붙여서 사용하겠습니다. 매 iteration 마다, 10개 데이터 인스턴스를 갖는 미니 배치를 읽어보겠습니다.

```{.python .input  n=3}
from mxnet.gluon import data as gdata

batch_size = 10
# Combine the features and labels of the training data
dataset = gdata.ArrayDataset(features, labels)
# Randomly reading mini-batches
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
```

 `data_iter` 는 이전 절에서 사용한 방식과 같습니다. 이를 이용해서 첫번째 미니 배치를 읽어서 내용을 출력합니다.

```{.python .input  n=5}
for X, y in data_iter:
    print(X, y)
    break
```

## 모델 정의하기

선형 회귀 모델을 직접 구현했을 때는 모델 파라메터를 정의하고, 모델을 수행하는 매 단계를 직접 작성했는데, 더 복잡한 모델을 이렇게 만들어야 한다면 구현이 더 복잡해질 것입니다. Gluon은 이미 정의된 다양한 레이어를 제공하고 있어서, 레이어를 어떻게 구현하는지가 아니라, 레이어를 사용한 모델을 설계하는데 집중할 수 있도록 해줍니다.

선형 모델을 구현하는 방법은, 우선 `nn` 모듈을 import하는 것으로 시작합니다. `nn`  은 neural network의 약자입니다. 이름이 의미하듯이 이 모듈을 많은 종류의 neural network 레이어들을 바로 사용할 수 있도록 제공합니다.  `Sequential` 인스턴스인  `net` 모델 변수를 정의합니다. Gluon에서  `Sequential` 인스턴스는 다양한 레이어를 순차적으로 담는 컨테이너로 사용됩니다.  모델을 정의할 때, 이 컨터이너에 레이어를 원하는 순서대로 추가합니다. 입력값이 주어지면, 컨테이너의 각 레이어 순서대로 계산이 이뤄지고, 각 레이어의 결과값은 다음 레이어의 입력으로 사용됩니다.

```{.python .input  n=5}
from mxnet.gluon import nn
net = nn.Sequential()
```

단일 레이어 네트워크를 다시 생각해봅시다. 레이어는 모든 입력들과 모든 출력들을 연결하는 fully connected로 구성되어 있고,  이는 matrix-vector 곱으로 표현했습니다. Gluon의   `Dense` 인스턴스는 fully connected layer를 의미합니다. 한개의 scalar 출력을 생성해야하기 때문에 갯수를 1로 정의합니다.

![Linear regression is a single-layer neural network. ](../img/singleneuron.svg)

```{.python .input  n=6}
net.add(nn.Dense(1))
```

Gluon의 특별한 점은, 각 레이어의 입력 shape를 별도로 지정할 필요가 없다는 것입니다. 예를 들면, 선형 회귀의 입력 개수 같은 것. 예를 들면  `net(X)` 가 수행되면, 모델이 데이터를 보면서 각 레이어의 입력 개수를 자동으로 계산해줍니다. 이 원리에 대한 내용은 "Deep Learning Computation" 장에서 자세히 다루겠습니다. Gluon의 이런 디자인은 모델 개발을 더 쉽게 만들어줍니다.

## 모델 파라메터들 초기화하기

`net` 을 사용하기전에 선형 회귀 모델(linear regression model)의 weight과 bias 같은 모델 파라메터들을 초기화를 해야합니다. 이를 위해서 MXNet의 `initializer`모듈을 import 합니다. 이 모듈을 이용하면 다양한 방법으로 모델 파라메터를 초기화할 수 있습니다. import 한 `init` 은 `initializer` 의 간략한 이름입니다. `init.Normal(sigma=0.01)` 을 통해서 평균이 0이고 표준편차가 0.01인 정규 분포를 따르는 난수값들로 각 weight 파라매터들을 초기화합니다. bias는 0으로 기본 설정되게 둡니다.

```{.python .input  n=7}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

위 코드는 매우 직관적으로 보이지만, 실제로는 상당히 이상한 일이 일어납니다. 이 단계에서 우리는 아직 Gluon에게 입력이 어떤 차원을 갖는지를 알려주지 않은 상태에서 파라메터를 초기화하고 있습니다. 네트워크를 어떻게 정의하는지에 따라서 따라서 2가 될 수도, 2,000이 될 수도 있기 때문에, 이 시점에서 메모리를 미리 할당할 수도 없습니다. 파라메터 할당 및 초기화는 최초의 데이터가 네트워크에 입력되는 시점으로 미뤄지는 방식으로 동작합니다. 이런 방식으로 모든 설정을 미리 준비한 후, 사용자는 설정들에 대해서 걱정하지 않아도 됩니다. 단지 유의해야할 점은 파라메터들이 아직 초기화가되지 않았기 때문에 값을 바꾸는 일을 못한다는 것입니다.

##  Loss 함수 정의하기

Gluon의 `loss` 모듈은 다양한 loss 함수를 정의하고 있습니다. `loss` 모듈을 import 할 때 이름을 `gloss` 바꾸고, squared loss를 모델의 loss 함수로 사용하도록 합니다.

```{.python .input  n=8}
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()  # The squared loss is also known as the L2 norm loss
```

## 최적화 알고리즘 정의하기

마찬가지로 미니 배치에 대한 stocahstic gradient descent를 직접 구현할 필요가 없습니다. Gluon을 import 한 후, `Trainer` 인스턴스를 만들면서, learning rate가 0.03를 갖는 미니 배치 stochastic gradient descent를 최적화(optimization) 알고리즘으로 설정하면 됩니다. 이 최적화 알고리즘은 `add` 함수를 이용해서 `net` 인스턴스에 추가된 레이어들의 모든 파라메터들에 적용할 것인데, 이 파라메터들은 `collect_params` 함수를 통해서 얻습니다.

```{.python .input  n=9}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

## 학습

Gluon을 이용해서 모델을 표현하는 것이 보다 간결하다는 것을 눈치챘을 것입니다. 예를 들면, 파라메터들을 일일이 선언하지 않았고, loss 함수를 직접 정의하지 않았고, stochastic gradient descent를 직접 구현할 필요가 없습니다. 더욱 복잡한 형태의 모델을 다루기 시작하면, Gluon의 추상화를 사용하는 것이 얼마나 많은 이득이 되는지 알게 될 것입니다. 기본적인 것들이 모두 설정된 후에는 학습 loop은 앞에서 직접 구현한 것과 비슷합니다.

앞서 구현했던 모델 학습 단계를 다시 짚어보면, 전체 학습 데이터(train_data)를 정해진 회수 만큼 (epoch) 반복해서 학습을 수행합니다. 하나의 epoch은 다시 미니 배치로 나눠지는데 미니 배치는 입력과 입력에 해당하는 ground-truth label들로 구성됩니다. 각 미니 배치에서는 다음 단계들이 수행됩니다.

* 네트워크를 따라 forward pass 를 수행해서 예측 값`net(x)`  을 생성하고, loss `l` 을 계산합니다. 
* `l.backward()` 수행으로 backward pass를 실행해서 gradient들을 계산합니다.
* SGD optimizer를 수행해서 모델 파라매터들을 업데이트합니다. (`trainer.step()` 을 보면, 어떤 파라매터를 업데이트 해야하는지를 명시하고 있지 않고 학습할 데이터만 전달하고 있습니다. 이유는, trainer를 초기화할 때 이미 학습할 파라메터를 명시했기 때문입니다.)

학습 상태를 관찰하기 위해서, 매 epoch 마다 전체 features 들에 대해서 loss 값을 계산해서 출력합니다.

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

학습된 모델 파라메터들과 실제 모델 파라메터를 비교해봅니다. 모델의 파라메터들의 값을 확인하는 방법은, 우선  `net` 인스턴스로부터 레이어를 얻어내고, 그 레이어의 weight(`weight`) 변수와 bias (`bias`) 변수를 접근하는 것입니다. 학습된 파라메터들과 실제 파라메터들이 매우 비슷한 것을 볼 수 있습니다.

```{.python .input  n=12}
w = net[0].weight.data()
print('Error in estimating w', true_w.reshape(w.shape) - w)
b = net[0].bias.data()
print('Error in estimating b', true_b - b)
```

## 요약

* Gluon을 이용해서 모델을 매우 간단하게 구현할 수 있습니다.
* Gluon에서 `data` 모듈은 데이터 프로세싱하는 도구를, `nn` 모듈을 많은 종류의 뉴럴 네트워크 레이어들의 정의를 그리고 `loss` 모듈은 다양한 loss 함수를 제공합니다.
* MXNet의 `initializer` 모듈은 모델 파라메터를 초기화하는 다양한 방법을 제공합니다.
* 모델 파라메터의 차원(dimensionality)과 저장 공간은 할당은 실제 사용될 때까지 미뤄집니다. (따라서, 초기화되기 전에 파라메터를 접근하는 경우에 주의해야 합니다.)

## 문제

1. `l = loss(output, y)` 를  `l = loss(output, y).mean()` 로 바꿀 경우, `trainer.step(batch_size)` 를 `trainer.step(1)` 바꿔야합니다. 왜 일까요?

1.  `gluon.loss` 와 `init` 모듈에 어떤 loss 함수와 초기화 방법들이 포함되어 있는지 MXNet 문서를 읽어보세요. loss 함수를 Huber's loss로도 바꿔보세요.
2.  `dense.weight` 의 gradient를 어떻게 접근할 수 있을까요?

## 

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2333)

![](../img/qr_linear-regression-gluon.svg)
