# 드롭아웃
:label:`sec_dropout`

:numref:`sec_weight_decay`에서는 가중치의 $L_2$ 표준을 처벌하여 통계 모델을 정규화하는 고전적인 접근 방식을 도입했습니다.확률 론적 측면에서, 우리는 가중치가 평균 0의 가우스 분포에서 값을 가져 오는 사전 믿음을 가정한다고 주장하여이 기술을 정당화 할 수 있습니다.좀 더 직관적으로, 우리는 잠재적 인 가짜 연관성에 너무 많이 의존하지 않고 많은 기능들 사이에서 가중치를 확산하도록 모델을 권장한다고 주장 할 수 있습니다.

## 초과 피팅 재방문

예보다 더 많은 기능에 직면하면 선형 모형이 과도하게 적합하는 경향이 있습니다.그러나 기능보다 더 많은 예를 감안할 때, 우리는 일반적으로 선형 모델을 과도하게 적합하지 않을 수 있습니다.불행히도 선형 모델이 일반화되는 신뢰성은 비용이 많이 듭니다.순진하게 적용된 선형 모델은 피처 간의 상호 작용을 고려하지 않습니다.모든 피쳐에 대해 선형 모형은 문맥을 무시하고 양수 또는 음수 가중치를 지정해야 합니다.

전통적인 텍스트에서 일반화 가능성과 유연성 사이의 근본적인 긴장은* 바이어스 분산 트레이드 오프*로 설명됩니다.선형 모델은 높은 바이어스를 가지고 있습니다: 그들은 단지 함수의 작은 클래스를 나타낼 수 있습니다.그러나 이러한 모형은 분산이 낮습니다. 데이터의 여러 랜덤 표본에서 유사한 결과를 제공합니다.

깊은 신경망은 바이어스 분산 스펙트럼의 반대쪽 끝에 서식합니다.선형 모델과 달리 신경망은 각 피쳐를 개별적으로 보는 데 국한되지 않습니다.기능 그룹 간의 상호 작용을 배울 수 있습니다.예를 들어, 이메일에 함께 나타나는 “나이지리아”와 “Western Union”이 스팸을 나타내지만 별도로 스팸을 나타내지는 않는다고 추론할 수 있습니다.

기능보다 훨씬 많은 예가 있더라도 심층 신경망은 과도하게 적합 할 수 있습니다.2017 년에 한 연구자 그룹이 무작위로 표시된 이미지에 깊은 그물을 훈련함으로써 신경망의 극도의 유연성을 입증했습니다.입력을 출력에 연결하는 실제 패턴이 없음에도 불구하고 확률 적 그래디언트 하강에 의해 최적화 된 신경망이 훈련 세트의 모든 이미지에 완벽하게 레이블을 지정할 수 있음을 발견했습니다.이것이 무엇을 의미하는지 생각해 보십시오.레이블이 무작위로 균일하게 할당되고 10 개의 클래스가있는 경우 분류자는 홀드 아웃 데이터에 대해 10% 이상의 정확도를 수행 할 수 없습니다.여기서 일반화 격차는 무려 90% 입니다.우리의 모델이 너무 표현력이 많아서 이것을 심하게 과도하게 부당 할 수 있다면, 우리는 언제 그들을 과도하게 사용하지 않기를 기대해야합니까?

깊은 네트워크의 수수께끼 일반화 속성에 대한 수학적 토대는 열린 연구 질문으로 남아 있으며, 이론적으로 지향하는 독자가 주제를 더 깊이 파고 들도록 장려합니다.지금은 깊은 그물의 일반화를 경험적으로 개선하는 경향이있는 실용적인 도구에 대한 조사를 시작하겠습니다.

## 섭동을 통한 견고성

우리가 좋은 예측 모델에서 기대하는 것에 대해 간단히 생각해 봅시다.우리는 그것이 보이지 않는 데이터에 잘 어울리기를 원합니다.고전적 일반화 이론은 기차와 테스트 성능 사이의 격차를 줄이기 위해, 우리는 간단한 모델을 목표로해야한다고 제안합니다.단순성은 적은 수의 치수 형태로 올 수 있습니다.:numref:`sec_model_selection`에서 선형 모델의 단일형 기반 함수에 대해 논의할 때 이 내용을 살펴봅니다.또한 :numref:`sec_weight_decay`에서 체중 감량 ($L_2$ 정규화) 을 논의 할 때 보았듯이 매개 변수의 (역) 규범은 또한 단순성의 유용한 척도를 나타냅니다.단순성의 또 다른 유용한 개념은 부드러움, 즉 함수가 입력에 대한 작은 변화에 민감하지 않아야한다는 것입니다.예를 들어 이미지를 분류 할 때 픽셀에 임의의 노이즈를 추가하는 것은 대부분 무해해야합니다.

1995 년 크리스토퍼 주교는 입력 소음 훈련이 Tikhonov 정규화 :cite:`Bishop.1995`와 동일하다는 것을 증명했을 때이 아이디어를 공식화했습니다.이 작업은 함수가 매끄럽고 단순하다는 요구 사항과 입력의 섭동에 탄력적이라는 요구 사항 사이에 명확한 수학적 연결을 만들었습니다.

그런 다음 2014년, 스리바스타바 (Srivastava) 등 :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`는 비숍의 아이디어를 네트워크의 내부 계층에도 적용하는 방법에 대한 영리한 아이디어를 개발했습니다.즉, 그들은 훈련 중에 후속 계층을 계산하기 전에 네트워크의 각 레이어에 노이즈를 주입 할 것을 제안했습니다.그들은 많은 레이어가있는 깊은 네트워크를 훈련 할 때 노이즈를 주입하면 입출력 매핑에 부드러움이 적용된다는 것을 깨달았습니다.

*dropout*이라고 불리는 그들의 아이디어는 순방향 전파 중에 각 내부 레이어를 계산하는 동안 노이즈를 주입하는 것을 포함하며 신경망 교육을위한 표준 기술이되었습니다.우리가 문자 그대로 있기 때문에이 방법은* 드롭 아웃*라고합니다
*훈련 중 일부 뉴런을 드롭*.
훈련을 통해 각 반복에서 표준 드롭 아웃은 후속 계층을 계산하기 전에 각 계층의 노드 중 일부를 제로 아웃하는 것으로 구성됩니다.

분명히하기 위해, 우리는 주교에 대한 링크와 함께 우리 자신의 이야기를 부과하고 있습니다.드롭 아웃에 대한 원본 논문은 성적 복제에 대한 놀라운 비유를 통해 직감을 제공합니다.저자들은 신경망 과잉 피팅은 각 레이어가 이전 계층의 특정 활성화 패턴에 의존하는 상태를 특징으로하며, 이 조건*공동 적응*이라고 부릅니다.드롭아웃은 성적 재생이 공동 적응 유전자를 분해한다고 주장하는 것처럼 공동 적응을 중단한다고 주장합니다.

그런 다음 가장 중요한 과제는 이 소음을 주입하는 방법입니다.한 가지 아이디어는* 편향된* 방식으로 노이즈를 주입하여 각 레이어의 예상 값 (다른 레이어를 고정하는 동안) 이 노이즈가없는 값과 같도록하는 것입니다.

비숍의 작업에서 그는 선형 모델의 입력에 가우시안 노이즈를 추가했습니다.각 훈련 반복에서 그는 $\epsilon \sim \mathcal{N}(0,\sigma^2)$가 평균 0인 분포에서 샘플링 된 잡음을 입력 $\mathbf{x}$에 추가하여 교란된 점 $\mathbf{x}' = \mathbf{x} + \epsilon$을 산출했습니다.기대에 따라, $E[\mathbf{x}'] = \mathbf{x}$.

표준 드롭 아웃 정규화에서는 유지되지 않은 (삭제되지 않은) 노드의 분수로 정규화하여 각 레이어를 debiasing합니다.즉, *드롭 아웃 확률* $p$를 사용하면 각 중간 활성화 $h$은 다음과 같이 랜덤 변수 $h'$로 대체됩니다.

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ with probability } p \\
    \frac{h}{1-p} & \text{ otherwise}
\end{cases}
\end{aligned}
$$

설계 상, 기대는 변함없이, 즉 $E[h'] = h$로 유지됩니다.

## 실천 드롭아웃

:numref:`fig_mlp`에서 숨겨진 레이어와 5 개의 숨겨진 유닛으로 MLP를 회상하십시오.숨겨진 레이어에 드롭 아웃을 적용하면 각 숨겨진 단위를 $p$ 확률로 0으로 설정하면 결과는 원래 뉴런의 하위 집합 만 포함하는 네트워크로 볼 수 있습니다.:numref:`fig_dropout2`에서, 그리고 $h_2$와 $h_5$에서 제거됩니다.따라서 출력의 계산은 더 이상 $h_2$ 또는 $h_5$에 의존하지 않으며 역전파를 수행할 때 각각의 그라데이션도 사라집니다.이러한 방식으로 출력 레이어의 계산은 $h_1, \ldots, h_5$의 한 요소에 지나치게 의존할 수 없습니다.

![MLP before and after dropout.](../img/dropout2.svg)
:label:`fig_dropout2`

일반적으로 테스트 시간에 드롭아웃을 비활성화합니다.훈련 된 모델과 새로운 예제를 감안할 때 노드를 삭제하지 않으므로 정상화 할 필요가 없습니다.그러나 몇 가지 예외가 있습니다. 일부 연구자들은 신경망 예측의* 불확실성*을 추정하기위한 경험적 방법으로 테스트 시간에 드롭 아웃을 사용합니다. 예측이 여러 가지 드롭 아웃 마스크에 동의하면 네트워크가 더 자신감을 가지고 있다고 말할 수 있습니다.

## 처음부터 구현

단일 계층에 대한 드롭 아웃 함수를 구현하기 위해 우리는 Bernoulli (이진) 랜덤 변수에서 많은 샘플을 그려야합니다. 여기서 임의의 변수는 확률 $1-p$ 및 $0$ (드롭) 확률로 $0$ (드롭) 를 사용합니다 $p$.이를 구현하는 한 가지 쉬운 방법은 먼저 균일 분포 $U[0, 1]$에서 샘플을 그리는 것입니다.그런 다음 해당 샘플이 $p$보다 큰 노드를 유지하고 나머지는 삭제할 수 있습니다.

다음 코드에서는 텐서 입력 `X`의 요소를 확률로 `dropout`으로 삭제하는 7323614 함수를 구현합니다. 생존자를 `1.0-dropout`로 나눕니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return np.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return torch.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = (torch.Tensor(X.shape).uniform_(0, 1) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return tf.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

몇 가지 예에서 `dropout_layer` 함수를 테스트 할 수 있습니다.다음 코드 줄에서 입력 `X`를 드롭 아웃 작업을 통해 각각 0, 0.5 및 1로 전달합니다.

```{.python .input}
X = np.arange(16).reshape(2, 8)
print(dropout_layer(X, 0))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1))
```

```{.python .input}
#@tab pytorch
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

### 모델 매개변수 정의

다시 말하지만, 우리는 :numref:`sec_fashion_mnist`에 도입 된 패션 Mnist 데이터 세트로 작업합니다.각각 256 단위를 포함하는 두 개의 숨겨진 레이어가 있는 MLP를 정의합니다.

```{.python .input}
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens1))
b1 = np.zeros(num_hiddens1)
W2 = np.random.normal(scale=0.01, size=(num_hiddens1, num_hiddens2))
b2 = np.zeros(num_hiddens2)
W3 = np.random.normal(scale=0.01, size=(num_hiddens2, num_outputs))
b3 = np.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
```

```{.python .input}
#@tab tensorflow
num_outputs, num_hiddens1, num_hiddens2 = 10, 256, 256
```

### 모델 정의하기

아래 모델은 각 숨겨진 레이어의 출력에 드롭아웃을 적용합니다 (활성화 기능에 따라).각 레이어에 대해 드롭 아웃 확률을 별도로 설정할 수 있습니다.일반적인 추세는 입력 계층에 더 가깝게 낮은 드롭아웃 확률을 설정하는 것입니다.아래에서는 첫 번째 및 두 번째 숨겨진 레이어에 대해 각각 0.2 및 0.5로 설정합니다.드롭아웃은 훈련 중에만 활성화되도록 합니다.

```{.python .input}
dropout1, dropout2 = 0.2, 0.5

def net(X):
    X = X.reshape(-1, num_inputs)
    H1 = npx.relu(np.dot(X, W1) + b1)
    # Use dropout only when training the model
    if autograd.is_training():
        # Add a dropout layer after the first fully connected layer
        H1 = dropout_layer(H1, dropout1)
    H2 = npx.relu(np.dot(H1, W2) + b2)
    if autograd.is_training():
        # Add a dropout layer after the second fully connected layer
        H2 = dropout_layer(H2, dropout2)
    return np.dot(H2, W3) + b3
```

```{.python .input}
#@tab pytorch
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()

        self.num_inputs = num_inputs
        self.training = is_training

        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)

        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # Use dropout only when training the model
        if self.training == True:
            # Add a dropout layer after the first fully connected layer
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # Add a dropout layer after the second fully connected layer
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

```{.python .input}
#@tab tensorflow
dropout1, dropout2 = 0.2, 0.5

class Net(tf.keras.Model):
    def __init__(self, num_outputs, num_hiddens1, num_hiddens2):
        super().__init__()
        self.input_layer = tf.keras.layers.Flatten()
        self.hidden1 = tf.keras.layers.Dense(num_hiddens1, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(num_hiddens2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs, training=None):
        x = self.input_layer(inputs)
        x = self.hidden1(x)
        if training:
            x = dropout_layer(x, dropout1)
        x = self.hidden2(x)
        if training:
            x = dropout_layer(x, dropout2)
        x = self.output_layer(x)
        return x

net = Net(num_outputs, num_hiddens1, num_hiddens2)
```

### 교육 및 테스트

이는 앞서 설명한 MLP 교육 및 테스트와 유사합니다.

```{.python .input}
num_epochs, lr, batch_size = 10, 0.5, 256
loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr, batch_size = 10, 0.5, 256
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 간결한 구현

고급 API를 사용하면 완전히 연결된 각 레이어 다음에 `Dropout` 레이어를 추가하여 드롭 아웃 확률을 생성자의 유일한 인수로 전달하면됩니다.훈련 중에 `Dropout` 레이어는 지정된 드롭 아웃 확률에 따라 이전 레이어의 출력 (또는 후속 레이어에 대한 입력) 을 무작위로 삭제합니다.트레이닝 모드가 아닐 때는 `Dropout` 레이어는 테스트 중에 데이터를 통과합니다.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Dense(256, activation="relu"),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # Add a dropout layer after the first fully connected layer
    tf.keras.layers.Dropout(dropout1),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # Add a dropout layer after the second fully connected layer
    tf.keras.layers.Dropout(dropout2),
    tf.keras.layers.Dense(10),
])
```

다음으로 모델을 훈련하고 테스트합니다.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 요약

* 치수 수와 웨이트 벡터의 크기를 제어하는 것 외에도 드롭아웃은 과잉 피팅을 방지하는 또 다른 도구입니다.종종 그들은 공동으로 사용됩니다.
* 드롭아웃은 정품 인증 $h$를 예상 값이 $h$인 랜덤 변수로 대체합니다.
* 드롭아웃은 훈련 중에만 사용됩니다.

## 연습 문제

1. 첫 번째 및 두 번째 레이어에 대한 드롭아웃 확률을 변경하면 어떻게 됩니까?특히 두 레이어 모두에 대해 레이어를 전환하면 어떻게됩니까?이러한 질문에 답하고, 결과를 정량적으로 설명하며, 질적 요소를 요약하는 실험을 설계합니다.
1. 에포크 수를 늘리고 드롭 아웃을 사용할 때 얻은 결과를 사용하지 않을 때의 결과와 비교하십시오.
1. 드롭아웃이 적용되거나 적용되지 않을 때 숨겨진 각 레이어의 활성화 차이는 무엇입니까?두 모델 모두에서 시간이 지남에 따라 이 양이 어떻게 진화하는지 보여 주는 그림을 그립니다.
1. 테스트 시간에 드롭아웃이 일반적으로 사용되지 않는 이유는 무엇입니까?
1. 이 섹션의 모델을 예로 들어 드롭아웃 및 무게 감쇠 사용의 효과를 비교합니다.드롭아웃과 체중 감량을 동시에 사용하면 어떻게 됩니까?결과가 첨가됩니까?반품이 줄어들었습니까 (또는 더 나쁜)?그들은 서로를 취소합니까?
1. 활성화가 아닌 가중치 행렬의 개별 가중치에 드롭아웃을 적용하면 어떻게됩니까?
1. 표준 드롭아웃 기술과 다른 각 레이어에 무작위 노이즈를 주입하는 또 다른 기술을 개발합니다.Fashion-mnist 데이터 세트 (고정 아키텍처의 경우) 에서 드롭 아웃을 능가하는 메소드를 개발할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/100)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/261)
:end_tab:
