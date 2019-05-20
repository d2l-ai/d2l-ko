# 다층 페셉트론(multilayer perceptron)의 간결한 구현

다층 페셉트론(multilayer perceptron, MLP)가 어떻게 작동하는지 이론적으로 배웠으니, 이제 직접 구현해보겠습니다. 우선 관련 패키지와 모듈을 import 합니다.

```{.python .input}
import sys
sys.path.insert(0, '..')

import d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
```

## 모델

Softmax 회귀 구현과 유일하게 다른 점은 한 개 대신에 두 개의 `Desse` (완전 연결) 층을 추가했다는 것입니다. 첫 번째는 256개의 은닉 유닛을 갖는 은닉 층이고, ReLU 활성화 함수를 사용합니다.

```{.python .input  n=5}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

위에서 처럼 우리는  `net.add()` 를 연속으로 여러 번 호출할 수 있고, 네트워크에 추가할 여러 층들을 전달하면서 한 번 호출하는 것도 가능하다는 것을 기억하세요. 즉, 위 코드를 `net.add(nn.Dense(256, activation='relu'), nn.Dense(10))` 로 동일하게 작성할 수 있습니다. 그리고 Gluon은 자동으로 각 층의 누락된 입력 차원을 추정한다는 것도 기억하세요.

Training the model follows the exact same steps as in our softmax regression implementation.

모델을 학습시키는 것은 softmax 회귀의 구현과 완전히 같은 절차로 이뤄집니다.

```{.python .input  n=6}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
```

## 연습문제

1. 은닉층(hidden layer)들을 더 추가해서 결과가 어떻게 변하는지 확인하세요.
1. 다른 활성화 함수(activation function)를 적용해보세요. 어떤 것이 가장 좋게 나오나요?
1. 가중치에 대한 초기화를 다르게 해보세요.

## QR 코드를 스캔해서 [논의하기](https://discuss.mxnet.io/t/2340)

![](../img/qr_mlp-gluon.svg)
