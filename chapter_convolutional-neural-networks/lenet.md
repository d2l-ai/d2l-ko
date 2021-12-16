# 컨벌루션 뉴럴 네트워크 (LeNet)
:label:`sec_lenet`

이제 모든 기능을 갖춘 CNN을 조립하는 데 필요한 모든 재료가 있습니다.이미지 데이터와의 초기 만남에서 패션-MNIST 데이터 세트의 의류 사진에 소프트맥스 회귀 모델 (:numref:`sec_softmax_scratch`) 과 MLP 모델 (:numref:`sec_mlp_scratch`) 을 적용했습니다.이러한 데이터를 소프트맥스 회귀 및 MLP에 적합하게 만들기 위해 먼저 $28\times28$ 행렬의 각 이미지를 고정 길이 $784$차원 벡터로 평탄화 한 다음 완전히 연결된 계층으로 처리했습니다.이제 컨벌루션 계층에 대한 핸들이 생겼으므로 이미지에 공간 구조를 유지할 수 있습니다.완전히 연결된 계층을 컨벌루션 계층으로 대체하는 또 다른 이점으로, 훨씬 적은 수의 매개 변수가 필요한 더 간결한 모델을 사용할 수 있습니다. 

이 섹션에서는 컴퓨터 비전 작업에 대한 성능에 대한 폭 넓은 관심을 끌기 위해 최초로 출판 된 CNN 중 하나인*LeNet*을 소개합니다.이 모델은 이미지 :cite:`LeCun.Bottou.Bengio.ea.1998`에서 손으로 쓴 숫자를 인식하기 위해 AT&T Bell Labs의 연구원이었던 Yann LeCun에 의해 소개되었습니다 (그리고 이름을 따서 명명되었습니다).이 연구는 기술 개발에 대한 10년간의 연구의 정점을 나타냅니다.1989년 LeCun은 역전파를 통해 CNN을 성공적으로 훈련시키는 첫 번째 연구를 발표했습니다. 

당시 LeNet은 지도 학습에서 지배적 인 접근 방식이었던 서포트 벡터 머신의 성능과 일치하는 뛰어난 결과를 달성했습니다.LeNet은 결국 ATM 기계에서 예금을 처리하기 위해 숫자를 인식하도록 조정되었습니다.오늘날까지 일부 ATM은 얀과 그의 동료 레온 보투가 1990 년대에 쓴 코드를 여전히 실행하고 있습니다! 

## 르넷

높은 수준에서 (**LeNet (LeNet-5) 는 (i) 두 개의 컨벌루션 계층으로 구성된 컨벌루션 인코더, (ii) 완전히 연결된 세 개의 레이어로 구성된 고밀도 블록**) 의 두 부분으로 구성됩니다. 아키텍처는 :numref:`img_lenet`에 요약되어 있습니다. 

![Data flow in LeNet. The input is a handwritten digit, the output a probability over 10 possible outcomes.](../img/lenet.svg)
:label:`img_lenet`

각 컨벌루션 블록의 기본 단위는 컨벌루션 계층, 시그모이드 활성화 함수 및 후속 평균 풀링 연산입니다.RELU와 최대 풀링이 더 잘 작동하지만 1990 년대에는 이러한 발견이 아직 이루어지지 않았습니다.각 컨벌루션 계층은 $5\times 5$ 커널과 시그모이드 활성화 함수를 사용합니다.이러한 레이어는 공간적으로 정렬된 입력을 여러 2차원 피처 맵에 매핑하여 일반적으로 채널 수를 늘립니다.첫 번째 컨벌루션 계층에는 6개의 출력 채널이 있고, 두 번째 컨벌루션 계층에는 16개의각 $2\times2$ 풀링 연산 (스트라이드 2) 은 공간 다운샘플링을 통해 차원성을 $4$의 계수만큼 줄입니다.컨벌루션 블록은 (배치 크기, 채널 수, 높이, 너비) 로 지정된 형태로 출력값을 방출합니다. 

컨벌루션 블록의 출력값을 dense 블록으로 전달하려면 미니배치의 각 예제를 플랫화해야 합니다.다시 말해, 이 4차원 입력을 완전히 연결된 계층에서 기대하는 2차원 입력으로 변환합니다. 다시 말해, 원하는 2차원 표현은 첫 번째 차원을 사용하여 미니배치의 예제를 인덱싱하고 두 번째 차원을 사용하여 평면 벡터 표현을 제공합니다.각 예제의.LeNet의 고밀도 블록에는 각각 120, 84 및 10개의 출력을 가진 3개의 완전히 연결된 레이어가 있습니다.아직 분류를 수행하고 있기 때문에 10차원 출력 계층은 가능한 출력 클래스의 개수에 해당합니다. 

LeNet 내부에서 일어나는 일을 진정으로 이해하는 시점에 도달하는 데 약간의 작업이 필요했을 수 있지만, 다음 코드 스 니펫을 통해 최신 딥 러닝 프레임 워크로 이러한 모델을 구현하는 것이 매우 간단하다는 것을 확신 할 수 있기를 바랍니다.`Sequential` 블록을 인스턴스화하고 적절한 레이어를 함께 연결하기만 하면 됩니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        # `Dense` will transform an input of the shape (batch size, number of
        # channels, height, width) into an input of the shape (batch size,
        # number of channels * height * width) automatically by default
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])
```

우리는 최종 레이어에서 가우스 활성화를 제거하여 원래 모델로 약간의 자유를 얻었습니다.이 외에도 이 네트워크는 원래 LeNet-5 아키텍처와 일치합니다. 

단일 채널 (흑백) $28 \times 28$ 이미지를 네트워크를 통해 전달하고 각 계층에서 출력 모양을 인쇄하면 [**모델 검사**] 를 통해 해당 작업이 :numref:`img_lenet_vert`에서 기대하는 것과 일치하는지 확인할 수 있습니다. 

![Compressed notation for LeNet-5.](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
X = np.random.uniform(size=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 28, 28, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
```

컨벌루션 블록 전체의 각 계층에서 표현의 높이와 너비가 줄어듭니다 (이전 계층과 비교).첫 번째 컨벌루션 계층은 $5 \times 5$ 커널을 사용할 때 발생할 수 있는 높이와 너비의 감소를 보상하기 위해 2픽셀의 패딩을 사용합니다.반면, 두 번째 컨벌루션 계층은 패딩을 사용하지 않으므로 높이와 너비는 모두 4픽셀씩 줄어듭니다.계층 스택을 올라가면 채널 수가 입력값의 1에서 첫 번째 컨벌루션 계층 후 6으로, 두 번째 컨벌루션 계층 뒤의 16으로 레이어 오버 레이어 수가 증가합니다.그러나 각 풀링 계층은 높이와 너비의 절반입니다.마지막으로, 완전히 연결된 각 계층은 차원을 줄여 최종적으로 차원이 클래스 수와 일치하는 출력값을 방출합니다. 

## 트레이닝

이제 모델을 구현했으므로 [**실험을 실행하여 LeNet이 Fashion-MNist에서 어떻게 운임하는지 확인해보자**].

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

CNN은 매개 변수가 적지 만 각 매개 변수가 더 많은 곱셈에 참여하기 때문에 비슷한 수준의 MLP보다 계산 비용이 더 많이 들 수 있습니다.GPU에 액세스할 수 있는 경우 훈련 속도를 높이기 위해 GPU를 실행하기에 좋은 시기가 될 수 있습니다.

:begin_tab:`mxnet, pytorch`
평가를 위해 :numref:`sec_softmax_scratch`에서 설명한 [**`evaluate_accuracy` 함수를 약간 수정**] 해야 합니다.전체 데이터셋은 메인 메모리에 있으므로 모델이 GPU를 사용하여 데이터세트로 계산하기 전에 GPU 메모리에 복사해야 합니다.
:end_tab:

```{.python .input}
def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    if not device:  # Query the first device where the first parameter is on
        device = list(net.collect_params().values())[0].list_ctx()[0]
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

또한 [**GPU를 처리하기 위해 훈련 함수를 업데이트해야합니다.**] :numref:`sec_softmax_scratch`에 정의된 `train_epoch_ch3`와 달리, 이제 앞으로 및 뒤로 전파하기 전에 데이터의 각 미니 배치를 지정된 장치 (GPU) 로 이동해야 합니다. 

훈련 함수 `train_ch6`은 :numref:`sec_softmax_scratch`에 정의된 `train_ch3`과 유사합니다.앞으로 많은 계층이 있는 네트워크를 구현할 예정이므로 주로 상위 수준 API에 의존할 것입니다.다음 학습 함수는 상위 수준 API에서 생성된 모델을 입력으로 가정하고 그에 따라 최적화됩니다.:numref:`subsec_xavier`에 도입된 Xavier 초기화를 사용하여 `device` 인수로 표시된 장치에서 모델 매개 변수를 초기화합니다.MLP와 마찬가지로 손실 함수는 교차 엔트로피이며 미니 배치 확률 경사 하강을 통해 최소화합니다.각 Epoch를 실행하는 데 수십 초가 걸리기 때문에 훈련 손실을 더 자주 시각화합니다.

```{.python .input}
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            # Here is the major difference from `d2l.train_epoch_ch3`
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab tensorflow
class TrainCallback(tf.keras.callbacks.Callback):  #@save
    """A callback to visiualize the training progress."""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = d2l.Timer()
        self.animator = d2l.Animator(
            xlabel='epoch', xlim=[1, num_epochs], legend=[
                'train loss', 'train acc', 'test acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name
    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()
    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(
            self.test_iter, verbose=0, return_dict=True)['accuracy']
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.animator.add(epoch + 1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')

#@save
def train_ch6(net_fn, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return net
```

[**이제 LeNet-5 모델을 학습시키고 평가해 보겠습니다.**]

```{.python .input}
#@tab all
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 요약

* CNN은 컨벌루션 계층을 사용하는 네트워크입니다.
* CNN에서는 컨벌루션, 비선형 및 (종종) 풀링 연산을 인터리브합니다.
* CNN에서 컨벌루션 계층은 일반적으로 채널 수를 늘리면서 표현의 공간 해상도를 점진적으로 낮추도록 배열됩니다.
* 전통적인 CNN에서, 컨벌루션 블록에 의해 인코딩된 표현은 출력을 방출하기 전에 하나 이상의 완전 연결 계층에 의해 처리된다.
* LeNet은 틀림없이 이러한 네트워크의 첫 번째 성공적인 배포였습니다.

## 연습문제

1. 평균 풀링을 최대 풀링으로 바꿉니다.어떻게 되나요?
1. 정확성을 높이기 위해 LeNet을 기반으로 보다 복잡한 네트워크를 구축해 보십시오.
    1. 컨볼루션 창 크기를 조정합니다.
    1. 출력 채널 수를 조정합니다.
    1. 활성화 기능 (예: ReLU) 을 조정합니다.
    1. 컨벌루션 계층의 수를 조정합니다.
    1. 완전 연결 레이어 수를 조정합니다.
    1. 학습률 및 기타 훈련 세부 사항 (예: 초기화 및 Epoch 수) 을 조정합니다.
1. 원래 MNIST 데이터셋에서 개선된 네트워크를 사용해 보십시오.
1. 다양한 입력 (예: 스웨터 및 코트) 에 대한 LeNet의 첫 번째 및 두 번째 계층의 활성화를 표시합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/275)
:end_tab:
