# 조밀하게 연결된 네트워크 (덴스넷)

ResNet은 심층 네트워크에서 함수를 매개 변수화하는 방법에 대한 관점을 크게 변경했습니다.*DenseNet* (고밀도 컨벌루션 네트워크) 는 이 :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`의 논리적 확장입니다.어떻게 도착하는지 이해하기 위해 수학을 조금 우회해 봅시다. 

## ResNet에서 덴스넷으로

함수에 대한 Taylor 확장을 상기하십시오.포인트 $x = 0$의 경우 다음과 같이 쓸 수 있습니다. 

$$f(x) = f(0) + f'(0) x + \frac{f''(0)}{2!}  x^2 + \frac{f'''(0)}{3!}  x^3 + \ldots.$$

핵심은 함수를 점점 더 높은 차수의 항으로 분해한다는 것입니다.비슷한 맥락에서 ResNet은 함수를 다음과 같이 분해합니다. 

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

즉, ResNet은 $f$를 단순한 선형 항과 더 복잡한 비선형 항으로 분해합니다.두 용어 이상의 정보를 캡처 (추가할 필요는 없음) 하려면 어떻게 해야 할까요?한 가지 해결책은 덴스넷 :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`였습니다. 

![The main difference between ResNet (left) and DenseNet (right) in cross-layer connections: use of addition and use of concatenation. ](../img/densenet-block.svg)
:label:`fig_densenet_block`

:numref:`fig_densenet_block`에서 볼 수 있듯이 ResNet과 DenseNet의 주요 차이점은 후자의 경우 출력이 추가되지 않고*연결* ($[,]$로 표시) 된다는 것입니다.결과적으로 점점 더 복잡해지는 함수 시퀀스를 적용한 후 $\mathbf{x}$에서 해당 값으로 매핑을 수행합니다. 

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2([\mathbf{x}, f_1(\mathbf{x})]), f_3([\mathbf{x}, f_1(\mathbf{x}), f_2([\mathbf{x}, f_1(\mathbf{x})])]), \ldots\right].$$

결국 이러한 모든 기능이 MLP에 결합되어 기능 수를 다시 줄입니다.구현 측면에서 이것은 매우 간단합니다. 용어를 추가하는 대신 용어를 연결합니다.DenseNet이라는 이름은 변수 간의 종속성 그래프가 상당히 밀집되어 있다는 사실에서 비롯됩니다.이러한 체인의 마지막 레이어는 이전의 모든 레이어에 밀집되어 있습니다.고밀도 연결은 :numref:`fig_densenet`에 나와 있습니다. 

![Dense connections in DenseNet.](../img/densenet.svg)
:label:`fig_densenet`

DenseNet을 구성하는 주요 구성 요소는*고밀도 블록* 및*전환 레이어*입니다.전자는 입력과 출력이 연결되는 방식을 정의하고 후자는 너무 크지 않도록 채널 수를 제어합니다. 

## [**조밀한 블록**]

DenseNet은 ResNet의 수정된 “배치 정규화, 활성화 및 컨볼루션” 구조를 사용합니다 (:numref:`sec_resnet`의 연습 참조).먼저, 이 컨볼루션 블록 구조를 구현합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

*dense 블록*은 각각 동일한 개수의 출력 채널을 사용하는 여러 컨벌루션 블록으로 구성됩니다.그러나 순방향 전파에서는 채널 차원에서 각 컨볼루션 블록의 입력과 출력을 연결합니다.

```{.python .input}
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
#@tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
#@tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

다음 예제에서는 10개의 출력 채널로 구성된 2개의 컨벌루션 블록을 사용하여 [**`DenseBlock` 인스턴스를 정의**] 합니다.3 채널의 입력을 사용하면 $3+2\times 10=23$ 채널의 출력을 얻을 수 있습니다.컨벌루션 블록 채널의 개수는 입력 채널 수에 비례하여 출력 채널 수의 증가를 제어합니다.이를 *성장률*이라고도 합니다.

```{.python .input}
blk = DenseBlock(2, 10)
blk.initialize()
X = np.random.uniform(size=(4, 3, 8, 8))
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab pytorch
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = DenseBlock(2, 10)
X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

## [**전환 레이어**]

밀도가 높은 각 블록은 채널 수를 증가시키기 때문에 채널을 너무 많이 추가하면 모델이 지나치게 복잡해집니다.*트랜지션 레이어*는 모델의 복잡성을 제어하는 데 사용됩니다.$1\times 1$ 컨벌루션 계층을 사용하여 채널 수를 줄이고 보폭이 2인 평균 풀링 계층의 높이와 너비를 절반으로 줄여 모델의 복잡성을 더욱 줄입니다.

```{.python .input}
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
#@tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

이전 예제에서 고밀도 블록의 출력에 10개의 채널이 있는 [**전환 레이어 적용**] 을 사용합니다.이렇게 하면 출력 채널 수가 10개로 줄어들고 높이와 너비는 절반으로 줄어듭니다.

```{.python .input}
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
#@tab pytorch
blk = transition_block(23, 10)
blk(Y).shape
```

```{.python .input}
#@tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

## [**고밀도 네트 모델**]

다음으로 DenseNet 모델을 구성합니다.DenseNet은 먼저 ResNet에서와 동일한 단일 컨벌루션 계층과 최댓값 풀링 계층을 사용합니다.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def block_1():
    return tf.keras.Sequential([
       tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
       tf.keras.layers.BatchNormalization(),
       tf.keras.layers.ReLU(),
       tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

그런 다음 ResNet에서 사용하는 잔여 블록으로 구성된 4개의 모듈과 유사하게 DenseNet은 4개의 고밀도 블록을 사용합니다.ResNet과 마찬가지로 각 고밀도 블록에 사용되는 컨벌루션 계층의 수를 설정할 수 있습니다.여기서는 :numref:`sec_resnet`의 ResNet-18 모델과 일치하도록 4로 설정했습니다.또한 고밀도 블록의 컨벌루션 계층에 대한 채널 수 (즉, 성장률) 를 32로 설정하여 각 고밀도 블록에 128개의 채널이 추가됩니다. 

ResNet에서는 보폭이 2인 잔차 블록만큼 각 모듈 사이의 높이와 너비가 줄어듭니다.여기에서는 전환 레이어를 사용하여 높이와 너비를 절반으로 줄이고 채널 수를 절반으로 줄입니다.

```{.python .input}
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that halves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        num_channels //= 2
        net.add(transition_block(num_channels))
```

```{.python .input}
#@tab pytorch
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that halves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
```

```{.python .input}
#@tab tensorflow
def block_2():
    net = block_1()
    # `num_channels`: the current number of channels
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # This is the number of output channels in the previous dense block
        num_channels += num_convs * growth_rate
        # A transition layer that halves the number of channels is added
        # between the dense blocks
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(TransitionBlock(num_channels))
    return net
```

ResNet과 마찬가지로 전역 풀링 계층과 완전 연결 계층이 끝에 연결되어 출력값을 생성합니다.

```{.python .input}
net.add(nn.BatchNorm(),
        nn.Activation('relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    net = block_2()
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(10))
    return net
```

## [**교육**]

여기서는 더 깊은 네트워크를 사용하고 있으므로 이 섹션에서는 계산을 단순화하기 위해 입력 높이와 너비를 224에서 96으로 줄입니다.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 요약

* 계층 간 연결의 경우 입력과 출력이 함께 추가되는 ResNet과 달리 DenseNet은 채널 차원에서 입력과 출력을 연결합니다.
* DenseNet을 구성하는 주요 구성 요소는 고밀도 블록과 전환 계층입니다.
* 네트워크를 구성할 때 채널 수를 다시 줄이는 전이 계층을 추가하여 차원을 제어해야 합니다.

## 연습문제

1. 전이 계층에서 최대 풀링이 아닌 평균 풀링을 사용하는 이유는 무엇입니까?
1. DenseNet 백서에서 언급한 장점 중 하나는 모델 매개 변수가 ResNet의 매개 변수보다 작다는 것입니다.왜 이런 경우일까요?
1. DenseNet이 비판받은 한 가지 문제는 높은 메모리 소비입니다.
    1. 이게 정말 사실인가요?실제 GPU 메모리 사용량을 확인하려면 입력 형태를 $224\times 224$로 변경해 보십시오.
    1. 메모리 소비를 줄이는 다른 방법을 생각해 볼 수 있습니까?프레임워크를 어떻게 변경해야 할까요?
1. 덴스넷 논문 :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`의 표 1에 제시된 다양한 덴스넷 버전을 구현합니다.
1. DenseNet 아이디어를 적용하여 MLP 기반 모델을 설계합니다.:numref:`sec_kaggle_house`의 주택 가격 예측 작업에 적용하십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/87)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/88)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/331)
:end_tab:
