# 병렬 연결을 사용하는 네트워크 (GoogLeNet)
:label:`sec_googlenet`

2014년에는*구글넷*이 이미지넷 챌린지에서 우승하여 nIN의 강점과 반복되는 블록 :cite:`Szegedy.Liu.Jia.ea.2015`의 패러다임을 결합한 구조를 제안했습니다.이 논문의 한 가지 초점은 어떤 크기의 컨볼루션 커널이 가장 좋은지에 대한 질문을 다루는 것이었습니다.결국 이전의 인기 네트워크는 $1 \times 1$만큼 작고 $11 \times 11$의 큰 선택을 사용했습니다.이 백서의 한 가지 통찰력은 때때로 다양한 크기의 커널을 조합하여 사용하는 것이 유리할 수 있다는 것입니다.이 섹션에서는 원래 모델의 약간 단순화 된 버전을 제시하는 GoogLeNet을 소개합니다. 교육을 안정화하기 위해 추가되었지만 더 나은 학습 알고리즘을 사용할 수 있으므로 불필요한 몇 가지 임시 기능을 생략합니다. 

## (**인셉션 블록**)

GoogLeNet의 기본 컨볼 루션 블록은*Inception 블록*이라고하며, 바이러스 밈을 시작한 영화*Inception* (“우리는 더 깊이 갈 필요가 있습니다”) 의 인용문으로 인해 이름이 지정되었을 수 있습니다. 

![Structure of the Inception block.](../img/inception.svg)
:label:`fig_inception`

:numref:`fig_inception`에 표시된 것처럼 시작 블록은 네 개의 평행 경로로 구성됩니다.처음 세 경로는 창 크기가 $1\times 1$, $3\times 3$ 및 $5\times 5$인 컨벌루션 계층을 사용하여 다양한 공간 크기에서 정보를 추출합니다.가운데 두 경로는 입력에서 $1\times 1$ 컨벌루션을 수행하여 채널 수를 줄여 모델의 복잡성을 줄입니다.네 번째 경로는 $3\times 3$ 최대 풀링 계층을 사용하고 그 뒤에 $1\times 1$ 컨벌루션 계층을 사용하여 채널 수를 변경합니다.네 개의 패스는 모두 적절한 패딩을 사용하여 입력과 출력에 동일한 높이와 너비를 지정합니다.마지막으로 각 경로의 출력은 채널 차원을 따라 연결되어 블록의 출력을 구성합니다.Inception 블록의 일반적으로 조정되는 하이퍼파라미터는 레이어당 출력 채널 수입니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class Inception(nn.Block):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # Concatenate the outputs on the channel dimension
        return np.concatenate((p1, p2, p3, p4), axis=1)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # Concatenate the outputs on the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class Inception(tf.keras.Model):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')


    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # Concatenate the outputs on the channel dimension
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])
```

이 네트워크가 왜 그렇게 잘 작동하는지에 대한 직관을 얻으려면 필터 조합을 고려하십시오.다양한 필터 크기로 이미지를 탐색합니다.즉, 다양한 크기의 필터를 통해 다양한 범위의 세부 정보를 효율적으로 인식할 수 있습니다.동시에 필터마다 다른 양의 매개 변수를 할당 할 수 있습니다. 

## [**구글넷 모델**]

:numref:`fig_inception_full`에서 볼 수 있듯이 GoogLeNet은 총 9개의 시작 블록 스택과 글로벌 평균 풀링을 사용하여 추정치를 생성합니다.시작 블록 간의 최대 풀링은 차원성을 감소시킵니다.첫 번째 모듈은 알렉스넷 및 르넷과 유사합니다.블록 스택은 VGG에서 상속되며 전역 평균 풀링은 끝에 완전히 연결된 계층의 스택을 피합니다. 

![The GoogLeNet architecture.](../img/inception-full.svg)
:label:`fig_inception_full`

이제 GoogleNet을 하나씩 구현할 수 있습니다.첫 번째 모듈은 64채널 $7\times 7$ 컨벌루션 계층을 사용합니다.

```{.python .input}
b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

두 번째 모듈은 두 개의 컨벌루션 계층을 사용합니다. 첫째, 64채널 $1\times 1$ 컨벌루션 계층과 채널 수를 3배로 늘리는 $3\times 3$ 컨벌루션 계층입니다.시작 블록의 두 번째 경로에 해당합니다.

```{.python .input}
b2 = nn.Sequential()
b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
       nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b2():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation='relu'),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

세 번째 모듈은 두 개의 완전한 Inception 블록을 직렬로 연결합니다.첫 번째 인셉션 블록의 출력 채널 수는 $64+128+32+32=256$이고, 네 개의 경로 중 출력 채널 수 비율은 $64:128:32:32=2:4:1:1$입니다.두 번째 및 세 번째 경로는 먼저 입력 채널 수를 각각 $96/192=1/2$ 및 $16/192=1/12$로 줄인 다음 두 번째 컨벌루션 계층을 연결합니다.두 번째 인셉션 블록의 출력 채널 수는 $128+192+96+64=480$로 증가하고, 네 개의 경로 중 출력 채널 수 비율은 $128:192:96:64 = 4:6:3:2$입니다.두 번째 및 세 번째 경로는 먼저 입력 채널 수를 각각 $128/256=1/2$ 및 $32/256=1/8$로 줄입니다.

```{.python .input}
b3 = nn.Sequential()
b3.add(Inception(64, (96, 128), (16, 32), 32),
       Inception(128, (128, 192), (32, 96), 64),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b3():
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

네 번째 모듈은 더 복잡합니다.다섯 개의 인셉션 블록을 직렬로 연결하며 각각 $192+208+48+64=512$, $160+224+64+64=512$, $128+256+64+64=512$, $112+288+64+64=528$ 및 $256+320+128+128=832$ 출력 채널을 가지고 있습니다.이러한 경로에 할당된 채널 수는 세 번째 모듈의 채널 수와 유사합니다. 즉, 컨벌루션 계층이 $3\times 3$ 인 두 번째 경로는 가장 많은 수의 채널을 출력하고 그 뒤에 $1\times 1$ 컨벌루션 계층만 있는 첫 번째 경로, $5\times 5$ 컨벌루션 계층이 있는 세 번째 경로 및$3\times 3$ 최대 풀링 계층이 있는 네 번째 경로입니다.두 번째 및 세 번째 경로는 먼저 비율에 따라 채널 수를 줄입니다.이러한 비율은 다른 시작 블록에서 약간 다릅니다.

```{.python .input}
b4 = nn.Sequential()
b4.add(Inception(192, (96, 208), (16, 48), 64),
       Inception(160, (112, 224), (24, 64), 64),
       Inception(128, (128, 256), (24, 64), 64),
       Inception(112, (144, 288), (32, 64), 64),
       Inception(256, (160, 320), (32, 128), 128),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b4():
    return tf.keras.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

다섯 번째 모듈에는 $256+320+128+128=832$ 및 $384+384+128+128=1024$ 출력 채널이 있는 두 개의 인셉션 블록이 있습니다.각 경로에 할당 된 채널 수는 세 번째 및 네 번째 모듈의 채널 수와 동일하지만 특정 값이 다릅니다.다섯 번째 블록 뒤에는 출력 계층이 나옵니다.이 블록은 nIn에서와 마찬가지로 전역 평균 풀링 계층을 사용하여 각 채널의 높이와 너비를 1로 변경합니다.마지막으로 출력값을 2차원 배열로 변환한 다음 출력값 수가 레이블 클래스 수인 완전 연결 계층으로 변환합니다.

```{.python .input}
b5 = nn.Sequential()
b5.add(Inception(256, (160, 320), (32, 128), 128),
       Inception(384, (192, 384), (48, 128), 128),
       nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))
```

```{.python .input}
#@tab pytorch
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```

```{.python .input}
#@tab tensorflow
def b5():
    return tf.keras.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten()
    ])
# Recall that this has to be a function that will be passed to
# `d2l.train_ch6()` so that model building/compiling need to be within
# `strategy.scope()` in order to utilize the CPU/GPU devices that we have
def net():
    return tf.keras.Sequential([b1(), b2(), b3(), b4(), b5(),
                                tf.keras.layers.Dense(10)])
```

GoogLeNet 모델은 계산적으로 복잡하므로 VGG에서와 같이 채널 수를 수정하기가 쉽지 않습니다.[**Fashion-MNIST에 대한 적절한 교육 시간을 갖기 위해 입력 높이와 너비를 224에서 96.로 줄입니다**] 이렇게 하면 계산이 간소화됩니다.다양한 모듈 간의 출력 모양 변화는 아래에 설명되어 있습니다.

```{.python .input}
X = np.random.uniform(size=(1, 1, 96, 96))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform(shape=(1, 96, 96, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

## [**교육**]

이전과 마찬가지로 패션-MNIST 데이터세트를 사용하여 모델을 학습합니다.교육 절차를 호출하기 전에 $96 \times 96$ 픽셀 해상도로 변환합니다.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 요약

* Inception 블록은 네 개의 경로가 있는 서브네트워크와 동일합니다.서로 다른 창 모양의 컨벌루션 계층과 최대 풀링 계층을 통해 병렬로 정보를 추출합니다. $1 \times 1$ 컨벌루션은 픽셀당 수준에서 채널 차원을 줄입니다.최대 풀링은 해상도를 감소시킵니다.
* GoogLeNet은 잘 설계된 여러 Inception 블록을 다른 레이어와 직렬로 연결합니다.Inception 블록에 할당된 채널 수의 비율은 ImageNet 데이터 세트에 대한 수많은 실험을 통해 얻어집니다.
* GoogLeNet과 후속 버전은 ImageNet에서 가장 효율적인 모델 중 하나였으며 계산 복잡성은 줄이면서 유사한 테스트 정확도를 제공합니다.

## 연습문제

1. 구글넷에는 몇 가지 반복이 있습니다.구현 및 실행을 시도합니다.그 중 일부는 다음과 같습니다.
    * 나중에 :numref:`sec_batch_norm`에 설명된 대로 배치 정규화 계층 :cite:`Ioffe.Szegedy.2015`를 추가합니다.
    * 인셉션 블록 :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`를 조정합니다.
    * 모델 정규화 :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`에 레이블 평활화를 사용합니다.
    * :numref:`sec_resnet`의 뒷부분에 설명된 대로 잔여 연결 :cite:`Szegedy.Ioffe.Vanhoucke.ea.2017`에 포함시킵니다.
1. GoogleNet이 작동하는 최소 이미지 크기는 얼마입니까?
1. 알렉스넷, VGG, Nin의 모델 모수 크기를 구글넷과 비교합니다.후자의 두 네트워크 아키텍처가 모델 파라미터 크기를 크게 줄이는 방법은 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/81)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/82)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/316)
:end_tab:
