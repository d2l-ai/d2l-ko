# 블록을 사용하는 네트워크 (VGG)
:label:`sec_vgg`

AlexNet은 심층 CNN이 좋은 결과를 얻을 수 있다는 경험적 증거를 제공했지만 후속 연구자들이 새로운 네트워크를 설계 할 수 있도록 안내하는 일반적인 템플릿을 제공하지 않았습니다.다음 섹션에서는 심층 네트워크를 설계하는 데 일반적으로 사용되는 몇 가지 휴리스틱 개념을 소개합니다. 

이 분야의 진전은 엔지니어가 트랜지스터를 배치하는 것에서 논리 요소, 논리 블록으로 전환한 칩 설계를 반영합니다.마찬가지로 신경망 아키텍처의 설계는 점점 더 추상적으로 성장했으며 연구자들은 개별 뉴런의 관점에서 전체 레이어로, 이제는 블록, 반복되는 레이어 패턴으로 이동했습니다. 

블록을 사용한다는 아이디어는 옥스퍼드 대학교의 [비주얼 지오메트리 그룹](http://www.robots.ox.ac.uk/~vgg/) (VGG) 에서 시조명으로*VGG* 네트워크에서 처음 등장했습니다.루프와 서브루틴을 사용하면 최신 딥 러닝 프레임워크를 사용하여 코드에서 이러한 반복되는 구조를 쉽게 구현할 수 있습니다. 

## (**VGG 블록**)
:label:`subsec_vgg-blocks`

기존 CNN의 기본 구성 요소는 다음과 같은 시퀀스입니다. (i) 해상도를 유지하기 위해 패딩이 있는 컨벌루션 계층, (ii) ReLU와 같은 비선형성, (iii) 최대 풀링 계층과 같은 풀링 계층.하나의 VGG 블록은 일련의 컨벌루션 계층과 공간 다운샘플링을 위한 최댓값 풀링 계층으로 구성됩니다.원본 VGG 논문 :cite:`Simonyan.Zisserman.2014`에서 저자는 패딩이 1 (높이와 너비 유지) 인 $3\times3$ 커널과 보폭이 2 인 $2 \times 2$ 최대 풀링 (각 블록 후 해상도를 절반으로 줄임) 이있는 컨볼루션을 사용했습니다.아래 코드에서는 하나의 VGG 블록을 구현하기 위해 `vgg_block`라는 함수를 정의합니다.

:begin_tab:`mxnet,tensorflow`
이 함수는 컨벌루션 계층 수 `num_convs` 및 출력 채널 수 `num_channels`에 해당하는 두 개의 인수를 취합니다.
:end_tab:

:begin_tab:`pytorch`
이 함수는 컨벌루션 계층 수 `num_convs`, 입력 채널 수 `in_channels` 및 출력 채널 수 `out_channels`에 해당하는 세 개의 인수를 취합니다.
:end_tab:

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,
                                    padding='same',activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

## [**VGG 네트워크**]

AlexNet 및 LeNet과 마찬가지로 VGG 네트워크는 두 부분으로 나눌 수 있습니다. 첫 번째 부분은 대부분 컨벌루션 계층과 풀링 계층으로 구성되고 두 번째 부분은 완전 연결 계층으로 구성됩니다.이것은 :numref:`fig_vgg`에 묘사되어 있습니다. 

![From AlexNet to VGG that is designed from building blocks.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

네트워크의 컨벌루션 부분은 :numref:`fig_vgg` (`vgg_block` 함수에도 정의됨) 의 여러 VGG 블록을 연속적으로 연결합니다.다음 변수 `conv_arch`은 튜플 목록 (블록당 하나) 으로 구성되며, 각 튜플에는 컨벌루션 계층 수와 출력 채널 수, 즉 `vgg_block` 함수를 호출하는 데 필요한 인수입니다.VGG 네트워크의 완전히 연결된 부분은 AlexNet에서 다루는 부분과 동일합니다. 

원래 VGG 네트워크에는 5개의 컨벌루션 블록이 있었는데, 그 중 처음 두 개에는 각각 하나의 컨벌루션 계층이 있고 후자 3개에는 각각 두 개의 컨벌루션 계층이 있습니다.첫 번째 블록에는 64개의 출력 채널이 있으며 각 후속 블록은 512에 도달할 때까지 출력 채널 수를 두 배로 늘립니다.이 네트워크는 8개의 컨벌루션 계층과 3개의 완전 연결 계층을 사용하므로 종종 VGG-11 이라고 합니다.

```{.python .input}
#@tab all
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

다음 코드는 VGG-11 을 구현합니다.이것은 `conv_arch`를 통해 for 루프를 실행하는 간단한 문제입니다.

```{.python .input}
def vgg(conv_arch):
    net = nn.Sequential()
    # The convolutional part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

net = vgg(conv_arch)
```

```{.python .input}
#@tab pytorch
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # The convolutional part
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # The fully-connected part
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

```{.python .input}
#@tab tensorflow
def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    # The convulational part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)]))
    return net

net = vgg(conv_arch)
```

다음으로 높이와 너비가 224인 단일 채널 데이터 예제를 구성하여 [**각 레이어의 출력 형태를 관찰**] 합니다.

```{.python .input}
net.initialize()
X = np.random.uniform(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for blk in net.layers:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t', X.shape)
```

보시다시피 각 블록에서 높이와 너비를 절반으로 줄이고 마침내 높이와 너비가 7에 도달한 다음 네트워크의 완전히 연결된 부분에서 처리하기 위해 표현을 병합합니다. 

## 트레이닝

[**VGG-11 은 AlexNet보다 계산량이 많기 때문에 채널 수가 적은 네트워크를 구성합니다.**] 이는 패션-MNIST 교육에 충분합니다.

```{.python .input}
#@tab mxnet, pytorch
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

```{.python .input}
#@tab tensorflow
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# Recall that this has to be a function that will be passed to
# `d2l.train_ch6()` so that model building/compiling need to be within
# `strategy.scope()` in order to utilize the CPU/GPU devices that we have
net = lambda: vgg(small_conv_arch)
```

약간 더 큰 학습률을 사용하는 것 외에도 [**모델 학습**] 프로세스는 :numref:`sec_alexnet`의 AlexNet의 프로세스와 유사합니다.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 요약

* VGG-11 은 재사용 가능한 컨벌루션 블록을 사용하여 네트워크를 구축합니다.각 블록의 컨벌루션 계층과 출력 채널 수의 차이로 서로 다른 VGG 모델을 정의할 수 있습니다.
* 블록을 사용하면 네트워크 정의를 매우 간결하게 표현할 수 있습니다.복잡한 네트워크를 효율적으로 설계할 수 있습니다.
* VGG 논문에서 Simonyan과 Ziserman은 다양한 아키텍처를 실험했습니다.특히, 그들은 깊고 좁은 컨볼루션의 여러 레이어 (즉, $3 \times 3$) 가 더 넓은 컨볼루션의 적은 레이어보다 더 효과적이라는 것을 발견했습니다.

## 연습문제

1. 레이어의 크기를 인쇄할 때 11이 아닌 8개의 결과만 볼 수 있었습니다.나머지 3개의 레이어 정보는 어디로 갔습니까?
1. AlexNet과 비교할 때 VGG는 계산 측면에서 훨씬 느리고 GPU 메모리도 더 많이 필요합니다.그 이유를 분석하십시오.
1. 패션-MNIST에서 이미지의 높이와 너비를 224에서 96으로 변경해 보세요.이것이 실험에 어떤 영향을 미칩니 까?
1. VGG-16 또는 VGG-19 과 같은 다른 일반 모델을 구성하려면 VGG 용지 :cite:`Simonyan.Zisserman.2014`의 표 1을 참조하십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/277)
:end_tab:
