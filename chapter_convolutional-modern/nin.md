# 네트워크 인 네트워크 (nIN)
:label:`sec_nin`

LeNet, AlexNet 및 VGG는 모두 공통된 설계 패턴을 공유합니다. 컨벌루션 및 풀링 계층의 시퀀스를 통해*공간* 구조를 활용하는 특징을 추출한 다음 완전 연결 계층을 통해 표현을 사후 처리합니다.AlexNet과 VGG의 LeNet의 개선 사항은 주로 이러한 이후 네트워크가 이러한 두 모듈을 확장하고 심화하는 방법에 있습니다.또는 프로세스 초기에 완전히 연결된 레이어를 사용하는 것을 상상할 수 있습니다.그러나 밀도가 높은 레이어를 부주의하게 사용하면 표현의 공간 구조가 완전히 포기될 수 있습니다.
*네트워크* (*Nin*) 블록의 네트워크는 대안을 제공합니다.
매우 간단한 통찰력을 기반으로 제안되었습니다. 즉, 각 픽셀의 채널에서 MLP를 별도로 :cite:`Lin.Chen.Yan.2013`로 사용하는 것입니다. 

## (**아홉 블록**)

컨벌루션 계층의 입력과 출력은 예제, 채널, 높이 및 너비에 해당하는 축을 가진 4차원 텐서로 구성됩니다.또한 완전히 연결된 계층의 입력과 출력은 일반적으로 예제 및 기능에 해당하는 2 차원 텐서입니다.nIn의 기본 개념은 각 픽셀 위치 (각 높이와 너비에 대해) 에 완전히 연결된 레이어를 적용하는 것입니다.각 공간 위치에 가중치를 연결하면 $1\times 1$ 컨벌루션 계층 (:numref:`sec_channels`에서 설명) 또는 각 픽셀 위치에서 독립적으로 작동하는 완전 연결 계층으로 생각할 수 있습니다.이를 확인하는 또 다른 방법은 공간 차원의 각 요소 (높이 및 너비) 를 예제와 동등하고 채널을 피처와 동등한 것으로 간주하는 것입니다. 

:numref:`fig_nin`는 VGG와 NiN 간의 주요 구조적 차이점과 해당 블록을 보여줍니다.nIN 블록은 하나의 컨벌루션 계층과 ReLU 활성화를 통해 픽셀당 완전 연결 계층 역할을 하는 두 개의 $1\times 1$ 컨벌루션 계층으로 구성됩니다.첫 번째 계층의 컨볼루션 창 모양은 일반적으로 사용자가 설정합니다.후속 창 쉐이프는 $1 \times 1$으로 고정됩니다. 

![Comparing architectures of VGG and NiN, and their blocks.](../img/nin.svg)
:width:`600px`
:label:`fig_nin`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def nin_block(num_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides,
                               padding=padding, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu')])
```

## [**닌 모델**]

원래 NiN 네트워크는 AlexNet 직후에 제안되었으며 분명히 영감을 얻었습니다.nIn은 창 모양이 $11\times 11$, $5\times 5$ 및 $3\times 3$인 컨벌루션 계층을 사용하며, 해당하는 출력 채널 수는 알렉스넷과 동일합니다.각 nIn 블록 뒤에는 보폭이 2이고 창 모양이 $3\times 3$인 최대 풀링 계층이 옵니다. 

nIN과 AlexNet의 한 가지 중요한 차이점은 nIn이 완전히 연결된 계층을 완전히 피한다는 것입니다.대신, nIn은 출력 채널 개수가 레이블 클래스의 개수와 같은 nIN 블록을 사용하고 그 뒤에*전역* 평균 풀링 계층을 사용하여 로짓으로 구성된 벡터를 생성합니다.NiN 설계의 장점 중 하나는 필요한 모델 매개 변수의 수를 크게 줄인다는 것입니다.그러나 실제로 이 설계에서는 모델 학습 시간을 늘려야 하는 경우가 있습니다.

```{.python .input}
net = nn.Sequential()
net.add(nin_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dropout(0.5),
        # There are 10 label classes
        nin_block(10, kernel_size=3, strides=1, padding=1),
        # The global average pooling layer automatically sets the window shape
        # to the height and width of the input
        nn.GlobalAvgPool2D(),
        # Transform the four-dimensional output into two-dimensional output
        # with a shape of (batch size, 10)
        nn.Flatten())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # There are 10 label classes
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # Transform the four-dimensional output into two-dimensional output with a
    # shape of (batch size, 10)
    nn.Flatten())
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        nin_block(96, kernel_size=11, strides=4, padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Dropout(0.5),
        # There are 10 label classes
        nin_block(10, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Reshape((1, 1, 10)),
        # Transform the four-dimensional output into two-dimensional output
        # with a shape of (batch size, 10)
        tf.keras.layers.Flatten(),
        ])
```

[**각 블록의 출력 형태**] 를 확인하기 위해 데이터 예제를 만듭니다.

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## [**교육**]

이전과 마찬가지로 패션 MNIST를 사용하여 모델을 교육합니다.nIN의 교육은 알렉스넷 및 VGG의 교육과 유사합니다.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 요약

* nIn은 컨벌루션 계층과 여러 개의 $1\times 1$개의 컨벌루션 계층으로 구성된 블록을 사용합니다.이것은 컨벌루션 스택 내에서 더 많은 픽셀당 비선형성을 허용하는 데 사용할 수 있습니다.
* nIn은 완전히 연결된 계층을 제거하고 채널 수를 원하는 출력 수 (예: Fashion-MNIST의 경우 10) 로 줄인 후 전역 평균 풀링 (즉, 모든 위치에 대한 합계) 으로 대체합니다.
* 완전히 연결된 레이어를 제거하면 과적합이 줄어듭니다.nIn은 매개 변수가 크게 적습니다.
* NiN 설계는 많은 후속 CNN 설계에 영향을 미쳤습니다.

## 연습문제

1. 초모수를 조정하여 분류 정확도를 높입니다.
1. nIN 블록에 두 개의 $1\times 1$개의 컨벌루션 계층이 있는 이유는 무엇입니까?그 중 하나를 제거한 다음 실험 현상을 관찰하고 분석하십시오.
1. NiN에 대한 리소스 사용량을 계산합니다.
    1. 매개 변수의 수는 몇 개입니까?
    1. 계산의 양은 얼마입니까?
    1. 훈련 중에 필요한 기억력은 얼마입니까?
    1. 예측 중에 필요한 메모리 양은 얼마입니까?
1. 한 번에 $384 \times 5 \times 5$ 표현을 $10 \times 5 \times 5$ 표현으로 줄일 때 발생할 수 있는 문제는 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/79)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/80)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/332)
:end_tab:
