# 잔여 네트워크 (ResNet)
:label:`sec_resnet`

점점 더 심층적인 네트워크를 설계함에 따라 계층을 추가하면 네트워크의 복잡성과 표현력이 어떻게 증가하는지 이해하는 것이 필수적입니다.더 중요한 것은 계층을 추가하면 네트워크가 단순히 다른 것이 아니라 표현력이 더 높아지는 네트워크를 설계하는 기능입니다.진전을 이루기 위해서는 약간의 수학이 필요합니다. 

## 함수 클래스

특정 네트워크 아키텍처 (학습률 및 기타 하이퍼파라미터 설정 포함) 가 도달할 수 있는 함수 클래스인 $\mathcal{F}$를 생각해 보십시오.즉, 모든 $f \in \mathcal{F}$에는 적절한 데이터 세트에 대한 교육을 통해 얻을 수 있는 몇 가지 매개 변수 집합 (예: 가중치 및 편향) 이 있습니다.$f^*$가 우리가 정말로 찾고 싶은 “진실” 함수라고 가정해 봅시다.$\mathcal{F}$에 있다면 우리는 몸매가 좋지만 일반적으로 운이 좋지는 않을 것입니다.대신 $\mathcal{F}$ 내에서 가장 좋은 방법 인 $f^*_\mathcal{F}$을 찾으려고 노력할 것입니다.예를 들어 기능 $\mathbf{X}$와 레이블이 $\mathbf{y}$인 데이터셋이 주어진 경우 다음 최적화 문제를 해결하여 찾을 수 있습니다. 

$$f^*_\mathcal{F} \stackrel{\mathrm{def}}{=} \mathop{\mathrm{argmin}}_f L(\mathbf{X}, \mathbf{y}, f) \text{ subject to } f \in \mathcal{F}.$$

다르고 더 강력한 아키텍처 $\mathcal{F}'$를 설계하면 더 나은 결과를 얻을 수 있다고 가정하는 것이 합리적입니다.즉, $f^*_{\mathcal{F}'}$이 $f^*_{\mathcal{F}}$보다 “더 낫다”고 예상할 수 있습니다.그러나 $\mathcal{F} \not\subseteq \mathcal{F}'$의 경우 이러한 일이 발생한다는 보장은 없습니다.실제로 $f^*_{\mathcal{F}'}$은 더 나빠질 수 있습니다.:numref:`fig_functionclasses`에서 설명했듯이 중첩되지 않은 함수 클래스의 경우 더 큰 함수 클래스가 항상 “실제” 함수 $f^*$에 가까워지는 것은 아닙니다.예를 들어, :numref:`fig_functionclasses`의 왼쪽에서는 $\mathcal{F}_3$가 $\mathcal{F}_1$보다 $f^*$에 더 가깝지만 $\mathcal{F}_6$은 멀리 이동하며 복잡성을 더 늘리면 거리가 $f^*$에서 줄어들 수 있다는 보장은 없습니다.:numref:`fig_functionclasses`의 오른쪽에 $\mathcal{F}_1 \subseteq \ldots \subseteq \mathcal{F}_6$인 중첩 함수 클래스를 사용하면 중첩되지 않은 함수 클래스에서 앞서 언급한 문제를 피할 수 있습니다. 

![For non-nested function classes, a larger (indicated by area) function class does not guarantee to get closer to the "truth" function ($f^*$). This does not happen in nested function classes.](../img/functionclasses.svg)
:label:`fig_functionclasses`

따라서 더 큰 함수 클래스가 더 작은 함수 클래스를 포함하는 경우에만 함수를 늘리면 네트워크의 표현력이 엄격하게 증가한다는 것이 보장됩니다.심층 신경망의 경우 새로 추가된 계층을 항등 함수 $f(\mathbf{x}) = \mathbf{x}$로 훈련시킬 수 있다면 새 모델은 원래 모델만큼 효과적입니다.새 모델이 훈련 데이터셋을 피팅하는 더 나은 솔루션을 얻을 수 있으므로 레이어를 추가하면 훈련 오류를 더 쉽게 줄일 수 있습니다. 

이것은 He et al. 이 매우 심층적 인 컴퓨터 비전 모델 :cite:`He.Zhang.Ren.ea.2016`에서 작업 할 때 고려한 질문입니다.제안된*잔여 네트워크* (*Resnet*) 의 핵심은 모든 추가 계층이 항등 함수를 요소 중 하나로 더 쉽게 포함해야 한다는 생각입니다.이러한 고려 사항은 다소 심오하지만 놀랍도록 간단한 해인 잔차 블록*으로 이어졌습니다.이를 통해 ResNet은 2015년 이미지넷 대규모 시각 인식 챌린지에서 우승했습니다.이 설계는 심층 신경망을 구축하는 방법에 큰 영향을 미쳤습니다. 

## (**잔여 블록**)

:numref:`fig_residual_block`에 묘사된 것처럼 신경망의 로컬 부분에 초점을 맞추겠습니다.입력을 $\mathbf{x}$으로 나타냅니다.학습을 통해 얻고 싶은 기본 매핑이 $f(\mathbf{x})$이라고 가정합니다. 이 매핑은 맨 위에 있는 활성화 함수에 대한 입력으로 사용됩니다.:numref:`fig_residual_block`의 왼쪽에 있는 점선 상자 내의 부분은 매핑 $f(\mathbf{x})$을 직접 학습해야 합니다.오른쪽에서 점선 상자 내의 부분은*잔차 매핑* $f(\mathbf{x}) - \mathbf{x}$를 학습해야 합니다. 이 매핑은 잔차 블록이 이름을 파생하는 방식입니다.아이덴티티 매핑 $f(\mathbf{x}) = \mathbf{x}$이 원하는 기본 매핑인 경우 잔차 매핑을 더 쉽게 배울 수 있습니다. 점선 상자 내의 상위 가중치 계층 (예: 완전 연결 계층 및 컨벌루션 계층) 의 가중치와 편향을 0으로 푸시하기만 하면 됩니다.:numref:`fig_residual_block`의 오른쪽 그림은 ResNet의*잔여 블록*을 보여줍니다. 여기서 더하기 연산자에 대한 레이어 입력 $\mathbf{x}$을 전달하는 실선을*잔여 연결* (또는*바로 가기 연결*) 이라고합니다.잔차 블록을 사용하면 입력값이 여러 계층의 잔차 연결을 통해 더 빠르게 순방향으로 전파될 수 있습니다. 

![A regular block (left) and a residual block (right).](../img/residual-block.svg)
:label:`fig_residual_block`

레스넷은 VGG의 전체 $3\times 3$ 컨벌루션 계층 설계를 따릅니다.잔차 블록에는 동일한 개수의 출력 채널을 갖는 두 개의 $3\times 3$개의 컨벌루션 계층이 있습니다.각 컨벌루션 계층 뒤에는 배치 정규화 계층과 ReLU 활성화 함수가 옵니다.그런 다음 이 두 컨벌루션 연산을 건너뛰고 최종 ReLU 활성화 함수 바로 앞에 입력을 추가합니다.이러한 종류의 설계에서는 두 컨벌루션 계층의 출력값이 입력값과 동일한 모양이어야 하므로 두 레이어를 함께 더할 수 있습니다.채널 수를 변경하려면 추가 $1\times 1$ 컨벌루션 계층을 추가하여 입력을 더하기 연산을 위해 원하는 모양으로 변환해야합니다.아래 코드를 살펴 보겠습니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class Residual(nn.Block):  #@save
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return npx.relu(Y + X)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):  #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class Residual(tf.keras.Model):  #@save
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, padding='same', kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(
                num_channels, kernel_size=1, strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)
```

이 코드는 두 가지 유형의 네트워크를 생성합니다. 하나는 `use_1x1conv=False`가 될 때마다 ReLU 비선형성을 적용하기 전에 출력에 입력을 추가하는 네트워크이고 다른 하나는 추가하기 전에 $1 \times 1$ 컨벌루션을 사용하여 채널과 해상도를 조정하는 네트워크입니다. :numref:`fig_resnet_block`는 이를 보여줍니다. 

![ResNet block with and without $1 \times 1$ convolution.](../img/resnet-block.svg)
:label:`fig_resnet_block`

이제 [**입력과 출력이 같은 상황**] 을 살펴보겠습니다.

```{.python .input}
blk = Residual(3)
blk.initialize()
X = np.random.uniform(size=(4, 3, 6, 6))
blk(X).shape
```

```{.python .input}
#@tab pytorch
blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = Residual(3)
X = tf.random.uniform((4, 6, 6, 3))
Y = blk(X)
Y.shape
```

[**출력 높이와 너비는 절반으로 줄이면서 출력 채널 수는 늘리기**] 옵션도 있습니다.

```{.python .input}
blk = Residual(6, use_1x1conv=True, strides=2)
blk.initialize()
blk(X).shape
```

```{.python .input}
#@tab pytorch
blk = Residual(3,6, use_1x1conv=True, strides=2)
blk(X).shape
```

```{.python .input}
#@tab tensorflow
blk = Residual(6, use_1x1conv=True, strides=2)
blk(X).shape
```

## [**레스넷 모델**]

ResNet의 처음 두 계층은 앞에서 설명한 GoogLeNet의 계층과 동일합니다. 출력 채널이 64개이고 보폭이 2인 $7\times 7$ 컨벌루션 계층 뒤에는 보폭이 2인 $3\times 3$ 최대 풀링 계층이 옵니다.차이점은 ResNet의 각 컨벌루션 계층 다음에 추가되는 배치 정규화 계층입니다.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
b1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

GoogLeNet은 인셉션 블록으로 구성된 4개의 모듈을 사용합니다.그러나 ResNet은 잔여 블록으로 구성된 4개의 모듈을 사용하며 각 모듈은 동일한 수의 출력 채널을 가진 여러 개의 잔여 블록을 사용합니다.첫 번째 모듈의 채널 수는 입력 채널 수와 같습니다.보폭이 2인 최대 풀링 레이어가 이미 사용되었으므로 높이와 너비를 줄일 필요가 없습니다.후속 모듈 각각에 대한 첫 번째 잔여 블록에서 채널 수는 이전 모듈에 비해 두 배가되고 높이와 너비는 절반으로 줄어 듭니다. 

이제 이 모듈을 구현합니다.첫 번째 모듈에서 특수 처리가 수행되었습니다.

```{.python .input}
def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
#@tab pytorch
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
```

```{.python .input}
#@tab tensorflow
class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False,
                 **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X
```

그런 다음 모든 모듈을 ResNet에 추가합니다.여기서는 각 모듈에 두 개의 잔차 블록이 사용됩니다.

```{.python .input}
net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))
```

```{.python .input}
#@tab pytorch
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
```

```{.python .input}
#@tab tensorflow
b2 = ResnetBlock(64, 2, first_block=True)
b3 = ResnetBlock(128, 2)
b4 = ResnetBlock(256, 2)
b5 = ResnetBlock(512, 2)
```

마지막으로 GoogLeNet과 마찬가지로 전역 평균 풀링 계층을 추가하고 그 다음에 완전히 연결된 계층 출력값을 추가합니다.

```{.python .input}
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
```

```{.python .input}
#@tab tensorflow
# Recall that we define this as a function so we can reuse later and run it
# within `tf.distribute.MirroredStrategy`'s scope to utilize various
# computational resources, e.g. GPUs. Also note that even though we have
# created b1, b2, b3, b4, b5 but we will recreate them inside this function's
# scope instead
def net():
    return tf.keras.Sequential([
        # The following layers are the same as b1 that we created earlier
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # The following layers are the same as b2, b3, b4, and b5 that we
        # created earlier
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=10)])
```

각 모듈에는 4개의 컨벌루션 계층이 있습니다 ($1\times 1$ 컨벌루션 계층 제외).첫 번째 $7\times 7$ 컨벌루션 계층 및 최종 완전 연결 계층과 함께 총 18개의 계층이 있습니다.따라서 이 모델은 일반적으로 ResNet-18로 알려져 있습니다.모듈에서 서로 다른 수의 채널과 잔여 블록을 구성하여 더 깊은 152 계층 ResNet-152와 같은 다양한 ResNet 모델을 만들 수 있습니다.ResNet의 기본 아키텍처는 GoogLeNet의 아키텍처와 유사하지만 ResNet의 구조는 더 간단하고 수정하기 쉽습니다.이러한 모든 요인으로 인해 ResNet이 빠르고 널리 사용되었습니다. :numref:`fig_resnet18`는 전체 ResNet-18을 묘사합니다. 

![The ResNet-18 architecture.](../img/resnet18.svg)
:label:`fig_resnet18`

ResNet을 훈련시키기 전에 [**ResNet의 여러 모듈에서 입력 모양이 어떻게 변하는지 관찰하십시오**].이전의 모든 아키텍처와 마찬가지로 해상도는 감소하는 반면 채널 수는 전역 평균 풀링 계층이 모든 특징을 집계하는 지점까지 증가합니다.

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
X = tf.random.uniform(shape=(1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## [**교육**]

이전과 마찬가지로 패션-MNIST 데이터 세트에서 ResNet을 훈련시킵니다.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 요약

* 중첩된 함수 클래스가 바람직합니다.심층 신경망의 추가 계층을 항등 함수로 학습하는 것은 매우 쉬워야 합니다 (극단적인 경우임에도 불구하고).
* 잔차 매핑은 가중치 계층의 파라미터를 0으로 푸시하는 것과 같이 항등 함수를 더 쉽게 학습할 수 있습니다.
* 잔차 블록을 가짐으로써 효과적인 심층 신경망을 훈련시킬 수 있습니다.입력값은 레이어 간 잔차 연결을 통해 더 빠르게 순방향으로 전파될 수 있습니다.
* ResNet은 컨벌루션 및 순차적 특성 모두에 대한 후속 심층 신경망의 설계에 큰 영향을 미쳤습니다.

## 연습문제

1. :numref:`fig_inception`의 시작 블록과 잔차 블록 간의 주요 차이점은 무엇입니까?Inception 블록에서 일부 경로를 제거한 후 이러한 경로는 서로 어떻게 연관됩니까?
1. 다양한 변형을 구현하려면 ResNet 백서 :cite:`He.Zhang.Ren.ea.2016`의 표 1을 참조하십시오.
1. 더 깊은 네트워크를 위해 ResNet은 모델 복잡성을 줄이기 위해 “병목” 아키텍처를 도입합니다.구현 해보십시오.
1. ResNet의 후속 버전에서 작성자는 “컨볼루션, 배치 정규화 및 활성화” 구조를 “배치 정규화, 활성화 및 컨볼루션” 구조로 변경했습니다.직접 개선하십시오.자세한 내용은 :cite:`He.Zhang.Ren.ea.2016*1`의 그림 1을 참조하십시오.
1. 함수 클래스가 중첩되어 있어도 바인딩되지 않은 함수의 복잡성을 늘릴 수 없는 이유는 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/85)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/86)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/333)
:end_tab:
