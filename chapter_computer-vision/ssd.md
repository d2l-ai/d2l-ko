# 단발 멀티박스 감지
:label:`sec_ssd`

:numref:`sec_bbox`—:numref:`sec_object-detection-dataset`에서는 경계 상자, 앵커 상자, 다중 축척 객체 감지 및 객체 감지를 위한 데이터 세트를 도입했습니다.이제 이러한 배경 지식을 사용하여 물체 감지 모델 (SSD) :cite:`Liu.Anguelov.Erhan.ea.2016`와 같은 단일 샷 멀티박스 감지 모델을 설계할 준비가 되었습니다.이 모델은 간단하고 빠르며 널리 사용됩니다.이것은 방대한 양의 객체 감지 모델 중 하나일 뿐이지 만, 이 섹션의 일부 설계 원칙 및 구현 세부 사항은 다른 모델에도 적용됩니다. 

## 모델

:numref:`fig_ssd`는 단발 멀티박스 감지의 설계에 대한 개요를 제공합니다.이 모델은 주로 기본 네트워크와 여러 다중 축척 특징 맵 블록으로 구성됩니다.기본 네트워크는 입력 영상에서 특징을 추출하기 위한 것이므로 심층 CNN을 사용할 수 있습니다.예를 들어, 원래의 단일 샷 멀티 박스 감지 용지는 분류 계층 :cite:`Liu.Anguelov.Erhan.ea.2016` 이전에 잘린 VGG 네트워크를 채택하고 ResNet도 일반적으로 사용되었습니다.설계를 통해 기본 네트워크가 더 큰 특징 맵을 출력하여 더 작은 물체를 감지하기 위해 더 많은 앵커 박스를 생성 할 수 있습니다.후속하여, 각각의 멀티스케일 특징 맵 블록은 이전 블록으로부터 특징 맵의 높이 및 폭을 감소시키고 (예를 들어, 절반으로), 특징 맵의 각 유닛이 입력 이미지 상의 자신의 수용 필드를 증가시키도록 한다. 

:numref:`sec_multiscale-object-detection`에서 심층 신경망에 의한 이미지의 층별 표현을 통해 다중 스케일 객체 감지 설계를 회상합니다.:numref:`fig_ssd`의 맨 위에 가까운 다중 축척 특징 맵은 더 작지만 수용 필드가 더 크기 때문에 더 작지만 더 큰 물체를 감지하는 데 적합합니다. 

간단히 말해서, 기본 네트워크와 여러 다중 축척 특징 맵 블록을 통해 단일 샷 다중 상자 감지는 크기가 다른 다양한 앵커 상자를 생성하고 이러한 앵커 상자 (따라서 경계 상자) 의 클래스와 오프셋을 예측하여 다양한 크기의 객체를 감지합니다.다중 스케일 객체 감지 모델. 

![As a multiscale object detection model, single-shot multibox detection mainly consists of a base network followed by several multiscale feature map blocks.](../img/ssd.svg)
:label:`fig_ssd`

다음에서는 :numref:`fig_ssd`에서 서로 다른 블록의 구현 세부 사항을 설명합니다.먼저 클래스와 경계 상자 예측을 구현하는 방법에 대해 설명합니다. 

### [**클래스 예측 계층**]

객체 클래스의 수를 $q$로 지정합니다.그런 다음 앵커 박스에는 $q+1$ 클래스가 있으며 여기서 클래스 0은 배경입니다.일부 축척에서 특징 맵의 높이와 너비가 각각 $h$과 $w$라고 가정합니다.이러한 피처 맵의 각 공간 위치를 중심으로 $a$ 앵커 박스가 생성되면 총 $hwa$ 앵커 박스를 분류해야 합니다.이로 인해 매개 변수화 비용이 많이 들기 때문에 완전히 연결된 계층으로 분류할 수 없는 경우가 많습니다.:numref:`sec_nin`에서 컨벌루션 계층의 채널을 사용하여 클래스를 예측한 방법을 생각해 보십시오.단발 멀티박스 감지는 동일한 기술을 사용하여 모델 복잡성을 줄입니다. 

구체적으로, 클래스 예측 계층은 피처 맵의 너비나 높이를 변경하지 않고 컨벌루션 계층을 사용합니다.이러한 방식으로 피처 맵의 동일한 공간 차원 (너비 및 높이) 에서 출력과 입력 간에 일대일 대응이 있을 수 있습니다.보다 구체적으로, 임의의 공간 위치 ($x$, $y$) 에서 출력 특징 맵의 채널은 입력 특징 맵의 ($x$, $y$) 을 중심으로 하는 모든 앵커 박스에 대한 클래스 예측을 나타낸다.유효한 예측을 생성하려면 $a(q+1)$개의 출력 채널이 있어야 합니다. 여기서 동일한 공간 위치에 대해 인덱스 $i(q+1) + j$을 갖는 출력 채널은 앵커 박스 $i$ ($0 \leq i < a$) 에 대한 클래스 $j$ ($0 \leq j \leq q$) 의 예측을 나타냅니다. 

아래에서는 인수 `num_anchors` 및 `num_classes`를 통해 각각 $a$과 $q$을 지정하는 이러한 클래스 예측 계층을 정의합니다.이 계층은 패딩이 1인 $3\times3$ 컨벌루션 계층을 사용합니다.이 컨벌루션 계층의 입력값과 출력값의 너비와 높이는 변경되지 않습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

### (**바운딩 박스 예측 레이어**)

경계 상자 예측 계층의 디자인은 클래스 예측 계층의 설계와 유사합니다.유일한 차이점은 각 앵커 박스의 출력 수에 있습니다. 여기서는 $q+1$ 클래스가 아닌 네 개의 오프셋을 예측해야 합니다.

```{.python .input}
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
```

```{.python .input}
#@tab pytorch
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

### [**여러 척도에 대한 예측 연결**]

앞서 언급했듯이 단발 멀티박스 탐지는 다중 축척 특징 맵을 사용하여 앵커 박스를 생성하고 해당 클래스와 오프셋을 예측합니다.축척에 따라 피처 맵의 모양이나 동일한 단위를 중심으로 한 앵커 상자의 수가 다를 수 있습니다.따라서 서로 다른 스케일에서 예측 출력의 모양이 다를 수 있습니다. 

다음 예에서는 `Y2`의 높이와 너비가 `Y1`의 높이와 너비의 절반인 동일한 미니배치에 대해 `Y1`와 `Y2`의 두 가지 축척으로 피쳐 맵을 구성합니다.클래스 예측을 예로 들어 보겠습니다.`Y1` 및 `Y2`의 모든 단위에 대해 각각 5개 및 3개의 앵커 상자가 생성된다고 가정합니다.객체 클래스 수가 10이라고 가정합니다.특징 맵 `Y1` 및 `Y2`의 경우 클래스 예측 출력의 채널 수는 각각 $5\times(10+1)=55$ 및 $3\times(10+1)=33$이며, 여기서 출력 형태는 (배치 크기, 채널 수, 높이, 너비) 입니다.

```{.python .input}
def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(np.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(np.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
Y1.shape, Y2.shape
```

```{.python .input}
#@tab pytorch
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
```

보시다시피 배치 크기 차원을 제외하고 나머지 세 차원은 모두 크기가 다릅니다.보다 효율적인 계산을 위해이 두 예측 출력을 연결하기 위해 이러한 텐서를보다 일관된 형식으로 변환합니다. 

채널 차원에는 중심이 같은 앵커 상자에 대한 예측이 포함됩니다.먼저 이 차원을 가장 안쪽까지 이동합니다.배치 크기는 다른 척도에 대해 동일하게 유지되므로 예측 출력을 모양 (배치 크기, 높이 $\times$ 너비 $\times$ 채널 수) 이있는 2 차원 텐서로 변환 할 수 있습니다.그런 다음 차원 1을 따라 다른 스케일로 이러한 출력을 연결할 수 있습니다.

```{.python .input}
def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)
```

```{.python .input}
#@tab pytorch
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
```

이러한 방식으로 `Y1` 및 `Y2`는 채널, 높이 및 너비가 서로 다른 크기를 갖지만 동일한 미니 배치에 대해 두 개의 서로 다른 스케일로 이러한 두 예측 출력을 연결할 수 있습니다.

```{.python .input}
#@tab all
concat_preds([Y1, Y2]).shape
```

### [**다운샘플링 블록**]

여러 스케일에서 객체를 감지하기 위해 입력 특징 맵의 높이와 너비를 절반으로 줄이는 다음 다운 샘플링 블록 `down_sample_blk`를 정의합니다.사실, 이 블록은 :numref:`subsec_vgg-blocks`에서 VGG 블록의 설계를 적용합니다.좀 더 구체적으로 말하자면, 각 다운샘플링 블록은 패딩이 1인 두 개의 $3\times3$ 컨벌루션 계층과 스트라이드가 2인 $2\times2$ 최대 풀링 계층으로 구성됩니다.아시다시피 패딩이 1인 $3\times3$개의 컨벌루션 계층은 특징 맵의 모양을 변경하지 않습니다.그러나 이후 $2\times2$ 최대 풀링은 입력 특징 맵의 높이와 너비를 절반으로 줄입니다.이 다운샘플링 블록의 입력 및 출력 특징 맵 모두에 대해 $1\times 2+(3-1)+(3-1)=6$이므로 출력의 각 장치에는 입력에 $6\times6$ 수용 필드가 있습니다.따라서 다운샘플링 블록은 출력 특징 맵에서 각 유닛의 수용 필드를 확대합니다.

```{.python .input}
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
```

```{.python .input}
#@tab pytorch
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
```

다음 예제에서 구성된 다운샘플링 블록은 입력 채널 수를 변경하고 입력 피처 맵의 높이와 너비는 절반으로 줄입니다.

```{.python .input}
forward(np.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape
```

```{.python .input}
#@tab pytorch
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```

### [**기본 네트워크 블록**]

기본 네트워크 블록은 입력 영상에서 특징을 추출하는 데 사용됩니다.단순화를 위해 각 블록에서 채널 수를 두 배로 늘리는 세 개의 다운 샘플링 블록으로 구성된 작은 기본 네트워크를 구성합니다.$256\times256$ 입력 이미지가 주어지면 이 기본 네트워크 블록은 $32 \times 32$ 특징 맵 ($256/2^3=32$) 을 출력합니다.

```{.python .input}
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

forward(np.zeros((2, 3, 256, 256)), base_net()).shape
```

```{.python .input}
#@tab pytorch
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
```

### 완벽한 모델

[**완전한 단일 샷 멀티박스 탐지 모델은 5개의 블록으로 구성됩니다.**] 각 블록에서 생성된 특징 맵은 (i) 앵커 박스를 생성하고 (ii) 이러한 앵커 박스의 클래스와 오프셋을 예측하는 데 사용됩니다.이 5개 블록 중에서 첫 번째 블록은 기본 네트워크 블록이고, 두 번째 블록에서 네 번째 블록은 다운샘플링 블록이며, 마지막 블록은 전역 최대 풀링을 사용하여 높이와 너비를 모두 1로 줄입니다.기술적으로 두 번째에서 다섯 번째 블록은 모두 :numref:`fig_ssd`의 다중 스케일 특징 맵 블록입니다.

```{.python .input}
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk
```

```{.python .input}
#@tab pytorch
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk
```

이제 각 블록에 대해 [**순방향 전파를 정의**] 합니다.이미지 분류 작업과는 달리, 여기서 출력에는 (i) CNN 특징 맵 `Y`, (ii) 현재 스케일에서 `Y`를 사용하여 생성된 앵커 박스, (iii) 이러한 앵커 박스에 대해 예측된 클래스 및 오프셋 (`Y` 기준) 이 포함됩니다.

```{.python .input}
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

```{.python .input}
#@tab pytorch
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

:numref:`fig_ssd`에서 맨 위에 더 가까운 다중 축척 특징 맵 블록은 더 큰 객체를 탐지하기 위한 것이므로 더 큰 앵커 상자를 생성해야 합니다.위의 순방향 전파에서 각 다중 스케일 특징 맵 블록에서 호출된 `multibox_prior` 함수의 `sizes` 인수를 통해 두 개의 척도 값 목록을 전달합니다 (:numref:`sec_anchor`에 설명).다음에서는 0.2와 1.05 사이의 구간을 5개의 섹션으로 균등하게 분할하여 5개 블록에서 더 작은 축척 값 (0.2, 0.37, 0.54, 0.71 및 0.88) 을 결정합니다.그런 다음 더 큰 척도 값은 $\sqrt{0.2 \times 0.37} = 0.272$, $\sqrt{0.37 \times 0.54} = 0.447$ 등으로 지정됩니다. 

[~~각 블록의 하이퍼파라미터~~]

```{.python .input}
#@tab all
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

이제 다음과 같이 [**전체 모델을 정의**] `TinySSD` 할 수 있습니다.

```{.python .input}
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = np.concatenate(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

```{.python .input}
#@tab pytorch
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

$256 \times 256$ 이미지 `X`의 미니배치에서 [**모델 인스턴스를 생성하여 순방향 전파를 수행하는 데 사용**] 합니다. 

이 섹션의 앞부분에서 보듯이 첫 번째 블록은 $32 \times 32$ 특징 맵을 출력합니다.두 번째에서 네 번째 다운샘플링 블록은 높이와 너비를 절반으로 줄이고 다섯 번째 블록은 전역 풀링을 사용합니다.피처 맵의 공간 차원을 따라 각 단위에 대해 4개의 앵커 박스가 생성되므로 5개 척도 모두에서 각 이미지에 대해 총 $(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$개의 앵커 박스가 생성됩니다.

```{.python .input}
net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

```{.python .input}
#@tab pytorch
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

## 트레이닝

이제 물체 감지를 위해 단발 멀티 박스 감지 모델을 훈련시키는 방법에 대해 설명하겠습니다. 

### 데이터세트 읽기 및 모델 초기화

먼저 :numref:`sec_object-detection-dataset`에 설명된 [**바나나 검출 데이터세트 읽기**] 를 하겠습니다.

```{.python .input}
#@tab all
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
```

바나나 검출 데이터셋에는 단 하나의 클래스만 있습니다.모델을 정의한 후에는 (**매개 변수를 초기화하고 최적화 알고리즘을 정의**) 해야합니다.

```{.python .input}
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})
```

```{.python .input}
#@tab pytorch
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
```

### [**손실 및 평가 함수 정의**]

물체 감지에는 두 가지 유형의 손실이 있습니다.첫 번째 손실은 앵커 박스의 클래스와 관련이 있습니다. 계산은 이미지 분류에 사용한 교차 엔트로피 손실 함수를 간단히 재사용 할 수 있습니다.두 번째 손실은 양수 (배경이 아닌) 앵커 박스의 오프셋과 관련이 있습니다. 이는 회귀 문제입니다.그러나 이 회귀 문제의 경우 :numref:`subsec_normal_distribution_and_squared_loss`에 설명된 손실 제곱을 사용하지 않습니다.대신 예측과 실측 차이의 절대값인 $L_1$ 노름 손실을 사용합니다.마스크 변수 `bbox_masks`는 손실 계산에서 음수 앵커 상자와 잘못된 (패딩) 앵커 상자를 필터링합니다.마지막으로 앵커 박스 클래스 손실과 앵커 박스 오프셋 손실을 합산하여 모델의 손실 함수를 얻습니다.

```{.python .input}
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
bbox_loss = gluon.loss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox
```

```{.python .input}
#@tab pytorch
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
```

정확도를 사용하여 분류 결과를 평가할 수 있습니다.오프셋에 $L_1$ 노름 손실이 사용되었기 때문에*평균 절대 오차*를 사용하여 예측된 경계 상자를 평가합니다.이러한 예측 결과는 생성된 앵커 박스와 그에 대한 예측된 오프셋으로부터 얻어집니다.

```{.python .input}
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

```{.python .input}
#@tab pytorch
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

### [**모델 교육**]

모델을 훈련시킬 때 다중 스케일 앵커 박스 (`anchors`) 를 생성하고 순방향 전파에서 해당 클래스 (`cls_preds`) 와 오프셋 (`bbox_preds`) 을 예측해야 합니다.그런 다음 레이블 정보 `Y`을 기반으로 생성된 앵커 박스의 클래스 (`cls_labels`) 및 오프셋 (`bbox_labels`) 에 레이블을 지정합니다.마지막으로 클래스와 오프셋의 예측 및 레이블이 지정된 값을 사용하여 손실 함수를 계산합니다.간결한 구현을 위해 테스트 데이터셋의 평가는 여기에서 생략됩니다.

```{.python .input}
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    for features, target in train_iter:
        timer.start()
        X = features.as_in_ctx(device)
        Y = target.as_in_ctx(device)
        with autograd.record():
            # Generate multiscale anchor boxes and predict their classes and
            # offsets
            anchors, cls_preds, bbox_preds = net(X)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors,
                                                                      Y)
            # Calculate the loss function using the predicted and labeled
            # values of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter._dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

```{.python .input}
#@tab pytorch
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # Generate multiscale anchor boxes and predict their classes and
        # offsets
        anchors, cls_preds, bbox_preds = net(X)
        # Label the classes and offsets of these anchor boxes
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # Calculate the loss function using the predicted and labeled values
        # of the classes and offsets
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

## [**예측**]

예측 중에 목표는 이미지에서 관심 있는 모든 객체를 감지하는 것입니다.아래에서는 테스트 이미지를 읽고 크기를 조정하여 컨볼 루션 계층에 필요한 4 차원 텐서로 변환합니다.

```{.python .input}
img = image.imread('../img/banana.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)
```

```{.python .input}
#@tab pytorch
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
```

아래의 `multibox_detection` 함수를 사용하면 앵커 상자와 예측된 오프셋에서 예측된 경계 상자를 가져옵니다.그런 다음 최대가 아닌 억제를 사용하여 유사한 예측 경계 상자를 제거합니다.

```{.python .input}
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

```{.python .input}
#@tab pytorch
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

마지막으로 [**모든 예측 경계 상자를 0.9 이상의 신뢰도로 표시**] 출력으로 표시합니다.

```{.python .input}
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output, threshold=0.9)
```

```{.python .input}
#@tab pytorch
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```

## 요약

* 싱글 샷 멀티박스 감지는 멀티스케일 물체 감지 모델입니다.단일 샷 멀티박스 탐지는 기본 네트워크와 여러 다중 축척 특징 맵 블록을 통해 크기가 다른 다양한 앵커 상자를 생성하고 이러한 앵커 상자 (경계 상자) 의 클래스와 오프셋을 예측하여 다양한 크기의 객체를 탐지합니다.
* 단발 멀티박스 탐지 모델을 훈련시킬 때 손실 함수는 앵커 박스 클래스와 오프셋의 예측 및 레이블이 지정된 값을 기반으로 계산됩니다.

## 연습문제

1. 손실 기능을 개선하여 단발 멀티박스 감지를 개선할 수 있습니까?예를 들어, 예측된 오프셋에 대해 $L_1$ 노름 손실을 부드러운 $L_1$ 노름 손실로 바꿉니다.이 손실 함수는 평활도를 위해 0에 가까운 제곱 함수를 사용하며, 이 함수는 하이퍼파라미터 $\sigma$에 의해 제어됩니다.

$$
f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \text{if }|x| < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \text{otherwise}
    \end{cases}
$$

$\sigma$가 매우 큰 경우 이 손실은 $L_1$ 표준 손실과 유사합니다.값이 작으면 손실 함수가 더 부드러워집니다.

```{.python .input}
sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = np.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = npx.smooth_l1(x, scalar=s)
    d2l.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

게다가 실험에서 클래스 예측을 위해 교차 엔트로피 손실을 사용했습니다. $p_j$로 지상 진실 클래스 $j$에 대한 예측 확률을 나타내면 교차 엔트로피 손실은 $-\log p_j$입니다.초점 손실 :cite:`Lin.Goyal.Girshick.ea.2017`를 사용할 수도 있습니다. 하이퍼 파라미터 $\gamma > 0$ 및 $\alpha > 0$이 주어지면이 손실은 다음과 같이 정의됩니다. 

$$ - \alpha (1-p_j)^{\gamma} \log p_j.$$

보시다시피 $\gamma$를 늘리면 잘 분류된 예 (예: $p_j > 0.5$) 의 상대적 손실을 효과적으로 줄일 수 있으므로 교육은 잘못 분류된 어려운 예에 더 집중할 수 있습니다.

```{.python .input}
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * np.log(x)

x = np.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,
                     label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

2. 공간 제한으로 인해 이 섹션에서는 싱글샷 멀티박스 탐지 모델의 일부 구현 세부 사항을 생략했습니다.다음과 같은 측면에서 모델을 더 개선할 수 있습니까?
    1. 객체가 이미지에 비해 훨씬 작으면 모델이 입력 이미지의 크기를 더 크게 조정할 수 있습니다.
    1. 일반적으로 수많은 네거티브 앵커 박스가 있습니다.클래스 분포의 균형을 높이기 위해 네거티브 앵커 박스를 다운샘플링할 수 있습니다.
    1. 손실 함수에서 클래스 손실과 오프셋 손실에 서로 다른 가중치 하이퍼파라미터를 할당합니다.
    1. 단일 샷 멀티박스 감지 용지 :cite:`Liu.Anguelov.Erhan.ea.2016`와 같은 다른 방법을 사용하여 물체 감지 모델을 평가합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/373)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1604)
:end_tab:
