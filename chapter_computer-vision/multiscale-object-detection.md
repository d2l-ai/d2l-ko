# 멀티스케일 객체 감지
:label:`sec_multiscale-object-detection`

:numref:`sec_anchor`에서는 입력 이미지의 각 픽셀을 중심으로 여러 앵커 상자를 생성했습니다.기본적으로 이러한 앵커 박스는 이미지의 여러 영역의 샘플을 나타냅니다.그러나*모든* 픽셀에 대해 생성되는 경우 앵커 박스가 너무 많아 계산할 수 없습니다.$561 \times 728$ 입력 이미지를 생각해 보십시오.각 픽셀을 중심으로 다양한 모양을 가진 다섯 개의 앵커 박스가 생성되면 이미지에서 2백만 개가 넘는 앵커 박스 ($561 \times 728 \times 5$) 에 레이블을 지정하고 예측해야 합니다.

## 멀티스케일 앵커 박스
:label:`subsec_multiscale-anchor-boxes`

이미지에서 앵커 상자를 줄이는 것이 어렵지 않다는 것을 알 수 있습니다.예를 들어 입력 이미지에서 픽셀의 작은 부분을 균일하게 샘플링하여 중앙에 앵커 박스를 생성 할 수 있습니다.또한 다양한 축척에서 크기가 다른 여러 앵커 상자를 생성 할 수 있습니다.직관적으로 작은 물체는 큰 물체보다 이미지에 나타날 가능성이 더 큽니다.예를 들어 $1 \times 1$, $1 \times 2$ 및 $2 \times 2$ 객체는 $2 \times 2$ 이미지에 각각 4, 2 및 1개의 가능한 방식으로 나타날 수 있습니다.따라서 작은 앵커 박스를 사용하여 더 작은 물체를 감지하면 더 많은 영역을 샘플링할 수 있고, 큰 객체의 경우 더 적은 영역을 샘플링할 수 있습니다.

여러 스케일로 앵커 박스를 생성하는 방법을 보여주기 위해 이미지를 읽어 보겠습니다.높이와 너비는 각각 561픽셀과 728픽셀입니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

:numref:`sec_conv_layer`에서는 컨벌루션 계층의 2차원 배열 출력값을 특징 맵이라고 부릅니다.특징 맵 모양을 정의하여 모든 이미지에서 균일하게 샘플링된 앵커 박스의 중심을 결정할 수 있습니다.

`display_anchors` 함수는 아래에 정의되어 있습니다.[**각 단위 (픽셀) 를 앵커 상자 중심으로 사용하여 형상 맵 (`fmap`) 에 앵커 박스 (`anchors`) 를 생성합니다.**] 앵커 박스 (`anchors`) 의 $(x, y)$축 좌표 값을 형상 맵 (`fmap`) 의 너비와 높이로 나눠졌기 때문에 이 값은 0과 1 사이입니다.피처 맵에서 앵커 박스의 상대적 위치를 나타냅니다.

앵커 박스 (`anchors`) 의 중심이 특징 맵 (`fmap`) 의 모든 유닛에 분산되어 있기 때문에 이러한 중심은 상대 공간 위치의 관점에서 임의의 입력 이미지에*균일하게* 분포되어야 합니다.보다 구체적으로, 특징 맵 `fmap_w` 및 `fmap_h`의 너비와 높이를 각각 고려할 때 다음 함수는 모든 입력 이미지에서 `fmap_h` 행과 `fmap_w` 열의 픽셀을 샘플링합니다*.균일하게 샘플링된 픽셀을 중심으로 `s` (목록 `s`의 길이가 1이라고 가정) 및 다른 종횡비 (`ratios`) 의 앵커 상자가 생성됩니다.

```{.python .input}
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

먼저 [**작은 물체의 감지 고려**] 를 하겠습니다.표시 할 때 쉽게 구분할 수 있도록 여기에 중심이 다른 앵커 상자가 겹치지 않습니다. 앵커 상자 배율은 0.15로 설정되고 피처 맵의 높이와 너비는 4로 설정됩니다.이미지에서 4 행과 4 열의 앵커 박스 중심이 균일하게 분포되어 있음을 알 수 있습니다.

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

[**피처 맵의 높이와 너비를 절반으로 줄이고 더 큰 앵커 박스를 사용하여 더 큰 물체를 감지합니다**] 로 넘어갑니다.축척이 0.4로 설정되면 일부 앵커 상자가 서로 겹칩니다.

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

마지막으로 [**피처 맵의 높이와 너비를 절반으로 줄이고 앵커 박스 스케일을 0.8로 늘립니다**].이제 앵커 박스의 중심이 이미지의 중심입니다.

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## 멀티스케일 감지

다중 스케일 앵커 박스를 생성했기 때문에 다양한 스케일에서 다양한 크기의 물체를 감지하는 데 사용할 것입니다.다음에서는 :numref:`sec_ssd`에서 구현할 CNN 기반 다중 스케일 객체 감지 방법을 소개합니다.

어떤 규모에서 $c$ 형상 $h \times w$의 특징 맵이 있다고 가정해 보겠습니다.:numref:`subsec_multiscale-anchor-boxes`의 방법을 사용하여 $hw$ 세트의 앵커 상자를 생성합니다. 여기서 각 세트에는 중심이 동일한 $a$ 앵커 상자가 있습니다.예를 들어, :numref:`subsec_multiscale-anchor-boxes`의 실험의 첫 번째 척도에서 10개 (채널 수) $4 \times 4$개의 피처 맵이 주어지면 16개의 앵커 박스 세트를 생성했습니다. 각 세트에는 중심이 같은 3개의 앵커 박스가 포함되어 있습니다.다음으로 각 앵커 박스에는 지상 실측 경계 상자를 기반으로 클래스 및 오프셋이 지정됩니다.현재 척도에서 객체 감지 모델은 입력 이미지에서 $hw$ 세트의 앵커 박스 클래스와 오프셋을 예측해야 합니다. 이 경우 세트마다 중심이 다릅니다.

여기서 $c$ 특징 맵은 입력 이미지를 기반으로 CNN 순방향 전파에 의해 얻어진 중간 출력이라고 가정합니다.각 특징 맵에는 $hw$개의 서로 다른 공간 위치가 있으므로 동일한 공간 위치는 $c$ 단위를 갖는 것으로 생각할 수 있습니다.:numref:`sec_conv_layer`의 수용장 정의에 따르면, 특징 맵의 동일한 공간 위치에있는 이러한 $c$ 단위는 입력 이미지에서 동일한 수용 필드를 갖습니다. 동일한 수용 필드에서 입력 이미지 정보를 나타냅니다.따라서 동일한 공간 위치에 있는 피쳐 맵의 $c$ 단위를 이 공간 위치를 사용하여 생성된 $a$ 앵커 박스의 클래스 및 오프셋으로 변환할 수 있습니다.본질적으로 특정 수용 필드에서 입력 이미지의 정보를 사용하여 입력 이미지의 수용 필드에 가까운 앵커 박스의 클래스와 오프셋을 예측합니다.

서로 다른 레이어의 특징 맵에 입력 영상에 다양한 크기의 수용 필드가 있는 경우 이 맵을 사용하여 다양한 크기의 객체를 탐지할 수 있습니다.예를 들어, 출력 계층에 더 가까운 특징 맵의 단위가 더 넓은 수용 필드를 가지므로 입력 이미지에서 더 큰 객체를 감지할 수 있는 신경망을 설계할 수 있습니다.

간단히 말해서, 다중 스케일 객체 감지를 위해 심층 신경망에 의해 여러 수준에서 이미지의 계층 적 표현을 활용할 수 있습니다.:numref:`sec_ssd`의 구체적인 예를 통해 이것이 어떻게 작동하는지 보여 드리겠습니다.

## 요약

* 여러 스케일에서 크기가 다른 앵커 박스를 생성하여 크기가 다른 물체를 감지 할 수 있습니다.
* 특징 맵의 모양을 정의하여 모든 이미지에서 균일하게 샘플링된 앵커 박스의 중심을 결정할 수 있습니다.
* 특정 수용 필드에서 입력 이미지의 정보를 사용하여 입력 이미지의 수용 필드에 가까운 앵커 박스의 클래스와 오프셋을 예측합니다.
* 딥 러닝을 통해 다중 스케일 객체 감지를 위해 여러 수준에서 이미지의 층별 표현을 활용할 수 있습니다.

## 연습문제

1. :numref:`sec_alexnet`의 논의에 따르면 심층 신경망은 이미지의 추상화 수준이 높아짐에 따라 계층적 특징을 학습합니다.다중 축척 객체 감지에서 다양한 축척의 특징 맵이 서로 다른 수준의 추상화에 대응합니까?왜, 왜 안되니?
1. :numref:`subsec_multiscale-anchor-boxes`의 실험의 첫 번째 척도 (`fmap_w=4, fmap_h=4`) 에서 겹칠 수 있는 균일하게 분산된 앵커 박스를 생성합니다.
1. 쉐이프가 $1 \times c \times h \times w$인 피쳐 맵 변수가 주어진 경우 여기서 $c$, $h$ 및 $w$는 각각 피쳐 맵의 채널 수, 높이 및 너비입니다.어떻게 이 변수를 앵커 박스의 클래스와 오프셋으로 변환할 수 있을까요?출력의 형태는 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1607)
:end_tab:
