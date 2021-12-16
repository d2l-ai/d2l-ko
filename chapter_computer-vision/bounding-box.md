# 물체 감지 및 경계 상자
:label:`sec_bbox`

이전 섹션 (예: :numref:`sec_alexnet`—:numref:`sec_googlenet`) 에서는 이미지 분류를 위한 다양한 모델을 소개했습니다.이미지 분류 작업에서는 이미지에*하나의* 주요 객체가 있다고 가정하고 범주를 인식하는 방법에만 중점을 둡니다.그러나 관심 이미지에*여러* 객체가 있는 경우가 많습니다.카테고리뿐만 아니라 이미지에서 특정 위치도 알고 싶습니다.컴퓨터 비전에서는*물체 감지* (또는*물체 인식*) 와 같은 작업을 말합니다. 

물체 감지는 많은 분야에서 널리 적용되었습니다.예를 들어 자율 주행은 캡처된 비디오 이미지에서 차량, 보행자, 도로 및 장애물의 위치를 감지하여 이동 경로를 계획해야 합니다.또한 로봇은 이 기술을 사용하여 환경 탐색 전반에 걸쳐 관심 물체를 감지하고 현지화할 수 있습니다.또한 보안 시스템은 침입자나 폭탄과 같은 비정상적인 물체를 탐지해야 할 수도 있습니다. 

다음 몇 단원에서는 객체 감지를 위한 몇 가지 딥러닝 방법을 소개하겠습니다.먼저 객체의*위치* (또는*위치*) 에 대한 소개로 시작하겠습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

이 섹션에서 사용할 샘플 이미지를 로드합니다.이미지 왼쪽에 개가 있고 오른쪽에 고양이가 있음을 알 수 있습니다.이 이미지에서 두 가지 주요 개체입니다.

```{.python .input}
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## 바운딩 박스

객체 감지에서는 일반적으로*경계 상자*를 사용하여 객체의 공간 위치를 설명합니다.경계 상자는 직사각형이며 직사각형의 왼쪽 위 모서리의 $x$ 및 $y$ 좌표와 오른쪽 아래 모서리의 해당 좌표에 의해 결정됩니다.일반적으로 사용되는 또 다른 경계 상자 표현은 경계 상자 중심의 $(x, y)$축 좌표와 상자의 너비와 높이입니다. 

[**여기서**] 이 (**두 표현**) 사이를 변환하는 함수를 정의합니다. `box_corner_to_center`는 두 모서리 표현에서 가운데 너비-높이 표현으로 변환하고 `box_center_to_corner`는 그 반대의 경우도 마찬가지입니다.입력 인수 `boxes`은 모양의 2차원 텐서 ($n$, 4) 여야 합니다. 여기서 $n$은 경계 상자의 수입니다.

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

좌표 정보를 기반으로 [**이미지에서 개와 고양이의 경계 상자를 정의**] 합니다.이미지에서 좌표의 원점은 이미지의 왼쪽 위 모서리이고 오른쪽과 아래쪽은 각각 $x$ 및 $y$ 축의 양의 방향입니다.

```{.python .input}
#@tab all
# Here `bbox` is the abbreviation for bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

두 번 변환하여 두 경계 상자 변환 함수의 정확성을 확인할 수 있습니다.

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

[**이미지에 경계 상자를 그리기**] 하여 정확한지 확인하겠습니다.그리기 전에 도우미 함수 `bbox_to_rect`를 정의합니다.`matplotlib` 패키지의 경계 상자 형식으로 경계 상자를 나타냅니다.

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

이미지에 경계 상자를 추가하면 두 객체의 기본 윤곽이 기본적으로 두 상자 안에 있음을 알 수 있습니다.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## 요약

* 물체 감지는 이미지에서 관심있는 모든 물체뿐만 아니라 그 위치도 인식합니다.위치는 일반적으로 직사각형 경계 상자로 표시됩니다.
* 일반적으로 사용되는 두 경계 상자 표현 사이를 변환할 수 있습니다.

## 연습문제

1. 다른 이미지를 찾아 객체가 포함된 경계 상자에 레이블을 지정해 봅니다.레이블 지정 경계 상자와 범주 비교: 일반적으로 시간이 더 오래 걸립니까?
1. `box_corner_to_center`와 `box_center_to_corner`의 입력 인수 `boxes`의 가장 안쪽 차원이 항상 4인 이유는 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1527)
:end_tab:
