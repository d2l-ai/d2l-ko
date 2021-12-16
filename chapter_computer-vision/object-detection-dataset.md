# 물체 감지 데이터세트
:label:`sec_object-detection-dataset`

물체 감지 분야에는 MNIST 및 패션-MNIST와 같은 작은 데이터 세트가 없습니다.물체 감지 모델을 빠르게 시연하기 위해 [**작은 데이터 세트를 수집하고 레이블을 지정했습니다**].먼저 사무실에서 무료 바나나 사진을 찍고 회전과 크기가 다른 1000 개의 바나나 이미지를 생성했습니다.그런 다음 각 바나나 이미지를 배경 이미지의 임의의 위치에 배치했습니다.결국 이미지에서 바나나에 대한 경계 상자에 레이블을 지정했습니다. 

## [**데이터세트 다운로드**]

모든 이미지와 csv 라벨 파일이 포함된 바나나 감지 데이터셋은 인터넷에서 직접 다운로드할 수 있습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## 데이터세트 읽기

아래 `read_data_bananas` 함수에서 [**바나나 검출 데이터세트 읽기**] 를 하겠습니다.데이터셋에는 객체 클래스 레이블에 대한 csv 파일과 왼쪽 위 및 오른쪽 아래 모서리에 있는 실측 경계 상자 좌표가 포함되어 있습니다.

```{.python .input}
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

`read_data_bananas` 함수를 사용하여 이미지와 레이블을 읽으면 다음 `BananasDataset` 클래스를 사용하면 바나나 감지 데이터 세트를 로드하기 위해 [**사용자 지정된 `Dataset` 인스턴스 만들기**] 를 사용할 수 있습니다.

```{.python .input}
#@save
class BananasDataset(gluon.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

마지막으로 `load_data_bananas` 함수를 정의하여 [**훈련 세트와 테스트 세트 모두에 대해 두 개의 데이터 반복기 인스턴스를 반환합니다.**] 테스트 데이터 세트의 경우 임의의 순서로 읽을 필요가 없습니다.

```{.python .input}
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

이 미니 배치에서 [**미니 배치를 읽고 이미지와 레이블의 모양을 인쇄**] 하겠습니다.이미지 미니 배치의 모양 (배치 크기, 채널 수, 높이, 너비) 은 친숙해 보입니다. 이전 이미지 분류 작업과 동일합니다.레이블 미니배치의 모양은 (배치 크기, $m$, 5) 입니다. 여기서 $m$는 모든 이미지가 데이터셋에 포함할 수 있는 가장 큰 경계 상자 수입니다. 

미니 배치에서의 계산이 더 효율적이지만 모든 이미지 예제에 동일한 수의 경계 상자가 포함되어 연결을 통해 미니 배치를 형성해야 합니다.일반적으로 이미지에는 다양한 수의 경계 상자가 있을 수 있습니다. 따라서 경계 상자가 $m$보다 적은 이미지는 $m$에 도달할 때까지 잘못된 경계 상자로 채워집니다.그런 다음 각 경계 상자의 레이블이 길이 5의 배열로 표시됩니다.배열의 첫 번째 요소는 경계 상자에 있는 객체의 클래스입니다. 여기서 -1은 채우기에 잘못된 경계 상자를 나타냅니다.배열의 나머지 4개 요소는 경계 상자의 왼쪽 위 모서리와 오른쪽 아래 모서리의 ($x$, $y$) 좌표값입니다 (범위는 0에서 1 사이임).바나나 데이터셋의 경우 각 이미지에 경계 상자가 하나뿐이므로 $m=1$이 있습니다.

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**데모**]

레이블이 지정된 실측 경계 상자를 사용하여 10개의 이미지를 시연해 보겠습니다.바나나의 회전, 크기 및 위치는 이러한 모든 이미지에서 다양하다는 것을 알 수 있습니다.물론 이것은 단순한 인공 데이터셋일 뿐입니다.실제로 실제 데이터 세트는 일반적으로 훨씬 더 복잡합니다.

```{.python .input}
imgs = (batch[0][0:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## 요약

* 수집한 바나나 감지 데이터세트는 물체 감지 모델을 시연하는 데 사용할 수 있습니다.
* 객체 감지를 위한 데이터 로딩은 이미지 분류의 데이터 로드와 유사합니다.그러나 객체 감지에서 레이블에는 이미지 분류에서 누락된 지상 실측 경계 상자의 정보도 포함됩니다.

## 연습문제

1. 바나나 탐지 데이터셋에서 실측 경계 상자를 사용하여 다른 이미지를 시연합니다.경계 상자 및 오브젝트와 관련하여 어떻게 다릅니까?
1. 객체 감지에 임의 자르기와 같은 데이터 증대를 적용하려고 한다고 가정해 보겠습니다.이미지 분류와 어떻게 다를수 있을까요?힌트: 잘린 이미지에 개체의 작은 부분만 포함되어 있으면 어떻게 될까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/372)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1608)
:end_tab:
