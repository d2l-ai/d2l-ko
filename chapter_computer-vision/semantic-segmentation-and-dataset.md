# 시맨틱 분할 및 데이터세트
:label:`sec_semantic_segmentation`

:numref:`sec_bbox`—:numref:`sec_rcnn`에서 객체 감지 작업을 논의할 때 사각형 경계 상자를 사용하여 이미지의 객체에 레이블을 지정하고 예측합니다.이 섹션에서는 이미지를 다른 시맨틱 클래스에 속하는 영역으로 나누는 방법에 초점을 맞추는*시맨틱 분할*의 문제에 대해 설명합니다.의미 론적 분할은 객체 감지와 달리 픽셀 수준의 이미지에 무엇이 있는지 인식하고 이해합니다. 의미 영역의 레이블 지정 및 예측은 픽셀 수준입니다. :numref:`fig_segmentation`는 의미 론적 분할에서 이미지의 개, 고양이 및 배경의 레이블을 보여줍니다.객체 감지와 비교할 때 의미 론적 분할로 레이블이 지정된 픽셀 수준 테두리는 분명히 더 세분화되어 있습니다. 

![Labels of the dog, cat, and background of the image in semantic segmentation.](../img/segmentation.svg)
:label:`fig_segmentation`

## 이미지 세분화 및 인스턴스 세분화

또한 컴퓨터 비전 분야에는 시맨틱 분할과 유사한 두 가지 중요한 작업, 즉 이미지 분할과 인스턴스 분할이 있습니다.다음과 같이 의미 론적 분할과 간단히 구분할 것입니다. 

* *이미지 분할*은 이미지를 여러 구성 영역으로 나눕니다.이러한 유형의 문제에 대한 방법은 일반적으로 이미지의 픽셀 간의 상관 관계를 사용합니다.훈련 중에는 이미지 픽셀에 대한 레이블 정보가 필요하지 않으며 세그먼트화된 영역이 예측 중에 얻고자 하는 의미를 가질 것이라고 보장할 수 없습니다.:numref:`fig_segmentation`의 이미지를 입력으로 가져 가면 이미지 분할은 개를 두 영역으로 나눌 수 있습니다. 하나는 주로 검은 색 입과 눈을 덮고 다른 하나는 주로 노란색 인 신체의 나머지 부분을 덮습니다.
* *인스턴스 세분화*는*동시 탐지 및 세분화*라고도 합니다.이미지에서 각 객체 인스턴스의 픽셀 수준 영역을 인식하는 방법을 연구합니다.의미론적 분할과는 달리, 인스턴스 분할은 의미론뿐만 아니라 다른 객체 인스턴스도 구별해야 합니다.예를 들어 이미지에 개가 두 개 있는 경우 인스턴스 세그먼테이션은 두 개 중 픽셀이 속한 개를 구분해야 합니다.

## 파스칼 VOC2012 시맨틱 분할 데이터셋

[**가장 중요한 의미론적 분할 데이터셋은 [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).**] 다음에서 이 데이터세트를 살펴보겠습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
```

데이터셋의 tar 파일은 약 2GB이므로 파일을 다운로드하는 데 시간이 걸릴 수 있습니다.추출된 데이터셋은 `../data/VOCdevkit/VOC2012`에 있습니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

`../data/VOCdevkit/VOC2012` 경로를 입력하면 데이터 세트의 다양한 구성 요소를 볼 수 있습니다.`ImageSets/Segmentation` 경로에는 훈련 샘플과 테스트 샘플을 지정하는 텍스트 파일이 포함되어 있으며 `JPEGImages` 및 `SegmentationClass` 경로는 각 예제의 입력 이미지와 레이블을 각각 저장합니다.여기서 레이블은 레이블이 지정된 입력 이미지와 크기가 같은 이미지 형식입니다.또한 레이블 이미지에서 색상이 같은 픽셀은 동일한 시맨틱 클래스에 속합니다.다음은 `read_voc_images` 함수가 [**모든 입력 이미지와 레이블을 메모리로 읽기**] 하도록 정의합니다.

```{.python .input}
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(image.imread(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(image.imread(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

```{.python .input}
#@tab pytorch
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

[**처음 5개의 입력 이미지와 해당 레이블을 그리기**] 합니다.레이블 이미지에서 흰색과 검정색은 각각 테두리와 배경을 나타내고 다른 색상은 서로 다른 클래스에 해당합니다.

```{.python .input}
n = 5
imgs = train_features[0:n] + train_labels[0:n]
d2l.show_images(imgs, 2, n);
```

```{.python .input}
#@tab pytorch
n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n);
```

다음으로 이 데이터셋의 모든 레이블에 대해 [**RGB 색상 값과 클래스 이름**] 을 열거합니다.

```{.python .input}
#@tab all
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
```

위에서 정의한 두 개의 상수를 사용하면 편리하게 [**레이블의 각 픽셀에 대한 클래스 인덱스를 찾기**] 할 수 있습니다.`voc_colormap2label` 함수를 정의하여 위의 RGB 색상 값에서 클래스 인덱스로의 매핑을 빌드하고 `voc_label_indices` 함수를 정의하여이 파스칼 VOC2012 데이터 세트의 모든 RGB 값을 클래스 인덱스에 매핑합니다.

```{.python .input}
#@save
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

```{.python .input}
#@tab pytorch
#@save
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

[**예**] 의 경우, 첫 번째 예제 이미지에서 비행기의 앞부분에 대한 클래스 색인은 1이고 배경 색인은 0입니다.

```{.python .input}
#@tab all
y = voc_label_indices(train_labels[0], voc_colormap2label())
y[105:115, 130:140], VOC_CLASSES[1]
```

### 데이터 전처리

:numref:`sec_alexnet`—:numref:`sec_googlenet`와 같은 이전 실험에서는 모델의 필수 입력 형태에 맞게 이미지의 배율을 조정합니다.그러나 의미론적 분할에서는 이렇게 하려면 예측된 픽셀 클래스를 입력 영상의 원래 모양으로 다시 스케일링해야 합니다.이러한 크기 조정은 부정확할 수 있으며, 특히 클래스가 다른 세그먼트화된 영역의 경우 더욱 그렇습니다.이 문제를 방지하기 위해 이미지를 다시 스케일링하는 대신*고정* 모양으로 자릅니다.구체적으로, [** 이미지 확대에서 무작위 자르기를 사용하여 입력 영상과 라벨의 동일한 영역을 자르기**].

```{.python .input}
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab pytorch
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
```

```{.python .input}
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

```{.python .input}
#@tab pytorch
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

### [**커스텀 시맨틱 분할 데이터셋 클래스**]

상위 수준 API에서 제공하는 `Dataset` 클래스를 상속하여 사용자 지정 의미 체계 세분화 데이터 세트 클래스 `VOCSegDataset`를 정의합니다.`__getitem__` 함수를 구현하면 데이터 세트에서 `idx`로 인덱싱된 입력 이미지와 이 이미지의 각 픽셀의 클래스 인덱스에 임의로 액세스할 수 있습니다.데이터셋의 일부 이미지는 임의 자르기의 출력 크기보다 크기가 작기 때문에 이러한 예는 사용자 지정 `filter` 함수로 필터링됩니다.또한 `normalize_image` 함수를 정의하여 입력 이미지의 세 RGB 채널 값을 표준화합니다.

```{.python .input}
#@save
class VOCSegDataset(gluon.data.Dataset):
    """A customized dataset to load the VOC dataset."""
    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose(2, 0, 1),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

### [**데이터세트 읽기**]

사용자 지정 `VOCSegDatase`t 클래스를 사용하여 각각 훈련 세트와 테스트 세트의 인스턴스를 만듭니다.무작위로 잘린 이미지의 출력 모양이 $320\times 480$라고 지정한다고 가정합니다.아래에서는 훈련 세트와 테스트 세트에 남아 있는 예제의 수를 볼 수 있습니다.

```{.python .input}
#@tab all
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)
```

배치 크기를 64로 설정하면 훈련 세트에 대한 데이터 반복기를 정의합니다.첫 번째 미니 배치의 모양을 인쇄해 보겠습니다.이미지 분류 또는 물체 감지와 달리 여기서 레이블은 3 차원 텐서입니다.

```{.python .input}
batch_size = 64
train_iter = gluon.data.DataLoader(voc_train, batch_size, shuffle=True,
                                   last_batch='discard',
                                   num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

```{.python .input}
#@tab pytorch
batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

### [**모든 것을 하나로 모으기**]

마지막으로 파스칼 VOC2012 시맨틱 세분화 데이터 세트를 다운로드하고 읽기 위해 다음 `load_data_voc` 함수를 정의합니다.훈련 데이터 세트와 테스트 데이터 세트 모두에 대한 데이터 반복기를 반환합니다.

```{.python .input}
#@save
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
```

## 요약

* 시맨틱 분할은 이미지를 서로 다른 시맨틱 클래스에 속하는 영역으로 나누어 픽셀 수준에서 이미지의 내용을 인식하고 이해합니다.
* 가장 중요한 의미론적 분할 데이터 세트 중 하나는 파스칼 VOC2012 입니다.
* 의미론적 분할에서는 입력 영상과 레이블이 픽셀에서 일대일로 대응하므로 입력 이미지는 스케일링되지 않고 고정된 모양으로 무작위로 잘립니다.

## 연습문제

1. 시맨틱 분할은 자율 주행 차량 및 의료 이미지 진단에 어떻게 적용될 수 있습니까?다른 응용 프로그램을 생각해 볼 수 있습니까?
1. :numref:`sec_image_augmentation`에서 데이터 증대에 대한 설명을 상기하십시오.이미지 분류에 사용되는 이미지 증강 방법 중 의미론적 분할에 적용할 수 없는 것은 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/375)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1480)
:end_tab:
