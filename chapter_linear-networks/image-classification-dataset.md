# 이미지 분류 데이터셋
:label:`sec_fashion_mnist`

(~~MNIST 데이터 세트는 이미지 분류에 널리 사용되는 데이터 세트 중 하나이지만 벤치 마크 데이터 세트로는 너무 간단합니다.유사하지만 더 복잡한 패션-MNIST 데이터세트~~) 

이미지 분류에 널리 사용되는 데이터 세트 중 하나는 MNIST 데이터 세트 :cite:`LeCun.Bottou.Bengio.ea.1998`입니다.벤치마크 데이터 세트로 잘 실행되었지만 오늘날의 표준에 따른 단순한 모델조차도 95% 이상의 분류 정확도를 달성하므로 더 강력한 모델과 약한 모델을 구분하는 데는 적합하지 않습니다.오늘날 MNIST는 벤치마크보다 더 많은 온전성 검사 역할을 합니다.분담금을 조금 높이기 위해 다음 섹션에서 2017 년에 출시 된 질적으로 유사하지만 비교적 복잡한 패션 MNIST 데이터 세트 :cite:`Xiao.Rasul.Vollgraf.2017`에 대한 토론에 초점을 맞출 것입니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon
import sys

d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms
from torch.utils import data

d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

## 데이터세트 읽기

[**프레임워크의 빌드 인 함수를 통해 Fashion-MNIST 데이터세트를 다운로드하여 메모리로 읽을 수 있습니다.**]

```{.python .input}
mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)
```

```{.python .input}
#@tab pytorch
# `ToTensor` converts the image data from PIL type to 32-bit floating point
# tensors. It divides all numbers by 255 so that all pixel values are between
# 0 and 1
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

```{.python .input}
#@tab tensorflow
mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST는 10개 범주의 이미지로 구성되며, 각 이미지는 훈련 데이터셋에서 6000개의 이미지로, 테스트 데이터셋에서는 1000개의 이미지로 표현됩니다.*테스트 데이터 세트* (또는*테스트 세트*) 는 모델 성능을 평가하는 데 사용되며 훈련에는 사용되지 않습니다.따라서 훈련 세트와 테스트 세트에는 각각 60000개 및 10000개의 이미지가 포함됩니다.

```{.python .input}
#@tab mxnet, pytorch
len(mnist_train), len(mnist_test)
```

```{.python .input}
#@tab tensorflow
len(mnist_train[0]), len(mnist_test[0])
```

각 입력 이미지의 높이와 너비는 모두 28픽셀입니다.데이터셋은 채널 수가 1인 회색조 이미지로 구성됩니다.간결하게, 이 책 전체에서 높이가 $h$, 너비 $w$ 픽셀인 이미지의 모양을 $h \times w$ 또는 ($h$, $w$) 로 저장합니다.

```{.python .input}
#@tab all
mnist_train[0][0].shape
```

[~~데이터 세트를 시각화하는 두 가지 유틸리티 함수 ~~] 

Fashion-MNIST의 이미지는 티셔츠, 바지, 풀오버, 드레스, 코트, 샌들, 셔츠, 운동화, 가방 및 발목 부츠와 관련이 있습니다.다음 함수는 숫자형 레이블 인덱스와 텍스트의 이름을 변환합니다.

```{.python .input}
#@tab all
def get_fashion_mnist_labels(labels):  #@save
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

이제 이러한 예제를 시각화하는 함수를 만들 수 있습니다.

```{.python .input}
#@tab mxnet, tensorflow
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(d2l.numpy(img))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

```{.python .input}
#@tab pytorch
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

다음은 훈련 데이터셋의 처음 몇 가지 예제에 대한 [**이미지와 해당 레이블**](텍스트) 입니다.

```{.python .input}
X, y = mnist_train[:18]

print(X.shape)
show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab pytorch
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab tensorflow
X = tf.constant(mnist_train[0][:18])
y = tf.constant(mnist_train[1][:18])
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y));
```

## 미니배치 읽기

훈련 세트와 테스트 세트를 읽을 때 삶을 더 쉽게 만들기 위해 처음부터 만드는 대신 내장 데이터 반복기를 사용합니다.각 반복에서 데이터 반복기는 [**매번 크기가 `batch_size`인 데이터의 미니 배치를 읽습니다.**] 또한 훈련 데이터 반복기에 대한 예제를 무작위로 섞습니다.

```{.python .input}
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data except for Windows."""
    return 0 if sys.platform.startswith('win') else 4

# `ToTensor` converts the image data from uint8 to 32-bit floating point. It
# divides all numbers by 255 so that all pixel values are between 0 and 1
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data."""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab tensorflow
batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices(
    mnist_train).batch(batch_size).shuffle(len(mnist_train[0]))
```

훈련 데이터를 읽는 데 걸리는 시간을 살펴보겠습니다.

```{.python .input}
#@tab all
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```

## 모든 것을 하나로 모으다

이제 [**Fashion-MNIST 데이터세트를 가져오고 읽는 `load_data_fashion_mnist` 함수**] 를 정의하겠습니다. 이 함수는 훈련 세트와 검증 세트 모두에 대한 데이터 반복자를 반환합니다.또한 이미지 크기를 다른 모양으로 조정하는 선택적 인수를 사용할 수 있습니다.

```{.python .input}
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab pytorch
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab tensorflow
def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))
```

아래에서는 `resize` 인수를 지정하여 `load_data_fashion_mnist` 함수의 이미지 크기 조정 기능을 테스트합니다.

```{.python .input}
#@tab all
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

이제 다음 섹션에서 패션-MNIST 데이터세트로 작업할 준비가 되었습니다. 

## 요약

* 패션-MNIST는 10개 카테고리를 나타내는 이미지로 구성된 의류 분류 데이터셋입니다.다음 섹션과 장에서 이 데이터세트를 사용하여 다양한 분류 알고리즘을 평가합니다.
* 우리는 높이가 $h$ 너비가 $w$ 픽셀인 모든 이미지의 모양을 $h \times w$ 또는 ($h$, $w$) 로 저장합니다.
* 데이터 반복기는 효율적인 성능을 위한 핵심 구성 요소입니다.훈련 루프의 속도를 늦추지 않도록 고성능 컴퓨팅을 활용하는 잘 구현된 데이터 반복기를 활용하십시오.

## 연습문제

1. `batch_size` (예: 1) 를 줄이면 읽기 성능에 영향을 줍니까?
1. 데이터 반복기 성능이 중요합니다.현재 구현이 충분히 빠르다고 생각하십니까?다양한 옵션을 탐색하여 개선하십시오.
1. 프레임워크의 온라인 API 문서를 확인하세요.사용할 수 있는 다른 데이터셋은 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/224)
:end_tab:
