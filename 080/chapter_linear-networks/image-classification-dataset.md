# The Image Classification Dataset
# 이미지 분류 데이터셋
:label:`sec_fashion_mnist`

One of the widely used dataset for image classification is the  MNIST dataset :cite:`LeCun.Bottou.Bengio.ea.1998`.
While it had a good run as a benchmark dataset,
even simple models by today's standards achieve classification accuracy over 95%,
making it unsuitable for distinguishing between stronger models and weaker ones.
Today, MNIST serves as more of sanity checks than as a benchmark.
To up the ante just a bit, we will focus our discussion in the coming sections
on the qualitatively similar, but comparatively complex Fashion-MNIST
dataset :cite:`Xiao.Rasul.Vollgraf.2017`, which was released in 2017.

MNIST 데이터셋 :cite:`LeCun.Bottou.Bengio.ea.1998`은 이미지 분류를 위한 데이터셋으로 많이 사용되는 것 중 하나입니다. 벤치마킹 데이터셋으로 잘 사용되어 왔지만, 오늘날의 표준으로 봤을 때 아주 간단한 모델도 95% 이상의 정확도를 달성하고 있어서, 강한 모델와 약한 모델을 구분하기에 적당하지 않게 되었습니다. 이제 MNIST는 벤치마킹 용도가 아니라 잘 동작하는지 체크하는 용도로 더 사용되고 있습니다. 조금 더 나아가기 위해서, 다음 절들에서 우리는 질적으로는 비슷하지만 비교적 복잡한 2017년에 발표된 Fashion-MNIST 데이터셋 :cite:`Xiao.Rasul.Vollgraf.2017`에 초점을 맞출 것입니다.

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

## Reading the Dataset
## 데이터셋 읽기

We can download and read the FashionMNIST dataset into memory via the the build-in functions in the framework.

프래임워크가 제공하는 함수들을 이용해서 FashionMNIST 데이터셋을 다운로드해서 메모리로 읽을 수 있습니다.

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

FashionMNIST consists of images from 10 categories, each represented
by 6000 images in the training set and by 1000 in the test set.
Consequently the training set and the test set
contain 60000 and 10000 images, respectively.

FashionMNIST는 10개 카테고리들에 속하는 이미지들로 구성되어 있고, 각 카테고리당 학습 데이터셋은 6000개 이미지, 테스트 데이터셋은 1000개 이미지가 있습니다. *테스트 데이터셋* (또는 *테스트 셋*)은 모델 성능을 평가하는데 사용되며, 학습에는 사용되지 않습니다. 결과적으로 학습 셋은 60000개, 테스트 셋은 10000개 이미지으로 구성되어 있습니다.

```{.python .input}
#@tab mxnet, pytorch
len(mnist_train), len(mnist_test)
```

```{.python .input}
#@tab tensorflow
len(mnist_train[0]), len(mnist_test[0])
```

The height and width of each input image are both 28 pixels.
Note that the dataset consists of grayscale images, whose number of channels is 1.
For brevity, throughout this book
we store the shape of any image with height $h$ width $w$ pixels as $h \times w$ or ($h$, $w$).

각 이미지의 높이와 폭는 모두 28 픽셀입니다. 데이터셋은 채널의 개수가 1인 회색 이미지로 구성되어 있습니다. 간결함을 위해서, 우리는 높이가 $h$이고 폭이 $w$ 픽셀인 이미지의 모양은 $h \times w$ 또는 ($h$, $w$)로 하겠습니다.

```{.python .input}
#@tab all
mnist_train[0][0].shape
```

```{.python .input}
#@tab tensorflow
mnist_train[0][0].shape
```

The images in Fashion-MNIST are associated with the following categories:
t-shirt, trousers, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot.
The following function converts between numeric label indices and their names in text.

Fashion-MNIST의 이미지들은 다음 카테고리들에 속해있습니다: 티셔츠, 바지, 풀오버, 드레스, 코드, 샌달, 셔츠, 스니커즈, 가방, 앵클 부츠. 다음 함수는 숫자 레이블들을 해당하는 테스트 레이블로 바꿔줍니다.

```{.python .input}
#@tab all
def get_fashion_mnist_labels(labels):  #@save
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

We can now create a function to visualize these examples.

이 예제들을 시각화하는 함수를 정의합니다.

```{.python .input}
#@tab all
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

Here are the images and their corresponding labels (in text)
for the first few examples in the training dataset.

다음은 학습 데이터셋의 처음 몇 개의 이미지들과 해당하는 레이블(텍스트 형태)들입니다.

```{.python .input}
X, y = mnist_train[:18]
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

## Reading a Minibatch
## 미니배치 읽기

To make our life easier when reading from the training and test sets,
we use the built-in data iterator rather than creating one from scratch.
Recall that at each iteration, a load loader
reads a minibatch of data with size `batch_size` each time.
We also randomly shuffle the examples for the training data iterator.

학습 데이터셋과 테스트 데이터셋을 읽는 것을 편하게 만들기 위해서, 직접 구현하지 않고 만들어진 데이터 반복자를 사용합니다. 매 반복마다, 데이터 로더는 매 번  `batch_size` 크기의 미니배치 데이터를 읽습니다. 또한 학습 데이터 반복자는 샘플들을 의의로 섞습니다.

```{.python .input}
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data expect for Windows."""
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

Let us look at the time it takes to read the training data.

학습 데이터를 읽는데 걸리는 시간을 측정해 보겠습니다.

```{.python .input}
#@tab all
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```

## Putting All Things Together
## 모두 합치기

Now we define the `load_data_fashion_mnist` function
that obtains and reads the Fashion-MNIST dataset.
It returns the data iterators for both the training set and validation set.
In addition, it accepts an optional argument to resize images to another shape.

Fashion-MNIST 데이터셋을 획득하고 읽는 함수인 `load_data_fashion_mnist` 를 정의하겠습니다. 이 함수는 학습 셋과 검증 셋에 대한 데이터 반복자들을 리턴합니다. 또한, 이미지를 다른 모양으로 크기를 변경하는 데 사용될 선택적인 변수도 받습니다.

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

Below we test the image resizing feature of the `load_data_fashion_mnist` function
by specifying the `resize` argument.

다음은 `resize` 변수를 설정해서 `load_data_fashion_mnist` 함수의 크기 변경 기능을 테스트합니다.

```{.python .input}
#@tab all
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

We are now ready to work with the Fashion-MNIST dataset in the sections that follow.

이제 우리는 이어질 절들에서 Fashion-MNIST 데이터셋을 사용할 준비를 마쳤습니다.

## Summary
## 요약

* Fashion-MNIST is an apparel classification dataset consisting of images representing 10 categories. We will use this dataset in subsequent sections and chapters to evaluate various classification algorithms.
* We store the shape of any image with height $h$ width $w$ pixels as $h \times w$ or ($h$, $w$).
* Data iterators are a key component for efficient performance. Rely on well-implemented data iterators that exploit high-performance computing to avoid slowing down your training loop.

* Fashion-MNIST는 의류 분류 데이터셋으로 10개 카테고리의 이미지들로 구성되어 있습니다. 우리는 이 데이터셋을 다음 절들과 장들에서 다양한 분류 알고리즘을 평가하는데 사용할 것입니다.
* 높이가 $h$, 폭이 $w$인 이미지의 모양을 $h \times w$ 또는 ($h$, $w$) 형태로 저장합니다.
* 데이터 반복자는 효과적인 성능을 위한 중요 컴포넌트입니다. 학습 룹의 속도 저하를 막기 위해서 고성능 컴퓨팅을 잘 이용하도록 잘 구현된 데이터 반복자를 사용하세요.

## Exercises
## 연습문제

1. Does reducing the `batch_size` (for instance, to 1) affect the reading performance?
1. The data iterator performance is important. Do you think the current implementation is fast enough? Explore various options to improve it.
1. Check out the framework's online API documentation. Which other datasets are available?

1. `batch_size`를 줄이는 것이(예를 들어 1) 읽기 성능에 영향을 미칠까요?
1. 데이터 반복자의 성능은 중요합니다. 현재 구현이 충분이 빠르다고 생각하나요? 향상 시키기 위한 다양한 옵션을 찾아보세요.
1. 프래임워크의 온라인 API 문서를 확인해보세요. 어떤 다른 데이터셋이 제공되나요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/224)
:end_tab:
