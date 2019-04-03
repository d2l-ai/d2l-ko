# 이미지 분류 데이터 (Fashion-MNIST)

softmax regression 구현에 앞서 적절한 데이터셋이 필요합니다. 시각적으로 돋보이는 것을 만들기 위해서, 분류 문제에서 선택해보겠습니다. 

다음 장들에서 모델 정확도의 차이를 관찰하거나, 비교 알고리즘의 연산 효율성에 대한 이야기를 할 때에도 반복해서 사용할 예제입니다. 가장 흔한 이미지 분류 데이터셋은 MNIST 손글씨 숫자 인식 데이터셋이 있습니다. 이 데이터셋은 1990년 대에 Lecun, Cortes와 Burges에 의해서 제안되었습니다. 하지만, 거의 모든 모델이 MNIST 데이터셋에 대해서 95% 이상의 정확도를 보여주기 때문에, 모델들 사이의 차이를 설명하기에 적합하지 않습니다. 알고리즘들의 차이를 보다 직관적으로 보여주기 위해서, 더 복잡한 데이터셋을 사용하겠습니다. 이 데이터셋은 Fashion-MNIST라를 것으로 2017년에 Xio, Rasul 그리고 Vollgraf가 제안했습니다. 

## 데이터 구하기

우선, 이 절에서 필요한 패키지와 모듈을  import 합니다.

```{.python .input}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet.gluon import data as gdata
import sys
import time
```

다음으로, Gluon의 `data` 패키지를 이용해서 이 데이터셋을 다운로드합니다. 데이터셋은 처음 불렸을 때, 인터넷으로부터 자동으로 다운로드됩니다. `train` 파라메터를 통해서 학습 데이터셋을 받을 것인지 테스트 데이터셋을 받을 것인지를 정할 수 있습니다. 테스트 데이터셋 또는 테스팅 데이터셋은 모델의 성능을 평가할 때만 쓰이고, 학습에는 사용되지 않는 데이터입니다.

```{.python .input  n=23}
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
```

학습 데이터셋과 테스트 데이터셋은 각 카테고리별로 각각 6,000개와 1,000개의 이미지들로 구성되어 있습니다. 카테고리 개수는 10개이기에, 학습 데이터는 총 60,000개의 이미지들로 테스팅 셋은 10,000개 이미지들을 가지고 있습니다.

```{.python .input}
len(mnist_train), len(mnist_test)
```

`[]` 을 이용하면 각 샘플을 접근할 수 있습니다. 첫번째 데이터의 이미지와 label를 얻어보겠습니다.

```{.python .input  n=24}
feature, label = mnist_train[0]
```

`feature` 변수는 높이와 넓이가 모두 28 픽셀인 이미지 데이터를 가지고 있습니다. 각 픽셀은 8-bit unsigned integer (uint8)이고, 0부터 255 사이의 값을 갖습니다. 이는 3차원 NDArray에 저장됩니다. 마지막 차원은 채널의 개수를 의미합니다. 데이터 셋이 회색 이미지이기 때문에, 채널의 수는 1이 됩니다. 간단하게 하기 위해서, 이미지의 모양이 높이 `h`, 넓이는 `w` 픽셀인 경우 이미지의 shape을  $h \times w$ 또는  `(h, w)` 로 표기하도록 하겠습니다.

```{.python .input}
feature.shape, feature.dtype
```

각 이미지에 대한 label은 NumPy의 scalar로 저장되어 있고, 이는 32-bit integer 형태입니다.

```{.python .input}
label, type(label), label.dtype
```

Fashion-MNIST에는 10개의 카테고리가 있는데, 이들은 티셔츠, 바지, 풀오버, 드레스, 코드, 센달, 셔츠, 스니커, 가방, 발목 부츠입니다. 숫자 형태의 label을 텍스트 label로 바꿔주는 함수를 아래와 같이 정의합니다.

```{.python .input  n=25}
# This function has been saved in the d2l package for future use
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

아래 함수는 한 줄에 여러 이미지와 그 이미지의 label을 그리는 것을 정의합니다.

```{.python .input}
# This function has been saved in the d2l package for future use
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # Here _ means that we ignore (not use) variables
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
```

학습 데이터셋의 처음 9개의 샘플들에 대한 이미지와 텍스트 label을 살펴보겠습니다.

```{.python .input  n=27}
X, y = mnist_train[0:9]
show_fashion_mnist(X, get_fashion_mnist_labels(y))
```

## 미니배치 읽기

학습 데이터나 테스트 데이터를 읽는 코드를  ["Linear Regression Implementation Starting from Scratch"](linear-regression-scratch.md) 에서처럼 직접 작성하지 않고 `DataLoad` 를 사용하도록 하겠습니다. 데이터 로더는 매 번 `batch_size` 개수의 샘플을 갖는 미니 배치를 읽습니다.

실제 수행을 할 때, 데이터를 읽는 것이 성능의 병목이 되는 것을 볼 수 있습니다. 특히, 모델이 간단하거나 컴퓨터가 빠를 경우에 더욱 그렇습니다. `DataLoader` 의 유용한 특징은 데이터 읽기 속도를 빠르게 하기 위해서 멀티 프로세스들 사용할 수 있다는 것입니다. (단 현재 Windows에서는 지원되지 않습니다) 예를 들면, `num_workers` 설정을 통해서 4개의 프로세스가 데이터를 읽도록 만들 수 있습니다.

추가적으로 `ToTensor` 클래스를 이용해서 이미지 데이터를 uint8에서 32 bit floating point number로 변환합니다. 이 후, 모든 숫자를 255로 나눠서 모든 픽셀의 값이 0과 1사이가 되도록 합니다. `ToTensor` 클래스는 이미지 채널을 마지막 차원에서 첫번째 차원으로 바꿔주는 기능이 있는데, 이는 다음에 소개할 convolutional neural network 계산과 관련이 있습니다. 데이터셋의 `transform_first` 함수를 이용하면,  `ToTensor` 의 변환을 각 데이터 샘플 (이미지와 label)의 첫번째 원소인 이미지에 적용할 수 있습니다.

```{.python .input  n=28}
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    # 0 means no additional processes are needed to speed up the reading of
    # data
    num_workers = 0
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=True,
                              num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, shuffle=False,
                             num_workers=num_workers)
```

Fashion-MNIST 데이터 셋을 가지고 와서 읽는 로직은 `g2l.load_data_fashion_mnist` 함수 내부에 구현되어 있습니다. 이 함수는 다음 장들에서 사용될 예정입니다. 이 함수는 `train_iter` 와 `test_iter` 두 변수를 리턴합니다. 이 책에서는 내용이 깊어짐에 따라 이 함수를 향상시켜보겠습니다. 전체 구현에 대한 자세한 내용은  ["Deep Convolutional Neural Networks (AlexNet)"](../chapter_convolutional-neural-networks/alexnet.md) 절에서 설명하겠습니다.

학습 데이터를 읽는데 걸리는 시간을 측정해 보겠습니다.

```{.python .input}
start = time.time()
for X, y in train_iter:
    continue
'%.2f sec' % (time.time() - start)
```

## 요약

* Fashion-MNIST는 의류 분류 데이터 셋으로 10개의 카테고리로 분류되어 있습니다. 다음 장들에서 다양한 알고리즘의 성능을 테스트하는데 사용할 예정입니다.
* 이미지의 shape은 높이 `h` 픽셀, 넓이 `w` 픽셀을 이용해서  $h \times w$ 나 `(h, w)` 로 저장됩니다.
* 데이터 이터레이터(iterator)는 효율적인 성능을 위한 중요한 컴포넌트입니다. 가능하면 제공되는 것들을 사용하세요.

## 문제

1. `batch_size` 를 줄이면 (예를 들면 1) 읽기 성능에 영향을 미칠까요?
1. Windows 사용자가 아니라면, `num_workers` 을 바꾸면서 읽기 성능이 어떻게 영향을 받는지 실험해보세요.
1. `mxnet.gluon.data.vision` 에서 어떤 데이터셋들이 제공되는지 MXNet 문서를 통해서 확인해보세요.
1. `mxnet.gluon.data.vision.transforms` 에서 어떤 변환들이 제공되는지 MXNet 문서를 통해서 확인해보세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2335)

![](../img/qr_fashion-mnist.svg)
