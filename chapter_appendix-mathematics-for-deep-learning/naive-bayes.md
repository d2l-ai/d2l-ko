# 나이브 베이즈
:label:`sec_naive_bayes`

이전 섹션에서 확률 이론과 랜덤 변수에 대해 배웠습니다.이 이론을 적용하기 위해*naive Bayes* 분류기를 소개하겠습니다.이것은 숫자 분류를 수행 할 수 있도록 확률적 기본 사항 만 사용합니다. 

학습은 모두 가정을 만드는 것입니다.이전에 보지 못했던 새로운 데이터 예제를 분류하려면 어떤 데이터 예제가 서로 유사한지에 대해 몇 가지 가정을 해야 합니다.널리 사용되고 매우 명확한 알고리즘인 Naive Bayes 분류기는 계산을 단순화하기 위해 모든 기능이 서로 독립적이라고 가정합니다.이 섹션에서는 이미지의 문자를 인식하기 위해 이 모델을 적용합니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## 광학 문자 인식

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998`는 널리 사용되는 데이터 세트 중 하나입니다.여기에는 훈련을 위한 60,000개의 이미지와 검증을 위한 10,000개의 이미지가 포함되어 있습니다.각 이미지에는 0에서 9까지의 손으로 쓴 숫자가 포함되어 있습니다.작업은 각 이미지를 해당 숫자로 분류하는 것입니다. 

Gluon은 `data.vision` 모듈에서 `MNIST` 클래스를 제공하여 인터넷에서 데이터 세트를 자동으로 검색합니다.그 후 Gluon은 이미 다운로드한 로컬 사본을 사용합니다.매개 변수 `train`의 값을 각각 `True` 또는 `False`로 설정하여 훈련 세트를 요청할지 테스트 세트를 요청할지 여부를 지정합니다.각 이미지는 너비와 높이가 모두 $28$이고 모양이 있는 그레이스케일 이미지입니다 ($28$,$28$,$1$).사용자 지정 변환을 사용하여 마지막 채널 차원을 제거합니다.또한 데이터셋은 부호 없는 $8$비트 정수로 각 픽셀을 나타냅니다.문제를 단순화하기 위해 이진 기능으로 양자화합니다.

```{.python .input}
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```{.python .input}
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# Original pixel values of MNIST range from 0-255 (as the digits are stored as
# uint8). For this section, pixel values that are greater than 128 (in the
# original image) are converted to 1 and values that are less than 128 are
# converted to 0. See section 18.9.2 and 18.9.3 for why
train_images = tf.floor(tf.constant(train_images / 128, dtype = tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype = tf.float32))

train_labels = tf.constant(train_labels, dtype = tf.int32)
test_labels = tf.constant(test_labels, dtype = tf.int32)
```

이미지와 해당 레이블이 포함 된 특정 예제에 액세스 할 수 있습니다.

```{.python .input}
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label.numpy()
```

여기 변수 `image`에 저장된 이 예제는 높이와 너비가 $28$픽셀인 이미지에 해당합니다.

```{.python .input}
#@tab all
image.shape, image.dtype
```

이 코드는 각 이미지의 레이블을 스칼라로 저장합니다.유형은 $32$비트 정수입니다.

```{.python .input}
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

```{.python .input}
#@tab tensorflow
label.numpy(), label.dtype
```

동시에 여러 예제에 액세스할 수도 있습니다.

```{.python .input}
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10, 38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10, 38)])
images.shape, labels.shape
```

```{.python .input}
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
images.shape, labels.shape
```

이러한 예제를 시각화해 보겠습니다.

```{.python .input}
#@tab all
d2l.show_images(images, 2, 9);
```

## 분류를 위한 확률적 모델

분류 작업에서는 예제를 카테고리에 매핑합니다.예를 들어 회색조 $28\times 28$ 이미지이고 범주는 숫자입니다.(자세한 설명은 :numref:`sec_softmax`를 참조하십시오.)분류 작업을 표현하는 자연스러운 방법 중 하나는 확률적 질문을 사용하는 것입니다. 특징 (예: 이미지 픽셀) 이 주어질 가능성이 가장 높은 레이블은 무엇입니까?예제의 특징을 $\mathbf x\in\mathbb R^d$로 나타내고 레이블은 $y\in\mathbb R$로 나타냅니다.여기서 특징은 이미지 픽셀입니다. 여기서 $2$차원 이미지를 벡터로 변경하여 $d=28^2=784$가 되고 레이블이 숫자가 되도록 할 수 있습니다.특징이 주어진 레이블의 확률은 $p(y  \mid  \mathbf{x})$입니다.이 예에서 $y=0, \ldots,9$에 대해 $p(y  \mid  \mathbf{x})$인 이러한 확률을 계산할 수 있는 경우 분류기는 다음 표현식으로 지정된 예측 $\hat{y}$을 출력합니다. 

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$

안타깝게도 이를 위해서는 $\mathbf{x} = x_1, ..., x_d$의 모든 값에 대해 $p(y  \mid  \mathbf{x})$를 추정해야 합니다.각 피처가 $2$개의 값 중 하나를 사용할 수 있다고 가정해 보십시오.예를 들어, $x_1 = 1$ 기능은 주어진 문서에 사과라는 단어가 나타나고 $x_1 = 0$는 사과가 나타나지 않음을 나타낼 수 있습니다.$30$개의 바이너리 기능이 있다면 $2^{30}$ (10억 개 이상!) 중 하나를 분류할 준비가 필요하다는 뜻입니다.입력 벡터 $\mathbf{x}$의 가능한 값입니다. 

또한 학습은 어디에 있습니까?해당 레이블을 예측하기 위해 가능한 모든 예제를 살펴봐야 한다면 실제로 패턴을 배우는 것이 아니라 데이터 세트를 암기하는 것입니다. 

## 나이브 베이즈 분류기

다행스럽게도 조건부 독립성에 대한 몇 가지 가정을 통해 귀납적 편향을 도입하고 비교적 겸손한 교육 예제를 통해 일반화 할 수있는 모델을 구축 할 수 있습니다.먼저 Bayes 정리를 사용하여 분류기를 다음과 같이 표현해 보겠습니다. 

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

분모는 정규화 항 $p(\mathbf{x})$이며 레이블 $y$의 값에 의존하지 않습니다.결과적으로 $y$의 서로 다른 값에서 분자를 비교하는 것에 대해서만 걱정할 필요가 있습니다.분모를 계산하는 것이 다루기 힘든 것으로 판명되더라도 분자를 평가할 수있는 한 분모를 무시하면 벗어날 수 있습니다.다행히도 정규화 상수를 회복하고 싶더라도 그렇게 할 수 있습니다.$\sum_y p(y  \mid  \mathbf{x}) = 1$ 이후 정규화 항을 항상 복구할 수 있습니다. 

이제 $p( \mathbf{x}  \mid  y)$에 초점을 맞추겠습니다.확률의 연쇄 법칙을 사용하여 $p( \mathbf{x}  \mid  y)$라는 용어를 다음과 같이 표현할 수 있습니다. 

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

이 표현 자체로는 더 이상 우리를 얻지 못합니다.우리는 여전히 대략 $2^d$개의 모수를 추정해야 합니다.그러나*레이블이 주어진*특징이 조건부로 서로 독립적이라고 가정하면 이 항이 $\prod_i p(x_i  \mid  y)$로 단순화되어 예측 변수를 제공하므로 갑자기 훨씬 더 나은 모양을 갖게 됩니다. 

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

우리가 모든 $i$와 $y$에 대해 $p(x_i=1  \mid  y)$를 추정하고 그 값을 $P_{xy}[i, y]$에 저장할 수 있다면, 여기서 $P_{xy}$는 $d\times n$ 행렬이며, $n$은 클래스의 수이고 $y\in\{1, \ldots, n\}$입니다. 그러면 이것을 사용하여 $p(x_i = 0 \mid y)$를 추정할 수도 있습니다. 

$$ 
p(x_i = t_i \mid y) = 
\begin{cases}
    P_{xy}[i, y] & \text{for } t_i=1 ;\\
    1 - P_{xy}[i, y] & \text{for } t_i = 0 .
\end{cases}
$$

또한 모든 $y$에 대해 $p(y)$을 추정하고 $P_y[y]$에 저장합니다. $P_y$는 $n$ 길이 벡터입니다.그런 다음 새로운 예제 $\mathbf t = (t_1, t_2, \ldots, t_d)$의 경우 다음을 계산할 수 있습니다. 

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$
:eqlabel:`eq_naive_bayes_estimation`

모든 $y$를 위해.따라서 조건부 독립성에 대한 가정은 모델의 복잡성을 특징 수 $\mathcal{O}(2^dn)$에 대한 지수 의존성에서 선형 종속성 ($\mathcal{O}(dn)$) 으로 가져갔습니다. 

## 트레이닝

이제 문제는 우리가 $P_{xy}$와 $P_y$을 모른다는 것입니다.따라서 먼저 일부 훈련 데이터가 주어지면 그 값을 추정해야합니다.이것은 모델을*훈련*하는 것입니다.$P_y$을 추정하는 것은 그리 어렵지 않습니다.$10$ 클래스만 다루기 때문에 각 자릿수에 대한 발생 횟수 $n_y$을 계산하여 총 데이터 양 $n$로 나눌 수 있습니다.예를 들어, 숫자 8이 $n_8 = 5,800$번 발생하고 총 $n = 60,000$개의 이미지가 있는 경우 확률 추정치는 $p(y=8) = 0.0967$입니다.

```{.python .input}
X, Y = mnist_train[:]  # All training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab tensorflow
X = train_images
Y = train_labels

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

이제 약간 더 어려운 것 $P_{xy}$로 넘어갑니다.흑백 이미지를 선택했기 때문에 $p(x_i  \mid  y)$은 클래스 $y$에 대해 픽셀 $i$이 켜질 확률을 나타냅니다.이전과 마찬가지로 이벤트가 발생하도록 $n_{iy}$의 횟수를 세고 $y$의 총 발생 횟수, 즉 $n_y$로 나눌 수 있습니다.그러나 약간 문제가 있습니다. 특정 픽셀은 검은색이 아닐 수 있습니다 (예: 잘 자른 이미지의 경우 모서리 픽셀이 항상 흰색일 수 있음).통계학자가 이 문제를 처리하는 편리한 방법은 모든 발생에 의사 카운트를 추가하는 것입니다.따라서 $n_{iy}$이 아닌 $n_{iy}+1$을 사용하고 $n_y$ 대신 $n_{y} + 1$를 사용합니다.*라플라스 스무딩*이라고도 합니다.임시로 보일 수 있지만 베이지안 관점에서 동기를 부여 할 수 있습니다.

```{.python .input}
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 1).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 1).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 1), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```

이러한 $10\times 28\times 28$ 확률 (각 클래스의 각 픽셀에 대해) 을 시각화하면 평균적인 숫자를 얻을 수 있습니다. 

이제 :eqref:`eq_naive_bayes_estimation`를 사용하여 새 이미지를 예측할 수 있습니다.$\mathbf x$이 주어지면 다음 함수는 모든 $y$에 대해 $p(\mathbf x \mid y)p(y)$을 계산합니다.

```{.python .input}
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = train_images[0], train_labels[0]
bayes_pred(image)
```

이건 끔찍하게 잘못되었습니다!이유를 알아보기 위해 픽셀당 확률을 살펴보겠습니다.일반적으로 $0.001$에서 $1$ 사이의 숫자입니다.우리는 그 중 $784$을 곱하고 있습니다.이 시점에서 컴퓨터에서 이러한 숫자를 계산하므로 지수에 대한 고정 범위를 사용한다는 점을 언급 할 가치가 있습니다.우리가*수치 언더 플로우*를 경험한다는 것입니다. 즉, 모든 작은 숫자를 곱하면 0으로 반올림 될 때까지 더 작은 것이 생깁니다.우리는 이것을 :numref:`sec_maximum_likelihood`에서 이론적 문제로 논의했지만 실제로는 현상을 분명히 볼 수 있습니다. 

이 섹션에서 설명한 것처럼 $\log a b = \log a + \log b$, 즉 합산 로그로 전환한다는 사실을 사용하여이 문제를 해결합니다.$a$와 $b$가 모두 작은 숫자이더라도 로그 값은 적절한 범위에 있어야 합니다.

```{.python .input}
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab pytorch
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab tensorflow
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*tf.math.log(a).numpy())
```

로그는 증가하는 함수이므로 :eqref:`eq_naive_bayes_estimation`를 다음과 같이 다시 작성할 수 있습니다. 

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$

다음 안정 버전을 구현할 수 있습니다.

```{.python .input}
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

이제 예측이 올바른지 확인할 수 있습니다.

```{.python .input}
# Convert label which is a scalar tensor of int32 dtype to a Python scalar
# integer for comparison
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

```{.python .input}
#@tab tensorflow
tf.argmax(py, axis=0, output_type = tf.int32) == label
```

이제 몇 가지 검증 예제를 예측하면 Bayes 분류기가 잘 작동하는 것을 볼 수 있습니다.

```{.python .input}
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item() 
            for x in X]

X = torch.stack([mnist_test[i][0] for i in range(18)], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(18)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab tensorflow
def predict(X):
    return [tf.argmax(
        bayes_pred_stable(x), axis=0, output_type = tf.int32).numpy()
            for x in X]

X = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

마지막으로 분류기의 전체 정확도를 계산해 보겠습니다.

```{.python .input}
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab tensorflow
X = test_images
y = test_labels
preds = tf.constant(predict(X), dtype=tf.int32)
# Validation accuracy
tf.reduce_sum(tf.cast(preds == y, tf.float32)).numpy() / len(y)
```

최신 딥 네트워크는 $0.01$ 미만의 오류율을 달성합니다.상대적으로 성능이 좋지 않은 이유는 모델에서 만든 잘못된 통계적 가정 때문입니다. 레이블에만 의존하여 각 픽셀이*독립적으로* 생성된다고 가정했습니다.이것은 인간이 숫자를 쓰는 방식이 아니며, 이 잘못된 가정은 지나치게 순진한 (Bayes) 분류기의 몰락으로 이어졌습니다. 

## 요약* 베이즈 규칙을 사용하면 관찰된 모든 기능이 독립적이라고 가정하여 분류자를 만들 수 있습니다.* 이 분류기는 레이블과 픽셀 값 조합의 발생 횟수를 계산하여 데이터 세트에 대해 학습시킬 수 있습니다.* 이 분류기는 수십 년 동안 스팸과 같은 작업에서 황금 표준이었습니다.발각. 

## 연습 문제

1.두 요소 $[0,1,1,0]$의 XOR에서 제공한 레이블이 있는 데이터셋 $[[0,0], [0,1], [1,0], [1,1]]$을 고려해 보십시오.이 데이터셋에 구축된 나이브 베이즈 분류기의 확률은 얼마입니까?포인트를 성공적으로 분류합니까?그렇지 않다면 어떤 가정을 위반합니까?
1.확률을 추정 할 때 Laplace 평활화를 사용하지 않았으며 훈련에서 관찰되지 않은 값을 포함하는 테스트 시간에 데이터 예제가 도착했다고 가정합니다.모델 출력은 어떻게 될까요?
1.나이브 베이즈 분류기는 확률 변수의 종속성이 그래프 구조로 인코딩되는 베이지안 네트워크의 특정 예입니다.전체 이론은 이 섹션의 범위를 벗어나지만 (자세한 내용은 :cite:`Koller.Friedman.2009` 참조) XOR 모델에서 두 입력 변수 간에 명시적 종속성을 허용하면 성공적인 분류자를 만들 수 있는 이유를 설명합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/418)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1101)
:end_tab:
