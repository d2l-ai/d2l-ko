# 기하학 및 선형 대수 연산
:label:`sec_geometry-linear-algebraic-ops`

:numref:`sec_linear-algebra`에서 우리는 선형 대수의 기본 사항을 접하고 데이터 변환을 위한 일반적인 연산을 표현하는 데 어떻게 사용될 수 있는지 확인했습니다.선형 대수는 딥 러닝과 기계 학습에서 더 광범위하게 수행하는 많은 작업의 기초가되는 핵심 수학적 기둥 중 하나입니다.:numref:`sec_linear-algebra`에는 최신 딥 러닝 모델의 메커니즘을 전달하기에 충분한 기계가 포함되어 있지만 주제에는 훨씬 더 많은 것이 있습니다.이 섹션에서는 선형 대수 연산에 대한 몇 가지 기하학적 해석을 강조하고 고유값과 고유벡터를 포함한 몇 가지 기본 개념을 소개하면서 더 자세히 살펴볼 것입니다. 

## 벡터의 기하학 

먼저, 공간의 점이나 방향으로 벡터의 두 가지 일반적인 기하학적 해석에 대해 논의해야 합니다.기본적으로 벡터는 아래 Python 목록과 같은 숫자 목록입니다.

```{.python .input}
#@tab all
v = [1, 7, 0, 1]
```

수학자들은 이것을*column* 또는*row* 벡터로 쓰는 경우가 가장 많습니다. 

$$
\mathbf{x} = \begin{bmatrix}1\\7\\0\\1\end{bmatrix},
$$

또는 

$$
\mathbf{x}^\top = \begin{bmatrix}1 & 7 & 0 & 1\end{bmatrix}.
$$

데이터 예제는 열 벡터이고 가중 합을 형성하는 데 사용되는 가중치는 행 벡터인 경우 해석이 서로 다른 경우가 많습니다.그러나 융통성이 있으면 도움이 될 수 있습니다.:numref:`sec_linear-algebra`에서 설명한 것처럼 단일 벡터의 기본 방향은 열 벡터이지만 테이블 형식 데이터 세트를 나타내는 모든 행렬에 대해 각 데이터 예제를 행렬의 행 벡터로 처리하는 것이 더 일반적입니다. 

벡터가 주어지면, 우리가 제공해야 할 첫 번째 해석은 공간의 한 점이다.2 차원 또는 3 차원에서 벡터의 구성 요소를 사용하여*origin*이라는 고정 참조와 비교하여 공간에서 점의 위치를 정의하여 이러한 점을 시각화 할 수 있습니다.이것은 :numref:`fig_grid`에서 볼 수 있습니다. 

![An illustration of visualizing vectors as points in the plane.  The first component of the vector gives the $x$-coordinate, the second component gives the $y$-coordinate.  Higher dimensions are analogous, although much harder to visualize.](../img/grid-points.svg)
:label:`fig_grid`

이 기하학적 관점을 통해 문제를 좀 더 추상적 수준으로 고려할 수 있습니다.사진을 고양이 나 개로 분류하는 것과 같은 극복 할 수없는 문제에 더 이상 직면하지 않으면 작업을 추상적으로 공간의 점 모음으로 고려하고 작업을 두 개의 서로 다른 점 군집을 분리하는 방법을 발견하는 것으로 묘사 할 수 있습니다. 

동시에 사람들이 종종 벡터를 취하는 두 번째 관점이 있습니다. 우주의 방향입니다.벡터 $\mathbf{v} = [3,2]^\top$을 오른쪽으로 $3$ 단위, 원점에서 $2$ 단위 위로 위치로 생각할 수 있을 뿐만 아니라 오른쪽으로 $3$ 단계를 밟고 $2$ 단계 올라가는 방향 자체라고 생각할 수도 있습니다.이런 식으로 그림 :numref:`fig_arrow`의 모든 벡터를 동일하게 간주합니다. 

![Any vector can be visualized as an arrow in the plane.  In this case, every vector drawn is a representation of the vector $(3,2)^\top$.](../img/par-vec.svg)
:label:`fig_arrow`

이러한 변화의 이점 중 하나는 벡터 덧셈의 행위를 시각적으로 이해할 수 있다는 것입니다.특히 :numref:`fig_add-vec`에서 볼 수 있듯이 한 벡터에 의해 주어진 방향을 따른 다음 다른 벡터에 의해 주어진 방향을 따릅니다. 

![We can visualize vector addition by first following one vector, and then another.](../img/vec-add.svg)
:label:`fig_add-vec`

벡터 뺄셈도 비슷한 해석이 있습니다.$\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$라는 정체성을 고려하면 벡터 $\mathbf{u}-\mathbf{v}$이 점 $\mathbf{v}$에서 점 $\mathbf{u}$로 이동하는 방향이라는 것을 알 수 있습니다. 

## 도트 곱과 

각도 :numref:`sec_linear-algebra`에서 보았 듯이 두 개의 열 벡터 $\mathbf{u}$와 $\mathbf{v}$을 취하면 다음을 계산하여 내적을 형성 할 수 있습니다. 

$$\mathbf{u}^\top\mathbf{v} = \sum_i u_i\cdot v_i.$$
:eqlabel:`eq_dot_def`

:eqref:`eq_dot_def`는 대칭이기 때문에 고전 곱셈의 표기법을 반영하고 

$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u},
$$

벡터의 순서를 교환하면 동일한 답이 생성된다는 사실을 강조합니다. 

내적 :eqref:`eq_dot_def`는 또한 기하학적 해석 : it is closely related to the angle between two vectors.  Consider the angle shown in :numref:`fig_angle`를 인정합니다. 

![Between any two vectors in the plane there is a well defined angle $\theta$.  We will see this angle is intimately tied to the dot product.](../img/vec-angle.svg)
:label:`fig_angle`

먼저 두 가지 특정 벡터를 고려해 보겠습니다. 

$$
\mathbf{v} = (r,0) \; \text{and} \; \mathbf{w} = (s\cos(\theta), s \sin(\theta)).
$$

벡터 $\mathbf{v}$는 길이 $r$이고 $x$축에 평행하게 실행되며, 벡터 $\mathbf{w}$은 길이가 $s$이고 $x$축과 각도 $\theta$입니다.이 두 벡터의 내적을 계산하면 

$$
\mathbf{v}\cdot\mathbf{w} = rs\cos(\theta) = \|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta).
$$

몇 가지 간단한 대수 조작으로 항을 재정렬하여 

$$
\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).
$$

요컨대, 이 두 특정 벡터에 대해 규범과 결합 된 내적은 두 벡터 사이의 각도를 알려줍니다.이 같은 사실이 일반적으로 사실입니다.그러나 여기서 표현을 도출하지는 않겠지 만, $\|\mathbf{v} - \mathbf{w}\|^2$를 두 가지 방법으로 작성하는 것을 고려하면 하나는 내적을 사용하고 다른 하나는 코사인 법칙을 기하학적으로 사용하면 완전한 관계를 얻을 수 있습니다.실제로 두 벡터 $\mathbf{v}$과 $\mathbf{w}$에 대해 두 벡터 사이의 각도는 다음과 같습니다. 

$$\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).$$
:eqlabel:`eq_angle_forumla`

계산에서 2차원을 참조하는 것이 없으므로 이는 좋은 결과입니다.실제로 우리는 문제 없이 3백만 또는 3백만 차원에서 이것을 사용할 수 있습니다. 

간단한 예로, 한 쌍의 벡터 사이의 각도를 계산하는 방법을 살펴보겠습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import gluon, np, npx
npx.set_np()

def angle(v, w):
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))

angle(np.array([0, 1, 2]), np.array([2, 3, 4]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
from torchvision import transforms
import torchvision

def angle(v, w):
    return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))

angle(torch.tensor([0, 1, 2], dtype=torch.float32), torch.tensor([2.0, 3, 4]))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

def angle(v, w):
    return tf.acos(tf.tensordot(v, w, axes=1) / (tf.norm(v) * tf.norm(w)))

angle(tf.constant([0, 1, 2], dtype=tf.float32), tf.constant([2.0, 3, 4]))
```

지금은 사용하지 않겠지 만 각도가 $\pi/2$ (또는 이에 상응하는 $90^{\circ}$) 인 벡터를*직교*로 언급한다는 것을 아는 것이 유용합니다.위의 방정식을 살펴보면 $\theta = \pi/2$가 $\cos(\theta) = 0$와 같은 경우에 이런 일이 발생한다는 것을 알 수 있습니다.이런 일이 발생할 수 있는 유일한 방법은 내적 자체가 0이고 두 벡터가 $\mathbf{v}\cdot\mathbf{w} = 0$인 경우에만 직교하는 경우입니다.이것은 물체를 기하학적으로 이해할 때 유용한 공식이 될 것입니다. 

다음과 같이 질문하는 것이 합리적입니다. 각도 계산이 유용한 이유는 무엇입니까?답은 데이터가 가질 것으로 예상되는 일종의 불변성에 있습니다.이미지와 모든 픽셀 값이 동일하지만 밝기는 $10\%$인 중복 이미지를 생각해 보십시오.개별 픽셀의 값은 일반적으로 원래 값과는 거리가 멀다.따라서 원본 이미지와 어두운 이미지 사이의 거리를 계산하면 거리가 커질 수 있습니다.그러나 대부분의 ML 응용 프로그램에서*콘텐츠*는 동일합니다. 고양이/개 분류기에 관한 한 여전히 고양이의 이미지입니다.그러나 각도를 고려하면 모든 벡터 
$\mathbf{v}$ 에서 $\mathbf{v}$ 와 $0.1\cdot\mathbf{v}$ 
사이의 각도가 0이라는 것을 알기가 어렵지 않습니다.이는 스케일링 벡터가 동일한 방향을 유지하고 길이만 변경한다는 사실에 해당합니다.각도는 어두운 이미지를 동일하게 간주합니다. 

이런 예는 어디에나 있습니다.텍스트에서 동일한 내용을 말하는 문서를 두 배 더 길게 작성하면 논의 중인 주제가 변경되지 않도록 할 수 있습니다.일부 인코딩 (예: 일부 어휘에서 단어 발생 횟수 계산) 의 경우 이는 문서를 인코딩하는 벡터의 두 배에 해당하므로 각도를 다시 사용할 수 있습니다. 

### 코사인 유사성 

두 벡터의 근접성을 측정하기 위해 각도가 사용되는 ML 맥락에서 실무자는*코사인 유사성*이라는 용어를 채택하여 

$$
\cos(\theta) = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}.
$$

코사인은 두 벡터가 같은 방향을 가리킬 때 최대값 $1$, 반대 방향을 가리킬 때 최소값 $-1$, 두 벡터가 직교인 경우 $0$ 값을 취합니다.고차원 벡터의 성분이 평균 $0$으로 랜덤하게 샘플링되는 경우 코사인은 거의 항상 $0$에 가깝습니다. 

## 하이퍼플레인

벡터로 작업하는 것 외에도 선형 대수에서 멀리 가기 위해 이해해야 할 또 다른 주요 객체는*하이퍼 평면*입니다. 이 객체는 선 (2 차원) 또는 평면 (3 차원) 의 더 높은 차원으로 일반화됩니다.$d$차원 벡터 공간에서 초평면은 $d-1$ 차원을 가지며 공간을 두 개의 절반 공간으로 나눕니다. 

예를 들어 시작하겠습니다.열 벡터 $\mathbf{w}=[2,1]^\top$가 있다고 가정합니다.우리는 “$\mathbf{w}\cdot\mathbf{v} = 1$과 함께 $\mathbf{v}$의 요점은 무엇입니까?”내적과 :eqref:`eq_angle_forumla` 이상의 각도 사이의 연결을 회상하면 이것이

$$
\|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta) = 1 \; \iff \; \|\mathbf{v}\|\cos(\theta) = \frac{1}{\|\mathbf{w}\|} = \frac{1}{\sqrt{5}}.
$$

![Recalling trigonometry, we see the formula $\|\mathbf{v}\|\cos(\theta)$ is the length of the projection of the vector $\mathbf{v}$ onto the direction of $\mathbf{w}$](../img/proj-vec.svg)
:label:`fig_vector-project`

이 표현의 기하학적 의미를 고려하면 $\mathbf{v}$을 $\mathbf{w}$의 방향으로 투영하는 길이가 :numref:`fig_vector-project`에 표시된 것처럼 정확히 $1/\|\mathbf{w}\|$라고 말하는 것과 동일하다는 것을 알 수 있습니다.이것이 참인 모든 점의 집합은 벡터 $\mathbf{w}$에 직각을 이루는 선입니다.원한다면 이 직선에 대한 방정식을 찾아 $2x + y = 1$ 또는 이에 상응하는 $y = 1 - 2x$임을 알 수 있습니다. 

이제 $\mathbf{w}\cdot\mathbf{v} > 1$ 또는 $\mathbf{w}\cdot\mathbf{v} < 1$으로 점 집합에 대해 질문 할 때 어떤 일이 발생하는지 살펴보면 예측이 각각 $1/\|\mathbf{w}\|$보다 길거나 짧은 경우임을 알 수 있습니다.따라서 이 두 부등식은 선의 양쪽을 정의합니다.이런 식으로 우리는 공간을 두 개의 절반으로 자르는 방법을 찾았습니다. 여기서 한쪽의 모든 점은 임계 값 아래의 내적을 가지며 다른 쪽은 :numref:`fig_space-division`에서 볼 수 있습니다. 

![If we now consider the inequality version of the expression, we see that our hyperplane (in this case: just a line) separates the space into two halves.](../img/space-division.svg)
:label:`fig_space-division`

더 높은 차원의 이야기는 거의 같습니다.이제 $\mathbf{w} = [1,2,3]^\top$를 취하고 $\mathbf{w}\cdot\mathbf{v} = 1$을 사용하여 3 차원의 점에 대해 물어 보면 주어진 벡터 $\mathbf{w}$에 직각으로 평면을 얻습니다.두 부등식은 :numref:`fig_higher-division`에 표시된 것처럼 평면의 양면을 다시 정의합니다. 

![Hyperplanes in any dimension separate the space into two halves.](../img/space-division-3d.svg)
:label:`fig_higher-division`

이 시점에서 시각화 능력은 부족하지만 수십, 수백 또는 수십억 차원에서 이러한 작업을 수행하는 것을 막을 수있는 것은 없습니다.기계 학습 모델에 대해 생각할 때 자주 발생합니다.예를 들어, :numref:`sec_softmax`와 같은 선형 분류 모델을 서로 다른 대상 클래스를 분리하는 초평면을 찾는 방법으로 이해할 수 있습니다.이러한 맥락에서 이러한 초평면을 종종*결정 평면*이라고 합니다.대부분의 심층 학습 분류 모델은 소프트 맥스에 공급되는 선형 계층으로 끝나므로 대상 클래스가 초평면에 의해 깨끗하게 분리 될 수 있도록 비선형 임베딩을 찾는 심층 신경망의 역할을 해석 할 수 있습니다. 

손으로 만든 예제를 제공하기 위해 의사 결정 평면을 정의하고 조잡한 임계 값을 안구하는 수단 사이의 벡터를 가져 와서 Fashion MNIST 데이터 세트 (:numref:`sec_fashion_mnist`에서 볼 수 있음) 에서 티셔츠와 바지의 작은 이미지를 분류하는 합리적인 모델을 생성 할 수 있습니다.먼저 데이터를 로드하고 평균을 계산합니다.

```{.python .input}
# Load in the dataset
train = gluon.data.vision.FashionMNIST(train=True)
test = gluon.data.vision.FashionMNIST(train=False)

X_train_0 = np.stack([x[0] for x in train if x[1] == 0]).astype(float)
X_train_1 = np.stack([x[0] for x in train if x[1] == 1]).astype(float)
X_test = np.stack(
    [x[0] for x in test if x[1] == 0 or x[1] == 1]).astype(float)
y_test = np.stack(
    [x[1] for x in test if x[1] == 0 or x[1] == 1]).astype(float)

# Compute averages
ave_0 = np.mean(X_train_0, axis=0)
ave_1 = np.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab pytorch
# Load in the dataset
trans = []
trans.append(transforms.ToTensor())
trans = transforms.Compose(trans)
train = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                          train=True, download=True)
test = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                         train=False, download=True)

X_train_0 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 0]).type(torch.float32)
X_train_1 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 1]).type(torch.float32)
X_test = torch.stack(
    [x[0] * 256 for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
y_test = torch.stack([torch.tensor(x[1]) for x in test
                      if x[1] == 0 or x[1] == 1]).type(torch.float32)

# Compute averages
ave_0 = torch.mean(X_train_0, axis=0)
ave_1 = torch.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab tensorflow
# Load in the dataset
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()


X_train_0 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 0]] * 256), dtype=tf.float32)
X_train_1 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 1]] * 256), dtype=tf.float32)
X_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 0]] * 256), dtype=tf.float32)
y_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 1]] * 256), dtype=tf.float32)

# Compute averages
ave_0 = tf.reduce_mean(X_train_0, axis=0)
ave_1 = tf.reduce_mean(X_train_1, axis=0)
```

이러한 평균을 자세히 조사하는 것이 도움이 될 수 있으므로 어떻게 보이는지 플로팅해 보겠습니다.이 경우 평균이 티셔츠의 흐릿한 이미지와 실제로 비슷하다는 것을 알 수 있습니다.

```{.python .input}
#@tab mxnet, pytorch
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(tf.reshape(ave_0, (28, 28)), cmap='Greys')
d2l.plt.show()
```

두 번째 경우에는 평균이 바지의 흐릿한 이미지와 비슷하다는 것을 다시 알 수 있습니다.

```{.python .input}
#@tab mxnet, pytorch
# Plot average trousers
d2l.plt.imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot average trousers
d2l.plt.imshow(tf.reshape(ave_1, (28, 28)), cmap='Greys')
d2l.plt.show()
```

완전히 기계 학습 된 솔루션에서는 데이터 세트에서 임계 값을 학습합니다.이 경우 훈련 데이터에서 잘 보이는 임계 값을 손으로 눈으로 보았습니다.

```{.python .input}
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
predictions = X_test.reshape(2000, -1).dot(w.flatten()) > -1500000

# Accuracy
np.mean(predictions.astype(y_test.dtype) == y_test, dtype=np.float64)
```

```{.python .input}
#@tab pytorch
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
# '@' is Matrix Multiplication operator in pytorch.
predictions = X_test.reshape(2000, -1) @ (w.flatten()) > -1500000

# Accuracy
torch.mean((predictions.type(y_test.dtype) == y_test).float(), dtype=torch.float64)
```

```{.python .input}
#@tab tensorflow
# Print test set accuracy with eyeballed threshold
w = tf.transpose(ave_1 - ave_0)
predictions = tf.reduce_sum(X_test * tf.nest.flatten(w), axis=0) > -1500000

# Accuracy
tf.reduce_mean(
    tf.cast(tf.cast(predictions, y_test.dtype) == y_test, tf.float32))
```

## 선형 변환의 지오메트리

:numref:`sec_linear-algebra`와 위의 논의를 통해 벡터, 길이 및 각도의 기하학에 대한 확실한 이해를 얻었습니다.그러나 우리가 생략한 중요한 대상이 하나 있는데, 이는 행렬로 표현되는 선형 변환에 대한 기하학적 이해입니다.잠재적으로 다른 두 고차원 공간 사이에서 데이터를 변환하기 위해 행렬이 수행할 수 있는 작업을 완전히 내재화하는 것은 상당한 관행이 필요하며 이 부록의 범위를 벗어납니다.그러나 우리는 두 가지 차원에서 직관을 쌓을 수 있습니다. 

행렬이 있다고 가정해 보겠습니다. 

$$
\mathbf{A} = \begin{bmatrix}
a & b \\ c & d
\end{bmatrix}.
$$

만약 우리가 이것을 임의의 벡터 $\mathbf{v} = [x, y]^\top$에 적용하고자 한다면, 우리는 곱해서 그것을 볼 수 있습니다. 

$$
\begin{aligned}
\mathbf{A}\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \\
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\d\end{bmatrix} \\
& = x\left\{\mathbf{A}\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\mathbf{A}\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{aligned}
$$

이것은 이상한 계산처럼 보일 수 있는데, 명확한 것이 다소 뚫을 수 없게되었습니다.그러나 행렬이*두 개의 특정 벡터*를 변환하는 방법 ($[1,0]^\top$ 및 $[0,1]^\top$) 의 관점에서*모든* 벡터를 변환하는 방법을 쓸 수 있음을 알려줍니다.잠시 고려해 볼 가치가 있습니다.우리는 본질적으로 무한 문제 (실수 쌍에 발생하는 일) 를 유한 문제 (이러한 특정 벡터에 발생하는 일) 로 줄였습니다.이 벡터는*basis*의 예로, 공간의 모든 벡터를 이러한*기초 벡터*의 가중 합으로 쓸 수 있습니다. 

특정 행렬을 사용할 때 어떤 일이 발생하는지 그려 보겠습니다. 

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}.
$$

특정 벡터 $\mathbf{v} = [2, -1]^\top$을 보면 이것이 $2\cdot[1,0]^\top + -1\cdot[0,1]^\top$이라는 것을 알 수 있습니다. 따라서 행렬 $A$가 이것을 $2(\mathbf{A}[1,0]^\top) + -1(\mathbf{A}[0,1])^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$로 보낼 것임을 알 수 있습니다.이 논리를 신중하게 따르면, 예를 들어 모든 정수 점 쌍의 그리드를 고려하면 행렬 곱셈이 그리드를 왜곡, 회전 및 배율 조정할 수 있지만 그리드 구조는 :numref:`fig_grid-transform`에서 볼 수 있듯이 유지되어야 함을 알 수 있습니다. 

![The matrix $\mathbf{A}$ acting on the given basis vectors.  Notice how the entire grid is transported along with it.](../img/grid-transform.svg)
:label:`fig_grid-transform`

이것은 행렬로 표현되는 선형 변환에 대해 내재화하는 가장 중요한 직관적 포인트입니다.행렬은 공간의 일부를 다른 부분과 다르게 왜곡할 수 없습니다.그들이 할 수 있는 일은 공간의 원래 좌표를 가져와 기울이기, 회전 및 크기 조정하는 것뿐입니다. 

일부 왜곡이 심할 수 있습니다.예를 들어, 행렬은 

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix},
$$

전체 2차원 평면을 한 줄로 압축합니다.이러한 변환을 식별하고 작업하는 것은 이후 섹션의 주제이지만 기하학적으로 이것이 위에서 본 변환 유형과 근본적으로 다르다는 것을 알 수 있습니다.예를 들어, 행렬 $\mathbf{A}$의 결과는 원래 격자선에 “뒤로 구부릴” 수 있습니다.행렬 $\mathbf{B}$의 결과는 벡터 $[1,2]^\top$의 출처를 알 수 없기 때문에 할 수 없습니다. $[1,1]^\top$ 또는 $[0, -1]^\top$입니까? 

이 그림은 $2\times2$ 행렬에 대한 것이었지만 배운 교훈을 더 높은 차원으로 끌어 올리는 것을 방해하는 것은 없습니다.$[1,0, \ldots,0]$와 같은 유사한 기저 벡터를 취하여 행렬이 어디에 전송하는지 확인하면 행렬 곱셈이 우리가 다루는 차원 공간에서 전체 공간을 어떻게 왜곡하는지 느낄 수 있습니다. 

## 선형 의존성

행렬을 다시 생각해 봅시다. 

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}.
$$

이렇게 하면 전체 평면을 아래로 압축하여 단일 선 $y = 2x$에 살게 됩니다.이제 질문이 생깁니다. 행렬 자체를 보는 것만으로도 이것을 감지할 수 있는 방법이 있을까요?답은 실제로 할 수 있다는 것입니다.$\mathbf{b}_1 = [2,4]^\top$과 $\mathbf{b}_2 = [-1, -2]^\top$를 $\mathbf{B}$의 두 기둥으로 삼자.행렬 $\mathbf{B}$로 변환 된 모든 것을 행렬 열의 가중 합으로 쓸 수 있습니다 (예: $a_1\mathbf{b}_1 + a_2\mathbf{b}_2$).이것을*선형 조합*이라고 부릅니다.$\mathbf{b}_1 = -2\cdot\mathbf{b}_2$이라는 사실은 우리가 이 두 열의 선형 조합을 완전히 $\mathbf{b}_2$이라는 용어로 쓸 수 있다는 것을 의미합니다. 

$$
a_1\mathbf{b}_1 + a_2\mathbf{b}_2 = -2a_1\mathbf{b}_2 + a_2\mathbf{b}_2 = (a_2-2a_1)\mathbf{b}_2.
$$

즉, 어떤 의미에서 기둥 중 하나는 공간에서 고유한 방향을 정의하지 않기 때문에 중복됩니다.이 행렬이 전체 평면을 한 줄로 축소하는 것을 이미 보았기 때문에 너무 놀라지 않을 것입니다.또한 선형 의존성 $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$가 이것을 포착한다는 것을 알 수 있습니다.두 벡터 사이의 대칭성을 높이기 위해 다음과 같이 작성합니다. 

$$
\mathbf{b}_1  + 2\cdot\mathbf{b}_2 = 0.
$$

일반적으로 계수 $a_1, \ldots, a_k$*모두 0과 같지는 않음*이 존재하는 경우 벡터 $\mathbf{v}_1, \ldots, \mathbf{v}_k$의 집합이*선형 의존적*이라고 말할 것입니다. 

$$
\sum_{i=1}^k a_i\mathbf{v_i} = 0.
$$

이 경우 벡터 중 하나를 다른 벡터의 일부 조합으로 풀고 효과적으로 중복으로 렌더링 할 수 있습니다.따라서 행렬의 열에 대한 선형 의존성은 행렬이 공간을 더 낮은 차원으로 압축하고 있다는 사실을 입증합니다.선형 종속성이 없으면 벡터가*선형 독립적*이라고 말합니다.행렬의 열이 선형으로 독립적이면 압축이 발생하지 않으며 연산을 취소할 수 있습니다. 

## 순위

일반적인 $n\times m$ 행렬이 있는 경우 행렬이 매핑되는 차원 공간을 묻는 것이 합리적입니다.*순위*로 알려진 개념이 우리의 해답이 될 것입니다.이전 섹션에서 우리는 선형 의존성이 공간을 더 낮은 차원으로 압축하는 것을 목격하므로 이것을 사용하여 순위 개념을 정의 할 수 있다고 언급했습니다.특히 행렬 $\mathbf{A}$의 랭크는 모든 열 부분 집합 중에서 가장 많은 선형 독립 열 수입니다.예를 들어, 행렬은 

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix},
$$

에는 $\mathrm{rank}(B)=1$가 있습니다. 두 열은 선형으로 종속적이지만 두 열 자체는 선형으로 종속되지 않습니다.좀 더 어려운 예를 들자면 

$$
\mathbf{C} = \begin{bmatrix}
1& 3 & 0 & -1 & 0 \\
-1 & 0 & 1 & 1 & -1 \\
0 & 3 & 1 & 0 & -1 \\
2 & 3 & -1 & -2 & 1
\end{bmatrix},
$$

예를 들어 처음 두 열은 선형으로 독립적이지만 세 개의 열로 구성된 네 개의 컬렉션 중 하나가 종속적이기 때문에 $\mathbf{C}$의 순위가 2임을 보여줍니다. 

이 절차는 설명 된대로 매우 비효율적입니다.주어진 행렬의 열의 모든 부분 집합을 살펴봐야하므로 열 수가 지수 일 수 있습니다.나중에 행렬의 순위를 계산하는 더 효율적인 방법을 보게 될 것입니다. 하지만 지금은 개념이 잘 정의되어 있고 의미를 이해하기에 충분합니다. 

## 불변성

위에서 선형 종속 열을 가진 행렬에 의한 곱셈은 취소 할 수 없다는 것을 보았습니다. 즉, 항상 입력을 복구 할 수있는 역 연산이 없습니다.그러나 전체 순위 행렬 (즉, 순위가 $n$인 $n \times n$ 행렬 인 일부 $\mathbf{A}$) 을 곱하면 항상 실행 취소 할 수 있어야합니다.행렬을 고려하세요 

$$
\mathbf{I} = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}.
$$

이 행렬은 대각선을 따라 1이고 다른 곳에 0이 있는 행렬입니다.이를 *정체성* 행렬이라고 합니다.적용 시 데이터가 변경되지 않은 상태로 유지되는 행렬입니다.행렬 $\mathbf{A}$가 수행한 작업을 취소하는 행렬을 찾으려면 다음과 같은 행렬 $\mathbf{A}^{-1}$를 찾으려고 합니다. 

$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} =  \mathbf{I}.
$$

이것을 시스템으로 보면 $n \times n$ 미지수 ($\mathbf{A}^{-1}$의 항목) 와 $n \times n$ 방정식 (제품 $\mathbf{A}^{-1}\mathbf{A}$의 모든 항목과 $\mathbf{I}$의 모든 항목 사이에 유지해야하는 평등) 이 있으므로 일반적으로 해결책이 존재할 것으로 예상해야합니다.실제로 다음 섹션에서는 행렬식이 0이 아닌 한 해를 찾을 수있는 속성을 가진*determinant*라는 수량을 볼 수 있습니다.이러한 행렬 $\mathbf{A}^{-1}$을*역* 행렬이라고 부릅니다.예를 들어, $\mathbf{A}$가 일반적인 $2 \times 2$ 행렬인 경우 

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

그러면 우리는 그 역행렬이 

$$
 \frac{1}{ad-bc}  \begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}.
$$

위의 공식으로 주어진 역수를 곱하는 것이 실제로 작동한다는 것을 확인하여이를 테스트 할 수 있습니다.

```{.python .input}
M = np.array([[1, 2], [1, 4]])
M_inv = np.array([[2, -1], [-0.5, 0.5]])
M_inv.dot(M)
```

```{.python .input}
#@tab pytorch
M = torch.tensor([[1, 2], [1, 4]], dtype=torch.float32)
M_inv = torch.tensor([[2, -1], [-0.5, 0.5]])
M_inv @ M
```

```{.python .input}
#@tab tensorflow
M = tf.constant([[1, 2], [1, 4]], dtype=tf.float32)
M_inv = tf.constant([[2, -1], [-0.5, 0.5]])
tf.matmul(M_inv, M)
```

### 수치 문제 행렬의 역행렬은 이론적으로 유용하지만, 대부분의 경우 실제로 문제를 풀기 위해 역행렬을*사용* 하고 싶지 않다고 말해야 합니다.일반적으로 다음과 같은 선형 방정식을 푸는 데 훨씬 더 수치적으로 안정적인 알고리즘이 있습니다. 

$$
\mathbf{A}\mathbf{x} = \mathbf{b},
$$

역수를 계산하고 곱하여 얻는 것보다 

$$
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}.
$$

작은 숫자로 나누면 수치 불안정성이 발생할 수있는 것처럼 순위가 낮은 행렬의 반전도 발생할 수 있습니다. 

또한 행렬 $\mathbf{A}$은*희소*인 것이 일반적이며, 이는 0이 아닌 소수의 값만 포함한다는 의미입니다.예를 살펴보면, 이것이 역이 희박하다는 것을 의미하지는 않는다는 것을 알 수 있습니다.$\mathbf{A}$이 7322936,150만 x 73229361,500만 개의 행렬인 경우에도 $5$만 개의 0이 아닌 항목이 있는 경우 (따라서 해당 $5$만 개의 항목만 저장하면 됨), 역행렬은 일반적으로 음수가 아닌 거의 모든 항목을 가지므로 모든 $1\text{M}^2$개의 항목 (즉, $1$조 달러) 을 저장해야 합니다.응모! 

선형 대수로 작업 할 때 자주 접하게되는 어려운 수치 문제에 대해 자세히 알아볼 시간이 없지만 신중하게 진행해야하는시기에 대한 직관을 제공하고자합니다. 일반적으로 실제로 반전을 피하는 것이 좋은 경험 법칙입니다. 

## 결정자 선형 대수의 기하학적 관점은*행렬식*으로 알려진 기본 양을 직관적으로 해석하는 방법을 제공합니다.이전의 그리드 이미지를 고려하지만 지금은 강조 표시된 영역 (:numref:`fig_grid-filled`) 이 있습니다. 

![The matrix $\mathbf{A}$ again distorting the grid.  This time, I want to draw particular attention to what happens to the highlighted square.](../img/grid-transform-filled.svg)
:label:`fig_grid-filled`

강조 표시된 사각형을 보세요.이것은 가장자리가 $(0, 1)$ 및 $(1, 0)$로 주어진 정사각형이므로 면적이 하나 있습니다.$\mathbf{A}$이 이 사각형을 변환 한 후 평행 사변형이되는 것을 알 수 있습니다.이 평행 사변형이 우리가 시작한 것과 동일한 영역을 가져야 할 이유가 없습니다. 실제로 여기에 표시된 특정 사례에서 

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix},
$$

이 평행 사변형의 면적을 계산하고 면적이 $5$인지 확인하는 것은 좌표 기하학의 연습입니다. 

일반적으로 행렬이 있으면 

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

일부 계산을 통해 결과 평행 사변형의 면적이 $ad-bc$라는 것을 알 수 있습니다.이 영역을*결정 요인*이라고 합니다. 

몇 가지 예제 코드를 사용하여 빠르게 확인해 보겠습니다.

```{.python .input}
import numpy as np
np.linalg.det(np.array([[1, -1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
torch.det(torch.tensor([[1, -1], [2, 3]], dtype=torch.float32))
```

```{.python .input}
#@tab tensorflow
tf.linalg.det(tf.constant([[1, -1], [2, 3]], dtype=tf.float32))
```

우리 가운데 독수리 눈을 가진 사람들은이 표현이 0이거나 심지어 음수가 될 수 있음을 알게 될 것입니다.음의 용어의 경우 이것은 일반적으로 수학에서 취해지는 관례의 문제입니다. 행렬이 그림을 뒤집으면 면적이 부정된다고 말합니다.이제 행렬식이 0일 때 더 많은 것을 배울 수 있습니다. 

고려해 보겠습니다. 

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix}.
$$

이 행렬의 행렬식을 계산하면 $2\cdot(-2 ) - 4\cdot(-1) = 0$을 얻습니다.위의 이해를 감안할 때 이것은 의미가 있습니다. $\mathbf{B}$는 원본 이미지의 사각형을 면적이 0 인 선 세그먼트로 압축합니다.그리고 실제로, 더 낮은 차원의 공간으로 압축되는 것이 변환 후 0 영역을 갖는 유일한 방법입니다.따라서 다음 결과가 true임을 알 수 있습니다. 행렬 $A$는 행렬식이 0이 아닌 경우에만 반전할 수 있습니다. 

마지막으로 비행기에 그림이 그려져 있다고 상상해보십시오.컴퓨터 과학자처럼 생각하면, 우리는 그 그림을 작은 사각형의 집합으로 분해할 수 있습니다. 그래서 그림의 면적은 본질적으로 분해의 제곱 수에 불과합니다.이제 그 그림을 행렬로 변환하면 이러한 각 사각형을 평행 사변형으로 보냅니다. 각 사각형은 행렬식에 의해 주어진 면적을 갖습니다.모든 그림에 대해 행렬식은 행렬이 그림의 면적을 스케일링하는 (부호있는) 숫자를 제공한다는 것을 알 수 있습니다. 

행렬이 큰 행렬에 대한 행렬식을 계산하는 것은 힘들 수 있지만 직관은 동일합니다.행렬식은 $n\times n$ 행렬이 $n$차원 볼륨의 크기를 조정하는 요인으로 남아 있습니다. 

## 텐서 및 일반 선형 대수 연산

:numref:`sec_linear-algebra`에서 텐서의 개념이 도입되었습니다.이 섹션에서는 텐서 수축 (행렬 곱셈과 동등한 텐서) 에 대해 자세히 알아보고 여러 행렬 및 벡터 연산에 대한 통합 뷰를 제공하는 방법을 살펴 보겠습니다. 

행렬과 벡터를 사용하여 데이터를 변환하기 위해 행렬과 벡터를 곱하는 방법을 알았습니다.텐서가 우리에게 유용하려면 텐서에 대해 비슷한 정의가 필요합니다.행렬 곱셈에 대해 생각해 보십시오. 

$$
\mathbf{C} = \mathbf{A}\mathbf{B},
$$

또는 동등하게 

$$ c_{i, j} = \sum_{k} a_{i, k}b_{k, j}.$$

이 패턴은 텐서에 대해 반복 할 수 있습니다.텐서의 경우 보편적으로 선택할 수있는 합산 사례가 하나도 없으므로 합산하려는 인덱스를 정확히 지정해야합니다.예를 들어 고려할 수 있습니다. 

$$
y_{il} = \sum_{jk} x_{ijkl}a_{jk}.
$$

이러한 변환을*텐서 수축*이라고합니다.행렬 곱셈만으로도 훨씬 더 유연한 변환 제품군을 나타낼 수 있습니다. 

자주 사용되는 표기법 단순화로서 합계가 표현식에서 두 번 이상 발생하는 인덱스 위에 있다는 것을 알 수 있습니다. 따라서 사람들은 종종*Einstein 표기법*을 사용하여 작업하는 경우가 많습니다.이렇게 하면 간결한 표현식이 제공됩니다. 

$$
y_{il} = x_{ijkl}a_{jk}.
$$

### 선형 대수의 일반적인 예

이전에 본 선형 대수 정의 중 몇 개를이 압축 된 텐서 표기법으로 표현할 수 있는지 살펴 보겠습니다. 

* $\mathbf{v} \cdot \mathbf{w} = \sum_i v_iw_i$
* $\|\mathbf{v}\|_2^{2} = \sum_i v_iv_i$
* $(\mathbf{A}\mathbf{v})_i = \sum_j a_{ij}v_j$
* $(\mathbf{A}\mathbf{B})_{ik} = \sum_j a_{ij}b_{jk}$
* $\mathrm{tr}(\mathbf{A}) = \sum_i a_{ii}$

이런 식으로 무수히 많은 특수 표기법을 짧은 텐서 표현식으로 대체 할 수 있습니다. 

### 코드 텐서의 표현은 

코드에서도 유연하게 작동할 수 있습니다.:numref:`sec_linear-algebra`에서 볼 수 있듯이 아래와 같이 텐서를 만들 수 있습니다.

```{.python .input}
# Define tensors
B = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab pytorch
# Define tensors
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab tensorflow
# Define tensors
B = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = tf.constant([[1, 2], [3, 4]])
v = tf.constant([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

아인슈타인 합계가 직접 구현되었습니다.아인슈타인 합계에서 발생하는 인덱스는 문자열로 전달 된 다음 동작중인 텐서가 올 수 있습니다.예를 들어, 행렬 곱셈을 구현하기 위해 위에서 본 아인슈타인 합계 ($\mathbf{A}\mathbf{v} = a_{ij}v_j$) 를 고려하고 인덱스 자체를 제거하여 구현을 얻을 수 있습니다.

```{.python .input}
# Reimplement matrix multiplication
np.einsum("ij, j -> i", A, v), A.dot(v)
```

```{.python .input}
#@tab pytorch
# Reimplement matrix multiplication
torch.einsum("ij, j -> i", A, v), A@v
```

```{.python .input}
#@tab tensorflow
# Reimplement matrix multiplication
tf.einsum("ij, j -> i", A, v), tf.matmul(A, tf.reshape(v, (2, 1)))
```

이는 매우 유연한 표기법입니다.예를 들어, 전통적으로 쓰여진 것을 계산하려는 경우 

$$
c_{kl} = \sum_{ij} \mathbf{b}_{ijk}\mathbf{a}_{il}v_j.
$$

아인슈타인 합계를 통해 다음과 같이 구현할 수 있습니다.

```{.python .input}
np.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab pytorch
torch.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab tensorflow
tf.einsum("ijk, il, j -> kl", B, A, v)
```

이 표기법은 인간에게 읽기 쉽고 효율적이지만 어떤 이유로 든 프로그래밍 방식으로 텐서 축소를 생성해야하는 경우 부피가 큽니다.이러한 이유로 `einsum`는 각 텐서에 정수 인덱스를 제공하여 대체 표기법을 제공합니다.예를 들어 동일한 텐서 수축은 다음과 같이 쓸 수도 있습니다.

```{.python .input}
np.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

```{.python .input}
#@tab pytorch
# PyTorch doesn't support this type of notation.
```

```{.python .input}
#@tab tensorflow
# TensorFlow doesn't support this type of notation.
```

어느 표기법이든 코드에서 텐서 수축을 간결하고 효율적으로 표현할 수 있습니다. 

## 요약

* 벡터는 기하학적으로 공간에서 점 또는 방향으로 해석될 수 있습니다.* 내적은 임의의 고차원 공간에 대한 각도의 개념을 정의합니다.
* 초평면은 선과 평면의 고차원 일반화입니다.분류 작업의 마지막 단계로 자주 사용되는 결정 평면을 정의하는 데 사용할 수 있습니다.
* 행렬 곱셈은 기하학적으로 기본 좌표의 균일한 왜곡으로 해석될 수 있습니다.벡터 변환은 매우 제한적이지만 수학적으로 깨끗한 벡터 변환 방법을 나타냅니다.
* 선형 종속성은 벡터 집합이 예상보다 낮은 차원 공간에 있을 때를 알 수 있는 방법입니다 (예: $2$차원 공간에 $3$개의 벡터가 있음).행렬의 랭크는 선형 독립된 열 중 가장 큰 부분 집합의 크기입니다. 
* 행렬의 역행렬이 정의되면 행렬 반전을 통해 첫 번째 행렬의 동작을 취소하는 다른 행렬을 찾을 수 있습니다.행렬 반전은 이론적으로 유용하지만 수치적 불안정성으로 인해 실제로 주의가 필요합니다.
* 행렬식을 사용하면 행렬이 공간을 얼마나 확장하거나 축소하는지 측정할 수 있습니다.0이 아닌 행렬식은 반전 가능한 (비특이적) 행렬을 의미하고, 값이 0인 행렬식은 행렬이 반전할 수 없는 (단수) 임을 의미합니다.
* 텐서 수축과 아인슈타인 합계는 머신러닝에서 볼 수 있는 많은 계산을 표현하기 위한 깔끔하고 깔끔한 표기법을 제공합니다. 

## 연습 문제 

1. What is the angle between
$$
\vec v_1 = \begin{bmatrix}
1 \\ 0 \\ -1 \\ 2
\end{bmatrix}, \qquad \vec v_2 = \begin{bmatrix}
3 \\ 1 \\ 0 \\ 1
\end{bmatrix}?
$$
2. True or false: $\begin{bmatrix}1 & 2\\0&1\end{bmatrix}$ and $\begin{bmatrix}1 & -2\\0&1\end{bmatrix}$ are inverses of one another?
3. Suppose that we draw a shape in the plane with area $100\mathrm{m}^2$.  What is the area after transforming the figure by the matrix
$$
\begin{bmatrix}
2 & 3\\
1 & 2
\end{bmatrix}.
$$
4. Which of the following sets of vectors are linearly independent?
 * $\left\{\begin{pmatrix}1\\0\\-1\end{pmatrix}, \begin{pmatrix}2\\1\\-1\end{pmatrix}, \begin{pmatrix}3\\1\\1\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}3\\1\\1\end{pmatrix}, \begin{pmatrix}1\\1\\1\end{pmatrix}, \begin{pmatrix}0\\0\\0\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}1\\1\\0\end{pmatrix}, \begin{pmatrix}0\\1\\-1\end{pmatrix}, \begin{pmatrix}1\\0\\1\end{pmatrix}\right\}$
5. Suppose that you have a matrix written as $A = \begin{bmatrix}c\\d\end{bmatrix}\cdot\begin{bmatrix}a & b\end{bmatrix}$ for some choice of values $a, b, c$, and $d$.  True or false: the determinant of such a matrix is always $0$?
6. The vectors $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$ and $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$ are orthogonal.  What is the condition on a matrix $A$ so that $Ae_1$ and $Ae_2$ are orthogonal?
7. How can you write $\mathrm{tr}(\mathbf{A}^4)$ in Einstein notation for an arbitrary matrix $A$?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/410)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1084)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1085)
:end_tab:
