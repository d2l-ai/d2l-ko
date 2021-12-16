# 선형 대수
:label:`sec_linear-algebra`

이제 데이터를 저장하고 조작할 수 있으므로 이 책에서 다루는 대부분의 모델을 이해하고 구현하는 데 필요한 기본 선형 대수의 하위 집합을 간략하게 살펴보겠습니다.아래에서는 선형 대수의 기본 수학 객체, 산술 및 연산을 소개하고 수학적 표기법과 해당 코드 구현을 통해 각각을 표현합니다.

## 스칼라

선형 대수학이나 기계 학습을 공부한 적이 없다면 과거 수학 경험은 아마도 한 번에 하나의 숫자를 생각하는 것으로 구성되었을 것입니다.그리고 수표 책의 균형을 맞추거나 식당에서 저녁 식사를 지불 한 적이 있다면 숫자 쌍을 더하고 곱하는 것과 같은 기본적인 작업을 수행하는 방법을 이미 알고 있습니다.예를 들어 팔로 알토의 온도는 화씨 $52$도입니다.공식적으로, 우리는 단지 하나의 숫자*스칼라*로 구성된 값을 호출합니다.이 값을 섭씨 (미터법 시스템의 보다 합리적인 온도 눈금) 로 변환하려면 $f$을 $52$으로 설정하여 $c = \frac{5}{9}(f - 32)$ 표현식을 평가해야 합니다.이 방정식에서 각 항 ($5$, $9$ 및 $32$) 은 스칼라 값입니다.자리 표시자 $c$ 및 $f$은*변수*라고 하며 알 수 없는 스칼라 값을 나타냅니다.

이 책에서는 스칼라 변수가 일반 소문자로 표시되는 수학적 표기법을 채택합니다 (예: $x$, $y$ 및 $z$).우리는 모든 (연속) *실수 값* 스칼라의 공간을 $\mathbb{R}$으로 나타냅니다.편의를 위해 정확히*공간*이 무엇인지에 대한 엄격한 정의를 펀트 할 것입니다. 하지만 지금은 $x \in \mathbb{R}$이라는 표현식이 $x$이 실수 값 스칼라라고 말하는 공식적인 방법이라는 것을 기억하십시오.기호 $\in$는 “in”으로 발음 할 수 있으며 단순히 집합의 구성원을 나타냅니다.마찬가지로 $x, y \in \{0, 1\}$을 작성하여 $x$과 $y$가 값이 $0$ 또는 $1$일 수 있는 숫자임을 나타낼 수 있습니다.

(**스칼라는 요소가 하나뿐인 텐서로 표시됩니다.**) 다음 스 니펫에서는 두 개의 스칼라를 인스턴스화하고 이들을 사용하여 친숙한 산술 연산, 즉 더하기, 곱셈, 나누기 및 지수를 수행합니다.

```{.python .input}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
#@tab pytorch
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

## 벡터

[**벡터는 단순히 스칼라 값의 목록이라고 생각할 수 있습니다.**] 이 값을 벡터의*요소* (*항목* 또는*구성 요소*) 라고 합니다.벡터가 데이터셋의 예제를 나타낼 때, 벡터의 값에는 실제 의미가 있습니다.예를 들어, 대출 채무 불이행의 위험을 예측하기 위해 모델을 훈련하는 경우 각 신청자를 소득, 고용 기간, 이전 채무 불이행 수 및 기타 요인에 해당하는 구성 요소의 벡터와 연관시킬 수 있습니다.병원 환자가 직면 할 수있는 심장 마비의 위험을 연구하는 경우 구성 요소가 가장 최근의 활력 징후, 콜레스테롤 수치, 하루 운동 시간 등을 포착하는 벡터로 각 환자를 나타낼 수 있습니다. 수학 표기법에서는 일반적으로 벡터를 굵고 소문자로 표시합니다.편지 (예: $\mathbf{x}$, $\mathbf{y}$ 및 $\mathbf{z})$.

우리는 일차원 텐서를 통해 벡터로 작업합니다.일반적으로 텐서는 컴퓨터의 메모리 제한에 따라 임의의 길이를 가질 수 있습니다.

```{.python .input}
x = np.arange(4)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(4)
x
```

아래 첨자를 사용하여 벡터의 모든 요소를 참조 할 수 있습니다.예를 들어 $x_i$에서 $\mathbf{x}$의 $i^\mathrm{th}$ 요소를 참조할 수 있습니다.요소 $x_i$는 스칼라이므로 글꼴을 참조할 때 글꼴을 굵게 표시하지 않습니다.광범위한 문헌에서는 열 벡터를 벡터의 기본 방향으로 간주하므로 이 책도 마찬가지입니다.수학에서 벡터 $\mathbf{x}$는 다음과 같이 쓸 수 있습니다.

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

여기서 $x_1, \ldots, x_n$는 벡터의 요소입니다.코드에서 우리는 (**텐서에 인덱싱하여 모든 요소에 액세스합니다.**)

```{.python .input}
x[3]
```

```{.python .input}
#@tab pytorch
x[3]
```

```{.python .input}
#@tab tensorflow
x[3]
```

### 길이, 치수 및 모양

:numref:`sec_ndarray`의 몇 가지 개념을 다시 살펴보겠습니다.벡터는 숫자의 배열일 뿐입니다.모든 배열에 길이가 있는 것처럼 모든 벡터도 마찬가지입니다.수학 표기법에서 벡터 $\mathbf{x}$이 $n$개의 실수 값 스칼라로 구성되어 있다고 말하고 싶다면 이를 $\mathbf{x} \in \mathbb{R}^n$로 표현할 수 있습니다.벡터의 길이는 일반적으로 벡터의*차원*이라고 합니다.

일반적인 파이썬 배열과 마찬가지로 파이썬의 내장 `len()` 함수를 호출하여 [**텐서의 길이에 액세스 할 수 있습니다**].

```{.python .input}
len(x)
```

```{.python .input}
#@tab pytorch
len(x)
```

```{.python .input}
#@tab tensorflow
len(x)
```

텐서가 정확하게 하나의 축을 가진 벡터를 나타내는 경우 `.shape` 속성을 통해 길이에 액세스 할 수도 있습니다.모양은 텐서의 각 축을 따라 길이 (차원) 를 나열하는 튜플입니다.(**축이 하나뿐인 텐서의 경우 모양에는 요소가 하나뿐입니다.**)

```{.python .input}
x.shape
```

```{.python .input}
#@tab pytorch
x.shape
```

```{.python .input}
#@tab tensorflow
x.shape
```

“차원”이라는 단어는 이러한 맥락에서 과부하가 걸리는 경향이 있으며 이는 사람들을 혼란스럽게하는 경향이 있습니다.명확히하기 위해*벡터* 또는*축*의 차원 성을 사용하여 길이, 즉 벡터 또는 축의 요소 수를 나타냅니다.그러나 텐서의 차원 성을 사용하여 텐서가 가진 축의 수를 나타냅니다.이런 의미에서 텐서의 일부 축의 차원은 해당 축의 길이가됩니다.

## 행렬

벡터가 0차수에서 1차까지 스칼라를 일반화하는 것처럼 행렬은 벡터를 1차부터 차수 2차까지 일반화합니다.일반적으로 굵은 대문자로 표시되는 행렬 (예: $\mathbf{X}$, $\mathbf{Y}$ 및 $\mathbf{Z}$) 은 코드에서 두 개의 축을 가진 텐서로 표시됩니다.

수학 표기법에서는 $\mathbf{A} \in \mathbb{R}^{m \times n}$을 사용하여 행렬 $\mathbf{A}$이 $m$개의 행과 $n$의 실수 값 스칼라 열로 구성되어 있음을 표현합니다.시각적으로 모든 행렬 $\mathbf{A} \in \mathbb{R}^{m \times n}$을 테이블로 설명할 수 있습니다. 여기서 각 요소 $a_{ij}$는 $i^{\mathrm{th}}$ 행과 $j^{\mathrm{th}}$ 열에 속합니다.

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`

모든 $\mathbf{A} \in \mathbb{R}^{m \times n}$의 경우, $\mathbf{A}$의 모양은 ($m$, $n$) 또는 $m \times n$입니다.특히 행렬의 행과 열 수가 같으면 모양이 정사각형이 되므로*정사각형 행렬*이라고 합니다.

텐서를 인스턴스화하기 위해 가장 좋아하는 함수를 호출 할 때 두 개의 구성 요소 $m$ 및 $n$로 모양을 지정하여 [**$m \times n$ 행렬**] 을 만들 수 있습니다.

```{.python .input}
A = np.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab pytorch
A = torch.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20), (5, 4))
A
```

$[\mathbf{A}]_{ij}$과 같은 행 ($i$) 과 열 ($j$) 에 대한 인덱스를 지정하여 :eqref:`eq_matrix_def`에서 행렬 $\mathbf{A}$의 스칼라 요소 $a_{ij}$에 액세스할 수 있습니다.:eqref:`eq_matrix_def`와 같이 행렬 $\mathbf{A}$의 스칼라 요소가 제공되지 않으면 행렬 $\mathbf{A}$의 소문자를 색인 첨자 $a_{ij}$과 함께 사용하여 $[\mathbf{A}]_{ij}$을 참조할 수 있습니다.표기법을 단순하게 유지하기 위해 $a_{2, 3j}$ 및 $[\mathbf{A}]_{2i-1, 3}$와 같이 필요한 경우에만 쉼표를 별도의 인덱스에 삽입합니다.

때로는 축을 뒤집고 싶을 때가 있습니다.행렬의 행과 열을 교환할 때 결과를 행렬의*transpose*라고 합니다.공식적으로, 우리는 행렬 $\mathbf{A}$의 전치를 $\mathbf{A}^\top$로 나타내고, $\mathbf{B} = \mathbf{A}^\top$인 경우 $i$ 및 $j$에 대해 $b_{ij} = a_{ji}$을 나타냅니다.따라서 :eqref:`eq_matrix_def`에서 $\mathbf{A}$의 전치는 $n \times m$ 행렬입니다.

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

이제 코드에서 (**행렬의 전치**) 에 액세스합니다.

```{.python .input}
A.T
```

```{.python .input}
#@tab pytorch
A.T
```

```{.python .input}
#@tab tensorflow
tf.transpose(A)
```

정방 행렬의 특수한 유형으로서 [**a*대칭 행렬* $\mathbf{A}$은 전치와 같습니다: $\mathbf{A} = \mathbf{A}^\top$.**] 여기서 대칭 행렬 `B`를 정의합니다.

```{.python .input}
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab pytorch
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab tensorflow
B = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

이제 `B`를 전치와 비교합니다.

```{.python .input}
B == B.T
```

```{.python .input}
#@tab pytorch
B == B.T
```

```{.python .input}
#@tab tensorflow
B == tf.transpose(B)
```

행렬은 유용한 데이터 구조입니다. 행렬을 사용하면 변동 양식이 다른 데이터를 구성 할 수 있습니다.예를 들어 행렬의 행은 다른 집 (데이터 예) 에 해당할 수 있지만 열은 다른 속성에 해당할 수 있습니다.스프레드시트 소프트웨어를 사용해 본 적이 있거나 :numref:`sec_pandas`를 읽은 적이 있다면 친숙하게 들릴 것입니다.따라서 단일 벡터의 기본 방향이 열 벡터이지만 테이블 형식 데이터 세트를 나타내는 행렬에서는 각 데이터 예제를 행렬의 행 벡터로 처리하는 것이 더 일반적입니다.그리고 이후 장에서 살펴보겠지만, 이 컨벤션은 일반적인 딥 러닝 관행을 가능하게 할 것입니다.예를 들어 텐서의 가장 바깥 쪽 축을 따라 데이터 예제의 미니 일괄 처리에 액세스하거나 열거 할 수 있으며 미니 배치가없는 경우 데이터 예제에만 액세스 할 수 있습니다.

## 텐서

벡터가 스칼라를 일반화하고 행렬이 벡터를 일반화하는 것처럼 훨씬 더 많은 축을 가진 데이터 구조를 만들 수 있습니다.[**텐서**](이 하위 섹션의 “텐서”는 대수 객체를 나타냅니다) (**임의의 수의 축을 가진 $n$ 차원 배열을 설명하는 일반적인 방법을 제공합니다.**) 예를 들어 벡터는 1 차 텐서이고 행렬은 2 차 텐서입니다.텐서는 특수 글꼴 (예: $\mathsf{X}$, $\mathsf{Y}$ 및 $\mathsf{Z}$) 의 대문자로 표시되며 인덱싱 메커니즘 (예: $x_{ijk}$ 및 $[\mathsf{X}]_{1, 2i-1, 3}$) 은 행렬과 유사합니다.

텐서는 이미지 작업을 시작할 때 더욱 중요해질 것입니다. 이 배열은 높이, 너비에 해당하는 3 개의 축과 색상 채널 (빨간색, 녹색 및 파란색) 을 쌓기위한* 채널* 축이있는 $n$ 차원 배열로 도착합니다.지금은 고차 텐서를 건너 뛰고 기본에 중점을 둘 것입니다.

```{.python .input}
X = np.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab pytorch
X = torch.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(24), (2, 3, 4))
X
```

## 텐서 연산의 기본 속성

임의의 수의 축의 스칼라, 벡터, 행렬 및 텐서 (이 하위 섹션의 “텐서”는 대수 객체를 나타냄) 에는 종종 유용한 몇 가지 멋진 속성이 있습니다.예를 들어 요소별 연산의 정의에서 요소별 단항 연산은 피연산자의 모양을 변경하지 않는다는 것을 알 수 있습니다.마찬가지로 [**동일한 모양을 가진 두 개의 텐서가 주어지면 이진 요소별 연산의 결과는 동일한 모양의 텐서가됩니다.**] 예를 들어, 동일한 모양의 행렬 두 개를 더하면이 두 행렬에 요소 별 덧셈이 수행됩니다.

```{.python .input}
A = np.arange(20).reshape(5, 4)
B = A.copy()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab pytorch
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  # No cloning of `A` to `B` by allocating new memory
A, A + B
```

구체적으로, [**두 행렬의 요소별 곱셈을 *하다마르 곱***](수학 표기법 $\odot$) 라고 합니다.행 $i$과 열 $j$의 요소가 $b_{ij}$인 행렬 $\mathbf{B} \in \mathbb{R}^{m \times n}$를 가정해 보겠습니다.행렬 $\mathbf{A}$ (:eqref:`eq_matrix_def`에 정의됨) 및 $\mathbf{B}$의 하다마르드 곱입니다.

$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
A * B
```

```{.python .input}
#@tab pytorch
A * B
```

```{.python .input}
#@tab tensorflow
A * B
```

[**텐서에 스칼라를 곱하거나 더하기**] 는 피연산자 텐서의 각 요소에 스칼라가 더하거나 곱해지는 텐서의 모양도 변경되지 않습니다.

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

## 감축
:label:`subseq_lin-alg-reduction`

임의의 텐서로 수행 할 수있는 유용한 연산 중 하나는 [**요소의 합계**] 를 계산하는 것입니다. 수학 표기법에서는 $\sum$ 기호를 사용하여 합계를 표현합니다.길이가 $d$인 벡터 $\mathbf{x}$에 요소의 합을 표현하기 위해 $\sum_{i=1}^d x_i$을 작성합니다.코드에서는 합계를 계산하는 함수를 호출 할 수 있습니다.

```{.python .input}
x = np.arange(4)
x, x.sum()
```

```{.python .input}
#@tab pytorch
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
#@tab tensorflow
x = tf.range(4, dtype=tf.float32)
x, tf.reduce_sum(x)
```

[**임의의 모양의 텐서 요소에 대한 합계**] 를 표현할 수 있습니다. 예를 들어, $m \times n$ 행렬 $\mathbf{A}$의 요소 합은 $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$로 작성할 수 있습니다.

```{.python .input}
A.shape, A.sum()
```

```{.python .input}
#@tab pytorch
A.shape, A.sum()
```

```{.python .input}
#@tab tensorflow
A.shape, tf.reduce_sum(A)
```

기본적으로 합계를 계산하는 함수를 호출합니다.
*는 모든 축을 따라 텐서를 스칼라로 줄입니다*.
또한 [**summation을 통해 텐서가 축소되는 축을 지정할 수 있습니다.**] 행렬을 예로 들어 보겠습니다.모든 행의 요소를 합산하여 행 차원 (축 0) 을 줄이려면 함수를 호출 할 때 `axis=0`를 지정합니다.입력 행렬이 축 0을 따라 축소되어 출력 벡터가 생성되므로 입력값의 축 0의 차원은 출력 형태에서 손실됩니다.

```{.python .input}
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis0 = tf.reduce_sum(A, axis=0)
A_sum_axis0, A_sum_axis0.shape
```

`axis=1`를 지정하면 모든 열의 요소를 합산하여 열 차원 (축 1) 이 줄어듭니다.따라서 입력의 축 1의 치수가 출력 형상에서 손실됩니다.

```{.python .input}
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis1 = tf.reduce_sum(A, axis=1)
A_sum_axis1, A_sum_axis1.shape
```

합을 통해 행과 열을 따라 행렬을 줄이는 것은 행렬의 모든 요소를 합산하는 것과 같습니다.

```{.python .input}
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab pytorch
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(A, axis=[0, 1])  # Same as `tf.reduce_sum(A)`
```

[**관련 수량은*평균*이며, 이는*평균*이라고도합니다.**] 합계를 총 요소 수로 나누어 평균을 계산합니다.코드에서는 임의의 모양의 텐서에서 평균을 계산하는 함수를 호출 할 수 있습니다.

```{.python .input}
A.mean(), A.sum() / A.size
```

```{.python .input}
#@tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

마찬가지로 평균을 계산하는 함수는 지정된 축을 따라 텐서를 줄일 수도 있습니다.

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab pytorch
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

### 비감면 합계
:label:`subseq_lin-alg-non-reduction`

그러나 합이나 평균을 계산하는 함수를 호출할 때 [**좌표축 개수를 변경하지 않음**] 하는 것이 유용할 수 있습니다.

```{.python .input}
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab pytorch
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A
```

예를 들어, `sum_A`는 각 행을 합산한 후에도 여전히 두 축을 유지하기 때문에 브로드캐스트를 사용하여 `A`를 `sum_A`로 나눌 수 있습니다.

```{.python .input}
A / sum_A
```

```{.python .input}
#@tab pytorch
A / sum_A
```

```{.python .input}
#@tab tensorflow
A / sum_A
```

[**어떤 축을 따라 `A`의 요소의 누적합**], 예를 들어 `axis=0` (행별) 를 계산하려면 `cumsum` 함수를 호출할 수 있습니다.이 함수는 축을 따라 입력 텐서를 줄이지 않습니다.

```{.python .input}
A.cumsum(axis=0)
```

```{.python .input}
#@tab pytorch
A.cumsum(axis=0)
```

```{.python .input}
#@tab tensorflow
tf.cumsum(A, axis=0)
```

## 도트 제품

지금까지 요소별 연산, 합계 및 평균만 수행했습니다.그리고 이것이 우리가 할 수 있는 전부라면, 선형 대수학은 아마도 자체 섹션을 가질 자격이 없을 것입니다.그러나 가장 기본적인 연산 중 하나는 내적입니다.두 개의 벡터 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$가 주어지면 내적* $\mathbf{x}^\top \mathbf{y}$ (또는 $\langle \mathbf{x}, \mathbf{y}  \rangle$) 은 동일한 위치 $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$에 있는 요소의 곱에 대한 합계입니다.

[~~두 벡터의 내적*내적*은 같은 위치에 있는 요소의 곱에 대한 합입니다 ~~]

```{.python .input}
y = np.ones(4)
x, y, np.dot(x, y)
```

```{.python .input}
#@tab pytorch
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
#@tab tensorflow
y = tf.ones(4, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

(**요소별 곱셈을 수행한 다음 sum: 을 수행하여 두 벡터의 내적을 동등하게 표현할 수 있습니다.**)

```{.python .input}
np.sum(x * y)
```

```{.python .input}
#@tab pytorch
torch.sum(x * y)
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x * y)
```

내적은 다양한 상황에서 유용합니다.예를 들어, 벡터 $\mathbf{x}  \in \mathbb{R}^d$로 표시된 일부 값 세트와 $\mathbf{w} \in \mathbb{R}^d$로 표시된 가중치 집합이 주어지면 가중치 $\mathbf{w}$에 따른 $\mathbf{x}$에 있는 값의 가중 합은 내적 $\mathbf{x}^\top \mathbf{w}$로 표현될 수 있습니다.가중치가 음수가 아니고 합이 1 (즉, $\left(\sum_{i=1}^{d} {w_i} = 1\right)$) 인 경우 내적은*가중 평균*을 나타냅니다.단위 길이를 갖도록 두 벡터를 정규화한 후 내적은 두 벡터 사이의 각도의 코사인을 표현합니다.이 섹션 후반부에서*길이*라는 개념을 공식적으로 소개하겠습니다.

## 매트릭스-벡터 제품

내적을 계산하는 방법을 알았으므로*행렬-벡터 곱*을 이해할 수 있습니다.각각 :eqref:`eq_matrix_def` 및 :eqref:`eq_vec_def`에서 정의되고 시각화된 행렬 $\mathbf{A} \in \mathbb{R}^{m \times n}$과 벡터 $\mathbf{x} \in \mathbb{R}^n$을 회상합니다.먼저 행렬 $\mathbf{A}$을 행 벡터로 시각화해 보겠습니다.

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

여기서 각 $\mathbf{a}^\top_{i} \in \mathbb{R}^n$는 행렬 $\mathbf{A}$의 $i^\mathrm{th}$ 행을 나타내는 행 벡터입니다.

[**행렬-벡터 곱 $\mathbf{A}\mathbf{x}$은 단순히 길이가 $m$인 열 벡터이며, 이 벡터의 $i^\mathrm{th}$ 요소는 내적 $\mathbf{a}^\top_i \mathbf{x}$입니다.**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

행렬 $\mathbf{A}\in \mathbb{R}^{m \times n}$에 의한 곱셈은 $\mathbb{R}^{n}$에서 $\mathbb{R}^{m}$로 벡터를 투영하는 변환으로 생각할 수 있습니다.이러한 변환은 매우 유용하다는 것이 밝혀졌습니다.예를 들어 회전을 정사각 행렬로 곱셈으로 표현할 수 있습니다.다음 장에서 볼 수 있듯이 행렬-벡터 곱을 사용하여 이전 계층의 값이 주어지면 신경망의 각 계층을 계산할 때 필요한 가장 집중적 인 계산을 설명 할 수도 있습니다.

:begin_tab:`mxnet`
텐서가있는 코드에서 행렬-벡터 곱을 표현하면 내적과 동일한 `dot` 함수를 사용합니다.행렬 `A`과 벡터 `x`를 사용하여 `np.dot(A, x)`를 호출하면 행렬-벡터 곱이 수행됩니다.`A` (축 1을 따르는 길이) 의 열 치수는 `x` (길이) 의 치수와 같아야 합니다.
:end_tab:

:begin_tab:`pytorch`
텐서가있는 코드로 행렬-벡터 곱을 표현하면 `mv` 함수를 사용합니다.행렬 `A`과 벡터 `x`를 사용하여 `torch.mv(A, x)`을 호출하면 행렬-벡터 곱이 수행됩니다.`A` (축 1을 따르는 길이) 의 열 치수는 `x` (길이) 의 치수와 같아야 합니다.
:end_tab:

:begin_tab:`tensorflow`
텐서가있는 코드로 행렬-벡터 곱을 표현하면 `matvec` 함수를 사용합니다.행렬 `A`과 벡터 `x`를 사용하여 `tf.linalg.matvec(A, x)`을 호출하면 행렬-벡터 곱이 수행됩니다.`A` (축 1을 따르는 길이) 의 열 치수는 `x` (길이) 의 치수와 같아야 합니다.
:end_tab:

```{.python .input}
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
#@tab pytorch
A.shape, x.shape, torch.mv(A, x)
```

```{.python .input}
#@tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

## 매트릭스-매트릭스 곱셈

내적과 행렬-벡터 곱이 익숙하다면*행렬-행렬 곱셈*은 간단해야 합니다.

$\mathbf{A} \in \mathbb{R}^{n \times k}$와 $\mathbf{B} \in \mathbb{R}^{k \times m}$라는 두 개의 행렬이 있다고 가정해 보겠습니다.

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$

행렬 $\mathbf{A}$의 $i^\mathrm{th}$ 행을 나타내는 행 벡터를 $\mathbf{a}^\top_{i} \in \mathbb{R}^k$로 나타내고, $\mathbf{b}_{j} \in \mathbb{R}^k$를 행렬 $\mathbf{B}$의 $j^\mathrm{th}$ 열에 있는 열 벡터로 지정합니다.행렬 곱 $\mathbf{C} = \mathbf{A}\mathbf{B}$를 생성하려면 행 벡터로 $\mathbf{A}$을 생각하고 열 벡터로 $\mathbf{B}$을 생각하는 것이 가장 쉽습니다.

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$

그런 다음 각 요소 $c_{ij}$을 내적 $\mathbf{a}^\top_i \mathbf{b}_j$로 계산하여 행렬 곱 $\mathbf{C} \in \mathbb{R}^{n \times m}$가 생성됩니다.

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**행렬-행렬 곱셈 $\mathbf{AB}$은 단순히 $m$ 행렬-벡터 곱을 수행하고 결과를 함께 스티칭하여 $n \times m$ 행렬을 형성하는 것으로 생각할 수 있습니다.**] 다음 스니펫에서는 `A` 및 `B`에서 행렬 곱셈을 수행합니다.여기서 `A`는 5개의 행과 4개의 열이 있는 행렬이고, `B`는 4개의 행과 3개의 열로 구성된 행렬입니다.곱셈 후 5 행과 3 열로 구성된 행렬을 얻습니다.

```{.python .input}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

```{.python .input}
#@tab pytorch
B = torch.ones(4, 3)
torch.mm(A, B)
```

```{.python .input}
#@tab tensorflow
B = tf.ones((4, 3), tf.float32)
tf.matmul(A, B)
```

행렬-행렬 곱셈은 단순히*행렬 곱셈*이라고 할 수 있으며, Hadamard 곱과 혼동해서는 안 됩니다.

## 규범
:label:`subsec_lin-algebra-norms`

선형 대수에서 가장 유용한 연산자는*norms*입니다.비공식적으로 벡터의 노름은 벡터가*큰*정도를 알려줍니다.여기서 고려중인*크기*의 개념은 차원성이 아니라 구성 요소의 크기와 관련이 있습니다.

선형 대수에서 벡터 노름은 벡터를 스칼라에 매핑하여 소수의 속성을 충족하는 함수 $f$입니다.벡터 $\mathbf{x}$이 주어지면 첫 번째 속성은 벡터의 모든 요소를 상수 인수 $\alpha$로 스케일링하면 해당 노름도 동일한 상수 인자의*절대 값*에 따라 스케일링된다고 말합니다.

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$

두 번째 속성은 익숙한 삼각형 부등식입니다.

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$

세 번째 속성은 단순히 규범이 음수가 아니어야한다고 말합니다.

$$f(\mathbf{x}) \geq 0.$$

대부분의 컨텍스트에서 가장 작은*크기*는 0이기 때문에 의미가 있습니다.최종 속성에서는 가장 작은 노름이 달성되고 모든 0으로 구성된 벡터에 의해서만 달성되어야 합니다.

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$

규범은 거리 측정과 매우 흡사하다는 것을 알 수 있습니다.그리고 초등학교에서 유클리드 거리 (피타고라스의 정리를 생각해보십시오) 를 기억한다면, 비 부정성과 삼각형 불평등의 개념이 종을 울릴 수 있습니다.실제로 유클리드 거리는 표준입니다. 특히 $L_2$ 표준입니다.$n$차원 벡터 $\mathbf{x}$의 요소가 $x_1, \ldots, x_n$라고 가정합니다.

[**$\mathbf{x}$의 $L_2$*표준*은 벡터 요소의 제곱합의 제곱근입니다. **]

(**$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$달러**)

여기서 첨자 $2$는 종종 $L_2$ 규범에서 생략됩니다. 즉, $\|\mathbf{x}\|$는 $\|\mathbf{x}\|_2$과 동일합니다.코드에서 다음과 같이 벡터의 $L_2$ 노름을 계산할 수 있습니다.

```{.python .input}
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
#@tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
#@tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

딥 러닝에서는 $L_2$ 제곱 노름으로 더 자주 작업합니다.

또한 벡터 요소의 절대값의 합으로 표현되는 [**$L_1$*norm***] 을 자주 접하게 됩니다.

(**$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$달러**)

$L_2$ 노름과 비교할 때 이상값의 영향을 덜 받습니다.$L_1$ 노름을 계산하기 위해 요소에 대한 합계를 사용하여 절대값 함수를 구성합니다.

```{.python .input}
np.abs(u).sum()
```

```{.python .input}
#@tab pytorch
torch.abs(u).sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(tf.abs(u))
```

$L_2$ 규범과 $L_1$ 규범은 모두 보다 일반적인 $L_p$*표준*의 특수한 경우입니다.

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

벡터의 $L_2$ 노름과 유사하게, [**행렬 $\mathbf{X} \in \mathbb{R}^{m \times n}$**] 의*프로베니우스 노름* 은 행렬 요소의 제곱합의 제곱근입니다.

[**$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

프로베니우스 노름은 벡터 노름의 모든 속성을 충족합니다.행렬형 벡터의 $L_2$ 노름인 것처럼 동작합니다.다음 함수를 호출하면 행렬의 프로베니우스 노름이 계산됩니다.

```{.python .input}
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
#@tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
#@tab tensorflow
tf.norm(tf.ones((4, 9)))
```

### 규범 및 목적
:label:`subsec_norms_and_objectives`

우리 자신보다 너무 앞서 가고 싶지는 않지만, 이러한 개념이 왜 유용한지에 대해 이미 직관을 세울 수 있습니다.딥 러닝에서는 종종 최적화 문제를 해결하려고 합니다.
*관측 데이터에 할당 된 확률을 최대화*;
*예측값 사이의 거리를 최소화*
그리고 지상 진실 관찰이 있습니다.유사한 항목 간의 거리가 최소화되고 서로 다른 항목 간의 거리가 최대화되도록 항목 (예: 단어, 제품 또는 뉴스 기사) 에 벡터 표현을 할당합니다.딥 러닝 알고리즘의 가장 중요한 구성 요소 (데이터 제외) 인 목표는 종종 규범으로 표현됩니다.

## 선형 대수에 대해 자세히 알아보기

이 섹션에서는 놀라운 현대 딥 러닝을 이해하는 데 필요한 모든 선형 대수를 가르쳤습니다.선형 대수학에는 훨씬 더 많은 것이 있으며 많은 수학이 기계 학습에 유용합니다.예를 들어 행렬은 인자로 분해될 수 있으며 이러한 분해는 실제 데이터셋에서 저차원 구조를 나타낼 수 있습니다.기계 학습에는 행렬 분해와 그 일반화를 고차 텐서에 사용하여 데이터 세트의 구조를 발견하고 예측 문제를 해결하는 데 중점을 둔 전체 하위 필드가 있습니다.하지만 이 책은 딥 러닝에 초점을 맞추고 있습니다.또한 실제 데이터 세트에 유용한 기계 학습 모델을 배포하면 더 많은 수학을 배우려는 경향이 훨씬 더 커질 것입니다.따라서 나중에 더 많은 수학을 소개 할 권리가 있지만, 이 섹션을 여기서 마무리하겠습니다.

선형 대수에 대해 더 자세히 알고 싶다면 [online appendix on linear algebraic operations](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html) 또는 기타 우수한 리소스 :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`를 참조 할 수 있습니다.

## 요약

* 스칼라, 벡터, 행렬 및 텐서는 선형 대수의 기본 수학 객체입니다.
* 벡터는 스칼라를 일반화하고 행렬은 벡터를 일반화합니다.
* 스칼라, 벡터, 행렬 및 텐서는 각각 0, 1, 2 및 임의 개수의 축을 갖습니다.
* 텐서는 지정된 축을 따라 `sum` 및 `mean`만큼 줄일 수 있습니다.
* 두 행렬의 요소별 곱셈을 Hadamard 곱이라고 합니다.행렬 곱셈과 다릅니다.
* 딥 러닝에서는 $L_1$ 표준, $L_2$ 표준 및 프로베니우스 규범과 같은 규범을 사용하는 경우가 많습니다.
* 스칼라, 벡터, 행렬 및 텐서에 대해 다양한 연산을 수행 할 수 있습니다.

## 연습문제

1. 행렬 $\mathbf{A}$의 전치의 전치가 $\mathbf{A}$:$(\mathbf{A}^\top)^\top = \mathbf{A}$임을 입증하십시오.
1. 두 행렬 $\mathbf{A}$과 $\mathbf{B}$가 주어지면 전치수의 합이 합계의 전치 ($\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$) 와 같음을 보여줍니다.
1. 정사각 행렬 $\mathbf{A}$가 주어지면 $\mathbf{A} + \mathbf{A}^\top$는 항상 대칭입니까?왜요?
1. 이 섹션에서는 모양 (2, 3, 4) 의 텐서 `X`를 정의했습니다.`len(X)`의 출력은 무엇입니까?
1. 임의의 모양의 텐서 `X`의 경우 `len(X)`는 항상 `X`의 특정 축의 길이와 일치합니까?그 축이 뭐죠?
1. `A / A.sum(axis=1)`를 실행하고 어떤 일이 발생하는지 확인합니다.이유를 분석해 볼 수 있나요?
1. 맨해튼의 두 지점 사이를 여행 할 때 좌표, 즉 도로와 거리 측면에서 커버해야 할 거리는 얼마입니까?대각선으로 여행할 수 있나요?
1. 모양 (2, 3, 4) 의 텐서를 고려하십시오.축 0, 1, 2에 따른 합계 출력의 형태는 무엇입니까?
1. 축이 3개 이상인 텐서를 `linalg.norm` 함수에 공급하고 출력을 관찰합니다.이 함수는 임의의 형태의 텐서에 대해 무엇을 계산합니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:
