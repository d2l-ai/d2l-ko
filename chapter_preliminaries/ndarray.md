# 데이터 조작
:label:`sec_ndarray`

어떤 작업을 수행하려면 데이터를 저장하고 조작할 수 있는 방법이 필요합니다.일반적으로 데이터로 해야 할 두 가지 중요한 작업이 있습니다. (i) 데이터를 획득하고 (ii) 데이터가 컴퓨터 내부에 있으면 처리합니다.데이터를 저장할 방법이 없으면 데이터를 수집 할 필요가 없으므로 먼저 합성 데이터를 가지고 놀면서 손을 더럽 히자.먼저, *텐서*라고도 하는 $n$차원 배열을 소개합니다. 

Python에서 가장 널리 사용되는 과학 컴퓨팅 패키지 인 NumPy와 함께 작업 한 적이 있다면이 섹션이 친숙하다는 것을 알게 될 것입니다.어떤 프레임워크를 사용하든, *텐서 클래스* (MXNet의 경우 `ndarray`, 파이토치와 텐서플로우 모두에서 `Tensor`) 는 몇 가지 킬러 기능을 갖춘 NumPy의 `ndarray`와 유사합니다.첫째, GPU는 계산을 가속화하기 위해 잘 지원되는 반면 NumPy는 CPU 계산만 지원합니다.둘째, 텐서 클래스는 자동 미분을 지원합니다.이러한 속성은 tensor 클래스를 딥러닝에 적합하게 만듭니다.책 전체에서 텐서라고 할 때 달리 명시되지 않는 한 텐서 클래스의 인스턴스를 참조합니다. 

## 시작하기

이 섹션에서는 책을 진행하면서 구축하게 될 기본 수학 및 수치 계산 도구를 갖추면서 시작하고 실행할 수 있도록 하는 것을 목표로 합니다.수학적 개념이나 라이브러리 함수 중 일부를 깨는 데 어려움을 겪더라도 걱정하지 마십시오.다음 섹션에서는 실제 예제의 맥락에서 이 자료를 다시 살펴보고 살펴볼 것입니다.반면에 이미 배경이 있고 수학적 내용에 대해 더 깊이 들어가고 싶다면이 섹션을 건너 뛰십시오.

:begin_tab:`mxnet`
시작하기 위해 MXNet에서 `np` (`numpy`) 및 `npx` (`numpy_extension`) 모듈을 가져옵니다.여기서 `np` 모듈에는 NumPy가 지원하는 기능이 포함되어 있으며 `npx` 모듈에는 Numpy와 유사한 환경에서 딥 러닝을 강화하기 위해 개발 된 확장 세트가 포함되어 있습니다.텐서를 사용할 때 거의 항상 `set_np` 함수를 호출합니다. 이는 MXNet의 다른 구성 요소에 의한 텐서 처리의 호환성을위한 것입니다.
:end_tab:

:begin_tab:`pytorch`
(**시작하기 위해 `torch`를 가져옵니다.파이토치라고 불리지만 `pytorch`.** 대신 `torch`를 가져와야 합니다.)
:end_tab:

:begin_tab:`tensorflow`
시작하기 위해 `tensorflow`를 가져옵니다.이름이 약간 길기 때문에 짧은 별칭 `tf`로 가져오는 경우가 많습니다.
:end_tab:

```{.python .input}
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

[**텐서는 숫자 값으로 구성된 (아마도 다차원) 배열을 나타냅니다.**] 하나의 축에서 텐서는*벡터*에 해당합니다 (수학적으로).축이 두 개인 경우 텐서는*행렬*에 해당합니다.축이 세 개 이상인 텐서에는 특별한 수학적 이름이 없습니다. 

먼저 `arange`를 사용하여 0으로 시작하는 처음 12개의 정수를 포함하는 행 벡터 `x`를 만들 수 있지만 기본적으로 부동 소수점으로 생성됩니다.텐서의 각 값을 텐서의*요소*라고합니다.예를 들어 텐서 `x`에는 12개의 요소가 있습니다.달리 지정하지 않는 한, 새 텐서는 주 메모리에 저장되고 CPU 기반 계산을 위해 지정됩니다.

```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(12)
x
```

(**`shape` 속성을 검사하여 텐서의*모양***) (~~및 총 요소 수~~) (각 축의 길이) 에 액세스 할 수 있습니다.

```{.python .input}
#@tab all
x.shape
```

텐서의 총 요소 수, 즉 모든 모양 요소의 곱을 알고 싶다면 크기를 검사 할 수 있습니다.여기서 벡터를 다루기 때문에 `shape`의 단일 요소는 크기와 동일합니다.

```{.python .input}
x.size
```

```{.python .input}
#@tab pytorch
x.numel()
```

```{.python .input}
#@tab tensorflow
tf.size(x)
```

[**요소 수 또는 값을 변경하지 않고 텐서의 모양을 변경**] 하려면 `reshape` 함수를 호출 할 수 있습니다.예를 들어, 텐서 `x`를 모양이 있는 행 벡터 (12,) 에서 모양이 있는 행렬 (3, 4) 으로 변환할 수 있습니다.이 새 텐서는 정확히 동일한 값을 포함하지만 3개의 행과 4개의 열로 구성된 행렬로 표시합니다.다시 말하지만, 모양은 변경되었지만 요소는 변경되지 않았습니다.크기는 형태를 변경해도 변경되지 않습니다.

```{.python .input}
#@tab mxnet, pytorch
X = x.reshape(3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

모든 치수를 수동으로 지정하여 형태를 변경할 필요가 없습니다.대상 모양이 모양 (높이, 너비) 이있는 행렬 인 경우 너비를 알면 높이가 암시 적으로 제공됩니다.왜 우리는 분열을 스스로 수행해야 하는가?위의 예에서 행이 3개인 행렬을 얻으려면 행 3개와 열이 4개인 행렬을 모두 지정했습니다.다행히도 텐서는 나머지가 주어지면 한 차원을 자동으로 해결할 수 있습니다.텐서가 자동으로 추론할 차원에 `-1`를 배치하여 이 기능을 호출합니다.우리의 경우 `x.reshape(3, 4)`으로 전화하는 대신 `x.reshape(-1, 4)` 또는 `x.reshape(3, -1)`라고 동등하게 부를 수 있었을 것입니다. 

일반적으로 행렬은 0, 1, 다른 상수 또는 특정 분포에서 무작위로 샘플링 된 숫자로 초기화되기를 원합니다.[**다음과 같이 모든 요소가 0**](~~또는 1~~) 로 설정되고 (2, 3, 4) 의 모양을 가진 텐서를 나타내는 텐서를 만들 수 있습니다.

```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

마찬가지로 다음과 같이 각 요소가 1로 설정된 텐서를 만들 수 있습니다.

```{.python .input}
np.ones((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.ones((2, 3, 4))
```

종종 일부 확률 분포에서 [**텐서의 각 요소에 대한 값을 무작위로 샘플링**] 하려고합니다.예를 들어 신경망에서 매개 변수로 사용할 배열을 만들 때 일반적으로 값을 임의로 초기화합니다.다음 스니펫은 모양 (3, 4) 의 텐서를 만듭니다.각 원소는 평균이 0이고 표준 편차가 1인 표준 가우스 (정규) 분포에서 랜덤하게 샘플링됩니다.

```{.python .input}
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
#@tab pytorch
torch.randn(3, 4)
```

```{.python .input}
#@tab tensorflow
tf.random.normal(shape=[3, 4])
```

숫자 값이 포함 된 Python 목록 (또는 목록 목록) 을 제공하여 원하는 텐서에서 [**각 요소에 대한 정확한 값**] 을 지정할 수도 있습니다.여기서 가장 바깥쪽 목록은 축 0에 해당하고 내부 목록은 축 1에 해당합니다.

```{.python .input}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## 오퍼레이션

이 책은 소프트웨어 엔지니어링에 관한 것이 아닙니다.우리의 관심사는 단순히 배열에서 데이터를 읽고 쓰는 것에 국한되지 않습니다.이러한 배열에 대해 수학적 연산을 수행하려고 합니다.가장 간단하고 유용한 연산 중 일부는*요소별* 연산입니다.이는 배열의 각 요소에 표준 스칼라 연산을 적용합니다.두 배열을 입력값으로 사용하는 함수의 경우 요소별 연산은 두 배열의 대응하는 각 요소 쌍에 표준 이항 연산자를 적용합니다.스칼라에서 스칼라로 매핑되는 모든 함수에서 요소 별 함수를 만들 수 있습니다. 

수학 표기법에서는 서명 $f: \mathbb{R} \rightarrow \mathbb{R}$로 이러한*단항* 스칼라 연산자 (하나의 입력을 취함) 를 나타냅니다.이것은 함수가 실수 ($\mathbb{R}$) 에서 다른 실수 ($\mathbb{R}$) 로 매핑된다는 의미입니다.마찬가지로 서명 $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$로*이진* 스칼라 연산자 (두 개의 실수 입력을 취하고 하나의 출력을 산출) 를 나타냅니다.동일한 모양*의 두 개의 벡터 $\mathbf{u}$과 $\mathbf{v}$*와 이항 연산자 $f$가 주어지면 모든 $i$에 대해 $c_i \gets f(u_i, v_i)$을 설정하여 벡터 $\mathbf{c} = F(\mathbf{u},\mathbf{v})$을 생성할 수 있습니다. 여기서 $c_i, u_i$과 $v_i$은 벡터 $\mathbf{c}, \mathbf{u}$의 $i^\mathrm{th}$ 요소이며 $\mathbf{v}$입니다.여기서는 스칼라 함수를 요소별 벡터 연산으로*lifting*하여 벡터 값 $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$을 생성했습니다. 

일반적인 표준 산술 연산자 (`+`, `-`, `*`, `/` 및 `**`) 는 모두 임의의 모양의 동일한 모양의 텐서에 대해 요소별 연산으로*들어 올려졌습니다*.같은 모양의 두 텐서에서 요소별 연산을 호출 할 수 있습니다.다음 예제에서는 쉼표를 사용하여 5 요소 튜플을 공식화합니다. 여기서 각 요소는 요소 별 연산의 결과입니다. 

### 오퍼레이션

[**일반적인 표준 산술 연산자 (`+`, `-`, `*`, `/` 및 `**`) 는 모두 요소별 연산으로*들어 올려졌습니다.**]

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

지수와 같은 단항 연산자를 포함하여 많은 (**더 많은 연산이 요소별로 적용될 수 있습니다**).

```{.python .input}
np.exp(x)
```

```{.python .input}
#@tab pytorch
torch.exp(x)
```

```{.python .input}
#@tab tensorflow
tf.exp(x)
```

요소별 계산 외에도 벡터 내적 및 행렬 곱셈을 포함한 선형 대수 연산을 수행 할 수도 있습니다.:numref:`sec_linear-algebra`에서 선형 대수 (사전 지식이 가정되지 않음) 의 중요한 부분을 설명합니다. 

또한 여러 텐서를 함께 *연결* 할 수 있으며, 이를 엔드 투 엔드로 쌓아 더 큰 텐서를 형성 할 수 있습니다.텐서 목록을 제공하고 시스템에 연결할 축을 알려주면됩니다.아래 예는 행 (축 0, 모양의 첫 번째 요소) 과 열 (축 1, 모양의 두 번째 요소) 을 따라 두 행렬을 연결할 때 발생하는 상황을 보여줍니다.첫 번째 출력 텐서의 축-0 길이 ($6$) 는 두 입력 텐서의 축-0 길이 ($3 + 3$) 의 합입니다. 두 번째 출력 텐서의 축 -1 길이 ($8$) 는 두 입력 텐서의 축 -1 길이 ($4 + 4$) 의 합입니다.

```{.python .input}
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
#@tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

때때로, 우리는 [**논리문을 통해 이진 텐서를 생성하길 원합니다*.**] `X == Y`를 예로 들어 보겠습니다.각 위치에 대해 `X` 및 `Y`이 해당 위치에서 같으면 새 텐서의 해당 항목은 1의 값을 취합니다. 즉, 논리문 `X == Y`가 해당 위치에서 참임을 의미합니다.

```{.python .input}
#@tab all
X == Y
```

[**텐서의 모든 요소를 합산**] 하면 요소가 하나뿐인 텐서가 생성됩니다.

```{.python .input}
#@tab mxnet, pytorch
X.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(X)
```

## 브로드캐스팅 메커니즘
:label:`subsec_broadcasting`

위 섹션에서는 동일한 모양의 두 텐서에서 요소 별 연산을 수행하는 방법을 살펴 보았습니다.특정 조건에서 모양이 다르더라도 [**방송 메커니즘*을 호출하여 요소 별 연산을 수행 할 수 있습니다.**] 이 메커니즘은 다음과 같은 방식으로 작동합니다. 첫째, 요소를 적절하게 복사하여 하나 또는 두 배열을 확장하여이 변환 후 두 텐서가같은 모양.둘째, 결과 배열에 대해 요소별 연산을 수행합니다. 

대부분의 경우 다음 예제와 같이 배열이 처음에는 길이가 1인 축을 따라 브로드캐스트합니다.

```{.python .input}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
#@tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
#@tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

`a`와 `b`는 각각 $3\times1$ 행렬과 $1\times2$ 행렬이므로 추가하려는 경우 모양이 일치하지 않습니다.다음과 같이 두 행렬의 항목을 더 큰 $3\times2$ 행렬로 브로드 캐스트합니다. 행렬 `a`의 경우 열을 복제하고 행렬 `b`의 경우 두 요소 모두를 합산하기 전에 행을 복제합니다.

```{.python .input}
#@tab all
a + b
```

## 인덱싱 및 슬라이싱

다른 파이썬 배열과 마찬가지로 텐서의 요소는 인덱스로 액세스 할 수 있습니다.다른 파이썬 배열에서와 마찬가지로 첫 번째 요소는 인덱스 0을 가지며 범위는 첫 번째이지만 마지막 요소는*앞에* 포함하도록 지정됩니다.표준 Python 목록에서와 같이 음수 인덱스를 사용하여 목록 끝까지의 상대적 위치에 따라 요소에 액세스 할 수 있습니다. 

따라서 [**`[-1]`는 마지막 요소를 선택하고 `[1:3]`는 다음과 같이 두 번째 및 세 번째 요소**] 를 선택합니다.

```{.python .input}
#@tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
읽는 것 외에도 (**인덱스를 지정하여 행렬의 요소를 쓸 수도 있습니다.**)
:end_tab:

:begin_tab:`tensorflow`
텐서플로우의 `Tensors`은 변경할 수 없으며 할당할 수 없습니다. 텐서플로우의 `Variables`는 할당을 지원하는 변경 가능한 상태 컨테이너입니다.텐서플로우의 그래디언트는 `Variable` 할당을 통해 거꾸로 흐르지 않습니다. 

전체 `Variable`에 값을 할당하는 것 외에도 인덱스를 지정하여 `Variable`의 요소를 작성할 수 있습니다.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X[1, 2] = 9
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

[**여러 요소에 동일한 값을 할당하려면 모든 요소를 인덱싱한 다음 값을 할당하기만 하면됩니다.**] 예를 들어 `[0:2, :]`는 첫 번째 행과 두 번째 행에 액세스합니다. 여기서 `:`는 축 1 (열) 을 따라 모든 요소를 가져옵니다.행렬에 대한 인덱싱에 대해 논의했지만 이는 벡터와 2차원 이상의 텐서에도 적용됩니다.

```{.python .input}
#@tab mxnet, pytorch
X[0:2, :] = 12
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
X_var
```

## 메모리 절약

[**작업을 실행하면 호스트 결과에 새 메모리가 할당될 수 있습니다.**] 예를 들어 `Y = X + Y`를 쓰면 `Y`이 가리키는 데 사용한 텐서를 역참조하고 대신 새로 할당된 메모리에서 `Y`을 가리킵니다.다음 예제에서는 메모리에서 참조 된 객체의 정확한 주소를 제공하는 Python의 `id()` 함수로 이것을 보여줍니다.`Y = Y + X`을 실행하면 `id(Y)`이 다른 위치를 가리키는 것을 알 수 있습니다.이는 파이썬이 먼저 `Y + X`를 평가하여 결과에 새 메모리를 할당한 다음 `Y`이 메모리의 이 새 위치를 가리키도록 하기 때문입니다.

```{.python .input}
#@tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

이는 두 가지 이유로 바람직하지 않을 수 있습니다.첫째, 항상 불필요하게 메모리를 할당하는 것을 원하지 않습니다.기계 학습에서는 수백 MB의 매개 변수가 있고 모든 매개 변수를 초당 여러 번 업데이트할 수 있습니다.일반적으로 이러한 업데이트를*제자리*로 수행하려고 합니다.둘째, 여러 변수에서 동일한 매개 변수를 가리킬 수 있습니다.제대로 업데이트하지 않으면 다른 참조가 여전히 이전 메모리 위치를 가리키므로 코드의 일부가 부주의하게 오래된 매개 변수를 참조할 수 있습니다.

:begin_tab:`mxnet, pytorch`
다행히도 (**현재 위치 내 작업 수행**) 은 쉽습니다.슬라이스 표기법을 사용하여 이전에 할당 된 배열에 연산 결과를 할당 할 수 있습니다 (예: `Y[:] = <expression>`).이 개념을 설명하기 위해 먼저 `zeros_like`를 사용하여 $0$ 항목 블록을 할당하는 다른 `Y`와 동일한 모양을 가진 새 행렬 `Z`을 만듭니다.
:end_tab:

:begin_tab:`tensorflow`
`Variables`는 텐서플로에서 변경 가능한 상태 컨테이너입니다.모델 매개변수를 저장하는 방법을 제공합니다.`assign`을 사용하여 작업 결과를 `Variable`에 할당할 수 있습니다.이 개념을 설명하기 위해 `zeros_like`를 사용하여 $0$ 항목의 블록을 할당하는 다른 텐서 `Y`과 동일한 모양을 가진 `Variable` `Z`을 만듭니다.
:end_tab:

```{.python .input}
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

:begin_tab:`mxnet, pytorch`
[**`X` 값이 후속 계산에서 재사용되지 않는 경우 `X[:] = X + Y` 또는 `X += Y`를 사용하여 연산의 메모리 오버헤드를 줄일 수도 있습니다.**]
:end_tab:

:begin_tab:`tensorflow`
`Variable`에 state를 지속적으로 저장하더라도 모델 매개 변수가 아닌 텐서에 대한 과도한 할당을 피하여 메모리 사용량을 더 줄일 수 있습니다. 

TensorFlow `Tensors`는 변경할 수 없으며 기울기가 `Variable` 할당을 통과하지 않기 때문에 TensorFlow는 개별 작업을 제자리에서 실행하는 명시적인 방법을 제공하지 않습니다. 

그러나 TensorFlow는 `tf.function` 데코레이터를 제공하여 실행 전에 컴파일되고 최적화되는 텐서플로우 그래프 내에서 계산을 래핑합니다.이를 통해 TensorFlow는 사용하지 않는 값을 정리하고 더 이상 필요하지 않은 이전 할당을 재사용할 수 있습니다.이렇게 하면 TensorFlow 계산의 메모리 오버헤드가 최소화됩니다.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
#@tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be re-used when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## 다른 파이썬 객체로 변환

:begin_tab:`mxnet, tensorflow`
[**NumPy 텐서 (`ndarray`) **] 로 변환하거나 그 반대로 변환하는 것은 쉽습니다.변환된 결과는 메모리를 공유하지 않습니다.이 사소한 불편함은 실제로 매우 중요합니다: CPU나 GPU에서 연산을 수행할 때, 파이썬의 NumPy 패키지가 동일한 메모리 청크로 다른 작업을 수행하기를 원하는지 기다리면서 계산을 중단하고 싶지 않습니다.
:end_tab:

:begin_tab:`pytorch`
[**NumPy 텐서 (`ndarray`) **] 로 변환하거나 그 반대로 변환하는 것은 쉽습니다.torch Tensor와 numpy 배열은 기본 메모리 위치를 공유하고 인플레이스 작업을 통해 하나를 변경하면 다른 메모리 위치도 변경됩니다.
:end_tab:

```{.python .input}
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
#@tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
#@tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

(**크기 1 텐서를 파이썬 스칼라로 변환**) 하기 위해 `item` 함수 또는 파이썬의 내장 함수를 호출 할 수 있습니다.

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

## 요약

* 딥러닝을 위해 데이터를 저장하고 조작하는 주요 인터페이스는 텐서 ($n$차원 배열) 입니다.기본 수학 연산, 브로드 캐스트, 인덱싱, 슬라이싱, 메모리 절약 및 다른 Python 객체로의 변환을 포함한 다양한 기능을 제공합니다.

## 연습문제

1. 이 섹션의 코드를 실행합니다.이 섹션의 조건문 `X == Y`을 `X < Y` 또는 `X > Y`로 변경 한 다음 어떤 종류의 텐서를 얻을 수 있는지 확인하십시오.
1. 방송 메커니즘의 요소별로 작동하는 두 개의 텐서를 다른 모양 (예: 3 차원 텐서) 으로 바꿉니다.결과가 예상과 같습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:
