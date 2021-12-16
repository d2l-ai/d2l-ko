# 고유 컴포지션
:label:`sec_eigendecompositions`

고유값은 종종 선형 대수를 공부할 때 접하게 될 가장 유용한 개념 중 하나이지만 초보자로서 그 중요성을 간과하기 쉽습니다.아래에서는 고유 구성을 소개하고 그것이 왜 그렇게 중요한지에 대한 감각을 전달하려고 노력합니다.  

다음 항목이 있는 행렬 $A$가 있다고 가정합니다. 

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$

임의의 벡터 $\mathbf{v} = [x, y]^\top$에 $A$을 적용하면 벡터 $\mathbf{A}\mathbf{v} = [2x, -y]^\top$을 얻습니다.이것은 직관적인 해석이 가능합니다. 즉, 벡터를 $x$ 방향으로 두 배 더 넓게 늘린 다음 $y$ 방향으로 뒤집습니다. 

그러나 무언가가 변경되지 않은*일부* 벡터가 있습니다.즉, $[1, 0]^\top$은 $[2, 0]^\top$로 전송되고 $[0, 1]^\top$는 $[0, -1]^\top$로 전송됩니다.이러한 벡터는 여전히 같은 선에 있으며 행렬이 각각 $2$ 및 $-1$의 인수로 늘이는 것이 유일한 수정입니다.이러한 벡터를*고유 벡터*라고 부르고*고유값*에 의해 확장되는 인자를 호출합니다. 

일반적으로 숫자 $\lambda$와 벡터 $\mathbf{v}$를 찾을 수 있다면  

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

우리는 $\mathbf{v}$가 $A$에 대한 고유 벡터이고 $\lambda$이 고유값이라고 말합니다. 

## 고유값 찾기 고유값을 찾는 방법을 알아보겠습니다.양쪽에서 $\lambda \mathbf{v}$를 뺀 다음 벡터를 분해하면 위의 내용이 다음과 같음을 알 수 있습니다. 

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_eigvalue_der`

:eqref:`eq_eigvalue_der`가 발생하려면 $(\mathbf{A} - \lambda \mathbf{I})$이 일부 방향을 0으로 압축해야하므로 반전 할 수 없으므로 행렬식은 0입니다.따라서 $\lambda$이 $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$인 것을 구하여*고유값*을 찾을 수 있습니다.고유값을 찾으면 $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$를 풀어 연관된*고유 벡터*를 찾을 수 있습니다. 

### 예제 좀 더 난이도 높은 매트릭스로 살펴보겠습니다. 

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}.
$$

$\det(\mathbf{A}-\lambda \mathbf{I}) = 0$을 고려하면 이것이 다항식 방정식 $0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$와 동일하다는 것을 알 수 있습니다.따라서 두 개의 고유값은 $4$와 $1$입니다.연관된 벡터를 구하려면 다음을 풀어야 합니다. 

$$
\begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \text{and} \;
\begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$

우리는 각각 벡터 $[1, -1]^\top$와 $[1, 2]^\top$를 사용하여 이 문제를 해결할 수 있습니다. 

내장 `numpy.linalg.eig` 루틴을 사용하여 코드에서 확인할 수 있습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64),
          eigenvectors=True)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

`numpy`는 고유 벡터를 길이가 1로 정규화하는 반면, 우리는 우리의 고유 벡터를 임의의 길이로 가져갔습니다.또한 기호 선택은 임의적입니다.그러나 계산된 벡터는 동일한 고유값을 사용하여 손으로 찾은 벡터와 평행합니다. 

## 행렬 분해하기 이전 예제를 한 단계 더 진행해 보겠습니다.하자 

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

열이 행렬 $\mathbf{A}$의 고유 벡터인 행렬이어야 합니다.하자 

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

대각선에 연관된 고유값이 있는 행렬이어야 합니다.그러면 고유값과 고유벡터의 정의는 다음과 같은 사실을 알 수 있습니다. 

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

행렬 $W$는 반전 가능하므로 오른쪽의 양쪽에 $W^{-1}$를 곱할 수 있습니다. 

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$
:eqlabel:`eq_eig_decomp`

다음 섹션에서 우리는 이것에 대한 몇 가지 좋은 결과를 보게 될 것입니다. 그러나 지금은 선형 적으로 독립적 인 고유 벡터의 전체 집합을 찾을 수있는 한 ($W$가 반전되도록) 그러한 분해가 존재한다는 것을 알아야합니다. 

## 고유 컴포지션에 대한 연산 고유 컴포지션 :eqref:`eq_eig_decomp`의 한 가지 좋은 점은 우리가 일반적으로 접하는 많은 연산을 고유 컴포지션 측면에서 깔끔하게 작성할 수 있다는 것입니다.첫 번째 예로 다음 사항을 살펴보겠습니다. 

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\text{$n$ times}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\text{$n$ times}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\text{$n$ times}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

이것은 행렬의 양의 거듭제곱에 대해 고유값을 동일한 거듭제곱으로 올리는 것만으로 고유 구성을 얻는다는 것을 알 수 있습니다.음의 거듭제곱에 대해서도 동일하게 표시 될 수 있으므로 행렬을 반전하려면 다음을 고려하면됩니다. 

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$

즉, 각 고유값을 반전하기만 하면 됩니다.이것은 각 고유값이 0이 아닌 한 작동하므로, 반전 가능은 고유값이 0이 없는 것과 동일하다는 것을 알 수 있습니다.   

실제로 추가 작업을 통해 $\lambda_1, \ldots, \lambda_n$가 행렬의 고유값인 경우 해당 행렬의 행렬식은 다음과 같다는 것을 알 수 있습니다. 

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$

또는 모든 고유값의 곱입니다.스트레칭 $\mathbf{W}$이 무엇이든 $W^{-1}$가 취소하기 때문에 직관적으로 의미가 있습니다. 따라서 결국 발생하는 유일한 스트레칭은 대각선 요소의 곱으로 볼륨을 늘리는 대각선 행렬 $\boldsymbol{\Sigma}$로 곱하는 것입니다. 

마지막으로, 랭크는 행렬에서 선형적으로 독립된 열의 최대 개수라는 것을 기억해 보겠습니다.고유 구성을 면밀히 살펴보면 순위가 $\mathbf{A}$의 0이 아닌 고유값의 수와 동일하다는 것을 알 수 있습니다. 

예제는 계속 될 수 있지만 요점은 분명하기를 바랍니다. eigendecomposition 은 많은 선형-대수 계산을 단순화 할 수 있으며 많은 수치 알고리즘과 선형 대수에서 수행하는 많은 분석의 기초가되는 기본 연산입니다.  

## 대칭 행렬의 고유구성 위의 프로세스가 작동하기에 충분한 선형 독립 고유벡터를 찾는 것이 항상 가능한 것은 아닙니다.예를 들어, 행렬은 

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$

에는 단일 고유 벡터, 즉 $(1, 0)^\top$만 있습니다.이러한 행렬을 처리하려면 다룰 수 있는 것보다 더 고급 기술 (예: Jordan 정규형 또는 특이값 분해) 이 필요합니다.우리는 종종 전체 고유 벡터 집합의 존재를 보장 할 수있는 행렬에주의를 제한해야 할 필요가 있습니다. 

가장 일반적으로 발생하는 패밀리는*대칭 행렬*이며, 이 행렬은 $\mathbf{A} = \mathbf{A}^\top$입니다.이 경우 $W$을*직교 행렬*으로 사용할 수 있습니다. 이 행렬은 열이 모두 서로 직각을 이루는 하나의 벡터이며, 여기서 $\mathbf{W}^\top = \mathbf{W}^{-1}$이며 모든 고유값은 실수입니다.따라서이 특별한 경우에는 :eqref:`eq_eig_decomp`를 다음과 같이 쓸 수 있습니다. 

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

## 게르쉬고린 원 정리 고유값은 직관적으로 추론하기 어려운 경우가 많습니다.임의의 행렬이 제시된다면, 고유값을 계산하지 않고는 고유값이 무엇인지에 대해 말할 수 있는 것은 거의 없습니다.그러나 가장 큰 값이 대각선에 있으면 쉽게 근사할 수 있는 정리가 하나 있습니다. 

$\mathbf{A} = (a_{ij})$를 임의의 정사각 행렬 ($n\times n$) 으로 지정합니다.우리는 $r_i = \sum_{j \neq i} |a_{ij}|$을 정의할 것입니다.$\mathcal{D}_i$가 중심이 $a_{ii}$ 반지름이 $r_i$인 복잡한 평면의 디스크를 나타내도록 합니다.그런 다음 $\mathbf{A}$의 모든 고유값은 $\mathcal{D}_i$ 중 하나에 포함됩니다. 

압축을 푸는 데 약간 걸릴 수 있으므로 예제를 살펴 보겠습니다.다음 행렬을 살펴보겠습니다. 

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

우리는 $r_1 = 0.3$, $r_2 = 0.6$, $r_3 = 0.8$ 및 $r_4 = 0.9$를 보유하고 있습니다.행렬이 대칭이므로 모든 고유값은 실수입니다.즉, 모든 고유값은 다음 범위 중 하나에 있습니다.  

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$

수치 계산을 수행하면 고유값이 약 $0.99$, $2.97$, $4.95$, $9.08$이며 모두 제공된 범위 내에 편안하게 있음을 알 수 있습니다.

```{.python .input}
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

이러한 방식으로 고유값을 근사화할 수 있으며 대각선이 다른 모든 요소보다 훨씬 큰 경우 근사치가 상당히 정확합니다.   

작은 일이지만 eigendecomposition 과 같은 복잡하고 미묘한 주제를 가지고 있기 때문에 우리가 할 수있는 직관적 인 이해를 얻는 것이 좋습니다. 

## 유용한 응용: 반복된 지도의 성장

이제 고유 벡터가 원칙적으로 무엇인지 이해했으므로 신경망 동작의 중심이 되는 문제, 즉 적절한 가중치 초기화에 대한 깊은 이해를 제공하는 데 어떻게 사용될 수 있는지 살펴보겠습니다.  

### 장기 행동으로서의 고유 벡터

심층 신경망의 초기화에 대한 완전한 수학적 조사는 텍스트의 범위를 벗어나지만 고유값이 이러한 모델의 작동 방식을 확인하는 데 어떻게 도움이되는지 이해하기 위해 여기에서 장난감 버전을 볼 수 있습니다.아시다시피 신경망은 선형 변환 계층과 비선형 연산을 산재하여 작동합니다.여기서 단순화하기 위해 비선형성이 없으며 변환이 단일 반복 행렬 연산 $A$라고 가정하여 모델의 출력은 다음과 같습니다. 

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

이러한 모델이 초기화되면 $A$가 가우스 항목을 갖는 랜덤 행렬로 간주되므로 그 중 하나를 만들어 보겠습니다.구체적으로 말하자면, 평균 0, 분산 1 가우스 분포 $5 \times 5$ 행렬부터 시작합니다.

```{.python .input}
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### 랜덤 데이터에 대한 동작 장난감 모델의 단순화를 위해 $\mathbf{v}_{in}$에서 제공하는 데이터 벡터가 임의의 5차원 가우스 벡터라고 가정합니다.우리가 어떤 일이 일어나길 원하는지 생각해 봅시다.문맥의 경우, 이미지와 같은 입력 데이터를 이미지가 고양이의 그림일 확률과 같은 예측으로 변환하려는 일반적인 ML 문제를 생각해 봅시다.$\mathbf{A}$를 반복적으로 적용하여 랜덤 벡터를 매우 길게 늘리면 입력의 작은 변화가 출력에서 큰 변화로 증폭됩니다. 입력 이미지를 조금만 수정하면 예측이 크게 달라집니다.이것은 옳지 않은 것 같습니다! 

반대로 $\mathbf{A}$가 랜덤 벡터를 더 짧게 축소하면 많은 레이어를 실행 한 후 벡터는 본질적으로 아무 것도 축소되지 않으며 출력은 입력에 의존하지 않습니다.이것도 분명히 옳지 않습니다! 

입력에 따라 출력이 변경되도록하기 위해 성장과 붕괴 사이의 좁은 선을 걸어야하지만 그다지 많지는 않습니다! 

행렬 $\mathbf{A}$를 무작위 입력 벡터에 반복적으로 곱하고 규범을 추적하면 어떤 일이 발생하는지 살펴 보겠습니다.

```{.python .input}
# Calculate the sequence of norms after repeatedly applying `A`
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Calculate the sequence of norms after repeatedly applying `A`
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Calculate the sequence of norms after repeatedly applying `A`
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

규범은 통제 할 수 없게 커지고 있습니다!실제로 몫 목록을 취하면 패턴을 볼 수 있습니다.

```{.python .input}
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

위 계산의 마지막 부분을 보면 랜덤 벡터가 `1.974459321485[...]`의 인수로 늘어나는 것을 볼 수 있습니다. 여기서 끝의 부분은 약간 이동하지만 스트레칭 인자는 안정적입니다.   

### 고유 벡터로 돌아가기

고유 벡터와 고유값은 어떤 것이 늘어나는 양에 해당하지만 특정 벡터와 특정 스트레치에 해당한다는 것을 확인했습니다.$\mathbf{A}$에 대한 정보를 살펴 보겠습니다.여기에 약간의 경고가 있습니다. 모든 것을 보려면 복소수로 이동해야한다는 것이 밝혀졌습니다.이를 늘이기 및 회전으로 생각할 수 있습니다.복소수의 노름 (실수 부분과 허수 부분의 제곱합의 제곱근) 을 취하면 스트레칭 인자를 측정 할 수 있습니다.또한 정렬해 보겠습니다.

```{.python .input}
# Compute the eigenvalues
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab pytorch
# Compute the eigenvalues
eigs = torch.eig(A)[0][:,0].tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# Compute the eigenvalues
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### An 관찰

여기서 예상치 못한 일이 일어나는 것을 볼 수 있습니다. 랜덤 벡터에 적용된 행렬 $\mathbf{A}$의 장기 스트레칭을 위해 이전에 확인한 숫자는*정확히* (소수점 13자리까지 정확합니다!)$\mathbf{A}$의 가장 큰 고유값입니다.이것은 분명히 우연이 아닙니다! 

하지만 기하학적으로 무슨 일이 일어나고 있는지 생각해 보면 이치에 맞기 시작합니다.랜덤 벡터를 생각해 보겠습니다.이 랜덤 벡터는 모든 방향을 약간 가리키므로 특히 가장 큰 고유값과 연관된 $\mathbf{A}$의 고유 벡터와 같은 방향을 약간 가리킵니다.이것은*원리 고유값* 및*원리 고유 벡터*라고 불릴 정도로 중요합니다.$\mathbf{A}$을 적용한 후 랜덤 벡터는 가능한 모든 고유 벡터와 연관된 것처럼 가능한 모든 방향으로 늘어나지만, 이 주 고유 벡터와 관련된 방향으로 대부분 늘어납니다.이것이 의미하는 바는 $A$에 적용한 후 랜덤 벡터가 더 길고 주 고유 벡터와 정렬되는 데 더 가까운 방향을 가리킨다는 것입니다.행렬을 여러 번 적용한 후에는 모든 실제 목적을 위해 랜덤 벡터가 주 고유 벡터로 변환 될 때까지 주 고유 벡터와의 정렬이 점점 더 가까워집니다!실제로 이 알고리즘은 행렬의 가장 큰 고유값과 고유벡터를 찾기 위해*거듭제곱 반복*으로 알려진 알고리즘의 기초입니다.자세한 내용은 예를 들어 :cite:`Van-Loan.Golub.1983`를 참조하십시오. 

### 정규화 수정

이제 위의 논의에서 무작위 벡터가 전혀 늘어나거나 찌그러지는 것을 원하지 않는다는 결론을 내 렸습니다. 랜덤 벡터가 전체 프로세스에서 거의 같은 크기를 유지하기를 원합니다.이를 위해 이제 가장 큰 고유값이 이제 하나가 되도록 이 원리 고유값으로 행렬을 다시 스케일링합니다.이 경우에 어떤 일이 일어나는지 봅시다.

```{.python .input}
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

또한 이전과 같이 연속적인 규범 간의 비율을 플로팅하고 실제로 안정화되는지 확인할 수 있습니다.

```{.python .input}
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## 결론

이제 우리는 우리가 바라는 것을 정확히 볼 수 있습니다!기본 고유값으로 행렬을 정규화 한 후 랜덤 데이터가 이전과 같이 폭발하지 않고 결국 특정 값과 평형을 이루는다는 것을 알 수 있습니다.첫 번째 원칙에서 이러한 작업을 수행 할 수 있으면 좋을 것입니다. 수학을 깊이 살펴보면 독립적 인 평균 0, 분산 1 가우스 항목을 가진 큰 확률 행렬의 가장 큰 고유값이 평균 약 $\sqrt{n}$ 또는 우리의 경우 $\sqrt{5} \approx 2.2$이라는 것을 알 수 있습니다.*순환 법칙* :cite:`Ginibre.1965`로 알려진 매혹적인 사실 때문입니다.랜덤 행렬의 고유값 (및 특이 값이라고하는 관련 객체) 간의 관계는 :cite:`Pennington.Schoenholz.Ganguli.2017` 및 후속 연구에서 논의 된 바와 같이 신경망의 적절한 초기화와 깊은 관련이있는 것으로 나타났습니다. 

## 요약 * 고유 벡터는 방향을 변경하지 않고 행렬에 의해 늘어나는 벡터입니다. * 고유값은 행렬을 적용하여 고유 벡터가 늘어나는 양입니다.* 행렬의 고유 구성을 사용하면 많은 연산을 고유값에 대한 연산으로 줄일 수 있습니다.*Gershgorin 원 정리는 행렬의 고유값에 대한 근사 값을 제공할 수 있습니다.* 반복된 행렬 거듭제곱의 동작은 주로 가장 큰 고유값의 크기에 따라 달라집니다.이러한 이해는 신경망 초기화 이론에서 많은 응용 분야가 있습니다. 

## 연습 문제 

1. What are the eigenvalues and eigenvectors of
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}?
$$
1.  What are the eigenvalues and eigenvectors of the following matrix, and what is strange about this example compared to the previous one?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}.
$$
1. Without computing the eigenvalues, is it possible that the smallest eigenvalue of the following matrix is less that $0.5$? *Note*: this problem can be done in your head.
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/411)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1086)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1087)
:end_tab:
