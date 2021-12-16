# 다층 퍼셉트론
:label:`sec_mlp`

:numref:`chap_linear`에서는 소프트맥스 회귀 (:numref:`sec_softmax`) 를 도입하여 알고리즘을 처음부터 구현하고 (:numref:`sec_softmax_scratch`) 상위 수준 API (:numref:`sec_softmax_concise`) 를 사용하고 저해상도 이미지에서 10개의 의류 범주를 인식하도록 분류자를 교육했습니다.그 과정에서 데이터를 논쟁하고, 출력을 유효한 확률 분포로 강제하고, 적절한 손실 함수를 적용하고, 모델의 매개 변수와 관련하여 최소화하는 방법을 배웠습니다.이제 간단한 선형 모델의 맥락에서 이러한 메커니즘을 마스터했으므로이 책이 주로 관련된 비교적 풍부한 모델 클래스 인 심층 신경망에 대한 탐구를 시작할 수 있습니다. 

## 숨겨진 레이어

우리는 :numref:`subsec_linear_model`에서 아핀 변환을 설명했는데, 이는 바이어스에 의해 추가된 선형 변환입니다.먼저 :numref:`fig_softmaxreg`에 설명된 소프트맥스 회귀 예제에 해당하는 모델 아키텍처를 생각해 보십시오.이 모델은 단일 아핀 변환을 통해 입력을 출력에 직접 매핑한 다음 소프트맥스 연산을 수행했습니다.레이블이 아핀 변환에 의해 입력 데이터와 실제로 관련되어 있다면 이 접근법만으로도 충분할 것입니다.그러나 아핀 변환의 선형성은*강한* 가정입니다. 

### 선형 모델이 잘못 될 수 있음

예를 들어, 선형성은*단조성*에 대한*약한* 가정을 의미합니다. 즉, 특징이 증가하면 항상 모델 출력이 증가하거나 (해당 가중치가 양수인 경우) 항상 모델 출력이 감소해야 합니다 (해당 가중치가 음수인 경우).때로는 의미가 있습니다.예를 들어, 개인이 대출금을 상환할지 여부를 예측하려는 경우, 다른 모든 것을 동등하게 보유하면 소득이 높은 신청자가 소득이 낮은 신청자보다 항상 상환 할 가능성이 더 높다고 합리적으로 상상할 수 있습니다.단조롭지 만 이 관계는 상환 확률과 선형적으로 연관되지 않을 수 있습니다.소득이 0에서 50,000으로 증가하면 상환 가능성이 1 백만에서 150 만 명으로 증가한 것보다 상환 가능성이 더 커질 수 있습니다.이를 처리하는 한 가지 방법은 예를 들어 소득 로그를 특징으로 사용하여 선형성이 더 그럴듯하게 되도록 데이터를 전처리하는 것입니다. 

단조 로움을 위반하는 예를 쉽게 생각해 낼 수 있습니다.예를 들어 체온에 따라 사망 확률을 예측하고 싶다고 가정 해 보겠습니다.체온이 37°C (98.6°F) 이상인 개인의 경우 온도가 높을수록 위험이 더 큽니다.그러나 체온이 37° C 미만인 개인의 경우 온도가 높을수록 위험이 낮다는 것을 나타냅니다!이 경우에도 영리한 전처리를 통해 문제를 해결할 수 있습니다.즉, 37°C에서의 거리를 특징으로 사용할 수 있습니다. 

하지만 고양이와 개 이미지를 분류하는 것은 어떨까요?위치 (13, 17) 에서 픽셀의 강도를 높이면 이미지가 개를 묘사할 가능성이 항상 증가해야 합니까 (또는 항상 감소) 해야 합니까?선형 모델에 대한 의존도는 고양이와 개를 구별하기 위한 유일한 요구 사항은 개별 픽셀의 밝기를 평가하는 것임을 암시적 가정에 해당합니다.이 접근 방식은 이미지를 반전하면 범주가 유지되는 세상에서 실패 할 운명입니다. 

그러나 이전 예제와 비교할 때 선형성의 명백한 부조리가 있음에도 불구하고 간단한 전처리 수정으로 문제를 해결할 수 있다는 것은 분명하지 않습니다.픽셀의 중요성은 컨텍스트 (주변 픽셀의 값) 에 따라 복잡한 방식으로 달라지기 때문입니다.기능 간의 관련 상호 작용을 고려한 데이터 표현이 존재할 수 있지만 그 위에 선형 모델이 적합하지만 손으로 계산하는 방법을 알지 못합니다.심층 신경망에서는 관측 데이터를 사용하여 은닉 레이어를 통한 표현과 해당 표현에 작용하는 선형 예측자를 공동으로 학습했습니다. 

### 숨겨진 레이어 통합

선형 모델의 이러한 한계를 극복하고 하나 이상의 은닉 레이어를 통합하여 보다 일반적인 함수 클래스를 처리할 수 있습니다.가장 쉬운 방법은 완전히 연결된 여러 레이어를 서로 겹쳐서 쌓는 것입니다.각 레이어는 출력을 생성할 때까지 그 위에 있는 레이어로 공급됩니다.첫 번째 $L-1$ 레이어를 표현으로 생각하고 최종 레이어를 선형 예측 변수로 생각할 수 있습니다.이 아키텍처는 일반적으로*다층 퍼셉트론*이라고 하며, 종종*MLP*로 약칭됩니다.아래에서는 MLP를 다이어그램으로 묘사합니다 (:numref:`fig_mlp`). 

![An MLP with a hidden layer of 5 hidden units. ](../img/mlp.svg)
:label:`fig_mlp`

이 MLP에는 4개의 입력, 3개의 출력이 있으며, 숨겨진 레이어에는 5개의 은닉 유닛이 있습니다.입력 계층에는 계산이 포함되지 않으므로 이 네트워크를 사용하여 출력값을 생성하려면 은닉 계층과 출력 계층 모두에 대한 계산을 구현해야 합니다. 따라서 이 MLP의 계층 수는 2입니다.이 두 레이어는 모두 완전히 연결되어 있습니다.모든 입력은 은닉 레이어의 모든 뉴런에 영향을 미치며, 각 입력은 출력 레이어의 모든 뉴런에 영향을 미칩니다.그러나 :numref:`subsec_parameterization-cost-fc-layers`에서 제안한 바와 같이 완전히 연결된 레이어를 가진 MLP의 파라미터화 비용은 엄청나게 높을 수 있으며, 이는 입력 또는 출력 크기 :cite:`Zhang.Tay.Zhang.ea.2021`를 변경하지 않고도 파라미터 저장과 모델 효율성 간의 절충을 유발할 수 있습니다. 

### 선형에서 비선형으로

이전과 마찬가지로 행렬 $\mathbf{X} \in \mathbb{R}^{n \times d}$에서 $n$ 예제의 미니 배치를 나타냅니다. 여기서 각 예제에는 $d$ 입력 (기능) 이 있습니다.은닉 레이어에 $h$개의 은닉 유닛이 있는 단일 은닉 레이어 MLP의 경우 은닉 레이어의 출력을 $\mathbf{H} \in \mathbb{R}^{n \times h}$로 나타냅니다.
*숨겨진 표현*.
수학 또는 코드에서 $\mathbf{H}$는*은닉 레이어 변수* 또는*은닉 변수*로도 알려져 있습니다.은닉 레이어와 출력 레이어가 모두 완전히 연결되어 있기 때문에 은닉 레이어 가중치 $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$과 바이어스 $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$, 출력 레이어 가중치 $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ 및 편향 $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$가 있습니다.공식적으로 단일 은닉 레이어 MLP의 출력 $\mathbf{O} \in \mathbb{R}^{n \times q}$을 다음과 같이 계산합니다. 

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

숨겨진 레이어를 추가한 후 이제 모델에서 추가 파라미터 세트를 추적하고 업데이트해야 합니다.그래서 우리는 그 대가로 무엇을 얻었을까요?위에서 정의한 모델에서*문제에 대해 아무것도 얻지 못한다는 사실에 놀랄 수도 있습니다*!그 이유는 명백합니다.위의 은닉 유닛은 입력값의 아핀 함수에 의해 주어지며, 출력값 (pre-softmax) 은 은닉 유닛의 아핀 함수에 불과합니다.아핀 함수의 아핀 함수 자체가 아핀 함수입니다.게다가 우리의 선형 모델은 이미 모든 아핀 함수를 나타낼 수 있었습니다. 

가중치의 모든 값에 대해 숨겨진 계층을 축소하여 매개 변수 $\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$ 및 $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$를 갖는 동등한 단일 계층 모델을 생성 할 수 있음을 증명함으로써 등가를 공식적으로 볼 수 있습니다. 

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

다층 아키텍처의 잠재력을 실현하기 위해서는 아핀 변환 후 각 은닉 유닛에 적용할 비선형*활성화 함수* $\sigma$라는 핵심 요소가 하나 더 필요합니다.활성화 함수 (예: $\sigma(\cdot)$) 의 출력을 *활성화*라고 합니다.일반적으로 활성화 함수가 있으면 MLP를 선형 모델로 축소할 수 없습니다. 

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

$\mathbf{X}$의 각 행은 미니 배치의 예와 일치하므로 표기법을 남용하여 비선형 성 $\sigma$을 행 방식으로, 즉 한 번에 하나의 예제로 입력에 적용 할 비선형 성 $\sigma$을 정의합니다.:numref:`subsec_softmax_vectorization`에서 행별 연산을 나타내는 것과 같은 방식으로 소프트맥스에 대한 표기법을 사용했습니다.이 섹션에서와 같이 히든 레이어에 적용하는 활성화 함수는 행별로 적용되는 것이 아니라 요소별로 적용되는 경우가 많습니다.즉, 레이어의 선형 부분을 계산한 후 다른 은닉 유닛이 취한 값을 보지 않고도 각 활성화를 계산할 수 있습니다.이는 대부분의 활성화 함수에 해당됩니다. 

보다 일반적인 MLP를 구축하기 위해 이러한 숨겨진 레이어 (예: $\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$ 및 $\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$) 를 서로 겹쳐 쌓아서 훨씬 더 표현력이 풍부한 모델을 생성 할 수 있습니다. 

### 유니버설 근사기

MLP는 각 입력의 값에 따라 달라지는 숨겨진 뉴런을 통해 입력 간의 복잡한 상호 작용을 포착 할 수 있습니다.숨겨진 노드를 쉽게 설계하여 임의의 계산 (예: 한 쌍의 입력에 대한 기본 논리 연산) 을 수행 할 수 있습니다.또한 활성화 기능의 특정 선택에 대해 MLP가 범용 근사치라는 것이 널리 알려져 있습니다.충분한 노드 (아마도 터무니없이 많은 노드) 와 올바른 가중치 세트가 주어진 단일 은폐 계층 네트워크를 사용하더라도 실제로 그 함수를 배우는 것이 어려운 부분이지만 모든 함수를 모델링 할 수 있습니다.신경망은 C 프로그래밍 언어와 약간 비슷하다고 생각할 수 있습니다.이 언어는 다른 현대 언어와 마찬가지로 계산 가능한 프로그램을 표현할 수 있습니다.하지만 실제로 사양에 맞는 프로그램을 만드는 것은 어려운 부분입니다. 

게다가 단일 은폐 계층 네트워크가
*모든 기능을 배울 수 있습니다
단일 은폐 계층 네트워크의 모든 문제를 해결하려고 시도해야 한다는 의미는 아닙니다.실제로 더 깊은 (대 더 넓은) 네트워크를 사용하여 많은 함수를 훨씬 더 간결하게 근사화할 수 있습니다.우리는 다음 장에서 더 엄격한 주장을 다룰 것입니다. 

## 활성화 함수
:label:`subsec_activation-functions`

활성화 함수는 가중 합계를 계산하고 바이어스를 추가하여 뉴런의 활성화 여부를 결정합니다.입력 신호를 출력으로 변환하는 미분 가능한 연산자이며 대부분은 비선형성을 추가합니다.활성화 함수는 딥러닝의 기본이기 때문에 (**몇 가지 일반적인 활성화 함수를 간단히 살펴보자**).

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

### 리루 함수

구현의 단순성과 다양한 예측 작업에서 우수한 성능으로 인해 가장 많이 사용되는 선택은*정류 선형 단위* (*Relu*) 입니다.[**ReLU는 매우 간단한 비선형 변환**] 을 제공합니다.요소 $x$가 주어지면 함수는 해당 요소의 최대값과 $0$로 정의됩니다. 

$$\operatorname{ReLU}(x) = \max(x, 0).$$

비공식적으로 ReLU 함수는 양수 요소만 유지하고 해당 활성화를 0으로 설정하여 모든 음수 요소를 삭제합니다.직관을 얻기 위해 함수를 플로팅할 수 있습니다.보시다시피 활성화 함수는 조각별 선형입니다.

```{.python .input}
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

입력값이 음수이면 ReLU 함수의 도함수가 0이고 입력값이 양수이면 ReLU 함수의 도함수가 1입니다.입력값이 정확히 0인 경우 ReLU 함수를 미분할 수 없습니다.이 경우 기본적으로 왼쪽 도함수를 사용하고 입력이 0일 때 도함수가 0이라고 말합니다.입력이 실제로 0이 아닐 수 있기 때문에 이 문제를 해결할 수 있습니다.미묘한 경계 조건이 중요하다면 공학이 아닌 (*실제*) 수학을 하고 있다는 오래된 격언이 있습니다.이러한 일반적인 지혜가 여기에 적용될 수 있습니다.아래에 표시된 ReLU 함수의 도함수를 플로팅합니다.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

ReLU를 사용하는 이유는 파생물이 특히 잘 동작하기 때문입니다. 사라지거나 인수를 통과시킵니다.이로 인해 최적화가 더 잘 작동하고 이전 버전의 신경망을 괴롭혔던 그래디언트가 사라지는 잘 문서화 된 문제가 완화되었습니다 (자세한 내용은 나중에 설명). 

ReLU 함수에는*매개변수화된 ReLU* (*PreLU*) 함수 :cite:`He.Zhang.Ren.ea.2015`를 포함하여 많은 변형이 있습니다.이 변형은 ReLU에 선형 항을 추가하므로 인수가 음수인 경우에도 일부 정보는 여전히 통과합니다. 

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### 시그모이드 함수

[***시그모이드 함수*는 값이 영역 $\mathbb{R}$에 있는 입력**] 을 변환합니다 (**구간 (0, 1) 에 있는 출력으로.**) 이러한 이유로 시그모이드는 종종*스쿼싱 함수*라고 합니다. 범위 (-inf, inf) 의 입력을 범위 (0, 1) 의 값으로 스쿼시합니다. 

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

초기 신경망에서 과학자들은*화재* 또는*발사하지 않는*생물학적 뉴런을 모델링하는 데 관심이 있었습니다.따라서이 분야의 개척자들은 인공 뉴런의 발명가 인 McCulloch와 Pitts로 돌아가 임계 단위에 중점을 두었습니다.임계값 활성화는 입력이 일부 임계값보다 작을 때는 값 0을, 입력이 임계값을 초과하면 값 1을 받습니다. 

주의가 기울기 기반 학습으로 전환되었을 때 시그모이드 함수는 분계점 단위에 대한 부드럽고 미분 가능한 근사치이기 때문에 자연스러운 선택이었습니다.시그모이드는 출력값을 이진 분류 문제에 대한 확률로 해석하고자 할 때 출력 단위의 활성화 함수로 널리 사용됩니다 (시그모이드는 소프트맥스의 특수한 경우라고 생각할 수 있음).그러나 시그모이드는 대부분 은닉 레이어에서 사용하기 위해 더 간단하고 쉽게 훈련 가능한 ReLU로 대체되었습니다.순환 신경망에 대한 이후 장에서는 시그모이드 단위를 활용하여 시간에 따른 정보 흐름을 제어하는 아키텍처에 대해 설명합니다. 

아래에서는 시그모이드 함수를 플로팅합니다.입력값이 0에 가까우면 시그모이드 함수가 선형 변환에 접근합니다.

```{.python .input}
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

시그모이드 함수의 도함수는 다음 방정식으로 계산됩니다. 

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$

시그모이드 함수의 도함수가 아래에 그려져 있습니다.입력값이 0이면 시그모이드 함수의 도함수가 최대값 0.25에 도달합니다.입력값이 0에서 어느 방향 으로든 갈라지면 도함수가 0에 접근합니다.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

### 탄 함수

시그모이드 함수와 마찬가지로 [**tanh (쌍곡선 탄젠트) 함수는 입력값을**] 스쿼시하여 구간 (**-1과 1** 사이) 의 요소로 변환합니다. 

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

아래에 tanh 함수를 플로팅합니다.입력값이 0에 가까워지면 tanh 함수가 선형 변환에 접근합니다.함수의 모양은 시그모이드 함수의 모양과 유사하지만 tanh 함수는 좌표계의 원점에 대한 점 대칭을 나타냅니다.

```{.python .input}
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

tanh 함수의 도함수는 다음과 같습니다. 

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

tanh 함수의 미분은 아래에 그려져 있습니다.입력값이 0에 가까워지면 tanh 함수의 도함수가 최대값 1에 근접합니다.시그모이드 함수에서 보았 듯이 입력이 어느 방향 으로든 0에서 멀어지면 tanh 함수의 도함수가 0에 접근합니다.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# Clear out previous gradients.
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

요약하면, 이제 비선형성을 통합하여 표현적인 다층 신경망 아키텍처를 구축하는 방법을 알게 되었습니다.참고로, 귀하의 지식은 이미 1990 년경 실무자와 유사한 툴킷을 지휘하고 있습니다.강력한 오픈 소스 딥 러닝 프레임워크를 활용하여 몇 줄의 코드만으로 모델을 빠르게 구축할 수 있기 때문에 어떤 면에서는 1990년대에 작업하는 사람보다 유리합니다.이전에는 이러한 네트워크를 훈련시키기 위해 연구원들은 수천 줄의 C와 포트란을 코딩해야 했습니다. 

## 요약

* MLP는 출력 레이어와 입력 레이어 사이에 완전히 연결된 은닉 레이어를 하나 또는 여러 개 추가하고 활성화 함수를 통해 은닉 레이어의 출력값을 변환합니다.
* 일반적으로 사용되는 활성화 함수에는 ReLU 함수, 시그모이드 함수 및 tanh 함수가 포함됩니다.

## 연습문제

1. PreLU 활성화 함수의 도함수를 계산합니다.
1. ReLU (또는 PreLU) 만 사용하는 MLP가 연속 조각별 선형 함수를 구성한다는 것을 보여줍니다.
1. $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$를 보여주세요.
1. 한 번에 하나의 미니배치에 적용되는 비선형성이 있다고 가정합니다.이로 인해 어떤 종류의 문제가 발생할 것으로 예상하십니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/226)
:end_tab:
