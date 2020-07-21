# 다층 퍼셉트론
:label:`sec_mlp`

:numref:`chap_linear`에서는 소프트맥스 회귀 (:numref:`sec_softmax`) 를 도입하여 알고리즘을 처음부터 구현하고 (:numref:`sec_softmax_scratch`) 고급 API (:numref:`sec_softmax_concise`) 및 교육 분류자를 사용하여 저해상도 이미지에서 10 가지 범주의 의류를 인식합니다.그 과정에서 우리는 데이터를 논쟁하고, 출력을 유효한 확률 분포로 강제 변환하고, 적절한 손실 함수를 적용하고, 모델의 매개 변수와 관련하여 최소화하는 방법을 배웠습니다.이제 우리는 단순한 선형 모델의 맥락에서 이러한 메커니즘을 마스터했으므로, 이 책이 주로 관련되어 있는 비교적 풍부한 종류의 모델인 깊은 신경망에 대한 탐사를 시작할 수 있습니다.

## 숨겨진 레이어

우리는 바이어스에 의해 추가 된 선형 변환 인 :numref:`subsec_linear_model`에서 아핀 변환을 설명했다.시작하려면 :numref:`fig_softmaxreg`에 설명된 softmax 회귀 예제에 해당하는 모델 아키텍처를 다시 생각해 보십시오.이 모델은 단일 아핀 변환을 통해 입력을 출력에 직접 매핑하고 softmax 작업을 수행했습니다.우리의 라벨이 아핀 변환에 의해 입력 데이터와 정말로 관련이 있다면, 이 접근법으로 충분할 것입니다.그러나 아핀 변환의 선형성은*강한* 가정입니다.

### 선형 모형이 잘못 될 수 있음

예를 들어, 선형성은*단량*의*약한* 가정을 의미합니다. 피쳐가 증가하면 항상 모델의 출력이 증가하거나 (해당 중량이 양수인 경우) 모델의 출력이 감소해야 합니다 (해당 중량이 음수인 경우).때로는 의미가 있습니다.예를 들어, 개인이 대출을 상환할지 여부를 예측하려는 경우, 다른 모든 것을 동등하게 유지하면 더 높은 소득을 가진 신청자가 항상 낮은 소득을 가진 사람보다 상환할 가능성이 더 높다고 합리적으로 상상할 수 있습니다.단조롭지 만, 이 관계는 선형 적으로 상환 확률과 관련이 없습니다.0에서 50,000으로 소득이 증가하면 1 백만에서 1.05 백만으로 증가하는 것보다 상환 가능성이 더 커질 수 있습니다.이를 처리하는 한 가지 방법은 소득의 대수를 우리의 특징으로 사용하여 선형성이 더 그럴듯하게되도록 데이터를 전처리하는 것입니다.

단조 로움을 위반하는 예제를 쉽게 제시 할 수 있습니다.예를 들어 우리는 체온에 따라 사망 확률을 예측하고 싶다고 가정 해보십시오.체온이 37°C (98.6°F) 이상인 경우 온도가 높으면 위험이 커집니다.그러나 체온이 37°C 이하인 개인의 경우 온도가 높을수록 위험이 낮아집니다!이 경우에도 영리한 전처리로 문제를 해결할 수 있습니다.즉, 우리는 우리의 기능으로 37° C에서 거리를 사용할 수 있습니다.

그러나 고양이와 강아지의 이미지를 분류하는 것은 어떨까요?위치 (13, 17) 에서 픽셀의 강도를 증가해야 항상 증가 (또는 항상 감소) 이미지가 개를 묘사 할 가능성을?선형 모델에 대한 의존도는 고양이 대 개를 구별하기 위한 유일한 요구 사항은 개별 픽셀의 밝기를 평가하는 것뿐이라는 암시적인 가정에 해당합니다.이 접근법은 이미지를 반전하면 카테고리가 보존되는 세계에서 실패 할 운명입니다.

그러나 앞의 예와 비교할 때 선형성의 명백한 부조리에도 불구하고 간단한 전처리 수정으로 문제를 해결할 수 있다는 것은 분명하지 않습니다.이는 모든 픽셀의 중요성이 컨텍스트 (주변 픽셀의 값) 에 복잡한 방식으로 의존하기 때문입니다.선형 모델이 적합 할 수있는 기능 간의 관련 상호 작용을 고려한 데이터 표현이 존재할 수도 있지만 단순히 손으로 계산하는 방법을 알지 못합니다.깊은 신경망을 통해 관측 데이터를 사용하여 숨겨진 레이어를 통한 표현과 그 표현에 따라 작동하는 선형 예측 변수를 공동으로 학습했습니다.

### 숨겨진 레이어 통합

선형 모델의 이러한 한계를 극복하고 하나 이상의 숨겨진 레이어를 통합하여 좀 더 일반적인 함수 클래스를 처리 할 수 있습니다.이를 수행하는 가장 쉬운 방법은 완전히 연결된 많은 레이어를 서로 쌓는 것입니다.각 레이어는 출력을 생성 할 때까지 그 위의 레이어로 공급됩니다.우리는 첫 번째 $L-1$ 층을 우리의 표현으로, 최종 계층을 선형 예측 변수로 생각할 수 있습니다.이 아키텍처는 일반적으로*다층 퍼셉트론*이라고 하며, 종종 *MLP*로 축약됩니다.아래, 우리는 다이어그램으로 MLP를 묘사 (:numref:`fig_mlp`).

![An MLP with a hidden layer of 5 hidden units. ](../img/mlp.svg)
:label:`fig_mlp`

이 MLP에는 4 개의 입력, 3 개의 출력이 있으며 숨겨진 레이어는 5 개의 숨겨진 단위를 포함합니다.입력 레이어에는 계산이 포함되지 않으므로 이 네트워크에서 출력을 생성하려면 숨겨진 레이어와 출력 레이어 모두에 대한 계산을 구현해야 합니다. 따라서 이 MLP의 레이어 수는 2입니다.이 레이어는 모두 완전히 연결되어 있습니다.모든 입력은 숨겨진 레이어의 모든 뉴런에 영향을 주며, 이들 각각은 출력 레이어의 모든 뉴런에 영향을 줍니다.

### 선형에서 비선형으로

이전과 마찬가지로 행렬 $\mathbf{X} \in \mathbb{R}^{n \times d}$에 의해 각 예제에 $d$ 입력 (기능) 이있는 7323621 예제의 미니 배치를 나타냅니다.숨겨진 레이어에 $h$ 숨겨진 단위가 있는 단일 숨겨진 레이어 MLP의 경우 숨겨진 레이어의 출력을 $\mathbf{H} \in \mathbb{R}^{n \times h}$로 나타냅니다.여기서 $\mathbf{H}$는*숨겨진 레이어 변수* 또는*숨겨진 변수*라고도 합니다.숨겨진 및 출력 레이어가 모두 완전히 연결되어 있기 때문에 숨겨진 레이어 가중치 $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ 및 바이어스 $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$ 및 출력 레이어 가중치 $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ 및 바이어스 $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$가 있습니다.공식적으로, 우리는 다음과 같이 하나의 숨겨진 레이어 MLP의 출력 $\mathbf{O} \in \mathbb{R}^{n \times q}$를 계산합니다.

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

숨겨진 레이어를 추가 한 후 모델에 추가 매개 변수 세트를 추적하고 업데이트해야합니다.그래서 우리는 대가로 무엇을 얻었습니까?위에 정의된 모델에서 이 사실을 알게 되면 놀라실 수도 있습니다* 우리는 우리의 문제를 위해 아무 것도 얻지 못합니다*!그 이유는 분명합니다.위의 숨겨진 단위는 입력의 아핀 함수에 의해 주어지며 출력 (pre-softmax) 은 숨겨진 단위의 아핀 함수입니다.아핀 함수의 아핀 함수 자체는 아핀 함수입니다.또한, 우리의 선형 모델은 이미 어떤 아핀 함수를 나타낼 수 있었다.

가중치의 모든 값에 대해 숨겨진 레이어를 축소하여 매개 변수 $\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$ 및 $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$가있는 동등한 단일 레이어 모델을 생성 할 수 있음을 증명함으로써 동등성을 공식적으로 볼 수 있습니다.

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

다층 아키텍처의 잠재력을 실현하기 위해서는 아핀 변환 후 각 숨겨진 유닛에 적용되는 비선형 * 활성화 기능* $\sigma$라는 또 하나의 핵심 성분이 필요합니다.활성화 함수 (예: $\sigma(\cdot)$) 의 출력을 *활성화*라고합니다.일반적으로 활성화 기능을 사용하면 MLP를 선형 모델로 축소 할 수 없습니다.

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

$\mathbf{X}$의 각 행은 미니 배치의 예와 일치하기 때문에 표기법을 남용하여 행 방식으로 입력에 적용 할 비선형 $\sigma$을 정의합니다 (예: 한 번에 하나의 예).우리는 :numref:`subsec_softmax_vectorization`에서 행 연산을 나타 내기 위해 같은 방식으로 softmax에 대한 표기법을 사용했습니다.이 섹션과 마찬가지로 숨겨진 레이어에 적용하는 활성화 기능은 단순히 행 방향이 아니라 요소별로 적용됩니다.즉, 레이어의 선형 부분을 계산한 후 다른 숨겨진 단위가 취한 값을 보지 않고도 각 활성화를 계산할 수 있습니다.이는 대부분의 정품 인증 기능에 해당됩니다.

좀 더 일반적인 MLP를 구축하기 위해 $\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$ 및 $\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$와 같은 숨겨진 레이어를 계속 쌓아 더 표현력이 풍부한 모델을 만들 수 있습니다.

### 유니버설 근사기

MLP는 각 입력의 값에 따라 숨겨진 뉴런을 통해 입력 간의 복잡한 상호 작용을 포착할 수 있습니다.우리는 숨겨진 노드를 쉽게 설계하여 임의의 계산을 수행 할 수 있습니다 (예: 입력 쌍에 대한 기본 논리 연산).또한 활성화 기능의 특정 선택에 대해 MLP는 보편적 인 근사기라는 것이 널리 알려져 있습니다.단일 숨겨진 계층 네트워크, 충분한 노드 (아마도 많은 노드) 와 올바른 가중치 세트가 주어지더라도 실제로 그 함수를 배우는 것이 어려운 부분이지만 모든 함수를 모델링 할 수 있습니다.신경망을 C 프로그래밍 언어와 약간 비슷하다고 생각할 수도 있습니다.이 언어는 다른 현대 언어와 마찬가지로 계산 가능한 프로그램을 표현할 수 있습니다.그러나 실제로 당신의 사양을 충족하는 프로그램과 함께 오는 것은 어려운 부분입니다.

또한 단일 숨겨진 계층 네트워크
*모든 기능을 배울 수 있습니다.
단일 숨겨진 계층 네트워크에서 모든 문제를 해결하려고 시도해야한다는 것을 의미하지는 않습니다.사실, 우리는 더 깊은 (더 넓은) 네트워크를 사용하여 많은 기능을 훨씬 더 콤팩트하게 근사시킬 수 있습니다.우리는 다음 장에서 더 엄격한 인수에 터치합니다.

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

## 활성화 기능

활성화 기능은 가중 합계를 계산하고 추가로 바이어스를 추가하여 뉴런을 활성화할지 여부를 결정합니다.이들은 입력 신호를 출력으로 변환하는 차별화 가능한 연산자이며, 대부분은 비선형성을 추가합니다.활성화 기능은 딥 러닝의 기본이기 때문에 몇 가지 일반적인 활성화 기능을 간략하게 살펴보겠습니다.

### reLu 함수

구현의 단순성과 다양한 예측 작업에서 우수한 성능으로 인해 가장 많이 사용되는 선택은*정류 선형 단위* (*RelU*) 입니다.RelU는 매우 간단한 비선형 변환을 제공합니다.요소 $x$가 주어지면 함수는 해당 요소의 최대 값과 $0$로 정의됩니다.

$$\operatorname{ReLU}(x) = \max(x, 0).$$

비공식적으로 reLu 함수는 양의 요소만 유지하고 해당 활성화를 0으로 설정하여 모든 음수 요소를 삭제합니다.직관을 얻기 위해 함수를 그릴 수 있습니다.보시다시피 활성화 기능은 조각 별 선형입니다.

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

입력이 음수이면 reLu 함수의 미분은 0이고 입력이 양수이면 reLu 함수의 미분은 1입니다.입력이 0과 정확히 같은 값을 취하는 경우 RelU 함수를 구분할 수 없습니다.이 경우, 우리는 왼쪽 유도체를 기본값으로 입력이 0 일 때 파생 상품이 0이라고 말합니다.입력이 실제로 0이 될 수 없기 때문에 우리는 이것으로 도망 갈 수 있습니다.미묘한 경계 조건이 중요하다면 우리는 아마도 공학이 아닌 (* 실제*) 수학을 할 것입니다.그 전통적인 지혜가 여기에 적용될 수 있습니다.우리는 아래에 그려진 reLu 함수의 미분을 플롯합니다.

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

RelU를 사용하는 이유는 파생 상품이 특히 잘 작동하기 때문입니다. 사라지거나 인수를 통과시킵니다.이렇게하면 최적화가 더 잘 작동하고 이전 버전의 신경망을 괴롭히는 그라디언트를 사라지는 잘 문서화 된 문제를 완화합니다 (자세한 내용은 나중에 참조).

*매개 변수화된 RelU* (*PreLU*) 함수 :cite:`He.Zhang.Ren.ea.2015`를 포함하여 ReLu 함수에는 많은 변형이 있습니다.이 변형은 Relu에 선형 용어를 추가하므로 인수가 음수 인 경우에도 일부 정보가 계속 전달됩니다.

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### 시그모이드 함수

* 시그모이드 함수*는 값이 도메인 $\mathbb{R}$에있는 입력을 간격 (0, 1) 에 놓인 출력으로 변환합니다.이러한 이유로 Sigmoid는 종종* 스쿼싱 함수*라고합니다. 범위 (-inf, inf) 의 모든 입력을 범위 (0, 1) 의 일부 값으로 스쿼시합니다.

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

초기 신경망에서 과학자들은 *fire* 또는*발광하지 않는 생물학적 뉴런을 모델링하는 데 관심이 있었습니다.따라서 인공 뉴런의 발명가인 McCulloch와 피츠로 돌아가는 이 분야의 개척자들은 문턱 장치에 초점을 맞추었습니다.입력이 임계 값을 초과하면 입력이 일부 임계 값 미만인 경우 임계 활성화는 값 0을 취합니다.

주의가 그라디언트 기반 학습으로 바뀌었을 때, Sigmoid 함수는 임계 단위에 대한 부드럽고 차별화 가능한 근사치이기 때문에 자연스러운 선택이었습니다.Sigmoids는 출력을 바이너리 분류 문제에 대한 확률로 해석하고자 할 때 출력 단위의 활성화 함수로 널리 사용됩니다 (Sigmoid를 softmax의 특별한 경우라고 생각할 수 있음).그러나 Sigmoid는 대부분 숨겨진 레이어에서 가장 사용하기 위해 간단하고 쉽게 교육 할 수있는 RELU로 대체되었습니다.반복되는 신경망에 대한 다음 장에서는 Sigmoid 단위를 활용하여 시간 경과에 따라 정보의 흐름을 제어하는 아키텍처를 설명합니다.

아래, 우리는 S 자 모이드 함수를 플롯합니다.입력이 0에 가까울 때, Sigmoid 함수는 선형 변환에 접근합니다.

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

Sigmoid 함수의 미분은 다음 방정식에 의해 주어진다:

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$

S 자 모이드 함수의 미분은 아래에 그려져 있습니다.입력이 0일 때, S 자 모이드 함수의 미분은 최대 0.25에 도달합니다.입력이 어느 방향 으로든 0에서 분기되면 미분은 0에 접근합니다.

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

### 탄 기능

Sigmoid 함수와 마찬가지로 tanh (쌍곡선 탄젠트) 함수는 입력을 스쿼시하여 -1과 1 사이의 간격으로 요소로 변환합니다.

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

우리는 아래의 tanh 함수를 플롯합니다.입력이 0에 가까워지면 tanh 함수가 선형 변환에 접근합니다.함수의 형상은 Sigmoid 함수의 형상과 유사하지만 tanh 함수는 좌표계의 원점에 대한 점 대칭을 나타냅니다.

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

tanh 함수의 미분은 다음과 같습니다.

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

tanh 함수의 미분은 아래에 그려져 있습니다.입력이 0에 가까워지면, tanh 함수의 미분 값은 최대 1에 접근한다.입력 어느 방향으로 0에서 멀리 이동으로 우리가 S 자 모이드 함수로 본 바와 같이, tanh 함수의 미분 0에 접근한다.

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

요약하면, 우리는 이제 비선형을 통합하여 표현적인 다층 신경망 아키텍처를 구축하는 방법을 알고 있습니다.부수적으로, 당신의 지식은 이미 1990 년경에 실무자와 유사한 툴킷을 명령합니다.몇 줄의 코드만 사용하여 모델을 빠르게 빌드하기 위해 강력한 오픈 소스 딥 러닝 프레임워크를 활용할 수 있기 때문에 어떤 면에서 1990년대에 일하는 사람보다 이점이 있습니다.이전에는 이러한 네트워크를 교육하는 연구원들이 수천 줄의 C와 Fortran을 코딩해야 했습니다.

## 요약

* MLP는 출력 레이어와 입력 레이어 사이에 하나 또는 여러 개의 완전히 연결된 숨겨진 레이어를 추가하고 활성화 기능을 통해 숨겨진 레이어의 출력을 변환합니다.
* 일반적으로 사용되는 활성화 함수에는 RelU 함수, Sigmoid 함수 및 tanh 함수가 포함됩니다.

## 연습 문제

1. PreLU 활성화 함수의 미분을 계산합니다.
1. ReLu (또는 PreLU) 만 사용하는 MLP가 연속 조각 별 선형 함수를 구성한다는 것을 보여줍니다.
1. 그 $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$를 보여줍니다.
1. 한 번에 하나의 미니배치에 적용되는 비선형성이 있다고 가정합니다.어떤 종류의 문제가 발생할 것으로 예상합니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/226)
:end_tab:
