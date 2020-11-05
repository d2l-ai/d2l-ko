# 수치 안정성 및 초기화
:label:`sec_numerical_stability`

지금까지 구현 한 모든 모델은 미리 지정된 배포판에 따라 매개 변수를 초기화해야했습니다.지금까지 우리는 이러한 선택이 어떻게 이루어지는지에 대한 세부 사항을 설명하면서 초기화 계획을 당연하게 취했습니다.이러한 선택이 특히 중요하지 않다는 인상을 받았을 수도 있습니다.반대로, 초기화 체계의 선택은 신경망 학습에서 중요한 역할을하며 수치 안정성을 유지하는 데 중요 할 수 있습니다.또한 이러한 선택은 비선형 활성화 기능을 선택하여 흥미로운 방식으로 묶을 수 있습니다.어떤 함수를 선택하고 매개 변수를 초기화하는 방법은 최적화 알고리즘이 얼마나 빨리 수렴되는지를 결정할 수 있습니다.여기서 잘못된 선택으로 인해 훈련 중에 그라디언트가 폭발하거나 사라질 수 있습니다.이 섹션에서는 이러한 주제를 자세히 살펴보고 딥 러닝에서 경력 전반에 걸쳐 유용하게 활용할 수 있는 유용한 휴리스틱에 대해 논의합니다.

## 그라디언트 소멸 및 분해

$L$ 레이어, 입력 $\mathbf{x}$ 및 출력 $\mathbf{o}$을 갖춘 딥 네트워크를 고려하십시오.숨겨진 변수가 $\mathbf{h}^{(l)}$ ($\mathbf{h}^{(0)} = \mathbf{x}$) 인 가중치에 의해 매개 변수화 된 변형 $f_l$ ($\mathbf{h}^{(0)} = \mathbf{x}$) 에 의해 정의 된 각 층, 우리의 네트워크는 다음과 같이 표현 될 수 있습니다:

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \text{ and thus } \mathbf{o} = f_L \circ \ldots \circ f_1(\mathbf{x}).$$

모든 숨겨진 변수와 입력이 벡터 인 경우 다음과 같이 $\mathbf{W}^{(l)}$의 매개 변수 세트에 대해 7323614의 그라데이션을 작성할 수 있습니다.

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\mathrm{def}}{=}} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\mathrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\mathrm{def}}{=}}.$$

즉, 이 그라데이션은 $L-l$ 행렬 $\mathbf{M}^{(L)} \cdot \ldots \cdot \mathbf{M}^{(l+1)}$와 그라데이션 벡터 $\mathbf{v}^{(l)}$의 곱입니다.따라서 우리는 너무 많은 확률을 곱할 때 종종 발생하는 수치 언더 플로와 동일한 문제에 취약합니다.확률을 다룰 때 일반적인 트릭은 로그 공간으로 전환하는 것입니다. 즉 가수에서 수치 표현의 지수로 압력을 이동하는 것입니다.불행히도 위의 문제는 더 심각합니다. 처음에는 행렬 $\mathbf{M}^{(l)}$가 다양한 고유 값을 가질 수 있습니다.그들은 작거나 클 수 있으며, 그들의 제품은*매우 큰* 또는*매우 작은* 수 있습니다.

불안정한 그라디언트에 의해 제기 된 위험은 수치 표현을 뛰어 넘습니다.예측할 수없는 크기의 그라디언트는 최적화 알고리즘의 안정성을 위협합니다.우리는 (i) 과도하게 크고 모델 (* 폭발하는 그라디언트* 문제) 을 파괴하는 매개 변수 업데이트에 직면 할 수 있습니다. 또는 (ii) 지나치게 작은 (* 사라지는 그라디언트* 문제), 매개 변수가 각 업데이트마다 거의 이동하지 못하므로 학습이 불가능합니다.

### 그라디언트 소멸

사라지는 그라디언트 문제를 일으키는 한 가지 빈번한 원인은 각 레이어의 선형 연산 다음에 추가되는 활성화 함수 $\sigma$을 선택하는 것입니다.역사적으로, S 자 모이드 함수 $1/(1 + \exp(-x))$ (:numref:`sec_mlp`에 도입) 는 임계 함수와 유사하기 때문에 대중적이었습니다.초기 인공 신경망은 생물학적 신경망에서 영감을 받았기 때문에 생물학적 뉴런과 같은*완전히* 또는*아님을 발사하는 뉴런에 대한 아이디어가 매력적으로 보였습니다.우리가 소멸 그라디언트를 일으킬 수있는 이유를 확인하기 위해 S 자 모이드 자세히 살펴 보자.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

보시다시피, Sigmoid의 그라디언트는 입력이 크고 작은 경우 모두 사라집니다.또한 많은 레이어를 통해 역 전파 할 때, 많은 Sigmoid의 입력이 0에 가까운 Goldilocks 영역에 있지 않으면 전체 제품의 그라디언트가 사라질 수 있습니다.우리 네트워크가 많은 레이어를 자랑 할 때 조심하지 않으면 그라디언트가 일부 레이어에서 잘릴 수 있습니다.사실, 이 문제는 깊은 네트워크 교육을 괴롭히는 데 사용되었습니다.결과적으로, Relus는 더 안정적이지만 (신경적으로는 그럴듯하지 않음) 실무자를위한 기본 선택으로 등장했습니다.

### 그라디언트 분해

그라디언트가 폭발 할 때 반대의 문제는 비슷하게 화살 수 있습니다.이것을 조금 더 잘 설명하기 위해 100 개의 가우스 랜덤 행렬을 그려 초기 행렬을 곱합니다.우리가 선택한 척도 (분산 $\sigma^2=1$의 선택) 의 경우 행렬 곱이 폭발합니다.심층 네트워크의 초기화로 인해 이런 일이 발생하면 그라디언트 디센트 옵티마이 저가 수렴 할 기회가 없습니다.

```{.python .input}
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))

print('after multiplying 100 matrices', M)
```

```{.python .input}
#@tab pytorch
M = torch.normal(0, 1, size=(4,4))
print('a single matrix \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('after multiplying 100 matrices\n', M)
```

```{.python .input}
#@tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))

print('after multiplying 100 matrices\n', M.numpy())
```

### 대칭 깨기

신경망 설계의 또 다른 문제점은 매개 변수화에 내재 된 대칭입니다.하나의 숨겨진 레이어와 두 개의 단위가있는 간단한 MLP가 있다고 가정합니다.이 경우 첫 번째 레이어의 가중치 $\mathbf{W}^{(1)}$를 순회하고 출력 층의 가중치를 순회하여 동일한 함수를 얻을 수 있습니다.첫 번째 숨겨진 단위와 두 번째 숨겨진 단위를 구별하는 특별한 것은 없습니다.즉, 각 층의 숨겨진 단위 사이에 순열 대칭이 있습니다.

이것은 단지 이론적 인 성가신 그 이상입니다.두 개의 숨겨진 단위가있는 앞서 언급 한 하나의 숨겨진 레이어 MLP를 고려하십시오.설명을 위해 출력 레이어가 숨겨진 두 단위를 하나의 출력 단위로만 변환한다고 가정합니다.일부 상수 $c$에 대해 숨겨진 레이어의 모든 매개 변수를 $\mathbf{W}^{(1)} = c$로 초기화하면 어떻게 될지 상상해보십시오.이 경우, 순방향 전파 중 하나 숨겨진 장치는 출력 장치에 공급되는 동일한 활성화를 생성, 동일한 입력 및 매개 변수를 취한다.역 전파 중에 매개 변수 $\mathbf{W}^{(1)}$와 관련하여 출력 단위를 구분하면 요소가 모두 동일한 값을 갖는 그라데이션이 제공됩니다.따라서 그라데이션 기반 반복 (예: 미니 배치 확률 적 그라데이션 강하) 후에도 $\mathbf{W}^{(1)}$의 모든 요소는 여전히 동일한 값을 취합니다.이러한 반복은 결코 대칭*을 깨뜨리지 않으며 네트워크의 표현력을 실현하지 못할 수도 있습니다.숨겨진 레이어는 마치 하나의 단위만 있는 것처럼 동작합니다.미니 배치 확률 적 그래디언트 강하는이 대칭을 깨뜨리지 않지만 드롭 아웃 정규화는 것입니다!

## 매개변수 초기화

위에서 제기 된 문제를 해결하거나 적어도 완화하는 한 가지 방법은 신중한 초기화를 사용하는 것입니다.최적화 및 적절한 정규화 중 추가 관리는 안정성을 더욱 향상시킬 수 있습니다.

### 기본 초기화

이전 섹션에서, 예를 들어, :numref:`sec_linear_concise`에서, 우리는 우리의 가중치의 값을 초기화하기 위해 정규 분포를 사용했다.초기화 방법을 지정하지 않으면 프레임 워크는 기본 무작위 초기화 방법을 사용합니다. 이 메소드는 보통 문제 크기에 대해 실제로 잘 작동합니다.

### 자비에르 초기화

우리가 완전히 연결된 레이어에 대한 출력 (예를 들어, 숨겨진 변수) $o_{i}$의 규모 분포를 살펴 보자
*비선형화 없음*.
이 레이어에 대한 $n_\mathrm{in}$ 입력 $x_j$ 및 관련 가중치 $w_{ij}$을 사용하면 출력은

$$o_{i} = \sum_{j=1}^{n_\mathrm{in}} w_{ij} x_j.$$

가중치 $w_{ij}$은 모두 동일한 분포에서 독립적으로 그려집니다.또한, 우리는이 분포가 제로 평균 및 분산 $\sigma^2$를 가지고 있다고 가정 해 봅시다.이것이 분포가 가우시안 (Gaussian) 이어야 한다는 것을 의미하지는 않으며 평균과 분산이 존재해야 한다는 것을 의미합니다.지금 들어, 우리는 층 $x_j$에 대한 입력은 제로 평균 및 분산 $\gamma^2$을 가지고 있다고 가정하자 그들은 $w_{ij}$ 독립적이고 서로의 독립적 인 것을.이 경우 다음과 같이 $o_i$의 평균과 분산을 계산할 수 있습니다.

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\mathrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\mathrm{in}} E[w_{ij}] E[x_j] \\&= 0, \\
    \mathrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

분산을 고정하는 한 가지 방법은 $n_\mathrm{in} \sigma^2 = 1$를 설정하는 것입니다.이제 역 전파를 고려하십시오.그라디언트가 출력에 더 가까운 레이어에서 전파되기는 하지만 비슷한 문제에 직면합니다.순방향 전파와 동일한 추론을 사용하여 $n_\mathrm{out} \sigma^2 = 1$이 아니라면 그라디언트의 분산이 폭파 될 수 있음을 알 수 있습니다. 여기서 $n_\mathrm{out}$는이 레이어의 출력 수입니다.이로 인해 딜레마에 빠지게됩니다. 두 조건을 동시에 만족시킬 수는 없습니다.대신, 우리는 단순히 만족시키려고 노력합니다.

$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}.
\end{aligned}
$$

이것은 제작자 :cite:`Glorot.Bengio.2010`의 첫 번째 저자의 이름을 따서 명명 된 현재 표준과 실질적으로 유익한* Xavier 초기화*의 근본적인 추론입니다.일반적으로 Xavier 초기화는 평균이 0이고 분산이 $\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}}$인 가우스 분포에서 가중치를 표본합니다.또한 Xavier의 직관을 조정하여 균일 분포에서 가중치를 샘플링 할 때 분산을 선택할 수 있습니다.균일 분포 $U(-a, a)$의 분산은 $\frac{a^2}{3}$입니다.$\frac{a^2}{3}$을 7323615의 조건에 연결하면 다음과 같이 초기화 할 제안이 생성됩니다.

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$

위의 수학적 추론에서 비선형이 존재하지 않는다는 가정은 신경망에서 쉽게 위반 될 수 있지만 Xavier 초기화 방법은 실제로 잘 작동하는 것으로 나타났습니다.

### 너머

위의 추론은 매개 변수 초기화에 대한 현대적인 접근법의 표면을 간신히 긁습니다.딥 러닝 프레임워크는 수십 가지 이상의 추론을 구현하는 경우가 많습니다.또한 파라미터 초기화는 딥 러닝에서 근본적인 연구의 뜨거운 영역입니다.이들 중에는 묶인 (공유) 매개 변수, 수퍼 해상도, 시퀀스 모델 및 기타 상황에 특화된 휴리스틱 (heuristics) 이 있습니다.예를 들어, Xiao et al은 신중하게 설계된 초기화 방법 :cite:`Xiao.Bahri.Sohl-Dickstein.ea.2018`를 사용하여 아키텍처 트릭없이 10000 계층 신경망을 훈련 할 수있는 가능성을 보여주었습니다.

주제에 관심이 있으시면 이 모듈의 제품에 대해 자세히 알아보고 각 경험적 방법을 제안하고 분석한 논문을 읽은 다음 주제에 대한 최신 간행물을 살펴보는 것이 좋습니다.아마도 당신은 영리한 아이디어를 우연히 발견하거나 심지어 발명하고 딥 러닝 프레임 워크에 구현을 기여할 것입니다.

## 요약

* 그라데이션을 소실하고 폭발하는 것은 딥 네트워크에서 흔히 발생하는 문제입니다.그라디언트와 매개 변수가 잘 제어되도록하려면 매개 변수 초기화에 큰주의가 필요합니다.
* 초기 그라디언트가 너무 크거나 너무 작지 않은지 확인하려면 초기화 휴리스틱 (heuristics) 이 필요합니다.
* RelU 활성화 기능은 사라지는 그라데이션 문제를 완화합니다.이것은 수렴을 가속화 할 수 있습니다.
* 무작위 초기화는 최적화 전에 대칭이 손상되도록하는 데 중요합니다.
* Xavier 초기화는 각 레이어에 대해 출력의 분산이 입력 수에 영향을받지 않으며 그라디언트의 분산은 출력 수에 영향을받지 않는다는 것을 암시합니다.

## 연습 문제

1. 신경망이 MLP의 계층에서 순열 대칭 외에 파괴가 필요한 대칭을 나타낼 수있는 다른 경우를 설계 할 수 있습니까?
1. 선형 회귀 또는 softmax 회귀 분석의 모든 가중치 매개 변수를 동일한 값으로 초기화 할 수 있습니까?
1. 두 행렬의 곱에 대한 고유값에 대한 분석 경계를 찾습니다.그래디언트가 잘 조절되도록 보장하는 것에 대해 무엇을 알려줍니까?
1. 어떤 용어가 갈라지는 것을 안다면 사실 후에 이것을 고칠 수 있습니까?영감을 :cite:`You.Gitman.Ginsburg.2017`에 대한 레이어 단위의 적응 속도 스케일링에 대한 논문을보십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/235)
:end_tab:
