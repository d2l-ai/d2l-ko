# 수치적 안정성 및 초기화
:label:`sec_numerical_stability`

지금까지 구현한 모든 모델은 미리 지정된 분포에 따라 매개 변수를 초기화해야 했습니다.지금까지 우리는 초기화 체계를 당연한 것으로 여기며 이러한 선택이 어떻게 이루어지는지에 대한 세부 사항을 설명했습니다.이러한 선택이 특별히 중요하지 않다는 인상을 받았을 수도 있습니다.반대로 초기화 체계의 선택은 신경망 학습에서 중요한 역할을하며 수치 안정성을 유지하는 데 중요 할 수 있습니다.또한 이러한 선택은 비선형 활성화 함수의 선택과 흥미로운 방식으로 연결될 수 있습니다.어떤 함수를 선택하고 파라미터를 초기화하는 방법은 최적화 알고리즘이 수렴하는 속도를 결정할 수 있습니다.여기서 잘못 선택하면 훈련 중에 그라디언트가 폭발하거나 사라지는 경우가 발생할 수 있습니다.이 섹션에서는 이러한 주제를 더 자세히 살펴보고 딥 러닝 경력 전반에 걸쳐 유용하게 사용할 수 있는 몇 가지 유용한 휴리스틱에 대해 논의합니다. 

## 사라짐 및 폭발 그라디언트

계층이 $L$이고 입력 $\mathbf{x}$이고 출력 $\mathbf{o}$이 있는 심층 네트워크를 가정해 보겠습니다.각 계층 $l$은 가중치 $\mathbf{W}^{(l)}$로 매개 변수화된 변환 $f_l$에 의해 정의되며, 숨겨진 변수는 $\mathbf{h}^{(l)}$ ($\mathbf{h}^{(0)} = \mathbf{x}$로 지정) 인 경우 네트워크는 다음과 같이 표현할 수 있습니다. 

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \text{ and thus } \mathbf{o} = f_L \circ \ldots \circ f_1(\mathbf{x}).$$

모든 숨겨진 변수와 입력값이 벡터이면 다음과 같이 매개 변수 $\mathbf{W}^{(l)}$ 집합에 대해 $\mathbf{o}$의 기울기를 쓸 수 있습니다. 

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\mathrm{def}}{=}} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\mathrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\mathrm{def}}{=}}.$$

즉, 이 기울기는 $L-l$ 행렬 $\mathbf{M}^{(L)} \cdot \ldots \cdot \mathbf{M}^{(l+1)}$와 기울기 벡터 $\mathbf{v}^{(l)}$의 곱입니다.따라서 우리는 너무 많은 확률을 곱할 때 종종 발생하는 동일한 수치 언더 플로우 문제에 취약합니다.확률을 다룰 때 일반적인 트릭은 로그 공간으로 전환하는 것입니다. 즉, 가수에서 숫자 표현의 지수로 압력을 이동하는 것입니다.안타깝게도 위의 문제는 더 심각합니다. 처음에는 행렬 $\mathbf{M}^{(l)}$에 다양한 고유값이 있을 수 있습니다.크기가 작거나 클 수 있으며 제품이*매우 큰* 또는*매우 작습니다*. 

불안정한 그래디언트로 인한 위험은 수치 표현을 넘어섭니다.예측할 수 없는 크기의 기울기도 최적화 알고리즘의 안정성을 위협합니다.(i) 지나치게 커서 모델을 파괴 (*폭발하는 그라디언트* 문제) 또는 (ii) 지나치게 작아서 (*사라지는 그라디언트* 문제) 매개 변수가 각 업데이트에서 거의 움직이지 않기 때문에 학습이 불가능한 매개 변수 업데이트에 직면 할 수 있습니다. 

### (**배니싱 그라디언트**)

소실 기울기 문제를 일으키는 빈번한 원인 중 하나는 각 레이어의 선형 연산 다음에 추가되는 활성화 함수 $\sigma$를 선택하는 것입니다.역사적으로 시그모이드 함수 $1/(1 + \exp(-x))$ (:numref:`sec_mlp`에 도입됨) 은 임계값 함수와 유사하기 때문에 널리 사용되었습니다.초기 인공 신경망은 생물학적 신경망에서 영감을 받았기 때문에* 완전히* 또는* 전혀* (생물학적 뉴런과 같이) 발사되는 뉴런에 대한 아이디어는 매력적으로 보였습니다.시그모이드를 자세히 살펴보고 그라디언트가 사라지는 이유를 살펴보겠습니다.

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

보시다시피, (**시그모이드의 기울기는 입력값이 크거나 작을 때 모두 사라집니다**).더욱이 많은 층을 통해 역전파 할 때 많은 시그모이드에 대한 입력이 0에 가까운 Goldilocks 영역에 있지 않으면 전체 곱의 기울기가 사라질 수 있습니다.네트워크가 많은 레이어를 자랑 할 때 조심하지 않으면 일부 레이어에서 그래디언트가 잘릴 수 있습니다.실제로 이 문제는 심층 네트워크 트레이닝을 괴롭히는 데 사용되었습니다.결과적으로 더 안정적이지만 (신경적으로 그럴듯하지는 않음) RELU가 실무자의 기본 선택으로 부상했습니다. 

### [**폭발하는 그라디언트**]

그라디언트가 폭발할 때 반대의 문제도 비슷하게 짜증날 수 있습니다.이를 좀 더 잘 설명하기 위해 100개의 가우스 랜덤 행렬을 그리고 초기 행렬을 곱합니다.우리가 선택한 척도 (분산 $\sigma^2=1$의 선택) 의 경우 행렬 곱이 폭발합니다.심층 네트워크의 초기화로 인해 이런 일이 발생하면 경사하강법 옵티마이저가 수렴할 기회가 없습니다.

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

### 대칭성 깨기

신경망 설계의 또 다른 문제는 매개 변수화에 내재된 대칭성입니다.하나의 은닉 레이어와 두 개의 유닛이 있는 단순한 MLP가 있다고 가정해 보겠습니다.이 경우 첫 번째 레이어의 가중치 $\mathbf{W}^{(1)}$를 순회하고 마찬가지로 출력 레이어의 가중치를 순회하여 동일한 함수를 얻을 수 있습니다.첫 번째 은닉 유닛과 두 번째 히든 유닛을 구분하는 특별한 것은 없습니다.즉, 각 레이어의 은닉 유닛 사이에 순열 대칭이 있습니다. 

이것은 단순한 이론적인 성가신 것 이상입니다.두 개의 은닉 유닛이 있는 앞서 언급한 단일 은닉 레이어 MLP를 생각해 보십시오.예를 들어, 출력 레이어가 숨겨진 두 단위를 하나의 출력 단위로만 변환한다고 가정해 보겠습니다.일부 상수 $c$에 대해 숨겨진 계층의 모든 매개 변수를 $\mathbf{W}^{(1)} = c$로 초기화하면 어떻게 될지 상상해보십시오.이 경우 순방향 전파 중에 숨겨진 장치는 동일한 입력 및 매개 변수를 사용하여 동일한 활성화를 생성하여 출력 장치에 공급됩니다.역전파 중에 매개 변수 $\mathbf{W}^{(1)}$와 관련하여 출력 단위를 차별화하면 요소가 모두 동일한 값을 갖는 기울기가 제공됩니다.따라서 기울기 기반 반복 (예: 미니배치 확률적 경사 하강) 후에도 $\mathbf{W}^{(1)}$의 모든 요소는 여전히 동일한 값을 갖습니다.이러한 반복은 그 자체로는 대칭을 깨뜨리지 않을 것이며 우리는 네트워크의 표현력을 결코 깨닫지 못할 수도 있습니다.숨겨진 레이어는 마치 하나의 단위만 있는 것처럼 동작합니다.미니배치 확률적 경사하강법이 이 대칭을 깨뜨리지는 않지만, 드롭아웃 정규화는 그렇게 될 것입니다! 

## 매개 변수 초기화

위에서 제기된 문제를 해결하거나 최소한 완화하는 한 가지 방법은 신중한 초기화를 이용하는 것입니다.최적화 중 추가 관리 및 적절한 정규화는 안정성을 더욱 향상시킬 수 있습니다. 

### 기본 초기화

이전 섹션 (예: :numref:`sec_linear_concise`) 에서는 정규 분포를 사용하여 가중치 값을 초기화했습니다.초기화 방법을 지정하지 않으면 프레임워크는 기본 임의 초기화 메서드를 사용합니다. 이 메서드는 보통 크기의 문제에 대해 실제로 잘 작동합니다. 

### 자비에르 초기화
:label:`subsec_xavier`

일부 완전 연결 계층에 대한 출력 (예: 숨겨진 변수) $o_{i}$의 스케일 분포를 살펴 보겠습니다.
*비선형성 없이*.
이 계층에 대한 $n_\mathrm{in}$ 입력 $x_j$ 및 관련 가중치 $w_{ij}$을 사용하는 경우 출력은 다음과 같이 지정됩니다. 

$$o_{i} = \sum_{j=1}^{n_\mathrm{in}} w_{ij} x_j.$$

가중치 $w_{ij}$은 모두 동일한 분포에서 독립적으로 그려집니다.또한 이 분포의 평균이 0이고 분산 $\sigma^2$이 있다고 가정해 보겠습니다.이는 분포가 가우스이어야 한다는 의미가 아니라 평균과 분산이 존재해야 한다는 것을 의미합니다.지금은 계층 $x_j$에 대한 입력값도 평균과 분산 $\gamma^2$가 0이고 $w_{ij}$과 독립적이며 서로 독립적이라고 가정해 보겠습니다.이 경우 다음과 같이 $o_i$의 평균과 분산을 계산할 수 있습니다. 

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\mathrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\mathrm{in}} E[w_{ij}] E[x_j] \\&= 0, \\
    \mathrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

분산을 고정하는 한 가지 방법은 $n_\mathrm{in} \sigma^2 = 1$을 설정하는 것입니다.이제 역전파를 고려해 보십시오.출력에 더 가까운 레이어에서 그라디언트가 전파되지만 비슷한 문제가 발생합니다.순방향 전파와 동일한 추론을 사용하여 $n_\mathrm{out} \sigma^2 = 1$가 아니면 그래디언트의 분산이 폭발할 수 있음을 알 수 있습니다. 여기서 $n_\mathrm{out}$는 이 계층의 출력 수입니다.이로 인해 우리는 딜레마에 빠지게됩니다. 두 조건을 동시에 만족시킬 수는 없습니다.대신, 우리는 단순히 다음을 만족시키려고 노력합니다. 

$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}.
\end{aligned}
$$

이것이 제작자 :cite:`Glorot.Bengio.2010`의 첫 번째 저자의 이름을 따서 명명 된 현재 표준적이고 실질적으로 유익한*Xavier 초기화*의 기초가되는 이유입니다.일반적으로 Xavier 초기화는 평균이 0이고 분산이 $\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}}$인 가우스 분포에서 가중치를 샘플링합니다.또한 균일한 분포에서 가중치를 샘플링할 때 Xavier의 직관을 조정하여 분산을 선택할 수 있습니다.균등 분포 $U(-a, a)$에는 분산 $\frac{a^2}{3}$가 있습니다.$\sigma^2$의 조건에 $\frac{a^2}{3}$를 연결하면 다음에 따라 초기화하라는 제안이 생성됩니다. 

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$

위의 수학적 추론에서 비선형성이 존재하지 않는다는 가정은 신경망에서 쉽게 위반될 수 있지만 Xavier 초기화 방법은 실제로 잘 작동하는 것으로 나타났습니다. 

### 너머

위의 이유는 매개 변수 초기화에 대한 현대적인 접근 방식의 표면을 거의 긁지 않습니다.딥 러닝 프레임워크는 십여 가지가 넘는 휴리스틱을 구현하는 경우가 많습니다.또한 파라미터 초기화는 딥 러닝의 기초 연구의 뜨거운 영역입니다.여기에는 연결된 (공유) 매개 변수, 초 고해상도, 시퀀스 모델 및 기타 상황에 특화된 휴리스틱이 있습니다.예를 들어, Xiao et al. 은 신중하게 설계된 초기화 방법 :cite:`Xiao.Bahri.Sohl-Dickstein.ea.2018`를 사용하여 아키텍처 트릭 없이 1000층 신경망을 훈련할 수 있는 가능성을 보여주었습니다. 

주제에 관심이 있다면 이 모듈의 오퍼링을 자세히 살펴보고 각 휴리스틱을 제안하고 분석한 논문을 읽은 다음 주제에 대한 최신 간행물을 탐색하는 것이 좋습니다.아마도 당신은 우연히 발견되거나 영리한 아이디어를 발명하고 딥 러닝 프레임 워크에 구현에 기여할 것입니다. 

## 요약

* 그라데이션 소멸 및 폭발은 심층 네트워크에서 흔히 발생하는 문제입니다.그래디언트 및 매개 변수를 잘 제어하려면 매개 변수 초기화에 세심한주의가 필요합니다.
* 초기 그래디언트가 너무 크거나 작지 않도록 하려면 초기화 휴리스틱이 필요합니다.
* ReLU 활성화 함수는 사라지는 기울기 문제를 완화합니다.이는 수렴을 가속화할 수 있습니다.
* 무작위 초기화는 최적화 전에 대칭이 깨지도록 하는 열쇠입니다.
* Xavier 초기화는 각 계층에 대해 출력의 분산이 입력 개수의 영향을 받지 않으며 기울기의 분산이 출력값 개수의 영향을 받지 않음을 나타냅니다.

## 연습문제

1. 신경망이 MLP 계층에서 순열 대칭 외에 파괴가 필요한 대칭을 나타낼 수 있는 다른 경우를 설계할 수 있습니까?
1. 선형 회귀 또는 소프트맥스 회귀에서 모든 가중치 파라미터를 동일한 값으로 초기화할 수 있습니까?
1. 두 행렬의 곱에 대한 고유값에 대한 분석적 한계를 찾습니다.그래디언트가 잘 조절되도록 하는 것에 대해 무엇을 알려줍니까?
1. 일부 용어가 서로 다르다는 것을 알고 있다면 사실 이후에 이 문제를 해결할 수 있을까요?영감을 얻기 위해 레이어별 적응형 속도 스케일링에 대한 논문을 살펴보십시오. :cite:`You.Gitman.Ginsburg.2017`.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/235)
:end_tab:
