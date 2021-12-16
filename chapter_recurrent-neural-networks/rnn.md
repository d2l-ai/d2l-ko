# 순환 신경망
:label:`sec_rnn`

:numref:`sec_language_model`에서는 $n$그램 모델을 도입했습니다. 여기서 시간 단계 $t$에서 단어 $x_t$의 조건부 확률은 $n-1$ 이전 단어에만 의존합니다.$x_t$에 시간 단계 $t-(n-1)$보다 이른 단어의 가능한 효과를 통합하려면 $n$을 늘려야 합니다.그러나 어휘 집합 $\mathcal{V}$에 대해 $|\mathcal{V}|^n$ 숫자를 저장해야하므로 모델 매개 변수의 수도 기하 급수적으로 증가합니다.따라서 $P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$을 모델링하는 대신 잠재 변수 모델을 사용하는 것이 좋습니다. 

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

여기서 $h_{t-1}$는 시간 단계 $t-1$까지의 시퀀스 정보를 저장하는*숨겨진 상태* (숨겨진 변수라고도 함) 입니다.일반적으로 언제든지 단계 $t$의 숨겨진 상태는 현재 입력 $x_{t}$과 이전의 숨겨진 상태 $h_{t-1}$를 기반으로 계산할 수 있습니다. 

$$h_t = f(x_{t}, h_{t-1}).$$
:eqlabel:`eq_ht_xt`

:eqref:`eq_ht_xt`에서 충분히 강력한 함수 $f$의 경우 잠재 변수 모델은 근사가 아닙니다.결국 $h_t$은 지금까지 관찰한 모든 데이터를 저장할 수 있습니다.그러나 계산과 스토리지 모두 비용이 많이 들 수 있습니다. 

:numref:`chap_perceptrons`에서 은닉 유닛이 있는 은닉 레이어에 대해 논의한 적이 있습니다.숨겨진 레이어와 숨겨진 상태는 매우 다른 두 가지 개념을 나타냅니다.숨겨진 레이어는 설명된 대로 입력에서 출력까지 경로의 뷰에서 숨겨진 레이어입니다.숨겨진 상태는 주어진 단계에서 수행하는 모든 작업에 대해 기술적으로*입력*을 말하며 이전 시간 단계의 데이터를 살펴보아야 만 계산할 수 있습니다. 

*순환 신경망* (RNN) 은 숨겨진 상태를 가진 신경망입니다.RNN 모델을 도입하기 전에 먼저 :numref:`sec_mlp`에 도입된 MLP 모델을 다시 살펴봅니다.

## 은닉 스테이트가 없는 신경망

숨겨진 레이어가 하나뿐인 MLP를 살펴보겠습니다.숨겨진 레이어의 활성화 함수를 $\phi$로 설정합니다.배치 크기가 $n$ 및 $d$ 입력인 예제 $\mathbf{X} \in \mathbb{R}^{n \times d}$의 미니배치가 주어진 경우, 은닉 레이어의 출력값 $\mathbf{H} \in \mathbb{R}^{n \times h}$은 다음과 같이 계산됩니다. 

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h).$$
:eqlabel:`rnn_h_without_state`

:eqref:`rnn_h_without_state`에는 은닉 레이어에 대한 가중치 파라미터 $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$, 바이어스 파라미터 $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$ 및 은닉 유닛 수 $h$가 있습니다.따라서 합산 중에 방송 (:numref:`subsec_broadcasting` 참조) 이 적용됩니다.다음으로, 은닉 변수 $\mathbf{H}$이 출력 계층의 입력으로 사용됩니다.출력 계층은 다음과 같이 지정됩니다. 

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q,$$

여기서 $\mathbf{O} \in \mathbb{R}^{n \times q}$은 출력 변수이고, $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$는 가중치 매개변수이며, $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$은 출력 계층의 편향 매개변수입니다.분류 문제인 경우 $\text{softmax}(\mathbf{O})$를 사용하여 출력 범주의 확률 분포를 계산할 수 있습니다. 

이는 이전에 :numref:`sec_sequence`에서 해결한 회귀 문제와 완전히 유사하므로 세부 사항을 생략합니다.특징-레이블 쌍을 무작위로 선택하고 자동 미분 및 확률 적 경사 하강을 통해 네트워크의 매개 변수를 학습 할 수 있다고 말하면 충분합니다. 

## 은닉 스테이트를 갖는 순환 신경망
:label:`subsec_rnn_w_hidden_states`

숨겨진 상태가 있으면 문제가 완전히 다릅니다.구조를 좀 더 자세히 살펴 보겠습니다. 

시간 단계 $t$에 입력 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$의 미니배치가 있다고 가정합니다.즉, $n$ 시퀀스 예제의 미니배치의 경우 $\mathbf{X}_t$의 각 행은 시퀀스의 시간 단계 $t$에 있는 하나의 예에 해당합니다.다음으로 시간 단계 $t$의 숨겨진 변수를 $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$로 나타냅니다.MLP와 달리 여기서는 이전 시간 단계의 숨겨진 변수 $\mathbf{H}_{t-1}$를 저장하고 새 가중치 매개 변수 $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$을 도입하여 현재 시간 단계에서 이전 시간 단계의 숨겨진 변수를 사용하는 방법을 설명합니다.구체적으로, 현재 시간 스텝의 은닉 변수 계산은 이전 시간 스텝의 히든 변수와 함께 현재 시간 스텝의 입력에 의해 결정됩니다. 

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$
:eqlabel:`rnn_h_with_state`

:eqref:`rnn_h_without_state`와 비교하여 :eqref:`rnn_h_with_state`는 용어 $\mathbf{H}_{t-1} \mathbf{W}_{hh}$을 하나 더 추가하여 :eqref:`eq_ht_xt`을 인스턴스화합니다.인접한 시간 스텝의 은닉 변수 $\mathbf{H}_t$와 $\mathbf{H}_{t-1}$ 사이의 관계에서 이러한 변수가 신경망의 현재 시간 스텝의 상태 또는 메모리와 마찬가지로 시퀀스의 과거 정보를 캡처하고 현재 시간 스텝까지 유지한다는 것을 알 수 있습니다.따라서 이러한 숨겨진 변수를*숨겨진 상태*라고 합니다.은닉 상태는 현재 시간 스텝의 이전 시간 스텝과 동일한 정의를 사용하므로 :eqref:`rnn_h_with_state`의 계산은*recurrent*입니다.따라서 반복 계산을 기반으로 숨겨진 상태를 가진 신경망의 이름이 지정됩니다.
*순환 신경망*.
RNN에서 :eqref:`rnn_h_with_state`의 계산을 수행하는 계층을 *순환 계층*이라고 합니다. 

RNN을 구성하는 방법에는 여러 가지가 있습니다.:eqref:`rnn_h_with_state`로 정의된 숨겨진 상태의 RNN은 매우 일반적입니다.시간 스텝 $t$의 경우 출력 계층의 출력은 MLP의 계산과 유사합니다. 

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

RNN의 파라미터는 출력 레이어의 가중치 ($\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$) 및 바이어스 ($\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$) 및 바이어스 ($\mathbf{b}_q \in \mathbb{R}^{1 \times q}$) 와 함께 은닉 레이어의 가중치 ($\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$) 및 바이어스 ($\mathbf{b}_h \in \mathbb{R}^{1 \times h}$) 를 포함한다.다른 시간 단계에서도 RNN은 항상 이러한 모델 매개 변수를 사용한다는 점을 언급 할 가치가 있습니다.따라서 시간 단계 수가 증가해도 RNN의 매개 변수화 비용은 증가하지 않습니다. 

:numref:`fig_rnn`는 인접한 세 시간 단계에서 RNN의 계산 논리를 보여줍니다.임의의 시간 스텝 ($t$) 에서, 은닉 상태의 계산은 (i) 현재 시간 스텝 ($t$) 에서의 입력 ($\mathbf{X}_t$) 과 이전 시간 스텝 ($t-1$) 에서의 은닉 상태 ($\mathbf{H}_{t-1}$) 를 연결하는 단계; (ii) 활성화와 함께 연결 결과를 완전히 연결된 계층으로 공급하는 것으로 취급될 수 있다.함수 $\phi$입니다.이러한 완전 연결 계층의 출력은 현재 시간 스텝 $t$의 숨겨진 상태 $\mathbf{H}_t$입니다.이 경우 모형 모수는 $\mathbf{W}_{xh}$와 $\mathbf{W}_{hh}$의 연결이고 치우침은 $\mathbf{b}_h$이며, 모두 :eqref:`rnn_h_with_state`입니다.현재 시간 스텝 $t$, $\mathbf{H}_t$의 숨겨진 상태는 다음 시간 스텝 $t+1$의 숨겨진 상태 $\mathbf{H}_{t+1}$을 계산하는 데 참여할 것이다.또한 $\mathbf{H}_t$는 현재 시간 스텝 $t$의 출력 $\mathbf{O}_t$를 계산하기 위해 완전히 연결된 출력 계층에 공급됩니다. 

![An RNN with a hidden state.](../img/rnn.svg)
:label:`fig_rnn`

우리는 방금 숨겨진 상태에 대한 $\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$의 계산이 $\mathbf{X}_t$과 $\mathbf{H}_{t-1}$의 연결의 행렬 곱셈과 $\mathbf{W}_{xh}$와 $\mathbf{W}_{hh}$의 결합과 동일하다고 언급했습니다.이것은 수학에서 입증될 수 있지만, 다음에서는 간단한 코드 스니펫을 사용하여 이를 보여줍니다.먼저 행렬 `X`, `W_xh`, `H` 및 `W_hh`을 정의합니다. 이 행렬의 모양은 각각 (3, 1), (1, 4), (3, 4) 및 (4, 4) 입니다.`X`에 `W_xh`를 곱하고 `H`에 각각 `W_hh`을 곱한 다음이 두 곱셈을 더하면 모양의 행렬 (3, 4) 을 얻습니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab mxnet, pytorch
X, W_xh = d2l.normal(0, 1, (3, 1)), d2l.normal(0, 1, (1, 4))
H, W_hh = d2l.normal(0, 1, (3, 4)), d2l.normal(0, 1, (4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
#@tab tensorflow
X, W_xh = d2l.normal((3, 1), 0, 1), d2l.normal((1, 4), 0, 1)
H, W_hh = d2l.normal((3, 4), 0, 1), d2l.normal((4, 4), 0, 1)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

이제 행렬 `X`와 `H`를 열 (축 1) 을 따라 연결하고 행렬 `W_xh`과 `W_hh`을 행 (축 0) 을 따라 연결합니다.이 두 연결은 각각 형상 (3, 5) 과 형상 (5, 4) 의 행렬을 생성합니다.이 두 개의 연결된 행렬을 곱하면 위와 동일한 모양의 출력 행렬 (3, 4) 을 얻습니다.

```{.python .input}
#@tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## RNN 기반 문자 수준 언어 모델

:numref:`sec_language_model`의 언어 모델링의 경우 현재 및 과거 토큰을 기반으로 다음 토큰을 예측하는 것을 목표로하므로 원래 시퀀스를 레이블로 하나의 토큰으로 이동합니다.Bengio et al. 은 언어 모델링을 위해 신경망을 사용할 것을 처음 제안했습니다.다음에서는 RNN을 사용하여 언어 모델을 구축하는 방법을 보여줍니다.미니 배치 크기를 1로 설정하고 텍스트 시퀀스를 “machine”으로 지정합니다.후속 섹션의 교육을 단순화하기 위해 텍스트를 단어가 아닌 문자로 토큰화하고*문자 수준 언어 모델*을 고려합니다. :numref:`fig_rnn_train`는 문자 수준 언어 모델링을 위해 RNN을 통해 현재 및 이전 문자를 기반으로 다음 문자를 예측하는 방법을 보여줍니다. 

![A character-level language model based on the RNN. The input and label sequences are "machin" and "achine", respectively.](../img/rnn-train.svg)
:label:`fig_rnn_train`

훈련 프로세스 중에 각 시간 스텝에 대해 출력 계층의 출력에 대해 softmax 연산을 실행 한 다음 교차 엔트로피 손실을 사용하여 모델 출력과 레이블 간의 오차를 계산합니다.은닉 레이어에서 은닉 상태의 반복 계산으로 인해 :numref:`fig_rnn_train`, $\mathbf{O}_3$의 시간 단계 3의 출력은 텍스트 시퀀스 “m”, “a” 및 “c”에 의해 결정됩니다.훈련 데이터에서 시퀀스의 다음 문자가 “h”이므로 시간 단계 3의 손실은 이 시간 단계의 특징 시퀀스 “m”, “a”, “c” 및 레이블 “h”를 기반으로 생성된 다음 문자의 확률 분포에 따라 달라집니다. 

실제로 각 토큰은 $d$차원 벡터로 표시되며 배치 크기 $n>1$을 사용합니다.따라서 시간 단계 $t$의 입력 $\mathbf X_t$은 $n\times d$ 행렬이 되며, 이는 :numref:`subsec_rnn_w_hidden_states`에서 논의한 것과 동일합니다. 

## 당혹
:label:`subsec_perplexity`

마지막으로, 후속 섹션에서 RNN 기반 모델을 평가하는 데 사용될 언어 모델 품질을 측정하는 방법에 대해 설명하겠습니다.한 가지 방법은 텍스트가 얼마나 놀라운 지 확인하는 것입니다.좋은 언어 모델은 다음에 보게 될 내용을 높은 정확도의 토큰으로 예측할 수 있습니다.다른 언어 모델에서 제안한 것처럼 “비가 내리고 있습니다”라는 문구의 다음과 같은 연속을 고려하십시오. 

1. “밖에 비가 내린다”
1. “비가 내리는 바나나 나무”
1. “비가 내리고 있습니다. kcj pepoiut”

품질면에서 예제 1이 분명히 최고입니다.단어는 합리적이고 논리적으로 일관성이 있습니다.의미 론적으로 따르는 단어를 정확하게 반영하지는 못할 수도 있지만 (“샌프란시스코에서”와 “겨울”은 완벽하게 합리적인 확장이었을 것입니다) 모델은 어떤 종류의 단어가 뒤 따르는지 포착 할 수 있습니다.예 2는 무의미한 확장을 생성하면 상당히 나쁩니다.그럼에도 불구하고 적어도 모델은 단어의 철자를 쓰는 방법과 단어 간의 어느 정도의 상관 관계를 배웠습니다.마지막으로, 예 3은 데이터를 제대로 피팅하지 못하는 제대로 훈련되지 않은 모델을 나타냅니다. 

시퀀스의 가능성을 계산하여 모델의 품질을 측정할 수 있습니다.불행히도 이것은 이해하기 어렵고 비교하기 어려운 숫자입니다.결국 긴 시퀀스보다 짧은 시퀀스가 발생할 가능성이 훨씬 높으므로 톨스토이의 매그넘 오푸스에서 모델을 평가합니다.
*전쟁과 평화*는 필연적으로 생텍쥐페리의 소설*어린 왕자*보다 훨씬 작은 가능성을 만들어 낼 것입니다.빠진 것은 평균과 같습니다.

정보 이론이 여기에 유용합니다.소프트맥스 회귀 (:numref:`subsec_info_theory_basics`) 를 도입했을 때 엔트로피, 놀라움 및 교차 엔트로피를 정의했으며 [online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html)에서 더 많은 정보 이론에 대해 논의합니다.텍스트를 압축하려면 현재 토큰 집합이 주어진 다음 토큰을 예측하는 방법을 물어볼 수 있습니다.더 나은 언어 모델을 통해 다음 토큰을 더 정확하게 예측할 수 있어야 합니다.따라서 시퀀스를 압축하는 데 더 적은 비트를 사용할 수 있습니다.따라서 시퀀스의 모든 $n$ 토큰에 대한 평균 교차 엔트로피 손실로 측정 할 수 있습니다. 

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
:eqlabel:`eq_avg_ce_for_lm`

여기서 $P$는 언어 모델에 의해 주어지고 $x_t$은 시퀀스의 시간 단계 $t$에서 관찰되는 실제 토큰입니다.따라서 길이가 다른 문서의 성능을 비교할 수 있습니다.역사적 이유로 자연어 처리 분야의 과학자들은*perplexity*라는 양을 사용하는 것을 선호합니다.간단히 말해서, 그것은 :eqref:`eq_avg_ce_for_lm`의 지수입니다. 

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$

당혹성은 다음에 선택할 토큰을 결정할 때 실제 선택 횟수의 조화 평균으로 가장 잘 이해할 수 있습니다.몇 가지 사례를 살펴 보겠습니다. 

* 최상의 시나리오에서 모델은 항상 레이블 토큰의 확률을 1로 완벽하게 추정합니다.이 경우 모델의 난처함은 1입니다.
* 최악의 시나리오에서 모델은 항상 레이블 토큰의 확률을 0으로 예측합니다.이 상황에서 당혹감은 양의 무한대입니다.
* 기준선에서 모델은 어휘의 사용 가능한 모든 토큰에 대해 균등 분포를 예측합니다.이 경우 혼란은 어휘의 고유 토큰 수와 같습니다.실제로 압축하지 않고 시퀀스를 저장한다면 인코딩할 수 있는 최선의 방법이 될 것입니다.따라서 이는 유용한 모형이 이겨야 하는 중요하지 않은 상한을 제공합니다.

다음 섹션에서는 문자 수준 언어 모델에 대한 RNN을 구현하고 이러한 모델을 평가하기 위해 perplexity를 사용합니다. 

## 요약

* 은닉 상태에 대해 반복 계산을 사용하는 신경망을 순환 신경망 (RNN) 이라고 합니다.
* RNN의 숨겨진 상태는 현재 시간 스텝까지 시퀀스의 히스토리 정보를 캡처할 수 있습니다.
* 시간 단계 수가 증가해도 RNN 모델 모수의 수는 증가하지 않습니다.
* RNN을 사용하여 문자 수준 언어 모델을 만들 수 있습니다.
* 당혹감을 사용하여 언어 모델의 품질을 평가할 수 있습니다.

## 연습문제

1. RNN을 사용하여 텍스트 시퀀스의 다음 문자를 예측하는 경우 출력에 필요한 차원은 무엇입니까?
1. RNN이 텍스트 시퀀스의 모든 이전 토큰을 기반으로 특정 시간 단계에서 토큰의 조건부 확률을 표현할 수 있는 이유는 무엇입니까?
1. 긴 시퀀스를 통해 역전파하면 그래디언트는 어떻게 됩니까?
1. 이 섹션에서 설명하는 언어 모델과 관련된 몇 가지 문제점은 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/337)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1050)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1051)
:end_tab:
