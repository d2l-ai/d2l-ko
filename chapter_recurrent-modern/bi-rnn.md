# 양방향 순환 신경망
:label:`sec_bi_rnn`

시퀀스 학습에서 지금까지 우리는 시계열의 맥락이나 언어 모델의 맥락에서 지금까지 본 것을 고려할 때 다음 출력을 모델링하는 것이 목표라고 가정했습니다.이것이 일반적인 시나리오이지만 우리가 접할 수 있는 유일한 시나리오는 아닙니다.이 문제를 설명하기 위해 텍스트 시퀀스에서 빈칸을 채우는 다음 세 가지 작업을 고려하십시오. 

* 저는 `___`입니다.
* 저는 `___`가 배고프다.
* 저는 `___`가 배고프고 돼지 반을 먹을 수 있습니다.

사용 가능한 정보의 양에 따라 “happy”, “not”, “very”와 같이 매우 다른 단어로 빈칸을 채울 수 있습니다.분명히 문구의 끝 부분 (사용 가능한 경우) 은 어떤 단어를 선택해야하는지에 대한 중요한 정보를 전달합니다.이 기능을 활용할 수 없는 시퀀스 모델은 관련 작업에서 제대로 수행되지 않습니다.예를 들어, 명명된 엔터티 인식 (예: “녹색”이 “Mr. Green”을 의미하는지 또는 색상을 의미하는지 인식) 을 잘 수행하려면 더 긴 범위의 컨텍스트도 똑같이 중요합니다.문제를 해결하기 위한 영감을 얻기 위해 확률적 그래픽 모델로 우회해 보겠습니다. 

## 히든 마르코프 모델의 동적 프로그래밍

이 하위 섹션에서는 동적 프로그래밍 문제를 설명합니다.구체적인 기술적 세부 사항은 딥 러닝 모델을 이해하는 데 중요하지 않지만 딥 러닝을 사용하는 이유와 특정 아키텍처를 선택하는 이유에 동기를 부여하는 데 도움이됩니다. 

확률적 그래픽 모델을 사용하여 문제를 해결하려면 예를 들어 다음과 같이 잠재 변수 모델을 설계할 수 있습니다.언제든지 단계 $t$에서 $P(x_t \mid h_t)$를 통해 관측된 배출량 $x_t$을 제어하는 잠복 변수 $h_t$이 존재한다고 가정합니다.또한, 임의의 전이 $h_t \to h_{t+1}$는 일부 상태 전이 확률 $P(h_{t+1} \mid h_{t})$에 의해 주어진다.이 확률적 그래픽 모델은 :numref:`fig_hmm`에서와 같이*숨겨진 마르코프 모델*입니다. 

![A hidden Markov model.](../img/hmm.svg)
:label:`fig_hmm`

따라서 $T$개의 관측치에 대해 관측된 상태와 숨겨진 상태에 대해 다음과 같은 공동 확률 분포가 있습니다. 

$$P(x_1, \ldots, x_T, h_1, \ldots, h_T) = \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t), \text{ where } P(h_1 \mid h_0) = P(h_1).$$
:eqlabel:`eq_hmm_jointP`

이제 $x_j$을 제외한 모든 $x_i$을 관찰하고 $P(x_j \mid x_{-j})$를 계산하는 것이 우리의 목표라고 가정합니다. 여기서 $x_{-j} = (x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_{T})$입니다.$P(x_j \mid x_{-j})$에는 잠재 변수가 없기 때문에 $h_1, \ldots, h_T$에 대해 가능한 모든 선택 조합을 합산하는 것을 고려합니다.$h_i$이 $k$의 고유 값 (유한 수의 상태) 을 취할 수 있는 경우, 이는 $k^T$ 개 이상의 용어를 합산해야 함을 의미합니다. 일반적으로 임무가 불가능합니다!다행히도 이를 위한 우아한 해결책이 있습니다: *동적 프로그래밍*. 

작동 방식을 확인하려면 잠재 변수 $h_1, \ldots, h_T$를 차례로 합산하는 것이 좋습니다.:eqref:`eq_hmm_jointP`에 따르면 다음과 같은 결과를 얻을 수 있습니다. 

$$\begin{aligned}
    &P(x_1, \ldots, x_T) \\
    =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_2, \ldots, h_T} \underbrace{\left[\sum_{h_1} P(h_1) P(x_1 \mid h_1) P(h_2 \mid h_1)\right]}_{\pi_2(h_2) \stackrel{\mathrm{def}}{=}}
    P(x_2 \mid h_2) \prod_{t=3}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_3, \ldots, h_T} \underbrace{\left[\sum_{h_2} \pi_2(h_2) P(x_2 \mid h_2) P(h_3 \mid h_2)\right]}_{\pi_3(h_3)\stackrel{\mathrm{def}}{=}}
    P(x_3 \mid h_3) \prod_{t=4}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t)\\
    =& \dots \\
    =& \sum_{h_T} \pi_T(h_T) P(x_T \mid h_T).
\end{aligned}$$

일반적으로*순방향 재귀*는 다음과 같습니다. 

$$\pi_{t+1}(h_{t+1}) = \sum_{h_t} \pi_t(h_t) P(x_t \mid h_t) P(h_{t+1} \mid h_t).$$

재귀는 $\pi_1(h_1) = P(h_1)$으로 초기화됩니다.추상적인 용어로 이것은 $\pi_{t+1} = f(\pi_t, x_t)$로 쓸 수 있으며, 여기서 $f$는 학습 가능한 함수입니다.이것은 RNN의 맥락에서 지금까지 논의한 잠재 변수 모델의 업데이트 방정식과 매우 흡사합니다!  

순방향 재귀와 완전히 유사하게 역방향 재귀를 사용하여 동일한 잠재 변수 집합을 합칠 수도 있습니다.이 결과는 다음과 같습니다 

$$\begin{aligned}
    & P(x_1, \ldots, x_T) \\
     =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot P(h_T \mid h_{T-1}) P(x_T \mid h_T) \\
    =& \sum_{h_1, \ldots, h_{T-1}} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_T} P(h_T \mid h_{T-1}) P(x_T \mid h_T)\right]}_{\rho_{T-1}(h_{T-1})\stackrel{\mathrm{def}}{=}} \\
    =& \sum_{h_1, \ldots, h_{T-2}} \prod_{t=1}^{T-2} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_{T-1}} P(h_{T-1} \mid h_{T-2}) P(x_{T-1} \mid h_{T-1}) \rho_{T-1}(h_{T-1}) \right]}_{\rho_{T-2}(h_{T-2})\stackrel{\mathrm{def}}{=}} \\
    =& \ldots \\
    =& \sum_{h_1} P(h_1) P(x_1 \mid h_1)\rho_{1}(h_{1}).
\end{aligned}$$

따라서*역방향 재귀*를 다음과 같이 쓸 수 있습니다. 

$$\rho_{t-1}(h_{t-1})= \sum_{h_{t}} P(h_{t} \mid h_{t-1}) P(x_{t} \mid h_{t}) \rho_{t}(h_{t}),$$

초기화 $\rho_T(h_T) = 1$를 사용합니다.순방향 및 역방향 재귀 모두 지수 시간이 아닌 $(h_1, \ldots, h_T)$의 모든 값에 대해 $\mathcal{O}(kT)$ (선형) 시간의 $T$ 개 이상의 잠재 변수를 합산할 수 있습니다.이는 그래픽 모델을 사용한 확률적 추론의 큰 이점 중 하나입니다.또한 일반적인 메시지 전달 알고리즘 :cite:`Aji.McEliece.2000`의 매우 특별한 예이기도 합니다.순방향 재귀와 역방향 재귀를 결합하여 다음을 계산할 수 있습니다. 

$$P(x_j \mid x_{-j}) \propto \sum_{h_j} \pi_j(h_j) \rho_j(h_j) P(x_j \mid h_j).$$

추상적인 용어로 역방향 재귀는 $\rho_{t-1} = g(\rho_t, x_t)$으로 작성할 수 있습니다. 여기서 $g$는 학습 가능한 함수입니다.다시 말하지만, 이것은 업데이트 방정식과 매우 흡사합니다. 지금까지 RNN에서 본 것과는 달리 거꾸로 실행됩니다.실제로 숨겨진 마르코프 모델은 향후 데이터가 사용 가능할 때 알 수 있다는 이점이 있습니다.신호 처리 과학자들은 미래의 관측치를 보간 대 외삽으로 아는 것과 알지 못하는 두 가지 경우를 구별합니다.:cite:`Doucet.De-Freitas.Gordon.2001`에 대한 자세한 내용은 순차 몬테카를로 알고리즘에 대한 책의 소개 장을 참조하십시오. 

## 양방향 모델

숨겨진 마르코프 모델에서와 비슷한 예측 기능을 제공하는 RNN에 메커니즘을 갖고 싶다면 지금까지 본 RNN 설계를 수정해야 합니다.다행히도 이것은 개념적으로 쉽습니다.첫 번째 토큰에서 시작하여 정방향 모드에서만 RNN을 실행하는 대신, 뒤에서 앞으로 실행되는 마지막 토큰에서 다른 토큰을 시작합니다. 
*양방향 RNNS*는 정보를 역방향으로 전달하는 은닉 계층을 추가하여 이러한 정보를보다 유연하게 처리합니다. :numref:`fig_birnn`는 단일 은닉 계층이있는 양방향 RNN의 아키텍처를 보여줍니다.

![Architecture of a bidirectional RNN.](../img/birnn.svg)
:label:`fig_birnn`

실제로 이것은 숨겨진 마르코프 모델의 동적 프로그래밍에서 순방향 및 역방향 재귀와 크게 다르지 않습니다.가장 큰 차이점은 앞의 경우 이러한 방정식이 특정 통계적 의미를 가졌다는 것입니다.이제 그들은 쉽게 접근 할 수있는 해석이 없으며 일반적이고 학습 가능한 함수로 취급 할 수 있습니다.이러한 전환은 현대 심층 네트워크의 설계를 안내하는 많은 원칙을 요약합니다. 먼저 고전적인 통계 모델의 기능적 종속성 유형을 사용한 다음 일반적인 형식으로 매개 변수화합니다. 

### 정의

양방향 RNN은 :cite:`Schuster.Paliwal.1997`에 의해 도입되었습니다.다양한 아키텍처에 대한 자세한 내용은 :cite:`Graves.Schmidhuber.2005` 백서를 참조하십시오.이러한 네트워크의 세부 사항을 살펴 보겠습니다. 

임의의 시간 단계 $t$에 대해 미니배치 입력 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (예제 수: $n$, 각 예제의 입력값 수: $d$) 이 주어지고 은닉 레이어 활성화 함수를 $\phi$로 설정합니다.양방향 아키텍처에서는 이 시간 스텝의 정방향 및 역방향 은닉 상태가 각각 $\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$ 및 $\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$이라고 가정합니다. 여기서 $h$는 은닉 유닛의 수입니다.앞으로 및 뒤로 숨김 상태 업데이트는 다음과 같습니다. 

$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)}),
\end{aligned}
$$

여기서 가중치 $\mathbf{W}_{xh}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{xh}^{(b)} \in \mathbb{R}^{d \times h}, \text{ and } \mathbf{W}_{hh}^{(b)} \in \mathbb{R}^{h \times h}$와 편향 $\mathbf{b}_h^{(f)} \in \mathbb{R}^{1 \times h} \text{ and } \mathbf{b}_h^{(b)} \in \mathbb{R}^{1 \times h}$는 모두 모형 모수입니다. 

다음으로, 출력 계층으로 공급할 숨겨진 상태 $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$을 얻기 위해 정방향 및 역방향 숨겨진 상태 $\overrightarrow{\mathbf{H}}_t$ 및 $\overleftarrow{\mathbf{H}}_t$를 연결합니다.은닉 레이어가 여러 개인 심층 양방향 RNN에서 이러한 정보는*input*으로 다음 양방향 계층으로 전달됩니다.마지막으로 출력 계층은 출력 $\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (출력 수: $q$) 를 계산합니다. 

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

여기서 가중치 행렬 $\mathbf{W}_{hq} \in \mathbb{R}^{2h \times q}$ 및 바이어스 $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$는 출력 계층의 모델 파라미터입니다.실제로 두 방향은 서로 다른 수의 숨겨진 단위를 가질 수 있습니다. 

### 계산 비용 및 응용

양방향 RNN의 주요 특징 중 하나는 시퀀스의 양쪽 끝의 정보가 출력을 추정하는 데 사용된다는 것입니다.즉, 미래 관측과 과거 관측치의 정보를 사용하여 현재 관측치를 예측합니다.다음 토큰 예측의 경우 이것은 우리가 원하는 것이 아닙니다.결국 우리는 다음 토큰을 예측할 때 다음 토큰을 알 수있는 사치가 없습니다.따라서 양방향 RNN을 순진하게 사용한다면 정확도가 높지 않을 것입니다. 훈련 중에 현재를 추정하기 위해 과거와 미래의 데이터가 있습니다.테스트 시간 동안 과거 데이터 만 있으므로 정확도가 떨어집니다.아래 실험에서 이에 대해 설명하겠습니다. 

부상에 대한 모욕을 더하기 위해 양방향 RNN도 매우 느립니다.그 주된 이유는 순방향 전파가 양방향 계층에서 순방향 및 역방향 재귀를 모두 필요로하고 역 전파가 순방향 전파의 결과에 의존하기 때문입니다.따라서 그래디언트는 매우 긴 종속성 체인을 갖게 됩니다. 

실제로 양방향 계층은 누락된 단어 채우기, 토큰 주석 달기 (예: 명명된 엔티티 인식) 및 시퀀스 처리 파이프라인의 한 단계로서 도매 시퀀스 인코딩 (예: 기계 번역) 과 같은 좁은 응용 프로그램 집합에만 매우 드물게 사용됩니다.:numref:`sec_bert` 및 :numref:`sec_sentiment_rnn`에서는 양방향 RNN을 사용하여 텍스트 시퀀스를 인코딩하는 방법을 소개합니다. 

## (**잘못된 애플리케이션에 대한 양방향 RNN 훈련**)

양방향 RNN이 과거와 미래의 데이터를 사용하고 단순히 언어 모델에 적용한다는 사실에 관한 모든 조언을 무시한다면 수용 가능한 혼란으로 추정치를 얻을 수 있습니다.그럼에도 불구하고 아래 실험에서 알 수 있듯이 미래 토큰을 예측하는 모델의 능력은 심각하게 손상됩니다.합리적인 혼란에도 불구하고 많은 반복 후에도 횡설수설이 발생합니다.아래 코드는 잘못된 컨텍스트에서 사용하는 것에 대한 경고 예제로 포함되어 있습니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import npx
from mxnet.gluon import rnn
npx.set_np()

# Load data
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Define the bidirectional LSTM model by setting `bidirectional=True`
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
lstm_layer = rnn.LSTM(num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
# Train the model
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Load data
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Define the bidirectional LSTM model by setting `bidirectional=True`
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# Train the model
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

위에서 설명한 이유로 출력이 분명히 만족스럽지 않습니다.양방향 RNN의 보다 효과적인 사용에 대한 논의는 :numref:`sec_sentiment_rnn`의 감성 분석 애플리케이션을 참조하십시오. 

## 요약

* 양방향 RNN에서, 각 시간 스텝에 대한 은닉 상태는 현재 시간 스텝 이전 및 이후의 데이터에 의해 동시에 결정된다.
* 양방향 RNN은 확률적 그래픽 모델에서 순방향 역방향 알고리즘과 매우 유사합니다.
* 양방향 RNN은 시퀀스 인코딩 및 양방향 컨텍스트가 주어진 관측치 추정에 주로 유용합니다.
* 양방향 RNN은 긴 그래디언트 체인으로 인해 훈련하는 데 매우 많은 비용이 듭니다.

## 연습문제

1. 서로 다른 방향에서 다른 개수의 은닉 유닛을 사용하는 경우 $\mathbf{H}_t$의 모양은 어떻게 변경됩니까?
1. 은닉 레이어가 여러 개인 양방향 RNN을 설계합니다.
1. 다산증은 자연어에서 흔히 볼 수 있습니다.예를 들어, “은행”이라는 단어는 “현금을 입금하기 위해 은행에 갔다”와 “은행에 가서 앉았다”는 맥락에서 다른 의미를 갖습니다.컨텍스트 시퀀스와 단어가 주어지면 컨텍스트에서 단어의 벡터 표현이 반환되도록 신경망 모델을 어떻게 설계 할 수 있습니까?다혈을 처리하기 위해 어떤 유형의 신경 아키텍처가 선호됩니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/339)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1059)
:end_tab:
