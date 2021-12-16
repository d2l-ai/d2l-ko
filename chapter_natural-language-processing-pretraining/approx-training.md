# 근사 교육
:label:`sec_approx_train`

:numref:`sec_word2vec`에서 우리의 토론을 상기하십시오.스킵 그램 모델의 주요 아이디어는 소프트맥스 연산을 사용하여 :eqref:`eq_skip-gram-softmax`에서 주어진 중심어 $w_c$을 기반으로 컨텍스트 워드 $w_o$을 생성할 조건부 확률을 계산하는 것이며, 해당 로그 손실은 :eqref:`eq_skip-gram-log`의 반대에 의해 주어집니다. 

소프트맥스 연산의 특성으로 인해 컨텍스트 단어는 사전 $\mathcal{V}$의 누군가가 될 수 있으므로 :eqref:`eq_skip-gram-log`의 반대쪽에는 어휘의 전체 크기만큼 항목의 합계가 포함됩니다.따라서 :eqref:`eq_skip-gram-grad`의 스킵 그램 모델에 대한 기울기 계산과 :eqref:`eq_cbow-gradient`의 연속 단어 모음 모델에 대한 기울기 계산에는 모두 합계가 포함됩니다.안타깝게도 큰 사전 (종종 수십만 또는 수백만 단어) 을 합산하는 이러한 그라디언트의 계산 비용은 엄청납니다! 

앞서 언급한 계산 복잡성을 줄이기 위해 이 섹션에서는 두 가지 근사 훈련 방법을 소개합니다.
*음수 샘플링* 및*계층적 소프트맥스*.
스킵 그램 모델과 연속 단어 모음 모델 간의 유사성으로 인해이 두 가지 근사 훈련 방법을 설명하기 위해 skip-gram 모델을 예로 들어 보겠습니다. 

## 음수 샘플링
:label:`subsec_negative-sampling`

음수 샘플링은 원래 목적 함수를 수정합니다.중심 단어 $w_c$의 컨텍스트 창을 감안할 때 모든 (컨텍스트) 단어 $w_o$가 이 컨텍스트 창에서 유래되었다는 사실은 다음과 같이 모델링된 확률을 가진 이벤트로 간주됩니다. 

$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

여기서 $\sigma$는 시그모이드 활성화 함수의 정의를 사용합니다. 

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$
:eqlabel:`eq_sigma-f`

단어 임베딩을 훈련시키기 위해 텍스트 시퀀스에서 이러한 모든 이벤트의 공동 확률을 최대화하는 것으로 시작하겠습니다.특히, 길이가 $T$인 텍스트 시퀀스가 주어진 경우, 시간 단계 $t$의 단어를 $w^{(t)}$로 나타내고 컨텍스트 창 크기를 $m$로 지정하면 접합 확률을 최대화하는 것이 좋습니다. 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$
:eqlabel:`eq-negative-sample-pos`

그러나 :eqref:`eq-negative-sample-pos`는 긍정적 인 예가 포함 된 사건만 고려합니다.결과적으로 모든 단어 벡터가 무한대인 경우에만 :eqref:`eq-negative-sample-pos`의 접합 확률이 1로 최대화됩니다.물론 이러한 결과는 의미가 없습니다.목적 함수를 보다 의미 있게 만들기 위해
*네거티브 샘플링*
에서는 사전 정의된 분포에서 샘플링된 부정적인 예를 추가합니다. 

컨텍스트 단어 $w_o$가 중심 단어 $w_c$의 컨텍스트 창에서 나온 이벤트를 $S$로 나타냅니다.$w_o$와 관련된 이 이벤트의 경우 이 컨텍스트 창에 속하지 않은 미리 정의된 분포 $P(w)$ 샘플 $K$*노이즈 단어*에서 가져온 것입니다.잡음 단어 $w_k$ ($k=1, \ldots, K$) 이 $w_c$의 컨텍스트 창에서 오지 않는 이벤트를 $N_k$로 나타냅니다.긍정적인 예와 부정적인 예 $S, N_1, \ldots, N_K$를 모두 포함하는 이러한 사건이 상호 독립적이라고 가정합니다.음수 샘플링은 :eqref:`eq-negative-sample-pos`에서 접합 확률 (긍정적 인 예만 포함) 을 다음과 같이 다시 작성합니다. 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

여기서 조건부 확률은 이벤트 $S, N_1, \ldots, N_K$를 통해 근사화됩니다. 

$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$
:eqlabel:`eq-negative-sample-conditional-prob`

텍스트 시퀀스의 시간 스텝 $t$와 노이즈 워드 $w_k$에서 단어 $w^{(t)}$의 인덱스를 각각 $i_t$ 및 $h_k$으로 나타냅니다.:eqref:`eq-negative-sample-conditional-prob`의 조건부 확률에 대한 로그 손실은 다음과 같습니다. 

$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$

이제 각 훈련 단계에서 기울기에 대한 계산 비용은 사전 크기와는 아무런 관련이 없지만 $K$에 선형으로 의존한다는 것을 알 수 있습니다.하이퍼파라미터 $K$를 더 작은 값으로 설정하면 음수 샘플링을 사용하는 각 훈련 단계에서 기울기에 대한 계산 비용이 더 적게 듭니다. 

## 계층적 소프트맥스

대안적인 근사 훈련 방법으로
*계층적 소프트맥스*
에서는 :numref:`fig_hi_softmax`에 설명된 데이터 구조인 이진 트리를 사용합니다. 여기서 트리의 각 리프 노드는 사전 $\mathcal{V}$의 단어를 나타냅니다. 

![Hierarchical softmax for approximate training, where each leaf node of the tree represents a word in the dictionary.](../img/hi-softmax.svg)
:label:`fig_hi_softmax`

루트 노드에서 리프 노드까지 이진 트리의 단어 $w$을 나타내는 경로의 노드 수 (양쪽 끝 포함) 를 $L(w)$으로 나타냅니다.$n(w,j)$를 이 경로의 $j^\mathrm{th}$ 노드로 지정합니다. 컨텍스트 단어 벡터는 $\mathbf{u}_{n(w, j)}$입니다.예를 들어 :numref:`fig_hi_softmax`에서 $L(w_3) = 4$을 입력합니다.계층적 소프트맥스는 :eqref:`eq_skip-gram-softmax`의 조건부 확률을 다음과 같이 근사합니다. 

$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \text{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$

여기서 함수 $\sigma$은 :eqref:`eq_sigma-f`에 정의되고 $\text{leftChild}(n)$은 노드 $n$의 왼쪽 하위 노드입니다. $x$가 참이면 $ [\![x]\!]= 1$; otherwise $ [\![x]\!]= -1$. 

설명하기 위해 :numref:`fig_hi_softmax`에서 단어 $w_c$이 주어진 단어 $w_3$를 생성 할 조건부 확률을 계산해 보겠습니다.이를 위해서는 $w_c$의 단어 벡터 $\mathbf{v}_c$과 루트에서 $w_3$까지의 경로 (:numref:`fig_hi_softmax`에서 굵게 표시된 경로) 에 있는 비리프 노드 벡터 사이의 내적이 필요합니다. 이 벡터는 왼쪽, 오른쪽, 왼쪽으로 이동합니다. 

$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$

$\sigma(x)+\sigma(-x) = 1$ 이후, $w_c$라는 단어를 기반으로 사전 $\mathcal{V}$의 모든 단어를 생성 할 조건부 확률은 최대 1입니다. 

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$
:eqlabel:`eq_hi-softmax-sum-one`

다행히 $L(w_o)-1$은 이진 트리 구조로 인해 $\mathcal{O}(\text{log}_2|\mathcal{V}|)$의 순서이므로 사전 크기 $\mathcal{V}$가 클 때 계층적 소프트맥스를 사용하는 각 훈련 단계의 계산 비용은 근사 훈련이 없는 것에 비해 현저히 줄어듭니다. 

## 요약

* 음수 샘플링은 양의 예와 음의 예를 모두 포함하는 상호 독립적인 사건을 고려하여 손실 함수를 생성합니다.훈련에 소요되는 계산 비용은 각 스텝의 노이즈 단어 수에 따라 선형적으로 달라집니다.
* 계층적 소프트맥스는 이진 트리의 루트 노드에서 리프 노드까지의 경로를 사용하여 손실 함수를 구성합니다.훈련 계산 비용은 각 단계에서 사전 크기의 로그에 따라 달라집니다.

## 연습문제

1. 음성 샘플링에서 노이즈 단어를 샘플링하려면 어떻게 해야 합니까?
1. :eqref:`eq_hi-softmax-sum-one`가 유지되는지 확인합니다.
1. 각각 음의 샘플링과 계층적 소프트맥스를 사용하여 연속 단어 모음 모델을 훈련시키는 방법은 무엇입니까?

[Discussions](https://discuss.d2l.ai/t/382)
