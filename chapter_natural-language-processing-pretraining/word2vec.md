# 단어 임베딩 (워드2vec)
:label:`sec_word2vec`

자연어는 의미를 표현하는 데 사용되는 복잡한 시스템입니다.이 체계에서 단어는 의미의 기본 단위입니다.이름에서 알 수 있듯이
*단어 벡터*는 단어를 나타내는 데 사용되는 벡터입니다.
특징 벡터 또는 단어 표현으로 간주할 수도 있습니다.단어를 실수 벡터에 매핑하는 기술을*단어 임베딩*이라고 합니다.최근 몇 년 동안 단어 임베딩은 점차 자연어 처리의 기본 지식이되었습니다. 

## 원-핫 벡터는 나쁜 선택입니다

:numref:`sec_rnn_scratch`에서 단어 (문자는 단어) 를 표현하기 위해 원핫 벡터를 사용했습니다.사전의 다른 단어 수 (사전 크기) 가 $N$이고 각 단어가 $0$에서 $N−1$까지 다른 정수 (색인) 에 해당한다고 가정합니다.인덱스가 $i$인 단어에 대한 원핫 벡터 표현을 얻기 위해 모든 0을 가진 길이-$N$ 벡터를 만들고 위치 $i$의 요소를 1로 설정합니다.이러한 방식으로 각 단어는 길이가 $N$인 벡터로 표시되며 신경망에서 직접 사용할 수 있습니다. 

원-핫 워드 벡터는 구성하기 쉽지만 일반적으로 좋은 선택이 아닙니다.주된 이유는 1-hot 단어 벡터가 우리가 자주 사용하는*코사인 유사성*과 같이 서로 다른 단어 간의 유사성을 정확하게 표현할 수 없기 때문입니다.벡터 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$의 경우 코사인 유사성은 벡터 사이의 각도의 코사인입니다. 

$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$

서로 다른 두 단어의 1-hot 벡터 간의 코사인 유사성은 0이므로 1-hot 벡터는 단어 간의 유사성을 인코딩할 수 없습니다. 

## 자가 지도 단어2vec

위의 문제를 해결하기 위해 [word2vec](https://code.google.com/archive/p/word2vec/) 도구가 제안되었습니다.각 단어를 고정 길이 벡터에 매핑하며 이러한 벡터는 서로 다른 단어 간의 유사성과 유추 관계를 더 잘 표현할 수 있습니다.단어2vec 도구에는 두 가지 모델, 즉 스킵 그램* :cite:`Mikolov.Sutskever.Chen.ea.2013`와*연속 단어 모음* (CBOW) :cite:`Mikolov.Chen.Corrado.ea.2013`가 포함되어 있습니다.의미 론적으로 의미있는 표현의 경우, 훈련은 조건부 확률에 의존하며, 조건부 확률은 상체에서 주변 단어 중 일부를 사용하여 일부 단어를 예측하는 것으로 볼 수 있습니다.감독은 레이블이 없는 데이터에서 이루어지기 때문에 스킵 그램과 연속 단어 모음은 모두 자체 감독 모델입니다. 

다음에서는 이 두 가지 모델과 그 훈련 방법을 소개합니다. 

## 스킵 그램 모델
:label:`subsec_skip-gram`

*skip-gram* 모델은 단어를 사용하여 텍스트 시퀀스에서 주변 단어를 생성할 수 있다고 가정합니다.텍스트 시퀀스 “the”, “남자”, “사랑”, “그의”, “아들”을 예로 들어 보겠습니다.*중심 단어*로 “loves”를 선택하고 컨텍스트 창 크기를 2로 설정해 보겠습니다.:numref:`fig_skip_gram`에서 볼 수 있듯이 중심 단어 “loves”가 주어지면 스킵 그램 모델은*컨텍스트 단어*를 생성하기 위한 조건부 확률을 고려합니다. “the”, “man”, “his” 및 “son”은 중심 단어에서 두 단어 이상 떨어져 있지 않습니다. 

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$

중심 단어 (즉, 조건부 독립성) 가 주어지면 문맥 단어가 독립적으로 생성된다고 가정합니다.이 경우 위의 조건부 확률은 다음과 같이 다시 작성할 수 있습니다. 

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$

![The skip-gram model considers the conditional probability of generating the surrounding context words given a center word.](../img/skip-gram.svg)
:label:`fig_skip_gram`

스킵 그램 모델에서 각 단어에는 조건부 확률을 계산하기 위한 두 개의 $d$차원 벡터 표현이 있습니다.보다 구체적으로, 사전에 색인이 $i$인 단어의 경우 $\mathbf{v}_i\in\mathbb{R}^d$ 및 $\mathbf{u}_i\in\mathbb{R}^d$로 각각 중심* 단어와*컨텍스트* 단어로 사용될 때 두 벡터를 나타냅니다.중심 단어 $w_c$ (사전에 인덱스 $c$ 포함) 이 주어진 컨텍스트 단어 $w_o$ (사전에 인덱스 $o$ 포함) 를 생성할 조건부 확률은 벡터 내적에 대한 소프트맥스 연산을 통해 모델링할 수 있습니다. 

$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)},$$
:eqlabel:`eq_skip-gram-softmax`

여기서 어휘 색인은 $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$을 설정합니다.길이가 $T$인 텍스트 시퀀스가 주어지며, 여기서 시간 단계 $t$의 단어는 $w^{(t)}$으로 표시됩니다.중심 단어가 주어지면 문맥 단어가 독립적으로 생성된다고 가정합니다.컨텍스트 창 크기 $m$의 경우 skip-gram 모델의 우도 함수는 가운데 단어가 지정된 경우 모든 컨텍스트 단어를 생성할 확률입니다. 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

여기서 $1$보다 작거나 $T$보다 큰 시간 스텝은 생략할 수 있습니다. 

### 트레이닝

스킵 그램 모델 매개 변수는 어휘의 각 단어에 대한 중심 단어 벡터와 문맥 단어 벡터입니다.훈련에서는 우도 함수 (즉, 최대 우도 추정) 를 최대화하여 모델 매개 변수를 학습합니다.이는 다음과 같은 손실 함수를 최소화하는 것과 같습니다. 

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)}).$$

확률 적 경사 하강을 사용하여 손실을 최소화하는 경우 각 반복에서 더 짧은 부분 시퀀스를 무작위로 샘플링하여 이 하위 시퀀스의 (확률 적) 기울기를 계산하여 모델 매개 변수를 업데이트 할 수 있습니다.이 (확률 적) 기울기를 계산하려면 중심 단어 벡터와 컨텍스트 단어 벡터에 대한 로그 조건부 확률의 기울기를 얻어야합니다.일반적으로 :eqref:`eq_skip-gram-softmax`에 따르면 중심 단어 $w_c$과 문맥 단어 $w_o$의 쌍을 포함하는 로그 조건부 확률은 다음과 같습니다. 

$$\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$
:eqlabel:`eq_skip-gram-log`

미분을 통해 중심 워드 벡터 $\mathbf{v}_c$에 대한 기울기를 다음과 같이 얻을 수 있습니다. 

$$\begin{aligned}\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.\end{aligned}$$
:eqlabel:`eq_skip-gram-grad`

:eqref:`eq_skip-gram-grad`의 계산에는 $w_c$를 중심 단어로 사용하여 사전에 있는 모든 단어의 조건부 확률이 필요합니다.다른 워드 벡터의 기울기도 같은 방법으로 얻을 수 있습니다. 

훈련 후 사전에 색인이 $i$인 단어에 대해 단어 벡터 $\mathbf{v}_i$ (중심 단어) 와 $\mathbf{u}_i$ (컨텍스트 단어) 를 모두 얻습니다.자연어 처리 응용 프로그램에서 스킵 그램 모델의 중심 단어 벡터는 일반적으로 단어 표현으로 사용됩니다. 

## 연속 단어 모음 (CBOW) 모델

*연속 단어 묶음* (CBOW) 모델은 스킵 그램 모델과 유사합니다.스킵 그램 모델과의 주요 차이점은 연속 단어 모음 모델은 텍스트 시퀀스의 주변 컨텍스트 단어를 기반으로 중심 단어가 생성된다고 가정한다는 것입니다.예를 들어 동일한 텍스트 시퀀스 “the”, “man”, “loves”, “his”및 “son”에서 “loves”를 중심 단어로 사용하고 컨텍스트 창 크기가 2인 경우 연속 단어 모음 모델은 컨텍스트 단어 “the”, “man”, “his”및 “son”을 기반으로 중심 단어 “loves”를 생성 할 조건부 확률을 고려합니다.“(:numref:`fig_cbow`에 표시된 대로), 이는 

$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$

![The continuous bag of words model considers the conditional probability of generating the center word given its surrounding context words.](../img/cbow.svg)
:eqlabel:`fig_cbow`

연속 단어 모음 모델에는 여러 컨텍스트 단어가 있으므로 이러한 컨텍스트 단어 벡터는 조건부 확률 계산에서 평균화됩니다.특히 사전에 색인이 $i$인 단어의 경우 $\mathbf{v}_i\in\mathbb{R}^d$ 및 $\mathbf{u}_i\in\mathbb{R}^d$으로 각각 컨텍스트* 단어와*중심* 단어 (스킵그램 모델에서 의미가 전환됨) 로 사용될 때 두 벡터를 나타냅니다.주변 컨텍스트 단어 $w_{o_1}, \ldots, w_{o_{2m}}$ (사전에 색인 $o_1, \ldots, o_{2m}$ 포함) 이 주어지면 중심 단어 $w_c$ (사전에 색인 $c$ 포함) 을 생성 할 조건부 확률은 다음과 같이 모델링 할 수 있습니다. 

$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}.$$
:eqlabel:`fig_cbow-full`

간결하게 하기 위해 $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$ 및 $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$을 입력합니다.그러면 :eqref:`fig_cbow-full`를 다음과 같이 단순화할 수 있습니다 

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$

길이가 $T$인 텍스트 시퀀스가 주어지며, 여기서 시간 단계 $t$의 단어는 $w^{(t)}$로 표시됩니다.문맥 창 크기 $m$의 경우 연속 단어 모음 모델의 우도 함수는 문맥 단어가 주어진 경우 모든 중심 단어를 생성할 확률입니다. 

$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

### 트레이닝

연속 단어 모음 모델을 훈련하는 것은 스킵 그램 모델을 훈련시키는 것과 거의 같습니다.연속형 단어 모음 모델의 최대우도 추정치는 다음 손실 함수를 최소화하는 것과 같습니다. 

$$  -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

주의 사항 

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$

차별화를 통해 모든 컨텍스트 단어 벡터 $\mathbf{v}_{o_i}$ ($i = 1, \ldots, 2m$) 에 대한 기울기를 얻을 수 있습니다. 

$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$
:eqlabel:`eq_cbow-gradient`

다른 워드 벡터의 기울기도 같은 방법으로 얻을 수 있습니다.스킵 그램 모델과 달리 연속 단어 모음 모델은 일반적으로 컨텍스트 단어 벡터를 단어 표현으로 사용합니다. 

## 요약

* 단어 벡터는 단어를 나타내는 데 사용되는 벡터이며 특징 벡터 또는 단어 표현으로 간주할 수도 있습니다.단어를 실제 벡터에 매핑하는 기술을 단어 임베딩이라고 합니다.
* word2vec 도구에는 단어 모델의 스킵 그램과 연속 백이 모두 포함되어 있습니다.
* skip-gram 모델은 단어를 사용하여 텍스트 시퀀스에서 주변 단어를 생성할 수 있다고 가정하고, Continuous Bag of words 모델은 중심 단어가 주변 컨텍스트 단어를 기반으로 생성된다고 가정합니다.

## 연습문제

1. 각 그래디언트를 계산하기 위한 계산 복잡도는 얼마입니까?사전 크기가 크면 어떤 문제가 될 수 있을까요?
1. 영어의 일부 고정 구문은 “뉴욕”과 같은 여러 단어로 구성됩니다.단어 벡터를 훈련시키는 방법?힌트: see Section 4 in the word2vec paper :cite:`Mikolov.Sutskever.Chen.ea.2013`.
1. 스킵 그램 모델을 예로 들어 word2vec 설계에 대해 생각해 보겠습니다.스킵 그램 모델에서 두 워드 벡터의 내적과 코사인 유사성 간의 관계는 무엇입니까?의미가 비슷한 단어 쌍의 경우, 스킵-그램 모델로 훈련된 단어 벡터의 코사인 유사성이 높은 이유는 무엇입니까?

[Discussions](https://discuss.d2l.ai/t/381)
