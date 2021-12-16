# 언어 모델 및 데이터세트
:label:`sec_language_model`

:numref:`sec_text_preprocessing`에서는 텍스트 데이터를 토큰에 매핑하는 방법을 살펴봅니다. 토큰은 단어나 문자와 같은 일련의 개별 관측값으로 볼 수 있습니다.길이가 $T$인 텍스트 시퀀스의 토큰이 차례로 $x_1, x_2, \ldots, x_T$라고 가정합니다.그런 다음 텍스트 시퀀스에서 $x_t$ ($1 \leq t \leq T$) 을 시간 단계 $t$에서 관측치 또는 레이블로 간주할 수 있습니다.이러한 텍스트 시퀀스가 주어지면*언어 모델*의 목표는 시퀀스의 공동 확률을 추정하는 것입니다.

$$P(x_1, x_2, \ldots, x_T).$$

언어 모델은 매우 유용합니다.예를 들어, 이상적인 언어 모델은 한 번에 하나의 토큰을 그리는 것만으로도 자연 텍스트를 자체적으로 생성 할 수 있습니다. $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$.타자기를 사용하는 원숭이와 달리 이러한 모델에서 나오는 모든 텍스트는 자연어 (예: 영어 텍스트) 로 전달됩니다.또한 이전 대화 상자 조각의 텍스트를 조절하는 것만으로도 의미 있는 대화 상자를 생성하는 데 충분할 것입니다.문법적으로 합리적인 콘텐츠를 생성하는 것이 아니라 텍스트를 이해해야하기 때문에 우리는 여전히 그러한 시스템을 설계하는 것과는 거리가 멀다.

그럼에도 불구하고 언어 모델은 제한된 형태로도 훌륭한 서비스를 제공합니다.예를 들어, “말을 인식하는 것”과 “멋진 해변을 망치기”라는 문구는 매우 비슷하게 들립니다.이로 인해 음성 인식이 모호해질 수 있으며, 이는 두 번째 번역을 엉뚱한 것으로 거부하는 언어 모델을 통해 쉽게 해결됩니다.마찬가지로 문서 요약 알고리즘에서 “개가 사람을 물다”는 것이 “남자가 개를 물다”보다 훨씬 더 빈번하거나 “할머니를 먹고 싶다”는 것이 다소 혼란스러운 진술이라는 것을 아는 것이 가치가 있습니다. “나는 먹고 싶다, 할머니”는 훨씬 더 양성이다.

## 언어 모델 학습

분명한 질문은 문서 또는 일련의 토큰을 모델링하는 방법입니다.단어 수준에서 텍스트 데이터를 토큰화한다고 가정해 보겠습니다.:numref:`sec_sequence`에서 시퀀스 모델에 적용한 분석에 의지 할 수 있습니다.기본 확률 규칙을 적용하여 시작하겠습니다.

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

예를 들어, 네 단어가 포함된 텍스트 시퀀스의 확률은 다음과 같습니다.

$$P(\text{deep}, \text{learning}, \text{is}, \text{fun}) =  P(\text{deep}) P(\text{learning}  \mid  \text{deep}) P(\text{is}  \mid  \text{deep}, \text{learning}) P(\text{fun}  \mid  \text{deep}, \text{learning}, \text{is}).$$

언어 모델을 계산하려면 앞의 몇 단어가 주어진 단어의 확률과 조건부 확률을 계산해야 합니다.이러한 확률은 본질적으로 언어 모델 매개 변수입니다.

여기서는 훈련 데이터 세트가 모든 Wikipedia 항목, [Project Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg) 및 웹에 게시된 모든 텍스트와 같은 큰 텍스트 코퍼스라고 가정합니다.단어의 확률은 훈련 데이터셋에 있는 주어진 단어의 상대적 단어 빈도로부터 계산할 수 있습니다.예를 들어, 추정값 $\hat{P}(\text{deep})$는 “deep”라는 단어로 시작하는 문장의 확률로 계산할 수 있습니다.약간 덜 정확한 접근법은 “deep”라는 단어의 모든 발생을 세고 코퍼스의 총 단어 수로 나누는 것입니다.이것은 특히 빈번한 단어의 경우 상당히 잘 작동합니다.계속 진행하면 추정하려고 시도 할 수 있습니다.

$$\hat{P}(\text{learning} \mid \text{deep}) = \frac{n(\text{deep, learning})}{n(\text{deep})},$$

여기서 $n(x)$ 및 $n(x, x')$는 각각 싱글톤과 연속적인 단어 쌍의 발생 횟수입니다.안타깝게도 단어 쌍의 확률을 추정하는 것은 다소 어렵습니다. “딥 러닝”의 발생 빈도가 훨씬 적기 때문입니다.특히 일부 비정상적인 단어 조합의 경우 정확한 추정치를 얻기에 충분한 단어를 찾는 것이 까다로울 수 있습니다.세 단어 조합과 그 이상의 경우 상황이 악화됩니다.데이터셋에서는 볼 수 없는 그럴듯한 세 단어 조합이 많이 있을 것입니다.이러한 단어 조합을 0이 아닌 개수로 할당하는 솔루션을 제공하지 않으면 언어 모델에서 사용할 수 없습니다.데이터셋이 작거나 단어가 매우 드문 경우 그중 하나도 찾지 못할 수 있습니다.

일반적인 전략은 어떤 형태의*라플라스 스무딩*을 수행하는 것입니다.해결책은 모든 카운트에 작은 상수를 추가하는 것입니다.훈련 세트에 포함된 총 단어 수를 $n$로 나타내고 고유한 단어의 수를 $m$로 나타냅니다.이 솔루션은 싱글톤 (예: via) 에 도움이 됩니다.

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

여기서 $\epsilon_1,\epsilon_2$과 $\epsilon_3$는 하이퍼파라미터입니다.$\epsilon_1$을 예로 들어 보겠습니다. $\epsilon_1 = 0$에서는 스무딩이 적용되지 않습니다. $\epsilon_1$이 양의 무한대에 가까워지면 $\hat{P}(x)$는 균일 확률 $1/m$에 접근합니다.위의 내용은 다른 기술이 :cite:`Wood.Gasthaus.Archambeau.ea.2011`를 달성 할 수있는 것의 다소 원시적 인 변형입니다.

안타깝게도 이와 같은 모델은 다음과 같은 이유로 다소 빨리 다루기 힘듭니다.먼저 모든 개수를 저장해야 합니다.둘째, 단어의 의미를 완전히 무시합니다.예를 들어, “고양이”와 “고양이”는 관련 상황에서 발생해야 합니다.이러한 모델을 추가 컨텍스트에 맞게 조정하는 것은 매우 어렵지만 딥 러닝 기반 언어 모델은 이를 고려하는 데 매우 적합합니다.마지막으로, 긴 단어 시퀀스는 거의 참신하다는 것이 확실하므로 이전에 본 단어 시퀀스의 빈도를 단순히 계산하는 모델은 그곳에서 제대로 수행되지 않을 것입니다.

## 마르코프 모델 및 $n$그램

딥 러닝과 관련된 솔루션을 논의하기 전에 용어와 개념이 더 필요합니다.:numref:`sec_sequence`에서 마르코프 모델에 대한 논의를 상기하십시오.이를 언어 모델링에 적용해 보겠습니다.시퀀스에 대한 분포는 $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$인 경우 1차 마르코프 속성을 충족합니다.주문이 높을수록 종속성이 길어집니다.이렇게 하면 시퀀스를 모델링하는 데 적용할 수 있는 여러 근사값이 생성됩니다.

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

하나, 둘, 세 개의 변수를 포함하는 확률 공식은 일반적으로*유니그램*, *바이그램* 및*트라이그램* 모형이라고 합니다.다음에서는 더 나은 모델을 설계하는 방법을 배웁니다.

## 자연어 통계

이것이 실제 데이터에서 어떻게 작동하는지 봅시다.:numref:`sec_text_preprocessing`에 소개된 타임머신 데이터세트를 기반으로 어휘를 구성하고 가장 자주 사용하는 상위 10개 단어를 인쇄합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
#@tab all
tokens = d2l.tokenize(d2l.read_time_machine())
# Since each text line is not necessarily a sentence or a paragraph, we
# concatenate all text lines
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]
```

우리가 볼 수 있듯이 (**가장 인기있는 단어는**) 실제로 보기에는 지루합니다.종종 (***중지 단어***) 라고 불리며 필터링됩니다.그럼에도 불구하고 그들은 여전히 의미를 지니고 있으며 우리는 여전히 그것을 사용할 것입니다.게다가 주파수라는 단어가 다소 빠르게 쇠퇴한다는 것은 분명합니다.$10^{\mathrm{th}}$의 가장 빈번한 단어는 가장 인기 있는 단어만큼 일반적인 $1/5$보다 작습니다.더 나은 아이디어를 얻기 위해 [**주파수라는 단어의 그림을 플로팅합니다**].

```{.python .input}
#@tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

우리는 여기서 아주 근본적인 것을 하고 있습니다. 주파수라는 단어는 잘 정의된 방식으로 빠르게 쇠퇴합니다.처음 몇 단어를 예외로 처리한 후 나머지 모든 단어는 로그 로그 플롯에서 대략적으로 직선을 따릅니다.즉, 단어가*Zipf의 법칙*을 충족한다는 것을 의미합니다. 즉, $i^\mathrm{th}$에서 가장 자주 사용되는 단어의 빈도 $n_i$는 다음과 같습니다.

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

이는 다음과 같습니다.

$$\log n_i = -\alpha \log i + c,$$

여기서 $\alpha$는 분포를 나타내는 지수이고 $c$는 상수입니다.통계를 세고 스무딩하여 단어를 모델링하려는 경우 이미 일시 중지 될 것입니다.결국 우리는 드문 단어라고도 알려진 꼬리의 빈도를 크게 과대 평가할 것입니다.하지만 [**바이그램, 트라이그램**] 같은 다른 단어 조합은 어떻습니까?바이 그램 주파수가 유니 그램 주파수와 동일한 방식으로 작동하는지 살펴 보겠습니다.

```{.python .input}
#@tab all
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

여기서 한 가지 주목할 만하다.가장 빈번한 10 개의 단어 쌍 중 9 개는 두 개의 중지 단어로 구성되며 실제 책과 관련된 단어 중 하나만 “시간”입니다.또한 트라이 그램 주파수가 동일한 방식으로 작동하는지 살펴 보겠습니다.

```{.python .input}
#@tab all
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

마지막으로 유니그램, 바이그램, 트라이그램의 세 가지 모델 중에서 [**토큰 빈도를 시각화**] 하겠습니다.

```{.python .input}
#@tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

이 수치는 여러 가지 이유로 매우 흥미 롭습니다.첫째, 유니그램 단어 외에도 시퀀스 길이에 따라 :eqref:`eq_zipf_law`의 지수 $\alpha$가 더 작지만 일련의 단어가 Zipf의 법칙을 따르는 것으로 보입니다.둘째, 뚜렷한 $n$그램의 수는 그다지 크지 않습니다.이것은 언어에 상당히 많은 구조가 있다는 희망을 안겨줍니다.셋째, 많은 $n$그램이 매우 드물게 발생하기 때문에 라플라스 스무딩이 언어 모델링에 적합하지 않습니다.대신 딥 러닝 기반 모델을 사용할 것입니다.

## 긴 시퀀스 데이터 읽기

시퀀스 데이터는 본질적으로 순차적이므로 처리 문제를 해결해야 합니다.우리는 :numref:`sec_sequence`에서 다소 임시적인 방식으로 그렇게 했습니다.시퀀스가 너무 길어 모델이 한꺼번에 처리할 수 없는 경우, 읽기를 위해 해당 시퀀스를 분할할 수 있습니다.이제 일반적인 전략을 설명하겠습니다.모델을 도입하기 전에 신경망을 사용하여 언어 모델을 훈련한다고 가정해 보겠습니다. 여기서 네트워크는 미리 정의된 길이의 시퀀스의 미니 배치 (예: $n$ 시간 단계) 를 한 번에 처리합니다.이제 문제는 [**특징과 라벨의 미니배치를 무작위로 읽는 방법**] 입니다.

우선 텍스트 시퀀스는*The Time Machine* 책 전체와 같이 임의로 길어질 수 있으므로 이러한 긴 시퀀스를 동일한 수의 시간 단계를 가진 하위 시퀀스로 나눌 수 있습니다.신경망을 훈련시킬 때 이러한 하위 시퀀스의 미니 배치가 모델에 공급됩니다.네트워크가 한 번에 $n$ 시간 스텝의 하위 시퀀스를 처리한다고 가정합니다. :numref:`fig_timemachine_5gram`는 원본 텍스트 시퀀스에서 하위 시퀀스를 얻는 다양한 방법을 보여 줍니다. 여기서 $n=5$과 각 시간 스텝의 토큰은 문자에 해당합니다.초기 위치를 나타내는 임의의 오프셋을 선택할 수 있기 때문에 상당히 자유롭습니다.

![Different offsets lead to different subsequences when splitting up text.](../img/timemachine-5gram.svg)
:label:`fig_timemachine_5gram`

따라서 :numref:`fig_timemachine_5gram` 중에서 어떤 것을 골라야합니까?사실, 모두 똑같이 훌륭합니다.그러나 오프셋을 하나만 선택하면 네트워크 훈련에 가능한 모든 하위 시퀀스의 적용 범위가 제한됩니다.따라서 랜덤 오프셋으로 시작하여 시퀀스를 분할하여*적용 범위*와*임의성*을 모두 얻을 수 있습니다.다음에서는 두 가지 모두에 대해 이 작업을 수행하는 방법을 설명합니다.
*랜덤 샘플링* 및*순차 분할* 전략.

### 랜덤 샘플링

(**랜덤 샘플링에서 각 예제는 원래의 긴 시퀀스에서 임의로 캡처된 하위 시퀀스입니다.**) 반복 중에 인접한 두 개의 무작위 미니 배치의 하위 시퀀스가 원래 시퀀스에서 반드시 인접하지는 않습니다.언어 모델링의 목표는 지금까지 본 토큰을 기반으로 다음 토큰을 예측하는 것입니다. 따라서 레이블은 원래 시퀀스이며 토큰 하나만큼 이동됩니다.

다음 코드에서는 매번 데이터에서 미니배치를 무작위로 생성합니다.여기서 인수 `batch_size`는 각 미니배치의 하위 시퀀스 예제 수를 지정하고 `num_steps`는 각 하위 시퀀스에서 미리 정의된 시간 단계 수입니다.

```{.python .input}
#@tab all
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset (inclusive of `num_steps - 1`) to partition a
    # sequence
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for
        # subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield d2l.tensor(X), d2l.tensor(Y)
```

[**0에서 34.까지의 시퀀스를 수동으로 생성합니다**] 배치 크기와 시간 단계 수는 각각 2와 5라고 가정합니다.즉, $\lfloor (35 - 1) / 5 \rfloor= 6$개의 특징-레이블 서브시퀀스 쌍을 생성할 수 있습니다.미니 배치 크기가 2이면 미니 배치는 3 개만 얻을 수 있습니다.

```{.python .input}
#@tab all
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

### 순차적 파티션

원래 시퀀스의 무작위 샘플링 외에도 [**반복 중에 인접한 두 미니 배치의 하위 시퀀스가 원래 시퀀스에 인접하도록 할 수 있습니다.**] 이 전략은 미니 배치를 반복 할 때 분할 하위 시퀀스의 순서를 유지하므로 순차라고 합니다.분할.

```{.python .input}
#@tab mxnet, pytorch
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

```{.python .input}
#@tab tensorflow
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs = d2l.reshape(Xs, (batch_size, -1))
    Ys = d2l.reshape(Ys, (batch_size, -1))
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

동일한 설정을 사용하여 순차적 분할로 읽은 후속 시퀀스의 [**인쇄 기능 `X` 및 각 미니 배치마다 `Y` 레이블**] 을 지정해 보겠습니다.반복하는 동안 인접한 두 미니 배치의 하위 시퀀스는 실제로 원래 시퀀스에서 인접합니다.

```{.python .input}
#@tab all
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

이제 위의 두 샘플링 함수를 클래스에 래핑하여 나중에 데이터 반복기로 사용할 수 있습니다.

```{.python .input}
#@tab all
class SeqDataLoader:  #@save
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```

[**마지막으로 데이터 반복자와 어휘를 모두 반환하는 함수 `load_data_time_machine`를 정의합니다**]. 따라서 :numref:`sec_fashion_mnist`에 정의된 `d2l.load_data_fashion_mnist`과 같이 `load_data` 접두사를 가진 다른 함수와 유사하게 사용할 수 있습니다.

```{.python .input}
#@tab all
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```

## 요약

* 언어 모델은 자연어 처리의 핵심입니다.
* $n$그램은 종속성을 잘라내어 긴 시퀀스를 처리하는 데 편리한 모델을 제공합니다.
* 긴 시퀀스는 매우 드물게 발생하거나 전혀 발생하지 않는다는 문제로 어려움을 겪습니다.
* Zipf의 법칙은 유니그램뿐만 아니라 다른 $n$그램에 대한 단어 분포에 적용됩니다.
* 구조는 많지만 Laplace 스무딩을 통해 드문 단어 조합을 효율적으로 처리하기에는 빈도가 충분하지 않습니다.
* 긴 시퀀스를 읽는 주요 선택은 무작위 샘플링과 순차적 분할입니다.후자는 반복하는 동안 인접한 두 미니 배치의 하위 시퀀스가 원래 시퀀스에서 인접하도록 할 수 있습니다.

## 연습문제

1. 훈련 데이터셋에 $100,000$개의 단어가 있다고 가정합니다.4그램에 얼마나 많은 단어 주파수와 다중 단어 인접 주파수를 저장해야 합니까?
1. 대화를 어떻게 모델링하시겠어요?
1. 유니그램, 바이그램, 트라이그램에 대한 Zipf 법칙의 지수를 추정합니다.
1. 긴 시퀀스 데이터를 읽는 데 사용할 수 있는 다른 방법은 무엇입니까?
1. 긴 시퀀스를 읽는 데 사용하는 랜덤 오프셋을 고려하십시오.
    1. 랜덤 오프셋을 갖는 것이 좋은 이유는 무엇입니까?
    1. 실제로 문서의 시퀀스에 대해 완벽하게 균일한 분포가 이루어지나요?
    1. 좀 더 균일하게 만들기 위해 무엇을 해야 할까요?
1. 시퀀스 예제를 완전한 문장으로 만들고 싶다면 미니 배치 샘플링에서 어떤 문제가 발생합니까?문제를 어떻게 해결할 수 있을까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:
