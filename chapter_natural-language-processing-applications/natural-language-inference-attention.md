# 자연어 추론: 주의 사용
:label:`sec_natural-language-inference-attention`

:numref:`sec_natural-language-inference-and-dataset`에서 자연어 추론 작업과 SNLI 데이터 세트를 도입했습니다.복잡하고 깊은 아키텍처를 기반으로하는 많은 모델을 고려할 때 Parikh는 주의력 메커니즘으로 자연어 추론을 해결할 것을 제안하고이를 “분해 가능한 관심 모델”이라고 불렀습니다. :cite:`Parikh.Tackstrom.Das.ea.2016`.그 결과 순환 계층이나 컨벌루션 계층이 없는 모델이 생성되어 훨씬 적은 수의 파라미터로 SNLI 데이터셋에서 최상의 결과를 얻을 수 있습니다.이 섹션에서는 :numref:`fig_nlp-map-nli-attention`에 설명된 대로 자연어 추론을 위한 주의 기반 방법 (MLP 사용) 을 설명하고 구현합니다. 

![This section feeds pretrained GloVe to an architecture based on attention and MLPs for natural language inference.](../img/nlp-map-nli-attention.svg)
:label:`fig_nlp-map-nli-attention`

## 더 모델

전제 및 가설에서 토큰의 순서를 유지하는 것보다 간단하게 한 텍스트 시퀀스의 토큰을 다른 텍스트 시퀀스의 모든 토큰에 정렬하고 그 반대의 경우도 마찬가지입니다. 그런 다음 이러한 정보를 비교 및 집계하여 전제와 가설 간의 논리적 관계를 예측할 수 있습니다.기계 번역에서 소스 문장과 대상 문장 사이의 토큰 정렬과 유사하게, 전제와 가설 사이의 토큰 정렬은 주의 메커니즘을 통해 깔끔하게 수행 할 수 있습니다. 

![Natural language inference using attention mechanisms.](../img/nli-attention.svg)
:label:`fig_nli_attention`

:numref:`fig_nli_attention`는 주의력 메커니즘을 사용한 자연어 추론 방법을 보여줍니다.상위 수준에서는 참석, 비교 및 집계라는 세 가지 공동 교육 단계로 구성됩니다.다음에서 단계별로 설명합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

### 참석

첫 번째 단계는 한 텍스트 시퀀스의 토큰을 다른 시퀀스의 각 토큰에 정렬하는 것입니다.전제가 “나는 수면이 필요해”라고 가정하고 가설은 “나는 피곤하다”고 가정합니다.의미 론적 유사성으로 인해 가설의 “i”를 전제의 “i”와 정렬하고 가설의 “피곤함”을 전제의 “수면”과 일치시킬 수 있습니다.마찬가지로, 우리는 전제의 “i”를 가설의 “i”와 정렬하고 전제의 “필요”와 “수면”을 가설의 “피곤함”과 일치시킬 수 있습니다.이러한 정렬은 가중 평균을 사용하여*부드러운*이며, 이상적으로는 큰 가중치가 정렬될 토큰과 연관되어 있습니다.시연을 쉽게하기 위해 :numref:`fig_nli_attention`는*하드* 방식으로 이러한 정렬을 보여줍니다. 

이제 주의 메커니즘을 사용하여 소프트 얼라인먼트에 대해 자세히 설명합니다.전제와 가설을 $\mathbf{A} = (\mathbf{a}_1, \ldots, \mathbf{a}_m)$와 $\mathbf{B} = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$로 나타내며, 토큰의 수는 각각 $m$과 $n$이며, 여기서 $\mathbf{a}_i, \mathbf{b}_j \in \mathbb{R}^{d}$ ($i = 1, \ldots, m, j = 1, \ldots, n$) 은 $d$차원의 단어 벡터입니다.소프트 정렬의 경우 주의 가중치 $e_{ij} \in \mathbb{R}$를 다음과 같이 계산합니다. 

$$e_{ij} = f(\mathbf{a}_i)^\top f(\mathbf{b}_j),$$
:eqlabel:`eq_nli_e`

여기서 함수 $f$은 다음 `mlp` 함수에 정의된 MLP입니다.출력 차원 $f$은 `mlp`의 `num_hiddens` 인수로 지정됩니다.

```{.python .input}
def mlp(num_hiddens, flatten):
    net = nn.Sequential()
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    return net
```

```{.python .input}
#@tab pytorch
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)
```

:eqref:`eq_nli_e`에서 $f$은 입력 $\mathbf{a}_i$과 $\mathbf{b}_j$를 입력으로 함께 사용하지 않고 개별적으로 입력한다는 점을 강조해야합니다.이러한*분해* 트릭은 $mn$ 응용 프로그램 (2차 복잡성) 이 아닌 $f$의 $m + n$ 응용 프로그램 (선형 복잡성) 으로 이어집니다. 

:eqref:`eq_nli_e`에서 주의 가중치를 정규화하면 가설의 모든 토큰 벡터의 가중 평균을 계산하여 전제에서 $i$에 의해 인덱싱된 토큰과 부드럽게 정렬된 가설의 표현을 얻습니다. 

$$
\boldsymbol{\beta}_i = \sum_{j=1}^{n}\frac{\exp(e_{ij})}{ \sum_{k=1}^{n} \exp(e_{ik})} \mathbf{b}_j.
$$

마찬가지로 가설에서 $j$로 인덱싱된 각 토큰에 대한 전제 토큰의 소프트 정렬을 계산합니다. 

$$
\boldsymbol{\alpha}_j = \sum_{i=1}^{m}\frac{\exp(e_{ij})}{ \sum_{k=1}^{m} \exp(e_{kj})} \mathbf{a}_i.
$$

아래에서는 `Attend` 클래스를 정의하여 입력 구내 `A`과의 가설 (`beta`) 의 소프트 정렬 및 입력 가설 `B`과 구내의 소프트 정렬 (`alpha`) 을 계산합니다.

```{.python .input}
class Attend(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (b`atch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = npx.batch_dot(f_A, f_B, transpose_b=True)
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = npx.batch_dot(npx.softmax(e), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = npx.batch_dot(npx.softmax(e.transpose(0, 2, 1)), A)
        return beta, alpha
```

```{.python .input}
#@tab pytorch
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (`batch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha
```

### 비교하기

다음 단계에서는 한 시퀀스의 토큰을 해당 토큰과 부드럽게 정렬된 다른 시퀀스와 비교합니다.소프트 정렬에서는 관심 가중치가 다를 수 있지만 한 시퀀스의 모든 토큰이 다른 시퀀스의 토큰과 비교됩니다.쉽게 시연할 수 있도록 :numref:`fig_nli_attention`는 토큰과 정렬된 토큰을 *하드* 방식으로 페어링합니다.예를 들어, 참석 단계에서 전제의 “필요”와 “수면”이 모두 가설에서 “피곤함”과 일치한다고 결정하면 “피곤함 - 수면 필요”쌍이 비교됩니다. 

비교 단계에서는 한 시퀀스에서 토큰의 연결 (연산자 $[\cdot, \cdot]$) 을 공급하고 다른 시퀀스의 정렬된 토큰을 함수 $g$ (MLP) 로 공급합니다. 

$$\mathbf{v}_{A,i} = g([\mathbf{a}_i, \boldsymbol{\beta}_i]), i = 1, \ldots, m\\ \mathbf{v}_{B,j} = g([\mathbf{b}_j, \boldsymbol{\alpha}_j]), j = 1, \ldots, n.$$

:eqlabel:`eq_nli_v_ab` 

:eqref:`eq_nli_v_ab`에서 $\mathbf{v}_{A,i}$은 전제의 토큰 $i$와 토큰 $i$와 부드럽게 정렬된 모든 가설 토큰 간의 비교입니다. 반면 $\mathbf{v}_{B,j}$은 가설의 토큰 $j$과 토큰 $j$과 부드럽게 정렬된 모든 전제 토큰 간의 비교입니다.다음 `Compare` 클래스는 비교 단계 등을 정의합니다.

```{.python .input}
class Compare(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(np.concatenate([A, beta], axis=2))
        V_B = self.g(np.concatenate([B, alpha], axis=2))
        return V_A, V_B
```

```{.python .input}
#@tab pytorch
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
```

### 집계

비교 벡터 $\mathbf{v}_{A,i}$ ($i = 1, \ldots, m$) 와 $\mathbf{v}_{B,j}$ ($j = 1, \ldots, n$) 의 두 세트를 사용하여 마지막 단계에서 이러한 정보를 집계하여 논리적 관계를 추론합니다.먼저 두 세트를 모두 요약합니다. 

$$
\mathbf{v}_A = \sum_{i=1}^{m} \mathbf{v}_{A,i}, \quad \mathbf{v}_B = \sum_{j=1}^{n}\mathbf{v}_{B,j}.
$$

다음으로 두 요약 결과를 함수 $h$ (MLP) 에 연결하여 논리적 관계의 분류 결과를 얻습니다. 

$$
\hat{\mathbf{y}} = h([\mathbf{v}_A, \mathbf{v}_B]).
$$

집계 단계는 다음 `Aggregate` 클래스에서 정의됩니다.

```{.python .input}
class Aggregate(nn.Block):
    def __init__(self, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_hiddens=num_hiddens, flatten=True)
        self.h.add(nn.Dense(num_outputs))

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(axis=1)
        V_B = V_B.sum(axis=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.h(np.concatenate([V_A, V_B], axis=1))
        return Y_hat
```

```{.python .input}
#@tab pytorch
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat
```

### 모든 것을 하나로 모으다

참석, 비교 및 집계 단계를 한데 모아 분해 가능한 주의력 모델을 정의하여 이 세 단계를 공동으로 훈련시킵니다.

```{.python .input}
class DecomposableAttention(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_hiddens)
        self.compare = Compare(num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_hiddens, 3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

```{.python .input}
#@tab pytorch
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

## 모델 훈련 및 평가

이제 SNLI 데이터 세트에서 정의된 분해 가능한 주의 모델을 훈련하고 평가합니다.먼저 데이터세트를 읽는 것으로 시작합니다. 

### 데이터세트 읽기

:numref:`sec_natural-language-inference-and-dataset`에 정의된 함수를 사용하여 SNLI 데이터 세트를 다운로드하고 읽습니다.배치 크기와 시퀀스 길이는 각각 $256$ 및 $50$로 설정됩니다.

```{.python .input}
#@tab all
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
```

### 모델 만들기

사전 훈련된 100차원 GLOVE 임베딩을 사용하여 입력 토큰을 나타냅니다.따라서 :eqref:`eq_nli_e`에서 벡터 $\mathbf{a}_i$과 $\mathbf{b}_j$의 차원을 100으로 미리 정의합니다.:eqref:`eq_nli_e`의 함수 $f$과 :eqref:`eq_nli_v_ab`의 $g$의 출력 치수는 200으로 설정됩니다.그런 다음 모델 인스턴스를 만들고 매개 변수를 초기화하고 GLOVE 임베딩을 로드하여 입력 토큰의 벡터를 초기화합니다.

```{.python .input}
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
net.initialize(init.Xavier(), ctx=devices)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
```

```{.python .input}
#@tab pytorch
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds);
```

### 모델 훈련 및 평가

텍스트 시퀀스 (또는 이미지) 와 같은 단일 입력을 취하는 :numref:`sec_multi_gpu`의 `split_batch` 함수와 달리 미니 배치의 전제 및 가설과 같은 여러 입력을 취하는 `split_batch_multi_inputs` 함수를 정의합니다.

```{.python .input}
#@save
def split_batch_multi_inputs(X, y, devices):
    """Split multi-input `X` and `y` into multiple devices."""
    X = list(zip(*[gluon.utils.split_and_load(
        feature, devices, even_split=False) for feature in X]))
    return (X, gluon.utils.split_and_load(y, devices, even_split=False))
```

이제 SNLI 데이터셋에서 모델을 훈련시키고 평가할 수 있습니다.

```{.python .input}
lr, num_epochs = 0.001, 4
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

### 모델 사용

마지막으로, 예측 함수를 정의하여 한 쌍의 전제와 가설 간의 논리적 관계를 출력합니다.

```{.python .input}
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    premise = np.array(vocab[premise], ctx=d2l.try_gpu())
    hypothesis = np.array(vocab[hypothesis], ctx=d2l.try_gpu())
    label = np.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

```{.python .input}
#@tab pytorch
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

훈련된 모델을 사용하여 샘플 문장 쌍에 대한 자연어 추론 결과를 얻을 수 있습니다.

```{.python .input}
#@tab all
predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
```

## 요약

* 분해 가능한 주의 모델은 전제와 가설 간의 논리적 관계를 예측하는 세 단계, 즉 참석, 비교 및 집계로 구성됩니다.
* 주의 메커니즘을 사용하면 한 텍스트 시퀀스의 토큰을 다른 텍스트 시퀀스의 모든 토큰에 정렬할 수 있으며 그 반대의 경우도 마찬가지입니다.이러한 정렬은 가중 평균을 사용하여 부드럽습니다. 이상적으로는 큰 가중치가 정렬될 토큰과 연관됩니다.
* 분해 트릭은 주의력 가중치를 계산할 때 2차 복잡도보다 더 바람직한 선형 복잡성을 유도합니다.
* 사전 훈련된 단어 벡터를 자연어 추론과 같은 다운스트림 자연어 처리 작업의 입력 표현으로 사용할 수 있습니다.

## 연습문제

1. 다른 하이퍼파라미터 조합을 사용하여 모델을 훈련시킵니다.테스트 세트의 정확도를 높일 수 있습니까?
1. 자연어 추론에 대한 분해 가능한 주의력 모델의 주요 단점은 무엇입니까?
1. 문장 쌍에 대해 의미 론적 유사성 수준 (예: 0과 1 사이의 연속형 값) 을 얻고 싶다고 가정합니다.데이터세트를 수집하고 레이블을 지정하려면 어떻게 해야 합니까?주의 메커니즘을 사용하여 모델을 설계할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/395)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1530)
:end_tab:
