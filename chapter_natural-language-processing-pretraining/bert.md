# 트랜스포머 (BERT) 의 양방향 엔코더 표현
:label:`sec_bert`

자연어 이해를 위한 몇 가지 단어 삽입 모델을 도입했습니다.사전 훈련 후 출력은 각 행이 미리 정의된 어휘의 단어를 나타내는 벡터인 행렬로 간주할 수 있습니다.실제로 이러한 단어 임베딩 모델은 모두*문맥에 독립적입니다*.이 속성을 설명하는 것으로 시작하겠습니다. 

## 문맥 독립에서 문맥에 따른 것까지

:numref:`sec_word2vec_pretraining` 및 :numref:`sec_synonyms`의 실험을 상기하십시오.예를 들어 word2vec과 GLOve는 단어의 문맥 (있는 경우) 에 관계없이 동일한 사전 훈련된 벡터를 동일한 단어에 할당합니다.공식적으로 모든 토큰 $x$의 컨텍스트 독립적인 표현은 $x$만 입력으로 취하는 함수 $f(x)$입니다.자연어로 된 다분성과 복잡한 의미론이 풍부하다는 점을 감안할 때 상황에 독립적 인 표현에는 명백한 한계가 있습니다.예를 들어, “크레인이 날고있다”와 “크레인 운전자가 왔습니다”라는 문맥에서 “크레인”이라는 단어는 완전히 다른 의미를 가지고 있습니다. 따라서 문맥에 따라 같은 단어에 다른 표현이 할당될 수 있습니다. 

이는 단어의 표현이 문맥에 따라 달라지는*문맥에 민감한* 단어 표현의 개발에 동기를 부여합니다.따라서 토큰 $x$의 상황에 맞는 표현은 $x$와 해당 컨텍스트 $c(x)$에 따라 함수 $f(x, c(x))$입니다.널리 사용되는 상황에 맞는 표현에는 TagLM (언어 모델 증강 시퀀스 태거) :cite:`Peters.Ammar.Bhagavatula.ea.2017`, CoVE (컨텍스트 벡터) :cite:`McCann.Bradbury.Xiong.ea.2017` 및 ElMo (언어 모델의 임베딩) :cite:`Peters.Neumann.Iyyer.ea.2018`이 포함됩니다. 

예를 들어, 전체 시퀀스를 입력으로 사용하는 ElMo는 입력 시퀀스의 각 단어에 표현을 할당하는 함수입니다.특히, ElMo는 사전 훈련된 양방향 LSTM의 모든 중간 계층 표현을 출력 표현으로 결합합니다.그런 다음 ElMo 표현이 기존 모델에서 eLMo 표현과 토큰의 원래 표현 (예: GLOVE) 을 연결하는 등의 추가 기능으로 다운스트림 작업의 기존 감독 모델에 추가됩니다.한편, 사전 훈련된 양방향 LSTM 모델의 모든 가중치는 ElMo 표현이 추가된 후 동결됩니다.반면에 기존의 감독 모델은 주어진 작업에 맞게 특별히 사용자 정의됩니다.당시 다양한 작업에 대해 서로 다른 최상의 모델을 활용하여 ElMo를 추가하여 감정 분석, 자연어 추론, 의미 론적 역할 라벨링, 상호 참조 해결, 명명된 엔터티 인식 및 질문 응답과 같은 6가지 자연어 처리 작업에서 최첨단 기술을 개선했습니다. 

## 특정 작업부터 작업에 구애받지 않는 작업까지

ElMo는 다양한 자연어 처리 작업에 대한 솔루션을 크게 개선했지만 각 솔루션은 여전히*작업 특정* 아키텍처에 달려 있습니다.그러나 모든 자연어 처리 작업에 대해 특정 아키텍처를 만드는 것은 사실상 간단하지 않습니다.GPT (생성 사전 교육) 모델은 상황에 맞는 표현 :cite:`Radford.Narasimhan.Salimans.ea.2018`에 대한 일반적인*작업에 구애받지 않음* 모델을 설계하려는 노력을 나타냅니다.트랜스포머 디코더에 구축된 GPT는 텍스트 시퀀스를 나타내는 데 사용될 언어 모델을 사전 트레이닝합니다.다운스트림 작업에 GPT를 적용하면 언어 모델의 출력이 추가된 선형 출력 계층으로 공급되어 작업의 레이블을 예측합니다.사전 훈련된 모델의 파라미터를 고정시키는 ElMo와는 대조적으로, GPT는 다운스트림 작업의 지도 학습 중에 사전 훈련된 트랜스포머 디코더의 파라미터를*모든* 미세 조정합니다.GPT는 자연어 추론, 질문 답변, 문장 유사성 및 분류의 12 가지 과제에 대해 평가되었으며 모델 아키텍처의 변경을 최소화하면서 그 중 9 개에서 최첨단 기술을 개선했습니다. 

그러나 언어 모델의 자기 회귀 특성으로 인해 GPT는 앞쪽 (왼쪽에서 오른쪽) 만 찾습니다.“현금을 입금하기 위해 은행에 갔다”와 “나는 앉기 위해 은행에 갔다”라는 맥락에서 “은행”은 왼쪽의 맥락에 민감하기 때문에 GPT는 “은행”에 대해 동일한 표현을 반환하지만 의미는 다릅니다. 

## BERT: 두 세계의 장점을 결합하다

지금까지 살펴본 것처럼 ElMo는 컨텍스트를 양방향으로 인코딩하지만 작업 별 아키텍처를 사용합니다. GPT는 작업에 구애받지 않지만 컨텍스트를 왼쪽에서 오른쪽으로 인코딩합니다.두 세계의 장점을 결합한 BERT (트랜스포머의 양방향 인코더 표현) 는 컨텍스트를 양방향으로 인코딩하며 광범위한 자연어 처리 작업 :cite:`Devlin.Chang.Lee.ea.2018`에 대해 최소한의 아키텍처 변경이 필요합니다.사전 훈련된 트랜스포머 인코더를 사용하여 BERT는 양방향 컨텍스트를 기반으로 모든 토큰을 나타낼 수 있습니다.다운스트림 작업의 지도 학습 중에 BERT는 두 가지 측면에서 GPT와 유사합니다.첫째, BERT 표현은 모든 토큰에 대한 예측과 전체 시퀀스에 대한 예측과 같은 작업의 특성에 따라 모델 아키텍처에 대한 최소한의 변경으로 추가 된 출력 계층으로 공급됩니다.둘째, 사전 훈련된 트랜스포머 엔코더의 모든 파라미터는 미세 조정되고 추가 출력 레이어는 처음부터 트레이닝됩니다. :numref:`fig_elmo-gpt-bert`는 ElMo, GPT 및 BERT 간의 차이점을 보여줍니다. 

![A comparison of ELMo, GPT, and BERT.](../img/elmo-gpt-bert.svg)
:label:`fig_elmo-gpt-bert`

BERT는 (i) 단일 텍스트 분류 (예: 정서 분석), (ii) 텍스트 쌍 분류 (예: 자연어 추론), (iii) 질문 답변, (iv) 텍스트 태그 지정 (예: 명명된 엔터티 인식) 의 광범위한 범주에서 11가지 자연어 처리 작업에 대한 최첨단 기술을 더욱 개선했습니다..상황에 맞는 ElMo부터 작업에 구애받지 않는 GPT 및 BERT에 이르기까지 2018 년에 제안 된 모든 것은 자연어에 대한 심층적 인 표현에 대한 개념적으로 단순하지만 경험적으로 강력한 사전 교육은 다양한 자연어 처리 작업에 대한 솔루션에 혁명을 일으켰습니다. 

이 장의 나머지 부분에서는 BERT의 사전 교육에 대해 자세히 알아볼 것입니다.자연어 처리 응용 프로그램이 :numref:`chap_nlp_app`에서 설명될 때 다운스트림 응용 프로그램에 대한 BERT의 미세 조정을 설명합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## 입력 표현
:label:`subsec_bert_input_rep`

자연어 처리에서 일부 작업 (예: 감정 분석) 은 단일 텍스트를 입력으로 사용하고 다른 작업 (예: 자연어 추론) 에서는 입력이 텍스트 시퀀스 쌍입니다.BERT 입력 시퀀스는 단일 텍스트와 텍스트 쌍을 명확하게 나타냅니다.전자에서 BERT 입력 시퀀스는 특수 분류 토큰 “<cls>”, 텍스트 시퀀스의 토큰 및 특수 분리 토큰 “<sep>” 의 연결입니다.후자의 경우 BERT 입력 시퀀스는 “<cls>”, 첫 번째 텍스트 시퀀스의 토큰, “<sep>”, 두 번째 텍스트 시퀀스의 토큰 및 “<sep>" 의 연결입니다.“BERT 입력 시퀀스”라는 용어를 다른 유형의 “시퀀스”와 일관되게 구분할 것입니다.예를 들어, 하나의*BERT 입력 시퀀스*에는 하나의*텍스트 시퀀스* 또는 두 개의*텍스트 시퀀스*가 포함될 수 있습니다. 

텍스트 쌍을 구별하기 위해, 학습된 세그먼트 임베딩 ($\mathbf{e}_A$ 및 $\mathbf{e}_B$) 은 각각 제 1 시퀀스 및 제 2 시퀀스의 토큰 임베딩에 추가된다.단일 텍스트 입력의 경우 $\mathbf{e}_A$만 사용됩니다. 

다음 `get_tokens_and_segments`는 한 문장 또는 두 문장을 입력으로 취한 다음 BERT 입력 시퀀스의 토큰과 해당 세그먼트 ID를 반환합니다.

```{.python .input}
#@tab all
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs."""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

BERT는 트랜스포머 엔코더를 양방향 아키텍처로 선택합니다.트랜스포머 엔코더에서 흔히 볼 수 있는 위치 임베딩은 BERT 입력 시퀀스의 모든 위치에 추가됩니다.그러나 원래 트랜스포머 엔코더와는 달리 BERT는*학습 가능* 위치 임베딩을 사용합니다.요약하면 :numref:`fig_bert-input`는 BERT 입력 시퀀스의 임베딩이 토큰 임베딩, 세그먼트 임베딩 및 위치 임베딩의 합계임을 보여줍니다. 

![BERT 입력 시퀀스의 임베딩은 토큰 임베딩, 세그먼트 임베딩 및 위치 임베딩의 합계입니다.](../img/bert-input.svg) :label:`fig_bert-input` 

다음 `BERTEncoder` 클래스는 :numref:`sec_transformer`에서 구현된 `TransformerEncoder` 클래스와 유사합니다.`TransformerEncoder`과 달리 `BERTEncoder`는 세그먼트 임베딩과 학습 가능한 위치 임베딩을 사용합니다.

```{.python .input}
#@save
class BERTEncoder(nn.Block):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(d2l.EncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = self.params.get('pos_embedding',
                                             shape=(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data(ctx=X.ctx)[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

```{.python .input}
#@tab pytorch
#@save
class BERTEncoder(nn.Module):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

어휘 크기가 10000이라고 가정합니다.`BERTEncoder`의 순방향 추론을 보여주기 위해 인스턴스를 만들고 매개 변수를 초기화하겠습니다.

```{.python .input}
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
num_layers, dropout = 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_layers, dropout)
encoder.initialize()
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
```

`tokens`를 길이가 8인 2개의 BERT 입력 시퀀스로 정의합니다. 여기서 각 토큰은 어휘의 인덱스입니다.입력 `tokens`와 함께 `BERTEncoder`의 순방향 추론은 인코딩된 결과를 반환합니다. 여기서 각 토큰은 길이가 하이퍼파라미터 `num_hiddens`에 의해 미리 정의된 벡터로 표시됩니다.이 하이퍼파라미터는 일반적으로 트랜스포머 엔코더의*hidden size* (은닉 유닛 수) 라고 합니다.

```{.python .input}
tokens = np.random.randint(0, vocab_size, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

```{.python .input}
#@tab pytorch
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

## 사전 교육 작업
:label:`subsec_bert_pretraining_tasks`

`BERTEncoder`의 순방향 추론은 입력 텍스트의 각 토큰과 삽입된 특수 토큰 “<cls>” 및 “" 의 BERT 표현을 제공합니다<seq>.다음으로 이러한 표현을 사용하여 BERT 사전 훈련에 대한 손실 함수를 계산합니다.사전 교육은 마스크 언어 모델링과 다음 문장 예측의 두 가지 작업으로 구성됩니다. 

### 마스크 언어 모델링
:label:`subsec_mlm`

:numref:`sec_language_model`에서 볼 수 있듯이 언어 모델은 왼쪽의 컨텍스트를 사용하여 토큰을 예측합니다.각 토큰을 표현하기 위해 양방향으로 컨텍스트를 인코딩하기 위해 BERT는 토큰을 무작위로 마스킹하고 양방향 컨텍스트의 토큰을 사용하여 마스크된 토큰을 자체 감독 방식으로 예측합니다.이 작업을*마스크 언어 모델*이라고 합니다. 

이 사전 학습 과제에서는 토큰의 15% 가 예측을 위해 마스크된 토큰으로 무작위로 선택됩니다.라벨을 사용하여 부정 행위 없이 마스킹된 토큰을 예측하려면, 한 가지 간단한 접근법은 항상 <mask>BERT 입력 시퀀스에서 특별한 “” 토큰으로 교체하는 것입니다.그러나 인위적인 특수 토큰 “<mask>" 은 미세 조정에 나타나지 않습니다.사전 학습과 미세 조정 간의 이러한 불일치를 피하기 위해 토큰이 예측을 위해 마스크 된 경우 (예: “이 영화는 훌륭합니다”에서 마스크되고 예측되도록 “great”가 선택됨) 입력에서 다음과 같이 대체됩니다. 

* <mask>80% 의 시간 동안 특별한 “” 토큰 (예: “이 영화는 훌륭하다”는 “이 영화는 훌륭하다<mask>”가 됨)
* 10% 의 시간 동안 임의의 토큰 (예: “이 영화는 훌륭합니다”는 “이 영화는 음료”가 됨)
* 10% 의 시간 동안 변경되지 않은 레이블 토큰 (예: “이 영화는 훌륭합니다”가 “이 영화는 훌륭합니다”가 됨).

15% 의 시간 중 10% 동안 무작위 토큰이 삽입됩니다.이 가끔 발생하는 노이즈는 양방향 컨텍스트 인코딩에서 BERT가 마스크된 토큰 (특히 레이블 토큰이 변경되지 않은 경우) 으로 편향되지 않도록 합니다. 

BERT 사전 교육의 마스크 언어 모델 작업에서 마스킹된 토큰을 예측하기 위해 다음 `MaskLM` 클래스를 구현합니다.예측에서는 단일 은폐 계층 MLP (`self.mlp`) 를 사용합니다.순방향 추론에서는 인코딩된 결과 `BERTEncoder`와 예측을 위한 토큰 위치의 두 가지 입력이 필요합니다.출력은 이러한 위치에서의 예측 결과입니다.

```{.python .input}
#@save
class MaskLM(nn.Block):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `np.array([0, 0, 0, 1, 1, 1])`
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class MaskLM(nn.Module):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

`MaskLM`의 순방향 추론을 시연하기 위해 인스턴스 `mlm`을 만들고 초기화합니다.`BERTEncoder`의 순방향 추론에서 나온 `encoded_X`은 2개의 BERT 입력 시퀀스를 나타냅니다.`mlm_positions`을 `encoded_X`의 BERT 입력 시퀀스에서 예측할 3개의 인덱스로 정의합니다.`mlm`의 순방향 추론은 `encoded_X`의 모든 마스크된 위치 `mlm_positions`에서 예측 결과 `mlm_Y_hat`를 반환합니다.각 예측에 대해 결과의 크기는 어휘 크기와 같습니다.

```{.python .input}
mlm = MaskLM(vocab_size, num_hiddens)
mlm.initialize()
mlm_positions = np.array([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

```{.python .input}
#@tab pytorch
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

마스크 아래에 예측된 토큰 `mlm_Y_hat`의 지상 실측 레이블 `mlm_Y`를 사용하면 BERT 사전 학습에서 마스킹된 언어 모델 작업의 교차 엔트로피 손실을 계산할 수 있습니다.

```{.python .input}
mlm_Y = np.array([[7, 8, 9], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

```{.python .input}
#@tab pytorch
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

### 다음 문장 예측
:label:`subsec_nsp`

마스크 언어 모델링은 단어를 표현하기 위한 양방향 컨텍스트를 인코딩할 수 있지만 텍스트 쌍 간의 논리적 관계를 명시적으로 모델링하지는 않습니다.두 텍스트 시퀀스 간의 관계를 이해하는 데 도움이 되도록 BERT는 사전 학습에서 이진 분류 작업인*다음 문장 예측*을 고려합니다.사전 학습을 위해 문장 쌍을 생성 할 때 절반 동안 실제로 레이블이 “True”인 연속 문장입니다. 나머지 절반 동안 두 번째 문장은 “False”라는 레이블로 코퍼스에서 무작위로 샘플링됩니다. 

다음 `NextSentencePred` 클래스는 한 개의 숨겨진 계층 MLP를 사용하여 두 번째 문장이 BERT 입력 시퀀스에서 첫 번째 문장의 다음 문장인지 여부를 예측합니다.트랜스포머 엔코더의 자체 주의로 인해 특수 토큰 “<cls>" 의 BERT 표현은 입력에서 두 문장을 모두 인코딩합니다.따라서, MLP 분류기의 출력 계층 (`self.output`) 은 `X`를 입력으로 취하고, 여기서 `X`는 입력이 인코딩된 “<cls>” 토큰인 MLP 은닉 계층의 출력이다.

```{.python .input}
#@save
class NextSentencePred(nn.Block):
    """The next sentence prediction task of BERT."""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Dense(2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

```{.python .input}
#@tab pytorch
#@save
class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT."""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

`NextSentencePred` 인스턴스의 순방향 추론이 각 BERT 입력 시퀀스에 대한 이진 예측을 반환한다는 것을 알 수 있습니다.

```{.python .input}
nsp = NextSentencePred()
nsp.initialize()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

```{.python .input}
#@tab pytorch
# PyTorch by default won't flatten the tensor as seen in mxnet where, if
# flatten=True, all but the first axis of input data are collapsed together
encoded_X = torch.flatten(encoded_X, start_dim=1)
# input_shape for NSP: (batch size, `num_hiddens`)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

두 이진 분류의 교차 엔트로피 손실도 계산할 수 있습니다.

```{.python .input}
nsp_y = np.array([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

```{.python .input}
#@tab pytorch
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

앞서 언급한 두 가지 사전 교육 작업의 모든 레이블은 수동 라벨링 작업 없이 사전 교육 코퍼스에서 간단하게 얻을 수 있다는 점은 주목할 만합니다.원래 BERT는 북코퍼스 :cite:`Zhu.Kiros.Zemel.ea.2015`와 영어 위키백과의 연결에 대해 사전 훈련되었습니다.이 두 텍스트 코퍼라는 엄청납니다. 각각 8억 단어와 25억 개의 단어가 있습니다. 

## 모든 것을 하나로 모으다

BERT를 사전 훈련할 때 최종 손실 함수는 마스크 언어 모델링을 위한 손실 함수와 다음 문장 예측의 선형 조합입니다.이제 세 가지 클래스 `BERTEncoder`, `MaskLM` 및 `NextSentencePred`을 인스턴스화하여 `BERTModel` 클래스를 정의할 수 있습니다.순방향 추론은 인코딩된 BERT 표현 `encoded_X`, 마스킹된 언어 모델링의 예측 `mlm_Y_hat` 및 다음 문장 예측 `nsp_Y_hat`을 반환합니다.

```{.python .input}
#@save
class BERTModel(nn.Block):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_layers, dropout, max_len)
        self.hidden = nn.Dense(num_hiddens, activation='tanh')
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class BERTModel(nn.Module):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

## 요약

* word2vec 및 GLOve와 같은 단어 임베딩 모델은 문맥에 독립적입니다.단어의 문맥 (있는 경우) 에 관계없이 동일한 사전 훈련된 벡터를 동일한 단어에 할당합니다.자연어로 다분성이나 복잡한 의미를 잘 다루기가 어렵습니다.
* ElMo 및 GPT와 같은 상황에 맞는 단어 표현의 경우 단어의 표현은 문맥에 따라 다릅니다.
* ElMo는 컨텍스트를 양방향으로 인코딩하지만 작업 별 아키텍처를 사용합니다 (그러나 모든 자연어 처리 작업에 대해 특정 아키텍처를 만드는 것은 거의 중요하지 않습니다). GPT는 작업에 구애받지 않지만 컨텍스트를 왼쪽에서 오른쪽으로 인코딩합니다.
* BERT는 두 세계의 장점을 결합합니다. 컨텍스트를 양방향으로 인코딩하고 광범위한 자연어 처리 작업을 위해 최소한의 아키텍처 변경이 필요합니다.
* BERT 입력 시퀀스의 임베딩은 토큰 임베딩, 세그먼트 임베딩 및 위치 임베딩의 합계입니다.
* 사전 학습 BERT는 마스크 언어 모델링과 다음 문장 예측의 두 가지 작업으로 구성됩니다.전자는 단어를 표현하기 위해 양방향 컨텍스트를 인코딩 할 수 있고 후자는 텍스트 쌍 간의 논리적 관계를 명시 적으로 모델링 할 수 있습니다.

## 연습문제

1. BERT는 왜 성공할까요?
1. 다른 모든 것들이 동일하다면, 마스크 언어 모델은 왼쪽에서 오른쪽 언어 모델보다 수렴하기 위해 더 많거나 적은 사전 훈련 단계가 필요합니까?왜요?
1. BERT의 원래 구현에서 `BERTEncoder`의 위치별 피드 포워드 네트워크 (`d2l.EncoderBlock`를 통해) 와 `MaskLM`의 완전 연결 계층은 모두 가우스 오류 선형 단위 (GELU) :cite:`Hendrycks.Gimpel.2016`를 활성화 함수로 사용합니다.GELU와 ReLU의 차이점을 연구합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/388)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1490)
:end_tab:
