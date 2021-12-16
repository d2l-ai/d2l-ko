# 바다나우 주의

:label:`sec_seq2seq_attention` 

우리는 :numref:`sec_seq2seq`에서 기계 번역 문제를 연구했으며, 여기서 시퀀스 대 시퀀스 학습을 위해 두 개의 RNN을 기반으로 인코더-디코더 아키텍처를 설계했습니다.구체적으로, RNN 인코더는 가변 길이 시퀀스를 고정 형상 컨텍스트 변수로 변환한 후, RNN 디코더는 생성된 토큰 및 컨텍스트 변수에 기초하여 출력 (타겟) 시퀀스 토큰을 토큰별로 생성한다.그러나 모든 입력 (소스) 토큰이 특정 토큰을 디코딩하는 데 유용하지는 않지만 전체 입력 시퀀스를 인코딩하는*동일한* 컨텍스트 변수는 여전히 각 디코딩 단계에서 사용됩니다. 

주어진 텍스트 시퀀스에 대한 필기 생성과 관련하여 별도이지만 관련된 과제에서 Graves는 정렬이 한 방향 :cite:`Graves.2013`로만 이동하는 훨씬 더 긴 펜 트레이스에 텍스트 문자를 정렬하기 위해 차별화 가능한 주의 모델을 설계했습니다.정렬 학습이라는 아이디어에서 영감을 얻은 Bahdanau et al. 은 심각한 단방향 정렬 제한 :cite:`Bahdanau.Cho.Bengio.2014`없이 차별화 가능한 주의 모델을 제안했습니다.토큰을 예측할 때 모든 입력 토큰이 관련성이 없는 경우 모델은 현재 예측과 관련된 입력 시퀀스 부분에만 정렬 (또는 참석) 합니다.이는 컨텍스트 변수를 주의력 풀링의 출력으로 처리함으로써 달성됩니다. 

## 모델

아래에서 RNN 인코더-디코더에 대한 Bahdanau의 관심을 설명 할 때 :numref:`sec_seq2seq`에서도 동일한 표기법을 따를 것입니다.새로운 주의 기반 모델은 :numref:`sec_seq2seq`의 컨텍스트 변수 $\mathbf{c}$이 임의의 디코딩 시간 스텝 $t'$에서 $\mathbf{c}_{t'}$로 대체된다는 점을 제외하면 :numref:`sec_seq2seq`의 모델과 동일합니다.입력 시퀀스에 $T$ 토큰이 있다고 가정하면 디코딩 시간 단계 $t'$의 컨텍스트 변수가 주의 풀링의 출력입니다. 

$$\mathbf{c}_{t'} = \sum_{t=1}^T \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_t) \mathbf{h}_t,$$

여기서 시간 스텝 $t' - 1$에서의 디코더 은닉 상태 ($\mathbf{s}_{t' - 1}$) 는 쿼리이고, 인코더 은닉 상태 ($\mathbf{h}_t$) 는 키 및 값 모두이고, 주의 가중치 ($\alpha$) 는 :eqref:`eq_additive-attn`에 의해 정의된 가산 주의 스코어링 함수를 사용하여 :eqref:`eq_attn-scoring-alpha`에서와 같이 계산된다. 

:numref:`fig_seq2seq_details`의 바닐라 RNN 인코더-디코더 아키텍처와는 약간 다른 점은 :numref:`fig_s2s_attention_details`에는 바다나우의 주목을 받은 동일한 아키텍처가 묘사되어 있습니다. 

![Layers in an RNN encoder-decoder model with Bahdanau attention.](../img/seq2seq-attention-details.svg)
:label:`fig_s2s_attention_details`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn, nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## 주의를 기울여 디코더 정의하기

Bahdanau의 주의를 기울여 RNN 인코더-디코더를 구현하려면 디코더를 재정의하기만 하면 됩니다.학습된 주의 가중치를 보다 편리하게 시각화하기 위해 다음 `AttentionDecoder` 클래스는 [**주의 메커니즘이 있는 디코더의 기본 인터페이스**] 를 정의합니다.

```{.python .input}
#@tab all
#@save
class AttentionDecoder(d2l.Decoder):
    """The base attention-based decoder interface."""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
```

이제 다음 `Seq2SeqAttentionDecoder` 클래스에서 [**바다나우의 주의를 기울여 RNN 디코더를 구현**] 하겠습니다.디코더의 상태는 (i) 모든 시간 스텝들에서 인코더 최종-계층 은닉 상태들 (주목의 키 및 값들로서); (ii) (디코더의 은닉 상태를 초기화하기 위해) 최종 시간 스텝에서의 인코더 전계층 은닉 상태; 및 (iii) 인코더 유효 길이 (를 제외하기 위해) 로 초기화된다.관심 풀링의 패딩 토큰).각각의 디코딩 시간 단계에서, 이전 시간 스텝에서의 디코더 최종-계층 은닉 상태가 주목의 쿼리로서 사용된다.그 결과, 주의 출력과 입력 임베딩이 모두 RNN 디코더의 입력으로 연결됩니다.

```{.python .input}
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs`: (`num_steps`, `batch_size`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            query = np.expand_dims(hidden_state[0][-1], axis=1)
            # Shape of `context`: (`batch_size`, 1, `num_hiddens`)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # Reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs`:
        # (`num_steps`, `batch_size`, `vocab_size`)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab pytorch
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs`: (`num_steps`, `batch_size`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of `context`: (`batch_size`, 1, `num_hiddens`)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs`:
        # (`num_steps`, `batch_size`, `vocab_size`)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab tensorflow
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens,
                                               num_hiddens, dropout)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]),
                                      return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`, `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs, hidden_state, enc_valid_lens)

    def call(self, X, state, **kwargs):
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`, `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X) # Input `X` has shape: (`batch_size`, `num_steps`)
        X = tf.transpose(X, perm=(1, 0, 2))
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            query = tf.expand_dims(hidden_state[-1], axis=1)
            # Shape of `context`: (`batch_size, 1, `num_hiddens`)
            context = self.attention(query, enc_outputs, enc_outputs,
                                     enc_valid_lens, **kwargs)
            # Concatenate on the feature dimension
            x = tf.concat((context, tf.expand_dims(x, axis=1)), axis=-1)
            out = self.rnn(x, hidden_state, **kwargs)
            hidden_state = out[1:]
            outputs.append(out[0])
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs`:
        # (`batch_size`, `num_steps`, `vocab_size`)
        outputs = self.dense(tf.concat(outputs, axis=1))
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

다음에서는 7개의 시간 스텝으로 구성된 4개의 시퀀스 입력으로 구성된 미니배치를 사용하여 Bahdanau의 주의를 기울여 [**구현된 디코더를 테스트**] 합니다.

```{.python .input}
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.initialize()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.initialize()
X = d2l.zeros((4, 7))  # (`batch_size`, `num_steps`)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

```{.python .input}
#@tab pytorch
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.eval()
X = d2l.zeros((4, 7), dtype=torch.long)  # (`batch_size`, `num_steps`)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

```{.python .input}
#@tab tensorflow
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
X = tf.zeros((4, 7))
state = decoder.init_state(encoder(X, training=False), None)
output, state = decoder(X, state, training=False)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

## [**교육**]

:numref:`sec_seq2seq_training`와 유사하게 여기서는 하이퍼파어미터를 지정하고, Bahdanau의 주의를 기울여 인코더와 디코더를 인스턴스화하고, 기계 번역을 위해 이 모델을 훈련시킵니다.새로 추가된 주의력 메커니즘으로 인해 이 훈련은 주의 메커니즘이 없는 :numref:`sec_seq2seq_training`보다 훨씬 느립니다.

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

모델을 학습한 후 [**영어 문장 몇 개를 번역**] 프랑스어로 번역하고 BLEU 점수를 계산하는 데 사용합니다.

```{.python .input}
#@tab mxnet, pytorch
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab tensorflow
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab all
attention_weights = d2l.reshape(
    d2l.concat([step[0][0][0] for step in dec_attention_weight_seq], 0),
    (1, 1, -1, num_steps))
```

마지막 영어 문장을 번역할 때 [**주의 가중치를 시각화**] 하면 각 쿼리가 키-값 쌍에 대해 균일하지 않은 가중치를 할당하는 것을 확인할 수 있습니다.각 디코딩 단계에서 입력 시퀀스의 서로 다른 부분이 주의력 풀링에서 선택적으로 집계된다는 것을 보여줍니다.

```{.python .input}
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1],
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
#@tab pytorch
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
#@tab tensorflow
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Key posistions', ylabel='Query posistions')
```

## 요약

* 토큰을 예측할 때 모든 입력 토큰이 적절하지 않은 경우 Bahdanau가 관심을 갖는 RNN 인코더-디코더는 입력 시퀀스의 다른 부분을 선택적으로 집계합니다.이는 컨텍스트 변수를 가산적 주의력 풀링의 출력으로 처리함으로써 달성됩니다.
* RNN 인코더-디코더에서 Bahdanau attention은 이전 시간 스텝의 디코더 은닉 상태를 쿼리로 취급하고 모든 시간 스텝의 인코더 은닉 상태를 키와 값으로 취급합니다.

## 연습문제

1. 실험에서 GRU를 LSTM으로 대체합니다.
1. 가법 주의력 점수 함수를 스케일링된 내적으로 대체하도록 실험을 수정합니다.교육 효율성에 어떤 영향을 미칩니 까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/347)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1065)
:end_tab:
