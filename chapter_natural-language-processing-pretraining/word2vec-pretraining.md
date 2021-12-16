# 사전 교육 단어2vec
:label:`sec_word2vec_pretraining`

계속해서 :numref:`sec_word2vec`에 정의된 스킵 그램 모델을 구현합니다.그런 다음 PTB 데이터 세트에서 음수 샘플링을 사용하여 word2vec을 사전 학습합니다.먼저 :numref:`sec_word2vec_data`에 설명된 `d2l.load_data_ptb` 함수를 호출하여 이 데이터 세트의 데이터 반복자와 어휘를 구해 보겠습니다.

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## 스킵 그램 모델

임베딩 레이어와 배치 행렬 곱셈을 사용하여 skip-gram 모델을 구현합니다.먼저 레이어 임베딩 작동 방식을 살펴보겠습니다. 

### 임베딩 레이어

:numref:`sec_seq2seq`에 설명된 대로 임베딩 레이어는 토큰의 인덱스를 특징 벡터에 매핑합니다.이 계층의 가중치는 행 수가 딕셔너리 크기 (`input_dim`) 와 같고 열 개수가 각 토큰의 벡터 차원 (`output_dim`) 과 동일한 행렬입니다.단어 임베딩 모델을 훈련시킨 후, 이 가중치가 우리에게 필요한 것입니다.

```{.python .input}
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.python .input}
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

임베딩 레이어의 입력은 토큰 (단어) 의 인덱스입니다.임의의 토큰 인덱스 $i$에 대해, 그 벡터 표현은 임베딩 계층에서 가중치 행렬의 $i^\mathrm{th}$ 행으로부터 얻어질 수 있다.벡터 차원 (`output_dim`) 이 4로 설정되었으므로 임베딩 계층은 형상 (2, 3) 을 갖는 토큰 인덱스의 미니배치에 대해 형상 (2, 3, 4) 의 벡터를 반환합니다.

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### 순방향 전파 정의

순방향 전파에서, 스킵-그램 모델의 입력은 형상의 중심 워드 인덱스 `center` (배치 크기, 1) 및 모양의 연결된 컨텍스트 및 노이즈 워드 인덱스 `contexts_and_negatives` (배치 크기, `max_len`) 을 포함하며, 여기서 `max_len`은 :numref:`subsec_word2vec-minibatch-loading`에 정의된다.이 두 변수는 먼저 임베딩 계층을 통해 토큰 인덱스에서 벡터로 변환된 다음 배치 행렬 곱셈 (:numref:`subsec_batch_dot`에 설명) 은 모양의 출력값 (배치 크기, 1, `max_len`) 을 반환합니다.출력값의 각 요소는 중심 워드 벡터와 컨텍스트 또는 노이즈 워드 벡터의 내적이 됩니다.

```{.python .input}
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```{.python .input}
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

몇 가지 예제 입력에 대해이 `skip_gram` 함수의 출력 모양을 인쇄해 보겠습니다.

```{.python .input}
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## 트레이닝

음수 샘플링으로 skip-gram 모델을 훈련시키기 전에 먼저 손실 함수를 정의하겠습니다. 

### 이진 교차 엔트로피 손실

:numref:`subsec_negative-sampling`에서 음수 샘플링에 대한 손실 함수의 정의에 따르면 이진 교차 엔트로피 손실을 사용합니다.

```{.python .input}
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # Binary cross-entropy loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

:numref:`subsec_word2vec-minibatch-loading`에서 마스크 변수와 레이블 변수에 대한 설명을 기억해 보십시오.다음은 주어진 변수에 대한 이진 교차 엔트로피 손실을 계산합니다.

```{.python .input}
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

아래는 이진 교차 엔트로피 손실에서 시그모이드 활성화 함수를 사용하여 위의 결과를 (덜 효율적인 방식으로) 계산하는 방법을 보여줍니다.두 출력을 마스크되지 않은 예측에 대해 평균화된 두 개의 정규화된 손실로 간주할 수 있습니다.

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### 모델 매개변수 초기화

어휘의 모든 단어가 각각 중심 단어와 문맥 단어로 사용될 때 두 개의 임베딩 레이어를 정의합니다.단어 벡터 차원 `embed_size`는 100으로 설정됩니다.

```{.python .input}
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```{.python .input}
#@tab pytorch
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### 훈련 루프 정의

훈련 루프는 아래에 정의되어 있습니다.패딩이 존재하기 때문에 손실 함수의 계산은 이전 훈련 함수와 약간 다릅니다.

```{.python .input}
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

이제 음수 샘플링을 사용하여 스킵 그램 모델을 훈련시킬 수 있습니다.

```{.python .input}
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## 단어 임베딩 적용
:label:`subsec_apply-word-embed`

word2vec 모델을 훈련 한 후 훈련 된 모델의 단어 벡터의 코사인 유사성을 사용하여 사전에서 입력 단어와 가장 의미 적으로 유사한 단어를 찾을 수 있습니다.

```{.python .input}
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## 요약

* 임베딩 레이어와 이진 교차 엔트로피 손실을 사용하여 네거티브 샘플링으로 스킵 그램 모델을 훈련시킬 수 있습니다.
* 단어 임베딩의 적용은 워드 벡터의 코사인 유사성에 기초하여 주어진 단어에 대해 의미론적으로 유사한 단어를 찾는 것을 포함한다.

## 연습문제

1. 훈련된 모델을 사용하여 다른 입력 단어에 대해 의미적으로 유사한 단어를 찾습니다.하이퍼파라미터를 조정하여 결과를 개선할 수 있습니까?
1. 훈련 코퍼스가 큰 경우 모델 매개 변수를 업데이트 할 때* 현재 미니 배치*의 중심 단어에 대한 컨텍스트 단어와 노이즈 단어를 샘플링하는 경우가 많습니다.즉, 동일한 중심 단어는 훈련 시대마다 다른 문맥 단어 또는 잡음 단어를 가질 수 있습니다.이 방법의 이점은 무엇입니까?이 교육 방법을 구현해 보십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/384)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1335)
:end_tab:
