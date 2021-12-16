# BERT 사전 교육
:label:`sec_bert-pretraining`

:numref:`sec_bert`에서 구현된 BERT 모델과 :numref:`sec_bert-dataset`의 위키텍스트-2 데이터세트에서 생성된 사전 학습 예제를 통해 이 섹션의 위키텍스트-2 데이터세트에 대해 BERT를 사전 학습할 것입니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

우선, 위키텍스트-2 데이터세트를 마스크 언어 모델링과 다음 문장 예측을 위한 사전 학습 예제의 미니배치로 로드합니다.배치 크기는 512이고 BERT 입력 시퀀스의 최대 길이는 64입니다.원래 BERT 모델에서 최대 길이는 512입니다.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```

## BERT 사전 교육

원래 BERT에는 서로 다른 모델 크기 :cite:`Devlin.Chang.Lee.ea.2018`의 두 가지 버전이 있습니다.기본 모델 ($\text{BERT}_{\text{BASE}}$) 은 768개의 히든 유닛 (숨겨진 크기) 과 12개의 자체 주의 헤드가 있는 12개의 레이어 (트랜스포머 엔코더 블록) 를 사용합니다.대형 모델 ($\text{BERT}_{\text{LARGE}}$) 은 1024개의 히든 유닛과 16개의 자습 헤드가 있는 24개의 레이어를 사용합니다.특히 전자는 1억 1천만 개의 매개변수를 가지고 있고 후자는 3억 4천만 개의 매개변수를 가지고 있습니다.쉽게 시연할 수 있도록 레이어 2개, 숨겨진 유닛 128개, 자기 주의 헤드 2개를 사용하여 작은 BERT를 정의합니다.

```{.python .input}
net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_layers=2, dropout=0.2)
devices = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=devices)
loss = gluon.loss.SoftmaxCELoss()
```

```{.python .input}
#@tab pytorch
net = d2l.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                    num_layers=2, dropout=0.2, key_size=128, query_size=128,
                    value_size=128, hid_in_features=128, mlm_in_features=128,
                    nsp_in_features=128)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()
```

훈련 루프를 정의하기 전에 도우미 함수 `_get_batch_loss_bert`를 정의합니다.훈련 예제의 샤드가 주어지면 이 함수는 마스크된 언어 모델링과 다음 문장 예측 작업 모두에 대한 손실을 계산합니다.BERT 사전 훈련의 최종 손실은 마스크된 언어 모델링 손실과 다음 문장 예측 손실의 합에 불과합니다.

```{.python .input}
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X_shards,
                         segments_X_shards, valid_lens_x_shards,
                         pred_positions_X_shards, mlm_weights_X_shards,
                         mlm_Y_shards, nsp_y_shards):
    mlm_ls, nsp_ls, ls = [], [], []
    for (tokens_X_shard, segments_X_shard, valid_lens_x_shard,
         pred_positions_X_shard, mlm_weights_X_shard, mlm_Y_shard,
         nsp_y_shard) in zip(
        tokens_X_shards, segments_X_shards, valid_lens_x_shards,
        pred_positions_X_shards, mlm_weights_X_shards, mlm_Y_shards,
        nsp_y_shards):
        # Forward pass
        _, mlm_Y_hat, nsp_Y_hat = net(
            tokens_X_shard, segments_X_shard, valid_lens_x_shard.reshape(-1),
            pred_positions_X_shard)
        # Compute masked language model loss
        mlm_l = loss(
            mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y_shard.reshape(-1),
            mlm_weights_X_shard.reshape((-1, 1)))
        mlm_l = mlm_l.sum() / (mlm_weights_X_shard.sum() + 1e-8)
        # Compute next sentence prediction loss
        nsp_l = loss(nsp_Y_hat, nsp_y_shard)
        nsp_l = nsp_l.mean()
        mlm_ls.append(mlm_l)
        nsp_ls.append(nsp_l)
        ls.append(mlm_l + nsp_l)
        npx.waitall()
    return mlm_ls, nsp_ls, ls
```

```{.python .input}
#@tab pytorch
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

앞서 언급한 두 도우미 함수를 호출하는 다음 `train_bert` 함수는 위키텍스트-2 (`train_iter`) 데이터세트에서 BERT (`net`) 를 사전 훈련시키는 절차를 정의합니다.BERT 교육은 매우 오래 걸릴 수 있습니다.다음 함수의 입력값 `num_steps`는 `train_ch13` 함수 (:numref:`sec_image_augmentation` 참조) 에서와 같이 훈련을 위한 Epoch 수를 지정하는 대신 훈련에 대한 반복 단계 수를 지정합니다.

```{.python .input}
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 0.01})
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch in train_iter:
            (tokens_X_shards, segments_X_shards, valid_lens_x_shards,
             pred_positions_X_shards, mlm_weights_X_shards,
             mlm_Y_shards, nsp_y_shards) = [gluon.utils.split_and_load(
                elem, devices, even_split=False) for elem in batch]
            timer.start()
            with autograd.record():
                mlm_ls, nsp_ls, ls = _get_batch_loss_bert(
                    net, loss, vocab_size, tokens_X_shards, segments_X_shards,
                    valid_lens_x_shards, pred_positions_X_shards,
                    mlm_weights_X_shards, mlm_Y_shards, nsp_y_shards)
            for l in ls:
                l.backward()
            trainer.step(1)
            mlm_l_mean = sum([float(l) for l in mlm_ls]) / len(mlm_ls)
            nsp_l_mean = sum([float(l) for l in nsp_ls]) / len(nsp_ls)
            metric.add(mlm_l_mean, nsp_l_mean, batch[0].shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

BERT 사전 훈련 중에 마스크 언어 모델링 손실과 다음 문장 예측 손실을 모두 그릴 수 있습니다.

```{.python .input}
#@tab all
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

## BERT로 텍스트 표현하기

BERT를 사전 학습한 후 이를 사용하여 단일 텍스트, 텍스트 쌍 또는 그 안의 토큰을 나타낼 수 있습니다.다음 함수는 `tokens_a` 및 `tokens_b`의 모든 토큰에 대한 버트 (`net`) 표현을 반환합니다.

```{.python .input}
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = np.expand_dims(np.array(vocab[tokens], ctx=devices[0]),
                               axis=0)
    segments = np.expand_dims(np.array(segments, ctx=devices[0]), axis=0)
    valid_len = np.expand_dims(np.array(len(tokens), ctx=devices[0]), axis=0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

```{.python .input}
#@tab pytorch
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

“크레인이 날고있다”라는 문장을 생각해보십시오.:numref:`subsec_bert_input_rep`에서 논의한 대로 BERT의 입력 표현을 회상합니다.특수 토큰 “<cls>" (분류에 사용됨) 과 “<sep>" (분리에 사용됨) 을 삽입한 후 BERT 입력 시퀀스의 길이는 6입니다.0은 “<cls>” 토큰의 인덱스이므로 `encoded_text[:, 0, :]`는 전체 입력 문장의 BERT 표현입니다.폴리 세미 토큰 “크레인”을 평가하기 위해 토큰의 BERT 표현의 처음 세 요소도 인쇄합니다.

```{.python .input}
#@tab all
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Tokens: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

이제 “크레인 운전사가 왔습니다”와 “방금 떠났습니다”라는 문장 쌍을 생각해보십시오.마찬가지로 `encoded_pair[:, 0, :]`는 사전 훈련된 BERT의 전체 문장 쌍의 인코딩된 결과입니다.polysemy 토큰 “crane”의 처음 세 요소는 컨텍스트가 다를 때의 요소와 다릅니다.이는 BERT 표현이 상황에 따라 달라지도록 지원합니다.

```{.python .input}
#@tab all
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# Tokens: '<cls>', 'a', 'crane', 'driver', 'came', '<sep>', 'he', 'just',
# 'left', '<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```

:numref:`chap_nlp_app`에서는 다운스트림 자연어 처리 애플리케이션을 위해 사전 훈련된 BERT 모델을 미세 조정할 것입니다. 

## 요약

* 원래 BERT에는 두 가지 버전이 있으며 기본 모델에는 1 억 1 천만 개의 매개 변수가 있고 대형 모델에는 3 억 4 천만 개의 매개 변수가 있습니다.
* BERT를 사전 학습한 후 이를 사용하여 단일 텍스트, 텍스트 쌍 또는 그 안의 토큰을 나타낼 수 있습니다.
* 실험에서 동일한 토큰은 컨텍스트가 다를 때 다른 BERT 표현을 갖습니다.이는 BERT 표현이 상황에 따라 달라지도록 지원합니다.

## 연습문제

1. 실험에서 마스크 언어 모델링 손실이 다음 문장 예측 손실보다 훨씬 높다는 것을 알 수 있습니다.왜요?
2. BERT 입력 시퀀스의 최대 길이를 512로 설정합니다 (원래 BERT 모델과 동일).$\text{BERT}_{\text{LARGE}}$와 같은 원래 BERT 모델의 구성을 사용합니다.이 섹션을 실행할 때 오류가 발생합니까?왜요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/390)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1497)
:end_tab:
