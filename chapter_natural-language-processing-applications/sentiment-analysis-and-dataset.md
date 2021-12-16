# 감성 분석 및 데이터세트
:label:`sec_sentiment`

온라인 소셜 미디어 및 리뷰 플랫폼이 확산됨에 따라 수많은 독단적 데이터가 기록되어 의사 결정 프로세스를 지원할 가능성이 큽니다.
*감성 분석*
제품 리뷰, 블로그 댓글 및 포럼 토론과 같은 제작 텍스트에서 사람들의 감정을 연구합니다.정치 (예: 정책에 대한 대중의 정서 분석), 금융 (예: 시장 정서 분석) 및 마케팅 (예: 제품 연구 및 브랜드 관리) 과 같은 다양한 분야에 폭넓게 적용됩니다. 

센티멘트는 개별 극성이나 척도 (예: 양수 및 음수) 로 분류될 수 있으므로 센티멘트 분석을 텍스트 분류 작업으로 간주하여 다양한 길이의 텍스트 시퀀스를 고정 길이 텍스트 범주로 변환할 수 있습니다.이 장에서는 감정 분석에 스탠포드의 [대형 영화 리뷰 데이터 세트](https://ai.stanford.edu/~amaas/data/sentiment/) 를 사용합니다.교육 세트와 테스트 세트로 구성되며 IMDb에서 다운로드한 25000개의 영화 리뷰가 포함되어 있습니다.두 데이터셋에는 동일한 수의 “양수” 레이블과 “음수” 레이블이 있으며 이는 서로 다른 센티멘트 극성을 나타냅니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

##  데이터세트 읽기

먼저 `../data/aclImdb` 경로에서 이 IMDb 검토 데이터 세트를 다운로드하여 추출합니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

다음으로 훈련 및 테스트 데이터세트를 읽습니다.각 예는 리뷰와 레이블입니다. “양수”의 경우 1, “음수”의 경우 0입니다.

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[0:60])
```

## 데이터세트 사전 처리

각 단어를 토큰으로 취급하고 5회 미만으로 나타나는 단어를 필터링하여 학습 데이터 세트에서 어휘를 만듭니다.

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

토큰화 후 검토 길이의 히스토그램을 토큰으로 플로팅해 보겠습니다.

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

예상대로 리뷰의 길이는 다양합니다.이러한 리뷰의 미니 배치를 매번 처리하기 위해 각 리뷰의 길이를 잘라내기 및 패딩과 함께 500으로 설정했습니다. 이는 :numref:`sec_machine_translation`의 기계 번역 데이터 세트에 대한 전처리 단계와 유사합니다.

```{.python .input}
#@tab all
num_steps = 500  # sequence length
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## 데이터 반복기 만들기

이제 데이터 이터레이터를 만들 수 있습니다.각 반복마다 예제의 미니배치가 반환됩니다.

```{.python .input}
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

```{.python .input}
#@tab pytorch
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

## 모든 것을 하나로 모으다

마지막으로 위의 단계를 `load_data_imdb` 함수로 마무리합니다.훈련 및 테스트 데이터 반복자와 IMDb 검토 데이터 세트의 어휘를 반환합니다.

```{.python .input}
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## 요약

* 감성 분석은 생성된 텍스트에서 사람들의 감정을 연구하며, 이는 다양한 길이의 텍스트 시퀀스를 변환하는 텍스트 분류 문제로 간주됩니다.
고정 길이 텍스트 범주로 변환합니다.
* 전처리 후 스탠포드의 대형 영화 리뷰 데이터 세트 (IMDb 리뷰 데이터 세트) 를 어휘와 함께 데이터 반복기에 로드할 수 있습니다.

## 연습문제

1. 교육 감정 분석 모델을 가속화하기 위해 이 섹션에서 수정할 수 있는 하이퍼파라미터는 무엇입니까?
1. 감성 분석을 위해 [Amazon reviews](https://snap.stanford.edu/data/web-Amazon.html)의 데이터세트를 데이터 반복기 및 레이블에 로드하는 함수를 구현할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/391)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1387)
:end_tab:
