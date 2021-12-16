# 단어 유사성 및 유추
:label:`sec_synonyms`

:numref:`sec_word2vec_pretraining`에서는 작은 데이터 세트에서 word2vec 모델을 훈련시키고 입력 단어에 대해 의미 론적으로 유사한 단어를 찾기 위해 적용했습니다.실제로 대체에 대해 사전 훈련된 워드 벡터는 다운스트림 자연어 처리 작업에 적용할 수 있으며, 이는 나중에 :numref:`chap_nlp_app`에서 다룰 예정입니다.큰 상체에서 사전 훈련 된 단어 벡터의 의미를 간단하게 보여주기 위해 유사성 및 비유 작업이라는 단어에 적용해 보겠습니다.

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

## 사전 훈련된 워드 벡터 불러오기

아래에는 차원 50, 100 및 300의 사전 훈련된 글러브 임베딩이 나열되어 있으며, 이는 [GloVe website](https://nlp.stanford.edu/projects/glove/)에서 다운로드할 수 있습니다.사전 학습된 FastText 임베딩은 여러 언어로 제공됩니다.여기서는 [fastText website](https://fasttext.cc/)에서 다운로드할 수 있는 영어 버전 (300차원 “위키.en”) 을 고려합니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

이러한 사전 훈련된 GLOVE 및 FastText 임베딩을 로드하기 위해 다음 `TokenEmbedding` 클래스를 정의합니다.

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

아래에는 50차원 GLOVE 임베딩 (위키백과 하위 집합에서 사전 훈련됨) 이 로드됩니다.`TokenEmbedding` 인스턴스를 만들 때 지정된 포함 파일이 아직 다운로드되지 않은 경우 다운로드해야 합니다.

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

어휘 크기를 출력합니다.어휘에는 400000 단어 (토큰) 와 알려지지 않은 특별한 토큰이 포함되어 있습니다.

```{.python .input}
#@tab all
len(glove_6b50d)
```

어휘에서 단어의 색인을 얻을 수 있고 그 반대도 마찬가지입니다.

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## 사전 훈련된 워드 벡터 적용하기

로드 된 GLOVE 벡터를 사용하여 다음 단어 유사성 및 유추 작업에 적용하여 의미를 보여줍니다. 

### 단어 유사성

:numref:`subsec_apply-word-embed`와 마찬가지로 단어 벡터 간의 코사인 유사성을 기반으로 입력 단어에 대해 의미 론적으로 유사한 단어를 찾기 위해 다음 `knn` ($k$-최근접이웃) 함수를 구현합니다.

```{.python .input}
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

그런 다음 `TokenEmbedding` 인스턴스 `embed`에서 사전 훈련된 단어 벡터를 사용하여 유사한 단어를 검색합니다.

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

`glove_6b50d`에서 사전 훈련된 단어 벡터의 어휘에는 400000개의 단어와 알려지지 않은 특수 토큰이 포함되어 있습니다.입력 단어와 알려지지 않은 토큰을 제외하고이 어휘 중에서 “칩”이라는 단어와 가장 의미 적으로 유사한 세 단어를 찾을 수 있습니다.

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

아래는 “아기”와 “아름다운”과 비슷한 단어를 출력합니다.

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### 워드 비유

비슷한 단어를 찾는 것 외에도 단어 비유 작업에 단어 벡터를 적용 할 수도 있습니다.예를 들어, “남자”: “여자”:: “아들”: “딸”은 단어 비유의 형태입니다. “남자”는 “아들”이 “딸”이기 때문에 “여자”입니다.구체적으로, 비유 완료 작업이라는 단어는 다음과 같이 정의할 수 있습니다. 단어 비유 $a : b :: c : d$의 경우 처음 세 단어 $a$, $b$ 및 $c$가 주어지면 $d$를 찾습니다.단어 $w$의 벡터를 $\text{vec}(w)$로 나타냅니다.비유를 완성하기 위해 벡터가 $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$의 결과와 가장 유사한 단어를 찾습니다.

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

로드 된 단어 벡터를 사용하여 “남성-여성”비유를 확인하겠습니다.

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

아래는 “수도 국가”비유를 완성합니다: “베이징”: “중국”:: “도쿄”: “일본”.이것은 사전 훈련된 워드 벡터의 의미를 보여줍니다.

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

“bad”: “worst”:: “big”: “big”: “big”와 같은 “형용사-최상급 형용사”비유의 경우 사전 훈련 된 단어 벡터가 구문 정보를 포착 할 수 있음을 알 수 있습니다.

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

사전 훈련 된 단어 벡터에서 과거 시제에 대한 캡처 된 개념을 보여주기 위해 “현재 시제-과거 시제”비유: “do”: “did”: “go”: “went”를 사용하여 구문을 테스트 할 수 있습니다.

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## 요약

* 실제로 대체에 대해 사전 훈련된 워드 벡터는 다운스트림 자연어 처리 작업에 적용할 수 있습니다.
* 사전 훈련된 단어 벡터를 단어 유사성 및 유추 작업에 적용할 수 있습니다.

## 연습문제

1. `TokenEmbedding('wiki.en')`를 사용하여 빠른 텍스트 결과를 테스트합니다.
1. 어휘가 매우 클 때 어떻게 비슷한 단어를 찾거나 단어 비유를 더 빨리 완성 할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1336)
:end_tab:
