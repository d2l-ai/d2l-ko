# 텍스트 전처리
:label:`sec_text_preprocessing`

시퀀스 데이터에 대한 통계 도구 및 예측 문제를 검토하고 평가했습니다.이러한 데이터는 다양한 형태를 취할 수 있습니다.특히 책의 여러 장에서 초점을 맞출 것이므로 텍스트는 시퀀스 데이터의 가장 인기있는 예 중 하나입니다.예를 들어 기사는 단순히 일련의 단어 또는 일련의 문자로 볼 수 있습니다.시퀀스 데이터에 대한 향후 실험을 용이하게 하기 위해 이 섹션에서는 텍스트의 일반적인 전처리 단계를 설명합니다.일반적으로 다음 단계는 다음과 같습니다. 

1. 텍스트를 문자열로 메모리에 로드합니다.
1. 문자열을 토큰으로 분할합니다 (예: 단어 및 문자).
1. 어휘 표를 작성하여 분할된 토큰을 숫자 인덱스에 매핑합니다.
1. 텍스트를 숫자 인덱스의 시퀀스로 변환하여 모델에서 쉽게 조작할 수 있습니다.

```{.python .input}
import collections
from d2l import mxnet as d2l
import re
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import re
```

```{.python .input}
#@tab tensorflow
import collections
from d2l import tensorflow as d2l
import re
```

## 데이터세트 읽기

시작하기 위해 H.G. 웰스의 [*The Time Machine*](http://www.gutenberg.org/ebooks/35)에서 텍스트를 로드합니다.이것은 30000단어가 조금 넘는 상당히 작은 말뭉치이지만, 우리가 설명하고 싶은 목적으로는 괜찮습니다.보다 사실적인 문서 컬렉션에는 수십억 개의 단어가 포함되어 있습니다.다음 함수 (**데이터셋을 텍스트 줄 목록으로 읽습니다**). 여기서 각 줄은 문자열입니다.단순화를 위해 여기서는 구두점과 대문자를 무시합니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])
```

## 토큰화

다음 `tokenize` 함수는 목록 (`lines`) 을 입력으로 사용합니다. 여기서 각 요소는 텍스트 시퀀스 (예: 텍스트 줄) 입니다.[**각 텍스트 시퀀스는 토큰 목록으로 분할됩니다**].*토큰*은 텍스트의 기본 단위입니다.결국 토큰 목록 목록이 반환되며 각 토큰은 문자열입니다.

```{.python .input}
#@tab all
def tokenize(lines, token='word'):  #@save
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

## 어휘

토큰의 문자열 유형은 숫자 입력을 받는 모델에서 사용하기가 불편합니다.이제 문자열 토큰을 0**부터 시작하는 숫자 인덱스에 매핑하기 위해*어휘*라고도 하는 사전을 빌드해 보겠습니다.이를 위해 먼저 훈련 세트의 모든 문서에서 고유 토큰, 즉*corpus*를 계산 한 다음 빈도에 따라 각 고유 토큰에 숫자 인덱스를 할당합니다.거의 나타나지 않는 토큰은 복잡성을 줄이기 위해 종종 제거됩니다.코퍼스에 존재하지 않거나 제거된 토큰은 알 수 없는 특수 토큰 “<unk>" 에 매핑됩니다.선택적으로 <pad>패딩의 경우 “”, 시퀀스의 <bos>시작을 나타내는 “”, <eos>시퀀스의 끝을 나타내는 “" 와 같은 예약된 토큰 목록을 추가합니다.

```{.python .input}
#@tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs

def count_corpus(tokens):  #@save
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

타임머신 데이터 세트를 코퍼스로 사용하여 [**어휘를 구성**] 합니다.그런 다음 처음 몇 개의 빈번한 토큰을 인덱스와 함께 인쇄합니다.

```{.python .input}
#@tab all
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

이제 (**각 텍스트 줄을 숫자 인덱스 목록으로 변환**) 할 수 있습니다.

```{.python .input}
#@tab all
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

## 모든 것을 하나로 모으다

위의 함수를 사용하여 토큰 인덱스 목록인 `corpus`와 타임머신 코퍼스의 어휘인 `vocab`를 반환하는 [**모든 것을 `load_corpus_time_machine` 함수로 패키징**] 합니다.여기서 수정 한 내용은 다음과 같습니다. (i) 이후 섹션에서 교육을 단순화하기 위해 텍스트를 단어가 아닌 문자로 토큰 화합니다. (ii) `corpus`는 타임머신 데이터 세트의 각 텍스트 행이 반드시 문장이나 단락이 아니기 때문에 토큰 목록 목록이 아닌 단일 목록입니다.

```{.python .input}
#@tab all
def load_corpus_time_machine(max_tokens=-1):  #@save
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## 요약

* 텍스트는 시퀀스 데이터의 중요한 형식입니다.
* 텍스트를 전처리하기 위해 일반적으로 텍스트를 토큰으로 분할하고, 토큰 문자열을 숫자 인덱스로 매핑하는 어휘를 만들고, 모델이 조작할 수 있도록 텍스트 데이터를 토큰 인덱스로 변환합니다.

## 연습문제

1. 토큰화는 주요 전처리 단계입니다.언어마다 다릅니다.텍스트를 토큰화하는 데 일반적으로 사용되는 다른 세 가지 방법을 찾아보십시오.
1. 이 섹션의 실험에서 텍스트를 단어로 토큰화하고 `Vocab` 인스턴스의 `min_freq` 인수를 변경합니다.이것이 어휘 크기에 어떤 영향을 미칩니 까?

[Discussions](https://discuss.d2l.ai/t/115)
