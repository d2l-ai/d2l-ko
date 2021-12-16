# 단어 임베딩 사전 학습을 위한 데이터셋
:label:`sec_word2vec_data`

이제 word2vec 모델의 기술적 세부 사항과 대략적인 훈련 방법을 알았으므로 구현을 살펴 보겠습니다.구체적으로 :numref:`sec_word2vec`에서는 스킵 그램 모델을, :numref:`sec_approx_train`에서는 네거티브 샘플링을 예로 들어 보겠습니다.이 섹션에서는 단어 임베딩 모델을 사전 학습하기 위한 데이터셋으로 시작합니다. 데이터의 원래 형식은 훈련 중에 반복할 수 있는 미니배치로 변환됩니다.

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
import os
import random
```

## 데이터세트 읽기

여기서 사용하는 데이터세트는 [펜 트리 뱅크 (PTB)](https://catalog.ldc.upenn.edu/LDC99T42) 입니다.이 코퍼스는 Wall Street Journal 기사에서 샘플링되며 교육, 검증 및 테스트 세트로 나뉩니다.원래 형식에서 텍스트 파일의 각 줄은 공백으로 구분된 단어 문장을 나타냅니다.여기서는 각 단어를 토큰으로 취급합니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract('ptb')
    # Read the training set.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'
```

훈련 세트를 읽은 후, 우리는 코퍼스에 대한 어휘를 만듭니다. 여기서 10 번 미만으로 나타나는 단어는 <unk>"“토큰으로 대체됩니다.원래 데이터셋에는 <unk>희귀 (알 수 없는) 단어를 나타내는 "“토큰도 포함되어 있습니다.

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## 서브샘플링

텍스트 데이터에는 일반적으로 “the”, “a” 및 “in”과 같은 빈도가 높은 단어가 있습니다. 매우 큰 상체에서 수십억 번 발생할 수도 있습니다.그러나 이러한 단어는 컨텍스트 창에서 여러 단어와 함께 발생하는 경우가 많으므로 유용한 신호가 거의 없습니다.예를 들어 컨텍스트 창에서 “칩”이라는 단어를 생각해보십시오. 직관적으로 저주파 단어 “intel”과의 동시 발생은 고주파 단어 “a”와의 동시 발생보다 훈련에 더 유용합니다.또한 방대한 양의 (고주파) 단어로 훈련하는 것은 느립니다.따라서 단어 임베딩 모델을 학습할 때 고주파 단어는*서브샘플링* :cite:`Mikolov.Sutskever.Chen.ea.2013`가 될 수 있습니다.특히, 데이터셋의 인덱싱된 단어 $w_i$는 확률과 함께 폐기됩니다. 

$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

여기서 $f(w_i)$는 데이터 세트의 총 단어 수에 대한 단어 수 $w_i$의 비율이고 상수 $t$은 하이퍼파라미터 (실험에서 $10^{-4}$) 입니다.상대 주파수 $f(w_i) > t$이 (고주파) 단어 $w_i$를 버릴 수 있고 단어의 상대 빈도가 높을수록 폐기 될 확률이 커진다는 것을 알 수 있습니다.

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """Subsample high-frequency words."""
    # Exclude unknown tokens '<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

다음 코드 스니펫은 서브샘플링 전후의 문장당 토큰 수에 대한 히스토그램을 플로팅합니다.예상대로 서브 샘플링은 고주파 단어를 삭제하여 문장을 크게 단축시켜 훈련 속도를 높입니다.

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

개별 토큰의 경우 고주파 단어 “the”의 샘플링 속도는 1/20보다 작습니다.

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

반대로 저주파 단어 “join”은 완전히 유지됩니다.

```{.python .input}
#@tab all
compare_counts('join')
```

서브 샘플링 후 토큰을 코퍼스에 대한 인덱스에 매핑합니다.

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## 중심 단어 및 문맥 단어 추출

다음 `get_centers_and_contexts` 함수는 `corpus`에서 모든 중심 단어와 문맥 단어를 추출합니다.컨텍스트 창 크기로 1에서 `max_window_size` 사이의 정수를 임의로 균일하게 샘플링합니다.가운데 단어로부터 거리가 샘플링된 컨텍스트 창 크기를 초과하지 않는 단어는 문맥 단어입니다.

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram."""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

다음으로 각각 7단어와 3단어로 구성된 두 문장이 포함된 인공 데이터 세트를 만듭니다.최대 컨텍스트 창 크기를 2로 설정하고 모든 가운데 단어와 문맥 단어를 인쇄합니다.

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

PTB 데이터세트에서 학습할 때 최대 컨텍스트 창 크기를 5로 설정했습니다.다음은 데이터셋의 모든 중심 단어와 문맥 단어를 추출합니다.

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## 음수 샘플링

근사 훈련에는 음수 샘플링을 사용합니다.미리 정의된 분포에 따라 노이즈 단어를 샘플링하기 위해 다음 `RandomGenerator` 클래스를 정의합니다. 여기서 (정규화되지 않았을 수 있는) 샘플링 분포는 인수 `sampling_weights`를 통해 전달됩니다.

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        # Exclude 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

예를 들어, 다음과 같이 표본 추출 확률이 $P(X=1)=2/9, P(X=2)=3/9$ 및 $P(X=3)=4/9$인 지수 1, 2, 3 중에서 10개의 랜덤 변수 $X$를 그릴 수 있습니다.

```{.python .input}
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

한 쌍의 중앙어와 문맥 단어에 대해 `K` (실험에서 5) 노이즈 단어를 무작위로 샘플링합니다.단어2vec 논문의 제안에 따르면, 노이즈 워드 $w$의 샘플링 확률 $P(w)$은 0.75 :cite:`Mikolov.Sutskever.Chen.ea.2013`의 거듭제곱으로 상승한 사전의 상대 주파수로 설정됩니다.

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling."""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## 미니배치로 훈련 예제 불러오기
:label:`subsec_word2vec-minibatch-loading`

모든 중심 단어를 문맥 단어 및 샘플링된 잡음 단어와 함께 추출한 후에는 훈련 중에 반복적으로 로드할 수 있는 예제의 미니 배치로 변환됩니다. 

미니배치에서 $i^\mathrm{th}$ 예제에는 가운데 단어와 해당 $n_i$ 컨텍스트 단어 및 $m_i$개의 노이즈 단어가 포함되어 있습니다.컨텍스트 창 크기가 다양하기 때문에 $n_i+m_i$는 $i$에 따라 다릅니다.따라서 각 예에서 컨텍스트 단어와 노이즈 단어를 `contexts_negatives` 변수에 연결하고 연결 길이가 $\max_i n_i+m_i$ (`max_len`) 에 도달 할 때까지 0을 채웁니다.손실 계산에서 패딩을 제외하기 위해 마스크 변수 `masks`를 정의합니다.`masks`의 요소와 `contexts_negatives`의 요소 간에는 일대일 대응이 있습니다. 여기서 `masks`의 0 (그렇지 않은 경우) 은 `contexts_negatives`의 패딩에 해당합니다. 

긍정적 예와 부정적인 예를 구별하기 위해 `labels` 변수를 통해 `contexts_negatives`의 컨텍스트 단어와 노이즈 단어를 분리합니다.`masks`와 마찬가지로 `labels`의 요소와 `contexts_negatives`의 요소 간에는 일대일 대응이 있습니다. 여기서 `labels`의 요소 (그렇지 않으면 0) 는 `contexts_negatives`의 컨텍스트 단어 (긍정적 예) 에 해당합니다. 

위의 아이디어는 다음 `batchify` 함수에서 구현됩니다.입력값 `data`는 길이가 배치 크기와 동일한 목록으로, 각 요소는 중심 단어 `center`, 컨텍스트 단어 `context` 및 잡음 단어 `negative`로 구성된 예입니다.이 함수는 마스크 변수 포함과 같이 훈련 중에 계산을 위해 로드할 수 있는 미니배치를 반환합니다.

```{.python .input}
#@tab all
#@save
def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

두 가지 예제의 미니 배치를 사용하여이 함수를 테스트해 보겠습니다.

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## 모든 것을 하나로 모으다

마지막으로 PTB 데이터 세트를 읽고 데이터 반복자와 어휘를 반환하는 `load_data_ptb` 함수를 정의합니다.

```{.python .input}
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab
```

데이터 반복기의 첫 번째 미니 배치를 인쇄 해 보겠습니다.

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## 요약

* 고주파 단어는 훈련에 그다지 유용하지 않을 수 있습니다.훈련 속도를 높이기 위해 하위 샘플링을 할 수 있습니다.
* 계산 효율성을 위해 예제를 미니 배치로 로드합니다.패딩과 패딩이 아닌 것을 구별하기 위해 다른 변수를 정의하고 긍정적 인 예와 부정적인 변수를 구별 할 수 있습니다.

## 연습문제

1. 서브샘플링을 사용하지 않을 경우 이 섹션의 코드 실행 시간은 어떻게 변경됩니까?
1. `RandomGenerator` 클래스는 `k` 랜덤 샘플링 결과를 캐시합니다.`k`를 다른 값으로 설정하고 이 값이 데이터 로드 속도에 어떤 영향을 미치는지 확인합니다.
1. 이 섹션의 코드에서 데이터 로드 속도에 영향을 줄 수 있는 다른 하이퍼파라미터는 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/383)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1330)
:end_tab:
