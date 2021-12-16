# 자연어 추론 및 데이터세트
:label:`sec_natural-language-inference-and-dataset`

:numref:`sec_sentiment`에서는 감정 분석의 문제에 대해 논의했습니다.이 작업은 단일 텍스트 시퀀스를 일련의 센티멘트 극성과 같은 사전 정의된 범주로 분류하는 것을 목표로 합니다.그러나 한 문장을 다른 문장으로 추론할 수 있는지 결정하거나 의미 상 동등한 문장을 식별하여 중복을 제거해야 할 경우 한 텍스트 시퀀스를 분류하는 방법을 아는 것만으로는 충분하지 않습니다.대신 텍스트 시퀀스 쌍을 추론할 수 있어야 합니다. 

## 자연어 추론

*자연어 추론*은*가설* 여부를 연구합니다
*premise*에서 추론할 수 있습니다. 여기서 둘 다 텍스트 시퀀스입니다.즉, 자연어 추론은 한 쌍의 텍스트 시퀀스 간의 논리적 관계를 결정합니다.이러한 관계는 일반적으로 세 가지 유형으로 나뉩니다. 

* *수반*: 가설은 전제에서 추론할 수 있습니다.
* *모순*: 가설의 부정은 전제에서 추론할 수 있습니다.
* *중립*: 다른 모든 경우

자연어 추론은 텍스트 수반 인식 작업이라고도합니다.예를 들어, 다음 쌍은*entailment*로 표시됩니다. 가설에서 “애정 표시”는 전제에서 “서로 껴안기”에서 추론 할 수 있기 때문입니다. 

> 전제: 두 명의 여성이 서로 껴안고 있습니다. 

> 가설: 두 명의 여성이 애정을 보이고 있습니다. 

다음은*모순*의 예입니다. “코딩 예제 실행”은 “잠자기”가 아닌 “잠자기 상태”를 나타냅니다. 

> 전제: 한 남자가 다이브 투 딥 러닝에서 코딩 예제를 실행하고 있습니다. 

> 가설: 남자가 자고 있습니다. 

세 번째 예는*중립성* 관계를 보여줍니다. 왜냐하면 “유명함”이나 “유명하지 않음”은 “우리를 위해 공연하고 있습니다”라는 사실에서 추론 할 수 없기 때문입니다.  

> 전제: 음악가들이 우리를 위해 공연하고 있습니다. 

> 가설: 음악가들이 유명합니다. 

자연어 추론은 자연어를 이해하는 데 핵심적인 주제였습니다.정보 검색에서 공개 도메인 질문 답변에 이르기까지 광범위한 응용 프로그램을 즐길 수 있습니다.이 문제를 연구하기 위해 먼저 널리 사용되는 자연어 추론 벤치마크 데이터 세트를 조사하는 것으로 시작하겠습니다. 

## 스탠포드 자연어 추론 (SNLI) 데이터세트

스탠포드 자연어 추론 (SNLI) 코퍼스는 500,000개 이상의 레이블이 지정된 영어 문장 쌍 :cite:`Bowman.Angeli.Potts.ea.2015`의 모음입니다.추출된 SNLI 데이터 세트를 `../data/snli_1.0` 경로에 다운로드하여 저장합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### 데이터세트 읽기

원래 SNLI 데이터 세트에는 실험에서 실제로 필요한 것보다 훨씬 풍부한 정보가 포함되어 있습니다.따라서 데이터 세트의 일부만 추출한 다음 전제, 가설 및 레이블 목록을 반환하는 함수 `read_snli`를 정의합니다.

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

이제 처음 3 쌍의 전제와 가설과 레이블 (“0", “1" 및 “2"는 각각 “수반”, “모순”및 “중립”에 해당) 을 인쇄 해 보겠습니다.

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

훈련 세트에는 약 550000쌍이 있고 테스트 세트에는 약 10000쌍이 있습니다.다음은 훈련 세트와 테스트 세트 모두에서 “수반”, “모순”및 “중립”이라는 세 가지 레이블이 균형을 이루고 있음을 보여줍니다.

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### 데이터세트 로드를 위한 클래스 정의

아래에서는 Gluon의 `Dataset` 클래스에서 상속하여 SNLI 데이터 세트를 로드하는 클래스를 정의합니다.클래스 생성자의 인수 `num_steps`는 시퀀스의 각 미니 배치가 동일한 모양을 갖도록 텍스트 시퀀스의 길이를 지정합니다.즉, 더 긴 시퀀스의 첫 번째 `num_steps` 이후의 토큰은 트리밍되고 특수 토큰 “<pad>" 은 길이가 `num_steps`가 될 때까지 더 짧은 시퀀스에 추가됩니다.`__getitem__` 함수를 구현하면 색인 `idx`을 사용하여 전제, 가설 및 레이블에 임의로 액세스 할 수 있습니다.

```{.python .input}
#@save
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### 모든 것을 하나로 모으다

이제 `read_snli` 함수와 `SNLIDataset` 클래스를 호출하여 SNLI 데이터 세트를 다운로드하고 훈련 세트의 어휘와 함께 훈련 세트와 테스트 세트 모두에 대한 `DataLoader` 인스턴스를 반환할 수 있습니다.훈련 세트에서 구성된 어휘를 테스트 세트의 어휘로 사용해야 한다는 점은 주목할 만하다.결과적으로 테스트 세트의 새 토큰은 학습 세트에서 학습된 모델에 알려지지 않습니다.

```{.python .input}
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

여기서는 배치 크기를 128로 설정하고 시퀀스 길이를 50으로 설정하고 `load_data_snli` 함수를 호출하여 데이터 반복기와 어휘를 얻습니다.그런 다음 어휘 크기를 인쇄합니다.

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

이제 첫 번째 미니 배치의 모양을 인쇄합니다.감정 분석과는 달리 전제와 가설의 쌍을 나타내는 두 개의 입력 `X[0]`와 `X[1]`가 있습니다.

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## 요약

* 자연어 추론은 둘 다 텍스트 시퀀스인 전제에서 가설을 추론할 수 있는지 여부를 연구합니다.
* 자연어 추론에서 전제와 가설 간의 관계에는 수반, 모순 및 중립이 포함됩니다.
* 스탠포드 자연어 추론 (SNLI) 코퍼스는 자연어 추론의 인기 벤치마크 데이터셋입니다.

## 연습문제

1. 기계 번역은 출력 번역과 실측 번역 간의 표면적 $n$g 일치를 기반으로 오랫동안 평가되어 왔습니다.자연어 추론을 사용하여 기계 번역 결과를 평가하는 방법을 설계할 수 있습니까?
1. 어휘 크기를 줄이기 위해 어떻게 하이퍼파라미터를 변경할 수 있을까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1388)
:end_tab:
