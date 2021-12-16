# 기계 번역 및 데이터세트
:label:`sec_machine_translation`

우리는 RNN을 사용하여 자연어 처리의 핵심인 언어 모델을 설계했습니다.또 다른 주력 벤치마크는 입력 시퀀스를 출력 시퀀스로 변환하는*시퀀스 변환* 모델의 중심 문제 영역인*기계 변환*입니다.다양한 최신 AI 애플리케이션에서 중요한 역할을 하는 시퀀스 변환 모델은 이 장의 나머지 부분과 :numref:`chap_attention`의 초점을 맞출 것입니다.이를 위해 이 섹션에서는 기계 번역 문제와 나중에 사용할 데이터세트에 대해 소개합니다. 

*기계 번역*은
시퀀스를 한 언어에서 다른 언어로 자동 번역합니다.사실, 이 분야는 디지털 컴퓨터가 발명 된 직후 1940 년대로 거슬러 올라갈 수 있습니다. 특히 제 2 차 세계 대전에서 언어 코드를 해독하기 위해 컴퓨터를 사용하는 것을 고려하면 더욱 그렇습니다.수십 년 동안 신경망을 사용한 종단 간 학습이 등장하기 전에이 분야 :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`에서 통계적 접근법이 지배적이었습니다.후자는 종종 불린다.
*신경 기계 번역*
자신을 구별하기 위해
*통계 기계 번역*
번역 모델 및 언어 모델과 같은 구성 요소의 통계 분석이 포함됩니다. 

엔드 투 엔드 학습을 강조하는 이 책은 신경망 기계 번역 방법에 초점을 맞출 것입니다.코퍼스가 하나의 단일 언어로 된 :numref:`sec_language_model`의 언어 모델 문제와는 달리, 기계 번역 데이터 세트는 각각 소스 언어와 대상 언어로 된 텍스트 시퀀스 쌍으로 구성됩니다.따라서 언어 모델링에 전처리 루틴을 재사용하는 대신 기계 번역 데이터 세트를 전처리하는 다른 방법이 필요합니다.다음에서는 훈련을 위해 전처리된 데이터를 미니배치로 로드하는 방법을 보여줍니다.

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
import os
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

## [**데이터세트 다운로드 및 전처리**]

먼저 [bilingual sentence pairs from the Tatoeba Project](http://www.manythings.org/anki/)로 구성된 영어-프랑스어 데이터 세트를 다운로드합니다.데이터셋의 각 줄은 탭으로 구분된 영어 텍스트 시퀀스와 번역된 프랑스어 텍스트 시퀀스의 쌍입니다.각 텍스트 시퀀스는 한 문장이거나 여러 문장으로 구성된 단락일 수 있습니다.영어가 프랑스어로 번역되는 이 기계 번역 문제에서 영어는*소스 언어*이고 프랑스어는*대상 언어*입니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """Load the English-French dataset."""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])
```

데이터세트를 다운로드한 후 원시 텍스트 데이터에 대해 [**몇 가지 전처리 단계를 진행합니다**].예를 들어, 줄 바꿈하지 않는 공백을 공백으로 바꾸고 대문자를 소문자로 변환하고 단어와 구두점 사이에 공백을 삽입합니다.

```{.python .input}
#@tab all
#@save
def preprocess_nmt(text):
    """Preprocess the English-French dataset."""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])
```

## [**토큰화**]

:numref:`sec_language_model`의 문자 수준 토큰화와 달리 기계 번역의 경우 단어 수준 토큰화를 선호합니다 (최첨단 모델은 고급 토큰화 기술을 사용할 수 있음).다음 `tokenize_nmt` 함수는 첫 번째 `num_examples` 텍스트 시퀀스 쌍을 토큰화합니다. 여기서 각 토큰은 단어 또는 문장 부호입니다.이 함수는 `source`와 `target`의 두 가지 토큰 목록 목록을 반환합니다.특히 `source[i]`은 소스 언어 (여기서는 영어) 로 된 $i^\mathrm{th}$ 텍스트 시퀀스의 토큰 목록이고 `target[i]`은 대상 언어 (여기서는 프랑스어) 로 표시됩니다.

```{.python .input}
#@tab all
#@save
def tokenize_nmt(text, num_examples=None):
    """Tokenize the English-French dataset."""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]
```

[**텍스트 시퀀스당 토큰 수의 히스토그램을 플로팅합니다.**] 이 간단한 영어-프랑스어 데이터 세트에서 대부분의 텍스트 시퀀스는 토큰이 20개 미만입니다.

```{.python .input}
#@tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);
```

## [**어휘**]

기계 번역 데이터 세트는 언어 쌍으로 구성되어 있으므로 소스 언어와 대상 언어 모두에 대해 두 개의 어휘를 별도로 작성할 수 있습니다.단어 수준 토큰화를 사용하면 어휘 크기가 문자 수준 토큰화를 사용하는 것보다 훨씬 커집니다.이를 완화하기 위해 2배 미만으로 나타나는 드문 토큰을 동일한 알 수 없는 (” <unk>“) 토큰으로 취급합니다.그 외에도 <pad>미니 배치에서 동일한 길이로 (” “) 시퀀스를 패딩하고 <bos><eos>시퀀스의 시작 (” “) 또는 끝 (” “) 을 표시하는 것과 같은 추가 특수 토큰을 지정합니다.이러한 특수 토큰은 일반적으로 자연어 처리 작업에 사용됩니다.

```{.python .input}
#@tab all
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

## 데이터세트 읽기
:label:`subsec_mt_data_loading`

언어 모델링 [**각 시퀀스 예제**] 에서 한 문장의 세그먼트 또는 여러 문장에 걸친 범위 (**고정 길이**) 를 기억하십시오. 이것은 :numref:`sec_language_model`의 `num_steps` (시간 단계 또는 토큰 수) 인수로 지정되었습니다.기계 번역에서 각 예제는 소스 및 대상 텍스트 시퀀스의 쌍이며, 각 텍스트 시퀀스의 길이는 다를 수 있습니다. 

계산 효율성을 위해*잘림* 및*패딩*을 사용하여 텍스트 시퀀스의 미니 배치를 한 번에 처리 할 수 있습니다.동일한 미니배치의 모든 시퀀스의 길이가 `num_steps`가 같아야 한다고 가정합니다.텍스트 시퀀스의 토큰이 `num_steps` 미만인 경우 <pad>길이가 `num_steps`에 도달할 때까지 끝에 특수 "“토큰을 계속 추가합니다.그렇지 않으면 첫 번째 `num_steps` 토큰만 사용하고 나머지 토큰은 폐기하여 텍스트 시퀀스를 자릅니다.이런 식으로 모든 텍스트 시퀀스는 동일한 모양의 미니 배치로 로드되는 동일한 길이를 갖습니다. 

다음 `truncate_pad` 함수는 앞에서 설명한 대로 (**텍스트 시퀀스를 자르거나 채우기**) 합니다.

```{.python .input}
#@tab all
#@save
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

이제 [**텍스트 시퀀스를 훈련을 위해 미니 배치로 변환하는 함수를 정의합니다.**] 시퀀스의 끝을 <eos>나타 내기 위해 모든 시퀀스의 끝에 특수 “" 토큰을 추가합니다.모델이 토큰 다음에 시퀀스 토큰을 생성하여 예측하는 경우 “<eos>” 토큰을 생성하면 출력 시퀀스가 완료되었음을 나타낼 수 있습니다.또한 패딩 토큰을 제외한 각 텍스트 시퀀스의 길이도 기록합니다.이 정보는 나중에 다룰 일부 모델에 필요합니다.

```{.python .input}
#@tab all
#@save
def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len
```

## [**모든 것을 하나로 모으기**]

마지막으로 소스 언어와 대상 언어 모두에 대한 어휘와 함께 데이터 반복자를 반환하는 `load_data_nmt` 함수를 정의합니다.

```{.python .input}
#@tab all
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
```

[**영어-프랑스어 데이터 세트에서 첫 번째 미니 배치를 읽습니다.**]

```{.python .input}
#@tab all
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', d2l.astype(X, d2l.int32))
    print('valid lengths for X:', X_valid_len)
    print('Y:', d2l.astype(Y, d2l.int32))
    print('valid lengths for Y:', Y_valid_len)
    break
```

## 요약

* 기계 번역이란 시퀀스를 한 언어에서 다른 언어로 자동 번역하는 것을 말합니다.
* 단어 수준 토큰화를 사용하면 어휘 크기가 문자 수준 토큰화를 사용하는 것보다 훨씬 커집니다.이를 완화하기 위해 자주 사용하지 않는 토큰을 동일한 알 수 없는 토큰으로 취급할 수 있습니다.
* 텍스트 시퀀스를 자르고 패딩하여 모든 시퀀스가 미니 배치에로드 할 길이가 같도록 할 수 있습니다.

## 연습문제

1. `load_data_nmt` 함수에서 `num_examples` 인수의 다른 값을 시도해 보십시오.이것이 소스 언어와 대상 언어의 어휘 크기에 어떤 영향을 미칩니 까?
1. 중국어 및 일본어와 같은 일부 언어의 텍스트에는 단어 경계 표시기 (예: 공백) 가 없습니다.이러한 경우에 워드 레벨 토큰화가 여전히 좋은 아이디어입니까?왜, 왜 안되니?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1060)
:end_tab:
