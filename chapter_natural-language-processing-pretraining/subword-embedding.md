# 서브워드 임베딩
:label:`sec_fasttext`

영어에서 “help”, “help”, “help”와 같은 단어는 동일한 단어 “help”의 변형된 형태입니다.“개”와 “개”의 관계는 “고양이”와 “고양이”의 관계와 동일하며, “소년”과 “남자 친구”의 관계는 “소녀”와 “여자 친구”의 관계와 동일합니다.프랑스어 및 스페인어와 같은 다른 언어에서는 많은 동사가 40 개가 넘는 변곡형을 가지고 있지만 핀란드어에서는 명사가 최대 15 개의 경우를 가질 수 있습니다.언어학에서 형태학은 단어 형성과 단어 관계를 연구합니다.그러나 단어의 내부 구조는 word2vec이나 GLOVE에서 탐구되지 않았습니다. 

## 패스트텍스트 모델

word2vec에서 단어가 어떻게 표현되는지 상기하십시오.스킵 그램 모델과 연속 단어 모음 모델 모두에서 동일한 단어의 서로 다른 변곡된 형식은 공유 매개 변수 없이 서로 다른 벡터로 직접 표현됩니다.형태학적 정보를 사용하기 위해*FastText* 모델은*하위 단어 임베딩* 접근 방식을 제안했습니다. 여기서 하위 단어는 문자 $n$g :cite:`Bojanowski.Grave.Joulin.ea.2017`입니다.단어 수준 벡터 표현을 학습하는 대신 FastText는 하위 단어 수준 건너뛰기 그램으로 간주할 수 있습니다. 여기서 각*중심 단어*는 하위 단어 벡터의 합으로 표시됩니다. 

“where”라는 단어를 사용하여 FastText의 각 중심 단어에 대한 하위 단어를 얻는 방법을 보여 드리겠습니다.먼저 <” and “> 단어의 시작과 끝에 특수 문자 “" 를 추가하여 접두사와 접미사를 다른 하위 단어와 구분합니다.그런 다음 단어에서 문자 $n$그램을 추출합니다.예를 들어, $n=3$인 경우 길이가 3인 모든 하위 단어: “<wh”, “whe”, “her”, “ere”, “re>" 와 특수 하위 단어 <where>"“를 얻습니다. 

FastText에서 모든 단어 $w$에 대해 $\mathcal{G}_w$로 길이가 3에서 6 사이인 모든 하위 단어와 특수 하위 단어의 합집합을 나타냅니다.어휘는 모든 단어의 하위 단어를 합한 것입니다.$\mathbf{z}_g$를 사전에 있는 하위 단어 $g$의 벡터가 되도록 하면, 스킵그램 모델에서 중심 단어인 단어 $w$에 대한 벡터 $\mathbf{v}_w$은 하위 단어 벡터의 합입니다. 

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

나머지 FastText는 스킵 그램 모델과 동일합니다.스킵 그램 모델과 비교할 때 FastText의 어휘는 더 커서 모델 매개 변수가 더 많습니다.또한 단어의 표현을 계산하려면 모든 하위 단어 벡터를 합산해야 하므로 계산 복잡성이 높아집니다.그러나 구조가 유사한 단어 사이에서 하위 단어의 공유 매개 변수 덕분에 희귀 단어와 어휘 외 단어조차도 FastText에서 더 나은 벡터 표현을 얻을 수 있습니다. 

## 바이트 페어 인코딩
:label:`subsec_Byte_Pair_Encoding`

FastText에서 추출된 모든 하위 단어는 $3$에서 $6$과 같이 지정된 길이여야 하므로 어휘 크기를 미리 정의할 수 없습니다.고정 크기 어휘에서 가변 길이 하위 단어를 허용하려면*바이트 쌍 인코딩* (BPE) 이라는 압축 알고리즘을 적용하여 하위 단어 :cite:`Sennrich.Haddow.Birch.2015`를 추출 할 수 있습니다. 

바이트 쌍 인코딩은 학습 데이터 세트의 통계 분석을 수행하여 임의의 길이의 연속 문자와 같은 단어 내의 공통 기호를 찾습니다.길이가 1인 심볼에서 시작하여 바이트 쌍 인코딩은 가장 빈번한 연속 심볼 쌍을 반복적으로 병합하여 더 긴 새로운 심볼을 생성합니다.효율성을 위해 단어 경계를 넘는 쌍은 고려되지 않습니다.결국 하위 단어와 같은 기호를 사용하여 단어를 분류 할 수 있습니다.바이트 쌍 인코딩 및 그 변형은 GPT-2 :cite:`Radford.Wu.Child.ea.2019` 및 RoberTA :cite:`Liu.Ott.Goyal.ea.2019`와 같은 널리 사용되는 자연어 처리 사전 교육 모델의 입력 표현에 사용되었습니다.다음에서는 바이트 쌍 인코딩이 작동하는 방식을 설명합니다. 

먼저 기호의 어휘를 모든 영어 소문자, 특수 단어 끝 기호 `'_'` 및 알 수없는 특수 기호 `'[UNK]'`로 초기화합니다.

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

단어의 경계를 넘는 기호 쌍을 고려하지 않기 때문에 데이터 세트에서 단어를 빈도 (발생 횟수) 에 매핑하는 사전 `raw_token_freqs` 만 있으면됩니다.특수 기호 `'_'`가 각 단어에 추가되어 출력 기호 시퀀스 (예: “a_ tall er_ man”) 에서 단어 시퀀스 (예: “키가 큰 남자”) 를 쉽게 복구 할 수 있습니다.단일 문자와 특수 기호로만 구성된 어휘에서 병합 프로세스를 시작하기 때문에 각 단어 내의 모든 연속 문자 쌍 사이에 공백이 삽입됩니다 (사전 `token_freqs`의 키).즉, 공백은 단어 내 기호 사이의 구분 기호입니다.

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

단어 내에서 가장 빈번한 연속 기호 쌍을 반환하는 다음 `get_max_freq_pair` 함수를 정의합니다. 여기서 단어는 입력 사전 `token_freqs`의 키에서 가져옵니다.

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```

연속 심볼의 빈도에 기반한 욕심 많은 접근 방식으로 바이트 쌍 인코딩은 다음 `merge_symbols` 함수를 사용하여 가장 빈번한 연속 심볼 쌍을 병합하여 새 심볼을 생성합니다.

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

이제 우리는 사전 `token_freqs`의 키에 대해 바이트 쌍 인코딩 알고리즘을 반복적으로 수행합니다.첫 번째 반복에서 가장 빈번한 연속 심볼 쌍은 `'t'` 및 `'a'`이므로 바이트 쌍 인코딩은 이들을 병합하여 새로운 심볼 `'ta'`을 생성합니다.두 번째 반복에서 바이트 쌍 인코딩은 `'ta'`과 `'l'`를 계속 병합하여 또 다른 새로운 심볼 `'tal'`을 생성합니다.

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

바이트 쌍 인코딩을 10번 반복한 후 목록 `symbols`에 다른 심볼에서 반복적으로 병합되는 10개의 심볼이 더 포함되어 있음을 알 수 있습니다.

```{.python .input}
#@tab all
print(symbols)
```

사전 `raw_token_freqs`의 키에 지정된 동일한 데이터 세트의 경우, 이제 데이터 세트의 각 단어가 바이트 쌍 인코딩 알고리즘의 결과로 하위 단어 “fast_”, “fast”, “er_”, “tall_” 및 “tall”으로 분할됩니다.예를 들어, “faster_”와 “taller_”라는 단어는 각각 “빠른 er_”과 “taller_”로 분류됩니다.

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

바이트 쌍 인코딩의 결과는 사용 중인 데이터세트에 따라 달라집니다.한 데이터셋에서 학습한 하위 단어를 사용하여 다른 데이터셋의 단어를 분할할 수도 있습니다.탐욕스러운 접근 방식으로 다음 `segment_BPE` 함수는 입력 인수 `symbols`에서 가능한 가장 긴 하위 단어로 단어를 분할하려고 시도합니다.

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

다음에서는 앞서 언급한 데이터세트에서 학습한 목록 `symbols`의 하위 단어를 사용하여 다른 데이터 세트를 나타내는 세그먼트 `tokens`에 사용합니다.

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## 요약

* FastText 모델은 하위 단어 삽입 방식을 제안합니다.word2vec의 스킵-그램 모델을 기반으로 하여 중심어를 하위 단어 벡터의 합으로 나타냅니다.
* 바이트 쌍 인코딩은 학습 데이터 세트의 통계 분석을 수행하여 단어 내의 공통 기호를 찾습니다.탐욕스러운 접근 방식으로 바이트 쌍 인코딩은 가장 빈번한 연속 기호 쌍을 반복적으로 병합합니다.
* 하위 단어 포함은 희귀 단어 및 사전 외 단어의 표현 품질을 향상시킬 수 있습니다.

## 연습문제

1. 예를 들어, 영어에는 약 $3\times 10^8$개의 가능한 $6$그램이 있습니다.하위 단어가 너무 많으면 어떤 문제가 발생합니까?이 문제를 어떻게 해결할 수 있을까요?힌트: refer to the end of Section 3.2 of the fastText paper :cite:`Bojanowski.Grave.Joulin.ea.2017`.
1. 연속 단어 모음 모델을 기반으로 하위 단어 임베딩 모델을 설계하는 방법은 무엇입니까?
1. 크기가 $m$인 어휘를 얻으려면 초기 기호 어휘 크기가 $n$일 때 몇 개의 병합 작업이 필요합니까?
1. 구문을 추출하기 위해 바이트 쌍 인코딩의 아이디어를 확장하는 방법은 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/386)
:end_tab:
