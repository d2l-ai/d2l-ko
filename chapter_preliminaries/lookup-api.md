# 설명서를 참조하십시오

:begin_tab:`mxnet`
이 책의 길이에 대한 제약으로 인해 모든 단일 MXNet 함수와 클래스를 소개 할 수는 없습니다 (아마도 원하지 않을 것입니다).API 설명서와 추가 자습서 및 예제는 책 이외의 많은 문서를 제공합니다.이 섹션에서는 MXNet API를 탐색하기 위한 몇 가지 지침을 제공합니다.
:end_tab:

:begin_tab:`pytorch`
이 책의 길이에 대한 제약으로 인해 모든 PyTorch 함수와 클래스를 소개 할 수는 없습니다 (그리고 아마도 원하지 않을 것입니다).API 설명서와 추가 자습서 및 예제는 책 이외의 많은 문서를 제공합니다.이 섹션에서는 PyTorch API를 탐색하기 위한 몇 가지 지침을 제공합니다.
:end_tab:

:begin_tab:`tensorflow`
이 책의 길이에 대한 제약으로 인해 모든 단일 TensorFlow 함수와 클래스를 소개 할 수는 없습니다 (아마도 원하지 않을 것입니다).API 설명서와 추가 자습서 및 예제는 책 이외의 많은 문서를 제공합니다.이 섹션에서는 TensorFlow API를 탐색하기 위한 몇 가지 지침을 제공합니다.
:end_tab:

## 모듈에서 모든 함수와 클래스 찾기

모듈에서 어떤 함수와 클래스를 호출할 수 있는지 알기 위해 `dir` 함수를 호출합니다.예를 들어, (**난수를 생성하기 위해 모듈의 모든 프로퍼티를 쿼리**) 할 수 있습니다:

```{.python .input  n=1}
from mxnet import np
print(dir(np.random))
```

```{.python .input  n=1}
#@tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input  n=1}
#@tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

일반적으로 `__` (파이썬의 특수 객체) 으로 시작하고 끝나는 함수나 단일 `_`로 시작하는 함수 (일반적으로 내부 함수) 는 무시할 수 있습니다.나머지 함수 또는 속성 이름을 기반으로 이 모듈이 균등 분포 (`uniform`), 정규 분포 (`normal`) 및 다항 분포 (`multinomial`) 의 샘플링을 포함하여 난수를 생성하는 다양한 방법을 제공한다는 추측을 위험에 빠뜨릴 수 있습니다. 

## 특정 함수 및 클래스의 사용법 찾기

주어진 함수나 클래스를 사용하는 방법에 대한 보다 구체적인 지침을 보려면 `help` 함수를 호출할 수 있습니다.예를 들어 [**텐서의 `ones` 함수에 대한 사용 지침을 살펴보십시오**].

```{.python .input}
help(np.ones)
```

```{.python .input}
#@tab pytorch
help(torch.ones)
```

```{.python .input}
#@tab tensorflow
help(tf.ones)
```

문서에서 `ones` 함수가 지정된 모양을 가진 새 텐서를 만들고 모든 요소를 값 1로 설정한다는 것을 알 수 있습니다.가능하면 (**빠른 테스트 실행**) 하여 해석을 확인해야 합니다.

```{.python .input}
np.ones(4)
```

```{.python .input}
#@tab pytorch
torch.ones(4)
```

```{.python .input}
#@tab tensorflow
tf.ones(4)
```

Jupyter 노트북에서는 `?`를 눌러 다른 창에 문서를 표시합니다.예를 들어, `list?`는 `help(list)`와 거의 동일한 컨텐츠를 생성하여 새 브라우저 창에 표시합니다.또한 `list?? 와 같은 두 개의 물음표를 사용하면`, 함수를 구현하는 파이썬 코드도 표시됩니다. 

## 요약

* 공식 문서에는 이 책의 범위를 벗어나는 많은 설명과 예가 나와 있습니다.
* `dir` 및 `help` 함수 또는 `? 를 호출하여 API 사용에 대한 설명서를 찾아볼 수 있습니다.` and `?`주피터 노트북에서.

## 연습문제

1. 딥러닝 프레임워크의 함수 또는 클래스에 대한 설명서를 찾아보십시오.프레임 워크의 공식 웹 사이트에서도 문서를 찾을 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:
