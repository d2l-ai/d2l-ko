# 문서(documentation)

이 책에서 MXNet 함수와 클래스를 모두 설명하기는 불가능하니, API 문서나 추가적인 튜토리얼과 예제를 참고하면 이 책에서 다루지 못한 많은 내용을 찾아볼 수 있습니다.

## 모듈의 모든 함수와 클래스 찾아보기

모듈에서 어떤 함수와 클래스가 제공되는지 알기 위해서 `dir` 함수를 이용합니다. 예를 들어, `nd.random` 모듈의 모든 맴버와 속성을 다음과 같이 조회할 수 있습니다.

```{.python .input  n=1}
from mxnet import nd
print(dir(nd.random))
```

일반적으로 이름이 `__` 로 시작하는 함수(Python에서 특별한 객체를 나타냄)나 `_` 로 시작하는 함수(보통은 내부 함수들)는 무시해도 됩니다. 나머지 맴버들에 대해서는 이름을 통해 추측해보면, 다양한 난수를 생성하는 메소드들로 추측할 수 있습니다. 즉, 균일한 분포에서 난수를 생성하는 `uniform`, 표준 분산에서 난수를 생성하는 `normal` 그리고 Poisson 샘플링인 `poisson` 등의 기능을 제공함을 알 수 있습니다.

## 특정 함수들과 클래스들의 사용법 찾아보기

`help` 함수를 이용하면 특정 함수나 클래스의 사용법 확인할 수 있습니다. NDArray의 `ones_like` 함수를 예로 살펴봅니다.

```{.python .input}
help(nd.ones_like)
```

문서를 보면, `ones_like` 함수는 NDArray 객체와 모두 1로 설정된 같은 모양(shape)의 새로운 객체를 만들어 줍니다. 확인해보겠습니다.

```{.python .input}
x = nd.array([[0, 0, 0], [2, 2, 2]])
y = x.ones_like()
y
```

Jupyter 노트북에서는 `?` 를 이용해서 다른 윈도우에 문서를 표시할 수 있습니다. 예를 들어 `nd.random.uniform?` 를 수행하면 `help(nd.random.uniform)` 과 거의 동일한 내용이 다른 윈도우에 나옵니다. 그리고, `nd.random.uniform??` 와 같이 `?` 를 두개 사용하면, 함수를 구현하는 코드도 함께 출력됩니다.

##  API 문서

API에 대한 더 자세한 내용은 MXNet 웹사이트 [http://mxnet.apache.org/](http://mxnet.apache.org/) 를 확인하세요. Python 및 이외의 다른 프로그램 언어에 대한 내용들을 웹 사이트에서 찾을 수 있습니다.

## 문제

1. API 문서에서 `ones_like` 와 `autograd` 를 찾아보세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2322)

![](../img/qr_lookup-api.svg)
