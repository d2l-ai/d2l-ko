#  Preliminaries

#  기초

:label:`chap_preliminaries`

To get started with deep learning, we will need to develop a few basic skills. All machine learning is concerned with extracting information from data. So we will begin by learning the practical skills for storing, manipulating, and preprocessing data.

딥러닝을 시작하려면 몇 가지 기초 기술을 알아야 합니다. 모든 머신러닝은 데이터에서 정보를 추출하는 것과 관련이 있습니다. 데이터 저장, 조작, 전처리의 실제 기술을 배우는 것으로 시작하겠습니다.



Moreover, machine learning typically requires working with large datasets, which we can think of as tables, where the rows correspond to examples and the columns correspond to attributes.
Linear algebra gives us a powerful set of techniques for working with tabular data.
We will not go too far into the weeds but rather focus on the basic of matrix operations and their implementation.

머신러닝은 일반적으로 대규모 데이터 세트를 다루는데, 이와 같은 데이터는 행이 예제, 열이 속성에 해당하는 테이블로 생각할 수 있습니다. 테이블 형식의 데이터를 다루는 강력한 기술을 선형대수학에서 배울 수 있습니다. 이 책에서는 지엽적인 세부사항보다 행렬 연산의 기본과 구현에 중점을 두겠습니다.



Additionally, deep learning is all about optimization. We have a model with some parameters and we want to find those that fit our data *the best*. Determining which way to move each parameter at each step of an algorithm requires a little bit of calculus, which will be briefly introduced. Fortunately, the `autograd` package automatically computes differentiation for us, and we will cover it next.

또한, 딥러닝은 최적화가 전부라고 해도 과언이 아닙니다. 주어진 모델에 대해서 우리가 가진 데이터에 *가장* 적합한 파라미터를 찾고자 합니다. 각각의 알고리즘 단계에서 매개 변수를 계산하기 위해 약간의 미적분이 필요합니다. `autograd` 패키지가 자동으로 미분값을 계산하며, 이 내용은 다음에 다루게 됩니다.



Next, machine learning is concerned with making predictions: what is the likely value of some unknown attribute, given the information that we observe? To reason rigorously under uncertainty we will need to invoke the language of probability.

다음으로 머신러닝은 예측을 하는 것과 관련이 있습니다. 우리가 관찰한 정보를 감안했을 때, 알려지지 않은 어떤 속성의 가능한 값은 얼마일까요? 불확실한 상황에서 엄격하게 추론하기 위해, 우리는 확률의 언어를 사용합니다.



In the end, the official documentation provides plenty of descriptions and examples that are beyond this book. To conclude the chapter, we will show you how to look up documentation for the needed information.

이 책의 범위를 넘어서는 풍부한 설명과 예제들은 공식 문서에서 찾을 수 있습니다. 필요한 정보를 문서에서 찾는 방법을 보여드린다는 말씀과 함께 이번 챕터를 마칩니다.



This book has kept the mathematical content to the minimum necessary to get a proper understanding of deep learning. However, it does not mean that this book is mathematics free. Thus, this chapter provides a rapid introduction to basic and frequently-used mathematics to allow anyone to understand at least *most* of the mathematical content of the book. If you wish to understand *all* of the mathematical content, further reviewing the [online appendix on mathematics](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html) should be sufficient.

이 책은 딥러닝을 이해하기 위한 최소한의 수학적인 내용만을 담고 있습니다. 그렇다고 이 책에 수학이 아예 나오지 않는다는 뜻은 아닙니다. 이번 챕터에서는  누구나 이 책의 수학 내용을 충분히 이해할 수 있도록 자주 나오는 기초 수학을 빠르게 소개하겠습니다. 수학 내용의 *전부*를 이해하고 싶다면, [수학에 대한 온라인 부록](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html) 을 참고하면 충분할 것입니다.



```toc
:maxdepth: 2

ndarray
pandas
linear-algebra
calculus
autograd
probability
lookup-api
```

