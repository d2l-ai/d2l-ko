# Using this Book

We aim to provide a comprehensive introduction to all aspects of deep learning from model construction to model training, as well as  applications in computer vision and natural language processing. We will not only explain the principles of the algorithms, but also demonstrate their implementation and operation in Apache MXNet. Each section of the book is a Jupyter notebook,  combining text, formulae, images, code, and running results. Not only can you read them directly, but you can run them to get an interactive learning experience. But since it is an introduction, we can only cover things so far. It is up to you, the reader, to explore further, to play with the toolboxes, compiler, and examples, tutorials and code snippets that are available in the research community. Enjoy the journey!

우리는 모델 만들기부터 모델 학습, 그리고 컴퓨터 비전 및 자연어 처리에 대한 응용까지 딥러닝의 모든 것들에 대해서 전반적인 소개를 하는 것을 목표로 합니다. 알고리즘의 원리만을 설명하는 것이 아니라, Apache MXNet을 이용해서 직접 구현하고 운영하는 것까지 보여줄 것입니다. 이 책의 각 절은 Jupyter 노트북, 텍스트, 공식, 이미지, 코드, 그리고 수행 결과로 구성되어 있습니다. 이를 통해서 이 책을 직접 읽는 것뿐만 아니라, 직접 수행하면서 상호적인 학습 경함을 할 수 있을 것 입니다. 하지만, 이 책을 여러분에 딥러닝에 대한 소개를 제공하는 것이기 때문에, 더 많은 것을 알고 싶다면, 연구 커뮤니티들에서 제공하는 툴박스, 컴파일러, 예제들, 튜터리얼, 그리고 소스 코드를 활용하는 것을 추천합니다. 그럼 여행을 시작봅시다.

## Target Audience

This book is for college students, engineers, and researchers who wish to learn deep learning, especially for those who are interested in applying deep learning in practice. Readers need not have a background in deep learning or machine learning. We will explain every concept from scratch. Although illustrations of deep learning techniques and applications involve mathematics and programming, you only need to know their basics, such as basic linear algebra, calculus, and probability, and basic Python programming. In the appendix we provide most of the mathematics covered in this book for your reference. Since it's an introduction, we prioritize intuition and ideas over mathematical rigor. There are many terrific books which can lead the interested reader further. For instance [Linear Analysis](https://www.amazon.com/Linear-Analysis-Introductory-Cambridge-Mathematical/dp/0521655773) by Bela Bollobas covers linear algebra and functional analysis in great depth. [All of Statistics](https://www.amazon.com/All-Statistics-Statistical-Inference-Springer/dp/0387402721) is a terrific guide to statistics. And if you have not used Python before, you may want to peruse the [Python tutorial](http://learnpython.org/). Of course, if you are only interested in the mathematical part, you can ignore the programming part, and vice versa.

이 책은 딥러닝을 배우고자 하는 대학교 학부생, 엔지니어, 연구원 그리고 특히 딥러닝을 실제 문제에 적용하고자 하는 사람들을 대상으로 하고 있습니다. 모든 개념을 기초부터 설명하고 있기 때문에, 딥러닝이나 머신러닝에 대한 배경지식이 없어도 됩니다. 딥러닝 기술이나, 응용을 설명할 때 수학이나 프로그래밍을 사용하지만, 기본적인 선형대수, 미적분, 확률 그리고 기초 Python 프로그래밍과 같은 기초적인 것들만 알고 있으면 충분합니다. 부록에서는 이 책에서 다루는 대부분의 수학을 포함하고 있으니 필요한 경우 참고하세요. 이 책은 소개서이기 때문에, 수학적으로 깊이 들어가는 것보다는 직관과 아이디어에 더 비중을 두고 있습니다. 흥미를 갖은 독자들은 많은 훌륭한 책들을 통해서 더 자세한 것을을 배울 수 있습니다. 예를 들면, Bela Bollobas의  [Linear Analysis](https://www.amazon.com/Linear-Analysis-Introductory-Cambridge-Mathematical/dp/0521655773) 는 선형대수와 함수분석을 아주 자세하게 다루고 있고, [All of Statistics](https://www.amazon.com/All-Statistics-Statistical-Inference-Springer/dp/0387402721) 은 통계에 대한 아주 훌륭한 가이드입니다. 만약 Python을 사용해보지 않았다면,  [Python tutorial](http://learnpython.org/) 을 참고하면 좋습니다. 물론, 수학적인 내용만 관심있다면, 프로그래밍 부분은 생략하면서 읽어도 됩니다. 물론 반대의 경우도 그렇습니다.


## Content and Structure

The book can be roughly divided into three sections:

이 책은 크게 세 부분으로 구성되어 있습니다.

* The first part covers prerequisites and basics. The first chapter offers an [Introduction to Deep Learning](../chapter_introduction/index.md) and how to use this book. [A Taste of Deep Learning](../chapter_crashcourse/index.md) provides the prerequisites required for hands-on deep learning, such as how to acquire and run the code covered in the book.  [Deep Learning Basics](../chapter_deep-learning-basics/index.md) covers the most basic concepts and techniques of deep learning, such as multi-layer perceptrons and regularization. If you are short on time or you only want to learn only about the most basic concepts and techniques of deep learning, it is sufficient to read the first section only.
* 첫번째 파트는 전제조건과 기본사항들을 다룹니다. 1장에서는 [딥러닝에 대한 소개](../chapter_introduction/index.md) 와 이 책에 대한 활용 방법을 설명합니다. [A Taste of Deep Learning](../chapter_crashcourse/index.md) 장에서는 이책의 코드를 어디서 받을 수 있고 어떻게 실행하는지 등과 같은 딥러닝 핸즈온에 필요한 것들을 이야기합니다. 만약 시간이 없거나 딥러닝의 가장 기본적인 개넘과 기법들만을 배우고자 한다면, 이 파트만 읽어도 충분합니다.
* The next three chapters focus on modern deep learning techniques. [Deep Learning Computation](../chapter_deep-learning-computation/index.md) describes the various key components of deep learning calculations and lays the groundwork for the later implementation of more complex models. [Convolutional Neural Networks](../chapter_convolutional-neural-networks/index.md) are explained next. They have made deep learning a success in computer vision in recent years. [Recurrent Neural Networks](../chapter_recurrent-neural-networks/index.md) are commonly used to process sequence data in recent years. Reading through the second section will help you grasp modern deep learning techniques.
* 두번째 파트인 다음 세장은 최근의 딥러닝 기법을 다룹니다. [Deep Learning Computation](../chapter_deep-learning-computation/index.md) 은 딥러닝 연산의 다양한 주요 요소들을 설명하면서, 더 복잡한 모델 구현을 위한 기본을 다질 수 있도록 합니다. 다음 장은 [Convolutional Neural Networks](../chapter_convolutional-neural-networks/index.md) 인데, 이는 최근 몇년동안 컴퓨터 비전에서 성과를 거두고 있는 딥러닝 기술입니다. 그리고 순서가 있는 데이터를 처리하는데 일반적으로 사용되는 [Recurrent Neural Networks](../chapter_recurrent-neural-networks/index.md) 를 다룹니다. 이 두분째 파트를 읽으면서 여러분은 최근 딥러닝 기술을 이해할 수 있을 것입니다.
* Part three discusses scalability, efficiency and applications. In particular, we discuss various [Optimization Algorithms](../chapter_optimization/index.md) used to train deep learning models. The next chapter examines several important factors that affect the [Performance](../chapter_computational-performance/index.md) of deep learning computation, such as regularization. Chapters 9 and 10  illustrate major applications of deep learning in computer vision and natural language processing respectively. This part is optional, depending on the reader's interests.
* 마지막 파트는 확장성, 효율성과 응용을 다룹니다. 딥러닝 모델을 학습시키는데 사용되는 다양한  [Optimization Algorithms](../chapter_optimization/index.md) 을 설명한 후, 정규화와 같이 딥러닝 연산 [Performance](../chapter_computational-performance/index.md) 에 영향을 주는 중요한 요소들에 대해서 살펴봅니다. 9장과 10장은 컴퓨터 비전과 자연어처리에서 사용되는 딥러닝의 응용에 대해서 알아봅니다. 이 파트는 여러분의 관심에 따라서 선택적으로 읽어도 됩니다.

An outline of the book is given below. The arrows provide a graph of prerequisites. If you want to learn the basic concepts and techniques of deep learning in a short time, simply read through the first three chapters; if you want to advance further, you will need the next three chapters. The last four chapters are optional, based on the reader's interests.

이책의 구성을 다음 그림과 같습니다. 화살표는 선행되어야하는 관계를 의미합니다. 만약 빠른 시간안에 딥러닝의 기본적인 개념과 기법들을 배워야한다면, 1장~3장만 읽으면됩니다. 더 심도있는 내용을 원하면, 그 다음 3장(4장~6장)을 읽으세요. 마지막 4장은 독자의 관심에 따라서 옵션입니다.

![Book structure](../img/book-org.svg)


## Code

This book features executable code in every section. The code can be modified and re-run to see how it affects the results. We recognize the importance of an interactive learning experience in deep learning. Unfortunately, deep learning remains to be poorly understood in theoretical terms. As a result, many arguments rely heavily on phenomenological experience that is best gained by experimentation with the code provided. The textual explanation may be insufficient to cover all the details, despite our best attempts. We are hopeful that this situation will improve in the future, as more theoretical progress is made. For now, we strongly advise that the reader further his understanding and gain insight by changing the code, observing the outcomes and summarizing the whole process.

이책은 모든 절에 동작하는 코드를 포함하고 있습니다. 코드들을 수정하고 다시 수행해서 결과에 어떤 영향을 미치는지도 확인할 수 있습니다. 이렇게 한 이유는 딥러닝에서 상호적인 학습 경험이 중요하다는 것을 알아냈기 때문입니다. 아쉽게도 딥러닝은 이론적으로 잘 이해되지 않고 있습니다. 그렇게 때문에, 많은 논의들은 코드를 수행해서 얻은 경험에 많이 의존하고 있습니다. 글로 설명하는 것은 최선의 노력을 해도 모든 자세한 것들을 다루기에 충분하지 않을 수 있습니다. 이론적인 진전이 더 많들어지면 그 떄는 이런 상황이 좋아질 것을 기대하지만, 지금은 독자들이 코드를 바꾸고, 결과를 관찰하고, 전반적인 과정을 요약하는 것을 통해서 이해도를 높이고, 직관을 얻는 것을 강력히 권장합니다.

Code in this book are based on the Apache MXNet. MXNet is an open-source framework for deep learning  which is the preferred choice of AWS (Amazon Cloud Services). It is used in many colleges and companies. All the code in this book have passed the test under MXNet 1.2.0. However, due to the rapid development of deep learning, some of the code *in the print edition* may not work properly in future versions of MXNet. The online version will remain up-to-date, though. In case of such problems, please refer to the section ["Installation and Running"](../chapter_prerequisite/install.md) to update the code and their runtime environment. In addition, to avoid unnecessary repetition, we encapsulate the frequently-imported and referred-to functions, classes, etc. in this book in the `d2l` package with version number 1.0.0.  We give a detailed overview of these functions and classes in the appendix [“d2l package index”](../chapter_appendix/d2l.md)

이책의 코드는 Apache MXNet을 기반으로 합니다. MXNet은 딥러닝을 위한 오픈소스 프래임워입니다. 이는 AWS(Amazon Cloud Services)가 선호한 선택이며, 많은 대학과 회사에서 사용되고 있습니다. 이책의 모든 코드는 MXNet 1.2.0을 이용해서 테스트되었으나, 딥러닝의 빠른 발전으로 어떤 코드는 이후 MXNet 버전에서는 이책의 **인쇄버전**의 코드가 잘 작동하지 않을 수 있습니다. 하지만, 온라인 버전은 계속 최신을 유지할 것입니다. 만약 그런 경우를 만난다면, ["Installation and Running"](../chapter_prerequisite/install.md) 를 참고해서 코드와 실행황경을 업데이트하세요. 그리고, 불필요한 반복을 피하기 위해서, 이책에서 자주 import 되거나 자주 참조되는 함수, 클래스 등은 `d2l` 패키지에 넣었습니다. `d2l` 패키지의 버전은 1.0.0 입니다. 포함된 함수와 클래스에 대한 자세한 내용은  [“d2l package index”](../chapter_appendix/d2l.md)에 있습니다.

This book can also serve as an MXNet primer. The main purpose of our code is to provide another way to learn deep learning algorithms in addition to text, images, and formulae. This book offers an interactive environment to understand the actual effects of individual models and algorithms on actual data. We only use the basic functionalities of MXNet's modules such as `ndarray`, `autograd`, `gluon`, etc. to familiarize yourself with the implementation details of deep learning algorithms. Even if you use other deep learning frameworks in your research or work, we hope that this code will help you better understand deep learning algorithms.

이 책은 MXNet 소개서로 활용될 수도 있습니다. 코드를 사용한 주요 목적은 텍스트, 이미지, 공식과 더불어 딥러닝 알고리즘을 배우는 또 다른 방법을 제공하는데 있습니다. 이 책은 각 모델과 알고리즘의 실제 데이터에 대한 실제 영향을 이해하기 위해서 인터엑티브한 환경을 제공합니다. 딥러닝 알고리즘의 구현에 대한 자세한 내용을 설명하기 위해서 MXNet 모듈의 기본적인 기능 - `ndarray`, `autograd`, `gluon`  등 - 만을 사용합니다. 여러분의 연구나 업무에서 다른 딥러닝 프래임워크를 사용하는 경우에도, 이 코드는 딥러닝 알고리즘에 대한 이해를 높이는데 도움이 될 것이라고 기대합니다.

## Forum

The discussion forum of this book is [discuss.mxnet.io](https://discuss.mxnet.io/). When you have questions on any section of the book, please scan the QR code at the end of the section to participate in its discussions. The authors of this book and MXNet developers are frequent visitors and participants on the forum.

이 책의 내용에 대한 논의 포럼은 [discuss.mxnet.io](https://discuss.mxnet.io/) 에 있습니다. 이 책의 어느 부분이던지 질문이 있을 때는, 각 절 끝에 있는 QR 코드를 스캔해서 논의에 참여할 수 있습니다. 이 책의 저자와 MXNet 개발자들이 포럼에 자주 방문하고 참여하고 있습니다.

## Problems

1. Register an account on the discussion forum of this book [discuss.mxnet.io](https://discuss.mxnet.io/).
1. Install Python on your computer.
1. 이 책의 논의 포럼 [discuss.mxnet.io](https://discuss.mxnet.io/) 에 계정을 생성하세요.
1. 여러분의 컴퓨터에 Python을 설치하세요.



## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2311)

![](../img/qr_how-to-use.svg)
