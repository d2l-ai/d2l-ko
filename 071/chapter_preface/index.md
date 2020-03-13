# Preface

# 서문



Just a few years ago, there were no legions of deep learning scientists
developing intelligent products and services at major companies and startups.
When the youngest among us (the authors) entered the field,
machine learning did not command headlines in daily newspapers.
Our parents had no idea what machine learning was,
let alone why we might prefer it to a career in medicine or law.
Machine learning was a forward-looking academic discipline
with a narrow set of real-world applications.
And those applications, e.g., speech recognition and computer vision,
required so much domain knowledge that they were often regarded
as separate areas entirely for which machine learning was one small component.
Neural networks then, the antecedents of the deep learning models
that we focus on in this book, were regarded as outmoded tools.

불과 몇 년 전만 해도, 대부분의 기업에서는 지능형 제품과 서비스를 개발하는 지금처럼 많은 딥러닝 과학자들을 볼 수 없었습니다. 이 책의 저자들 중에서 가장 어린 막내가 입사했을 때도, 머신 러닝이 지금처럼 일간지의 헤드라인을 장식하는 일은 드물었습니다. 저희들의 부모님은 왜 우리가 의학이나 법학보다 머신 러닝을 좋아하는지는 커녕, 도대체 그게 뭔지도 전혀 알지 못하셨습니다. 머신 러닝은 실제 응용 분야가 좁은 미래지향적인 학문이었습니다. 또한 음성 인식, 컴퓨터 비전과 같은 응용 분야들은 많은 도메인 지식이 필요하기 때문에, 종종 머신 러닝이 작은 구성요소로 포함된 별도의 독립적인 영역으로 간주되었습니다. 이 책에서 우리가 중점적으로 다루는 딥 러닝 모델의 선구자격인 신경망은 낡은 도구로 간주되었습니다.

In just the past five years, deep learning has taken the world by surprise,
driving rapid progress in fields as diverse as computer vision,
natural language processing, automatic speech recognition,
reinforcement learning, and statistical modeling.
With these advances in hand, we can now build cars that drive themselves with more autonomy than ever before (and less autonomy than some companies might have you believe), smart reply systems that automatically draft the most mundane emails, helping people dig out from oppressively large inboxes,
and software agents that dominate the world's best humans
at board games like Go, a feat once thought to be decades away.
Already, these tools exert ever-wider impacts on industry and society,
changing the way movies are made, diseases are diagnosed,
and playing a growing role in basic sciences---from astrophysics to biology.

지난 5년간 딥러닝은 컴퓨터 비전, 자연어 처리, 음성 인식, 강화 학습, 통계적 모델링 등, 다양한 분야에서 빠르게 발전하며 세상을 놀라게 했습니다. 이러한 진보를 통해 우리는 이제 그 어느 때보다 더 자율적으로 운전하는 자동차 (어떤 회사들이 주장하는 것 만큼은 아니지만), 뻔한 이메일을 자동으로 답장해서 산처럼 쌓인 메일함에서 사람들을 구해내는 스마트 응답 시스템, 인간을 이기려면 수십년이 걸릴거라고 믿었던 바둑 같은 게임을 지배하는 소프트웨어 에이전트를 만들 수 있습니다. 이미 딥러닝은 영화 제작이나 질병 진단의 방식을 바꾸고 천체물리학에서 생물학에 이르는 기초 과학 분야에서도 점점 더 많은 역할을 하는 등, 우리 사회와 산업계에 넓은 영향을 미치고 있습니다.



## About This Book

## 이 책에 대해

This book represents our attempt to make deep learning approachable,
teaching you both the *concepts*, the *context*, and the *code*.

이 책은 딥러닝의 *개념*, *문맥*과 *코드*를 동시에 설명하는 방법으로 쉽게 접근할 수 있게 하려는 저희의 노력의 결과물입니다.



### One Medium Combining Code, Math, and HTML

### 코드, 수학, HTML이 통합된 책

For any computing technology to reach its full impact,
it must be well-understood, well-documented, and supported by
mature, well-maintained tools.
The key ideas should be clearly distilled,
minimizing the onboarding time needing to bring new practitioners up to date.
Mature libraries should automate common tasks,
and exemplar code should make it easy for practitioners
to modify, apply, and extend common applications to suit their needs.
Take dynamic web applications as an example.
Despite a large number of companies, like Amazon,
developing successful database-driven web applications in the 1990s,
the potential of this technology to aid creative entrepreneurs
has been realized to a far greater degree in the past ten years,
owing in part to the development of powerful, well-documented frameworks.

어떤 컴퓨팅 기술이 최대의 영향력을 발휘하려면 충분한 이해를 바탕으로 문서화 되어야 하며, 잘 유지되는 성숙된 도구들이 지원되어야 합니다. 핵심적인 아이디어가 명확하게 전달되면 신참자들이 최신 내용을 익히는데 걸리는 시간을 최소화할 수 있습니다. 반복 작업을 자동화할 수 있는 성숙된 라이브러리가 있어야 하고, 실무자들이 필요에 따라 수정, 적용, 확장할 수 있도록 대표적인 작업들의 예제 코드가 준비되어야 합니다. 동적 웹 애플리케이션을 예로 들어 보겠습니다. Amazon과 같은 많은 회사들이 1990년대에 데이터베이스 기반 웹 애플리케이션을 성공적으로 개발했음에도 불구하고, 창조적 기업가를 뒷받침할 수 있는 이 기술의 잠재력은 최근 10년에 와서야 실현되었습니다. 그 중 한 가지 이유는 강력하고 잘 문서화된 프레임워크가 개발되었기 때문입니다.











Testing the potential of deep learning presents unique challenges
because any single application brings together various disciplines.
Applying deep learning requires simultaneously understanding
(i) the motivations for casting a problem in a particular way;
(ii) the mathematics of a given modeling approach;
(iii) the optimization algorithms for fitting the models to data;
and (iv) and the engineering required to train models efficiently,
navigating the pitfalls of numerical computing
and getting the most out of available hardware.
Teaching both the critical thinking skills required to formulate problems,
the mathematics to solve them, and the software tools to implement those
solutions all in one place presents formidable challenges.
Our goal in this book is to present a unified resource
to bring would-be practitioners up to speed.

딥러닝의 실현은, 모든 어플리케이션은 다양한 분야의 통합을 가져왔기 때문에, 고유한 과제를 제시합니다. 딥러닝을 적용하기 위해서는, (i) 특정 방식으로 문제제기를 위한 동기, (ii) 주어진 모델링 접근을 위한 수학, (iii) 모델을 데이터에 맞추기 위한 최적화 알고리즘 (iv) 모델을 효율적으로 훈련하는데 필요한 엔지니어링들을 이해하고 수치 연산의 위험(Pitfalls)을 탐색하고 가용한 하드웨어의 최대한 활용하세요. 문제를 수식으로 만들기 위한 비판적 사고 기술, 그것을 풀 수 있는 수학, 그리고 이러한 솔루션들을 모두 구현하기 위한 소프트웨어 도구, 이 모두를 한 곳에서 가리치는 것은 매우 어려운 과제 입니다. 이 책에서 우리의 목표는 독자들이 최대한 빠르게 실무자가 될 수 있도록, 통합된 자원을 제시하는 것입니다.

We started this book project in July 2017 when we needed
to explain MXNet's (then new) Gluon interface to our users.
At the time, there were no resources that simultaneously
(i) were up to date; (ii) covered the full breadth
of modern machine learning with substantial technical depth;
and (iii) interleaved exposition of the quality one expects
from an engaging textbook with the clean runnable code
that one expects to find in hands-on tutorials.
We found plenty of code examples for
how to use a given deep learning framework
(e.g., how to do basic numerical computing with matrices in TensorFlow)
or for implementing particular techniques
(e.g., code snippets for LeNet, AlexNet, ResNets, etc)
scattered across various blog posts and GitHub repositories.
However, these examples typically focused on
*how* to implement a given approach,
but left out the discussion of *why* certain algorithmic decisions are made.
While some interactive resources have popped up sporadically
to address a particular topic, e.g., the engaging blog posts
published on the website [Distill](http://distill.pub), or personal blogs,
they only covered selected topics in deep learning,
and often lacked associated code.
On the other hand, while several textbooks have emerged,
most notably :cite:`Goodfellow.Bengio.Courville.2016`,
which offers a comprehensive survey of the concepts behind deep learning,
these resources do not marry the descriptions
to realizations of the concepts in code,
sometimes leaving readers clueless as to how to implement them.
Moreover, too many resources are hidden behind the paywalls
of commercial course providers.

우리는 MXNet의 새로운 Gluon 인터페이스를 사용자에게 설명해야 했던, 2017년 7월에 이 책 프로젝트를 시작했습니다. 동시에 당시에는, (1) 최신 상태이고, (2) 기술적 깊이와 유사한 것으로 현대의 머신 러닝의 전체 폭을 다루고 (3) 실행가능한 코드가 있는 교재부터 활발한 튜토리얼 같은 것들이 없었습니다. 우리는 딥러닝과 관련된 프레임워크를 사용하는 방법 (예: TensorFlow에서 행렬을 사용하여 기본 수치 계산을 수행하는 방법) 또는 특정 기술 (예: LeNet, AlexNet, ResNet 등의 코드 일부들)을 구현해 내는 방법 혹은 예제 코드들을 블로그 게시물 형태 또는 GitHub에서 많이 발견했습니다. 그러나 이 예제는 일반적으로 주어진 접근 방식을 구현하는 *방법* 에 초점을 맞추었지만, *왜* 그런 특정 알고리즘이 선택되었는지에 대한 논의를 생략했습니다. 웹 사이트 [Distill](http://distill.pub/) 또는 개인 블로그와 같은 곳에서 산발적인 주제들이 논의 되는 동안, 딥러닝에서 선택한 주제만을 다루거나, 종종 관련된 코드가 부족한 경우가 많았습니다. 다른 한편으로는, 딥러닝의 개념에 대한 훌륭한 자료를 제공하는 [Goodfellow, Bengio and Courville, 2016](https://www.deeplearningbook.org/)에 여러 교과서가 등장하지만 이러한 자료는 설명과 코드로 그 개념을 실현하는 방법을 잘 녹아내지 못했습니다. 결국 독자들은 그것을 구현하기위한 단서들을 스스로 찾도록 방치되었습니다. 또한 너무 많은 자료가 유료 교육과정 제공 업체에 숨겨져 있습니다.

We set out to create a resource that could
(1) be freely available for everyone;
(2) offer sufficient technical depth to provide a starting point on the path
to actually becoming an applied machine learning scientist;
(3) include runnable code, showing readers *how* to solve problems in practice;
(4) that allowed for rapid updates, both by us
and also by the community at large;
and (5) be complemented by a [forum](http://discuss.mxnet.io)
for interactive discussion of technical details and to answer questions.

우리는 그래서 자료를 만들기 시작했습니다. 이 자료는

(1) 모든 사람이 자유롭게 이용할 수 있고, (2) 실제로 응용 머신 러닝 과학자가 될때 필요한, 충분한 기술적 깊이를 제공하고, (3) 실행 가능한 코드를 포함시키고, 실제로 문제를 해결하는 *방법* 을 독자에게 보여 주며, (4) 우리뿐만 아니라 많은 커뮤니티에 의해 신속하게 업데이트 할 수 있으며, (5) 기술 세부 사항에 대한 상호 토론과 질문에 답하기 위해 [forum](http://discuss.mxnet.io/) 를 이용하여 보완됩니다.

These goals were often in conflict.
Equations, theorems, and citations are best managed and laid out in LaTeX.
Code is best described in Python.
And webpages are native in HTML and JavaScript.
Furthermore, we want the content to be
accessible both as executable code, as a physical book,
as a downloadable PDF, and on the internet as a website.
At present there exist no tools and no workflow
perfectly suited to these demands, so we had to assemble our own.
We describe our approach in detail in :numref:`sec_how_to_contribute`.
We settled on Github to share the source and to allow for edits,
Jupyter notebooks for mixing code, equations and text,
Sphinx as a rendering engine to generate multiple outputs,
and Discourse for the forum.
While our system is not yet perfect,
these choices provide a good compromise among the competing concerns.
We believe that this might be the first book published
using such an integrated workflow.

이러한 목표는 종종 충돌했습니다. 방정식, 이론 및 인용식은 LaTeX에서 가장 잘 관리되고 배치됩니다. 코드는 파이썬을 이용하야 가장 잘 설명 됩니다. 그리고 웹페이지들은 기본적으로 HTML과 Javascript로 구성됩니다. 더 나아가, 우리는 코드가 물리적인 책, 다운로드 가능한 PDF, 인터넷 웹페이지로 모두 접근 가능하지만 동시에 실행 가능하 길 바랍니다. 현재 이러한 이러한 요구에 완벽하게 부합하는 도구나 워크 플로우가 존재하지 않으므로 직접 만들어야 했습니다. 우리는 우리의 접근 방식을 [부록](https://github.com/d2l-ai/d2l-ko/blob/master/chapter_appendix/how-to-contribute.md)에 상세하게 기록하였습니다. 우리는 이를 위해, 소스를 공유하고 편집을 허용하기 위해 Github를, 다양한 코드, 방적식과 문장들을 다루기위한 Jupyter 노트북, 다양한 출력을 생산하기 위한 렌더링 엔진으로 Sphinx, 포럼을 위해 Discourse를 각각 설정하였습니다. 아직 우리의 시스템이 완벽하지는 않지만, 이러한 선택은 경쟁적인 상황에서 좋은 절충안을 제공합니다. 우리는 이러한 통합 워크 플로우를 사용하여 출판된 첫 번째 책이 될 거라고 생각합니다.

### Learning by Doing

## 직접 하면서 배우기

Many textbooks teach a series of topics, each in exhaustive detail.
For example, Chris Bishop's excellent textbook :cite:`Bishop.2006`,
teaches each topic so thoroughly, that getting to the chapter
on linear regression requires a non-trivial amount of work.
While experts love this book precisely for its thoroughness,
for beginners, this property limits its usefulness as an introductory text.

많은 교과서는 일련의 주제들을 각각 철저히 상세하게 가르칩니다. 예를 들어, Chris Bishop의 훌륭한 교과서 [Pattern Recognition and Machine Learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738)는 각 주제를 철저히 가르쳐 줍니다. 선형 회귀 분석의 장에 들어가려면 작지 않은 양의 작업이 필요합니다. 전문가들은 이 책을 철저하기 때문에 좋아하지만, 초보자에게는 이 점이 소개 텍스트로써의 유용함을 감소시킵니다.

In this book, we will teach most concepts *just in time*.
In other words, you will learn concepts at the very moment
that they are needed to accomplish some practical end.
While we take some time at the outset to teach
fundamental preliminaries, like linear algebra and probability,
we want you to taste the satisfaction of training your first model
before worrying about more esoteric probability distributions.

이 책에서는 대부분의 개념을 *딱 그시점(Just in time)* 에 가르쳐 드리겠습니다. 즉, 실제 목적을 달성하기 위해 필요한 바로 그 순간에 개념을 배우게 될 것입니다. 선형대수학이나 확률과 같은 기본적인 사항을 가르치기 위해 처음에는 약간의 시간이 걸리지만, 좀 더 색다른 확률 분포에 대해 걱정하기 전에, 첫 번째 모델을 훈련하는 만족감을 느끼기를 바랍니다.

Aside from a few preliminary notebooks that provide a crash course
in the basic mathematical background,
each subsequent chapter introduces both a reasonable number of new concepts
and provides single self-contained working examples---using real datasets.
This presents an organizational challenge.
Some models might logically be grouped together in a single notebook.
And some ideas might be best taught by executing several models in succession.
On the other hand, there is a big advantage to adhering
to a policy of *1 working example, 1 notebook*:
This makes it as easy as possible for you to
start your own research projects by leveraging our code.
Just copy a notebook and start modifying it.

기본적인 수학적 배경에 대한 특별 과정을 제공하는 몇 가지 예비 노트북을 제외하고, 각각의 후속 노트북은 합리적인 수의 새로운 개념을 소개하고 실제 데이터셋을 사용하여 하나의 독립적인 작업 예제를 제공합니다. 이것은 구조적인 과제를 제시한다. 일부 모델은 논리적으로 단일 노트북으로 그룹화될 수 있습니다. 그리고 어떤 아이디어들은 여러 모델을 연속적으로 실행하여 가장 잘 가르쳐 질 수 있습니다. 반면에 *1 작업 예제, 1 노트북* 의 정책을 준수하면 큰 이점이 있습니다. 이렇게 하면 우리의 코드를 활용하여 여러분의 연구 프로젝트를 가능한 한 쉽게 시작할 수 있습니다. 단지 단일 노트북을 복사하고 수정하기만 하면 됩니다.

We will interleave the runnable code with background material as needed.
In general, we will often err on the side of making tools
available before explaining them fully (and we will follow up by
explaining the background later).
For instance, we might use *stochastic gradient descent*
before fully explaining why it is useful or why it works.
This helps to give practitioners the necessary
ammunition to solve problems quickly,
at the expense of requiring the reader
to trust us with some curatorial decisions.

우리는 필요에 따라 실행 가능한 코드를 배경 자료와 함께 배치합니다. 일반적으로, 우리는 종종 일반적으로 우리는 도구를 완전히 설명하기 전에 사용할 수 있도록 만드는 측면에서 종종 오류를 겪을 것입니다 (나중에 배경을 설명하여 후속 조치를 취합니다). 예를 들어, 왜 유용한지 또는 왜 작동하는지 완전히 설명하기 전에 *확률적 경사 하강법(Stochastic Gradient Descent)* 을 사용할 수 있습니다. 이것은 실무자에게 문제를 신속하게 해결하는 데 필요한 정보를 제공하는 데 도움이되며, 이를 위해 독자가 적어도 단기적으로, 몇 가지 결정에 있어서는 우리를 신뢰해야 합니다.

Throughout, we will be working with the MXNet library,
which has the rare property of being flexible enough for research
while being fast enough for production.
This book will teach deep learning concepts from scratch.
Sometimes, we want to delve into fine details about the models
that would typically be hidden from the user
by Gluon's advanced abstractions.
This comes up especially in the basic tutorials,
where we want you to understand everything
that happens in a given layer or optimizer.
In these cases, we will often present two versions of the example:
one where we implement everything from scratch,
relying only on the NumPy interface and automatic differentiation,
and another, more practical example,
where we write succinct code using Gluon.
Once we have taught you how some component works,
we can just use the Gluon version in subsequent tutorials.

전반적으로 MXNet 라이브러리로 작업하게 될 것입니다. MXNet 라이브러리는 연구하기에 충분히 유연하면서도 완성품 만들기에 있어서도 충분히 빠르다는 드문 특성을 가지고 있습니다. 이 책은 딥러닝 개념을 처음부터 가르칠 것입니다. 때때로, 우리는 `Gluon`의 고급 기능에 의해 사용자로부터 숨겨진 모델에 대한 세부 사항을 탐구 하고자 합니다. 이것은 특히 기본 튜토리얼에서 나옵니다. 여기서는 주어진 층에서 일어나는 모든 것을 이해하기를 원합니다. 이 경우, 우리는 일반적으로 두 가지 버전의 예제를 제시합니다. 하나는 처음부터 모든 것을 구현하고 NDArray 와 자동미분(automatic differentitation)에만 의존하고 다른 하나는 `Gluon`을 사용하여 간결하게 수행하는 방법을 보여줍니다. 한번만 층(layer)이 어떻게 동작하는지 가르쳐 주면, 후속 자습서에서 `Gluon` 버전을 사용할 수 있습니다.


### Content and Structure

The book can be roughly divided into three parts,
which are presented by different colors in :numref:`fig_book_org`:

![Book structure](../img/book-org.svg)
:label:`fig_book_org`


* The first part covers basics and preliminaries.
:numref:`chap_introduction` offers an introduction to deep learning.
Then, in :numref:`chap_preliminaries`,
we quickly bring you up to speed on the prerequisites required
for hands-on deep learning, such as how to store and manipulate data,
and how to apply various numerical operations based on basic concepts
from linear algebra, calculus, and probability.
:numref:`chap_linear` and :numref:`chap_perceptrons`
cover the most basic concepts and techniques of deep learning,
such as linear regression, multilayer perceptrons and regularization.

* The next five chapters focus on modern deep learning techniques.
:numref:`chap_computation` describes the various key components of deep
learning calculations and lays the groundwork
for us to subsequently implement more complex models.
Next, in :numref:`chap_cnn` and :numref:`chap_modern_cnn`,
we introduce convolutional neural networks (CNNs), powerful tools
that form the backbone of most modern computer vision systems.
Subsequently, in :numref:`chap_rnn` and :numref:`chap_modern_rnn`, we introduce
recurrent neural networks (RNNs), models that exploit
temporal or sequential structure in data, and are commonly used
for natural language processing and time series prediction.
In :numref:`chap_attention`, we introduce a new class of models
that employ a technique called attention mechanisms
and they have recently begun to displace RNNs in natural language processing.
These sections will get you up to speed on the basic tools
behind most modern applications of deep learning.

* Part three discusses scalability, efficiency, and applications.
First, in :numref:`chap_optimization`,
we discuss several common optimization algorithms
used to train deep learning models.
The next chapter, :numref:`chap_performance` examines several key factors
that influence the computational performance of your deep learning code.
In :numref:`chap_cv` and :numref:`chap_nlp`, we illustrate
major applications of deep learning in computer vision
and natural language processing, respectively.



### Code
:label:`sec_code`

Most sections of this book feature executable code because of our belief
in the importance of an interactive learning experience in deep learning.
At present, certain intuitions can only be developed through trial and error,
tweaking the code in small ways and observing the results.
Ideally, an elegant mathematical theory might tell us
precisely how to tweak our code to achieve a desired result.
Unfortunately, at present, such elegant theories elude us.
Despite our best attempts, formal explanations for various techniques
are still lacking, both because the mathematics to characterize these models
can be so difficult and also because serious inquiry on these topics
has only just recently kicked into high gear.
We are hopeful that as the theory of deep learning progresses,
future editions of this book will be able to provide insights
in places the present edition cannot.

Most of the code in this book is based on Apache MXNet.
MXNet is an open-source framework for deep learning
and the preferred choice of AWS (Amazon Web Services),
as well as many colleges and companies.
All of the code in this book has passed tests under the newest MXNet version.
However, due to the rapid development of deep learning, some code
*in the print edition* may not work properly in future versions of MXNet.
However, we plan to keep the online version remain up-to-date.
In case you encounter any such problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

At times, to avoid unnecessary repetition, we encapsulate
the frequently-imported and referred-to functions, classes, etc.
in this book in the `d2l` package.
For any block block such as a function, a class, or multiple imports
to be saved in the package, we will mark it with
`# Saved in the d2l package for later use`.
The `d2l` package is light-weight and only requires
the following packages and modules as dependencies:

```{.python .input  n=1}
# Saved in the d2l package for later use
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
import os
import pandas as pd
import random
import re
import sys
import tarfile
import time
import zipfile
```

We offer a detailed overview of these functions and classes in :numref:`sec_d2l`.


### Target Audience

This book is for students (undergraduate or graduate),
engineers, and researchers, who seek a solid grasp
of the practical techniques of deep learning.
Because we explain every concept from scratch,
no previous background in deep learning or machine learning is required.
Fully explaining the methods of deep learning
requires some mathematics and programming,
but we will only assume that you come in with some basics,
including (the very basics of) linear algebra, calculus, probability,
and Python programming.
Moreover, in the Appendix, we provide a refresher
on most of the mathematics covered in this book.
Most of the time, we will prioritize intuition and ideas
over mathematical rigor.
There are many terrific books which can lead the interested reader further.
For instance, Linear Analysis by Bela Bollobas :cite:`Bollobas.1999`
covers linear algebra and functional analysis in great depth.
All of Statistics :cite:`Wasserman.2013` is a terrific guide to statistics.
And if you have not used Python before,
you may want to peruse this [Python tutorial](http://learnpython.org/).


### Forum

Associated with this book, we have launched a discussion forum,
located at [discuss.mxnet.io](https://discuss.mxnet.io/).
When you have questions on any section of the book,
you can find the associated discussion page by scanning the QR code
at the end of the section to participate in its discussions.
The authors of this book and broader MXNet developer community
frequently participate in forum discussions.

## Acknowledgments

## 감사의 글

We are indebted to the hundreds of contributors for both
the English and the Chinese drafts.
They helped improve the content and offered valuable feedback.
Specifically, we thank every contributor of this English draft
for making it better for everyone.
Their GitHub IDs or names are (in no particular order):
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, vishwesh5, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, IgorDzreyev, Ha Nguyen, pmuens,
alukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, prasanth5reddy, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, ruslo, Rafael Schlatter, liusy182, Giannis Pappas, ruslo,
ati-ozgur, qbaza, dchoi77, Adam Gerson, lkhphuc, Mark Atwood, christabella, vn09.

We thank Amazon Web Services, especially Swami Sivasubramanian,
Raju Gulabani, Charlie Bell, and Andrew Jassy for their generous support in writing this book. Without the available time, resources, discussions with colleagues, and continuous encouragement this book would not have happened.

우리는 영어와 중국 초안 모두에 대한 수백 명의 기여자에게 빚을 지고 있습니다. 그들은 콘텐츠를 개선하는 데 도움이 귀중한 피드백을 제공했습니다. 특히, 우리는 모두를 위해 더 나은 버전을 만들기 위해 힘써준 영어 초안의 모든 기여자에게 감사드립니다. 그들의 Github ID와 이름 (제공된 경우)은 : bowen0701, ChaiBapChya (Chaitanya Prakash Bapat), kirk86, MLWhiz (Rahul Agarwal), mstewart141, muelleme (Mike Müller), sfermigier, sundeepteki, vishaalkapoor, YaYaB. 더해서, 아마존 웹서비시즈에 감사 드리며, 특히 Swami Sivasubramanian, Raju Gulabani, Charlie Bell, and Andrew Jassy에게 이 책을 쓰도록 충분히 지원해 주신 대해 감사드립니다. 사용 가능한 시간, 자원, 동료와의 토론, 지속적인 격려 덕분에 책이 만들어 질 수 있었습니다.


## Summary

* Deep learning has revolutionized pattern recognition, introducing technology that now powers a wide range of  technologies, including computer vision, natural language processing, automatic speech recognition.
* To successfully apply deep learning, you must understand how to cast a problem, the mathematics of modeling, the algorithms for fitting your models to data, and the engineering techniques to implement it all.
* This book presents a comprehensive resource, including prose, figures, mathematics, and code, all in one place.
* To answer questions related to this book, visit our forum at https://discuss.mxnet.io/.
* Apache MXNet is a powerful library for coding up deep learning models and running them in parallel across GPU cores.
* Gluon is a high level library that makes it easy to code up deep learning models using Apache MXNet.
* Conda is a Python package manager that ensures that all software dependencies are met.
* All notebooks are available for download on GitHub.
* If you plan to run this code on GPUs, do not forget to install the necessary drivers and update your configuration.


## Exercises

1. Register an account on the discussion forum of this book [discuss.mxnet.io](https://discuss.mxnet.io/).
1. Install Python on your computer.
1. Follow the links at the bottom of the section to the forum, where you will be able to seek out help and discuss the book and find answers to your questions by engaging the authors and broader community.
1. Create an account on the forum and introduce yourself.


## [Discussions](https://discuss.mxnet.io/t/2311)

![](../img/qr_preface.svg)
