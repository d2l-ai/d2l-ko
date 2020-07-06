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
With these advances in hand, we can now build cars that drive themselves
with more autonomy than ever before (and less autonomy
than some companies might have you believe),
smart reply systems that automatically draft the most mundane emails,
helping people dig out from oppressively large inboxes,
and software agents that dominate the world's best humans
at board games like Go, a feat once thought to be decades away.
Already, these tools exert ever-wider impacts on industry and society,
changing the way movies are made, diseases are diagnosed,
and playing a growing role in basic sciences---from astrophysics to biology.

지난 5년간 딥러닝은 컴퓨터 비전, 자연어 처리, 음성 인식, 강화 학습, 통계적 모델링 등, 다양한 분야에서 빠르게 발전하며 세상을 놀라게 했습니다. 이러한 진보를 통해 우리는 이제 그 어느 때보다 더 자율적으로 운전하는 자동차 (어떤 회사들이 주장하는 것 만큼은 아니지만), 뻔한 이메일을 자동으로 답장해서 산처럼 쌓인 메일함에서 사람들을 구해내는 스마트 응답 시스템, 인간을 이기려면 수십년이 걸릴거라고 믿었던 바둑 같은 게임을 지배하는 소프트웨어 에이전트를 만들 수 있습니다. 이미 딥러닝은 영화 제작이나 질병 진단의 방식을 바꾸고 천체물리학에서 생물학에 이르는 기초 과학 분야에서도 점점 더 많은 역할을 하는 등, 우리 사회와 산업계에 넓은 영향을 미치고 있습니다.



## About This Book

## 이 책에 대해

This book represents our attempt to make deep learning approachable,
teaching you the *concepts*, the *context*, and the *code*.

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

어떤 컴퓨팅 기술이 최대의 영향력을 발휘하려면 충분한 이해를 바탕으로 문서화 되어야 하며, 잘 유지되는 성숙된 도구들이 지원되어야 합니다. 핵심적인 아이디어가 명확하게 전달되면 신참자들이 최신 내용을 익히는데 걸리는 시간을 최소화할 수 있습니다. 반복 작업을 자동화할 수 있는 성숙된 라이브러리가 있어야 하고, 실무자들이 필요에 따라 수정, 적용, 확장할 수 있도록 대표적인 작업들의 예제 코드가 준비되어야 합니다. 동적 웹 애플리케이션을 예로 들어 보겠습니다. 아마존과 같은 많은 회사들이 1990년대에 데이터베이스 기반 웹 애플리케이션을 성공적으로 개발했음에도 불구하고, 창조적 기업가를 뒷받침할 수 있는 이 기술의 잠재력은 최근 10년에 와서야 실현되었습니다. 그 중 한 가지 이유는 강력하고 잘 문서화된 프레임워크가 개발되었기 때문입니다.



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

간단한 딥러닝 애플리케이션라도 다양한 학문을 알아야 하기 때문에, 딥러닝의 잠재력을 테스트하는 것은 상당히 어렵습니다. 딥러닝을 적용하기 위해서는 (i) 특정한 방식으로 문제를 정의하기 위한 동기, (ii) 모델링에 사용된 수학, (iii) 모델을 데이터에 맞추기 위한 최적화 알고리즘, (iv) 모델을 효율적으로 훈련하는데 필요한 엔지니어링을 한꺼번에 이해하면서 동시에 수치 연산의 함정을 피하고 하드웨어를 최대한 활용할 수 있어야 합니다. 문제를 공식화하는 데 필요한 비판적 사고 능력, 그것을 풀 수 있는 수학, 그리고 이러한 솔루션들을 구현하기 위한 소프트웨어 도구를 한꺼번에 가르치는 것은 매우 어려운 과제입니다. 이 책에서 우리의 목표는 독자들이 최대한 빠르게 실무자가 될 수 있도록 통합된 리소스를 제시하는 것입니다.



At the time we started this book project,
there were no resources that simultaneously
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

우리가 이 책 프로젝트를 시작했을 때에는 (1) 최신 내용이고 (2) 현대 머신러닝의 전체 범위를 상당한 기술적 깊이로 다루면서 (3) 동시에, 매력적인 교과서와 핸즈온 튜토리얼과 같은 실행가능한 깨끗한 코드가 섞여 있는 교재가 존재하지 않았습니다. 우리는 딥러닝과 관련된 프레임워크를 사용하는 방법(예: TensorFlow에서 행렬을 사용한 기본 수치 계산) 또는 특정 기술(예: LeNet, AlexNet, ResNet 등의 코드 일부들)을 구현하는 방법에 대한 많은 예제 코드들을 여러 블로그와 GitHub 저장소에서 발견했습니다. 그러나 이 예제들은 대부분, 주어진 접근 방식을 구현하는 *방법*에 초점을 맞추느라 *왜* 그런 특정 알고리즘이 선택되었는지에 대한 설명은 생략하고 있었습니다. 특정 주제에 대한 리소스가 산발적으로 (예를 들면 웹 사이트 [Distill](http://distill.pub/) 또는 개인 블로그들에서) 나타나기도 했지만, 딥러닝의 몇몇 주제만 다루거나 코드가 없는 경우가 종종 있었습니다. 한편, 딥 러닝의 개념을 포괄적으로 다루는 여러 교과서(특히 :cite:`Goodfellow.Bengio.Courville.2016`)가 등장했지만, 이런 책들은 설명과 (이론을 구현한) 코드를 결합하지 못해서 독자들이 직접 구현하기 어려운 경우가 있었습니다. 또한 유료 교육기관에 등록해야만 볼 수 있는 자료들도 많습니다.



We set out to create a resource that could
(i) be freely available for everyone;
(ii) offer sufficient technical depth to provide a starting point on the path
to actually becoming an applied machine learning scientist;
(iii) include runnable code, showing readers *how* to solve problems in practice;
(iv) allow for rapid updates, both by us
and also by the community at large;
and (v) be complemented by a [forum](http://discuss.d2l.ai)
for interactive discussion of technical details and to answer questions.

우리는 (i) 누구나 무료로 사용할 수 있고 (ii) 머신러닝 과학자가 되고 싶은 사람이 출발점으로 삼을 수 있을 만큼 충분한 기술적 깊이를 제공하며 (iii) 실제로 문제를 푸는 *방법*을 보여주는, 실행가능한 코드를 포함하고  (iv) 저자들과 커뮤니티에 의해 빠르게 업데이트될 수 있으며 (v) 상세한 기술 내용에 대해 대화식 토론과 답변이 가능한 [게시판] (http://discuss.d2l.ai)으로 보완 가능한 자료를 만들기 시작했습니다. 



These goals were often in conflict.
Equations, theorems, and citations are best managed and laid out in LaTeX.
Code is best described in Python.
And webpages are native in HTML and JavaScript.
Furthermore, we want the content to be
accessible both as executable code, as a physical book,
as a downloadable PDF, and on the Internet as a website.
At present there exist no tools and no workflow
perfectly suited to these demands, so we had to assemble our own.
We describe our approach in detail in :numref:`sec_how_to_contribute`.
We settled on GitHub to share the source and to allow for edits,
Jupyter notebooks for mixing code, equations and text,
Sphinx as a rendering engine to generate multiple outputs,
and Discourse for the forum.
While our system is not yet perfect,
these choices provide a good compromise among the competing concerns.
We believe that this might be the first book published
using such an integrated workflow.

이러한 목표는 종종 충돌했습니다. 수학식, 정리와 인용은 LaTeX에서 가장 잘 관리되고 레이아웃 될 수 있습니다. 코드는 파이썬으로 가장 잘 설명됩니다. 웹 페이지는 HTML과 자바 스크립트에서 기본입니다. 우리는 실행 가능한 코드, 실제 책, 다운로드 가능한 PDF, 인터넷 웹 사이트 모두에서 액세스 가능한 콘텐츠를 원했습니다. 현재는 이러한 요구조건을 완벽하게 지원하는 도구나 워크 플로우가 존재하지 않기 때문에, 이 책에서는 독자적인 방법을 만들어 사용했습니다. 이에 대해서는 :numref:`sec_how_to_contribute` 에서 자세히 설명하겠습니다. 소스를 공유하고 편집하기 위해서 Github를 사용했고, 코드, 수학식, 텍스트를 동시에 사용하기 위해 Jupyter 노트북을, 다양한 출력물을 생성하는 렌더링 엔진으로 Sphinx를, 게시판을 위해 Discourse를 사용했습니다. 우리의 시스템이 아직 완벽하지는 않지만, 이와 같은 선택의 결과로 적절한 타협점을 찾을 수 있었습니다. 아마도 이 책은 이와 같은 통합 워크플로를 사용해서 출판된 첫번째 책일지도 모릅니다.





### Learning by Doing

### 직접 하면서 배우기

Many textbooks teach a series of topics, each in exhaustive detail.
For example, Chris Bishop's excellent textbook :cite:`Bishop.2006`,
teaches each topic so thoroughly, that getting to the chapter
on linear regression requires a non-trivial amount of work.
While experts love this book precisely for its thoroughness,
for beginners, this property limits its usefulness as an introductory text.

In this book, we will teach most concepts *just in time*.
In other words, you will learn concepts at the very moment
that they are needed to accomplish some practical end.
While we take some time at the outset to teach
fundamental preliminaries, like linear algebra and probability,
we want you to taste the satisfaction of training your first model
before worrying about more esoteric probability distributions.

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

This book will teach deep learning concepts from scratch.
Sometimes, we want to delve into fine details about the models
that would typically be hidden from the user
by deep learning frameworks' advanced abstractions.
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
In :numref:`chap_cv`,
we illustrate
major applications of deep learning in computer vision.
In :numref:`chap_nlp_pretrain` and :numref:`chap_nlp_app`,
we show how to pretrain language representation models and apply
them to natural language processing tasks.


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

At times, to avoid unnecessary repetition, we encapsulate
the frequently-imported and referred-to functions, classes, etc.
in this book in the `d2l` package.
For any block such as a function, a class, or multiple imports
to be saved in the package, we will mark it with
`#@save`. We offer a detailed overview of these functions and classes in :numref:`sec_d2l`.
The `d2l` package is light-weight and only requires
the following packages and modules as dependencies:

```{.python .input  n=1}
#@tab all
#@save
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`

Most of the code in this book is based on Apache MXNet.
MXNet is an open-source framework for deep learning
and the preferred choice of AWS (Amazon Web Services),
as well as many colleges and companies.
All of the code in this book has passed tests under the newest MXNet version.
However, due to the rapid development of deep learning, some code
*in the print edition* may not work properly in future versions of MXNet.
However, we plan to keep the online version up-to-date.
In case you encounter any such problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

Here is how we import modules from MXNet.
:end_tab:

:begin_tab:`pytorch`

Most of the code in this book is based on PyTorch.
PyTorch is an open-source framework for deep learning, which is extremely
popular in the research community.
All of the code in this book has passed tests under the the newest PyTorch.
However, due to the rapid development of deep learning, some code
*in the print edition* may not work properly in future versions of PyTorch.
However, we plan to keep the online version up-to-date.
In case you encounter any such problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

Here is how we import modules from PyTorch.
:end_tab:

:begin_tab:`tensorflow`

Most of the code in this book is based on TensorFlow.
TensorFlow is an open-source framework for deep learning, which is extremely
popular in both the research community and industrial.
All of the code in this book has passed tests under the the newest TensorFlow.
However, due to the rapid development of deep learning, some code
*in the print edition* may not work properly in future versions of TensorFlow.
However, we plan to keep the online version up-to-date.
In case you encounter any such problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

Here is how we import modules from TensorFlow.
:end_tab:

```{.python .input  n=1}
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input  n=1}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
```

```{.python .input  n=1}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

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
located at [discuss.d2l.ai](https://discuss.d2l.ai/).
When you have questions on any section of the book,
you can find the associated discussion page link at the end of each chapter.


## Acknowledgments

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
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, ruslo, Rafael Schlatter, liusy182, Giannis Pappas, ruslo,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
tiepvupsu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, duythanhvn, paulaurel, graytowne, minhduc0711,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
cuongvng, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap.

We thank Amazon Web Services, especially Swami Sivasubramanian,
Raju Gulabani, Charlie Bell, and Andrew Jassy for their generous support in writing this book. Without the available time, resources, discussions with colleagues, and continuous encouragement this book would not have happened.

우리는 영어와 중국 초안 모두에 대한 수백 명의 기여자에게 빚을 지고 있습니다. 그들은 콘텐츠를 개선하는 데 도움이 귀중한 피드백을 제공했습니다. 특히, 우리는 모두를 위해 더 나은 버전을 만들기 위해 힘써준 영어 초안의 모든 기여자에게 감사드립니다. 그들의 Github ID와 이름 (제공된 경우)은 : bowen0701, ChaiBapChya (Chaitanya Prakash Bapat), kirk86, MLWhiz (Rahul Agarwal), mstewart141, muelleme (Mike Müller), sfermigier, sundeepteki, vishaalkapoor, YaYaB. 더해서, 아마존 웹서비시즈에 감사 드리며, 특히 Swami Sivasubramanian, Raju Gulabani, Charlie Bell, and Andrew Jassy에게 이 책을 쓰도록 충분히 지원해 주신 대해 감사드립니다. 사용 가능한 시간, 자원, 동료와의 토론, 지속적인 격려 덕분에 책이 만들어 질 수 있었습니다.




## Summary

* Deep learning has revolutionized pattern recognition, introducing technology that now powers a wide range of  technologies, including computer vision, natural language processing, automatic speech recognition.
* To successfully apply deep learning, you must understand how to cast a problem, the mathematics of modeling, the algorithms for fitting your models to data, and the engineering techniques to implement it all.
* This book presents a comprehensive resource, including prose, figures, mathematics, and code, all in one place.
* To answer questions related to this book, visit our forum at https://discuss.d2l.ai/.
* All notebooks are available for download on GitHub.


## Exercises

1. Register an account on the discussion forum of this book [discuss.d2l.ai](https://discuss.d2l.ai/).
1. Install Python on your computer.
1. Follow the links at the bottom of the section to the forum, where you will be able to seek out help and discuss the book and find answers to your questions by engaging the authors and broader community.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/186)
:end_tab:
