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

교과서들은 대개 일련의 주제들을 차례로 상세하게 가르치는 방식으로 만들어져 있습니다. 예를 들어 Chris Bishop의 유명한 교과서 [Pattern Recognition and Machine Learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738)는 각 주제를 철저히 가르쳐 주는 반면, 선형 회귀 분석 챕터까지 가기 위해 꽤 많은 분량을 소화해야만 합니다. 이 책의 방대한 설명을 좋아하는 전문가들도 있지만, 초보자에게 입문서로 사용하기에는 다소 적절하지 않습니다.



In this book, we will teach most concepts *just in time*.
In other words, you will learn concepts at the very moment
that they are needed to accomplish some practical end.
While we take some time at the outset to teach
fundamental preliminaries, like linear algebra and probability,
we want you to taste the satisfaction of training your first model
before worrying about more esoteric probability distributions.

이 책에서는 대부분의 개념을 *필요한 시점(just in time)*에 가르칠 것입니다. 즉, 실제 목표를 달성하는데 필요한 바로 그 순간에 해당 개념을 배우게됩니다. 선형대수학이나 확률과 같은 기본적인 내용을 배우기 위해 처음에 약간의 시간이 걸리겠지만, 난해한 확률 분포를 걱정하기 전에 첫 번째 모델을 훈련시키는 만족감을 느끼기를 바랍니다.



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

기본적인 수학적 배경에 대한 특별 과정을 제공하는 몇 가지 노트북을 제외하면, 이후의 각 챕터에서는 적절한 수의 새로운 개념을 소개하고 실제 데이터셋을 사용하는 1개의 작업 예제를 제공합니다. 이와 같은 접근 방법 때문에 책 구성이 쉽지 않았습니다. 어떤 모델은 논리적으로 하나의 노트북으로 그룹화될 수 있습니다. 한편 어떤 아이디어들은 여러 모델을 연속적으로 실행해서 가장 잘 배울 수 있습니다. 반면에, *작업 예제 1개, 노트북 1개*의 원칙을 지키면 큰 이점이 있습니다. 이렇게 하면 이 책의 코드를 활용해 자신만의 연구 프로젝트를 쉽게 시작할 수 있습니다. 노트북 하나를 복사하고 수정하기만 하면 됩니다.



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

실행 코드는 배경 설명과 필요에 따라 번갈아가며 배치됩니다. 우리는 종종 충분히 설명하기 전에 어떤 도구를 사용하는 (즉, 배경 설명은 나중에 하는) 실수를 하기도 할 것입니다. 예를 들어, 왜 유용한지 또는 왜 작동하는지 완전히 설명하기 전에 *확률적 경사 하강법(Stochastic Gradient Descent)* 을 사용하기도 합니다. 그 결과, 이 책의 독자는 문제를 풀기 위한 정보를 (저자들의 접근법을 믿어준다면) 빠르게 얻게 됩니다.



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

이 책은 딥러닝의 개념을 처음부터 가르칩니다. 때로는 딥러닝 프레임워크의 고급 추상화를 통해 일반적으로 사용자로부터 숨겨진 모델의 세부적인 부분까지 탐구하기도 합니다. 특히 특정 레이어나 옵티마이저에서 일어나는 모든 것을 이해해야 하는 기본 튜토리얼에서 접근하는 방식입니다. 이런 경우, 우리는 대개 두 가지 버전의 예제를 제시합니다. 하나는 NumPy 인터페이스와 자동 미분에만 의존해 모든 것을 처음부터 구현하는 예제이고, 다른 하나는 Gluon을 사용해 간결한 코드를 작성하는 현실적인 예제입니다. 일단 몇 가지 구성요소의 동작을 배운 다음에는, 이후의 튜토리얼에서는 Gluon 버전만을 사용하기도 합니다.



### Content and Structure

###콘텐츠 및 구조

The book can be roughly divided into three parts,
which are presented by different colors in :numref:`fig_book_org`:

이 책은 :numref:`fig_book_org`에 다른 색상으로 표시된 것처럼 대략 세 부분으로 나눌 수 있습니다.



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
* 첫 번째 부분은 기본 사항과 예비 지식을 다룹니다. :numref: `chap_introduction`은 딥러닝에 대한 소개를 제공합니다. 그런 다음, :numref: `chap_preliminaries`에서는 데이터를 저장하고 조작하는 방법, 선형 대수, 미적분 및 확률의 기본 개념을 기반으로 다양한 수치 연산을 적용하는 방법 등, 딥러닝의 핸즈온에 필요한 기초 내용을 빠르게 설명합니다. :numref: `chap_linear`와 :numref: `chap_perceptrons'는 선형 회귀, 다계층 퍼셉트론, 정규화와 같은 딥러닝의 가장 기본적인 개념과 기술을 다룹니다.




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
* 그 다음 다섯 챕터에서는 최신 딥러닝 기술에 초점을 맞춥니다. :numref: `chap_computation`은 딥러닝 연산의 다양한 핵심 구성 요소를 설명하고, 이후에 더 복잡한 모델을 구현하기 위한 기반을 마련합니다. 다음으로 :numref: `chap_cnn`과 :numref: `chap_modern_cnn`에서는 최신 컴퓨터 비전 시스템의 근간이 되는 강력한 도구인 합성곱 신경망(CNN, Convolutional Neural Network)을 소개합니다. 이어서 :numref: `chap_rnn` 및 :numref: `chap_modern_rnn`에서는 데이터의 시간적/순차적 구조를 활용해 자연어 처리 및 시계열 예측에 흔히 사용되는 순환 신경망(RNN, Recurrent Neural Network)을 소개합니다. :numref: `chap_attention`에서는, 최근에 자연어 처리에서 RNN을 대체하기 시작한 어텐션(attention) 메커니즘이라는 기술을 사용하는 새로운 클래스의 모델을 소개합니다. 이 부분을 통해, 최신 딥러닝 애플리케이션에 사용된 기본 도구들에 대해 이해할 수 있을 것입니다.




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
* 세 번째 부분에서는 확장성, 효율성 및 애플리케이션에 대해 설명합니다. 첫째, :numref: `chap_optimization`에서는 딥러닝 모델의 훈련에 사용되는 몇 가지 일반적인 최적화 알고리즘에 대해 설명합니다. 다음 챕터인 :numref: `chap_performance`에서는 딥러닝 코드의 계산 성능에 영향을 미치는 몇 가지 핵심 요소를 살펴봅니다. :numref: `chap_cv`에서는 컴퓨터 비전 분야의 주요 딥 러닝 응용 프로그램을 예를 들어 설명하고, :numref: `chap_nlp_pretrain` 및 :numref: `chap_nlp_app`에서는 언어 표현 모델을 사전 학습하고 자연어 처리 작업에 적용하는 방법을 보입니다.



### Code

###코드

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

저자들은 딥러닝에서 대화형 학습이 중요하다고 믿기 때문에, 이 책의 대부분의 섹션에 실행 코드가 포함되어 있습니다. 코드를 조금씩 바꿔가면서 결과를 관찰하는 시행착오를 통해서만 배울 수 있는 직관이 있기 때문입니다. 원하는 결과를 얻기 위해 코드를 수정하는 방법을 우아한 수학 이론으로 얻을 수 있을지도 모릅니다. 하지만 불행하게도 아직까지는 그런 멋진 이론을 만나지 못하고 있습니다. 최선의 노력에도 불구하고 여러 기술에 대한 공식적인 설명이 여전히 부족합니다. 이런 모델을 표현하기 위한 수학이 너무 어렵기도 하고 이 주제에 대한 진지한 연구가 최근에야 급속히 진행되고 있기 때문입니다. 이 책의 향후 판에서는 이번 판에서 통찰력을 제공하지 못하는 부분에서도 딥러닝 이론이 발전함에 따라 통찰력을 제공할 수 있기를 바랍니다.



At times, to avoid unnecessary repetition, we encapsulate
the frequently-imported and referred-to functions, classes, etc.
in this book in the `d2l` package.
For any block such as a function, a class, or multiple imports
to be saved in the package, we will mark it with
`#@save`. We offer a detailed overview of these functions and classes in :numref:`sec_d2l`.
The `d2l` package is light-weight and only requires
the following packages and modules as dependencies:

불필요한 반복 작업을 피하기 위해, 이 책에서 자주 import하거나 참조하는 함수와 클래스들은 `d2l` 패키지로 캡슐화되어 있습니다. 이 패키지에 저장될 함수, 클래스, 여러 import 들은 `#@save`와 같이 표시됩니다. 이러한 함수와 클래스에 대한 자세한 개요는 :numref:`sec_d2l`을 참조하시기 바랍니다. `d2l` 패키지는 경량이며 아래의 패키지와 모듈에만 의존합니다.



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

이 책의 대부분의 코드는 아파치 MXNet을 기반으로 합니다. MXNet은 딥러닝을 위한 오픈소스 프레임워크이며 많은 대학, 회사들과 AWS(Amazon Web Services, 아마존웹서비스)에서 선호되고 있습니다. 이 책의 모든 코드는 최신 MXNet 버전에서 테스트를 통과했습니다. 딥러닝 기술이 빠르게 발전함에 따라 *인쇄판*의 일부 코드가 MXNet의 향후 버전에서 제대로 동작하지 않을 수도 있습니다만, 온라인 버전은 최신 상태로 유지할 계획입니다. 이러한 문제가 발생할 경우 :ref:`chap_installation`을 참조하여 코드와 런타임 환경을 업데이트하시기 바랍니다.

다음은 MXNet에서 모듈을 import하는 방법입니다.

:end_tab:

:begin_tab:`pytorch`

Most of the code in this book is based on PyTorch.
PyTorch is an open-source framework for deep learning, which is extremely
popular in the research community.
All of the code in this book has passed tests under the newest PyTorch.
However, due to the rapid development of deep learning, some code
*in the print edition* may not work properly in future versions of PyTorch.
However, we plan to keep the online version up-to-date.
In case you encounter any such problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

이 책의 대부분의 코드는 PyTorch를 기반으로 합니다. PyTorch는 딥러닝을 위한 오픈소스 프레임워크이며 연구자들 사이에서 많은 인기를 끌고 있습니다. 이 책의 모든 코드는 최신 PyTorch 버전에서 테스트를 통과했습니다. 딥러닝 기술이 빠르게 발전함에 따라 *인쇄판*의 일부 코드가 PyTorch의 향후 버전에서 제대로 동작하지 않을 수도 있습니다만, 온라인 버전은 최신 상태로 유지할 계획입니다. 이러한 문제가 발생할 경우 :ref:`chap_installation`을 참조하여 코드와 런타임 환경을 업데이트하시기 바랍니다.

:end_tab:

:begin_tab:`tensorflow`

Most of the code in this book is based on TensorFlow.
TensorFlow is an open-source framework for deep learning, which is extremely
popular in both the research community and industrial.
All of the code in this book has passed tests under the newest TensorFlow.
However, due to the rapid development of deep learning, some code
*in the print edition* may not work properly in future versions of TensorFlow.
However, we plan to keep the online version up-to-date.
In case you encounter any such problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

Here is how we import modules from TensorFlow.

이 책의 대부분의 코드는 TensorFlow를 기반으로 합니다. TensorFlow는 딥러닝을 위한 오픈소스 프레임워크이며 연구자들과 산업계에서 많은 인기를 끌고 있습니다. 이 책의 모든 코드는 최신 TensorFlow 버전에서 테스트를 통과했습니다. 딥러닝 기술이 빠르게 발전함에 따라 *인쇄판*의 일부 코드가 TensorFlow의 향후 버전에서 제대로 동작하지 않을 수도 있습니다만, 온라인 버전은 최신 상태로 유지할 계획입니다. 이러한 문제가 발생할 경우 :ref:`chap_installation`을 참조하여 코드와 런타임 환경을 업데이트하시기 바랍니다.

다음은 TensorFlow에서 모듈을 import하는 방법입니다.

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

###대상 독자

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

이 책은 실용적인 딥러닝 기술을 확실히 이해하고자 하는 학생 (학부 또는 대학원), 엔지니어 및 연구원을 위한 것입니다. 이 책에서는 모든 개념을 처음부터 설명하기 때문에 딥러닝 또는 머신러닝에 대한 사전 지식이 필요하지 않습니다. 딥러닝 방법론을 완전히 설명하려면 수학과 프로그래밍이 필요하지만, 이 책에서는 여러분이 (기초)선형대수, 미적분, 확률, 파이썬 프로그래밍의 기본 지식만 있다고 가정할 것입니다. 또한 부록에서는, 이 책에서 다루는 대부분의 수학에 대한 참고 자료를 제공합니다. 대부분의 경우, 수학적 엄격함보다 직관과 아이디어를 우선시해서 설명하겠습니다. 관심있는 독자를 더 이끌어 줄 훌륭한 책이 많이 있습니다. 한 예로, Bela Bollobas의 "Linear Analysis" :cite:`Bollobas.1999`는 선형대수와 함수 분석을 심도있게 다루고 있습니다. "All of Statistics" :cite:`Wasserman.2013`은 통계학에 대한 훌륭한 안내서입니다. 이전에 파이썬을 써보지 않아다면, [Python tutorial] (http://learnpython.org/)을 살펴보시기 바랍니다.

### Forum

###게시판

Associated with this book, we have launched a discussion forum,
located at [discuss.d2l.ai](https://discuss.d2l.ai/).
When you have questions on any section of the book,
you can find the associated discussion page link at the end of each chapter.

이 책과 관련해 [discuss.d2l.ai] (https://discuss.d2l.ai/)에 게시판을 열었습니다. 이 책의 어디에서든 궁금한 점이 있다면, 각 챕터의 끝 부분에 있는 관련 게시판의 링크를 찾아보시기 바랍니다.



## Acknowledgments

## 감사의 말

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
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
tiepvupsu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, duythanhvn, paulaurel, graytowne, minhduc0711,
sl7423, tbaums, cuongvng, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap.

우리는 영어와 중국어 초안 작성에 수백 명의 도움을 받았습니다. 이 분들은 콘텐츠 개선에 도움을 주시고 귀중한 피드백을 제공하셨습니다. 특히, 모든 이를 위해 더 나은 영어 초안이 되도록 기여해주신 모든 분들에게 감사의 말씀 드립니다. 이 분들의 GitHub ID 또는 이름은 다음과 같습니다(무순). alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat, cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu, Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller, NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki, topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen, Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens, Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta, uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee, Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy, lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner, Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong, Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas, ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09, Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil, Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp, tiepvupsu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto, Katarzyna Biesialska, Gregory Bruss, duythanhvn, paulaurel, graytowne, minhduc0711, sl7423, tbaums, cuongvng, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic, the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom, abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang, StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl.



We thank Amazon Web Services, especially Swami Sivasubramanian,
Raju Gulabani, Charlie Bell, and Andrew Jassy for their generous support in writing this book. Without the available time, resources, discussions with colleagues, and continuous encouragement this book would not have happened.

Amazon Web Services, 특히 Swami Sivasubramanian에게 감사드립니다. Raju Gulabani, Charlie Bell, Andrew Jassy가 이 책을 쓰는 데 많은 도움을 주셨습니다. 지원해주신 시간과 자원, 동료들과의 토론, 지속적인 격려 덕분에 이 책이 쓰여질 수 있었습니다.



## Summary


## 요약

* Deep learning has revolutionized pattern recognition, introducing technology that now powers a wide range of  technologies, including computer vision, natural language processing, automatic speech recognition. 
  패턴인식에 혁명을 일으킨 딥러닝은 컴퓨터 비전, 자연어 처리, 자동 음성 인식 등 다양한 기술에 적용되고 있습니다.

  

* To successfully apply deep learning, you must understand how to cast a problem, the mathematics of modeling, the algorithms for fitting your models to data, and the engineering techniques to implement it all.
  딥 러닝을 성공적으로 적용하려면, 문제를 제기하는 방법, 모델링 수학, 모델을 데이터에 맞추는 알고리즘, 모든 구현을 위한 엔지니어링 기술을 이해해야 합니다.

  

* This book presents a comprehensive resource, including prose, figures, mathematics, and code, all in one place.
  이 책은 텍스트, 그림, 수학식과 코드를 동시에 제공하는 리소스입니다.

  

* To answer questions related to this book, visit our forum at https://discuss.d2l.ai/.
  이 책과 관련된 질문에 대답하려면 게시판(https://discuss.d2l.ai/)을 찾아보시기 바랍니다.

  

* All notebooks are available for download on GitHub.
  모든 노트북은 GitHub에서 다운로드 할 수 있습니다.




## Exercises


## 연습 문제

1. Register an account on the discussion forum of this book [discuss.d2l.ai](https://discuss.d2l.ai/). 
   이 책의 토론 게시판에 계정을 등록하세요: [Discussions](https://discuss.d2l.ai/).

1. Install Python on your computer. 
   컴퓨터에 파이썬을 설치합니다.

1. Follow the links at the bottom of the section to the forum, where you will be able to seek out help and discuss the book and find answers to your questions by engaging the authors and broader community. 
섹션 하단의 게시판 링크를 따라 가면 저자들과 커뮤니티를 통해 도움을 얻거나 책을 토론하고 질문에 대한 답변을 얻을 수 있습니다.
  
   

:begin_tab:
[Discussions](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/186)
:end_tab:

