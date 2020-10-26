# Deep Learning Computation
# 딥러닝 연산

:label:`chap_computation`
:label:`chap_computation`

0.15.0

Alongside giant datasets and powerful hardware,
great software tools have played an indispensable role
in the rapid progress of deep learning.
Starting with the pathbreaking Theano library released in 2007,
flexible open-source tools have enabled researchers
to rapidly prototype models, avoiding repetitive work
when recycling standard components
while still maintaining the ability to make low-level modifications.
Over time, deep learning's libraries have evolved
to offer increasingly coarse abstractions.
Just as semiconductor designers went from specifying transistors
to logical circuits to writing code,
neural networks researchers have moved from thinking about
the behavior of individual artificial neurons
to conceiving of networks in terms of whole layers,
and now often design architectures with far coarser *blocks* in mind.

거대한 데이터셋과 강력한 하드웨어와 함께, 훌륭한 소프트웨어 도구들은 딥러닝의 빠른 발전에 없어서는 안 될 역할을 해왔습니다. 2007년에 발표된 획기적인 Theano 라이브러리를 시작으로 유연한 오픈소스 도구들은 표준 컴포넌트들을 재사용하면서 여전히 저수준의 변경을 할 수 있도록 했기 때문에 연구자들이 모델을 빠르게 프로토타이핑을 할 수 있도록 해왔습니다. 시간이 지남에 따라 딥러닝 라이브러리들은 더 많은 추상화를 지원하도록 진화를 했습니다. 반도체 설계자가 트랜지스터를 설계에서 논리 회호로 코드를 작성하는 것으로 변한 것처럼, 뉴럴 네트워크 연구자들은 각 인공 뉴런들의 행동에 대한 생각에서 전체 층들의 관점에서 네트워크를 생각하는 것으로 이동했고, 이제는 훨씬 더 거친 *블록* 을 염두에 두고 아키텍처를 설계합니다.

So far, we have introduced some basic machine learning concepts,
ramping up to fully-functional deep learning models.
In the last chapter,
we implemented each component of an MLP from scratch
and even showed how to leverage high-level APIs
to roll out the same models effortlessly.
To get you that far that fast, we *called upon* the libraries,
but skipped over more advanced details about *how they work*.
In this chapter, we will peel back the curtain,
digging deeper into the key components of deep learning computation,
namely model construction, parameter access and initialization,
designing custom layers and blocks, reading and writing models to disk,
and leveraging GPUs to achieve dramatic speedups.
These insights will move you from *end user* to *power user*,
giving you the tools needed to reap the benefits
of a mature deep learning library while retaining the flexibility
to implement more complex models, including those you invent yourself!
While this chapter does not introduce any new models or datasets,
the advanced modeling chapters that follow rely heavily on these techniques.

지금까지 우리는 완전한 기능을 하는 딥러닝 모델을 개발하면서 몇 가지 기본적인 머신러닝 개념을 소개했습니다. 앞 장에서 MLP의 각 컴포넌트를 직접 구현했고, 상위 수준 API들을 사용해서 같은 모델들을 노력을 덜 들이고 만드는 방법도 알아봤습니다. 여기까지 빨리 도달하기 위해서 우리는 라이브러리들을 *사용* 했으니, *라이브러리가 어떻게 동작하는지* 에 대한 더 자세한 설명은 생략했습니다. 이 장에서 우리는 모델 만들기, 파라메터 접근 및 초기화와 같은 딥러닝 연산의 주요 컴포넌트들을 깊이 살펴보고, 커스텀 층과 블럭을 설계하기, 모델을 디스크에서 읽고 쓰기 그리고 굉장히 빠른 속도를 위한 GPU들을 활용하는 방법을 알아볼 것입니다. 이 통찰력은 여러분을 *최종 사용자(end user)* 에서 *고급 사용자(power user)*로 바꿔줄 것입니다.  여러분이 발명할 모델을 포함해, 더 복잡한 모델들을 구현하기 위한 유연성을 유지하면서도, 성숙한 딥러닝 라이브러리의 이점을 누리는데 필요한 도구들을 제공할 것입니다. 이 장은 새로운 모델이나 데이터셋을 소개하지 않지만, 앞으로 이어질 고급 모델링을 다루는 장들은 이 장에서 소개하는 기법들에 많이 의존할 것입니다.

```toc
:maxdepth: 2

model-construction
parameters
deferred-init
custom-layer
read-write
use-gpu

모델 생성
파라미터들
지연된 초기화
커스텀 층
읽기-쓰기
GPU 사용
```

