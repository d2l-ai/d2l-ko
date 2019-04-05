# Introduction

Before we could begin writing, the authors of this book, like much of
the work force, had to become caffeinated.  We hopped in the car and
started driving.  Using an iPhone, Alex called out 'Hey Siri',
awakening the phone's voice recognition system.  Then Mu commanded
'directions to Blue Bottle coffee shop'.  The phone quickly displayed
the transcription of his command.  It also recognized that we were
asking for directions and launched the Maps application to fulfill our request.  Once launched, the Maps app identified a number of routes. Next to each route, the phone displayed a predicted transit time. While we fabricated this story for pedagogical convenience, it demonstrates that in the span of just a few seconds, our everyday interactions with a smartphone can engage several machine learning
models.

저자들은 이 책을 쓰기 전에, 많은 노동력이 필요한 일처럼, 많은 카페인이 필요했습니다. 상상해 봅시다, 우리는 차에 올라타서 운전을 하기 시작했습니다. 아이폰을 사용하는 Alex는 핸드폰의 음성 인식 시스템을 부르기 위해서 'Hey Siri'라고 외쳤습니다. 그리고는 Mu는 '블루 보틀 커피샵으로 가는길을 알려줘'라고 명령을 했습니다. 핸드폰은 그의 명령을 글로 바꿔서 화면에 빠르게 보여줍니다. 우리가 길을 묻는 것을 알아채고는 우리의 요청에 응하기 위해서 지도 앱을 띄웁니다. 지도 앱이 실행되자 마자 여러 경로를 찾아냅니다. 각 경로 옆에는 예상 소요 시간이 함께 표시됩니다. 설명하기 위해서 이야기를 지어낸 것이지만, 이 몇 초 동안에도 스마트폰을 사용하면서 여러 머신러닝 모델을 사용하는 것을 보여주고 있습니다.

If you've never worked with machine learning before, you might be
wondering what we're talking about.  You might ask, 'isn't that just
programming?' or 'what does *machine learning* even mean?'  First, to
be clear, we implement all machine learning algorithms by writing
computer programs.  Indeed, we use the same languages and hardware as
other fields of computer science, but not all computer programs
involve machine learning.  In response to the second question,
precisely defining a field of study as vast as machine learning is
hard.  It's a bit like answering, 'what is math?'.  But we'll try to
give you enough intuition to get started.

여러분이 지금까지 머신러닝을 다뤄본적이 없다면, 무엇에 대해서 이야기를 하고 있는지 모를 수도 있습니다. 어쩌면 '그냥 프로그래밍으로 작동하는거 아닌가요?' 라고 묻거나 '*머신러닝* 이 무엇을 의미하나요?' 라는 질문을 던질 수도 있습니다. 우선 확실하게 해두기 위해서, 모든 머신러닝 알고리즘은 컴퓨터 프로그램을 작성해서 구현됩니다. 사실 우리는 다른 컴퓨터 과학의 분야와 동일한 언어와 하드웨어를 사용합니다. 하지만, 모든 컴퓨터 프로그램이 머신러닝을 포함하는 것은 아닙니다. 두번째 질문에 대한 답은,  머신러닝은 방대한 분야이기 때문의 정의하기 어렵습니다. 이 질문은 마치 '수학이 무엇인가요?'라는 질문과 비슷합니다. 하지만, 여러분이 공부를 시작할 수 있도록 충분한 직관적인 설명을 해보겠습니다.


## A motivating example

Most of the computer programs we interact with every day
can be coded up from first principles.
When you add an item to your shopping cart,
you trigger an e-commerce application to store an entry
in a *shopping cart* database table,
associating your user ID with the product's ID.
We can write such a program from first principles,
launch without ever having seen a real customer.
When it's this easy to write an application
*you should not be using machine learning*.

우리가 매일 사용하는 컴퓨터 프로그램의 대부분은 제일 원칙(first principles)을 활용해서 코드화될 수 있습니다. 여러분이 쇼핑 카트에 물건을 담으면, 이커머스 어플리케이션은 어떤 항목(여러분의 user ID와 제품 ID를 연관시키는)을 *쇼핑 카트* 데이터베이스 태이블에 저장합니다. 우리는 이런 프로그램을 제일 원칙에 따라서 작성하고, 실제 고객을 본적이 없이도 런치할 수 있습니다. 어플리케이션을 만드는 것이 쉬울 경우에는 *머신러닝을 사용하지 말아야합니다.*

Fortunately (for the community of ML scientists)
for many problems, solutions aren't so easy.
Returning to our fake story about going to get coffee,
imagine just writing a program to respond to a *wake word*
like 'Alexa', 'Okay, Google' or 'Siri'.
Try coding it up in a room by yourself
with nothing but a computer and a code editor.
How would you write such a program from first principles?
Think about it... the problem is hard.
Every second, the microphone will collect roughly 44,000 samples.
What rule could map reliably from a snippet of raw audio
to confident predictions ``{yes, no}``
on whether the snippet contains the wake word?
If you're stuck, don't worry.
We don't know how to write such a program from scratch either.
That's why we use machine learning.

(머신러닝 과학자 커뮤니티에게는) 다행히도, 많은 문제들에 대한 해결책이 그리 쉽지 않습니다. 커피를 사러가는 이야기로 돌아가서, 'Alexa', 'Okay, Google', 이나 'Siri' 같은 *wake word* 에 응답하는 프로그램을 작성한다고 생각해보세요. 컴퓨터와 코드 편집기만 사용해서 여러분의 방에서 혼자 코드를 만들기를 해보세요. 제일 원칙을 이용해서 어떨게 그런 프로그램을 작성할 것인가요? 생각해보면... 이 문제는 어렵습니다. 매초마다 마이크로폰은 대략 44,000개 샘플을 수집합니다. 소리 조각으로 부터 그 소리 조각이 wake word를 포함하는지 신뢰있게  `{yes, no}` 로 예측하는 룰을 만들 수 있나요? 어떻게할지를 모른다고 해도 걱정하지 마세요. 우리도 그런 프로그램을 처음부터 어떻게 작성해야하는지 모릅니다. 이것이 바로 우리가 머신러닝을 사용하는 이유입니다.

![](../img/wake-word.svg)

Here's the trick.
Often, even when we don't know how to tell a computer
explicitly how to map from inputs to outputs,
we are nonetheless capable of performing the cognitive feat ourselves.
In other words, even if you don't know *how to program a computer* to recognize the word 'Alexa',
you yourself *are able* to recognize the word 'Alexa'.
Armed with this ability,
we can collect a huge *data set* containing examples of audio
and label those that *do* and that *do not* contain the wake word.
In the machine learning approach, we do not design a system *explicitly* to recognize
wake words right away.
Instead, we define a flexible program with a number of *parameters*.
These are knobs that we can tune to change the behavior of the program.
We call this program a model.
Generally, our model is just a machine that transforms its input into some output.
In this case, the model receives as *input* a snippet of audio,
and it generates as output an answer ``{yes, no}``,
which we hope reflects whether (or not) the snippet contains the wake word.

트릭을 알려드리겠습니다. 우리는 컴퓨터에게 명시적으로 입력과 출력을 어떻게 매핑해야하는지를 알려주는 것은 모르지만, 우리 자신은 인지적인 활동을 할 수 있는 능력이 있습니다. 즉, 우리는 'Alexa'라는 단어를 인지하도록 *컴퓨터를 프로그램하는 방법* 은 모르지만, 여러분은 그 단어를 *인지할 수* 있습니다. 이 능력을 사용해서, 우리는 오디오 샘플과 그 오디오 샘플이 wake word를 포함하는지 여부를 알려주는 레이블을 아주 많이 수집할 수 있습니다. 머신러닝 접근 방법은 우리가 wake word를 바로 인식할 수 있도록 *명시적* 인 시스템 디자인을 할 수 없지만, 대신 우리는 많은 수의 *파라메터*들을 갖는 유연한 프로그램을 정의할 수 있습니다. 이것들은 프로그램의 행동을 바꾸기 위해서 조작하는데 사용하는 손잡이들입니다. 이 프로그램을 모델이라고 부릅니다. 일반적으로 우리의 모델은 입력을 어떤 결과로 변환하는 머신일 뿐입니다. 이 경우에는, 모델은 오디오 조각을 *입력*으로 받아서, `{yes, no}` 답을 출력으로 생성하는데, 우리는 이 결과가 wake word의 포함 여부를 담기를 원합니다.

If we choose the right kind of model,
then there should exist one setting of the knobs
such that the model fires ``yes`` every time it hears the word 'Alexa'.
There should also be another setting of the knobs that might fire ``yes``
on the word 'Apricot'.
We expect that the same model should apply to 'Alexa' recognition and 'Apricot' recognition
because these are similar tasks.
However, we might need a different model to deal with fundamentally different inputs or outputs.
For example, we might choose a different sort of machine to map from images to captions,
or from English sentences to Chinese sentences.

만약 여러분이 좋은 모델을 선택했다면, 'Alexa' 단어를 들을 때 마다 `yes` 를 출력하는 모델을 만드는 파라메터 세트 하나가 존재할 것입니다. 마찬가지로 'Apricot' 단어에 대해서 `yes` 를 출력하는 것이 다른 조합이 있을 수 있습니다. 우리는 이 두가지가 비슷하기 때문에, 동일한 모델이 'Alexa' 인식과 'Apricot' 인식에 적용되기를 기대합니다. 하지만 근본적으로 다른 입력 또는 출력을 다루기 위해서는 다른 모델이 필요할 수도 있습니다. 예를 들어, 이미지와 캡션을 매팅하는 머신과 영어 문장을 중국어 문장으로 매핑하는 모델은 서로 다른 것을 사용할 것입니다.

As you might guess, if we just set the knobs randomly,
the model will probably recognize neither 'Alexa', 'Apricot',
nor any other English word.
Generally, in deep learning, the *learning*
refers precisely
to updating the model's behavior (by twisting the knobs)
over the course of a *training period*.

이미 예상했겠지만, 이 손잡이를 아무렇게 설명할 경우, 아마도 그 모델은 'Alexa', 'Apricot'  또는 어떤 영어 단어도 인식하지 못할 것입니다. 일반적으로 딥러닝에서는 *학습(learning)* 은 여러 *학습 기간*에 걸쳐서 모델의 행동 (손잡이를 돌리면서)을 업데이트하는 것을 말합니다.

The training process usually looks like this:

학습 과정은 보통 다음과 같습니다.

1. Start off with a randomly initialized model that can't do anything useful.
1. Grab some of your labeled data (e.g. audio snippets and corresponding ``{yes,no}`` labels)
1. Tweak the knobs so the model sucks less with respect to those examples
1. Repeat until the model is awesome.
1. 임의로 초기화되서 아무것도 유용한 것을 못하는 모델로 시작합니다.
1. 레이블을 갖은 데이터를 수집합니다. (예, 오디오 조각과 그에 해당하는 `{yes,no}` 레이블들)
1. 주어진 예제들에 모델이 조금 덜 이상하게 작동하도록 손잡이를 조정합니다.
1. 모델이 좋아질 때까지 반복합니다.

![](../img/ml-loop.svg)

To summarize, rather than code up a wake word recognizer,
we code up a program that can *learn* to recognize wake words,
*if we present it with a large labeled dataset*.
You can think of this act
of determining a program's behavior by presenting it with a dataset
as *programming with data*.

요약하면, wake word를 인식하는 코드를 작성하는 것이 아니라, *아주 많은 레이블이 있는 데이터셋이 있을 경우에* wake word를 인식하는 것을 *배우는* 프로그램을 작성하는 것입니다. 데이터셋을 제공해서 프로그램의 행동를 결정하는 것을 *programming with data*라고 생각할 수 있습니다.

We can 'program' a cat detector by providing our machine learning system with many examples of cats and dogs, such as the images below:

우리는 아래 이미지들과 같은 아주 많은 고양이와 개 샘플들을 모신러닝 시스템에 제공해서 고양이 탐지기 프로그램을 만들 수 있습니다.

|![](../img/cat1.png)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|
|:---------------:|:---------------:|:---------------:|:---------------:|
|cat|cat|dog|dog|

This way the detector will eventually learn to emit a very large positive number if it's a cat, a very large negative number if it's a dog, and something closer to zero if it isn't sure, but this is just barely scratching the surface of what machine learning can do.

이런 방법으로 탐지기는 결국에 고양이를 입력으로 받으면 아주 큰 양수를 결과로 나오게, 그리고 개를 입력으로 받으면 아주 큰 음수를 결과로 나오게 학습될 것입니다. 이 모델은 잘 모르겠으면 0과 가까운 수를 결과로 출력할 것입니다. 이 예는 머신러닝으로 할 수 있는 아주 일부입니다.


## The dizzying versatility of machine learning

This is the core idea behind machine learning:
Rather than code programs with fixed behavior,
we design programs with the ability to improve
as they acquire more experience.
This basic idea can take many forms.
Machine learning can address many different application domains,
involve many different types of models,
and update them according to many different learning algorithms.
In this particular case, we described an instance of *supervised learning*
applied to a problem in automated speech recognition.

이것이 머신러닝의 핵심 아이디어입니다. 정해진 행동에 대한 코드를 만드는 것이 아닌, 더 많은 경험을 하면 능력이 향상되는 프로그램을 디자인하는 것입니다. 이 기본 아이디어는 여러 형태들이 될 수 있습니다. 머신러닝은 여러 종류의 어플리케이션 도메인 문제를 풀수 있고, 모델의 다른 형태를 포함하고, 여러 학습 알고리즘에 따라 업데이트를 합니다. 앞에서 든 예의 경우 자동 음성 인식 문제에 적용된 *지도학습(supervised learning)*의 한 예입니다.

Machine Learning is a versatile set of tools that lets you work with data in many different situations where simple rule-based systems would fail or might be very difficult to build. Due to its versatility, machine learning can be quite confusing to newcomers.
For example, machine learning techniques are already widely used
in applications as diverse as search engines, self driving cars,
machine translation, medical diagnosis, spam filtering,
game playing (*chess*, *go*), face recognition,
data matching, calculating insurance premiums, and adding filters to photos.

간단한 규칙 기반의 시스템이 실패하거나 만들기 어려운 경우에 데이터를 활용할 수 있도록 하는 다양한 도구들의 집합이 머신러닝입니다. 이 다양함 때문에 머신러닝을 처음 접하는 경우 혼란스러울 수 있습니다. 예를 들어, 머신러닝 기술은 검색엔진, 자율주행차, 기계번역, 의료진단, 스팸 필터링, 게임, 얼굴 인식, 데이터 메칭, 보험 프리미엄 계산, 사진에 필터 적용 등 다양한 응용에서 이미 널리 사용되고 있습니다. 

Despite the superficial differences between these problems many of them share a common structure
and are addressable with deep learning tools.
They're mostly similar because they are problems where we wouldn't be able to program their behavior directly in code,
but we can *program them with data*.
Often times the most direct language for communicating these kinds of programs is *math*.
In this book, we'll introduce a minimal amount of mathematical notation,
but unlike other books on machine learning and neural networks,
we'll always keep the conversation grounded in real examples and real code.

이 문제들은 겉보기에는 달라보이지만, 많은 것들은 공통적인 구조를 가지고 있고, 딥러닝 도구를 이용해서 풀 수 있습니다. 이 문제들은 코드로 직접 프로그램이 어떻게 행동해야하는지를 작성하는 것이 불가능하지만, 데이터로 프로그램을 만들 수 있다는 것이 때문에 이들 대부분이 비슷합니다. 대부분의 경우 이런 종류의 프로그램을 설명하는 가장 직접적인 언어가 *수학*입니다. 이 책은 다른 머신러닝이나 뉴럴 네트워크 책과는 다르게 수학 표현은 최소화하고, 실제 예제와 실제 코드를 중심으로 설명하겠습니다.

## Basics of machine learning
When we considered the task of recognizing wake-words,
we put together a dataset consisting of snippets and labels.
We then described (albeit abstractly)
how you might train a machine learning model
to predict the label given a snippet.
This set-up, predicting labels from examples, is just one flavor of ML
and it's called *supervised learning*.
Even within deep learning, there are many other approaches,
and we'll discuss each in subsequent sections.
To get going with machine learning, we need four things:

wake word를 인식하는 문제를 이야기할 때, 음성 조각과 레이블로 구성된 데이터셋을 언급했습니다. (추상적이긴 하지만) 음성 조각이 주어졌을 때 레이블을 예측하는 머신러닝 모델을 어떻게 학습시킬 수 있는지 설명했습니다. 예제로부터 레이블을 예측하는 설정은 ML의 한 종류로 *지도학습(supervised learning)* 이라고 부릅니다. 딥러닝에서도 많은 접근법들이 있는데, 다른 절들에서 다루겠습니다. 머신러닝을 진행하기 위해서는 다음 4가지가 필요합니다.

1. Data
2. A model of how to transform the data
3. A loss function to measure how well we're doing
4. An algorithm to tweak the model parameters such that the loss function is minimized
5. 데이터
6. 데이터를 어떻게 변환할지에 대한 모델
7. 우리가 얼마나 잘하고 있는지를 측정하는 loss 함수
8. loss 함수를 최소화하도록 모델 파라메터를 바꾸는 알고리즘

### Data

Generally, the more data we have, the easier our job becomes.
When we have more data, we can train more powerful models. Data is at the heart of the resurgence of deep learning and many of most exciting models in deep learning don't work without large data sets. Here are some examples of the kinds of data machine learning practitioners often engage with:

일반적으로는 데이터가 많아질 수 록 일이 더 쉬워집니다. 더 많은 데이터가 있다면, 더 강력한 모델을 학습시킬 수 있습니다. 데이터는 딥러닝 부활의 중심이고 딥러닝에서 아주 흥미로운 많은 모델들은 많은 데이터가 없으면 만들어지지 못 했습니다. 머신러닝를 수행하는 여러분들이 자주 접하게될 몇가지 종류의 데이터는 다음과 같습니다.

* **Images:** Pictures taken by smartphones or harvested from the web, satellite images, photographs of medical conditions, ultrasounds, and radiologic images like CT scans and MRIs, etc.
* **Text:** Emails, high school essays, tweets, news articles, doctor's notes, books, and corpora of translated sentences, etc.
* **Audio:** Voice commands sent to smart devices like Amazon Echo, or iPhone or Android phones, audio books, phone calls, music recordings, etc.
* **Video:** Television programs and movies, YouTube videos, cell phone footage, home surveillance, multi-camera tracking, etc.
* **Structured data:** Webpages, electronic medical records, car rental records, electricity bills, etc.
* **이미지**: 스마트폰으로 찍거니 웹에서 수집한 사진들, 인공위성 이미지, 의료 사진, 초음파, CT 또는 MRI 같은 방사선 이미지등
* **텍스트**: 이메일, 고등학교 에세이, 트윗, 뉴스 기사, 의사의 기록, 책, 변역된 문장 등
* **오디오**: Amazon Echo, 아이폰, 또는 안드로이드 폰과 같은 스마트 디바이스에 전달될 음성 명령, 오디오 책, 전화 통화, 음악 녹음 등
* **비디오**: 텔레비전 프로그램, 영화, 유투브 비디오, 휴대전화 수신 범위(cell phone footage), 가정 감시 카메라, 다중 카메라를 이용한 추적 등

### Models

Usually the data looks quite different from what we want to accomplish with it.
For example, we might have photos of people and want to know whether they appear to be happy.
We might desire a model capable of ingesting a high-resolution image and outputting a happiness score.
While some simple problems might be addressable with simple models, we're asking a lot in this case.
To do its job, our happiness detector needs to transform hundreds of thousands of low-level features (pixel values)
into something quite abstract on the other end (happiness scores).
Choosing the right model is hard, and different models are better suited to different datasets.
In this book, we'll be focusing mostly on deep neural networks.
These models consist of many successive transformations of the data that are chained together top to bottom,
thus the name *deep learning*.
On our way to discussing deep nets, we'll also discuss some simpler, shallower models.

보통의 경우에 데이터는 이를 통해서 이루고자 하는 것과는 아주 다릅니다. 예를 들면 사람들의 사진을 가지고 있고, 사람이 행복한지 아닌지를 알아내고자 합니다. 모델이 고해상도 이미지를 받아서 행복 점수를 결과로 내도록 할 수 있습니다. 단순한 문제는 단순한 모델로 해결될 수 있지만, 이 경우에는 많은 것을 묻고 있습니다. 이를 하기 위해서는 우리의 행복 탐지기가 수십만개의 저수준(low-level) 피처들 (픽셀값들)를 행복 점수와 같은 상당히 추상적인 것으로 변환해야합니다. 정확한 모델을 선택하는 것은 어려운 일이고, 다른 모델들은 다른 데이터셋에 더 적합합니다. 이 책에서 우리는 대부분 딥 뉴럴 네트워크에 집중할 예정입니다. 이 모델들은 데이터 변환이 많이 연속적으로 구성되어있고, 따라서 이를 *딥 러닝(deep learning)* 이라고 합니다. 딥 넷을 논의하는 과정으로 우선 보다 간단한 또는 얕은 모델을 먼저 살펴보겠습니다.


###  Loss functions

To assess how well we're doing we need to compare the output from the model with the truth.
Loss functions give us a way of measuring how *bad* our output is.
For example, say we trained a model to infer a patient's heart rate from images.
If the model predicted that a patient's heart rate was 100bpm,
when the ground truth was actually 60bpm,
we need a way to communicate to the model that it's doing a lousy job.

우리의 모델이 얼마나 잘하고 있는지 평가하기 위해서 모델의 결과와 정답(truth)를 비교할 필요가 있습니다. Loss 함수는 우리 결과가 얼마나 나쁜지를 측정하는 방법을 제공합니다. 예를 들어, 이미지로 부터 환자의 심장 박동을 추론하는 모델을 학습시켰다고 하겠습니다. 환자의 실제 심장 박동은 60bpm인데, 모델이 환자의 심장 박동을 100bpm이하고 예측했다면, 모델이 틀린 일을 하고 있다는 것을 이야기할 방법이 필요합니다.

Similarly if the model was assigning scores to emails indicating the probability that they are spam,
we'd need a way of telling the model when its predictions are bad.
Typically the *learning* part of machine learning consists of minimizing this loss function.
Usually, models have many parameters.
The best values of these parameters is what we need to 'learn', typically by minimizing the loss incurred on a *training data*
of observed data.
Unfortunately, doing well on the training data
doesn't guarantee that we will do well on (unseen) test data,
so we'll want to keep track of two quantities.

다른 예로, 모델이 이메일이 스팸일 가능성을 점수로 알려준다면, 예측이 틀렸을 때 모델에게 알려줄 방법이 필요합니다. 일반적으로 머신러닝의 *학습* 부분은 이 loss 함수를 초소화하는 것으로 구성됩니다. 보통은 모델이 많은 파라메터를 갖습니다. 이 파라메터들의 가장 좋은 값은 관찰된 데이터의 *학습 데이터(training data)*에 대한 loss를 최소화하는 것을 통해서 '배우'기를 원하는 것입니다. 불행하게도, 학습데이터에 대해서 잘 하는 것이 (본적이 없는) 테스트 데이터에도 잘 작동한다는 것이 보장되지 않습니다. 그렇기 때문에, 우리는 두 값을 추적해야합니다.

 * **Training Error:** This is the error on the dataset used to train our model by minimizing the loss on the training set. This is equivalent to doing well on all the practice exams that a student might use to prepare for the real exam. The results are encouraging, but by no means guarantee success on the final exam.
 * **Test Error:** This is the error incurred on an unseen test set. This can deviate quite a bit from the training error. This condition, when a model fails to generalize to unseen data, is called *overfitting*. In real-life terms, this is the equivalent of screwing up the real exam despite doing well on the practice exams.
 * **학습 오류(training error)**: 학습 데이터에 대해 loss를 최소화 하면서 모델을 학습시킨 데이터에 대한 오류입니다. 비유하자면, 실제 시험을 준비하는 학생이 연습 시험에 대해서 모두 잘하는 것과 동일합니다. 이 결과가 좋긴하지만, 실제 시험에서도 잘본다는 보장은 없습니다.
 * **테스트 오류(test error)**: 보지 않은 테스트 셋에 대한 오류입니다. 학습 오류와는 상당히 다를 수 있습니다. 이런 경우 즉 보지 않은 데이터에 대한 일반화를 실패한 경우를 우리는 *오버피팅(overfitting)* 이라고 합니다. 실제 생활과 비유하면, 연습 시험은 모두 잘했는데 실제 시험은 망친것입니다.


### Optimization algorithms

Finally, to minimize the loss, we'll need some way of taking the model and its loss functions,
and searching for a set of parameters that minimizes the loss.
The most popular optimization algorithms for work on neural networks
follow an approach called gradient descent.
In short, they look to see, for each parameter which way the training set loss would move if you jiggled the parameter a little bit. They then update the parameter in the direction that reduces the loss.

마지막으로 loss를 최소화하기 위해서 우리는 모델과 loss 함수를 사용해서 loss를 최소화하는 파라메터 집합을 찾는 방법이 필요합니다. 뉴럴 네트워크에서 가장 유명한 최적화 알고리즘은 gradient descent 라고 불리는 방법을 따르고 있습니다. 간략하게 말하면, 각 파라메터에 대해서 파라메터를 조금 바꿨을 때 학습 셋에 대한 loss가 어느 방향으로 움직이는지를 봐서, loss 가 감소하는 방향으로 파라메터를 업데이트 합니다.

In the following sections, we will discuss a few types of machine learning in some more detail. We begin with a list of *objectives*, i.e. a list of things that machine learning can do. Note that the objectives are complemented with a set of techniques of *how* to accomplish them, i.e. training, types of data, etc. The list below is really only sufficient to whet the readers' appetite and to give us a common language when we talk about problems. We will introduce a larger number of such problems as we go along.

지금부터는 우리는 머신러닝의 몇가지 종류에 대해서 조금 자세히 살펴보겠습니다. 머신러닝이 할 수 있는 것들을 나열하는 것으로 시작합니다. 목적은 이를 달성하는 방법에 대한 기술 집합(즉, 학습, 데이터 종류 등)과 보완되는 것을 기억해두세요. 아래 목록은 여러분들이 공부를 시작하고, 우리가 문제를 이야기할 때 공동 언어를 쓸 수 있을 정도입니다. 더 많은 문제는 앞으로 계속 다룰 예정입니다.

## Supervised learning

Supervised learning addresses the task of predicting *targets* given input data.
The targets, also commonly called *labels*, are generally denoted *y*.
The input data points, also commonly called *examples* or *instances*, are typically denoted $\boldsymbol{x}​$.
The goal is to produce a model $f_\theta​$ that maps an input $\boldsymbol{x}​$ to a prediction $f_{\theta}(\boldsymbol{x})​$

지도학습(supervised learning)은 주어진 입력 데이터에 대한 *타켓(target)*을 예측하는 문제를 풉는 것입니다. 타겟은 종종 *레이블(label)* 이라고 불리고, 기호로는  *y* 로 표기합니다. 입력 데이터 포인트는 *샘플(sample)* 또는 *인스턴스(instance)* 라고 불리기도 하고,  $\boldsymbol{x}$ 로 표기됩니다. 입력  $\boldsymbol{x}$ 를 예측 on $f_{\theta}(\boldsymbol{x})$ 로 매핑하는 모델  $f_\theta$ 을 생성하는 것이 목표입니다.

To ground this description in a concrete example,
if we were working in healthcare,
then we might want to predict whether or not a patient would have a heart attack.
This observation, *heart attack* or *no heart attack*,
would be our label $y$.
The input data $\boldsymbol{x}$ might be vital signs such as heart rate, diastolic and systolic blood pressure, etc.

이해를 돕기 위해서 예를 들어보겠습니다. 여러분이 의료분야에서 일을 한다면, 어떤 환자에게 심장 마비가 일어날지 여부를 예측하기를 원할 것입니다. 이 관찰, *심장 마비* 또는 *정상*, 은 우리의 레이블 $y$ 가 됩니다. 입력 $x$ 는 심박동, 이완기 및 수축 혈압 등 바이탈 사인들이 될 것입니다.

The supervision comes into play because for choosing the parameters $\theta$, we (the supervisors) provide the model with a collection of *labeled examples* ($\boldsymbol{x}_i, y_i$), where each example $\boldsymbol{x}_i$ is matched up against its correct label.

파라메터 $\theta$ 를 선택하는데, 우리는(감독자) *레이블이 있는 예들* , ($\boldsymbol{x}_i, y_i$)을 모델에게 제공하기 때문에 감독이 작동합니다. 이 때,  $\boldsymbol{x}_i$ 는 정확한 레이블과 매치되어 있습니다.

In probabilistic terms, we typically are interested estimating
the conditional probability $P(y|x)​$.
While it's just one among several approaches to machine learning,
supervised learning accounts for the majority of machine learning in practice.
Partly, that's because many important tasks
can be described crisply as estimating the probability of some unknown given some available evidence:

확률 용어로는 조건부 확률  $P(y|x)$을 추정하는데 관심이 있습니다. 이것은 머신러닝의 여러 접근 방법 중에 하나이지만, 지도학습은 실제로 사용되는 머신러닝의 대부분을 설명합니다. 부분적으로, 많은 중요한 작업들이 몇 가지 이용 가능한 증거가 주어졌을 때 알 수 없는 것의 확률을 추정하는 것으로 설명될 수 있기 때문입니다 :

* Predict cancer vs not cancer, given a CT image.
* Predict the correct translation in French, given a sentence in English.
* Predict the price of a stock next month based on this month's financial reporting data.
* CT 이미지를 보고 암 여부를 예측하기
* 영어 문장에 대한 정확한 프랑스어 번역을 예측하기
* 이번달의 제정 보고 데이터를 기반으로 다음달 주식 가격을 예측하기

Even with the simple description 'predict targets from inputs'
supervised learning can take a great many forms and require a great many modeling decisions,
depending on the type, size, and the number of inputs and outputs.
For example, we use different models to process sequences (like strings of text or time series data)
and for processing fixed-length vector representations.
We'll visit many of these problems in depth throughout the first 9 parts of this book.

'입력으로 부터 타겟을 예측한다'라고 간단하게 설명했지만, 지도학습은 입력과 출력의 테입, 크기 및 개수에 따라서 아주 다양한 형식이 있고 다양한 모델링 결정을 요구합니다. 예를 들면, 텍스트의 문자열 또는 시계열 데이터와 같은 시퀀스를 처리하는 것과 고정된 백처 표헌을 처리하는데 다른 모델을 사용합니다. 이책의 처음 9 파트에서 이런 문제들에 대해서 상세하게 다룹니다.

Put plainly, the learning process looks something like this.
Grab a big pile of example inputs, selecting them randomly.
Acquire the ground truth labels for each.
Together, these inputs and corresponding labels (the desired outputs)
comprise the training set.
We feed the training dataset into a supervised learning algorithm.
So here the *supervised learning algorithm* is a function that takes as input a dataset,
and outputs another function, *the learned model*.
Then, given a learned model,
we can take a new previously unseen input, and predict the corresponding label.

명백히 말하면, 학습 과정은 다음과 같습니다. 예제 입력을 많이 수집해서, 임의로 고릅니다. 각각에 대해서 ground truth를 얻습니다. 입력과 해당하는 레이블 (원하는 결과)를 합쳐서 학습 데이터를 구성합니다. 학습 데이터를 지도학습 알고리즘에 입력합니다. 여기서 *지도학습 알고리즘(supervised learning algorithm)* 은 데이터셋을 입력으로 받아서 어떤 함수(학습된 모델)를 결과로 내는 함수 입니다. 그렇게 얻어진 학습된 모델을 이용해서 이전에 보지 않은 새로운 입력에 대해서 해당하는 레이블을 예측합니다.

![](../img/supervised-learning.svg)



### Regression

Perhaps the simplest supervised learning task to wrap your head around is Regression.
Consider, for example a set of data harvested
from a database of home sales.
We might construct a table, where each row corresponds to a different house,
and each column corresponds to some relevant attribute,
such as the square footage of a house, the number of bedrooms, the number of bathrooms,
and the number of minutes (walking) to the center of town.
Formally, we call one row in this dataset a *feature vector*,
and the object (e.g. a house) it's associated with an *example*.

아마도 여러분의 머리에 떠오르는 가장 간단한 지도학습은 회귀(regression)일 것입니다. 주택 판매 데이터베이스에서 추출된 데이터를 예로 들어보겠습니다. 각 행은 하나의 집을, 각 열은 관련된 속성(집의 면적, 침실 개수, 화장실 개수, 도심으로 부터의 도보 거리 등)을 갖는 테이블을 만듭니다. 우리는 이 데이터셋의 하나의 행을 *속성백터(feature vector)* 라고 부르고, 이와 연관된 객체는 *예제(example)* 이라고 부릅니다.

If you live in New York or San Francisco, and you are not the CEO of Amazon, Google, Microsoft, or Facebook,
the (sq. footage, no. of bedrooms, no. of bathrooms, walking distance) feature vector for your home
might look something like: $[100, 0, .5, 60]​$.
However, if you live in Pittsburgh,
it might look more like $[3000, 4, 3, 10]​$.
Feature vectors like this are essential for all the classic machine learning problems.
We'll typically denote the feature vector for any one example $\mathbf{x_i}​$
and the set of feature vectors for all our examples $X​$.

만약 여러분이 뉴욕이나 샌프란시스코에서 살고, 아마존, 구글, 마이크소프트, 패이스북의 CEO가 아니라면, 여러분 집의 속성 백터(집 면적, 침실수, 화장실수, 도심까지 도보 거리)는 아마도 $[100, 0, .5, 60]$ 가 될 것입니다. 하지만, 피츠버그에 산다면  $[3000, 4, 3, 10]$ 와 가까울 것입니다. 이런 속성 백터는 모든 전통적인 머신러닝 문제에 필수적인 것입니다. 우리는 일반적으로 어떤 예제에 대한 속성 백터를 $\mathbf{x_i}$ 로 표기하고, 모든 예제에 대한 속성 백터의 집합은  $X$ 로 표기합니다.

What makes a problem *regression* is actually the outputs.
Say that you're in the market for a new home,
you might want to estimate the fair market value of a house,
given some features like these.
The target value, the price of sale, is a *real number*.
We denote any individual target $y_i​$ (corresponding to example $\mathbf{x_i}​$)
and the set of all targets $\mathbf{y}​$ (corresponding to all examples X).
When our targets take on arbitrary real values in some range,
we call this a regression problem.
The goal of our model is to produce predictions (guesses of the price, in our example)
that closely approximate the actual target values.
We denote these predictions $\hat{y}_i​$
and if the notation seems unfamiliar, then just ignore it for now.
We'll unpack it more thoroughly in the subsequent chapters.

결과에 따라서 어떤 문제가 *회귀(regression)*인지를 결정됩니다. 새집을 사기 위해서 부동산을 돌아다니고 있다고 하면, 여러분은 주어진 속성에 대해서 합당한 집 가격을 추정하기를 원합니다. 타겟 값, 판매 가격,은 *실제 숫자(real number)*가 됩니다. 샘플  $\mathbf{x_i}$에 대한하는 각 타겟은  $y_i$ 로 표시하고, 모든 예제 X 에 대한 모든 타겟들은  $\mathbf{y}$ 로 적습니다. 타겟이 어떤 범위에 속하는 임의의 실수값을 갖는 다면, 우리는 이를 회귀 문제라고 부릅니다. 우리의 모델의 목표는 실제 타겟 값을 근접하게 추정하는 예측 (이 경우에는 집가격 추측)을 생성하는 것입니다. 이 예측을 $\hat{y}_i$ 로 표기합니다. 만약 표기법이 익숙하지 않다면, 다음 장들에서 더 자세히 설명할 것이기 때문에 지금은 그냥 무시해도 됩니다.  

Lots of practical problems are well-described regression problems.
Predicting the rating that a user will assign to a movie is a regression problem,
and if you designed a great algorithm to accomplish this feat in 2009,
you might have won the [$1 million Netflix prize](https://en.wikipedia.org/wiki/Netflix_Prize).
Predicting the length of stay for patients in the hospital is also a regression problem.
A good rule of thumb is that any *How much?* or *How many?* problem should suggest regression.

많은 실질적인 문제들이 잘 정의된 회귀 문제들입니다. 관객이 영화에 줄 평점을 예측하는 것은 회귀의 문제인데, 여러분이 2009년에 이를 잘 예측하는 대단한 알고리즘을 디자인했다면  [$1 million Netflix prize](https://en.wikipedia.org/wiki/Netflix_Prize) 를 받았을 것입니다. 환자가 입원일 수를 예측하는 것 또한 회귀 문제입니다. 문제가 회귀의 문제인지를 판단하는 좋은 경험의 법칙은 *얼마나 만큼*  또는 *얼마나 많이* 로 대답이되는지 보는 것입니다.

* 'How many hours will this surgery take?' - *regression*
* 'How many dogs are in this photo?' - *regression*.
* 이 수술은 몇 시간이 걸릴까요? - *회귀*
* 이 사진에 개가 몇 마리 있나요? - *회귀*

However, if you can easily pose your problem as 'Is this a _ ?',
then it's likely, classification, a different fundamental problem type that we'll cover next.
Even if you've never worked with machine learning before,
you've probably worked through a regression problem informally.
Imagine, for example, that you had your drains repaired
and that your contractor spent $x_1=3​$ hours removing gunk from your sewage pipes.
Then she sent you a bill of $y_1 = \$350​$.
Now imagine that your friend hired the same contractor for $x_2 = 2​$ hours
and that she received a bill of $y_2 = \$250​$.
If someone then asked you how much to expect on their upcoming gunk-removal invoice
you might make some reasonable assumptions,
such as more hours worked costs more dollars.
You might also assume that there's some base charge and that the contractor then charges per hour.
If these assumptions held, then given these two data points,
you could already identify the contractor's pricing structure:
\$100 per hour plus \$50 to show up at your house.
If you followed that much then you already understand the high-level idea behind linear regression (and you just implicitly designed a linear model with bias).

그런데 만약 주어진 문제에 대한 질문을 '이것은 ... 인가요?' 라고 쉽게 바꿀 수 있다면, 분류의 문제입니다. 이는 다른 기본적인 문제 유형입니다. 머신러닝을 이전에 다뤄보지 않은 경우에도 비공식적으로는 회귀의 문제들을 다뤄왔습니다. 예를 들어, 여러분의 집의 배수구를 수리하고, 수리공이  $x_1=3$ 시간이 걸려서 하수관에서 덩어리를 제거했습니다. 이에 대해서 수리공은 $y_1 = \$350$ 청구를 합니다. 여러분의 친구가 같은 수리공을 공용해서 or $x_2 = 2$ 시간 걸려서 일하고,  $y_2 = \$250$ 를 청구했습니다. 어떤 사람이 하수관에서 덩어리를 제거하는 데 비용이 얼마가 될지를 물어보면, 여러분은 논리적인 추정 - 시간이 더 소요되면 더 비싸다 -을 할 것입니다. 기본 비용이 있고, 시간당 비용이 있을 것이라고까지 추정할 것입니다. 이 가정이 맞다면, 위 두 데이터 포인트를 활용해서 수리공의 가격 구조를 알아낼 수 있습니다: 시간당 100달러 및 기본 비용 50달러. 여러분이 여기까지 잘 따라왔다면 선형 회귀에 대한 고차원의 아이디어를 이미 이해한 것입니다. (선형모델을 bias를 사용해서 디자인했습니다.)

In this case, we could produce the parameters that exactly matched the contractor's prices.
Sometimes that's not possible, e.g., if some of the variance owes to some factors besides your two features.
In these cases, we'll try to learn models that minimize the distance between our predictions and the observed values.
In most of our chapters, we'll focus on one of two very common losses,
the
[L1 loss](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L1Loss)
where

위 예에서는 수리공의 가격을 정확하게 계산하는 파라메터를 찾아낼 수 있었습니다. 때로는 불가능한데, 예를 들면 만약 어떤 차이가 이 두 피쳐외에 작용하는 경우가 그렇습니다. 그런 경우에는 우리는 우리의 예측과 관찰된 값의 차이를 최소화하는 모델을 학습시키고자 노력합니다. 대부분 장들에서 우리는 아주 일반적인 loss 둘 중에 하나에 집중할 것입니다. 하나는 [L1 loss](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L1Loss) 로, 다음과 같고, 

$$l(y,y') = \sum_i |y_i-y_i'|$$

and the least mean squares loss, aka

[L2 loss](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L2Loss)
where

다른 하나는 최소 평균 제곱 손실(least mean square loss), 즉 [L2 loss](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L2Loss) 입니다. 이는 다음과 같이 표기 됩니다.

$$l(y,y') = \sum_i (y_i - y_i')^2.$$

As we will see later, the $L_2$ loss corresponds to the assumption that our data was corrupted by Gaussian noise, whereas the $L_1$ loss corresponds to an assumption of noise from a Laplace distribution.

나중에 보겠지만, $L_2$ loss는 우리의 데이터가 가우시안 노이즈에 영향을 받았다고 가정에 관련이 되고, $L_1$ loss는 라플라스 분포(Laplace distribution)의 노이즈를 가정합니다.

### Classification

While regression models are great for addressing *how many?* questions,
lots of problems don't bend comfortably to this template. For example,
a bank wants to add check scanning to their mobile app.
This would involve the customer snapping a photo of a check with their smartphone's camera
and the machine learning model would need to be able to automatically understand text seen in the image.
It would also need to understand hand-written text to be even more robust.
This kind of system is referred to as optical character recognition (OCR),
and the kind of problem it solves is called a classification.
It's treated with a distinct set of algorithms than those that are used for regression.

회귀 모델은 *얼마나 많이* 라는 질문에 답을 주는데는 훌륭하지만, 많은 문제들이 이 템플렛에 잘 들어맞지 않습니다. 예를 들면, 은행이 모바일앱에 수표 스캐닝 기능을 추가하고자 합니다. 이를 위해서 고객은 스마트폰의 카메라로 수표를 찍으면, 이미지에 있는 텍스트를 자동으로 이해하는 기능을 하는 머신러닝 모델이 필요합니다. 손으로 쓴 글씨에 더 잘 동작을 해야할 필요가 있습니다. 이런 시스템은 문자인식(OCR, optical character recognition)이라고 하고, 이것이 풀려는 문제의 종류를 분류라고 합니다. 회귀 문제에 사용되는 알고리즘과는 아주 다른 알고리즘이 이용됩니다.

In classification, we want to look at a feature vector, like the pixel values in an image,
and then predict which category (formally called *classes*),
among some set of options, an example belongs.
For hand-written digits, we might have 10 classes,
corresponding to the digits 0 through 9.
The simplest form of classification is when there are only two classes,
a problem which we call binary classification.
For example, our dataset $X​$ could consist of images of animals
and our *labels* $Y​$ might be the classes $\mathrm{\{cat, dog\}}​$.
While in regression, we sought a regressor to output a real value $\hat{y}​$,
in classification, we seek a *classifier*, whose output $\hat{y}​$ is the predicted class assignment.

분류는 이미지의 픽셀값과 같은 속석 백터를 보고, 그 예제가 주어진 종류들 중에서 어떤 카테고리에 속하는지를 예측합니다. 손으로 쓴 숫자의 경우에는 숫자 0부터 9까지 10개의 클래스가 있습니다. 가장 간단한 분류의 형태는 단 두개의 클래스가 있는 경우로, 이를 이진 분류(binary classificatio)이라고 부릅니다. 예를 들어, 데이터셋 $X$ 가 동물들의 사진이고, 이에 대한 *레이블*  $Y$ 이 {고양이, 강아지}인 경우를 들 수 있습니다. 회귀에서는 결과가 실수 값 $\hat{y}$ 가 되지만, 분류에서는 결과가 예측된 클래스인 *분류기* 를 만들고자 합니다.

For reasons that we'll get into as the book gets more technical, it's pretty hard to optimize a model that can only output a hard categorical assignment, e.g. either *cat* or *dog*.
It's a lot easier instead to express the model in the language of probabilities.
Given an example $x$, the model assigns a probability $\hat{y}_k$ to each label $k$.
Because these are probabilities, they need to be positive numbers and add up to $1$.
This means that we only need $K-1$ numbers to give the probabilities of $K$ categories.
This is easy to see for binary classification.
If there's a 0.6 (60%) probability that an unfair coin comes up heads,
then there's a 0.4 (40%) probability that it comes up tails.
Returning to our animal classification example, a classifier might see an image
and output the probability that the image is a cat $\Pr(y=\mathrm{cat}| x) = 0.9​$.
We can interpret this number by saying that the classifier is 90% sure that the image depicts a cat.
The magnitude of the probability for the predicted class is one notion of confidence.
It's not the only notion of confidence and we'll discuss different notions of uncertainty in more advanced chapters.

이 책에서 더 기술적인 내용을 다룰 때, 고정된 카테고리 - 예를 들면 고양이 또는 개 -에 대한 결과만을 예측하는 모델을 최적화하는 것은 어려워질 것입니다. 대신 확률에 기반한 모델로 표현하는 것이 훨씬 더 쉽습니다. 즉, 예제 $x$ 가 주어졌을 때, 모델은 각 레이블 $k$ 에 확률  $\hat{y}_k$ 를 할당하는 것입니다. 결과가 확률값이기 때문에 모두 양수이고, 합은 1이됩니다. 이는 $K$ 개의 카테고리에 대한 확률을 구하기 위해서는 $K-1$ 개의 숫자만 필요하다는 것을 의미합니다. 이진 분류를 예로 들어보겠습니다. 공정하지 않은 동전을 던져서 앞면이 나올 확률이 0.6 (60%)라면, 뒷면이 나올 확률은 0.4 (40%)다 됩니다. 동물 분류의 예로 돌아가보면, 분류기는 이미지를 보고 이미지가 고양이일 확률 $\Pr(y=\mathrm{cat}| x) = 0.9$ 을 출력합니다. 우리는 이 숫자를 이미지가 고양이를 포할 것이라고 90% 정도 확신한다라고 해석할 수 있습니다. 예측된 클래스에 대한 확률의 정도는 신뢰에 대한 개념을 나타냅니다. 신뢰의 개념일 뿐만 아니라, 고급 내용을 다루는 장에서는 여러 비신뢰의 개념도 논의하겠습니다.

When we have more than two possible classes, we call the problem *multiclass classification*.
Common examples include hand-written character recognition `[0, 1, 2, 3 ... 9, a, b, c, ...]`.
While we attacked regression problems by trying to minimize the L1 or L2 loss functions,
the common loss function for classification problems is called cross-entropy.
In MXNet Gluon, the corresponding loss function can be found [here](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SoftmaxCrossEntropyLoss).

두개보다 많은 클래스가 있을 경우에 우리는 이 문제를 *다중클래스 분류(multiclass classification)* 이라고 합니다. 흔한 예로는 손으로 쓴 글씨 -  `[0, 1, 2, 3 ... 9, a, b, c, ...]` - 를 인식하는 예제가 있습니다. 우리는 회귀 문제를 풀 때 L1 또는 L2 loss 함수를 최소화하는 시도를 했는데, 분류 문제에서 cross-entropy 함수가 흔히 사용되는 loss 함수는 입니다. MXNet Gluon에서는 관련된 loss 함수에 대한 내용을 [여기](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SoftmaxCrossEntropyLoss)에서 볼 수 있습니다.

Note that the most likely class is not necessarily the one that you're going to use for your decision. Assume that you find this beautiful mushroom in your backyard:

가장 그럴듯한 클래스가 결정을 위해서 사용하는 것이 꼭 아닐 수도 있습니다. 여러분의 뒷뜰에서 이 아름다운 버섯을 찾는다고 가정해보겠습니다.

|![](../img/death_cap.jpg)|
|:-------:|
|Death cap - do not eat!|

Now, assume that you built a classifier and trained it
to predict if a mushroom is poisonous based on a photograph.
Say our poison-detection classifier outputs $\Pr(y=\mathrm{death cap}|\mathrm{image}) = 0.2​$.
In other words, the classifier is 80% confident that our mushroom *is not* a death cap.
Still, you'd have to be a fool to eat it.
That's because the certain benefit of a delicious dinner isn't worth a 20% risk of dying from it.
In other words, the effect of the *uncertain risk* by far outweighs the benefit.
Let's look at this in math. Basically, we need to compute the expected risk that we incur, i.e. we need to multiply the probability of the outcome with the benefit (or harm) associated with it:

자, 사진이 주어졌을 때 버섯이 독이 있는 것인지를 예측하는 분류기를 만들어서 학습했다고 가정합니다. 우리의 독버섯 탐기 분류기의 결과가 $\Pr(y=\mathrm{death cap}|\mathrm{image}) = 0.2$ 로 나왔습니다. 다르게 말하면, 이 분류기는 80% 확신을 갖고 이 버섯이 알광대버섯(death cap)이 *아니다*라고 말하고 있습니다. 하지만, 이것을 먹지는 않을 것입니다. 이 버섯으로 만들어질 멋진 저녁식사의 가치가 독버섯을 먹고 죽을 20%의 위험보다 가치가 없기 때문입니다. 이것을 수학적으로 살펴보겠습니다. 기본적으로 우리는 예상된 위험을 계산해야합니다. 즉, 결과에 대한 확률에 그 결과에 대한 이익 (또는 손해)를 곱합니다.

$$L(\mathrm{action}| x) = \mathbf{E}_{y \sim p(y| x)}[\mathrm{loss}(\mathrm{action},y)]$$

Hence, the loss $L​$ incurred by eating the mushroom is $L(a=\mathrm{eat}| x) = 0.2 * \infty + 0.8 * 0 = \infty​$, whereas the cost of discarding it is $L(a=\mathrm{discard}| x) = 0.2 * 0 + 0.8 * 1 = 0.8​$.

따라서 버섯을 먹을 경우 우리가 얻는 loss $L$ 은  $L(a=\mathrm{eat}| x) = 0.2 * \infty + 0.8 * 0 = \infty$ 인 반면에, 먹지 않을 경우 cost 또는 loss는  $L(a=\mathrm{discard}| x) = 0.2 * 0 + 0.8 * 1 = 0.8$ 이 됩니다.

Our caution was justified: as any mycologist would tell us, the above actually *is* a death cap.
Classification can get much more complicated than just binary, multiclass, of even multi-label classification.
For instance, there are some variants of classification for addressing hierarchies.
Hierarchies assume that there exist some relationships among the many classes.
So not all errors are equal - we prefer to misclassify to a related class than to a distant class.
Usually, this is referred to as *hierarchical classification*.
One early example is due to [Linnaeus](https://en.wikipedia.org/wiki/Carl_Linnaeus),
who organized the animals in a hierarchy.

우리의 주의 깊음이 올았습니다. 균학자들은 위 버섯이 실제로 독버섯인 알광대버섯이라고 알려줄 것이기 때문입니다. 분류 문제는 이진 분류보다 복잡해질 수 있습니다. 즉, 다중클래스 분류 문제이거나 더 나아가서는 다중 레이블 분류의 문제일 수 있습니다.  예를 들면, 계층을 푸는 분류의 종류들이 있습니다. 계층은 많은 클래스들 사이에 관계가 있는 것을 가정합니다. 따라서, 모든 오류가 동일하지 않습니다. 즉, 너무 다른 클래스로 예약하는 것보다는 관련된 클래스로 예측하는 것을 더 선호합니다. 이런 문제를 *계층적 분류(hierarchical classification)* 이라고 합니다. 계층적 분류의 오랜 예는 [Linnaeus](https://en.wikipedia.org/wiki/Carl_Linnaeus) 가 동물을 계층으로 분류한 것을 들수 있습니다.

![](../img/sharks.png)

In the case of animal classification, it might not be so bad to mistake a poodle for a schnauzer,
but our model would pay a huge penalty if it confused a poodle for a dinosaur.
What hierarchy is relevant might depend on how you plan to use the model.
For example, rattle snakes and garter snakes might be close on the phylogenetic tree,
but mistaking a rattler for a garter could be deadly.

동물 분류의 경우 푸들을 슈나이저라고 실수로 분류하는 것이 그렇게 나쁘지 않을 수 있지만, 푸들을 공룡이라고 분류한다면 그 영향이 클 수도 있습니다. 어떤 계층이 적절할지는 여러분이 모델을 어떻게 사용할 것인지에 달려있습니다. 예를 들면, 딸랑이 뱀(rattle snake)와 가터스 뱀(garter snake)은 계통 트리에서는 가까울 수 있지만, 딸랑이 뱀를 가터스 뱀으로 잘못 분류한 결과는 치명적일 수 있기때문입니다.

### Tagging

Some classification problems don't fit neatly into the binary or multiclass classification setups.
For example, we could train a normal binary classifier to distinguish cats from dogs.
Given the current state of computer vision,
we can do this easily, with off-the-shelf tools.
Nonetheless, no matter how accurate our model gets, we might find ourselves in trouble when the classifier encounters an image of the Bremen Town Musicians.

어떤 분류의 문제는 이진 또는 다중 클래스 분류 형태로 딱 떨어지지 않습니다. 예들 들자면,  고양이와 강아지를 구분하는 정상적인 이진 분류기를 학습시킬 수 있습니다. 현재의 컴퓨터 비전의 상태를 고려하면, 이는 상용도구을 이용해서도 아주 쉽게 할 수 있습니다. 그럼에도 불구하고, 우리의 모델이 얼마나 정확하든지 상관없이 브레맨 음악대의 사진지 주어진다면 문제가 발생할 수도 있습니다.

![](../img/stackedanimals.jpg)

As you can see, there's a cat in the picture, and a rooster, a dog and a donkey, with some trees in the background.
Depending on what we want to do with our model ultimately,
treating this as a binary classification problem
might not make a lot of sense.
Instead, we might want to give the model the option
of saying the image depicts a cat *and* a dog *and* a donkey *and* a rooster.

사진에는 고양이, 수닭, 강아지, 당나귀 그리고 배경에는 나무들이 있습니다. 우리의 모델을 이용해서 주로 무엇을 할 것인지에 따라서, 이 문제를 이진 분류의 문제로 다룰 경우 소용이 없어질 수 있습니다. 대신, 우리는 모델이 이미지에 고양이, 강아지, 당나귀 그리고 수닭이 있는 것을 알려주도록 하고 싶을 것입니다.

The problem of learning to predict classes
that are *not mutually exclusive*
is called multi-label classification.
Auto-tagging problems are typically best described
as multi-label classification problems.
Think of the tags people might apply to posts on a tech blog,
e.g., 'machine learning', 'technology', 'gadgets',
'programming languages', 'linux', 'cloud computing', 'AWS'.
A typical article might have 5-10 tags applied
because these concepts are correlated.
Posts about 'cloud computing' are likely to mention 'AWS'
and posts about 'machine learning' could also deal with 'programming languages'.

*서로 배타적이 아닌(not mutually exclusive)* 아닌 클래스들을 예측하는 문제를 멀티-레이블 분류라고 합니다. 자동 태깅 문제가 전형적인 멀티 레이블 분류 문제입니다. 태그의 예는 기술 문서에 붙이는 태그 - 즉, '머신러닝', '기술', '가젯', '프로그램언어', '리눅스', 클라우드 컴퓨팅', 'AWS' - 를 생각해봅시다. 일반적으로 기사는 5-10개 태그를 갖는데, 그 이유는 테그들이 서로 관련이 있기 때문입니다. '클라우드 컴퓨팅'에 대한 글은 'AWS'를 언급할 가능성이 높고, '머신러닝' 관련 글은 '프로그램 언어'와 관련된 것일 수 있습니다.

We also have to deal with this kind of problem when dealing with the biomedical literature,
where correctly tagging articles is important
because it allows researchers to do exhaustive reviews of the literature.
At the National Library of Medicine, a number of professional annotators
go over each article that gets indexed in PubMed
to associate each with the relevant terms from MeSH,
a collection of roughly 28k tags.
This is a time-consuming process and the annotators typically have a one year lag between archiving and tagging. Machine learning can be used here to provide provisional tags
until each article can have a proper manual review.
Indeed, for several years, the BioASQ organization has [hosted a competition](http://bioasq.org/)
to do precisely this.

우리는 연구자들이 리뷰를 많이 할 수 있도록 하기 위해서 올바른 태그를 다는 것이 중요한 생물 의학 문헌을 다룰 때 이런 문제를 다뤄야합니다. 의학 국립 도서관에는 많은 전문 주석자들이 PubMed에 색인된 아티클들을 하나씩 보면서 MeSH (약 28,000개 태그의 집합) 중에 관련 된 태그를 연관시키는 일을 하고 있습니다. 이것은 시간이 많이 소모되는 일로서, 주석자들이 태그를 다는데는 보통 1년이 걸립니다. 머신러닝을 사용해서 임시 태그를 달고, 이후에 매뉴얼 리뷰를 하는 것이 가능합니다. 실제로 몇년에 동안 [BioASQ](http://bioasq.org/) 에서는 이에 대한 대회를 열었었습니다.


### Search and ranking

Sometimes we don't just want to assign each example to a bucket or to a real value. In the field of information retrieval, we want to impose a ranking on a set of items. Take web search for example, the goal is less to determine whether a particular page is relevant for a query, but rather, which one of the plethora of search results should be displayed for the user. We really care about the ordering of the relevant search results and our learning algorithm needs to produce ordered subsets of elements from a larger set. In other words, if we are asked to produce the first 5 letters from the alphabet, there is a difference between returning ``A B C D E`` and ``C A B E D``. Even if the result set is the same, the ordering within the set matters nonetheless.

떄로는 각 예제들에 대해서 어떤 클래스 또는 실제 값을 할당하는 것만을 원하지 않습니다. 정보 검색 분야의 경우에는 아이템 집합에 순위를 매기고 싶어합니다. 웹 검색을 예로 들어보면, 목표는 특정 페이지가 쿼리에 관련이 있는지 여부를 판단하는 것보다는 검색 결과들 중에 어떤 것이 사용자에게 먼저 보여줘야하는 것에 있습니다. 관련 검색 결과의 순서에 대해서 관심이 많고, 우리의 러닝 알고리즘은 큰 집합의 일부에 대한 순서를 매길 수 있어야합니다. 즉, 알파벳에서 처음 5개 글자가 무엇인지를 물어봤을 경우,  ``A B C D E`` 를 결과로 주는 것과  ``C A B E D`` 를 결과로 주는 것에는 차이가 있습니다. 결과 집합은 같은 경우라도,  집합안에서 순서도 중요합니다.

One possible solution to this problem is to score every element in the set of possible sets along with a corresponding relevance score and then to retrieve the top-rated elements. [PageRank](https://en.wikipedia.org/wiki/PageRank) is an early example of such a relevance score. One of the peculiarities is that it didn't depend on the actual query. Instead, it simply helped to order the results that contained the query terms. Nowadays search engines use machine learning and behavioral models to obtain query-dependent relevance scores. There are entire conferences devoted to this subject.

이 문제에 대한 가능한 해결방법은 가능한 집합의 원소들에 관련성 점수를 부여하고, 점수가 높은 항목들을 검색하는 것입니다[PageRank](https://en.wikipedia.org/wiki/PageRank) 가 관련성 점수를 적용한 예로, 특성 중 하나는 이것은 실제 쿼리에 의존하지 않는다는 것입니다. 대신, 쿼리 단어들을 포함한 결과들을 순서를 부여하는 것을 합니다. 요즘의 검색 엔진은 머신러닝과 행동 모델을 이용해서 쿼리와 관련된 관련성 점수를 얻습니다. 이 주제만 다루는 컨퍼런스가 있습니다.

<!-- Add / clean up-->

### Recommender systems

Recommender systems are another problem setting that is related to search and ranking. The problems are  similar insofar as the goal is to display a set of relevant items to the user. The main difference is the emphasis on *personalization* to specific users in the context of recommender systems. For instance, for movie recommendations, the results page for a SciFi fan and the results page for a connoisseur of Woody Allen comedies might differ significantly.

추천 시스템은 검색과 랭킹과 관련된 또다른 문제 세팅입니다. 사용자에게 관련된 상품을 보여주는 것이 목표이기에 문제는 비스합니다. 주요 차이점은 추천 시스템에서는 특정 사용자에 대한 *개인화(personalization)* 를 중점으로 한다는 것입니다. 예를 들어, 영화 추천의 경우에는 SciFi 에 대한 결과 페이지와 우디 엘런 코미디에 대한 결과 페이지가 아주 다르게 나옵니다.

Such problems occur, e.g. for movie, product or music recommendation. In some cases, customers will provide explicit details about how much they liked the product (e.g. Amazon product reviews). In some other cases, they might simply provide feedback if they are dissatisfied with the result (skipping titles on a playlist). Generally, such systems strive to estimate some score $y_{ij}$, such as an estimated rating or probability of purchase, given a user $u_i$ and product $p_j$.

이런 문제는 영화, 제품 또는 음악 추천에서 발생합니다. 어떤 경우에는 고객은 얼마나 그 제품을 좋아하는지를 직접 알려주기도 합니다 (예를 들면 아마존의 제품 리뷰). 어떤 경우에는 결과에 만족하지 못한 경우 피드백을 간단하게 주기도 합니다 (재생 목록의 타이틀을 건너띄는 형식으로). 일반적으로는 이런 시스템은 어떤 점수 $y_{ij}$ 를 예측하고자 하는데, 이 예측은 사용자 $u_i$ 와 제품  $p_j$가 주어졌을 때 예상된 평점 또는 구매 확률이 될 수 있습니다.

Given such a model, then for any given user, we could retrieve the set of objects with the largest scores $y_{ij}​$, which are then used as a recommendation. Production systems are considerably more advanced and take detailed user activity and item characteristics into account when computing such scores. The following image is an example of deep learning books recommended by Amazon based on personalization algorithms tuned to the author's preferences.

이런 모델은 어떤 사용자에 대해서 가장 큰 점수  $y_{ij}$ 를 갖는 객체들의 집합을 찾아주는데, 이것이 추천으로 사용됩니다. 운영 시스템은 매우 복잡하고, 점수를 계산할 때 자세한 사용자의 활동과 상품의 특징까지 고려합니다. 아래 이미지는 아마존이 저자들의 관심을 반영한 개인화 알고리즘을 기반으로 아마존이 추천한 딥러닝 책들의 예입니다.

![](../img/deeplearning_amazon.png)


### Sequence Learning

So far we've looked at problems where we have some fixed number of inputs
and produce a fixed number of outputs.
Before we considered predicting home prices from a fixed set of features:
square footage, number of bedrooms, number of bathrooms, walking time to downtown.
We also discussed mapping from an image (of fixed dimension),
to the predicted probabilities that it belongs to each of a fixed number of classes,
or taking a user ID and a product ID, and predicting a star rating.
In these cases, once we feed our fixed-length input into the model to generate an output,
the model immediately forgets what it just saw.

This might be fine if our inputs truly all have the same dimensions
and if successive inputs truly have nothing to do with each other.
But how would we deal with video snippets?
In this case, each snippet might consist of a different number of frames.
And our guess of what's going on in each frame
might be much stronger if we take into account
the previous or succeeding frames.
Same goes for language.
One popular deep learning problem is machine translation:
the task of ingesting sentences in some source language
and predicting their translation in another language.

These problems also occur in medicine.
We might want a model to monitor patients in the intensive care unit and to fire off alerts
if their risk of death in the next 24 hours exceeds some threshold.
We definitely wouldn't want this model to throw away everything it knows about the patient history each hour,
and just make its predictions based on the most recent measurements.

These problems are among the more exciting applications of machine learning
and they are instances of *sequence learning*.
They require a model to either ingest sequences of inputs
or to emit sequences of outputs (or both!).
These latter problems are sometimes referred to as ``seq2seq`` problems.
Language translation is a ``seq2seq`` problem.
Transcribing text from spoken speech is also a ``seq2seq`` problem.
While it is impossible to consider all types of sequence transformations,
a number of special cases are worth mentioning:

#### Tagging and Parsing

This involves annotating a text sequence with attributes. In other words, the number of inputs and outputs is essentially the same. For instance, we might want to know where the verbs and subjects are. Alternatively, we might want to know which words are the named entities. In general, the goal is to decompose and annotate text based on structural and grammatical assumptions to get some annotation. This sounds more complex than it actually is. Below is a very simple example of annotating a sentence with tags indicating which words refer to named entities.

|Tom | has | dinner | in | Washington | with | Sally.|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Ent | - | - | - | Ent | - | Ent|


#### Automatic Speech Recognition

With speech recognition, the input sequence $x$ is the sound of a speaker,
and the output $y$ is the textual transcript of what the speaker said.
The challenge is that there are many more audio frames (sound is typically sampled at 8kHz or 16kHz) than text, i.e. there is no 1:1 correspondence between audio and text,
since thousands of samples correspond to a single spoken word.
These are seq2seq problems where the output is much shorter than the input.

|`-D-e-e-p- L-ea-r-ni-ng-`|
|:--------------:|
|![Deep Learning](../img/speech.png)|

#### Text to Speech

Text to Speech (TTS) is the inverse of speech recognition.
In other words, the input $x$ is text
and the output $y$ is an audio file.
In this case, the output is *much longer* than the input.
While it is easy for *humans* to recognize a bad audio file,
this isn't quite so trivial for computers.

#### Machine Translation

Unlike the case of speech recognition, where corresponding inputs and outputs occur in the same order (after alignment),
in machine translation, order inversion can be vital.
In other words, while we are still converting one sequence into another,
neither the number of inputs and outputs
nor the order of corresponding data points
are assumed to be the same.
Consider the following illustrative example of the obnoxious tendency of Germans
(*Alex writing here*)
to place the verbs at the end of sentences.

|German |Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?|
|:------|:---------|
|English|Did you already check out this excellent tutorial?|
|Wrong alignment |Did you yourself already this excellent tutorial looked-at?|

A number of related problems exist.
For instance, determining the order in which a user reads a webpage
is a two-dimensional layout analysis problem.
Likewise, for dialogue problems,
we need to take world-knowledge and prior state into account.
This is an active area of research.


## Unsupervised learning

All the examples so far were related to *Supervised Learning*,
i.e. situations where we feed the model
a bunch of examples and a bunch of *corresponding target values*.
You could think of supervised learning as having an extremely specialized job and an extremely anal boss.
The boss stands over your shoulder and tells you exactly what to do in every situation until you learn to map from situations to actions.
Working for such a boss sounds pretty lame.
On the other hand, it's easy to please this boss. You just recognize the pattern as quickly as possible and imitate their actions.

In a completely opposite way,
it could be frustrating to work for a boss
who has no idea what they want you to do.
However, if you plan to be a data scientist, you had better get used to it.
The boss might just hand you a giant dump of data and tell you to *do some data science with it!*
This sounds vague because it is.
We call this class of problems *unsupervised learning*,
and the type and number of questions we could ask
is limited only by our creativity.
We will address a number of unsupervised learning techniques in later chapters. To whet your appetite for now, we describe a few of the questions you might ask:

* Can we find a small number of prototypes that accurately summarize the data? Given a set of photos, can we group them into landscape photos, pictures of dogs, babies, cats, mountain peaks, etc.? Likewise, given a collection of users' browsing activity, can we group them into users with similar behavior? This problem is typically known as **clustering**.
* Can we find a small number of parameters that accurately capture the relevant properties of the data? The trajectories of a ball are quite well described by velocity, diameter, and mass of the ball. Tailors have developed a small number of parameters that describe human body shape fairly accurately for the purpose of fitting clothes. These problems are referred to as **subspace estimation** problems. If the dependence is linear, it is called **principal component analysis**.
* Is there a representation of (arbitrarily structured) objects in Euclidean space (i.e. the space of vectors in $\mathbb{R}^n$) such that symbolic properties can be well matched? This is called **representation learning** and it is used to describe entities and their relations, such as Rome - Italy + France = Paris.
* Is there a description of the root causes of much of the data that we observe? For instance, if we have demographic data about house prices, pollution, crime, location, education, salaries, etc., can we discover how they are related simply based on empirical data? The field of **directed graphical models** and **causality** deals with this.
* An important and exciting recent development is **generative adversarial networks**. They are basically a procedural way of synthesizing data. The underlying statistical mechanisms are tests to check whether real and fake data are the same. We will devote a few notebooks to them.


## Interacting with an environment

So far, we haven't discussed where data actually comes from,
or what actually *happens* when a machine learning model generates an output.
That's because supervised learning and unsupervised learning
do not address these issues in a very sophisticated way.
In either case, we grab a big pile of data up front,
then do our pattern recognition without ever interacting with the environment again.
Because all of the learning takes place after the algorithm is disconnected from the environment,
this is called *offline learning*.
For supervised learning, the process looks like this:

![](../img/data-collection.svg)


This simplicity of offline learning has its charms.
The upside is we can worry about pattern recognition in isolation without these other problems to deal with,
but the downside is that the problem formulation is quite limiting.
If you are more ambitious, or if you grew up reading Asimov's Robot Series,
then you might imagine artificially intelligent bots capable not only of making predictions,
but of taking actions in the world.
We want to think about intelligent *agents*, not just predictive *models*.
That means we need to think about choosing *actions*, not just making *predictions*.
Moreover, unlike predictions, actions actually impact the environment.
If we want to train an intelligent agent,
we must account for the way its actions might
impact the future observations of the agent.


Considering the interaction with an environment opens a whole set of new modeling questions. Does the environment:

* remember what we did previously?
* want to help us, e.g. a user reading text into a speech recognizer?
* want to beat us, i.e. an adversarial setting like spam filtering (against spammers) or playing a game (vs an opponent)?
* not  care (as in most cases)?
* have shifting dynamics (steady vs shifting over time)?

This last question raises the problem of *covariate shift*,
(when training and test data are different).
It's a problem that most of us have experienced when taking exams written by a lecturer,
while the homeworks were composed by his TAs.
We'll briefly describe reinforcement learning, and adversarial learning,
two settings that explicitly consider interaction with an environment.


### Reinforcement learning

If you're interested in using machine learning to develop an agent that interacts with an environment and takes actions, then you're probably going to wind up focusing on *reinforcement learning* (RL).
This might include applications to robotics, to dialogue systems,
and even to developing AI for video games.
*Deep reinforcement learning* (DRL), which applies deep neural networks
to RL problems, has surged in popularity.
The breakthrough [deep Q-network that beat humans at Atari games using only the visual input](https://www.wired.com/2015/02/google-ai-plays-atari-like-pros/) ,
and the [AlphaGo program that dethroned the world champion at the board game Go](https://www.wired.com/2017/05/googles-alphago-trounces-humans-also-gives-boost/) are two prominent examples.

Reinforcement learning gives a very general statement of a problem,
in which an agent interacts with an environment over a series of *time steps*.
At each time step $t$, the agent receives some observation $o_t$ from the environment,
and must choose an action $a_t$ which is then transmitted back to the environment.
Finally, the agent receives a reward $r_t$ from the environment.
The agent then receives a subsequent observation, and chooses a subsequent action, and so on.
The behavior of an RL agent is governed by a *policy*.
In short, a *policy* is just a function that maps from observations (of the environment) to actions.
The goal of reinforcement learning is to produce a good policy.

![](../img/rl-environment.svg)

It's hard to overstate the generality of the RL framework.
For example, we can cast any supervised learning problem as an RL problem.
Say we had a classification problem.
We could create an RL agent with one *action* corresponding to each class.
We could then create an environment which gave a reward
that was exactly equal to the loss function from the original supervised problem.

That being said, RL can also address many problems that supervised learning cannot.
For example, in supervised learning we always expect
that the training input comes associated with the correct label.
But in RL, we don't assume that for each observation,
the environment tells us the optimal action.
In general, we just get some reward.
Moreover, the environment may not even tell us which actions led to the reward.

Consider for example the game of chess.
The only real reward signal comes at the end of the game when we either win, which we might assign a reward of 1,
or when we lose, which we could assign a reward of -1.
So reinforcement learners must deal with the *credit assignment problem*.
The same goes for an employee who gets a promotion on October 11.
That promotion likely reflects a large number of well-chosen actions over the previous year.
Getting more promotions in the future requires figuring out what actions along the way led to the promotion.

Reinforcement learners may also have to deal with the problem of partial observability.
That is, the current observation might not tell you everything about your current state.
Say a cleaning robot found itself trapped in one of many identical closets in a house.
Inferring the precise location (and thus state) of the robot
might require considering its previous observations before entering the closet.

Finally, at any given point, reinforcement learners might know of one good policy,
but there might be many other better policies that the agent has never tried.
The reinforcement learner must constantly choose
whether to *exploit* the best currently-known strategy as a policy,
or to *explore* the space of strategies,
potentially giving up some short-run reward in exchange for knowledge.


### MDPs, bandits, and friends

The general reinforcement learning problem
is a very general setting.
Actions affect subsequent observations.
Rewards are only observed corresponding to the chosen actions.
The environment may be either fully or partially observed.
Accounting for all this complexity at once may ask too much of researchers.
Moreover not every practical problem exhibits all this complexity.
As a result, researchers have studied a number of *special cases* of reinforcement learning problems.

When the environment is fully observed, we call the RL problem a *Markov Decision Process* (MDP).
When the state does not depend on the previous actions,
we call the problem a *contextual bandit problem*.
When there is no state, just a set of available actions with initially unknown rewards,
this problem is the classic *multi-armed bandit problem*.

## Summary

Machine Learning is vast. We cannot possibly cover it all. On the other hand, neural networks are simple and only require elementary mathematics. So let's get started (but first, let's install MXNet).

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2314)

![](../img/qr_introduction.svg)
