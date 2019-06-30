# Introduction
:label:`chapter_introduction`

Until recently, nearly all of the computer programs
that we interacted with every day were coded
by software developers from first principles.
Say that we wanted to write an application to manage an e-commerce platform.
After huddling around a whiteboard for a few hours to ponder the problem,
we would come up with the broad strokes of a working solution
that would probably look something like this:
(i) users would interact with the application
through an interface running in a web browser or mobile application
(ii) our application would rely on a commerical database engine
to keep track of each user's state and maintain records
of all historical transactions
(ii) at the heart of our application, running in parallel across many servers, the *business logic* (you might say, the *brains*)
would map out in methodical details the appropriate action to take
in every conceivable circumstance.

최근까지 우리가 매일 사용하는 거의 모든 컴퓨터 프로그램들은 첫번째 원칙에 입각하여 소프트웨어 개발자에 의해 코드로 만들어졌습니다.
E-커머스(e-commerce)플랫폼을 관리 하는 어플리케이션을 만들려고 한다면, 몇시간의 숙고와 화이트보딩을 통한 후에야, 해결 가능한 방법들을 폭넓게 시도할 생각을 하게 될 겁니다. 그것들은 아마도 다음과 슷할 겁니다.
(i) 사용자는 웹브라우저나 모바일앱을 이용할 것이다.
(ii) 우리의 어플리케이션은 각 유저의 상태, 모든 트랜잭션 기록을 위해 상용 데이터페이스 앤진을 사용할 것이다.
(iii) 병렬로 많은 수의 서버를 통해 동작할 우리 어플리케이션의 핵심 *비즈니스로직*(business logic)은 생각할 수 있는 모든 상황에 대해 체계적으로 세부사항 까지 적절한 액션을 취할 것입니다. (여러분은 아마 *지능*(brain)이라고 부를지 모르겠다.)

To build the *brains* of our application,
we'd have to step through every possible corner case
that we anticipate encountering, devising appropriate rules.
Each time a customer clicks to add an item to their shopping cart,
we add an entry to the shopping cart database table,
associating that user's ID with the requested product’s ID.
While few developers ever get it completely right the first time
(it might take some test runs to work out the kinks),
for the most part, we could write such a program from first principles
and confidently launch it *before ever seeing a real customer*.
Our ability to design automated systems from first principles
that drive functioning products and systems,
often in novel situations, is a remarkable cognitive feat.
And when you're able to devise solutions that work $100\%$ of the time,
*you should not be using machine learning*.

이 어플리케이션의 *지능*(brains)를 만들기 위해, 발생할 수 있는 모든 사항을 대응 할수 있도록 적절한 규칙을 만들어야 합니다.
고객이 매번 하나의 상품을 쇼핑 카트에 넣을 때마다, 우리는 쇼핑키트 데이터 테이블에, 고객의 사용자 ID와 상품ID를 연관시켜야 합니다.
프로그래머들이 처음부터 완벽하게 만들지는 못할 것입니다.
(몇몇의 문제를 풀기위해 몇개의 테스트를 만들어야 할 지도 모릅니다.) 대부분의 경우 *진짜 고객들이 보기 전에* 첫번째원칙에 따라 프로그램을 자신있게 만들수 있습니다. 제일 원칙에따라 자동화된 시스템을 설계할수 있는, 제품과 시스템의 기능을 운영하는 능력은, 종종 새로운 상황에서 주목할 만한 인지 능력 입니다. 
그리고 당신이 $100\%$시간 동안 운영 가능한 솔루션을 고안해 낼수 있수 있다면, *당신은 머신러닝을 사용하지 말아야 합니다*.


Fortunately—for the growing community of ML scientists—many
problems in automation don't bend so easily to human ingenuity.
Imagine huddling around the whiteboard with the smartest minds you know,
but this time you are tackling any of the following problems:

* Write a program that predicts tomorrow's weather given geographic
information, satellite images, and a trailing window of past weather.
* Write a program that takes in a question, expressed in free-form text, and
 answers it correctly.
* Write a program that given an image can identify all the people it contains,
 drawing outlines around each.
* Write a program that presents users with products that they are likely to
  enjoy but unlikely, in the natural course of browsing, to encounter.

다행스럽게도, 성장하는 머신러닝 과학자들의 공동체 덕에, 자동화 과정의 많은 문제들이 너무 쉽게 인간의 독창성에 무릎꿇지 않았습니다. 당신이 갖고 있는 가장 똑똑한 방법으로 화이트보드에 끄적이는 것을 상상해 봅시다, 다만 이번에는 다음과 같은 문제들을 다루고 있습니다.:

* 지정학적 정보, 위성 이미지 그리고 과거 날씨 이력등을 이용하여 내일의 날씨를 예측하는 프로그램 작성
* 자유 형식으로 작성된 질문을 받아, 올바르게 답을 하는 프로그램 작성.
* 주어진 이미지에 포함된 모든 사람들을 식별하고, 각각의 아웃라인을 그리는 프로그램 작성.
* 있을 것 같지는 않지만, 자연스러운 브라우징 중에 겪게될, 그들이 즐길것 같은 제품을 주는 프로그램 작성

In each of these cases, even elite programmers
are incapable of coding up solutions from scratch.
The reasons for this can vary.
Sometimes the program that we are looking for
follows a pattern that changes over time,
and we need our programs to adapt.
In other cases, the relationship
(say between pixels, and abstract categories)
may be too complicated, requiring thousands or millions of computations
that are beyond our conscious understanding
(even if our eyes manage the task effortlessly).
Machine learning (ML) is the study of powerful techniques
that can *learn behavior* from *experience*.
As ML algorithm accumulates more experience,
typically in the form of observational data
or interactions with an environment, their performance improves.
Contrast this with our deterministic e-commerce platform,
which performs according to the same business logic,
no matter how much experience accrues,
until the developers themselves *learn* and decide
that it's time to update the software.
In this book, we will teach you the fundamentals of machine learning,
and focus in particular on deep learning,
a powerful set of techniques driving innovations
in areas as diverse as computer vision, natural language processing,
healthcare, and genomics.

이러한 각각의 경우, 우수한 프로그래머 조차도 처음부터 솔루션을 만들수 없습니다.
그 이유는 매우 다양합니다. 때때로 우리가 찾는 프로그램은 시간이 흐름에 따라 변하는 패턴을 찾아내고, 적응할 수 있어야 합니다. 
다른 한편으로, 픽셀(pixel)과 추상 카테고리 사이의 관계는 너무 복잡하여, 이 관계를 추론하기엔 (우리의 눈이 별다른 노력없이 작업을 관리하더라도) 의식적인 이해를 초월하는, 수천 혹은 수백만의 연산을 필요로 합니다.
기계학습(Machine Learning)은 *경험*으로 부터 *행동을 배울수* 있는 강력한 기술에 대한 연구 입니다.
일반적으로 기계학습 알고리즘이 관찰데이터 혹은 환경과의 상호작용에 따른, 더 많은 경험의 축적을 통해 성능이 향상됩니다. 
개발자가 스스로 *배우고*, 소프트웨어를 수정할 때까지, 얼마나 많은 경험이 쌓였는지와 상관 없이, 정해진대로 동일한 비즈니스 로직에 따라동작하는 우리의 e-커머스(E-Commerce) 플랫폼과 비교해 보세요. 
이 책에서, 기계학습의 기초를 배우고, 특별히 자연어처리, 헬쓰케어(HealthCare), 유전체학 등과 같이 다양한 분야에서 강략한 도구인 컴퓨터비젼(Computer Vision)의 혁신을 주도하는 강략한 딥러닝(DeepLearning)에 중점을 둡니다.

## A Motivating Example

Before we could begin writing, the authors of this book,
like much of the work force, had to become caffeinated.
We hopped in the car and started driving.
Using an iPhone, Alex called out 'Hey Siri',
awakening the phone's voice recognition system.
Then Mu commanded 'directions to Blue Bottle coffee shop'.
The phone quickly displayed the transcription of his command.
It also recognized that we were asking for directions
and launched the Maps application to fulfill our request.
Once launched, the Maps app identified a number of routes.
Next to each route, the phone displayed a predicted transit time.
While we fabricated this story for pedagogical convenience,
it demonstrates that in the span of just a few seconds,
our everyday interactions with a smartphone
can engage several machine learning models.

저자들은 이 책을 쓰기 전에, 많은 노동력이 필요한 일처럼, 많은 카페인이 필요했습니다.. 상상해 봅시다. 우리는 차에 올라타서 운전을 하기 시작했습니다. 아이폰을 사용자인 Alex는 핸드폰의 음성 인식 시스템을 부르기 위해서 'Hey Siri'라고 외쳤습니다. 그러자 Mu는 '블루 보틀 커피샵으로 가는길을 알려줘'라고 명령 했습니다. 핸드폰은 그의 명령을 글로 바꿔서 화면에 빠르게 보여줍니다. 우리가 길을 묻는 것을 알아채고는 우리의 요청에 응하기 위해서 지도 앱을 띄웁니다. 지도 앱이 실행되자 마자 여러 경로를 찾아냅니다. 각 경로 옆에는 예상 소요 시간이 함께 표시됩니다. 설명을 위해서 지어낸 이야기이긴 하지만, 이 짧은 시나리오는 스마트폰을 통해 다양한 머신 러닝 모델이 사용되는 것을 보여주고 있습니다.

Imagine just writing a program to respond to a *wake word*
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
That's why we use ML.

 'Alexa', 'Okay, Google', 이나 'Siri' 같은 wake word 에 응답하는 프로그램을 작성한다고 생각해보세요. 컴퓨터와 코드 편집기만 사용해서 코드를 만들어 나간다고 했을때 제일 원칙을 이용해서 어떻게 그런 프로그램을 작성할 것인가요? 조금만 생각해 봐도 이 문제가 쉽지 않다는 것을 알 수 있습니다. 매 초마다 마이크는 대략 44,000개의 샘플을 수집합니다. 소리 조각으로 부터 그 소리 조각이 wake word를 포함하는지 신뢰있게 {yes, no} 로 예측하는 룰을 만들 수 있나요? 어떻게 할지를 모른다고 해도 걱정하지 마세요. 우리도 그런 프로그램을 처음부터 어떻게 작성해야하는지 모릅니다. 이것이 바로 우리가 머신 러닝을 사용하는 이유입니다.
 
![Identify an awake word.](../img/wake-word.svg)


Here's the trick.
Often, even when we don't know how to tell a computer
explicitly how to map from inputs to outputs,
we are nonetheless capable of performing the cognitive feat ourselves.
In other words, even if you don't know *how to program a computer*
to recognize the word 'Alexa',
you yourself *are able* to recognize the word 'Alexa'.
Armed with this ability,
we can collect a huge *dataset* containing examples of audio
and label those that *do* and that *do not* contain the wake word.
In the ML approach, we do not design a system *explicitly*
to recognize wake words.
Instead, we define a flexible program
whose behavior is determined by a number of *parameters*.
Then we use the dataset to determine
the best possible set of parameters,
those that improve the performance of our program
with respect to some measure of performance on the task of interest.

여기에 트릭이 있습니다. 컴퓨터에 명시적으로, 입력을 출력으로 맵핑하는 방법을 모르는 경우에도,  우리는 스스로 인지해서 이것을 해낼 수 있습니다. 즉, 당신이 'Alexa'라는 단어를 인식하도록 컴퓨터를 프로그래밍하는 방법을 모르지만, 당신은 'Alexa'라는 단어를 인식 할 수 있습니다. 이 기능을 탑제하여, 오디오 예제를 포함하는 거대한 데이터 세트를 수집하고, 웨이크 워드를 포함하지 않는 라벨을 표시 할 수 있습니다. ML 접근법에서 우리는 웨이크워드(wake word)를 인식하도록 시스템을 명시 적으로 설계하지 않습니다. 대신, 우리는 여러 가지 파라미터로 동작이 결정되는 유연한 프로그램을 정의합니다. 그런 다음 데이터 집합을 사용하여 가능한 파라미터 집합을 결정합니다. 파라미터 집합은 관심있는 작업의 성능 측정과 관련하여 프로그램 성능을 향상시킵니다.

You can think of the parameters as knobs that we can turn,
manipulating the behavior of the program.
Fixing the parameters, we call the program a *model*.
The set of all distinct programs (input-output mappings)
that we can produce just by manipulating the parameters
is called a *family* of models.
And the *meta-program* that uses our dataset
to choose the parameters is called a *learning algorithm*.

Before we can go ahead and engage the learning algorithm,
we have to define the problem precisely,
pinning down the exact nature of the inputs and outputs,
and choosing an appropriate model family.
In this case, our model receives a snippet of audio as *input*,
and it generates a selection among ``{yes, no}`` as *output*—which,
if all goes according to plan,
will closely approximate whether (or not)
the snippet contains the wake word.

If we choose the right family of models,
then there should exist one setting of the knobs
such that the model fires ``yes`` every time it hears the word 'Alexa'.
Because the exact choice of the wake word is arbitrary,
we'll probably need a model family capable, via another setting of the knobs,
of firing ``yes`` on the word 'Apricot'.
We expect that the same model should apply to 'Alexa' recognition and 'Apricot' recognition because these are similar tasks.
However, we might need a different family of models entirely
if we want to deal with fundamentally different inputs or outputs,
say if we wanted to map from images to captions,
or from English sentences to Chinese sentences.

As you might guess, if we just set all of the knobs randomly,
it's not likely that our model will recognize 'Alexa', 'Apricot',
or any other English word.
In deep learning, the *learning* is the process
by which we discover the right setting of the knobs
coercing the desired behaviour from our model.

The training process usually looks like this:

1. Start off with a randomly initialized model that can't do anything useful.
1. Grab some of your labeled data (e.g. audio snippets and corresponding ``{yes,no}`` labels)
1. Tweak the knobs so the model sucks less with respect to those examples
1. Repeat until the model is awesome.

![A typical training process. ](../img/ml-loop.svg)

To summarize, rather than code up a wake word recognizer,
we code up a program that can *learn* to recognize wake words,
*if we present it with a large labeled dataset*.
You can think of this act
of determining a program's behavior by presenting it with a dataset
as *programming with data*.
We can "program" a cat detector by providing our machine learning system
with many examples of cats and dogs, such as the images below:

| ![cat1](../img/cat1.png) | ![cat2](../img/cat2.jpg) | ![dog1](../img/dog1.jpg) |![dog2](../img/dog2.jpg) |
|:---------------:|:---------------:|:---------------:|:---------------:|
|cat|cat|dog|dog|

This way the detector will eventually learn to emit
a very large positive number if it's a cat,
a very large negative number if it's a dog,
and something closer to zero if it isn't sure,
and this barely scratches the surface of what ML can do.

Deep learning is just one among many
popular frameworks for solving machine learning problems.
While thus far, we've only talked about machine learning broadly
and not deep learning, there's a couple points worth sneaking in here:
First, the problems that we've discussed thus far:
learning from raw audio signal,
directly from the pixels in images,
and mapping between sentences of arbitrary lengths and across languages
are problems where deep learning excels and traditional ML tools faltered.
Deep models are *deep* in precisely the sense that they learn
many *layers* of computation.
It turns out that these many-layered (or hierarchical) models
are capable of addressing low-level perceptual data
in a way that previous tools could not.
In bygone days, the crucial part of applying ML to these problems
consisted of coming up with manually engineered ways of transforming
the data into some form amenable to *shallow* models.
One key advantage of deep learning is that it replaces not only the *shallow* models
at the end of traditional learning pipelines,
but also the labor-intensive feature engineering.
Secondly, by replacing much of the *domain-specific preprocessing*,
deep learning has eliminated many of the boundaries
that previously separated computer vision, speech recognition,
natural language processing, medical informatics, and other application areas,
offering a unified set of tools for tackling diverse problems.


## The Key Components: Data, Models, and Algorithms

In our *wake-word* example, we described a dataset
consisting of audio snippets and binary labels
gave a hand-wavy sense of how we might *train*
a model to approximate a mapping from snippets to classifications.
This sort of problem, where we try to predict a designated unknown *label*
given known *inputs* (also called *features* or *covariates*),
and examples of both is called *supervised learning*,
and it's just one among many *kinds* of machine learning problems.
In the next section, we'll take a deep dive into the different ML problems.
First, we'd like to shed more light on some core components
that will follow us around, no matter what kind of ML problem we take on:

1. The **data** that we can learn from
2. A **model** of how to transform the data
3. A **loss** function that quantifies the *badness* of our model
4. An **algorithm** to adjust the model's parameters to minimize the loss


### Data

It might go without saying that you cannot do data science without data.
We could lose hundreds of pages pondering the precise nature of data
but for now we'll err on the practical side and focus on the key properties
to be concerned with.
Generally we are concerned with a collection of *examples*
(also called *data points*, *samples*, or *instances*).
In order to work with data usefully, we typically
need to come up with a suitable numerical representation.
Each *example* typically consists of a collection
of numerical attributes called *features* or *covariates*.

If we were working with image data,
each individual photograph might constitute an *example*,
each represented by an ordered list of numerical values
corresponding to the brightness of each pixel.
A $200\times200$ color photograph would consist of $200\times200\times3=120000$
numerical values, corresponding to the brightness
of the red, green, and blue channels corresponding to each spatial location.
In a more traditional task, we might try to predict
whether or not a patient will survive,
given a standard set of features such as age, vital signs, diagnoses, etc.

When every example is characterized by the same number of numerical values,
we say that the data consists of *fixed-length* vectors
and we describe the (constant) length of the vectors
as the *dimensionality* of the data.
As you might imagine, fixed length can be a convenient property.
If we wanted to train a model to recognize cancer in microscopy images,
fixed-length inputs means we have one less thing to worry about.

However, not all data can easily be represented as fixed length vectors.
While we might expect microscrope images to come from standard equipment,
we can't expect images mined from the internet to all show up in the same size.
While we might imagine cropping images to a standard size,
text data resists fixed-length representations even more stubbornly.
Consider the product reviews left on e-commerce sites like Amazon or TripAdvisor. Some are short: "it stinks!". Others ramble for pages.
One major advantage of deep learning over traditional methods
is the comparative grace with which modern models
can handle *varying-length* data.

Generally, the more data we have, the easier our job becomes.
When we have more data, we can train more powerful models,
and rely less heavily on pre-conceived assumptions.
The regime change from (comparatively small) to big data
is a major contributor to the success of modern deep learning.
To drive the point home, many of the most exciting models in deep learning either don't work without large data sets.
Some others work in the low-data regime,
but no better than traditional approaches.

Finally it's not enough to have lots of data and to process it cleverly.
We need the *right* data.
If the data is full of mistakes, or if the chosen features are not predictive of the target quantity of interest, learning is going to fail.
The situation is well captured by the cliché: *garbage in, garbage out*.
Moreover, poor predictive performance isn't the only potential consequence.
In sensitive applications of machine learning,
like predictive policing, resumé screening, and risk models used for lending,
we must be especially alert to the consequences of garbage data.
One common failure mode occurs in datasets where some groups of people
are unrepresented in the training data.
Imagine applying a skin cancer recognition system in the wild
that had never seen black skin before.
Failure can also occur when the data doesn't merely under-represent some groups,
but reflects societal prejudices.
For example if past hiring decisions are used to train a predictive model
that will be used to screen resumes, then machine learning models could inadvertently capture and automate historical injustices.
Note that this can all happen without the data scientist being complicit,
or even aware.


### Models


Most machine learning involves *transforming* the data in some sense.
We might want to build a system that ingests photos and predicts *smiley-ness*.
Alternatively, we might want to ingest a set of sensor readings
and predict how *normal* vs *anomalous* the readings are.
By *model*, we denote the computational machinery for ingesting data
of one type, and spitting out predictions of a possibly different type.
In particular, we are interested in statistical models
that can be estimated from data.
While simple models are perfectly capable of addressing
appropriately simple problems the problems
that we focus on in this book stretch the limits of classical methods.
Deep learning is differentiated from classical approaches
principally by the set of powerful models that it focuses on.
These models consist of many successive transformations of the data
that are chained together top to bottom, thus the name *deep learning*.
On our way to discussing deep neural networks, we'll discuss some more traditional methods.


###  Objective functions

Earlier, we introduced machine learning as "learning behavior from experience".
By *learning* here, we mean *improving* at some task over time.
But who is to say what constitutes an improvement?
You might imagine that we could propose to update our model,
and some people might disagree on whether the proposed update
constitued an improvement or a decline.

In order to develop a formal mathematical system of learning machines,
we need to have formal measures of how good (or bad) our models are.
In machine learning, and optimization more generally,
we call these objective functions.
By convention, we usually define objective funcitons
so that *lower* is *better*.
This is merely a convention. You can take any function $f$
for which higher is better, and turn it into a new function $f'$
that is qualitatively identical but for which lower is better
by setting $f' = -f$.
Because lower is better, these functions are sometimes called
*loss functions* or *cost functions*.

When trying to predict numerical values,
the most common objective function is squared error $(y-\hat{y})^2$.
For classification, the most common objective is to minimize error rate,
i.e., the fraction of instances on which
our predictions disagree with the ground truth.
Some objectives (like squared error) are easy to optimize.
Others (like error rate) are difficult to optimize directly,
owing to non-differentiability or other complications.
In these cases, it's common to optimize a surrogate objective.

Typically, the loss function is defined
with respect to the model's parameters
and depends upon the dataset.
The best values of our model's parameters are learned
by minimizing the loss incurred on a *training set*
consisting of some number of *examples* collected for training.
However, doing well on the training data
doesn't guarantee that we will do well on (unseen) test data.
So we'll typically want to split the available data into two partitions:
the training data (for fitting model parameters)
and the test data (which is held out for evaluation),
reporting the following two quantities:

 * **Training Error:**
 The error on that data on which the model was trained.
 You could think of this as being like
 a student's scores on practice exams
 used to prepare for some real exam.
 Even if the results are encouraging,
 that does not guarantee success on the final exam.
 * **Test Error:** This is the error incurred on an unseen test set.
 This can deviate significantly from the training error.
 When a model fails to generalize to unseen data,
 we say that it is *overfitting*.
 In real-life terms, this is like flunking the real exam
 despite doing well on practice exams.


### Optimization algorithms

Once we've got some data source and representation,
a model, and a well-defined objective function,
we need an algorithm capable of searching
for the best possible parameters for minimizing the loss function.
The most popular optimization algorithms for neural networks
follow an approach called gradient descent.
In short, at each step, they check to see, for each parameter,
which way the training set loss would move
if you perturbed that parameter just a small amount.
They then update the parameter in the direction that reduces the loss.


## Kinds of Machine Learning

In the following sections, we will discuss a few types of machine learning in some more detail. We begin with a list of *objectives*, i.e. a list of things that machine learning can do. Note that the objectives are complemented with a set of techniques of *how* to accomplish them, i.e. training, types of data, etc. The list below is really only sufficient to whet the readers' appetite and to give us a common language when we talk about problems. We will introduce a larger number of such problems as we go along.

### Supervised learning

Supervised learning addresses the task of predicting *targets* given input data.
The targets, also commonly called *labels*, are generally denoted *y*.
The input data points, also commonly called *examples* or *instances*, are typically denoted $\boldsymbol{x}$.
The goal is to produce a model $f_\theta$ that maps an input $\boldsymbol{x}$ to a prediction $f_{\theta}(\boldsymbol{x})$

To ground this description in a concrete example,
if we were working in healthcare,
then we might want to predict whether or not a patient would have a heart attack.
This observation, *heart attack* or *no heart attack*,
would be our label $y$.
The input data $\boldsymbol{x}$ might be vital signs such as heart rate, diastolic and systolic blood pressure, etc.

The supervision comes into play because for choosing the parameters $\theta$, we (the supervisors) provide the model with a collection of *labeled examples* ($\boldsymbol{x}_i, y_i$), where each example $\boldsymbol{x}_i$ is matched up against its correct label.

In probabilistic terms, we typically are interested in estimating
the conditional probability $P(y|x)$.
While it's just one among several approaches to machine learning,
supervised learning accounts for the majority of machine learning in practice.
Partly, that's because many important tasks
can be described crisply as estimating the probability of some unknown given some available evidence:

* Predict cancer vs not cancer, given a CT image.
* Predict the correct translation in French, given a sentence in English.
* Predict the price of a stock next month based on this month's financial reporting data.

Even with the simple description 'predict targets from inputs'
supervised learning can take a great many forms and require a great many modeling decisions,
depending on the type, size, and the number of inputs and outputs.
For example, we use different models to process sequences (like strings of text or time series data)
and for processing fixed-length vector representations.
We'll visit many of these problems in depth throughout the first 9 parts of this book.

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

![Supervised learning.](../img/supervised-learning.svg)



#### Regression

Perhaps the simplest supervised learning task to wrap your head around is Regression.
Consider, for example a set of data harvested
from a database of home sales.
We might construct a table, where each row corresponds to a different house,
and each column corresponds to some relevant attribute,
such as the square footage of a house, the number of bedrooms, the number of bathrooms,
and the number of minutes (walking) to the center of town.
Formally, we call one row in this dataset a *feature vector*,
and the object (e.g. a house) it's associated with an *example*.

If you live in New York or San Francisco, and you are not the CEO of Amazon, Google, Microsoft, or Facebook,
the (sq. footage, no. of bedrooms, no. of bathrooms, walking distance) feature vector for your home
might look something like: $[100, 0, .5, 60]$.
However, if you live in Pittsburgh,
it might look more like $[3000, 4, 3, 10]$.
Feature vectors like this are essential for all the classic machine learning problems.
We'll typically denote the feature vector for any one example $\mathbf{x_i}$
and the set of feature vectors for all our examples $X$.

What makes a problem a *regression* is actually the outputs.
Say that you're in the market for a new home,
you might want to estimate the fair market value of a house,
given some features like these.
The target value, the price of sale, is a *real number*.
We denote any individual target $y_i$ (corresponding to example $\mathbf{x_i}$)
and the set of all targets $\mathbf{y}$ (corresponding to all examples X).
When our targets take on arbitrary real values in some range,
we call this a regression problem.
The goal of our model is to produce predictions (guesses of the price, in our example)
that closely approximate the actual target values.
We denote these predictions $\hat{y}_i$
and if the notation seems unfamiliar, then just ignore it for now.
We'll unpack it more thoroughly in the subsequent chapters.


Lots of practical problems are well-described regression problems.
Predicting the rating that a user will assign to a movie is a regression problem,
and if you designed a great algorithm to accomplish this feat in 2009,
you might have won the [$1 million Netflix prize](https://en.wikipedia.org/wiki/Netflix_Prize).
Predicting the length of stay for patients in the hospital is also a regression problem.
A good rule of thumb is that any *How much?* or *How many?* problem should suggest regression.

* 'How many hours will this surgery take?' - *regression*
* 'How many dogs are in this photo?' - *regression*.

However, if you can easily pose your problem as 'Is this a _ ?',
then it's likely, classification, a different fundamental problem type that we'll cover next.
Even if you've never worked with machine learning before,
you've probably worked through a regression problem informally.
Imagine, for example, that you had your drains repaired
and that your contractor spent $x_1=3$ hours removing gunk from your sewage pipes.
Then she sent you a bill of $y_1 = \$350$.
Now imagine that your friend hired the same contractor for $x_2 = 2$ hours
and that she received a bill of $y_2 = \$250$.
If someone then asked you how much to expect on their upcoming gunk-removal invoice
you might make some reasonable assumptions,
such as more hours worked costs more dollars.
You might also assume that there's some base charge and that the contractor then charges per hour.
If these assumptions held true, then given these two data points,
you could already identify the contractor's pricing structure:
\$100 per hour plus \$50 to show up at your house.
If you followed that much then you already understand the high-level idea behind linear regression (and you just implicitly designed a linear model with bias).

In this case, we could produce the parameters that exactly matched the contractor's prices.
Sometimes that's not possible, e.g., if some of the variance owes to some factors besides your two features.
In these cases, we'll try to learn models that minimize the distance between our predictions and the observed values.
In most of our chapters, we'll focus on one of two very common losses,
the
[L1 loss](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L1Loss)
where

$$l(y,y') = \sum_i |y_i-y_i'|$$

and the least mean squares loss, aka
[L2 loss](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L2Loss)
where

$$l(y,y') = \sum_i (y_i - y_i')^2.$$

As we will see later, the $L_2$ loss corresponds to the assumption that our data was corrupted by Gaussian noise, whereas the $L_1$ loss corresponds to an assumption of noise from a Laplace distribution.

#### Classification

While regression models are great for addressing *how many?* questions,
lots of problems don't bend comfortably to this template. For example,
a bank wants to add check scanning to their mobile app.
This would involve the customer snapping a photo of a check with their smartphone's camera
and the machine learning model would need to be able to automatically understand text seen in the image.
It would also need to understand hand-written text to be even more robust.
This kind of system is referred to as optical character recognition (OCR),
and the kind of problem it solves is called a classification.
It's treated with a distinct set of algorithms than those that are used for regression.

In classification, we want to look at a feature vector, like the pixel values in an image,
and then predict which category (formally called *classes*),
among some set of options, an example belongs.
For hand-written digits, we might have 10 classes,
corresponding to the digits 0 through 9.
The simplest form of classification is when there are only two classes,
a problem which we call binary classification.
For example, our dataset $X$ could consist of images of animals
and our *labels* $Y$ might be the classes $\mathrm{\{cat, dog\}}$.
While in regression, we sought a *regressor* to output a real value $\hat{y}$,
in classification, we seek a *classifier*, whose output $\hat{y}$ is the predicted class assignment.

For reasons that we'll get into as the book gets more technical, it's pretty hard to optimize a model that can only output a hard categorical assignment, e.g. either *cat* or *dog*.
It's a lot easier instead to express the model in the language of probabilities.
Given an example $x$, the model assigns a probability $\hat{y}_k$ to each label $k$.
Because these are probabilities, they need to be positive numbers and add up to $1$.
This means that we only need $K-1$ numbers to give the probabilities of $K$ categories.
This is easy to see for binary classification.
If there's a 0.6 (60%) probability that an unfair coin comes up heads,
then there's a 0.4 (40%) probability that it comes up tails.
Returning to our animal classification example, a classifier might see an image
and output the probability that the image is a cat $\Pr(y=\mathrm{cat}| x) = 0.9$.
We can interpret this number by saying that the classifier is 90% sure that the image depicts a cat.
The magnitude of the probability for the predicted class is one notion of confidence.
It's not the only notion of confidence and we'll discuss different notions of uncertainty in more advanced chapters.

When we have more than two possible classes, we call the problem *multiclass classification*.
Common examples include hand-written character recognition `[0, 1, 2, 3 ... 9, a, b, c, ...]`.
While we attacked regression problems by trying to minimize the L1 or L2 loss functions,
the common loss function for classification problems is called cross-entropy.
In MXNet Gluon, the corresponding loss function can be found [here](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SoftmaxCrossEntropyLoss).

Note that the most likely class is not necessarily the one that you're going to use for your decision. Assume that you find this beautiful mushroom in your backyard:

![Death cap - do not eat!](../img/death_cap.jpg)
:width:`400px`

Now, assume that you built a classifier and trained it
to predict if a mushroom is poisonous based on a photograph.
Say our poison-detection classifier outputs $\Pr(y=\mathrm{death cap}|\mathrm{image}) = 0.2$.
In other words, the classifier is 80% confident that our mushroom *is not* a death cap.
Still, you'd have to be a fool to eat it.
That's because the certain benefit of a delicious dinner isn't worth a 20% risk of dying from it.
In other words, the effect of the *uncertain risk* by far outweighs the benefit.
Let's look at this in math. Basically, we need to compute the expected risk that we incur, i.e. we need to multiply the probability of the outcome with the benefit (or harm) associated with it:

$$L(\mathrm{action}| x) = \mathbf{E}_{y \sim p(y| x)}[\mathrm{loss}(\mathrm{action},y)]$$

Hence, the loss $L$ incurred by eating the mushroom is $L(a=\mathrm{eat}| x) = 0.2 * \infty + 0.8 * 0 = \infty$, whereas the cost of discarding it is $L(a=\mathrm{discard}| x) = 0.2 * 0 + 0.8 * 1 = 0.8$.

Our caution was justified: as any mycologist would tell us, the above mushroom actually *is* a death cap.
Classification can get much more complicated than just binary, multiclass, or even multi-label classification.
For instance, there are some variants of classification for addressing hierarchies.
Hierarchies assume that there exist some relationships among the many classes.
So not all errors are equal - we prefer to misclassify to a related class than to a distant class.
Usually, this is referred to as *hierarchical classification*.
One early example is due to [Linnaeus](https://en.wikipedia.org/wiki/Carl_Linnaeus),
who organized the animals in a hierarchy.

![Classify sharks](../img/sharks.png)
:width:`500px`

In the case of animal classification, it might not be so bad to mistake a poodle for a schnauzer,
but our model would pay a huge penalty if it confused a poodle for a dinosaur.
Which hierarchy is relevant might depend on how you plan to use the model.
For example, rattle snakes and garter snakes might be close on the phylogenetic tree,
but mistaking a rattler for a garter could be deadly.

#### Tagging

Some classification problems don't fit neatly into the binary or multiclass classification setups.
For example, we could train a normal binary classifier to distinguish cats from dogs.
Given the current state of computer vision,
we can do this easily, with off-the-shelf tools.
Nonetheless, no matter how accurate our model gets, we might find ourselves in trouble when the classifier encounters an image of the Town Musicians of Bremen.

![A cat, a roster, a dog and a donkey](../img/stackedanimals.jpg)
:width:`500px`


As you can see, there's a cat in the picture, and a rooster, a dog, a donkey and a bird, with some trees in the background.
Depending on what we want to do with our model ultimately,
treating this as a binary classification problem
might not make a lot of sense.
Instead, we might want to give the model the option
of saying the image depicts a cat *and* a dog *and* a donkey *and* a rooster *and* a bird.

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

We also have to deal with this kind of problem when dealing with the biomedical literature,
where correctly tagging articles is important
because it allows researchers to do exhaustive reviews of the literature.
At the National Library of Medicine, a number of professional annotators
go over each article that gets indexed in PubMed
to associate it with the relevant terms from MeSH,
a collection of roughly 28k tags.
This is a time-consuming process and the annotators typically have a one year lag between archiving and tagging. Machine learning can be used here to provide provisional tags
until each article can have a proper manual review.
Indeed, for several years, the BioASQ organization has [hosted a competition](http://bioasq.org/)
to do precisely this.


#### Search and ranking

Sometimes we don't just want to assign each example to a bucket or to a real value. In the field of information retrieval, we want to impose a ranking on a set of items. Take web search for example, the goal is less to determine whether a particular page is relevant for a query, but rather, which one of the plethora of search results should be displayed for the user. We really care about the ordering of the relevant search results and our learning algorithm needs to produce ordered subsets of elements from a larger set. In other words, if we are asked to produce the first 5 letters from the alphabet, there is a difference between returning ``A B C D E`` and ``C A B E D``. Even if the result set is the same, the ordering within the set matters nonetheless.

One possible solution to this problem is to score every element in the set of possible sets along with a corresponding relevance score and then to retrieve the top-rated elements. [PageRank](https://en.wikipedia.org/wiki/PageRank) is an early example of such a relevance score. One of the peculiarities is that it didn't depend on the actual query. Instead, it simply helped to order the results that contained the query terms. Nowadays search engines use machine learning and behavioral models to obtain query-dependent relevance scores. There are entire conferences devoted to this subject.

<!-- Add / clean up-->

#### Recommender systems

Recommender systems are another problem setting that is related to search and ranking. The problems are  similar insofar as the goal is to display a set of relevant items to the user. The main difference is the emphasis on *personalization* to specific users in the context of recommender systems. For instance, for movie recommendations, the results page for a SciFi fan and the results page for a connoisseur of Woody Allen comedies might differ significantly.

Such problems occur, e.g. for movie, product or music recommendation. In some cases, customers will provide explicit details about how much they liked the product (e.g. Amazon product reviews). In some other cases, they might simply provide feedback if they are dissatisfied with the result (skipping titles on a playlist). Generally, such systems strive to estimate some score $y_{ij}$, such as an estimated rating or probability of purchase, given a user $u_i$ and product $p_j$.

Given such a model, then for any given user, we could retrieve the set of objects with the largest scores $y_{ij}$, which are then used as a recommendation. Production systems are considerably more advanced and take detailed user activity and item characteristics into account when computing such scores. The following image is an example of deep learning books recommended by Amazon based on personalization algorithms tuned to the author's preferences.

![Deep learning books recommended by Amazon.](../img/deeplearning_amazon.png)


#### Sequence Learning

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

These problems are among the most exciting applications of machine learning
and they are instances of *sequence learning*.
They require a model to either ingest sequences of inputs
or to emit sequences of outputs (or both!).
These latter problems are sometimes referred to as ``seq2seq`` problems.
Language translation is a ``seq2seq`` problem.
Transcribing text from spoken speech is also a ``seq2seq`` problem.
While it is impossible to consider all types of sequence transformations,
a number of special cases are worth mentioning:

##### Tagging and Parsing

This involves annotating a text sequence with attributes. In other words, the number of inputs and outputs is essentially the same. For instance, we might want to know where the verbs and subjects are. Alternatively, we might want to know which words are the named entities. In general, the goal is to decompose and annotate text based on structural and grammatical assumptions to get some annotation. This sounds more complex than it actually is. Below is a very simple example of annotating a sentence with tags indicating which words refer to named entities.

```text
Tom has dinner in Washington with Sally.
Ent  -    -    -     Ent      -    Ent
```


##### Automatic Speech Recognition

With speech recognition, the input sequence $x$ is the sound of a speaker,
and the output $y$ is the textual transcript of what the speaker said.
The challenge is that there are many more audio frames (sound is typically sampled at 8kHz or 16kHz) than text, i.e. there is no 1:1 correspondence between audio and text,
since thousands of samples correspond to a single spoken word.
These are ``seq2seq`` problems where the output is much shorter than the input.

![`-D-e-e-p- L-ea-r-ni-ng-`](../img/speech.png)
:width:`700px`

##### Text to Speech

Text-to-Speech (TTS) is the inverse of speech recognition.
In other words, the input $x$ is text
and the output $y$ is an audio file.
In this case, the output is *much longer* than the input.
While it is easy for *humans* to recognize a bad audio file,
this isn't quite so trivial for computers.

##### Machine Translation

Unlike the case of speech recognition, where corresponding inputs and outputs occur in the same order (after alignment),
in machine translation, order inversion can be vital.
In other words, while we are still converting one sequence into another,
neither the number of inputs and outputs
nor the order of corresponding data points
are assumed to be the same.
Consider the following illustrative example of the obnoxious tendency of Germans
<!-- Alex writing here -->
to place the verbs at the end of sentences.

```text
German:           Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?
English:          Did you already check out this excellent tutorial?
Wrong alignment:  Did you yourself already this excellent tutorial looked-at?
```

A number of related problems exist.
For instance, determining the order in which a user reads a webpage
is a two-dimensional layout analysis problem.
Likewise, for dialogue problems,
we need to take world-knowledge and prior state into account.
This is an active area of research.


### Unsupervised learning

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
However, if you plan to be a data scientist, you'd better get used to it.
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


### Interacting with an Environment

So far, we haven't discussed where data actually comes from,
or what actually *happens* when a machine learning model generates an output.
That's because supervised learning and unsupervised learning
do not address these issues in a very sophisticated way.
In either case, we grab a big pile of data up front,
then do our pattern recognition without ever interacting with the environment again.
Because all of the learning takes place after the algorithm is disconnected from the environment,
this is called *offline learning*.
For supervised learning, the process looks like this:

![Collect data for supervised learning from an environment.](../img/data-collection.svg)


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
* have shifting dynamics (steady vs. shifting over time)?

This last question raises the problem of *covariate shift*,
(when training and test data are different).
It's a problem that most of us have experienced when taking exams written by a lecturer,
while the homeworks were composed by his TAs.
We'll briefly describe reinforcement learning and adversarial learning,
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

![The interaction between reinforcement learning and an environment.](../img/rl-environment.svg)

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

체스게임의 예를 생각해보면,
실제의 보상 신호는 게임이 끝났을 때나옵니다. 이기면 1의 보상을 , 지면 -1의 보상을 줄 수 있습니다.
그래서 강화학습은 학습자는  *credit assignment problem*을 다루어야 합니다. 같은 방식으로 10월 11일에 승진한 직원이 있다면, 이 승진은
이전 년도에 걸쳐 수많은 잘 선택된 행동(Action)이 반영된 것과 같습니다. 앞으로 더 많은 프로모션을 하기 위해선, 어떠한 행동들이 프로모션에 영향을 미치는지 알아 내야 합니다.

Reinforcement learners may also have to deal with the problem of partial observability.
That is, the current observation might not tell you everything about your current state.
Say a cleaning robot found itself trapped in one of many identical closets in a house.
Inferring the precise location (and thus state) of the robot
might require considering its previous observations before entering the closet.

강화학습 학습자는 또한 부분적인 관찰가능성의 문제를 다룰 수 있어야 합니다. 그것은, 현재 관찰된 사항이 현재 상태의 모든 정보를 말하고 있지 않을 수 있기 때문입니다.
청소로봇이 집안의 옷장에 자주 갇히는 것으로 나타났습니다. 정확한 로봇의 위치를 추정하는 것은 옷장에 들어가기 전에 이전의 관찰 정보를 고려하도록 해야 할 수 있습니다.

Finally, at any given point, reinforcement learners might know of one good policy,
but there might be many other better policies that the agent has never tried.
The reinforcement learner must constantly choose
whether to *exploit* the best currently-known strategy as a policy,
or to *explore* the space of strategies,
potentially giving up some short-run reward in exchange for knowledge.

마지막으로, 어떠한 주어진 점수에 따라, 강화학습자는 하나의 좋은 정책(Policy)를 알게 됩니다. 그러나 에이전트(agent)가 시도해 보지 않은, 많은 더 좋은 정책들이 있을 수 있습니다. 강화학습 러너는 항상, 지금까지 알려진 최적의 전략에 따라 *탐험*을 할지, 이외의 전략 공간을 *모험*을 할지 선택해야 합니다.


#### MDPs, bandits, and friends

The general reinforcement learning problem
is a very general setting.
Actions affect subsequent observations.
Rewards are only observed corresponding to the chosen actions.
The environment may be either fully or partially observed.
Accounting for all this complexity at once may ask too much of researchers.
Moreover not every practical problem exhibits all this complexity.
As a result, researchers have studied a number of *special cases* of reinforcement learning problems.

일반적인 강화학습 문제는 매우 일반적인 설정입니다.
액션들은 이후의 관찰들에 영향을 미칩니다.
선택된 액션과 일치하는 상황이 관찰될 때에만 보상이 이루어 집니다. 
환경은 완전하게 혹은 부분적으로 관찰될 수 있습니다.
이 모든 복잡도를 한번에 설명하는 것은 매우 많은 연구자들에게 물어봐야 할 수도 있다.
더욱이 모든 실제 문제들이 이러한 복잡도를 나타내지도 않습니다.
그러한 결과로, 연구자들은 강화학습 문제들 중 몇개의 *특별한*를 연구 하였다.

When the environment is fully observed,
we call the RL problem a *Markov Decision Process* (MDP).
When the state does not depend on the previous actions,
we call the problem a *contextual bandit problem*.
When there is no state, just a set of available actions with initially unknown rewards,
this problem is the classic *multi-armed bandit problem*.

완전히 관찰 가능한 환경일 때, 우리는 RL 문제를 *Markov Decision Process* (MDP)라고 부릅니다.
이전 액션을 의존하지 않을 때, 우리는 문제를 *contextual bandit problem* 이라고 부릅니다.
상태가 없고, 초기의 알수 없는 보상을 갖는 가능한 액션 조합만이 있을 때, 이 문제는 고전적인 *multi-armed bandit problem* 입니다.

# 딥러닝 소개

2016년, 네임드 데이터 과학자인 죠엘 그루스(Joel Grus)는 한 유명 인터넷 기업에서 [면접](http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/)을 보았습니다. 보통 그러하듯이 인터뷰 담당자는 그의 프로그래밍 기술을 평가하는 문제를 냈습니다. 간단한 어린이 게임인 FizzzBuzz를 구현하는 것이 과제였습니다. 그 안에서 플레이어는 1부터 카운트하면서 3으로 나눌 수 있는 숫자는 'fizz' 로, 5로 나눌 수 있는 숫자는 'buzz' 로 바꿉니다. 15로 나눌 수 있는 숫자는 'FizzBuzz' 가 됩니다. 즉, 플레이어는 시퀀스를 생성합니다.

```
1 2 fizz 4 buzz fizz 7 8 fizz buzz 11 ...
```

전혀 예상하지 못한 일이 벌어졌습니다. 거의 몇 줄의 Python 코드로 *알고리즘* 을 구현해서 문제를 해결하는 대신, 그는 데이터를 활용한 프로그램으로 문제를 풀기로 했습니다. 그는 (3, fizz), (5, buzz), (7, 7), (2, 2), (15, fizzbuzz) 의 쌍을 활용하여 분류기를 학습시켰습니다. 그가 작은 뉴럴 네트워크(neural network)를 만들고, 그것을 이 데이터를 활용하여 학습시켰고 그 결과 꽤 높은 정확도를 달성하였습니다(면접관이 좋은 점수를 주지 않아서 채용되지는 못했습니다).

이 인터뷰와 같은 상황은 프로그램 설계가 데이터에 의한 학습으로 보완 되거나 대체되는 컴퓨터 과학의 획기적인 순간입니다. 예로 든 상황은 면접이라서가 아니라 어떠한 목표를 손쉽게 달성 할 수 있게 해 준다는 점에서 중요성을 가집니다. 일반적으로는 위에서 설명한 오버스러운 방식으로 FizzBuzz를 해결하지는 않겠지만, 얼굴을 인식하거나, 사람의 목소리 또는 텍스트로 감정을 분류하거나, 음성을 인식할 때는 완전히 다른 이야기입니다. 좋은 알고리즘, 많은 연산 장치 및 데이터, 그리고 좋은 소프트웨어 도구들로 인해 이제는 대부분의 소프트웨어 엔지니어가 불과 10년전에는 최고의 과학자들에게도 너무 도전적이라고 여겨졌던 문제를 해결하는 정교한 모델을 만들 수 있게 되었습니다.

이 책은 머신러닝을 구현하는 여정에 들어선 엔지니어를 돕는 것을 목표로 합니다. 우리는 수학, 코드, 예제를 쉽게 사용할 수 있는 패키지로 결합하여 머신러닝을 실용적으로 만드는 것을 목표로 합니다. 온라인으로 제공되는 Jupyter 노트북 예제들은 노트북이나 클라우드 서버에서 실행할 수 있습니다. 우리는 이를 통해서 새로운 세대의 프로그래머, 기업가, 통계학자, 생물학자 및 고급 머신러닝 알고리즘을 배포하는 데 관심이 있는 모든 사람들이 문제를 해결할 수 있기를 바랍니다.

## 데이터를 활용하는 프로그래밍

코드를 이용하는 프로그래밍과 데이터를 활용하는 프로그래밍의 차이점을 좀 더 자세히 살펴 보겠습니다. 이 둘은 보이는 것보다 더 심오하기 때문입니다. 대부분의 전통적인 프로그램은 머신러닝을 필요로 하지 않습니다. 예를 들어 전자 레인지용 사용자 인터페이스를 작성하려는 경우 약간의 노력으로 몇 가지 버튼을 설계할 수 있습니다. 다양한 조건에서 전자 레인지의 동작을 정확하게 설명하는 몇 가지 논리와 규칙을 추가하면 완료됩니다. 마찬가지로 사회 보장 번호의 유효성을 확인하는 프로그램은 여러 규칙이 적용되는지 여부를 테스트하면 됩니다. 예를 들어, 이러한 숫자는 9 자리 숫자를 포함해야 하며 000으로 시작하지 않아야 한다와 같은 규칙입니다.

위의 두 가지 예에서 프로그램의 논리를 이해하기 위해 현실 세계에서 데이터를 수집할 필요가 없으며, 그 데이터의 특징을 추출할 필요가 없다는 점에 주목할 가치가 있습니다. 많은 시간이 있다면, 우리의 상식과 알고리즘 기술은 우리가 작업을 완료하기에 충분합니다.

우리가 전에 관찰 한 바와 같이, 심지어 최고의 프로그래머의 능력을 넘는 많은 예가 있지만, 많은 아이들, 심지어 많은 동물들이 쉽게 그들을 해결할 수 있습니다. 이미지에 고양이가 포함되어 있는지 여부를 감지하는 문제를 고려해보겠습니다. 어디서부터 시작해야 할까요? 이 문제를 더욱 단순화해 보겠습니다. 모든 이미지가 동일한 크기 (예, 400x400 픽셀)이라고 가정하고, 각 픽셀이 빨강, 녹색 및 파랑 값으로 구성된 경우 이미지는 480,000 개의 숫자로 표시됩니다. 우리의 고양이 탐지기가 관련된 정보가 어디에 있는지 결정하는 것은 불가능합니다. 그것은 모든 값의 평균일까요? 네 모서리의 값일까요? 아니면 이미지의 특정 지점일까요? 실제로 이미지의 내용을 해석하려면 가장자리, 질감, 모양, 눈, 코와 같은 수천 개의 값을 결합 할 때만 나타나는 특징을 찾아야합니다. 그래야만 이미지에 고양이가 포함되어 있는지 여부를 판단할 수 있습니다.

다른 전략은 최종 필요성에 기반한 솔루션을 찾는 것입니다. 즉, 이미지 예제 및 원하는 응답 (cat, cat 없음) 을 출발점으로 사용하여 *데이터로 프로그래밍하는 것*입니다. 우리는 고양이의 실제 이미지 (인터넷에서 인기있는 주제)들과 다른 것들을 수집할 수 있습니다. 이제 우리의 목표는 이미지에 고양이가 포함되어 있는지 여부를 *배울 수 있는* 함수를 찾는 것입니다. 일반적으로 함수의 형태 (예, 다항식)는 엔지니어에 의해 선택되며 그 함수의 파라미터들은 데이터에서 *학습*됩니다.

일반적으로 머신러닝은 고양이 인식과 같은 문제를 해결하는 데 사용할 수 있는 다양한 종류의 함수를 다룹니다. 딥 러닝은 특히 신경망에서 영감을 얻은 특정 함수의 클래스를 사용해서, 이것을 특별한 방법으로 학습(함수의 파라미터를 계산하는 것)시키는 방법입니다. 최근에는 빅 데이터와 강력한 하드웨어 덕분에 이미지, 텍스트, 오디오 신호 등과 같은 복잡한 고차원 데이터를 처리하는 데 있어 딥 러닝이 사실상의 표준으로(de facto choice) 자리잡았습니다.

## 기원

딥 러닝은 최근의 발명품이지만, 인간은 데이터를 분석하고 미래의 결과를 예측하려는 욕구를 수세기 동안 가지고 있어왔습니다. 사실, 자연 과학의 대부분은 이것에 뿌리를 두고 있습니다. 예를 들어, 베르누이 분포는 [야곱 베르누이 (1655-1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli) 의 이름을 따서 명명되었으며, 가우시안 분포는 [칼 프리드리히 가우스 (1777-1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss) 에 의해 발견되었습니다. 예를 들어, 그는 최소 평균 제곱 알고리즘을 발명했는데, 이것은 보험 계산부터 의료 진단까지 다양한 분야에서 오늘날 까지도 계속 사용되고 있습니다. 이러한 기술들은 자연 과학에서 실험적인 접근법을 불러 일으켰습니다. 예를 들어 저항기의 전류 및 전압에 관한 옴의 법칙은 선형 모델로 완벽하게 설명됩니다.

중세 시대에도 수학자들은 예측에 대한 예리한 직관을 가지고 있었습니다. 예를 들어, [야곱 쾨벨 (1460-1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry)의 기하학책에서는 발의 평균 길이를 얻기 위해 성인 남성 발 16개의 평균을 사용했습니다.

![Estimating the length of a foot](../img/koebel.jpg)

그림 1.1은 이 평균이 어떻게 얻어졌는지를 보여줍니다. 16명의 성인 남성은 교회를 떠날 때 한 줄로 정렬하도록 요구받았습니다. 그런 다음 총 길이를 16으로 나누어 현재 1피트 금액에 대한 추정치를 얻습니다. 이 '알고리즘'은 나중에 잘못된 모양의 발을 다루기 위해 개선되었습니다 - 각각 가장 짧고 긴 발을 가진 2 명의 남성은 제외하고 나머지 발들에 대해서만 평균값을 계산합니다. 이것은 절사 평균 추정치의 초기 예 중 하나입니다.

통계는 실제로 데이터의 수집 및 가용성으로 시작되었습니다. 거장 중 한명인 [로널드 피셔 (1890-1962)](https://en.wikipedia.org/wiki/Ronald_Fisher)는 이론과 유전학의 응용에 크게 기여했습니다. 그의 알고리즘들 (예, 선형 판별 분석)과 수식들(예, Fisher 정보 매트릭스)은 오늘날에도 여전히 자주 사용되고 있습니다 (1936년에 발표한 난초(Iris) 데이터셋도 머신러닝 알고리즘을 설명하는 데 사용되기도 합니다).

머신러닝에 대한 두 번째 영향은 정보 이론 [(클로드 섀넌, 1916-2001)](https://en.wikipedia.org/wiki/Claude_Shannon) 과 [앨런 튜링 (1912-1954)](https://en.wikipedia.org/wiki/Alan_Turing)의 계산 이론에서 나왔습니다. 튜링은 그의 유명한 논문, [기계 및 지능 컴퓨팅, Computing machinery and intelligence](https://www.jstor.org/stable/2251299) (Mind, 1950년10월)에서, 그는  “기계가 생각할 수 있습니까?” 라는 질문을 제기했습니다. 그의 튜링 테스트로 설명 된 것처럼, 인간 평가자가 텍스트 상호 작용을 통해 기계와 인간의 응답을 구별하기 어려운 경우 ''기계는 지능적이다''라고 간주될 수 있습니다. 오늘날까지 지능형 기계의 개발은 신속하고 지속적으로 변화하고 있습니다.

또 다른 영향은 신경 과학 및 심리학에서 발견 될 수 있습니다. 결국, 인간은 분명히 지적인 행동을 합니다. 이러한 행동 및 이에 필요한 통찰력을 설명하고, 아마도 리버스 엔지니어링 할 수 있는지 여부를 묻는 것은 합리적입니다. 이를 달성하기위한 가장 오래된 알고리즘 중 하나는 [도널드 헤브 (1904-1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb) 에 의해 공식화 되었습니다.

그의 획기적인 책 [행동의 조직, The Organization of Behavior](http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf) (John Wiley & Sons, 1949) 에서, 그는 뉴런이 긍정적인 강화를 통해 학습할 것이라고 가정했습니다. 이것은 Hebbian 학습 규칙으로 알려지게 되었습니다. 그것은 Rosenblatt의 퍼셉트론 학습 알고리즘의 원형이며 오늘날 딥 러닝을 뒷받침하는 많은 stochastic gradient descent 알고리즘의 기초를 마련했습니다: 뉴럴 네트워크의 좋은 가중치를 얻기 위해 바람직한 행동은 강화하고 바람직하지 않은 행동을 감소시킵니다.

생물학적 영감으로부터 신경망(Neural Network)이라는 명칭이 탄생하였습니다. 알렉산더 베인(1873)과 제임스 셰링턴(1890)이 신경망 모델을 제안한 이래 한세기 이상 연구자들은 상호 작용하는 뉴런들의 네트워크와 유사한 계산 회로를 구현하려고 시도해 왔습니다. 시간이 지남에 따라 생물학과의 연관성은 느슨해 졌지만 신경망이라는 이름은 그대로 사용하고 있습니다. 오늘날 대부분의 신경망에서 찾을 수 있는 몇 가지 주요 핵심 원칙은 다음과 같습니다:

* '레이어'라고 불리우는 선형 및 비선형 처리 유닛들의 교차 구조
* 체인 규칙 (일명 역 전파)을 사용하여 한 번에 전체 네트워크의 매개 변수를 조정

초기 급속한 진행 이후, 뉴럴 네트워크의 연구는 1995년경부터 2005년까지 쇠퇴했습니다. 여러 가지 이유들이 있습니다. 우선 네트워크 학습은 매우 많은 계산량을 필요로합니다. 지난 세기 말경에 이르러 메모리는 충분해 졌지만 계산 능력이 부족했습니다. 두번째 이유는 데이터셋이 상대적으로 작았다는 것 입니다. 실제로 1932년에 나온 피셔(Fisher)의 '[난초(Iris) 데이터셋](https://en.wikipedia.org/wiki/Iris_flower_data_set)'은 각각 50장의 세 종류의 난초 사진으로 구성되어 있는데 알고리즘의 효능을 테스트하는 데 널리 사용되는 도구였습니다. 지금은 학습 예제에 흔히 쓰이는 60,000개의 손으로 쓴 숫자들로 구성된 MNIST조차 당시에는 너무 거대한 데이터로 취급되었습니다.

데이터 및 계산능력이 부족한 경우, 커널 방법, 의사 결정 트리 및 그래픽 모델과 같은 강력한 통계 도구쪽이 우수한 성능을 보여줍니다. 신경망과는 달리 이것들은 훈련하는 데 몇 주씩 걸리지 않으면서도 강력한 이론적 보장으로 예측 가능한 결과를 제공합니다.

## 딥 러닝으로의 길

시간이 흐르면서 월드 와이드 웹이 등장하고 수억명의 온라인 사용자에게 서비스를 제공하는 회사가 출현하였으며 저렴한 고품질의 센서, 데이터 저장 비용 감소([Kryder의 법칙](https://en.wikipedia.org/wiki/Mark_Kryder)), 그리고 특히 원래 컴퓨터 게임을 위해 설계된 GPU의 가격 하락([무어의 법칙](https://ko.wikipedia.org/wiki/%EB%AC%B4%EC%96%B4%EC%9D%98_%EB%B2%95%EC%B9%99))이 진행됨에 따라 많은 것들이 바뀌게 되었습니다. 갑자기 계산이 불가능한 것처럼 보이는 알고리즘과 모델이 의미를 지니게 된 것입니다(반대의 경우도 마찬가지입니다). 이것은 아래 표에 가장 잘 설명되어 있습니다.

|연대|데이터셋|메모리|초당 부동소수점 연산수|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (House prices in Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optical character recognition)|10 MB|10 MF (Intel 80486)|
|2000|10 M (web pages)|100 MB|1 GF (Intel Core)|
|2010|10 G (advertising)|1 GB|1 TF (NVIDIA C2050)|
|2020|1 T (social network)|100 GB|1 PF (NVIDIA DGX-2)|

RAM이 데이터의 증가와 보조를 맞추지 않은 것은 매우 분명합니다. 동시에 계산 능력의 향상은 사용 가능한 데이터의 증가를 앞서고 있습니다. 즉, 통계 모델은 메모리 효율성이 향상 되어야하고 (일반적으로 비선형을 추가하여 달성됨) 컴퓨팅 예산 증가로 인해 이러한 매개변수를 최적화하는 데 더 많은 시간을 할애해햐야 했습니다. 결국 머신러닝 및 통계에 적절한 방법은 선형 모델 및 커널 방법에서 딥 네트워크로 이동했습니다. 이것은 다층 퍼셉트론(Multilayer Perceptron) (예, 맥컬록 & 피츠, 1943), 컨볼루션 뉴럴 네트워크(Convolutional Neural Network) (Le Cun, 1992), Long Short Tem Memory (Hochreiter & Schmidhuber, 1997), Q-러닝 (왓킨스, 1989) 과 같은 딥 러닝의 많은 주류 이론들이 상당 시간 동안 휴면기에 있다가 최근 10년간 재발견된 이유 중에 하나입니다.

통계 모델, 응용 프로그램 및 알고리즘의 최근 발전은 때때로 캄브리아 폭발 (Cambrian Dexplosion) 에 비유되고 있습니다: 종의 진화가 급속히 진행되는 순간입니다. 실제로, 현 시점에서 가장 좋은 성과들(state of art)은 수십 년동안 만들어져온 오래된 알고리즘이 적용된 결과가 아닙니다. 아래 목록은 연구자들이 지난 10년간 엄청난 진전을 달성하는 데 도움이 된 아이디어의 극히 일부분 입니다.

* 드롭아웃(Drop out) [3] 과 같은 새로운 용량 제어 방법, 즉 학습 데이터의 큰 부분을 암기하는 위험 없이 비교적 큰 네트워크의 학습이 가능합니다. 이것은 학습 목적을 위해 무작위 변수로 가중치를 대체하여 네트워크 전체에 노이즈 주입 [4] 을 적용하여 달성되었습니다.
* 어텐션 메커니즘(Attention Mechanism)은 1 세기 이상 통계를 괴롭히던 두 번째 문제를 해결했습니다: 수를 늘리지 않고 시스템의 메모리와 복잡성을 증가시키는 방법, 학습 가능한 매개 변수. [5] 는 학습 가능한 포인터 구조로만 볼 수 있는 것을 사용하여 우아한 해결책을 찾았습니다. 즉, 전체 문장을 기억할 필요없이 (예: 고정 차원 표현의 기계 번역의 경우) 저장해야하는 모든 것은 번역 프로세스의 중간 상태에 대한 포인터였습니다. 이것은 문장을 생성하기 전에 모델이 더 이상 전체 문장을 기억할 필요가 없기 때문에 긴 문장의 정확도를 크게 높일 수 있었습니다. 
* 다단계 디자인 (예: Memory Network [6] 및 Neural programmer-interpreters [7]) 를 통해 통계 모델러가 추론에 대한 반복적인 접근법을 이용해 묘사할 수 있게 하였습니다. 이러한 기술은 딥 네트워크의 내부 상태가 반복적으로 변경가능하도록 하였습니다. 그에따라 추론 체인의 후속단계를 진행하고, 이는 프로세서가 계산을 위해 메모리를 수정할 수 있는 것과 유사합니다. 
* 또 다른 중요한 발전은 적대적 생성 신경망(Generative Adversarial Netoworks)의 발명 입니다[8]. 밀도추정 및 생성모델에 대한 전통적인 통계 방법은, 적절한 확률 분포와 그들로부터 샘플링에 대한 (종종 근사) 알고리즘을 찾는 데 초점을 맞추었습니다. 결과적으로, 이러한 알고리즘은 통계 모델에 내재 된 유연성 부족으로 인해 크게 제한되었습니다. GAN의 중요한 혁신은 샘플러를 다른 파라미터들을 가진 임의의 알고리즘으로 대체한 것입니다. 그런 다음 Discriminator(사실상 두 샘플 테스트)에 의해 가짜와 실제 데이터를 구분할 수 없도록 조정됩니다. 임의의 알고리즘을 이용하여 데이터를 생성하는 기술을 통해, 다양한 분야의 밀도추정이 가능해 졌습니다. 달리는 얼룩말[9] 과 가짜 유명인 얼굴 [10] 의 예는 이러한 발전에 대한 증명입니다.
* 대부분의 경우 단일 GPU는 학습에 필요한 많은 양의 데이터를 처리하기에는 부족합니다. 지난 10년간 병렬 분산 학습 알고리즘을 개발하는 능력은 크게 향상되었습니다. 확장 가능한 알고리즘을 설계할 때, 가장 큰 과제 중 하나는 딥 러닝 최적화, 즉 확률적 그래디언트 디센트의 핵심은 처리할 데이터에 비해 상대적으로 작은 미니배치(minibatches)에 의존한다는 점입니다. 이러한 미니배치(minibatch) 그래서, 하나의 작은 batch 때문에 GPU를 최대한 활용하지 못합니다. 따라서 1024개의 GPU로, batch당 32개의 이미지를 처리하는 미니배치 학습은, 한번의 병합된 32K개의 이미지 처리와 같습니다. 최근에는, 처음 Li [11] 에 이어 You[12] 와 Jia[13]가 최대 64K개의 데이터를 ResNet50으로 ImageNet을 학습 시간을 7분 미만으로 단축시켰습니다. 초기에 이 학습 시간 측정은 일 단위 이었습니다.
* 계산을 병렬화 하는 능력은, 시뮬레이션이 가능한 상황에서 강화학습(Reinforcement learning) 분야에 결정적인 기여를 하였습니다. 이것은 바둑, 아타리 게임, 스타크래프트, 그리고 물리 시뮬레이션(예를 들면 MuJoCo의 사용) 분야에서 컴퓨터가 인간능력에 다가서거나 넘어 설 수 있도록 하는데 필요한 중요한 발전을 이끌어 내었습니다. 실예로, Silver [18]는 AlphaGo가 어떻게 이것을 달성 했는지 설명하고 있습니다. 요컨대, 많은 양의 상태, 행동, 보상의 세가지 조합 데이터들을 사용할 수 있다면, 강화학습은 각각의 데이터들을 어떻게 연관시킬 수 있을 지, 매우 많은 양의 데이터로 시도해 볼수 있게 합니다. 시뮬레이션은 그런 방법을 제공합니다. 
* 딥러닝 프레임워크는 아이디어를 널리 퍼트리는데 중요한 역할을 했습니다. 손쉬운 모델링을 위한 프레임워크의 첫 번째 세대는 [Caffe](https://github.com/BVLC/caffe), [Torch](https://github.com/torch), [Theano](https://github.com/Theano/Theano) 입니다. 많은 영향력 있는 논문들이 이 도구를 이용해 작성 되었습니다. 지금에 이르러 이들은 TensorFlow[TensorFlow](https://github.com/tensorflow/tensorflow)로 대체 되었습니다. 고수준 API 인 [Keras](https://github.com/keras-team/keras), [CNTK](https://github.com/Microsoft/CNTK), [Caffe 2](https://github.com/caffe2/caffe2) 및  [Apache MxNet](https://github.com/apache/incubator-mxnet)도 이를 사용합니다. 3 세대 툴, 즉 딥러닝을 위한 명령형 툴은 틀림없이, 모델을 기술하기 위해 파이썬 NumPy와 유사한 구문을 사용하는 [Chainer](https://github.com/chainer/chainer)에 의해 주도될 것입니다. 이 아이디어는 [PyTorch](https://github.com/pytorch/pytorch)와 MxNet의 [Gluon API](https://github.com/apache/incubator-mxnet)에도 채택 되었습니다. 이 책에서는 MxNet의 Gluon API을 사용하였습니다.

학습을 위해 더 나은 툴을 만드는 연구자와 더 나은 신경망을 만들기 위한 통계모델러 간, 작업 시스템 분리는 많은 것들을 단순화 하였습니다. 한 예로, 2014년 카네기멜론대학의 머신러닝 박사과정 학생에게 선형 회귀 분석 모델을 학습시키는 것은 매우 중요한 과제 였습니다. 지금은 이 작업은 10줄이 안되는 코드로 완료 가능하고, 프로그래머들이 확실히 이해할수 있게 만들었습니다.

## 성공 사례

인공지능은 풀기 어려웠던 여러가지 문제들을 다른 방법으로 해결해 온 오래된 역사가 있습니다. 하나의 예로, 편지는 광학 문자 인식 기술을 이용해 정렬됩니다. 이 시스템은 90년대부터 사용되었습니다.(이것이 결국, 유명한 MINIST 및 USPS 필기 숫자셋의 출처 입니다.) 동시에 은행 예금의 수표책과 신청자의 신용 점수를 읽는 데에도 적용됩니다. 또 금융 거래에서는 자동으로 사기 여부를 체크 합니다. 페이팔, 스트라이프, 알리페이, 위챗, 애플, 비자, 마스터카드 등과 같은 많은 e-커머스 지불 시스템의 근간을 이룹니다. 체스 프로그램은 수십 년간 경쟁력을 유지해 왔습니다. 머신러닝은 인터넷상에서 검색, 추천, 개인화 및 랭킹을 제공해 왔습니다. 즉, 인공 지능과 머신러닝은 비록, 종종 시야에서 숨겨져 있지만, 널리 퍼져 있습니다.

최근에야 AI가 각광을 받고 있는데, 이는 대부분 이전에 다루기 어려운 문제들을 AI가 해결하고 있기 때문입니다.

* 애플의 시리 (Siri), 아마존의 알렉사 (Alexa), 구글의 조수 (assistant)와 같은 지능형 조수들은 말로 전달된 질문에 대해 합당한 정도의 정확도로 대답 할 수 있습니다. 여기에는 조명 스위치를 켜고 이발사와 약속을 잡고, 대화형 전화 지원을 제공하는 등의 일상적인 작업이 포함됩니다. 이것은 AI가 우리 삶에 영향을 미치는 가장 두드러진 사례들 일 것입니다.
* 디지털 조수의 핵심 요소는 말을 정확하게 인식 할 수있는 능력입니다. 점차적으로 이러한 시스템의 정확도는 특정 응용 분야에서 인간과 유사한 수준에 도달 할 정도로 올라갔습니다[14].
* 마찬가지로 객체 인식은 먼 길을왔다. 그림에서 개체를 추정하는 것은 2010 년에 상당히 어려운 작업이었습니다. ImageNet 벤치 마크 Lin 외. [15] 는 28% 의 상위 5 오류율을 달성했습니다. 2017 후 등. [16] 이 오류율을 2.25% 로 줄였습니다. 마찬가지로 놀라운 결과는 새를 식별하거나 피부암을 진단하기 위해 달성되었습니다. 
* 게임은 인간의 지능의 보루로 사용되었습니다. TDGammon에서 시작 [23], 시간 차이 (TD) 강화 학습, 알고리즘 및 계산 진행을 사용하여 주사위 놀이를 재생하는 프로그램은 광범위한 응용 프로그램을위한 알고리즘을 주도하고있다. 주사위 놀이와 달리 체스는 훨씬 더 복잡한 상태 공간과 일련의 행동을 가지고 있습니다. DeepBlue는 게리 카스파로프를 이길, 캠벨 등. [17], 게임 트리를 통해 대규모 병렬 처리, 특수 목적 하드웨어와 효율적인 검색을 사용하여. 이동은 거대한 상태 공간 때문에 여전히 더 어렵습니다. AlphaGo는 2015 년에 인간의 패리티를 달성, 실버 등. [18] 몬테 카를로 트리 샘플링과 결합 된 딥 러닝을 사용하여. 포커의 도전은 상태 공간이 크고 완전히 관찰되지 않는다는 것입니다 (우리는 상대방의 카드를 모릅니다). 천교는 효율적으로 구조화 된 전략을 사용하여 포커에서 인간의 성능을 초과; 브라운과 샌드 홀름 [19]. 이것은 게임의 인상적인 진전과 고급 알고리즘이 그들에게 중요한 역할을 했다는 사실을 보여줍니다. 
* AI의 진보의 또 다른 표시는 자율 주행 자동차와 트럭의 출현입니다. 아직 완전한 자율 주행에 도달한  것은 아니지만, [모멘타](http://www.momenta.com), [테슬라](https://www.tesla.com/), [엔비디아](http://www.nvidia.com), [모바일아이](http://www.mobileye.com), [Waymo.com](http://www.waymo.com) 등의 회사들은 적어도 부분적인 자율 주행을 구현하는 놀라운 진전을 이루어 냈습니다. 완전한 자율성을 너무 어렵게 만드는 것은 올바른 운전을 위해서는 규칙을 인식하고 추론하고 시스템에 통합하는 능력이 필요하다는 것 입니다. 현재 딥 러닝은 컴퓨터 영상 처리 측면의 문제를 해결하기 위해 주로 사용됩니다. 나머지는 엔지니어에 의해 많이 조정됩니다.

다시 말하지만, 위의 목록은 지능적인 것으로 간주되는 것과 머신러닝이 분야에서 일어난 놀라운 발견들의 극히 일부분에 불과합니다. 오늘날의 로봇 공학, 물류, 전산생물학, 입자 물리학, 천문학은 크던 작던 머신러닝의 발전의 혜택을 누리고 있습니다. 이제 머신러닝은 엔지니어와 과학자를 위한 범용적인 도구가 되어가고 있는 것 입니다.

종종 AI 종말론이나 인공 지능 특이성에 대한 질문들이 비기술적인 기사에서 제기되곤 합니다. 머신러닝 시스템이 지각을 갖게 될 것이고, 그것을 만든 프로그래머와는 독립적으로 인간의 생활에 직접적인 영향을 끼칠 것들을 결정할 것이라는 것을 두려워합니다. 하지만 이미 AI는 인간의 삶에 영향을 미치고 있습니다. 신용도가 자동으로 평가되고, 오토파일럿(autopilot)은 자동차를 안전하게 운전할 수 있게 해 주며, 통계 데이터를 입력을 사용해서 보석 허용 여부를 결정하고 있습니다. 조금 더 친근한 사례로 우리는 Alexa에게 커피 머신을 켜달라고 요청할 수 있으며, Alexa가 장치에 연결되어 있다면 요청을 수행할 수 있습니다.

다행히도 우리는 인간 창조자를 노예로 만들거나 커피를 태울 준비가 된 지각 있는 AI 시스템과는 거리가 멀었습니다. 첫째, AI 시스템은 특정 목표 지향적 방식으로 설계, 학습, 배포됩니다. 그들의 행동은 범용AI에 대한 환상을 줄 수 있지만, 어디까지나 현재 인공지능 디자인의 기초는 규칙, 휴리스틱, 통계 모델의 조합입니다. 둘째, 아직까지는 일반적인 일을 수행하면서 스스로 개선하고, 스스로에 대해서 사고, 스스로의 아키텍처를 개선확장하고 개선하는 일반적인 인공지능을 위한 도구는 존재하지 않습니다.

훨씬 더 현실적인 관심사는 AI가 일상생활에서 어떻게 사용되는지입니다. 트럭 운전사 및 상점 보조자가 수행하는 사소한 일들이 자동화될 수 있고 자동화될 가능성이 있습니다. 농장 로봇은 유기 농업 비용을 줄일 수 있있고, 또한 수확 작업을 자동화할 것입니다. 산업 혁명의 이 단계는 사회의 많은 이들의 삶에 있어서 중대한 변화를 가져올 것입니다. 트럭 운전사와 매장 점원은 많은 주에서 가장 일반적인 직업중 하나입니다. 게다가 통계 모델이 부주의하게 적용되면 인종적, 성별 또는 연령 편견이 발생할 수 있습니다. 이러한 알고리즘이 세심한 주의를 가지고 사용되는지 확인하는 것이 중요합니다. 이것은 인류를 멸망시킬 수 있는 악의적인 초지능의 탄생에 대해 걱정하는 것보다 훨씬 더 현실적이고 중요한 문제입니다.

## 주요 요소들

머신러닝은 데이터를 사용하여 예제 간의 변환을 학습합니다. 예를 들어 숫자 이미지는 0에서 9 사이의 정수로 변환되고, 오디오는 텍스트(음성 인식)로 변환되고, 텍스트는 다른 언어의 텍스트로 변환되거나(기계 번역), 머그샷이 이름으로 변환됩니다(얼굴 인식). 그렇게 할 때, 알고리즘이 데이터를 처리하는 데 적합한 방식으로 데이터를 표현해야 하는 경우가 종종 있습니다. 이러한 특징 변환(feature transformation)의 정도는 표현 학습을 위한 수단으로 딥 러닝을 언급하는 이유로서 종종 사용됩니다. 사실, 국제 학습 표현 회의(the International Conference on Learning Representations)의 명칭은 이것으로부터 유래합니다. 동시에 머신러닝은 통계(특정 알고리즘이 아닌 매우 큰 범위의 질문까지)와 데이터 마이닝(확장성 처리)을 똑같이 사용합니다.

현기증 나는 알고리즘 및 응용 프로그램 집합으로 인해 딥 러닝을 위한 성분이 무엇인지 *구체적으로* 평가하기가 어렵습니다. 이것은 피자에 필요한 재료를 고정시키는 것만큼 어렵습니다. 거의 모든 구성 요소는 대체 가능합니다. 예를 들어 다층 퍼셉트론이 필수 성분이라고 가정할 수 있습니다. 그러나 convolution 만 사용하는 컴퓨터 비전 모델이 있습니다. 다른 것들은 시퀀스 모델만 사용하기도 합니다.

틀림없이 이러한 방법에서 가장 중요한 공통점은 종단간(end-to-end) 학습을 사용하는 것입니다. 즉, 개별적으로 튜닝된 구성 요소를 기반으로 시스템을 조립하는 대신 시스템을 구축한 다음 성능을 공동으로 튜닝합니다. 예를 들어, 컴퓨터 비전 과학자들은 머신러닝 모델을 구축하는 과정과 특징 엔지니어링 프로세스를 분리하곤 했습니다. Canny 에지 검출기 [20] 와 Lowe의 SIFT 특징 추출기 [21] 는 이미지를 형상 벡터에 매핑하기 위한 알고리즘으로 10여 년간 최고로 통치했습니다. 불행히도 알고리즘에 의해 자동으로 수행 될 때 수천 또는 수백만 가지 선택에 대한 일관된 평가와 관련하여 인간이 독창성으로 성취 할 수있는 많은 것들이 있습니다. 딥 러닝이 적용되었을 때, 이러한 특징 추출기는 자동으로 튜닝된 필터로 대체되어 뛰어난 정확도를 달성했습니다.

마찬가지로 자연 언어 처리에서 Salton과 McGill [22] 의 bag-of-words 모델은 오랫동안 기본으로 선택되었습니다. 여기서 문장의 단어는 벡터로 매핑되며 각 좌표는 특정 단어가 발생하는 횟수에 해당합니다. 이것은 단어 순서 ('개가 사람을 물었다' 대 '사람이 개를 물었다') 또는 구두점 ('먹자, 할머니' 대 '할머니를 먹자') 을 완전히 무시합니다. 불행히도, 더 나은 특징을 *수동으로* 엔지니어링하는 것은 다소 어렵습니다. 반대로 알고리즘은 가능한 특징(feature) 설계의 넓은 공간을 자동으로 검색 할 수 있습니다. 이것은 엄청난 진전을 이끌어 왔습니다. 예를 들어 의미상 관련성이 있는 단어 임베딩은 벡터 공간에서 '베를린 - 독일 + 이탈리아 = 로마' 형식의 추론을 허용합니다. 다시 말하지만, 이러한 결과는 전체 시스템의 end-to-end 학습을 통해 달성됩니다.

End-to-end 학습 외에도 두 번째로 중요한 것은 파라미터 기반의 통계 설명에서 완전 비파라미터 기반의 모델로의 전환을 경험하고 있다는 것입니다. 데이터가 부족한 경우, 유용한 모델을 얻기 위해서는 현실에 대한 가정을 단순화하는 데 의존해야합니다 (예, 스펙트럼 방법을 통해). 데이터가 풍부하면 현실에 더 정확하게 맞는 비파라미터 기반의 모형으로 대체될 수 있습니다. 어느 정도, 이것은 컴퓨터의 가용성과 함께 이전 세기 중반에 물리학이 경험한 진전과 비슷합니다. 전자가  어떻게 동작하는지에 대한 파라메트릭 근사치를 직접 해결하는 대신, 이제 연관된 부분 미분 방정식의 수치 시뮬레이션에 의존 할 수 있습니다. 이것은 설명 가능성을 희생시키면서 종종 훨씬 더 정확한 모델을 이끌어 냈습니다. 

예를 들어 Generative Aversarial Networks가 있습니다. 그래픽 모델이 적절한 확률적 공식 없이도 데이터 생성 코드로 대체됩니다. 이것은 현혹적으로 현실적으로 보일 수 있는 이미지의 모델을 이끌어 냈는데 이는 오랜 시간 동안 너무 어려운 것으로 여겨졌왔던 것입니다.

이전 작업의 또 다른 차이점은 볼록하지 않은 비선형 최적화 문제(nonconvex nonlinear optimization problem)를 다루면서 차선책 솔루션을 받아들이고, 이를 증명하기 전에 시도하려는 의지입니다. 통계적 문제를 다루는 새로운 경험주의와 인재의 급속한 유입은 실질적인 알고리즘의 급속한 발전으로 이어졌습니다 (많은 경우에도 불구하고 수십 년간 존재했던 도구를 수정하고 다시 발명하는 대신). 

마지막으로 딥 러닝 커뮤니티는 학술 및 기업 경계를 넘어 도구를 공유하는 것을 자랑으로 하고 있으며, 많은 우수한 라이브러리, 통계 모델 및 학습된 네트워크를 오픈 소스로 공개합니다. 이 과정을 구성하는 노트북은 배포 및 사용이 자유됩다는 것이 이러한 정신입니다. 우리는 모든 사람들이 딥 러닝에 대해 배울 수 있는 접근의 장벽을 낮추기 위해 열심히 노력했으며 독자가 이것의 혜택을 누릴 수 있기를 바랍니다.

## 요약

* 머신러닝은 컴퓨터 시스템이 어떻게 데이터를 사용하여 성능을 향상시킬 수 있는지 연구합니다. 통계, 데이터 마이닝, 인공 지능 및 최적화의 아이디어를 결합합니다. 종종 인위적으로 지능형 솔루션을 구현하는 수단으로 사용됩니다. 
* 머신러닝의 클래스로서 표현 학습은 데이터를 나타내는 적절한 방법을 자동으로 찾는 방법에 초점을 맞춥니다. 이것은 종종 학습된 변환의 진행에 의해 성취됩니다. 
* 최근 진전의 대부분은 값싼 센서와 인터넷 규모 응용 프로그램에서 발생하는 풍부한 데이터와 주로 GPU를 통한 계산의 상당한 진전으로 인해 트리거되었습니다. 
* 전체 시스템 최적화가 핵심입니다. 구성 요소를 사용하여 좋은 성능을 얻을 수 있습니다. 효율적인 딥 러닝 프레임워크의 가용성으로 인해 이 프레임워크의 설계와 구현이 훨씬 쉬워졌습니다.

## 문제

1. 현재 작성중인 코드의 어느 부분이 '학습' 될 수 있습니까? 즉, 코드에서 만들어진 디자인 선택을 학습하고 자동으로 결정함으로써 개선 될 수 있습니까? 코드에 휴리스틱 디자인 선택이 포함되어 있습니까? 
1. 어떤 문제가 발생하는지이를 해결하는 방법에 대한 많은 예가 있지만 자동화하는 구체적인 방법은 없습니다. 이들은 딥 러닝을 사용하기위한 주요 후보 일 수 있습니다. 
1. 새로운 산업 혁명으로 인공 지능의 발전을 보고, 알고리즘과 데이터의 관계는 무엇인가? 증기 엔진과 석탄과 비슷합니까 (근본적인 차이점은 무엇입니까)? 
1. End-to-end 학습 접근 방식을 어디에서 적용할 수 있습니까? 물리학? 엔지니어링? 경제학? 
1. 왜 인간의 뇌처럼 구조화된 딥 네트워크를 만들고 싶습니까? 장점은 무엇입니까? 왜 그렇게하고 싶지 않습니까 (마이크로 프로세서와 뉴런의 주요 차이점은 무엇입니까)?

## 참고문헌

[1] Turing, A. M. (1950). Computing machinery and intelligence. Mind, 59(236), 433.

[2] Hebb, D. O. (1949). The organization of behavior; a neuropsychological theory. A Wiley Book in Clinical Psychology. 62-78.

[3] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1), 1929-1958.

[4] Bishop, C. M. (1995). Training with noise is equivalent to Tikhonov regularization. Neural computation, 7(1), 108-116.

[5] Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.

[6] Sukhbaatar, S., Weston, J., & Fergus, R. (2015). End-to-end memory networks. In Advances in neural information processing systems (pp. 2440-2448).

[7] Reed, S., & De Freitas, N. (2015). Neural programmer-interpreters. arXiv preprint arXiv:1511.06279.

[8] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., … & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).

[9] Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint.

[10] Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017). Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196.

[11] Li, M. (2017). Scaling Distributed Machine Learning with System and Algorithm Co-design (Doctoral dissertation, PhD thesis, Intel).

[12] You, Y., Gitman, I., & Ginsburg, B. Large batch training of convolutional networks. ArXiv e-prints.

[13] Jia, X., Song, S., He, W., Wang, Y., Rong, H., Zhou, F., … & Chen, T. (2018). Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes. arXiv preprint arXiv:1807.11205.

[14] Xiong, W., Droppo, J., Huang, X., Seide, F., Seltzer, M., Stolcke, A., … & Zweig, G. (2017, March). The Microsoft 2016 conversational speech recognition system. In Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on (pp. 5255-5259). IEEE.

[15] Lin, Y., Lv, F., Zhu, S., Yang, M., Cour, T., Yu, K., … & Huang, T. (2010). Imagenet classification: fast descriptor coding and large-scale svm training. Large scale visual recognition challenge.

[16] Hu, J., Shen, L., & Sun, G. (2017). Squeeze-and-excitation networks. arXiv preprint arXiv:1709.01507, 7.

[17] Campbell, M., Hoane Jr, A. J., & Hsu, F. H. (2002). Deep blue. Artificial intelligence, 134 (1-2), 57-83.

[18] Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., … & Dieleman, S. (2016). Mastering the game of Go with deep neural networks and tree search. Nature, 529 (7587), 484.

[19] Brown, N., & Sandholm, T. (2017, August). Libratus: The superhuman ai for no-limit poker. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence.

[20] Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6), 679-698.

[21] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2), 91-110.

[22] Salton, G., & McGill, M. J. (1986). Introduction to modern information retrieval.

[23] Tesauro, G. (1995), Transactions of the ACM, (38) 3, 58-68

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2310)

![](../img/qr_deep-learning-intro.svg)
