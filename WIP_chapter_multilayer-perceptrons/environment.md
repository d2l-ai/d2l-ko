# 환경에 대한 고려

*지금까지 우리는 데이터가 어디서 왔는지 모델이 어떻게 배포되는지에 대해서는 걱정하지 않았습니다. 하지만, 이것들을 고려하지 않는 것은 문제가 됩니다. 실패한 많은 머신 러닝 배포들의 원인을 추적해보면 이런 상황이 원인이 됩니다. 이 절에서는 이 상황을 초기에 발견하고, 완화하는 방법을 알아봅니다. 상황에 따라서, 정확한 데이터를 사용하면되는 다소 간단한 문제일 수 있기도 하지만, 강화학습 시스템을 만드는 것과 같이 어려운 문제이기도 합니다.*

So far, we have worked through a number of hands-on implementations
fitting machine learning models to a variety of datasets.
And yet, until now we skated over the matter 
of where are data comes from in the first place,
and what we plan to ultimately *do* with the outputs from our models.
Too often in the practice of machine learning, 
developers rush ahead with the development of models
tossing these fundamental considerations aside.

지금까지 우리는 여러 종류의 데이터셋을 사용해서 머신러닝 모델을 학습시키는 것에 대한 핸즈온 구현을 살펴봤습니다. 하지만 지금까지는 데이터가 처음에 어디에서 왔는지, 모델의 결과를 사용해서 진짜로 무엇을 *할 것*인지에 대한 이야기는 가볍게만 해왔습니다. 머신러닝의 사용할 때 개발자들은 자주 모델 개발에 열중하지만 이 기본적인 고려 사항들을 뒷전에 둡니다.

Many failed machine learning deployments can be traced back to this situation. 
Sometimes the model does well as evaluated by test accuracy
only to fail catastrophically in the real world
when the distribution of data suddenly shifts.
More insidiously, sometimes the very deployment of a model
can be the catalyst which perturbs the data distribution.
Say for example that we trained a model to predict loan defaults,
finding that the choice of footware was associated with risk of default 
(Oxfords indicate repayment, sneakers indicate default).
We might be inclined to thereafter grant loans 
to all applicants wearing Oxfords
and to deny all appplicants wearing sneakers.
But our ill-conceived leap from pattern recognition to decision-making 
and our failure to think critically about the environment
might have disastrous consequences. 
For starters, as soon as we began making decisions based on footware,
customers would catch on and change their behavior. 
Before long, all applicants would be wearing Oxfords, 
and yet there would be no coinciding improvement in credit-worthiness.
Think about this deeply because similar issues abound in the application of machine learning: by introducing our model-based decisions to the environnment,
we might break the model.

많은 실패한 머신러닝 배포들은 이런 상황으로 추적될 수 있습니다. 종종 모델들이 테스트 정확도를 이용한 검증은 좋게 보이지만, 데이터 분포가 갑자기 변화는 실제 상황에서 급격하게 실패합니다. 어떤 경우에는 모르는 사이에 모델의 배포 자체가 데이터 분포를 변동시키는 촉매가 될 수도 있습니다. 예를 들어 신발 선택이 채무 불이행의 위험과 연관된다는 것을 기반으로 대출 불이행을 예측하는 모델을 학습시켰습니다. (옥스포드(신발의 종류)는 상환을 나타내고, 스키커즈는 채무 불이행을 나타냅니다) 우리는 옥스포드를 신은 모든 신청자들에게는 대출을 허가하고, 스니커즈를 신은 모든 신청차들은 거부하게 됩니다. 하지만, 패턴 인식부터 의사 결정까지의 의도하지 않는 도약과 환경에 대한 비판적으로 생각하지 못한 것이 비참한 결과를 초래할 수 있습니다. 우리가 신발의 종류에 근거해서 결정을 내리기 시작하자마자, 고객들은 이를 알아내고서 그들의 행동을 바꿀 것입니다. 오래되지 않아 모든 지원자들은 옥스포드를 신고 올 것이고, 신용 가치에 대한 일치하는 향상은 없을 것입니다. 머신러닝의 응용에도 비슷한 이슈가 일어나기 때문에 이 예제를 깊게 생각해 봐야합니다: 즉, 우리의 모델 기반의 결정을 어떤 환경에 적용하는 것으로 모델을 망가트릴 수도 있습니다.

In this chapter, we describe some common concerns
and aim to get you started acquiring the critical thinking 
that you will need in order to detect these situations early,
mitigate the damage, and use machine learning responsibly. 
Some of the solutions are simple (ask for the 'right' data) 
some are technically difficult (implement a reinforcement learning system),
and others require that we entre the realm of philosophy
and grapple with difficult questions concerning ethics and informed consent.

우리는 이 장에서 공통적인 우려점들을 설명하고, 여러분이 이런 상황을 조기에 탐지하고, 피해를 방지하고, 머신러닝을 책임있게 사용하도록 해주는 비판적인 생각법을 습득할 수 있도록 돕는 것이 목표입니다. 어떤 해결책들은 간단하고('정확한' 데이터를 요구함), 어떤 것들은 기술적으로 어렵고(강화학습 시스템 구현), 그리고 다른 것들은 철학의 영역을 다루거나 윤리 및 정보에 근거한 동의와 관련된 어려운 문제들이 될 수 있습니다.


## 분포 변화(Distribution Shift)

To begin, we return to the observartional setting,
putting aside for now the impacts of our actions 
on the environment.
In the following sections, we take a deeper look 
at the various ways that data distributions might shift,
and what might be done to salvage model performance. 
From the outset, we should warn that if 
the data-generating distribution $p(\mathbf{x},y)$
can shift in arbitrary ways at any point in time,
then learning a robust classifier is impossible.
In the most pathological case, if the label defnitions themselves
can change at a moments notice: if suddenly
what we called "cats" are now dogs
and what we previously called "dogs" are now in fact cats,
without any perceptible change in the distribution of inputs $p(\mathbf{x})$,
then there is nothing we could do to detect the change 
or to correct our classifier at test time.
Fortunately, under some restricted assumptions 
on the ways our data might change in the future,
principled algorithms can detect shift and possibly even
adapt, achieving higher accuracy 
than if we naively continued to rely on our original classifer. 

우선은 우리의 행동이 환경에 미칠 영향은 논외로 하고, 관찰 설정으로 돌아가겠습니다. 아래 절들에서 우리는 데이터 분포가 바뀔 수 있는 여러 방법들과 모델의 성능을 유지시키기 위해서 해야하는 것들에 대해서 자세히 보겠습니다. 데이터 생성 분포  $p(\mathbf{x},y)$ 가 언제든이 임의의 방향으로 변화될 수 있다면, 견고한 분류기를 학습시키는 것은 불가능하다는 것을 처음부터 경고해야합니다. 예를 들어 만약 레이블 정의 자체가 입력의 분포  $p(\mathbf{x})$가 인지할 수 있는 변화 없이 어떤 순간 바뀔 수 있다면, 즉 우리가 "고양이"라고 불렀던 것이 개가 되고, 이전에 "개"라고 부르던 것이 사실 지금은 고양이라면, 변화를 감지하거나 테스트 시점에 분류기를  고칠기 위해서 할 수 있는 일이 없을 것입니다. 운이 좋게도 우리의 데이터가 미래에 바뀌는 방법에 대한 제한된 가정하에서 주요 알고리즘은 원래 분류기에 계속 의존하는 것이 아니라 변화를 감지하고 높은 정확도를 달성하면서 적응을 할 수 있습니다.

## 공변량 변화(covariate shift)

*이해하는 것은 쉽지만, 놓치기 쉬운 문제가 있습니다. 강아지와 고양을 구분하는 문제를 생각해봅시다. 학습 데이터는 다음과 같이 주어졌습니다.*

One of the best-studied forms of distribution shift is *covariate shift*.
Here we assume that although the distirbution of inputs may change over time,
the labeling function, i.e., the conditional distribution $p(y|\mathbf{x})$
does not change. 
While this problem is easy to understand 
its also easy to overlook it in practice. 
Consider the challenge of distinguishing cats and dogs. 
Our training data consists of images of the following kind:

분포 변화에 대해서 가장 잘 연구된 형태 중 하나는 *공변량 변화(covariate shift)* 입니다. 여기서는 입력 분포가 시간이 지남에 따라서 변화할지라도 조건 분포  $p(y|\mathbf{x})$인 레이블 함수는 변화지 않는다고 가정합니다. 이런 문제를 이해하기는 쉽지만 현실에서는 관과되기도 쉽습니다. 고양이와 개를 구분하는 것에 대한 문제를 생각해 봅시다. 우리의 학습 데이터는 다음과 같은 종의 이미지로 구성되어 있습니다.

|고양이|고양이|강아지|강아지|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat3.jpg)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|

테스트에서는 다음 그림을 분류하도록 요청 받습니다.

|고양이|고양이|강아지|강아지|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat-cartoon1.png)|![](../img/cat-cartoon2.png)|![](../img/dog-cartoon1.png)|![](../img/dog-cartoon2.png)|

당연하게 이것은 잘 작동하지 않습니다. 학습 데이터는 실제 사진으로 구성되어 있지만, 테스트셋은 만화 그림으로 되어있습니다. 색상도 현실적이지 않습니다. 새로운 도메인에 어떻게 적용할지 계획이 없이 테스트셋과 다른 데이터로 학습을 시키는 것은 나쁜 아이디어입니다. 불행하게도 이것은 흔히 빠지는 함정입니다. *통계학자들은 이것을  **공변량 변화(covariate shift)** 라고 합니다. 즉, 공변량(covariates)(학습 데이터)의 분포가 테스트 데이터의 분포가 다른 상황을 의미합니다. 수학적으로 말하자면,  $p(x)$ 는 변화하는데,  $p(y|x)$ 는 그대로 있는 경우를 의미합니다.*

Statisticians call this *covariate shift*
because the root of the problem owed to 
a shift in the distribution of features (i.e., of *covariates*).
Mathematically, we could say that $p(\mathbf{x})$ changes 
but that $p(y|\mathbf{x})$ remains unchanged.
Although its usefulness is not restricted to this setting,
when we believe $\mathbf{x}$ causes $y$, covariate shift is usually 
the right assumption to be working with.

이 문제의 근원은 특성들 또는 공변량(covariate)들의 분포의 이동이기 때문에, 통계학자들은 이를 *공변량 변화(covariate shift)* 라고 부릅니다. 수학적으로 설명하면, $p(\mathbf{x})$ 는 변화하지만 $p(y|\mathbf{x})$ 는 변화하지 않는다고 말할 수 있습니다. 이 개념은 이 세팅에만 국한하지 않지만, $\mathbf{x}$ 가 $y$ 에 대한 원인이 될 때,  보통은 공변량 변화가 우리가 다룰 적당한 가정입니다.

### 레이블 변화(Label Shift)

The converse problem emerges when we believe that what drives the shift
is a change in the marginal distribution over the labels $p(y)$
but that the class-conditional distributions are invariant $p(\mathbf{x}|y)$.
Label shift is a reasonable assumption to make 
when we beleive that $y$ causes $\mathbf{x}$. 
For example, commonly we want to predict a diagnosis given its manifestations.
In this case we beleive that the diagnosis causes the manifestations,
i.e., diseases cause symptoms.
Sometimes the label shift and covariate shift assumptions 
can hold simultaneously.
For example, when the true labeling function is deterministic and unchanging,
then covariate shift will always hold, 
including if label shift holds too.
Interestingly, when we expect both label shift and covariate shift hold,
it's often advantageous to work with the methods 
that flow from the label shift assumption.
That's because these methods tend to involve manipulating objects 
that look like the label, 
which (in deep learning) tends to be comparatively easy
compared to working with the objects that look like the input,
which tends (in deep learning) to be a high-dimensional object.

변화를 일으키는 것이 레이블에 대한 주변확률분포(marginal distribution), $p(y)$,의 변화이고 하지만 클래스-조건 분포, $p(\mathbf{x}|y)$,는 변화지 않는다고 믿는 경우에는 반대의 문제가 발생합니다.  $y$ 가 $\mathbf{x}$ 의 원인이라고 믿는 경우에 레이블 변화는 합리적인 가정입니다. 예를 들면, 보통은 증상이 주어졌을 때 진단을 예측하기를 원합니다. 이 경우, 우리는 진단이 증상에 영향을 미친다고 믿습니다. 즉, 병이 증상의 원인이 된다고 믿습니다. 종종 레이블 변화와 공변량 변화에 대한 가정은 동시에 성립하기도 합니다. 예를 들어, 진짜 레이블을 할당하는 함수가 결정적이고 변화하지 않을 때는 공변량 변화는 항상 유지됩니다. 흥미롭게도 레이블 변화와 공변량 변화가 동시에 유지되는 것을 예상하는 경우에는 레이블 변화에 대한 방법들을 사용하는 것이 때로는 더 유용합니다. 그 이유는 이런 방법들은 레이블처럼 보이는 객체를 조작하는 경향이 있고, 딥러닝에서는  입력처럼 보이는 객체들을 다루는 것보다는 상대적으로 쉽기 때문입니다. 왜냐하면, 딥러닝에서 입력은 고차원 객체인 경향이 있기 때문입니다.

## 개념 변화(concept shift)

One more related problem arises in *concept shift*, 
the situation in which the very label definitions change. 
This sounds weird—after all, a *cat* is a *cat*.
Indeed the definition of a cat might not change, 
but can we say the same about soft drinks?
It turns out that if we navigate aroudn the United States,
shifting the source of our data by geography,
we'll find considerable concept shift regaring 
the definition of even this simple term:

관련된 또 다른 문제는 *개념 변화(concept shift)* 로 인해서 발생합니다. 이 상황은 레이블에 대한 정의 자체가 변화나는 경우입니다. 이것은 이상하게 들릴 것입니다 - 결국 *고양이* 는 *고양이* 입니다. 실제로는 고양이에 대한 정의가 바뀌지는 않습니다. 하지만, 소프트 드링크의 경우도 같을까요? 미국 전역을 돌아다니면서 우리의 데이터의 소스를 지역에 따라서 변화시켜보면, 이 간단한 용어 정의에 대한 개념 변화가 상당하다는 것을 알아낼 수 있습니다.

![](../img/popvssoda.png)

If we were to build a machine translation system, 
the distribution $p(y|x)$ might be different 
depending on our location. 
This problem can be tricky to spot. 
A saving grace is that often the $p(y|x)$ only shifts gradually. 
Before we go into further detail and discuss remedies, 
we can discuss a number of situations where covariate and concept shift 
may not be so obvious.

우리가 만약 기계번역 시스템을 만든 다면, 분산  $p(y|x)$ 은 우리의 지역에 따라서 달라질 것 입니다. 이 문제를 찾아내는 것은 까다롭습니다. 다행인 것은 많은 경우에  $p(y|x)$ 는 점진적으로만 변화한다는 것입니다. 더 자세한 내용과 해결 방법을 알아보기 전에, 공변량 변화와 개념 변화가 명확해 보이지 않는 다양한 상황에 대해서 설명하겠습니다.

## 예제

### 의학 분석

암을 진단하는 알고리즘을 설계하는 것을 상상해보세요. 건강한 사람과 아픈 사람의 데이터를 얻은 후, 알고리즘을 학습시킵니다. 학습된 모델이 높은 정확도를 보여주면서 잘 동작합니다. 당신은 이제 의료 분석 분야에서 성공적인 경력을 시작할 수 있다고 판단합니다. 하지만 너무 이릅니다.

많은 것들이 잘못될 수 있습니다. 특히, 학습에 사용한 분포와 실제 만나게 될 분포는 상당히 다를 수 있습니다. 실제로 알랙스가 수년 전에 스타트업 회사를 컨설팅하면서 겪었던 일입니다. 이 회사는 주로 나이 많은 남성에서 발견되는 질병에 대한 혈액 테스트를 개발하고 있었습니다. 이를 위해서 환자들로부터 상당히 많은 샘플을 수집할 수 있었습니다. 하지만, 윤리적인 이유로 건강한 남자의 혈액 샘플을 구하는 것은 상당히 어려웠습니다. 이를 해결하기 위해서, 캠퍼스의 학생들에게 혈액을 기증 받아서 테스트를 수행했습니다. 그리고, 그 회사는 나에게 질병을 분류하는 모델을 만드는 것에 대한 도움을 요청했습니다. 나는 그들에게 거의 완벽에 가까운 정확도로 두 데이터셋을 구분하는 것은 아주 쉽다고 알려줬습니다. 결국, 모든 테스트 대상은 나이, 호르몬 레벨, 신체 활동, 식이 상태, 알콜 섭취, 그리고 질병과 연관이 없는 아주 많은 요소들이 달랐습니다. 하지만, 이는 실제 환자의 경우와 차이가 있습니다. 그들의 샘플링 절차는 *소스*와 *타겟*  분포 사이에 극심한 공변량 변화가 발생하기 쉽게 만들었습니다. 그리고 그것은 전통적인 방법으로는 고쳐질 수 없었습니다. 달리 말하면, 학습 데이터와 테스트 데이터가 너무나 달라서 어떤 유용한 일도 할 수 없었고, 결국 상당히 많은 돈을 낭비만 했습니다.

### 자율 주행 자동차

자율 주행차를 위한 머신러닝 시스템을 만들고자 하는 한 회사가 있다고 하겠습니다. 도로를 탐지하는 것이 중요한 컴포넌트 중에 하나입니다. 실제 답을 다는 것이 너무 비싸기 때문에, 게임 렌더링 엔진을 사용해서 생성한 데이터를 추가 학습 데이터로 사용하기로 했습니다. 이렇게 학습된 모델은 렌더링 엔진으로 만들어진 '테스트 데이터'에는 잘 동작했습니다. 하지만, 실제 차에서는 재앙이었습니다. 이유는 렌더링 된 도로가 너무 단순한 텍스처를 사용했기 때문 였습니다. 더 중요한 것은 모든 도로 경계가 같은 텍스터로 렌더되었기에, 도로 탐지기는 이 '특징'을 너무 빨리 배워버렸습니다.

미국 군대에서 숲 속에서 있는 탱크를 탐지하는 것을 하려고 했을 때도 비슷한 문제가 발생했습니다. 탱크가 없는 숲의 항공 사진을 찍고, 탱크를 숲으로 몰고 가서 다른 사진을 찍었습니다. 이렇게 학습된 분류기는 아주 완벽하게 동작했습니다. 하지만 불행히도 이 모델은 그늘이 있는 나무들과 그늘이 없는 나무를 구분하고 있었습니다. 이유는 첫번째 사진은 이른 아침에 찍었고, 두번째 사진은 정오에 찍었기 때문이었습니다.

### 정적이지 않은 분포(nonstationary distribution)

더 알아내기 힘든 상황은 분포가 천천히 변화하는 상황에서 모델을 적절하게 업데이트를 하지 않을 때 일어납니다. 전형적인 몇 가지 사례가  있습니다.

* 광고 모델을 학습시킨 후, 자주 업데이트하는 것을 실패한 경우. (예를 들면, iPad 라는 새로운 디바이스가 막 출시된 것을 반영하는 것을 잊은 경우)
* 스팸 필더를 만들었습니다. 이 스팸 필터는 우리가 봤던 모든 스팸을 모두 잘 탐지합니다. 하지만, 스팸을 보내는 사람들이 이를 알고 이전에 봐왔던 것과는 다른 새로운 메시지를 만듭니다.
* 상품 추천 시스템을 만들었습니다. 겨울 동안에는 잘 동작합니다. 하지만, 크리스마스가 지난 후 오랫동안 산타 모자를 계속 추천하고 있습니다.

### 더 많은 예제들

* 얼굴 인식기를 만듭니다. 모든 밴치마크에서 잘 동작합니다. 하지만, 테스트 데이터에서는 그렇지 못합니다. 실패한 이미지를 보니 이미지 전체를 얼굴이 차지하는 클로즈업 사진들입니다.
* 미국 마켓을 위한 웹 검색 엔진을 만들어서 영국에 배포하고 싶습니다.
* We train an image classifier by compiling a large dataset where each among a large set of classes is equally represented in the dataset, say 1000 categories, represented by 1000 images each. Then we deploy the system in the real world, where the actual label distirbution of photographs is decidedly non-uniform.
* 큰 데이터셋을 사용해서 이미지 분류기를 학습시킵니다. 이 데이터셋의 각 클래스는 동일한 개수의 샘플을 갖고 있습니다. 즉, 1000개 카테고리는 각각 1000장의 이미지를 가지고 있습니다. 학습 후, 실제 환경에 배포하는데, 사진의 실제 레이블 분포는 명확하게 균일하지 않습니다.

*요약하면, 학습 데이터의 분포와 테스트 데이터의 분포가 다른 다양한 사례가 있습니다. 어떤 경우에는 운이 좋아서 covariate shift가 있음에도 불구하고 모델이 잘 동작할 수 있습니다. 자 지금부터 원칙적인 해결 전략에 대해서 이야기하겠습니다. 경고 - 약간의 수학과 통계가 필요합니다.*

In short, there are many cases where training and test distributions 
$p(\mathbf{x}, y)$ are different. 
In some cases, we get lucky and the models work 
despite covariate, label, or concept shift. 
In other cases, we can do better by employing 
principled strategies to cope with the shift. 
The remainder of this section grows considerably more technical. 
The impatient reader could continue on to the next section 
as this material is not prerequisite to subsequent concepts.

요약하면, 학습과 테스트 분포 $p(\mathbf{x}, y)$ 가 다른 경우가 많이 있습니다. 어떤 경우에는 운이 좋아서 공변량, 레이블, 또는 개념 변화에도 불구하고 모델이 잘 동작할 수 있습니다. 다른 경우에는 변화를 다루기 위한 원칙적 전략들을 사용해서 더 좋게 만들 수 있습니다. 이 절의 나머지는 더 기술적인 내용을 다룹니다. 이 내용들은 이어지는 개념들에 대한 선행이 아니므로, 급한 독자들은 다음 절도 넘어가도 됩니다.

## 공변량 변화(covariate shift) 교정 <— 여기부터 다시

*레이블을 달아놓은 데이터  $(x_i,y_i)$ 에 대한 의존도  $p(y|x)$ 를 추정하는 것을 한다고 가정합니다. 그런데,  $x_i$ 가 올바른 분포인  $p(x)$ 가 아닌 다른 분포 $q(x)$ 를 갖는 곳에서 추출됩니다. 먼저, 우리는 학습 과정에 정확하게 어떤 일이 일어나는지에 대해서 잘 생각해볼 필요가 있습니다. 즉, 학습 데이터와 연관된 레이블을 반복하면서, 매 미니 배치 이후에 모델의 가중치 벡터(weight vector)들을 업데이트합니다.*

*경우에 따라서 우리는 파라미터에 가중치 감쇠(weight decay), 드롭아웃(dropout), 존아웃(zoneout) 또는 유사한 패널티를 적용합니다. 즉, 학습은 대부분 손실(loss)을 최초화하는 것을 의미합니다.*
$$
\mathop{\mathrm{minimize}}_w \frac{1}{n} \sum_{i=1}^n l(x_i, y_i, f(x_i)) + \mathrm{some~penalty}(w)
$$

*통계학자들인 첫번째 항을 경험적인 평균(empirical average)이라고 합니다. 즉, 이것은  $p(x) p(y|x)$ 확률로 선택된 데이터에 구해진 평균을 의미합니다. 만약 데이터가 잘못된 분포 $q$ 에서 선택된다면, 다음과 같이 간단한 아이덴터티(identity)를 사용해서 수정할 수 있습니다.*
$$
\begin{aligned}
\int p(x) f(x) dx & = \int p(x) f(x) \frac{q(x)}{p(x)} dx \\
& = \int q(x) f(x) \frac{p(x)}{q(x)} dx
\end{aligned}
$$

Assume that we want to estimate some dependency $p(y|\mathbf{x})$ 
for which we have labeled data $(\mathbf{x}_i,y_i)$. 
Unfortunately, the observations $x_i$ are drawn 
from some *target* distribution $q(\mathbf{x})$ 
rather than the *source* distribution $p(\mathbf{x})$. 
To make progress, we need to reflect about what exactly 
is happening during training: 
we iterate over training data and associated labels 
$\{(\mathbf{x}_1, y_1), \ldots (\mathbf{x}_n, y_n)\}$
and update the weight vectors of the model after every minibatch.
We sometimes additionally apply some penalty to the parameters, 
using weight decay, dropout, or some other related technique. 
This means that we largely minimize the loss on the training.

레이블을 부여한 데이터,$(\mathbf{x}_i,y_i)$들에 대한 어떤 의존도  $p(y|\mathbf{x})$ 를 구하는 것을 생각해보겠습니다. 하지만 불행히도 관찰 $x_i$ 가 *소스* 분포  $p(\mathbf{x})$ 로 부터 추출된 것이 아니라 *타켓* 분포 $q(\mathbf{x})$ 에서 추출됩니다. 우선 우리는 학습 과정에서 정확히 무슨 일이 일어나는지 생각해봐야합니다: 우리는 학습 데이터와 그에 대한 레이블들,$\{(\mathbf{x}_1, y_1), \ldots (\mathbf{x}_n, y_n)\}$,을 반복하면서, 매 미니배치 후에 모델의 가중차 벡터를 업데이트합니다. 때로는 가중치 감쇠, 드롭아웃 또는 다른 관련된 기법을 사용해서 파라미터들에 패널티를 적용하기도 합니다. 이렇게 해서 우리는 학습에 대한 손실을 최소화합니다.
$$
\mathop{\mathrm{minimize}}_w \frac{1}{n} \sum_{i=1}^n l(x_i, y_i, f(x_i)) + \mathrm{some~penalty}(w)
$$

Statisticians call the first term an *empirical average*, 
i.e., an average computed over the data drawn from $p(x) p(y|x)$. 
If the data is drawn from the 'wrong' distribution $q$, 
we can correct for that by using the following simple identity:

통계학에서는 이 첫번째 항목을 *경험 평균(empirical average)* 라고 부릅니다. 즉, 이는  $p(x) p(y|x)$ 에서 추출된 데이터들에 대해서 계산된 평균입니다. 만약에 데이터라 '잘못된' 분포 $q$ 에서 추출되었다면, 우리는 다음과 같이 간단한 아이덴티티(identity)를 사용해서 고칠 수 있습니다.
$$
\begin{aligned}
\int p(\mathbf{x}) f(\mathbf{x}) dx & = \int p(\mathbf{x}) f(\mathbf{x}) \frac{q(\mathbf{x})}{q(\mathbf{x})} dx \\
& = \int q(\mathbf{x}) f(\mathbf{x}) \frac{p(\mathbf{x})}{q(\mathbf{x})} dx
\end{aligned}
$$
다르게 설명해보면, 데이터가 추출 되어야하는 올바른 분포에 대한 확률의 비율 $\beta(\mathbf{x}) := p(\mathbf{x})/q(\mathbf{x})$ 을 곱해서 각 샘플의 가중치를 조절하면 됩니다. 하지만 안타깝게도 이 비율을 알지 못 합니다. 따라서, 우선 해야하는 일은 이 값을 추정하는 것입니다. 이를 추정하는 다양한 방법이 존재합니다. 예로는 다소 멋진 이론적인 연산 방법이 있습니다. 이는 예상치를 계산하는 연산을 재조정하는 것으로, 이는 최소-놈(minimum-norm)이나 최대 엔트로피(maximum entropy) 원칙을 직접 이용하는 방법입니다. 이런 방법들은 두 분포에서 샘플들을 수집해야하는 것을 염두해 두세요. 즉, 학습 데이터를 이용해서 진짜 $p$ , 그리고 학습 데이터셋을 $q$ 를 만드는데 사용한 분포를 의미합니다.

Note however, that we only need samples $\mathbf{x} \sim q(\mathbf{x})$;
we do not to access labels labels $y \sim q(y)$.

하지만 우리는 샘플 $\mathbf{x} \sim q(\mathbf{x})$ 만 필요하고, 레이블 $y \sim q(y)$ 을 사용할 필요가 없음을 기억해두세요.

이 경우 좋은 결과를 주는 효과적인 방법이 있는데, 그것은 바로 선형 회귀(logistic regression)입니다. 선형 회귀를 이용하면 확률 비율을 계산해낼 수 있습니다. $p(\mathbf{x})$ 로 부터 추출된 데이터와 $q(x)$ 로 부터 추출된 데이터를 구분하기 위한 분리 모델을 학습 시키실 수 있습니다. 두 분포를 구별하는 것이 불가능하다면, 샘플들은 두 분포 중에 하나에서 나왔다는 것을 의미합니다. 반면에 분류가 잘 되는 샘플들은 오버웨이트(overweighted) 되었거나 언더웨이트(underweight)되어 있을 것입니다. 간단하게 설명하기 위해서, 두 분포로부터 같은 개수만큼 샘플을 추출했다고 가정하겠습니다. 이를 각각 $x_i \sim p(x)$ 와 $x_i′ \sim q(x)$ 로 표기합니다. $p$ 로부터 추출된 경우 $z_i$ 를 1로, $q$ 로 부터 추출된 경우에는 -1로 값을 할당합니다. 그러면, 섞인 데이터셋의 확률은 다음과 같이 표현됩니다.

$$p(z=1|\mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \text{ 이고 따라서 } \frac{p(z=1|\mathbf{x})}{p(z=-1|\mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}$$

따라서, $p(z=1|\mathbf{x})=\frac{1}{1+\exp(−f(\mathbf{x}))}$를 만족시키는 선형 회귀(logistic regression) 방법을 사용하면, 이 비율은 아래와 같은 수식으로 계산됩니다.

$$
\beta(\mathbf{x}) = \frac{1/(1 + \exp(-f(\mathbf{x})))}{\exp(-f(\mathbf{x})/(1 + \exp(-f(\mathbf{x})))} = \exp(f(\mathbf{x}))
$$

결론적으로 우리는 두 문제를 풀어야합니다. 첫번째 문제는 두 분포에서 추출된 데이터를 구분하는 것이고, 두번째는 가중치를 다시 적용한 최소화 문제입니다. 가중치 조정은  $\beta$ 를 이용하는데, 이는 해드 그래디언트(head gradient)를 이용합니다. 레이블이 없는 학습셋 $X$ 와 테스트셋 $Z$ 을 사용하는 프로토타입의 알고리즘은 아래와 같습니다.

1. 학습셋  $\{(\mathbf{x}_i, -1) ... (\mathbf{z}_j, 1)\}$ 을 생성합니다.
1. 로지스틱 회귀(Logistic regression)를 이용해서 이진(binary) 분류기를 학습시킵니다. 이를 함수 $f$ 라고 하겠습니다.
1. $\beta_i = \exp(f(\mathbf{x}_i))$ 또는  $\beta_i = \min(\exp(f(\mathbf{x}_i)), c)$를 이용해서 학습 데이터에 가중치를 적용합니다.
1. 데이터 $X$ 와 이에 대한 레이블 $Y$ 에 대한 학습을 수행할 때, 가중치  $\beta_i$ 를 이용합니다.

Note that this method relies on a crucial assumption.
For this scheme to work, we need that each data point 
in the tartget (test time)distribution 
had nonzero probability of occuring at training time.
If we find a point where $q(\mathbf{x}) > 0$ but $p(\mathbf{x}) = 0$,
then the corresponding importance weight should be infinity.

이 방법은 결정적인 가정에 의존함을 알아야합니다. 이 방법을 사용하기 위해서는 각 타겟 (테스트 수행시) 분산의 데이터 포인트들이 학습시 일어날 확률이 0이 아니어야 한다는 것입니다. 만약 $q(\mathbf{x}) > 0$ 이지만 $p(\mathbf{x}) = 0$ 를 만족하는 데이터 포인드가 있다면, 그에 대한 중요 가중치(importance weight)는 무한대가 됩니다.

*Generative Adversarial Networks 는 위에서 설명한 아이디어를 이용해서, 참조 데이터 셋과 구분이 어려운 데이터를 만드는 데이터 생성기(data generator)를 만듭니다. 네트워크 $f$ 는 진짜와 가짜 데이터는 구분하고, 다른 네트워크 $g$ 는 판정하는 역할을 하는 $f$ 를 속이는 역할, 즉 가짜 데이터를 진짜라고 판별하도록하는 역할을 수행합니다. 이에 대한 자세한 내용은 다시 다루겠습니다.*

**Generative Adversarial Networks** 
use a very similar idea to that described above 
to engineer a *data generator* that outputs data 
that cannot be distinguished 
from examples sampled from a reference dataset. 
In these approaches, we use one network, $f$ 
to distinguish real versus fake data 
and a second network $g$ that tries to fool the discriminator $f$ 
into accepting fake data as real. 
We will discuss this in much more detail later.

**Generative Adversarial Network** 는 참고되는 데이터셋에서 추출된 예제들과 구별이 되지 않는 결과를 *데이터 생성기* 를 만드는데 위에서 설명한 것과 유사한 아이디어를 사용합니다. 이 방법들은 진짜와 가짜 데이터를 구분하는 첫번째 네트워크 $f$ 와 식별자 *f* 를 속여서 가짜 데이터를 진짜로 생각하게 하는 두번째 네트워크 $g$ 를 사용합니다. 더 자세한 내용은 다음에 살펴보겠습니다.

### Label Shift Correction

For the discussion of label shift, 
we'll assume for now that we are dealing
with a $k$-way multiclass classification task.
When the distribution of labels shifts over time $p(y) \neq q(y)$
but the class-conditional distributions stay the same 
$p(\mathbf{x})=q(\mathbf{x})$,
our importance weights will correspond to the 
label likelihood ratios $q(y)/p(y)$.
One nice thing about label shift is that 
if we have a reasonably good model (on the source distribution) 
then we can get consistent estimates of these weights 
without ever having to deal with the ambient dimension 
(in deep learning, the inputs are often high-dimensional perceptual objects like images, while the labels are often easier to work, 
say vectors whose length corresponds to the number of classes).

레이블 변화를 논의하기 위해서 우리가 지금 $k$개에 대한 다중 클래스 분류 과제를 수행하고 있다고 하겠습니다. 시간이 지나면서 레이블의 분포가 변화하고($p(y) \neq q(y)$)그렇지만 클래스에 대한 조건 분포는 변하지 않을 때()$p(\mathbf{x})=q(\mathbf{x})$), 중요 가중치(importance weight)들은 레이블 유사 비율 $q(y)/p(y)$ 에 대응할 것입니다. 레이블 변화의 한가지 좋은 점은 (소스 분포에 대해서) 합리적으로 좋은 모델이 있다면, 부가적인 차원을 다룰 필요없이 이 가중치들에 대한 일관적인 추정이 가능하다는 것입니다. (딥러닝에서 입력은 종종 차원이 큰 개념적인 객체이나, 레이블은 클래스의 개수와 같은 수의 길이의 벡터와 같이 다루기 쉬운 형태입니다.)

To estimate calculate the target label distribution,
we first take our reasonably good off the shelf classifier 
(typically trained on the training data)
and compute its confusion matrix using the validation set
(also from the training distribution).
The confusion matrix C, is simply a $k \times k$ matrix
where each column corresponsd to the *actual* label
and each row corresponds to our model's predicted label.
Each cell's value $c_{ij}$ is the fraction of predictions 
where the true label was $j$ *and* our model predicted $y$.

타켓 레이블 분포를 추정하기 위해서 우선 (일반적으로는 학습 데이터를 사용해서 학습한) 합리적으로 좋은 분류기를 얻어야 하고, 검증 데이터셋 (이는 학습 분포를 따름니다)을 이용해서 오차 행렬(confusion matrix)를 계산합니다. 오차 행렬 C는 간단하게 $k \times k$ 행렬로 각 컬럼은 *실제* 레이블, 각 행은 모델이 예측한 레이블을 나타냅니다. $j$ 는 실제 레이블, $i$ 는 모델이 예측한 값에 대한 인덱스에 대한 각 원소의 값 $c_{ij}$ 은 해당 구분에 대해서 예측된 값들의 개수를 나타냅니다.

Now we can't calculate the confusion matrix
on the target data directly,
because we don't get to see the labels for the examples 
that we see in the wild,
unless we invest in a complex real-time annotation pipeline.
What we can do, however, is average all of our models predictions 
at test time together, yielding the mean model output $\mu_y$.

지금 우리는 타겟 데이터에 대해서 직접 오차 행렬을 계산할 수 없습니다. 그 이유는 복잡한 실시간 어노테이션 파이프라인에 대한 투자를 하지 않았다면 실제 우리가 보는 예제들에 대한 레이블을 알 수가 없기 때문입니다. 하지만, 우리는 테스트를 수행할 때 모델의 예측값 전체에 대한 평균을 계산해서 평균 모델 결과 $\mu_y$ 를 얻을 수 있습니다.

It turns out that under some mild conditions—
if our classifier was reasonably accurate in the first place,
if the target data contains only classes of images that we've seen before,
and if the label shift assumption holds in the first place 
(far the strongest assumption here),
then we can recover the test set label distribution by solving
a simple linear system $C \cdot q(y) = \mu_y$.
If our classifier is sufficiently accurate to begin with,
then the confusion C will be invertible, 
and we get a solution $q(y) = C^{-1} \mu_y$.
Here we abuse notation a bit, using $q(y)$ 
to denote the vector of label frequencies.
Because we observe the labels on the source data,
it's easy to estimate the distribution $p(y)$.
Then for any training example $i$ with label $y$, 
we can take the ratio of our estimates $\hat{q}(y)/\hat{p}(y)$
to calculate the weight $w_i$,
and plug this into the weighted risk minimization algorithm above.

—> 여기 부터 계속 !!!!!!!!

## 개념 변화(concept shift) 교정

*개념 변화(concept shift)는 개념적으로 해결하기 훨씬 어렵습니다. 예를 들면, 고양이와 강아지를 구분하는 문제에서 흰색과 검은색 동물을 구분하는 문제로 갑자기 바뀌었다고 하면, 새로운 레이블을 이용해서 새로 학습을 시키는 것보다 더 잘 동작시키는 것을 기대하는 것은 무리일 것입니다. 다행히, 실제 상황에서는 이렇게 심한 변화는 발생하지 않습니다. 대신, 변화가 천천히 일어나는 것이 보통의 경우입니다. 더 정확하게 하기 위해서, 몇가지 예를 들어보겠습니다.*

Concept shift is much harder to fix in a principled manner. 
For instance, in a situation where suddenly the problem changes 
from distinguishing cats from dogs to one of 
distinguishing white from black animals, 
it will be unreasonable to assume 
that we can do much better than just collecting new labels
and training from scratch. 
Fortunately, in practice, such extreme shifts are rare. 
Instead, what usually happens is that the task keeps on changing slowly. 
To make things more concrete, here are some examples:

* 광고에서 새로운 상품이 출시되고, 이전 상품의 인기는 떨어집니다. 즉, 광고의 분포와 인기도는 서서히 변화되기 때문에, click-through rate 예측 모델은 그에 따라서 서서히 바뀌어야 합니다.
* 교통 카메라 렌즈는 환경의 영향으로 서서히 성능이 떨어지게 되고, 그 결과 이미지 품질에 영향을 미칩니다.
* 뉴스 내용이 서서히 바뀝니다. (즉, 대부분의 뉴스는 바뀌지 않지만, 새로운 이야기가 추가됩니다.)

이런 경우에 네트워크 학습에 사용한 것과 같은 방법을 데이터의 변화에 적응시키는 데 사용할 수 있습니다. 즉, 네트워크를 처음부터 다시 학습시키는 것이 아니라, 현재 가중치 값을 갖는 네트워크에 새로이 추가된 데이터를 이용해서 학습시키는 것입니다. 

## 학습 문제의 분류

 $p(x)$ 과  $p(y|x)$ 이 바뀔 때 어떻게 다뤄야하는지에 대해서 알아봤으니, 머신 러닝을 이용해서 풀 수 있는 여러가지 문제들에 대해서 알아보겠습니다.

* **배치 러닝**. 학습 데이터와 레이블 쌍  $\{(x_1, y_1), \ldots (x_n, y_n)\}$ 을 사용해서 네트워크 $f(x,w)$ 를 학습시킨다고 생각해봅니다. 모델을 학습시킨 후, 학습 데이터와 같은 분포에서 새로운 데이터 $(x,y)$ 를  뽑아서 이 모델에 적용합니다. 우리가 여기서 논의하는 대부분의 문제는 이 기본적인 가정을 포함하고 있습니다. 예를 들면, 고양이와 강아지 사진을 사용해서 고양이 탐지 모델을 학습시킵니다. 모델을 학습시킨 후, 고양이만 들어올 수 있도록 하는 컴퓨터 비전을 이용한 고양이 전용 문 시스템에 이 모델을 사용합니다. 이 시스템을 고객의 가정에 설치한 후에 모델을 다시 업데이트하지 않습니다.
* **온라인 러닝**. 데이터 $(x_i, y_i)$ 가 한번에 하나씩 들어오는 것을 가정합니다. 조금 더 명확하게 말하자면, 우선 $x_i$ 가 관찰되면, $f(x_i,w)$ 를 통해서 추측을 수행 한 이후에만  $y_i$ 를 알 수 있는 경우를 가정합니다. 이 후, 추측 결과에 대한 보상 또는 loss 를 계산합니다. 많은 실제 문제가 이러한 분류에 속합니다. 예를 들면, 다음 날의 주식 가격을 예측하는 경우를 생각해보면, 예측된 주가에 근거해서 거래를 하고, 그날의 주식시장이 끝나면 예측이 수익을 가져다 줬는지 알 수 있습니다. 달리 말하면, 새로운 관찰을 통해서 모델을 지속적으로 발전시키는 다음과 같은 사이클을 만들 수 있습니다.

$$
\mathrm{model} ~ f_t \longrightarrow
\mathrm{data} ~ x_t \longrightarrow
\mathrm{estimate} ~ f_t(x_t) \longrightarrow
\mathrm{observation} ~ y_t \longrightarrow
\mathrm{loss} ~ l(y_t, f_t(x_t)) \longrightarrow
\mathrm{model} ~ f_{t+1}
$$

* **반딧**. 반딧은 위 문제의 *특별한 경우*입니다. 대부분의 학습 문제는 연속된 값을 출력하는 함수  $f$ 의 파라메터(예를 들면 딥 네트워크)를 학습하는 경우이지만, 반딧 문제는 선택할 수 있는 종류가 유한한 경우 (즉, 취할 수 있는 행동이 유한한 경우)입니다. 이 간단한 문제의 경우, 최적화의 측면에서 강력한 이론적인 보증을 얻을 수 있다는 것이 당연합니다. 이 문제를 별도로 분류한 이유는 이 문제를 종종 distinct learning과 혼동하기 때문입니다.
* **제어 (그리고 비대립적 강화 학습).** 많은 경우에 환경은 우리가 취한 행동을 기억합니다. 적의적인 의도가 아닌 경우에도, 단순히 기억하고, 이전에 일어난 일에 근거해서 반응하는 환경들이 있습니다. 즉, 커피 포트 제어기의 경우 이전에 데웠는지 여부에 따라서 다른 온도를 감지하기도 합니다. PID (Propotional Integral Derivative) 제어 알고리즘도 [유명한 예](http://pidkits.com/alexiakit.html)입니다. 비슷한 예로, 뉴스 사이트에 대한 사용자의 행동은 이전에 무엇을 무엇이었는지 영향을 받습니다. 이런 종류의 많은 알고리즘들은 그 결정들이 임의의 선택으로 보이지 않도록 모델을 만들어냅니다. (즉, 분산을 줄이는 방향으로)
* **강화 학습**. 기억을 하는 환경의 더 일반적인 예로 우리와 협력을 시도하는 환경(non-zero-sum 게임과 같이 협력적인 게임)이나 이기려고 하는 환경이 있습니다. 체스나, 바둑, 서양주사위놀이(Backgammon) 또는 스타크래프트가 경쟁하는 환경의 예들입니다. 마찬가지로, 자율주행차를 위한 좋은 제어기를 만드는 것도 생각해볼 수 있습니다. 이 경우 다른 차량들은 자율주행차의 운전 스타일에 여러가지로 반응을 합니다. 때로는 피하려고 하거나, 사고를 내려고 하거나, 같이 잘 주행하려고 하는 등 여러 반응을 보일 것입니다.

위에 설명한 다양한 상황들 간의 주요 차이점은 안정적인 환경에서 잘 작동하는 전략이 환경이 변화는 상황에서는 잘 작동하지 않을 수 있다는 것입니다. 예를 들면, 거래자가 발견한 차익 거래 기회는 한번 실행되면 사라질 가능성이 높습니다. 환경이 변화하는 속도나 형태는 계속해서 사용할 수 있는 알고리즘의 형태를 많이 제약합니다. 예를 들면, 어떤 것이 천천히 변화할 것이라고 알고 있을 경우, 예측 모델 또한 천천히 바뀌도록 할 수 있습니다. 만약, 환경이 불규적으로 순간적으로 바뀐다고 알고 있는 경우에는, 이에 대응 하도록 만들 수 있습니다. 이런 종류의 지식은 풀고자 하는 문제가 시간에 따라서 바뀌는 상황, 즉 개념 변화(concerpt shit)를 다루는 야심 찬 데이터 사이언티스트에게 아주 중요합니다.

## Fairness, Accountability, and Transparency in machine Learning

Finally, it's important to remember 
that when you deploy machine learning systems
you aren't simply minimizing negative log likelihood
or maximizing accuracy—you are automating some kind of decision process.
Often the automated decision-making systems that we deploy
can have consequences for those subject to its decisions.
If we are deploying a medical diagnostic system,
we need to know for which populations it may work and which it may not.
Overlooking forseeable risks to the welfare of a subpopulation
would run afoul of basic ethical principles.
Moreover, "accuracy" is seldom the right metric.
When translating predictions in to actions
we'll often want to take into account the potential cost sensitivity
of erring in various ways. 
If one way that you might classify an image could be perceived as a racial sleight, while misclassification to a different category
would be harmless, then you might want to adjust your thresholds
accordingly, accounting for societal values 
in designing the decision-making protocol.
We also want to be careful about how prediction systems 
can lead to feedback loops. 
For example, if prediction systems are applied naively to predictive policing,
allocating patrol officers accordingly, a vicious cycle might emerge.
Neighborhoods that have more crimes, get more patrols, get more crimes discovered, get more training data, get yet more confident predictions, leading to even more patrols, even more crimes discovered, etc. 
Additionally, we want to be careful about whether we are addressing the right problem in the first place. Predictive algorithms now play an outsize role in mediating the dissemination of information.
Should what news someone is exposed to be determined by which Facebook pages they have *Liked*? These are just a few among the many profound ethical dilemmas that you might encounter in a career in machine learning.

## 요약

* 많은 경우에 학습셋와 테스트셋는 같은 분포로부터 얻어지지 않습니다. 이런 상황을 우리는 공변량 변화(covariate shift)라고 합니다.
* 공변량 변화(covariate shift)는 변화가 아주 심하지 않을 경우에 탐지하고 이를 교정할 수 있습니다. 만약 그렇게 하지 못하면, 테스트 시점에 좋지 않은 결과가 나옵니다.
* 어떤 경우에는 환경이 우리가 취한 것을 기억하고, 예상하지 못한 방법으로 결과를 줄 수도 있습니다. 모델을 만들에 이점을 유의해야합니다.

## 연습문제

1. 검색 엔진의 행동을 바꾸면 어떤 일이 일어날까요? 사용자는 어떻게 반응할까요? 광고주는 어떨까요?
1. 공변량 변화(covariate shift) 탐기지를 구현해보세요. 힌트 - 분류기를 만들어 봅니다.
1. 공변량 변화(covariate shift)t 교정기를 구현해보세요.
1. 학습 세트와 테스트 세트가 많이 다를 경우 무엇이 잘못될 수 있을까요? 샘플 weight들에는 어떤 일이 일어날까요?

## QR 코드를 스캔해서 [논의하기](https://discuss.mxnet.io/t/2347)

![](../img/qr_environment.svg)
