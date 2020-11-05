# Softmax Regression
# 소프트맥스 회귀(Softmax Regression)
:label:`sec_softmax`

In :numref:`sec_linear_regression`, we introduced linear regression,
working through implementations from scratch in :numref:`sec_linear_scratch`
and again using high-level APIs of a deep learning framework
in :numref:`sec_linear_concise` to do the heavy lifting.

:numref:`sec_linear_regression`에서 우리는 선형 회귀를 소개했고, :numref:`sec_linear_scratch`에서 이를 처음부터 구현해봤고, :numref:`sec_linear_concise`에서는 반복되는 것들을 대신 해주는 딥러닝 프래임워크의 고차원 API를 사용해 봤습니다.

Regression is the hammer we reach for when
we want to answer *how much?* or *how many?* questions.
If you want to predict the number of dollars (price)
at which a house will be sold,
or the number of wins a baseball team might have,
or the number of days that a patient will remain hospitalized before being discharged,
then you are probably looking for a regression model.

회귀는 *몇 개*인지, *얼마*인지 질문 대한 답을 구할 때 사용하는 도구입니다. 예를 들면 집이 얼마에 팔릴지, 어떤 야구팀이 몇 번 승리를 할 것인지,  또는 환자가 몇 일 만에 퇴원할 것인지 예측 등을 예측하고 싶다면, 여러분은 아마도 회귀 모델을 찾고 있을 것입니다.

In practice, we are more often interested in *classification*:
asking not "how much" but "which one":

현실에서는 "얼마나 많이"를 묻기 보다는 "어떤 것"을 묻는 *분류*에 종종 더 관심을 갖는다.

* Does this email belong in the spam folder or the inbox?
* Is this customer more likely *to sign up* or *not to sign up* for a subscription service?
* Does this image depict a donkey, a dog, a cat, or a rooster?
* Which movie is Aston most likely to watch next?

* 이 이메일이 스팸 폴더에 속할지 받은 편지함에 속할지?
* 이 고객이 구독 서비스에 *가입을 할 것인지* 아니면 *가입을 하지 않을 것인지*?
* 이 이미지가 당나귀, 강아지, 고양지 또는 닭 중 어떤 것인지?
* Aston(저자 중 한명)이 다음으로 볼 영화가 무엇일지?

Colloquially, machine learning practitioners
overload the word *classification*
to describe two subtly different problems:
(i) those where we are interested only in
hard assignments of examples to categories (classes);
and (ii) those where we wish to make soft assignments,
i.e., to assess the probability that each category applies.
The distinction tends to get blurred, in part,
because often, even when we only care about hard assignments,
we still use models that make soft assignments.

평소에 사용하는 용어로 머신러닝 전문가들은 미묘하게 다른 두 가지 문제를 설명할 때 *분류*라는 단어를 사용합니다: (i) 샘플들을 카테고리들(또는 클래스들)로 하드 할당(hard assignment)하는 것에 관심이 있는 문제와 (ii) 각 카테고리에 할당될 활률을 평가하는 것과 같은 소프트 할당(soft assignmetn)를 원하는 문제. 우리는 하드 할당에만 관심이 있을 때도 소프트 할당을 하는 모델을 여전히 사용하기 때문에, 어떤 면에서는 이를 구분하는 것은 점점 모호해지고 있습니다.

## Classification Problem
## 분류 문제
:label:`subsec_classification-problem`

To get our feet wet, let us start off with
a simple image classification problem.
Here, each input consists of a $2\times2$ grayscale image.
We can represent each pixel value with a single scalar,
giving us four features $x_1, x_2, x_3, x_4$.
Further, let us assume that each image belongs to one
among the categories "cat", "chicken", and "dog".

발을 담구기 위해서 간단한 이미지 분류 문제부터 시작하겠습니다. 각 입력은 $2\times2$ 회색 이미지로 구성되어 있습니다. 각 픽셀 값을 한 개의 스칼라로 표현할 수 있고, 이는 4개 피처 $x_1, x_2, x_3, x_4$가 됩니다. 더 나아가 각 이미지는 "고양이", "닭", 또는 "강아지" 카테고리 중에 하나에 속한다고 가정합니다.

Next, we have to choose how to represent the labels.
We have two obvious choices.
Perhaps the most natural impulse would be to choose $y \in \{1, 2, 3\}$,
where the integers represent $\{\text{dog}, \text{cat}, \text{chicken}\}$ respectively.
This is a great way of *storing* such information on a computer.
If the categories had some natural ordering among them,
say if we were trying to predict $\{\text{baby}, \text{toddler}, \text{adolescent}, \text{young adult}, \text{adult}, \text{geriatric}\}$,
then it might even make sense to cast this problem as regression
and keep the labels in this format.

다음으로 우리는 레이블을 어떻게 표현할지를 선택해야 합니다. 두 가지 자명한 방법이 있습니다. 아마도 가장 자연스러운 방법은 $y \in \{1, 2, 3\}$ 로 선택하는 것인데, 각 정수는 $\{\text{강아지}, \text{고양이}, \text{닭}\}$을 각각 의미합니다. 이것은 컴퓨터에 정보를 *저장*하는 훌륭한 방법입니다. 만약 카테고리들이 그것들 사이에 자연스러운 순서가 있다면, 예를 들면 $\{\text{아이}, \text{유아}, \text{청소년}, \text{청년}, \text{성인}, \text{노인}\}$을 예측하는 경우, 이 문제를 회귀의 문제로 정의하고, 이 형태로 레이블을 부여하는 것이 의미가 있을 것입니다.

But general classification problems do not come with natural orderings among the classes.
Fortunately, statisticians long ago invented a simple way
to represent categorical data: the *one-hot encoding*.
A one-hot encoding is a vector with as many components as we have categories.
The component corresponding to particular instance's category is set to 1
and all other components are set to 0.
In our case, a label $y$ would be a three-dimensional vector,
with $(1, 0, 0)$ corresponding to "cat", $(0, 1, 0)$ to "chicken",
and $(0, 0, 1)$ to "dog":

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$

하지만 일반적인 분류 문제는 클래스들 사이에 자연스러운 순서가 있지 않습니다. 운이 좋게도 통계학자들은 오래전에 카테고리 데이터를 표현하는 간단한 방법을 고안했습니다: 바로 *원-핫 인코딩(one-hot encoding)*입니다. 원-핫 인코딩은 카테고리 수만큰의 원소를 갖는 벡터입니다. 특정 인스턴스의 카테고리에 속한 컴포넌트는 1로 설정하고, 다른 모든 컴포넌트는 0으로 설정합니다. 우리의 경우 레이블 $y$는 3차원 벡터가 될 것이고, "고양이"는 $(1, 0, 0)$, "닭"은  $(0, 1, 0)$, 그리고 "강아지"는  $(0, 0, 1)$이 될 것입니다.

## Network Architecture
## 네트워크 아키텍처

In order to estimate the conditional probabilities associated with all the possible classes,
we need a model with multiple outputs, one per class.
To address classification with linear models,
we will need as many affine functions as we have outputs.
Each output will correspond to its own affine function.
In our case, since we have 4 features and 3 possible output categories,
we will need 12 scalars to represent the weights ($w$ with subscripts),
and 3 scalars to represent the biases ($b$ with subscripts).
We compute these three *logits*, $o_1, o_2$, and $o_3$, for each input:

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

모든 가능한 클래스 부여되는 조건부 확률을 추정하기 위해서 우리는 각 클래스별로 한 개씩인 다중 출력을 갖는 모델이 필요합니다. 선형 모델로 분류 문제를 해결하기 위해 결과 개수만큼의 아핀 함수가 필요합니다. 각 출력은 그것만의 아핀 함수에 대응할 것입니다. 우리의 경우 4개 피처와 3개의 가능한 출력 카테고리가 있기 때문에, 가중치들  ($w$ with subscripts)을 표현할 12개 스칼라와 편향 ($b$ with subscripts)을 표현하는 3개의 스칼라가 필요합니다. 우리는 각 입력에 대해서 3개 *로짓(logit)*, $o_1, o_2$, 과 $o_3$을 계산합니다.

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

We can depict this calculation with the neural network diagram shown in :numref:`fig_softmaxreg`.
Just as in linear regression, softmax regression is also a single-layer neural network.
And since the calculation of each output, $o_1, o_2$, and $o_3$,
depends on all inputs, $x_1$, $x_2$, $x_3$, and $x_4$,
the output layer of softmax regression can also be described as fully-connected layer.

![Softmax regression is a single-layer neural network.](../img/softmaxreg.svg)
:label:`fig_softmaxreg`

우리는 이 계산을 :numref:`fig_softmaxreg`에 보이는 것처럼 뉴럴 네트워크 다이어그램으로 표현할 수 있습니다. 선형 회귀에서 처럼, 소프트맥스 회귀 역시 단일 층 뉴럴 네트워크입니다. 각 출력, $o_1, o_2$, 과 $o_3$,의 계산이 모든 입력,  $x_1$, $x_2$, $x_3$, 과 $x_4$에 의존하기 때문에, 소프트맥스 회귀의 출력 층은 완전-연결 층으로 표현될 수 있습니다.

![Softmax regression is a single-layer neural network.](../img/softmaxreg.svg)
:label:`fig_softmaxreg`

To express the model more compactly, we can use linear algebra notation.
In vector form, we arrive at 
$\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$,
a form better suited both for mathematics, and for writing code.
Note that we have gathered all of our weights into a $3 \times 4$ matrix
and that for features of a given data point $\mathbf{x}$,
our outputs are given by a matrix-vector product of our weights by our input features
plus our biases $\mathbf{b}$.

모델을 보다 간결하게 표현하기 위해서 우리는 선형 대수 표기법을 사용할 수 있습니다. 벡터 형식으로 표현하면, $\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$ 이 되는데, 수학과 코드 작성 모두에 더 적합한 형태입니다. 모든 가중치를 $3 \times 4$ 행렬로 모았고, 주어진 데이터 포인트의 피처,$\mathbf{x}$에 대해서 결과는 가중치와 입력 피처의 행렬-벡터 곱에 편향 $\mathbf{b}$을 더하는 것으로 구해집니다.

## Softmax Operation
## 소프트맥스 연산

The main approach that we are going to take here
is to interpret the outputs of our model as probabilities.
We will optimize our parameters to produce probabilities
that maximize the likelihood of the observed data.
Then, to generate predictions, we will set a threshold,
for example, choosing the label with the maximum predicted probabilities.

우리가 여기서 취할 주요 접근법은 모델의 출력을 확률로 해석하는 것입니다. 우리는 관찰된 데이터의 가능도를 최대화하는 확률을 만들도록 파라미터를 최적화할 것입니다. 그리고, 예측을 생성하기 위해서 우리는 임계갑을 정합니다. 예를 들어 최대 예측 확률이 있는 레이블을 선택합니다.

Put formally, we would like any output $\hat{y}_j$
to be interpreted as the probability
that a given item belongs to class $j$.
Then we can choose the class with the largest output value
as our prediction $\operatorname*{argmax}_j y_j$.
For example, if $\hat{y}_1$, $\hat{y}_2$, and $\hat{y}_3$
are 0.1, 0.8, and 0.1, respectively,
then we predict category 2, which (in our example) represents "chicken".

공식적으로 우리는 출력 $\hat{y}_j$가 그 아이템이 클래스 $j$에 속할 확률로 해석하고 싶습니다. 그렇게되면 우리는 가장 큰 출력 값을 갖는 클래스를 우리의 예측, $\operatorname{argmax}_j y_j$으로 선택할 수 있습니다. 예를 들면, 만약 $\hat{y}_1$, $\hat{y}_2$ 와 $\hat{y}_3$가 각각 0.1, 0.8, 0.1이면, 우리는 "닭"을 의미하는 카테고리 2로 예측합니다.

You might be tempted to suggest that we interpret
the logits $o$ directly as our outputs of interest.
However, there are some problems with directly
interpreting the output of the linear layer as a probability.
On one hand,
nothing constrains these numbers to sum to 1.
On the other hand, depending on the inputs, they can take negative values.
These violate basic axioms of probability presented in :numref:`sec_prob`

로짓 $o$를 직접 출력으로 해석하면 것을 제안하고 싶을 것입니다. 하지만, 선형 층의 결과를 직접 확률로 해석하는 것에는 몇 가지 문제가 있습니다. 하나는 출력의 숫자들의 합이 1이 되도록 하는 장치가 없습니다. 다른 한편으로는 입력에 따라서 출력의 값이 음수가 될 수도 있습니다. 이것들은 :numref:`sec_prob`에 제시된 확률의 기본 공리를 위반합니다.

To interpret our outputs as probabilities,
we must guarantee that (even on new data),
they will be nonnegative and sum up to 1.
Moreover, we need a training objective that encourages
the model to estimate faithfully probabilities.
Of all instances when a classifier outputs 0.5,
we hope that half of those examples
will actually belong to the predicted class.
This is a property called *calibration*.

출력을 확률로 해석하기 위해서는 이 값들이 음수가 아니고 합이 1이 되는 것을 보장되어야 합니다. 또한, 모델이 충실하게 확률을 추정하게 만들는 학습 오브젝티브가 필요합니다. 전체에 대해서 분류기의 출력이 0.5라면, 이 샘플들의 반이 실제로 예측된 클래스에 속할 것이라고 희망합니다. 이것은 *칼리브래이션(calibration)*이라는 속성입니다.

The *softmax function*, invented in 1959 by the social scientist
R. Duncan Luce in the context of *choice models*,
does precisely this.
To transform our logits such that they become nonnegative and sum to 1,
while requiring that the model remains differentiable,
we first exponentiate each logit (ensuring non-negativity)
and then divide by their sum (ensuring that they sum to 1):

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{where}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}. $$
:eqlabel:`eq_softmax_y_and_o`

1959년에 사회과학자인 R. Duncan Luce가 *선택 모델*과 관련해서 발명한 *소프트맥스 함수*는 정확하게 이것을 수행합니다. 모델이 미분가능한 상태로 유지되면서 로짓이 음이 아닌 수가 되고 합이 1이 되도록 변형하기 위해서 우리는 각 로짓에 지수를 취하고(음이 아닌 수가 되도록), 그것들의 합으로 나눕니다(합이 1이 되도록).

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{where}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}. $$
:eqlabel:`eq_softmax_y_and_o`

It is easy to see $\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$
with $0 \leq \hat{y}_j \leq 1$ for all $j$.
Thus, $\hat{\mathbf{y}}$ is a proper probability distribution
whose element values can be interpreted accordingly.
Note that the softmax operation does not change the ordering among the logits $\mathbf{o}$,
which are simply the pre-softmax values
that determine the probabilities assigned to each class.
Therefore, during prediction we can still pick out the most likely class by

$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$

모든 $j$에 대해서 $0 \leq \hat{y}_j \leq 1$일 때, $\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$이 됨을 확인하는 것은 쉽습니다. 즉, $\hat{\mathbf{y}}$는 적절한 확률 분포로 각 원소의 값이 적절히 해설될 수 있습니다. 각 클래스에 속할 확률을 결정하는 소프트맥스 적용 전의 값인 로짓 $\mathbf{o}$, 의 순서를 소프트맥스 연산에 의해서 바꾸지 않는 다는 것을 유념하세요. 따라서, 예측할 때 우리는 가장 가능성이 높은 클래스를 다음과 같이 뽑을 수 있습니다.

Although softmax is a nonlinear function,
the outputs of softmax regression are still *determined* by
an affine transformation of input features;
thus, softmax regression is a linear model.

소프트맥스가 비선형 함수일지라도, 소프트맥스 회귀의 결과는 여전히 입력 피처에 대한 아핀 변환으로 *결정*됩니다. 따라서, 소프트맥스 회귀는 선형 모델인 것입니다.

## Vectorization for Minibatches
## 미니배치의 벡터화
:label:`subsec_softmax_vectorization`

To improve computational efficiency and take advantage of GPUs,
we typically carry out vector calculations for minibatches of data.
Assume that we are given a minibatch $\mathbf{X}$ of examples
with feature dimensionality (number of inputs) $d$ and batch size $n$.
Moreover, assume that we have $q$ categories in the output.
Then the minibatch features $\mathbf{X}$ are in $\mathbb{R}^{n \times d}$,
weights $\mathbf{W} \in \mathbb{R}^{d \times q}$,
and the bias satisfies $\mathbf{b} \in \mathbb{R}^{1\times q}$.

$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$
:eqlabel:`eq_minibatch_softmax_reg`

연산의 효율성을 높이고 GPU의 장점을 살리기 위해서, 일반적으로 데이터의 미니배치에 대한 벡터 계산을 수행합니다. 피처의 차원(입력의 개수)이 $d$이고, 배치 크기가 $n$인 예제들의 미니배치 $\mathbf{X}$가 주어졌다고 가정합니다. 또한, 출력은 $q$ 개의 카테고리를 갖는다고 가정합니다. 그러면, 미니배치 피처들 $\mathbf{X}$은 $\mathbb{R}^{n \times d}$에 속하고, 가중치들은 $\mathbf{W} \in \mathbb{R}^{d \times q}$, 그리고 편향들은 $\mathbf{b} \in \mathbb{R}^{1\times q}$를 만족합니다.

$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$
:eqlabel:`eq_minibatch_softmax_reg`

This accelerates the dominant operation into
a matrix-matrix product $\mathbf{X} \mathbf{W}$
vs. the matrix-vector products we would be executing
if we processed one example at a time.
Since each row in $\mathbf{X}$ represents a data point,
the softmax operation itself can be computed *rowwise*:
for each row of $\mathbf{O}$, exponentiate all entries and then normalize them by the sum.
Triggering broadcasting during the summation $\mathbf{X} \mathbf{W} + \mathbf{b}$ in :eqref:`eq_minibatch_softmax_reg`,
both the minibatch logits $\mathbf{O}$ and output probabilities $\hat{\mathbf{Y}}$
are $n \times q$ matrices.

이것은 대부분 연산을 행렬-행렬 곱 $\mathbf{X} \mathbf{W}$으로 가속화시킵니다. 만약 샘플을 한 번에 하나씩 처리하는 경우는 행렬-벡터 곱이 됩니다. $\mathbf{X}$의 각 행은 데이터 포인트이기 때문에, 소프트맥스 연산 자체는 *행을 따라(rowwise)* 계산될 수 있습니다. 즉, $\mathbf{O}$의 각 행에 대해서, 모든 원소들의 지수승을 구한 후, 합으로 정규화를 합니다. :eqref:`eq_minibatch_softmax_reg`의 $\mathbf{X} \mathbf{W} + \mathbf{b}$ 합을 수행할 때는 브로드케스팅이 적용되면서, 미니배치 로짓 $\mathbf{O}$와 결과 확률들 $\hat{\mathbf{Y}}$은 $n \times q$ 크기의 행렬입니다.

## Loss Function
## 손실 함수

Next, we need a loss function to measure
the quality of our predicted probabilities.
We will rely on maximum likelihood estimation,
the very same concept that we encountered
when providing a probabilistic justification
for the mean squared error objective in linear regression
(:numref:`subsec_normal_distribution_and_squared_loss`).

다음으로는 예측된 확률의 품질을 측정할 손실 함수가 필요합니다. 우리는 선형 회귀의 평균 제곱 오류 오브젝티브를 위한 확률적인 평가를 제공할 때 사용한 매우 동일한 개념(:numref:`subsec_normal_distribution_and_squared_loss`)인, 최대 가능도 추정(maximum likelihood estimation)에 의존할 것입니다. 

### Log-Likelihood
### 로그-가능도

The softmax function gives us a vector $\hat{\mathbf{y}}$,
which we can interpret as estimated conditional probabilities
of each class given any input $\mathbf{x}$, e.g.,
$\hat{y}_1$ = $P(y=\text{cat} \mid \mathbf{x})$.
Suppose that the entire dataset $\{\mathbf{X}, \mathbf{Y}\}$ has $n$ examples,
where the example indexed by $i$
consists of a feature vector $\mathbf{x}^{(i)}$ and a one-hot label vector $\mathbf{y}^{(i)}$.
We can compare the estimates with reality
by checking how probable the actual classes are
according to our model, given the features:

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).
$$

소프트맥스 함수는 주어진 임의의 입력 $\mathbf{x}$에 대해서 각 클래스의 추정된 조건부 확률로 해석될 수 있는 $\hat{\mathbf{y}}$ 벡터를 줍니다. 즉, $\hat{y}_1$ = $P(y=\text{cat} \mid \mathbf{x})$. 전체 데이터셋 $\{\mathbf{X}, \mathbf{Y}\}$이 $n$개의 예제를 가지고 있고, $i$로 인덱스된 예제는 피처 벡터  $\mathbf{x}^{(i)}$와 원-핫 레이블 벡터 $\mathbf{y}^{(i)}$로 구성되어 있다고 가정하겠습니다. 주어진 피처에 대한 모델의 결과에 따라 실제 클래스들이 얼마나 가능하지를 확인하고, 그것으로 추정과 실제를 비교할 수 있습니다.

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).
$$

According to maximum likelihood estimation,
we maximize $P(\mathbf{Y} \mid \mathbf{X})$,
which is 
equivalent to minimizing the negative log-likelihood:

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$

where for any pair of label $\mathbf{y}$ and model prediction $\hat{\mathbf{y}}$ over $q$ classes,
the loss function $l$ is

$$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$
:eqlabel:`eq_l_cross_entropy`

최대 가능도 추정에 따르면, $P(\mathbf{Y} \mid \mathbf{X})$를 최대화하는데, 이것은 네가티브 로그-가능도를 최소화하는 것과 동일합니다:

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$

여기서 레이블 $\mathbf{y}$와 $q$ 클래스들에 대한 모델의 예측 $\hat{\mathbf{y}}$ 쌍에 대해서 손실 함수 $l$은 다음과 같습니다.

$$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$
:eqlabel:`eq_l_cross_entropy`

For reasons explained later on, the loss function in :eqref:`eq_l_cross_entropy`
is commonly called the *cross-entropy loss*.
Since $\mathbf{y}$ is a one-hot vector of length $q$,
the sum over all its coordinates $j$ vanishes for all but one term.
Since all $\hat{y}_j$ are predicted probabilities,
their logarithm is never larger than $0$.
Consequently, the loss function cannot be minimized any further
if we correctly predict the actual label with *certainty*,
i.e., if the predicted probability $P(\mathbf{y} \mid \mathbf{x}) = 1$ for the actual label $\mathbf{y}$.
Note that this is often impossible.
For example, there might be label noise in the dataset
(some examples may be mislabeled).
It may also not be possible when the input features
are not sufficiently informative
to classify every example perfectly.

나중에 설명할 이유로 :eqref:`eq_l_cross_entropy`의 손실 함수는 흔히 *크로스-인트로피 손실(cross-entropy loss)*라고 불립니다. $\mathbf{y}$는 길이가 $q$인 원-핫 벡터이기 때문에, 모든 원소 $j$에 대한 합은 한개의 항을 제외하고는 사라집니다. 모든 $\hat{y}_j는 예측된 확률들이기 떄문에, 그 것들의 로그 값은 $0$보다 커질 수 없습니다. 결과적으로 우리가 실제 레이블을 *확신을 갖고* 올바르게 예측한다면, 즉, 실제 레이블 $\mathbf{y}$에 대한 예측된 확률이 $P(\mathbf{y} \mid \mathbf{x}) = 1$일 경우,  손실 함수는 더 이상 최소화될 수 없습니다. 이것은 종종 불가능합니다. 예를 들면, 데이터셋에 노이즈가 있을 수 있습니다. (어떤 예제들이 잘못 레이블될 수 있습니다) 입력 피처들이 모든 예제들을 완벽하게 분류할 만큼 입력 피처가 충분히 유용하지 않은 경우에도 가능하지 않을 것입니다.

### Softmax and Derivatives
### 소프트맥스와 미분
:label:`subsec_softmax_and_derivatives`

Since the softmax and the corresponding loss are so common,
it is worth understanding a bit better how it is computed.
Plugging :eqref:`eq_softmax_y_and_o` into the definition of the loss
in :eqref:`eq_l_cross_entropy`
and using the definition of the softmax we obtain:

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

소프트맥스와 관련된 손실이 너무 흔하기 때문에 어떻게 계산되는지 조금 더 잘 이해하는 것은 가치가 있습니다. :eqref:`eq_softmax_y_and_o`를 :eqref:`eq_l_cross_entropy`의 손실 함수 정의에 대입하고, 소프트맥스 정의를 사용하면, 다음을 얻습니다.

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

To understand a bit better what is going on,
consider the derivative with respect to any logit $o_j$. We get

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$

무엇이 일어나고 있는지 좀 더 잘 이해하기 위해서, 어떤 로짓 $o_j$에 대한 미분을 고려해봅니다. 우리는 다음을 얻습니다.

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$

In other words, the derivative is the difference
between the probability assigned by our model,
as expressed by the softmax operation,
and what actually happened, as expressed by elements in the one-hot label vector.
In this sense, it is very similar to what we saw in regression,
where the gradient was the difference
between the observation $y$ and estimate $\hat{y}$.
This is not coincidence.
In any exponential family (see the
[online appendix on distributions](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/distributions.html)) model,
the gradients of the log-likelihood are given by precisely this term.
This fact makes computing gradients easy in practice.

다르게 설명하면, 미분은 소프트맥스 연산으로 표현된 모델이 할당한 확률과 원-핫 레이블 벡터의 원소로 표현된 실제로 일어난 것의 차입니다. 이런 의미에서 경사가 관찰 $y$와 추정 $\hat{y}$의 차였던 회귀에서 보았던 것과 비슷합니다. 이것은 우연이 아닙니다. 어떤 지수 패밀리In any exponential family (
[online appendix on distributions](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/distributions.html) 참고) 모델에서든 로그-가능도의 경사는 정확이 이 항목으로 주어집니다. 이 사실은 현실에서 경사 게산을 쉽게 만듭니다.

### Cross-Entropy Loss
### 크로스-엔트로피 손실

Now consider the case where we observe not just a single outcome
but an entire distribution over outcomes.
We can use the same representation as before for the label $\mathbf{y}$.
The only difference is that rather than a vector containing only binary entries,
say $(0, 0, 1)$, we now have a generic probability vector, say $(0.1, 0.2, 0.7)$.
The math that we used previously to define the loss $l$ 
in :eqref:`eq_l_cross_entropy`
still works out fine,
just that the interpretation is slightly more general.
It is the expected value of the loss for a distribution over labels.
This loss is called the *cross-entropy loss* and it is
one of the most commonly used losses for classification problems.
We can demystify the name by introducing just the basics of information theory.
If you wish to understand more details of information theory,
you may further refer to the [online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html).

자 이제는 하나의 결과에 대한 관찰을 하는 경우가 아니라, 결과들에 대한 전체 분포를 다루는 경우를 생각해봅시다. 레이블 $\mathbf{y}$에 대한 표기를 이전과 동일하게 사용할 수 있습니다. 오직 다른 점은 $(0, 0, 1)$ 과 같이 이진(binary) 값을 갖는 것이 아니라 $(0.1, 0.2, 0.7)$과 같이 일반적인 확률 벡터를 사용한다는 것입니다. :eqref:`eq_l_cross_entropy`의 손실 $l$ 정의도 동일한 수학을 사용하지만, 이에 대한 해석은 조금 더 일반적입니다. 레이블들의 분포에 대한 손실의 기대값을 의미합니다. 이 손실은 *크로스-엔트로피 손실*이라고 불리며, 분류 문제에서 가장 흔하게 사용되는 손실들 중 하나입니다. 우리는 정보 이론을 기본만을 소개함으로써 그 이름을 해독할 수 있다. 정보 이론에 대해서 더 자세한 내용을 알고 싶다면,  [online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html) 참고하길 바랍니다.

## Information Theory Basics
## 정보 이론 기초

*Information theory* deals with the problem of encoding, decoding, transmitting,
and manipulating information (also known as data) in as concise form as possible.
*정보 이론*은 가능한 간결한 형식으로 (데이터로 알려지기도 한)정보의 인코딩, 디코딩, 전달, 그리고 변조하는 문제를 다룹니다.

### Entropy
### 엔트로피

The central idea in information theory is to quantify the information content in data.
This quantity places a hard limit on our ability to compress the data.
In information theory, this quantity is called the *entropy* of a distribution $p$,
and it is captured by the following equation:

$$H[p] = \sum_j - p(j) \log p(j).$$
:eqlabel:`eq_softmax_reg_entropy`

정보 이론의 중심 아이디어는 데이터의 정보 내용을 정량화하는 것입니다. 이 정량화는 데이터를 압축하는 우리의 능력을 강하게 제한합니다. 정보 이론에서 이 정량화는 분포 $p$의 *엔트로피*라고 불리고, 다음 방정식에 의해서 표현됩니다. 

$$H[p] = \sum_j - p(j) \log p(j).$$
:eqlabel:`eq_softmax_reg_entropy`

One of the fundamental theorems of information theory states
that in order to encode data drawn randomly from the distribution $p$,
we need at least $H[p]$ "nats" to encode it.
If you wonder what a "nat" is, it is the equivalent of bit
but when using a code with base $e$ rather than one with base 2.
Thus, one nat is $\frac{1}{\log(2)} \approx 1.44$ bit.

정보 이론의 근본적인 이론 중에 하나로 분포 $p$ 로부터 임의로 추출된 데이터를 인코드하기 위해서는 최소  $H[p]$ 개의 'nat'이 필요하다는 것이 있습니다. 여기서 'nat'은 비트와 동일하나, 베이스 2가 아니라 베이스 $e$ 를 이용합니다. 따라서, 1 nat은 $\frac{1}{\log(2)} \approx 1.44$  비트입니다.

### Surprisal
### 놀라움

You might be wondering what compression has to do with prediction.
Imagine that we have a stream of data that we want to compress.
If it is always easy for us to predict the next token,
then this data is easy to compress!
Take the extreme example where every token in the stream always takes the same value.
That is a very boring data stream!
And not only it is boring, but it is also easy to predict.
Because they are always the same, we do not have to transmit any information
to communicate the contents of the stream.
Easy to predict, easy to compress.

압축이 예측과 무슨 관계가 있는지 궁금할 것입니다. 압축하기를 원하는 데이터 스트림이 있다고 상상해보세요. 만약 다음 토큰을 예측하는 것이 항상 쉽다면, 이 데이터는 입축하기 쉬울 것입니다! 스트림의 모든 토큰이 항상 같은 값을 갖는 극단적인 예를 들어봅니다. 이것은 아주 지루한 데이터 스트림입니다! 지루할 뿐만 아니라, 예측하기도 쉽습니다. 항상 같은 값이기 때문에, 스트림의 내용을 전달하기 위해서 어떤 정보도 전송할 필요가 없습니다. 예측하기가 쉬우면, 압축하기도 쉽습니다.

However if we cannot perfectly predict every event,
then we might sometimes be surprised.
Our surprise is greater when we assigned an event lower probability.
Claude Shannon settled on $\log \frac{1}{P(j)} = -\log P(j)$
to quantify one's *surprisal* at observing an event $j$
having assigned it a (subjective) probability $P(j)$.
The entropy defined in :eqref:`eq_softmax_reg_entropy` is then the *expected surprisal*
when one assigned the correct probabilities
that truly match the data-generating process.

하지만, 모든 이벤트를 완벽하게 예측하지 못한다면, 우리는 종종 놀라게 됩니다. 우리가 낮은 확률을 이벤트에 할당했었을 때 우리의 놀라움은 더 커집니다. Claude Shannon은 어떤 사람이 (주관적) 확률 $P(j)$를 할당 한 이벤트 $j$를 관찰했을 때의 *놀라움(surprisal)*를 정량화하기 위해서  $\log \frac{1}{P(j)} = -\log P(j)$를 정했습니다. 그러면, 어떤 사람이 데이터 생성 과정과 잘 일치하는 올바른 확률을 할당했을 때, :eqref:`eq_softmax_reg_entropy`에 정의된 엔트로피는 *기대 놀라움(expected surprisal)* 입니다.

### Cross-Entropy Revisited
### 크로스-엔트로피 다시 보기

So if entropy is level of surprise experienced
by someone who knows the true probability,
then you might be wondering, what is cross-entropy?
The cross-entropy *from* $p$ *to* $q$, denoted $H(p, q)$,
is the expected surprisal of an observer with subjective probabilities $q$
upon seeing data that were actually generated according to probabilities $p$.
The lowest possible cross-entropy is achieved when $p=q$.
In this case, the cross-entropy from $p$ to $q$ is $H(p, p)= H(p)$.

그러면, 만약에 엔트로피가 실제 확률을 아는 어떤 사람이 겪을 놀라움의 정도라면, 크로스-엔트로피는 무엇인지 궁금할 것입니다. $H(p, q)$로 표기되는 $p$*에서 부터* $q$*까지* 크로스-엔트로피는 확률 $p$에 따라서 실제로 만들어진 데이터를 주관적인 확률 $q$로 보는 관찰자의 기대 놀라움입니다. 가장 낮은 크로스-엔트로피틑 $p=q$일 때 얻어집니다. 이 경우, $p$ 부터 $q$까지의 크로스-엔트로피는 $H(p, p)= H(p)$가 됩니다.

In short, we can think of the cross-entropy classification objective
in two ways: (i) as maximizing the likelihood of the observed data;
and (ii) as minimizing our surprisal (and thus the number of bits)
required to communicate the labels.

간략하게 말해서, 크로스-엔트로피 뷴류 오브젝티브를 두 가지로 생각할 수 있습니다: (i) 관찰된 데이터의 가능도를 최대화하는 것으로, 그리고 (ii) 레이블 소통에 필요한 우리의 놀라움 (그리고 즉 비트의 수)를 최소화하는 것.

## Model Prediction and Evaluation
## 모델 예측과 평가

After training the softmax regression model, given any example features,
we can predict the probability of each output class.
Normally, we use the class with the highest predicted probability as the output class.
The prediction is correct if it is consistent with the actual class (label).
In the next part of the experiment,
we will use *accuracy* to evaluate the model's performance.
This is equal to the ratio between the number of correct predictions and the total number of predictions.

소프트맥스 회귀 모델을 학습한 후,  예제 피처들이 주어지면  각 출력 클래스에 대한 확률을 예측할 수 있습니다. 보통은 가장 큰 예측 확률값을 갖는 클래스를 출력 클래스로 사용합니다. 실제 클래스 (레이블)과 일치하면 예측을 정확한 것입니다. 실험의 다음 파트에서 우리는 모델의 성능을 평가하기 위해서 *정확도*를 사용할 것입니다. 이것은 정확한 예측의 수와 전체 예측의 비율과 같습니다.

## Summary
## 요약

* The softmax operation takes a vector and maps it into probabilities.
* Softmax regression applies to classification problems. It uses the probability distribution of the output class in the softmax operation.
* Cross-entropy is a good measure of the difference between two probability distributions. It measures the number of bits needed to encode the data given our model.

* 소프트맥스 연산은 벡터에 적용되고서 확률로 매핑한다.
* 소프트맥스 회귀는 분류 문제에 적용된다. 이것은 소프트맥스 연산의 결과 클래스의 활률 분포를 사용한다.
* 크로스-엔트로피는 두 확률 분포의 차이에 대한 좋은 측정이다. 이것은 모델이 주어졌을 때 데이터를 인코딩할 때 필요한 비트수를 측정한다.

## Exercises
## 연습문제

1. We can explore the connection between exponential families and the softmax in some more depth.
    * Compute the second derivative of the cross-entropy loss $l(\mathbf{y},\hat{\mathbf{y}})$ for the softmax.
    * Compute the variance of the distribution given by $\mathrm{softmax}(\mathbf{o})$ and show that it matches the second derivative computed above.
1. Assume that we have three classes which occur with equal probability, i.e., the probability vector is $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
    * What is the problem if we try to design a binary code for it?
    * Can you design a better code? Hint: what happens if we try to encode two independent observations? What if we encode $n$ observations jointly?
1. Softmax is a misnomer for the mapping introduced above (but everyone in deep learning uses it). The real softmax is defined as $\mathrm{RealSoftMax}(a, b) = \log (\exp(a) + \exp(b))$.
    * Prove that $\mathrm{RealSoftMax}(a, b) > \mathrm{max}(a, b)$.
    * Prove that this holds for $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b)$, provided that $\lambda > 0$.
    * Show that for $\lambda \to \infty$ we have $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a, b)$.
    * What does the soft-min look like?
    * Extend this to more than two numbers.

1. 지수승 집단과 소프트맥스 관계를 어느정도 심도있게 알아볼 수 있습니다.
    * 소프트맥스에 대한 크로스-엔트로피 손실 $l(\mathbf{y},\hat{\mathbf{y}})$의 이차 미분을 계산하세요.
    * $\mathrm{softmax}(\mathbf{o})$로 주어지는 분포의 분산을 계산하고, 위에서 계산한 이차 미분과 같은을 증명하세요.
1. 동일한 확률로 일어나는 3개 클래스가 있다고 가정합니다. 즉, 확률 벡터가 $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$ 입니다.
    * 이 문제를 위한 이진 코드(binary code)를 설계하고자 하면 어떤 문제가 있을까요?
    * 더 좋은 코드를 설계할 수 있나요? 힌트 - 두 독립적인 관찰을 인코딩하려면 어떤일이 생기나요? $n$개의 관찰을 연관해서 인코드하면 어떨까요?
1. Softmax는 위에서 소개된 매핑에 대한 잘못된 이름입니다 (하지만 딥러닝에서 많은 사람들이 쓰고 있습니다.) 실제 softmax는 $\mathrm{RealSoftMax}(a,b) = \log (\exp(a) + \exp(b))$ 로 정의됩니다.
    * $\mathrm{RealSoftMax}(a,b) > \mathrm{max}(a,b)$ 임을 증명하세요.
    * $\lambda > 0$ 일 경우, 모든 $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b)$ 에 대해서 이것이 성립함을 증명하세요
    * $\lambda \to \infty$ 이면, $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a,b)$ 임을 증명하세요.
    * soft-min은 어떻게 생겼을까요?
    * 이를 두개 이상의 숫자들로 확장해보세요.

[Discussions](https://discuss.d2l.ai/t/46)
