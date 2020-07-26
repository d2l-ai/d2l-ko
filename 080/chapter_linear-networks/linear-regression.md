# Linear Regression
# 선형 회귀
:label:`sec_linear_regression`

*Regression* refers to a set of methods for modeling
the relationship between one or more independent variables
and a dependent variable.
In the natural sciences and social sciences,
the purpose of regression is most often to
*characterize* the relationship between the inputs and outputs.
Machine learning, on the other hand,
is most often concerned with *prediction*.

*회귀(regression)*는 하나 또는 그 이상의 독립 변수들과 종속 변수의 관계를 모델링하는 방법들을 지칭합니다. 자연 과학이나 사회 과학에서의 회귀의 목적은 입력값과 결과값의 관계를 *특징(characterize)* 짓는데 주로 사용됩니다. 반면, 머신 러닝에서는 이 관계에 대한 *예측(prediction)*에 더 관점을 두고 있습니다.

Regression problems pop up whenever we want to predict a numerical value.
Common examples include predicting prices (of homes, stocks, etc.),
predicting length of stay (for patients in the hospital),
demand forecasting (for retail sales), among countless others.
Not every prediction problem is a classic regression problem.
In subsequent sections, we will introduce classification problems,
where the goal is to predict membership among a set of categories.

회귀 문제는 어떤 수를 예측하고자 하는 여러 곳에서 발견될 수 있는데, 일반적인 예로는 (주택 또는 주식 등의) 가격을 예측하기, 환자의 입원 일수 예측하기, 또는 소매 판매에 대한 수요를 예측하는 것 등이 있습니다. 하지만 모든 문제가 고전적인 회귀 문제는 아닙니다. 이 장에서는 또한 분류 문제(classification problem)도 다룰 예정입니다. 주어진 카테고리들 중에 어떤 것에 속하는지를 예측하는 것을 목표로 하는 것을 분류 문제라고 합니다.


## Basic Elements of Linear Regression
## 선형 회귀의 기본 요소들

*Linear regression* may be both the simplest
and most popular among the standard tools to regression.
Dating back to the dawn of the 19th century,
linear regression flows from a few simple assumptions.
First, we assume that the relationship between
the independent variables $\mathbf{x}$ and the dependent variable $y$ is linear,
i.e., that $y$ can be expressed as a weighted sum
of the elements in $\mathbf{x}$,
given some noise on the observations.
Second, we assume that any noise is well-behaved
(following a Gaussian distribution).

*선형 회귀*는 회귀에 대한 표준 도구들 중에서 가장 간단하고 가장 유명한 방법일 것입니다. 19세기 초반에 몇 가지 간단한 가정에서 선형 회귀가 만들어졌습니다. 첫 번째 가정은 독립 변수들  $\mathbf{x}$와 종속 변수 $y$의 관계가 선형적이라는 것입니다. 즉, 관찰들에서 약간의 노이즈가 있다는 전재하에 $y$는  $\mathbf{x}$ 원소들의 가중치를 적용한 합으로 표현될 수 있다는 것입니다. 두 번째 가정은 노이즈는 가우시안 분포(Gaussian distribution)를 잘 따른다는 것입니다.

To motivate the approach, let us start with a running example.
Suppose that we wish to estimate the prices of houses (in dollars)
based on their area (in square feet) and age (in years).
To actually fit a model for predicting house prices,
we would need to get our hands on a dataset
consisting of sales for which we know
the sale price, area, and age for each home.
In the terminology of machine learning,
the dataset is called a *training dataset* or *training set*,
and each row (here the data corresponding to one sale)
is called an *example* (or *data instance*, *data point*, *sample*).
The thing we are trying to predict (price)
is called a *label* (or *target*).
The independent variables (age and area)
upon which the predictions are based
are called *features* (or *covariates*).

실제 예제를 통해서 어떤 접근 방법을 취하는 것인지 알아보겠습니다. (제곱 피트 단위의) 집의 면적과 (년수로 표현된) 집의 년차가 주어졌을 때 (달러로 표현된) 집 가격을 예측하고자 합니다. 집 가격을 예측하는 모델을 만들기 위해서는 가격, 면적, 연차가 알려진 각 주택 판매에 대한 데이터셋을 확보해야합니다. 머신 러닝의 용어로는 이 데이터셋을 *학습 데이터(training data)* 또는 *학습 세트(training set)*라고 하고, 하나의 판매에 대한 데이터인 각 행은 *예제(example)* (또는 *데이터 포인트*, *데이터 인스턴스*, 또는 *샘플*)라고 합니다. 예측하고자 하는 것(가격)을 *레이블(label)* (또는 *타겟(target)*) 이라고 부릅니다. 예측에 사용되는 독립 변수들(년수와 면적) 은 *피처(feature)* 또는 *공변량(covariate)*라고 합니다.

Typically, we will use $n$ to denote
the number of examples in our dataset.
We index the data instances by $i$, denoting each input
as $\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}]^\top$
and the corresponding label as $y^{(i)}$.

일반적으로 데이터셋의 예제의 수는 $n$로 표현합니다. 데이터 포인트들은 $i$로 색인을 하는데, 즉 각 입력 데이터 포인트를  $x^{(i)} = [x_1^{(i)}, x_2^{(i)}]$로 이 입력에 대한 레이블은 $y^{(i)}$로 표기합니다.

### Linear Model
### 선형 모델
:label:`subsec_linear_model`

The linearity assumption just says that the target (price)
can be expressed as a weighted sum of the features (area and age):

간단하게 말하자면, 선형성의 가정은 타겟(가격)이 특성들(면적과 연수)의 가중합으로 표현될 수 있다는 의미합니다.

$$\mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b.$$
:eqlabel:`eq_price-area`

$$\mathrm{가격} = w_{\mathrm{면적}} \cdot \mathrm{면적} + w_{\mathrm{연수}} \cdot \mathrm{연수} + b$$

In :eqref:`eq_price-area`, $w_{\mathrm{area}}$ and $w_{\mathrm{age}}$
are called *weights*, and $b$ is called a *bias*
(also called an *offset* or *intercept*).
The weights determine the influence of each feature
on our prediction and the bias just says
what value the predicted price should take
when all of the features take value 0.
Even if we will never see any homes with zero area,
or that are precisely zero years old,
we still need the bias or else we will
limit the expressivity of our model.
Strictly speaking, :eqref:`eq_price-area` is an *affine transformation*
of input features,
which is characterized by
a *linear transformation* of features via weighted sum, combined with 
a *translation* via the added bias.

:eqref:`eq_price-area`에서 , $w_{\mathrm{area}}$와 $w_{\mathrm{age}}$는 *가중치(weight)*, $b$는 *편향(bais)*라고 합니다. (편향은 *오프셋(offset)* 또는 *절편(intercept)* 라고도 합니다.) 가중치들은 각 피처(feature)가 예측에 미치는 영향 정도를 결정하고, 편향은 모든 피처가 $0$일 경우 예측된 가격이 얼마가 되어야 하는지를 말해줍니다. 사실 집의 면적이 $0$이거나 연수가 $0$인 경우는 없지만, 편향은 여전히 필요합니다. 그렇지 않다면, 선형 모델이 표현할 수 있는 것들에 제약이 생기기 때문입니다. 정확하게 말하자만, :eqref:`eq_price-area`는 입력 피처들의 *아핀 변환(affine transformation)*이고, 이 변환은 가중합을 통한 피처들의 *선형 변환*과 추가적인 편향을 통한 *이동(translation)*으로 정의됩니다.

Given a dataset, our goal is to choose
the weights $\mathbf{w}$ and the bias $b$ such that on average,
the predictions made according to our model
best fit the true prices observed in the data.
Models whose output prediction
is determined by the affine transformation of input features
are *linear models*,
where the affine transformation is specified by the chosen weights and bias.

우리의 목표는 모델이 주어진 데이터셋에 있는 관찰된 실제 가격을 평균적으로 가장 잘 예측할 수 있는 가중치 $\mathbf{w}$와 편향 $b$를 선택하는 것입니다. 출력 예측이 입력 피처의 아핀 변환으로 결정되는 모델들은 선택된 가중치와 편향으로 정의되는 *선형 모델*입니다. 

In disciplines where it is common to focus
on datasets with just a few features,
explicitly expressing models long-form like this is common.
In machine learning, we usually work with high-dimensional datasets,
so it is more convenient to employ linear algebra notation.
When our inputs consist of $d$ features,
we express our prediction $\hat{y}$ (in general the "hat" symbol denotes estimates) as

$$\hat{y} = w_1  x_1 + ... + w_d  x_d + b.$$

주로 적은 수의 피처들을 가진 데이터셋에 집중하는 분야들의 경우에는 모델을 다음과 같은 긴 수식으로 직접 표현하는 것이 일반적입니다. 머신 러닝에서는 주로 고 차원(high dimensional)의 데이터셋을 다루기에, 선형 대수 표기법을 사용하는 것이 보다 간편합니다. 만약 입력이 $d$ 개의 피처들로 구성되어 있다면, 예측 $\hat{y}$(일반적으로 "햇(모자)" 기호는 예측을 표기함)을 아래와 같이 표기합니다.

$$\hat{y} = w_1 \cdot x_1 + ... + w_d \cdot x_d + b$$

Collecting all features into a vector $\mathbf{x} \in \mathbb{R}^d$
and all weights into a vector $\mathbf{w} \in \mathbb{R}^d$,
we can express our model compactly using a dot product:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$
:eqlabel:`eq_linreg-y`

모든 피처들을 벡터 $\mathbf{x} \in \mathbb{R}^d$로 합치고, 모든 가중치들을 벡터 $\mathbf{w} \in \mathbb{R}^d$로 합치면, 닷 곱(dot product)로 모델을 간결하게 표현할 수 있습니다.

In :eqref:`eq_linreg-y`, the vector $\mathbf{x}$ corresponds to features of a single data point.
We will often find it convenient
to refer to features of our entire dataset of $n$ examples
via the *design matrix* $\mathbf{X} \in \mathbb{R}^{n \times d}$.
Here, $\mathbf{X}$ contains one row for every example
and one column for every feature.

:eqref:`eq_linreg-y`에서 벡터 $\mathbf{x}$는 한 개의 데이터 포인트의 피처에 해당합니다. $n$ 개 예제들로 구성된 전체 데이터셋에 대한 피처를 *계획 행렬(design matrix)*  $\mathbf{X} \in \mathbb{R}^{n \times d}$ 형태로 표현하는 것이 편리하다는 것을 알게될 것입니다. 즉, $X$의 각 행은 각 샘플을 각 열은 각 피처(feature)를 나타냅니다.

For a collection of features $\mathbf{X}$,
the predictions $\hat{\mathbf{y}} \in \mathbb{R}^n$
can be expressed via the matrix-vector product:

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,$$

where broadcasting (see :numref:`subsec_broadcasting`) is applied during the summation.
Given features of a training dataset $\mathbf{X}$
and corresponding (known) labels $\mathbf{y}$,
the goal of linear regression is to find
the weight vector $\mathbf{w}$ and the bias term $b$
that given features of a new data point
sampled from the same distribution as $\mathbf{X}$,
the new data point's label will (in expectation) be predicted with the lowest error.

예를 들여, 데이터 포인들의 집합이 $\mathbf{X}$이며, 예측들 $\hat{\mathbf{y}} \in \mathbb{R}^n$은 행렬-벡터의 곱으로 다음과 같이 표현됩니다.

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b$$

여기서 합을 할 때는 브로드케스팅( :numref:`subsec_broadcasting` 참조)이 적용됩니다. 학습 데이터셋 $\mathbf{X}$의 피처들과 그에 대응하는 (알려진) 레이블 $\mathbf{y}$가 주어졌을 때, $\mathbf{X}$ 와 같은 분포에서 뽑아진 새로운 데이터 포인트의 피처들에 대해서 낮은 오류로 레이블들을 예측할 수 있는 가중치 벡터 $\mathbf{w}$와 편향항 $b$를 찾는 것이 선형 회귀의 목표입니다.

Even if we believe that the best model for
predicting $y$ given $\mathbf{x}$ is linear,
we would not expect to find a real-world dataset of $n$ examples where
$y^{(i)}$ exactly equals $\mathbf{w}^\top \mathbf{x}^{(i)}+b$
for all $1 \leq i \leq n$.
For example, whatever instruments we use to observe
the features $\mathbf{X}$ and labels $\mathbf{y}$
might suffer small amount of measurement error.
Thus, even when we are confident
that the underlying relationship is linear,
we will incorporate a noise term to account for such errors.

우리가 $\mathbf{x}$에 대해서 $y$를 예측하는 최고의 모델이 선형이라고 믿을지라도, 모든  $1 \leq i \leq n$에 대해서, $y^{(i)}$가  정확히 $\mathbf{w}^\top \mathbf{x}^{(i)}+b$ 와 일치하는 $n$ 개의 예제를 갖는 데이터셋을 현실에서 찾을 수 없습니다. 예를 들어 피처 $\mathbf{X}$와 레이블 $\mathbf{y}$를 관찰하기 위해서 어떤 측정 방법을 사용하더라도 작은 양의 측정 오류를 피할 수 없습니다. 따라서, 실제 관계가 선형이라고 확신을 하는 경우에라도 이런 오류를 반영하기 위한 노이즈 항을 사용해야 합니다.

Before we can go about searching for the best *parameters* (or *model parameters*) $\mathbf{w}$ and $b$,
we will need two more things:
(i) a quality measure for some given model;
and (ii) a procedure for updating the model to improve its quality.

가장 좋은 *파라메터(또는 모델 파라메터)* $\mathbf{w}$와 $b$,를 찾기에 앞서 우리는 두 가지가 더 필요합니다: (i) 주어진 모델에 대한 품질 측정과 (ii) 품질을 향상시키기 위해서 모델을 업데이트하는 절차.

### Loss Function
### 손실(Loss) 함수

Before we start thinking about how *to fit* our model,
we need to determine a measure of *fitness*.
The *loss function* quantifies the distance
between the *real* and *predicted* value of the target.
The loss will usually be a non-negative number
where smaller values are better
and perfect predictions incur a loss of 0.
The most popular loss function in regression problems
is the squared error.
When our prediction for an example $i$ is $\hat{y}^{(i)}$
and the corresponding true label is $y^{(i)}$,
the squared error is given by:

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

모델을 *학습(fit)* 시키는 방법을 생각하기 전에 우리는 *학습의 정도(fitness)*를 측정하는 방법을 정해야 합니다. *손실 함수(loss function)*은 타겟의 *실제* 값과 *예측된* 값의 차이를 측정하는 함수입이다. 보통 손실은 음수가 아닌 숫자로 표현되고, 작은 값일 수록 더 좋고, 완벽한 예측은 손실 값이 0이 됩니다. 회쉬 문제에서 가장 유명한 손실 함수는 제곱 오류(squared error)이다. 예제 $i$에 대한 예측이 $\hat{y}^{(i)}$이고 실제 값이 $y^{(i)}$ 인 경우, 제곱 오류는 다음과 같이 계산됩니다.

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

The constant $\frac{1}{2}$ makes no real difference
but will prove notationally convenient,
canceling out when we take the derivative of the loss.
Since the training dataset is given to us, and thus out of our control,
the empirical error is only a function of the model parameters.
To make things more concrete, consider the example below
where we plot a regression problem for a one-dimensional case
as shown in :numref:`fig_fit_linreg`.

![Fit data with a linear model.](../img/fit_linreg.svg)
:label:`fig_fit_linreg`

상수값  $\frac{1}{2}$ 는 실제 차이를 만들지는 않지만, 손실을 미분했을 때 사라지는 효과가 있어서 표기상의 편리함을 주고 있습니다. 학습 데이터셋은 주어지는 것이지 우리가 제어할 수 있는 것이 아니기 때문에 경험적인 오류는 모델 파라미터들의 함수가 됩니다. 더 정확하게 하기 위해서 예를 들어보면, 1차원 케이스에 대한 회귀 문제를 그래프로 그려보면 :numref:`fig_fit_linreg`과 같습니다.

![Fit data with a linear model.](../img/fit_linreg.svg)
:label:`fig_fit_linreg`

Note that large differences between
estimates $\hat{y}^{(i)}$ and observations $y^{(i)}$
lead to even larger contributions to the loss,
due to the quadratic dependence.
To measure the quality of a model on the entire dataset of $n$ examples,
we simply average (or equivalently, sum)
the losses on the training set.

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

예측값 $\hat{y}^{(i)}$과 관찰된 값 $y^{(i)}$의 차이가 크면 2차 의존성으로 인해서 손실에 더 크게 반영되는 것을 확인할 수 있습다. $n$ 개의 예제들로 구성된 전체 데이터셋에 대한 모델의 품질을 측정은 학습셋에 대한 손실에 대한 평균 (또는 합)으로 할 수 있습니다.

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

When training the model, we want to find parameters ($\mathbf{w}^*, b^*$)
that minimize the total loss across all training examples:

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

모델을 학습하는 것은 모든 학습 예제들에 대한 전체 손실을 최소화하는 파라미터들 ($\mathbf{w}^*, b^*$) 찾는 것입니다.

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### Analytic Solution
### 분석적 솔루션

Linear regression happens to be an unusually simple optimization problem.
Unlike most other models that we will encounter in this book,
linear regression can be solved analytically by applying a simple formula.
To start, we can subsume the bias $b$ into the parameter $\mathbf{w}$
by appending a column to the design matrix consisting of all ones.
Then our prediction problem is to minimize $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$.
There is just one critical point on the loss surface
and it corresponds to the minimum of the loss over the entire domain.
Taking the derivative of the loss with respect to $\mathbf{w}$
and setting it equal to zero yields the analytic (closed-form) solution:

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}.$$

선형 회귀는 아주 간단한 최적화 문제입니다. 이 책에서 다루게될 대부분 모델들과 다르게, 선형 회귀는 간단한 방적식을 적용해서 분석적으로 풀 수가 있습니다. 우선, 편향 $b$를 파라미터 $\mathbf{w}$에 포함시킬 수 있는데, 방법은 모두 1로 이뤄진 설계 행렬에 편향을 새로운 열로 추가하는 것입다. 이제 예측 문제는 $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$를 최소화하는 것이 됩니다. 손실 평면에는 단 하나의 임계점(critical point)가 있는데, 이 점이 바로 전체 영역에서 손실을 최소화하는 것에 해당됩니다.  $\mathbf{w}$에 대해서 손실 함수를 미분을 구하고 이를 0에 대해서 풀면, (닫힌 형태의) 분석적 해법이 됩니다.

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}.$$

While simple problems like linear regression
may admit analytic solutions,
you should not get used to such good fortune.
Although analytic solutions allow for nice mathematical analysis,
the requirement of an analytic solution is so restrictive
that it would exclude all of deep learning.

선형 회귀 같은 간단한 문제들은 분석적인 해법이 있을 수 있지만, 이런 행운을 기대해서는 안된다. 분석적 해법이 훌륭한 수학적 분석을 가능하게 할 수 있지만, 분석적인 해법에 대한 요구 사항은 너무 제한적이기 때문에 딥러닝에는 적용할 수 없습니다.

### Minibatch Stochastic Gradient Descent
### 미니배치 확률적 경사 하강법(Stochastic Gradient Descent)

Even in cases where we cannot solve the models analytically,
it turns out that we can still train models effectively in practice.
Moreover, for many tasks, those difficult-to-optimize models
turn out to be so much better that figuring out how to train them
ends up being well worth the trouble.

분석적인 방법으로 모델을 구할 수 없는 경우에도 우리는 실제로는 모델을 효과적으로 학습시킬 수 있습니다. 더군다나 많은 과제들에서 최적화하기 어려운 모델들은 이 모델들을 학습시키는 방법을 알아내는 것이 가치있는 문제가 되기도 합니다.

The key technique for optimizing nearly any deep learning model,
and which we will call upon throughout this book,
consists of iteratively reducing the error
by updating the parameters in the direction
that incrementally lowers the loss function.
This algorithm is called *gradient descent*.

이 책의 전반에 결쳐서 사용하게 될, 그리고 거의 모든 딥러닝 모델을 최적화하는 주요 기법은 끊임없이 손실 함수를 낮추는 방향으로 파라미터들을 업데이트하면서 오류를 반복적으로 줄이는 것으로 구성됩니다. 이 알고리즘을 *경사 하강법(gradient descent)*라고 합니다.

The most naive application of gradient descent
consists of taking the derivative of the loss function,
which is an average of the losses computed
on every single example in the dataset.
In practice, this can be extremely slow:
we must pass over the entire dataset before making a single update.
Thus, we will often settle for sampling a random minibatch of examples
every time we need to compute the update,
a variant called *minibatch stochastic gradient descent*.

경사 하강법의 가장 단순한 적용법은 손실 함수의 미분을 취하는 것으로 구성할 수 있는데, 이는 데이터셋의 각 예제들에 대한 손실의 평균값이다. 실제로 이 방법은 매우 느립니다: 모든 데이터셋을 사용해서 한 번의 업데이트를 하기 때문입니다. 따라서, 업데이트를 계산할 때 임의로 선택한 예제들의 미니배체를 사용하는데, 이를 *미니배치 확률적 경사 하강법(minibatch stochastic gradient descent)*라고 합니다.

In each iteration, we first randomly sample a minibatch $\mathcal{B}$
consisting of a fixed number of training examples.
We then compute the derivative (gradient) of the average loss
on the minibatch with regard to the model parameters.
Finally, we multiply the gradient by a predetermined positive value $\eta$
and subtract the resulting term from the current parameter values.

반복할 때 마다 우리는 고정된 개수의 학습 예제들로 구성된 미니배치를 임의로  $\mathcal{B}$ 추출합니다. 그리고, 미니배치에 대한 평균 손실을 모델 파라미터들에 대해서 미분을 계산합니다. 최종적으로 우리는 사전에 정의된 양수 값  $\eta$를 미분값에 곱한 값을 현재 파라미터 값에서 뺍니다.

We can express the update mathematically as follows
($\partial$ denotes the partial derivative):

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

업데이트를 수핟적으로 표현하면 다음과 같습니다. ($\partial$는 편미분을 의미합니다.)

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

To summarize, steps of the algorithm are the following:
(i) we initialize the values of the model parameters, typically at random;
(ii) we iteratively sample random minibatches from the data,
updating the parameters in the direction of the negative gradient.
For quadratic losses and linear functions,
we can write this out explicitly as follows:

$$\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

요약하면, 알고리즘은 다음 순서로 구성됩니다.
(i) 보통은 난수를 사용해서 모델 파라미터들의 값을 초기화합니다.
(ii) 데이터에서 임의의 미니배치를 추출하고, 음의 경사(negative gradient) 방향으로 파라미터를 업데이트하는 것을 반복합니다. 2차 손실과 선형 함수에 대해서 다음과 같이 표현할 수 있습니다.

$$\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

Note that $\mathbf{w}$ and $\mathbf{x}$ are vectors in :eqref:`eq_linreg_batch_update`.
Here, the more elegant vector notation makes the math
much more readable than expressing things in terms of coefficients,
say $w_1, w_2, \ldots, w_d$.
The set cardinality
$|\mathcal{B}|$ represents
the number of examples in each minibatch (the *batch size*)
and $\eta$ denotes the *learning rate*.
We emphasize that the values of the batch size and learning rate
are manually pre-specified and not typically learned through model training.
These parameters that are tunable but not updated
in the training loop are called *hyperparameters*.
*Hyperparameter tuning* is the process by which hyperparameters are chosen,
and typically requires that we adjust them
based on the results of the training loop
as assessed on a separate *validation dataset* (or *validation set*).

:eqref:`eq_linreg_batch_update`에서  $\mathbf{w}$와 $\mathbf{x}$는 벡터이다.  $w_1, w_2, \ldots, w_d$를 계수로 사용해서 표현하는 것보다, 더 우아한 벡터 표기법을 사용하면 수식이 더 읽기 쉬워집니다. $|\mathcal{B}|$ 으로 표현되는 집한 크기는 각 미니배치의 예제 수 (*배치 크기(batch size)*)를 나타낸다. 배치 크기와 학습 속도는 모델 학습을 통해서 학습되는 값이 아니라 사전에 직접 설정하는 값들임을 알아둡시다. 설정을 할 수 있지만 학습 과정에서 업데이트되지 않는 파라미터들을 *하이퍼파라미터(hyperparameter)* 라고 합니다. *하아퍼파라미터 튜닝(hyperparameter tuning)*은 하이퍼파라미터들을 선택하고, 별도로 준비된 *검증 데이터셋(validation dataset)*에 학습된 모델을 적용한 결과에 따라 다시 값을 조정하는 과정입니다.

After training for some predetermined number of iterations
(or until some other stopping criteria are met),
we record the estimated model parameters,
denoted $\hat{\mathbf{w}}, \hat{b}$.
Note that even if our function is truly linear and noiseless,
these parameters will not be the exact minimizers of the loss
because, although the algorithm converges slowly towards the minimizers
it cannot achieve it exactly in a finite number of steps.

미리 정의된 회수만큼 학습을 반복한 후 (또는 학습 종료 조건을 충족할 때까지 학습을 수행한 후), 추정된 모델 파라미터 $\hat{\mathbf{w}}, \hat{b}$ 를 저장한다. 우리의 함수가 완전히 선형이고 노이즈가 없다고 해도, 이 파라미터들은 손실을 정확하게 최소화하지 않을 것임을 기억합시다. 그 이유는, 이 알고리즘은 최소값으로 천천히 수렴지만, 정해한 만큼 반복해서는 도달할 수 없기 때문입니다.

Linear regression happens to be a learning problem where there is only one minimum
over the entire domain.
However, for more complicated models, like deep networks,
the loss surfaces contain many minima.
Fortunately, for reasons that are not yet fully understood,
deep learning practitioners seldom struggle to find parameters
that minimize the loss *on training sets*.
The more formidable task is to find parameters
that will achieve low loss on data
that we have not seen before,
a challenge called *generalization*.
We return to these topics throughout the book.

선형 회귀는 전체 영역에 대해서 단 하나의 최소점이 존재하는 학습 문제입니다. 하지만, 딥 네트워크와 같이 보다 복잡한 모델의 경우 손실 표면은 여러 최소값을 갖습니다. 다행하게도 아직은 완전히 이해할 수 없는 이유로 딥 러닝 실무자들은  *학습 데이터셋*에 대한 손실을 최소화하는 파라미터를 찾기 위해서 고생을 하는 경우는 적습니다. 더 어려운 일은 우리가 보지 못한 데이터셋에 대해서 낮은 손실을 달성하는 파라미터를 찾는 것입니다. 이를 *일반화(generalization)*이라고 한다. 이 책 전반에서 우리는 이 주제들에 대해서 다시 다루겠습니다.

### Making Predictions with the Learned Model
### 학습된 모델로 예측하기

Given the learned linear regression model
$\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$,
we can now estimate the price of a new house
(not contained in the training data)
given its area $x_1$ and age $x_2$.
Estimating targets given features is
commonly called *prediction* or *inference*.

학습된 선형 회귀 모델, $\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$,을 사용해서 (학습 데이터셋에 포함되지 않은) 새로운 집에 대한 면적 $x_1$ 와 연수 $x_2$을 사용해서 가격을 예측할 수 있습니다. 주어진 피처에 대해서 타겟을 예측하는 것을 일반적으로 *예측(prediction)* 또는 *추론(inference)*라고 부릅니다.

We will try to stick with *prediction* because
calling this step *inference*,
despite emerging as standard jargon in deep learning,
is somewhat of a misnomer.
In statistics, *inference* more often denotes
estimating parameters based on a dataset.
This misuse of terminology is a common source of confusion
when deep learning practitioners talk to statisticians.

*추론*이 딥러닝에서 표준 용어로 사용되고 있지만 이 용어는 다소 잘못된 것이기 때문에, 우리는 *예측*이라는 용어를 사용하겠다. 통계학에서 *추론*은 데이터셋이 주어졌을 때 파라미터들을 추정하는 것을 가르킵니다. 딥러닝 전문가가 통계학자와 이야기를 할 때, 이 용어의 잘못된 사용이 혼란을 가져오는 일반적인 원인입니다.

## Vectorization for Speed
## 속도를 위한 벡터화

When training our models, we typically want to process
whole minibatches of examples simultaneously.
Doing this efficiently requires that we vectorize the calculations
and leverage fast linear algebra libraries
rather than writing costly for-loops in Python.

모델을 학습 시킬 때, 일반적으로 예제들의 미니배치 전체를 동시에 처리하기를 바랍니다. 이를 효과적으로 수행하기 위해서는 연산을 벡터화하고, Python의 비싼 for-룹을 사용하는 것 보다는 빠른 선형 대수 라이브러리를 사용하는 것이 필요합니다. 

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np
import time
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import numpy as np
import time
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
import numpy as np
import time
```

To illustrate why this matters so much,
we can consider two methods for adding vectors.
To start we instantiate two 10000-dimensional vectors
containing all ones.
In one method we will loop over the vectors with a Python for-loop.
In the other method we will rely on a single call to `+`.

이 것이 왜 중요한지 설명하기 위해서, 벡터들을 더하는 두 가지 방법을 생각해 봅시다. 우선 1000차원 벡터 두 개를 값을 모두 1로 설정해서 만듭니다. 첫 번째 방법은 Python의 for-룹에서 벡터의 원소를 하나씩 더하는 것이고, 다른 방법은 `+` 연산을 한 번 사용하는 것입니다.

```{.python .input}
#@tab all
n = 10000
a = d2l.ones(n)
b = d2l.ones(n)
```

Since we will benchmark the running time frequently in this book,
let us define a timer.

우리는 이 책에서 벤치마크를 자주 수행할 것이기 때문에 타이버를 정의하겠습니다.

```{.python .input}
#@tab all
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
```

Now we can benchmark the workloads.
First, we add them, one coordinate at a time,
using a for-loop.

자 이제 워크로드의 속도를 측정할 수 있습니다. 우선, for-룹 안에서 벡터의 요소 하나씩 더해봅니다.

```{.python .input}
#@tab mxnet, pytorch
c = d2l.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'
```

```{.python .input}
#@tab tensorflow
c = tf.Variable(d2l.zeros(n))
timer = Timer()
for i in range(n):
    c[i].assign(a[i] + b[i])
f'{timer.stop():.5f} sec'
```

Alternatively, we rely on the reloaded `+` operator to compute the elementwise sum.

다른 방법으로, 우리는 리로드된 `+` 연산자를 사용해서 원소별 합을 구해봅시다.

```{.python .input}
#@tab all
timer.start()
d = a + b
f'{timer.stop():.5f} sec'
```

You probably noticed that the second method
is dramatically faster than the first.
Vectorizing code often yields order-of-magnitude speedups.
Moreover, we push more of the mathematics to the library
and need not write as many calculations ourselves,
reducing the potential for errors.

아마 두 번째 방법이 첫 번째 방법보다 상당히 빠르다는 것을 확인했을 것입니다. 벡터화 코드를 사용하면 종종 속도를 상당히 향상시킬 수 있습니다. 더구나 많은 수학 연산을 라이브러리를 사용해서 할 수 있고, 계산하는 코드를 직접 만드는 일을 할 필요하가 없으며, 계산 코드를 직접 작성하면서 발생할 수 있는 잠재적인 오류도 줄일 수 있습니다.

## The Normal Distribution and Squared Loss
## 표준 분포(normal distribution)과 제곱 손실(squared loss)
:label:`subsec_normal_distribution_and_squared_loss`

While you can already get your hands dirty using only the information above,
in the following we can more formally motivate the square loss objective
via assumptions about the distribution of noise.

지금까지의 이야기한 내용만으로도 충분히 실험을 했겠지만, 손실 분포에 대한 가능으로 제곱 손실 오브젝티브에 대해서 좀 더 설명해 보겠습니다.

Linear regression was invented by Gauss in 1795,
who also discovered the normal distribution (also called the *Gaussian*).
It turns out that the connection between
the normal distribution and linear regression
runs deeper than common parentage.
To refresh your memory, the probability density
of a normal distribution with mean $\mu$ and variance $\sigma^2$ (standard deviation $\sigma$)
is given as

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

선형 회귀는 *가우시안*이라고 불리기도 하는 정규 분포(normal distribution)를 발견한 가우스(Gauss)에 의해서 1795년에서 발명되었습니다. 정규 분포와 선형 회귀와의 연관성은 일반적인 종속 관계보다 더 깊다는 것이 밝혀졌습니다. 여러분의 기억을 되집어보면, 평균이 $\mu$이고 분산이 $\sigma^2$ (표준 편차는 $\sigma$)인 정규 분포의 확률 밀도는 다음과 같습니다.

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

Below we define a Python function to compute the normal distribution.

아래 코드는 정규 분표를 계산하는 Python 함수를 정의합니다.

```{.python .input}
#@tab all
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
```

We can now visualize the normal distributions.

이제 우리는 정규 분포를 시각화할 수 있습니다.

```{.python .input}
#@tab all
# Use numpy again for visualization
x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

As we can see, changing the mean corresponds to a shift along the $x$-axis,
and increasing the variance spreads the distribution out, lowering its peak.

그림에서 보이듯이 평균을 바꾸면 $x$-축을 따라서 이동하고, 분산을 크게하면 그래프를 옆으로 더 펼쳐지게 하고, 최대점은 낮아짐니다.

One way to motivate linear regression with the mean squared error loss function (or simply square loss)
is to formally assume that observations arise from noisy observations,
where the noise is normally distributed as follows:

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \text{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

평균 제곱 오류 손실 함수(또는 간단하게 제곱 손실)를 사용한 선형 회귀에 대한 접근 중에 하나는  다음과 같은 정규 분포를 따르는 노이즈가 있는 관찰로 부터 관찰 값들이 나왔다는 가정을 하는 것입니다.

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \text{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

Thus, we can now write out the *likelihood*
of seeing a particular $y$ for a given $\mathbf{x}$ via

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

따라서 어떤 $\mathbf{x}$에 대해서 특정한 $y$를 관찰 할 *가능도(likelihood)*를 다음과 같습니다.

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

Now, according to the principle of maximum likelihood,
the best values of parameters $\mathbf{w}$ and $b$ are those
that maximize the *likelihood* of the entire dataset:

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)}).$$

그렇게 되면, 최대 가능도 원칙(the principle of maximum likelihood)에 따르면 파라미터 $\mathbf{w}$ 와 $b$의 가장 좋은 값은 전체 데이터셋에 대한 *가능도(likelihood)*를 최대화하는 값입니다.

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)}).$$

Estimators chosen according to the principle of maximum likelihood
are called *maximum likelihood estimators*.
While, maximizing the product of many exponential functions,
might look difficult,
we can simplify things significantly, without changing the objective,
by maximizing the log of the likelihood instead.
For historical reasons, optimizations are more often expressed
as minimization rather than maximization.
So, without changing anything we can minimize the *negative log-likelihood*
$-\log P(\mathbf y \mid \mathbf X)$.
Working out the mathematics gives us:

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

최대 가능도 원칙에 따라서 선택된 추정(estimators, 파라미터)는 *최대 가능도 추정(maximum likelihood estimator)*라고 합니다. 많은 제곱 함수들의 곱을 최대화하는 것은 어려워 보이는데, 가능도에 로그를 적용하면 오브젝티브를 변경하지 않고도 이를 아주 간단하게 만들 수 있습니다. 역사적인 이유로 최적화는 최대화보다는 최소화로 종종 표현됩니다. 따라서, 우리는 아무것도 바꾸지 않고 *음의 로그-가능도(negative log-likelihood)* $-\log P(\mathbf y \mid \mathbf X)$.를 최소화한다. 이를 수학식으로 표현하면 다음과 같습니다.

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

Now we just need one more assumption that $\sigma$ is some fixed constant.
Thus we can ignore the first term because
it does not depend on $\mathbf{w}$ or $b$.
Now the second term is identical to the squared error loss introduced earlier,
except for the multiplicative constant $\frac{1}{\sigma^2}$.
Fortunately, the solution does not depend on $\sigma$.
It follows that minimizing the mean squared error
is equivalent to maximum likelihood estimation
of a linear model under the assumption of additive Gaussian noise.

$\sigma$가 고정된 상수라는 가정이 하나 더 필요합니다. 즉,  $\mathbf{w}$ 또는 $b$에 의존하는 값이 아니기 때문에 첫 번째 항을 무시할 수 있습니다. 그럼 두 번째 항은 곱하는 상수 값 $\frac{1}{\sigma^2}$를 제외하고는 이미 소개한 제곱 오류 손실과 동일합니다. 따라서, 평균 제곱 오류를 최소화하는 것은 가산적 가우스 노이즈(additive Gaussian noise) 가정하에 선형 모델에 대한 가능도 추정을 최대화하는 것과 동일합니다.

## From Linear Regression to Deep Networks
## 선형 회귀에서 딥 네트워크로

So far we only talked about linear models.
While neural networks cover a much richer family of models,
we can begin thinking of the linear model
as a neural network by expressing it in the language of neural networks.
To begin, let us start by rewriting things in a "layer" notation.

지금까지 우리는 선형 모델에 대해서만 이야기했습니다. 뉴럴 네트워크는 아주 다양한 모델의 종류들을 포함하고 있지만, 우리는 선형 모델을 뉴럴 네트워크 용어를 사용해서 뉴럴 네트워크로 여기는 것으로 시작하겠습니다. 모든 것들을 "층(layer)" 표기로 다시 적는 것부터 시작합니다.

### Neural Network Diagram
### 뉴럴 네트워크 다이어그램

Deep learning practitioners like to draw diagrams
to visualize what is happening in their models.
In :numref:`fig_single_neuron`,
we depict our linear regression model as a neural network.
Note that these diagrams highlight the connectivity pattern
such as how each input is connected to the output,
but not the values taken by the weights or biases.

![Linear regression is a single-layer neural network.](../img/singleneuron.svg)
:label:`fig_single_neuron`

딥러닝 실무자들은 그들의 모델에서 어떤 일이 일어나는지를 시각화하기 위해서 다이어그램을 그리는 것을 좋아합니다. :numref:`fig_single_neuron`에서 우리는 선형 회귀 모델을 뉴럴 네트워크로 표현했습니다. 이 다이어그램들은 입력과 출력이 어떻게 연결되는지에 대한 연결 패턴을 중요하게 표현하는 반면, 가충치와 편향의 값이 무엇인지는 관심이 없다는 것을 기억하세요.

![Linear regression is a single-layer neural network.](../img/singleneuron.svg)
:label:`fig_single_neuron`

For the neural network shown in :numref:`fig_single_neuron`,
the inputs are $x_1, \ldots, x_d$,
so the *number of inputs* (or *feature dimensionality*) in the input layer is $d$.
The output of the network in :numref:`fig_single_neuron` is $o_1$,
so the *number of outputs* in the output layer is 1.
Note that the input values are all *given*
and there is just a single *computed* neuron.
Focusing on where computation takes place,
conventionally we do not consider the input layer when counting layers.
That is to say,
the *number of layers* for the neural network in :numref:`fig_single_neuron` is 1.
We can think of linear regression models as neural networks
consisting of just a single artificial neuron,
or as single-layer neural networks.

:numref:`fig_single_neuron`에 표현된 뉴럴 네트워크에서  $x_1, \ldots, x_d$ 이 입력들입니다. 즉, 입력 층의 *입력의 개수* (또는 *피처 차원(feature dimensionality)*)은 $d$ 입니다. :numref:`fig_single_neuron` 의 네트워크 출력은 $o_1$ 이고, 출력 층의 *출력 개수*는 1입니다. 입력 값들은 *주어진 것*이고, *계산되는* 뉴런은 단 한 개임을 주의하세요. 연산이 일어나는 곳들에만 집중으로 하기에, 입력 층은 네트워크 층의 개수를 셀 때 관습적으로 포함하지 않습니다. 다시 말하자면, :numref:`fig_single_neuron` 의 뉴럴 네트워크의 *층의 개수*는 1입니다. 우리는 선형 회귀를 단 한 개 뉴런으로 구성된 뉴럴 네트워크 또는 단-층 뉴럴 네트워크로 생각할 수 있습니다.

Since for linear regression, every input is connected
to every output (in this case there is only one output),
we can regard this transformation (the output layer in :numref:`fig_single_neuron`)
as a *fully-connected layer* or *dense layer*.
We will talk a lot more about networks composed of such layers
in the next chapter.

선형 회귀에서 모든 입력값이 모든 출력(이 경우는 한 개의 출력)과 연결되어 있기 떄문에, 우리는 이 변환(:numref:`fig_single_neuron` 의 출력 층)을 *완전 연결 층(fully-connected layer)* 또는 *덴스 층(dense layer)*로 여길 수 있습니다. 다음 장에서 이런 층들로 구성된 네트워크에 대해서 더 많이 이야기할 예정입니다.

### Biology
### 생물학

Since linear regression (invented in 1795)
predates computational neuroscience,
it might seem anachronistic to describe
linear regression as a neural network.
To see why linear models were a natural place to begin
when the cyberneticists/neurophysiologists
Warren McCulloch and Walter Pitts began to develop
models of artificial neurons,
consider the cartoonish picture
of a biological neuron in :numref:`fig_Neuron`, consisting of
*dendrites* (input terminals),
the *nucleus* (CPU), the *axon* (output wire),
and the *axon terminals* (output terminals),
enabling connections to other neurons via *synapses*.

![The real neuron.](../img/Neuron.svg)
:label:`fig_Neuron`

선형 회귀는 (1795년에 발명됨)은 계산 신경과학(computaional neuroscience)보다 먼저 나왔기 때문에, 선형 회귀를 뉴럴 네트워크로 설명하는 것이 시대 착오적으로 보일 수 있습니다. 인공두뇌학자/신경 생리학자인 워렌 맥철록(Warren McCulloch)와 월터 피츠(Walter Pitt)가 인공 뉴런들의 모델을 만들기 시작했을 때, 왜 선형 모델이 시작하기 자연스러운 것이 었는지를 보기 위해서, :numref:`fig_Neuron` 에 있는 생물학적 뉴런의 만화 그림을 들어보겠습니다. 이는 *수상돌기(dendrites)*(입력 단자), *핵(nucleus)*(CPU), *축삭(axon)*(출력 와이어), 그리고 *출삭 단자(axon terminal)*(출력 단자)로 구성되며, *시넵스(synapse)*를 통해서 다른 뉴런에 연결될 수 있습니다.

![The real neuron.](../img/Neuron.svg)
:label:`fig_Neuron`

Information $x_i$ arriving from other neurons
(or environmental sensors such as the retina)
is received in the dendrites.
In particular, that information is weighted by *synaptic weights* $w_i$
determining the effect of the inputs
(e.g., activation or inhibition via the product $x_i w_i$).
The weighted inputs arriving from multiple sources
are aggregated in the nucleus as a weighted sum $y = \sum_i x_i w_i + b$,
and this information is then sent for further processing in the axon $y$,
typically after some nonlinear processing via $\sigma(y)$.
From there it either reaches its destination (e.g., a muscle)
or is fed into another neuron via its dendrites.

다른 뉴런들(혹은 망막과 같은 환경 신경 센서)에서 온 정보 $x_i$는 수상 돌기로 전달됩니다. 구체적으로는 그 정보는 *시넵스 가중치* $w_i$로 가중치가 매겨지고, 이는 입력의 영향을 결정합니다. (예를 들어, $x_i w_i$ 곱을 통한 활성화 또는 억제). 여러 소스들로 부터 전달된 가중치가 적용된 입력들을 핵에서 $y = \sum_i x_i w_i + b$인 가중합으로 모아지고, 보통은 $\sigma(y)$와 같은 비선형 프로세스를 적용됩니다. 여기서부터 이 값들은 (근육과 같은) 목적지에 도착하거나, 그 뉴런의 수상돌기를 통해서 다른 뉴런으로 보내집니다.

Certainly, the high-level idea that many such units
could be cobbled together with the right connectivity
and right learning algorithm,
to produce far more interesting and complex behavior
than any one neuron alone could express
owes to our study of real biological neural systems.

물론, 그런 많은 단위들이 올바른 연결성과 올바른 학습 알고리즘으로 함께 뭉쳐져서 하나의 뉴런이 표현할 수 있는 것도다 훨씬 더 흥미롭고 복잡한 행동을 만들어 낼 수 있다는 높은 수준의 생각은 실제 생물학적인 뉴럴 시스템의 연구에 덕입니다.

At the same time, most research in deep learning today
draws little direct inspiration in neuroscience.
We invoke Stuart Russell and Peter Norvig who,
in their classic AI text book
*Artificial Intelligence: A Modern Approach* :cite:`Russell.Norvig.2016`,
pointed out that although airplanes might have been *inspired* by birds,
ornithology has not been the primary driver
of aeronautics innovation for some centuries.
Likewise, inspiration in deep learning these days
comes in equal or greater measure from mathematics,
statistics, and computer science.

동시에 최근 딥러닝 연구들은 신경과학에서 직접적인 영감을 거의 얻지 않습니다. 스튜어트 러셀(Stuart Russel)과 피터 노르빅(Peter Norvig)은 그들의 고적 AI 교과서 *Artificial Intelligence: A Modern Approach* :cite:`Russell.Norvig.2016` 에서 비행기가 새들을 부터 *영감*을 얻었을지라도, 조류학은 몇 세기 동안 항공 혁신의 주요 동력이 되지 않았음을 지적했습니다. 마찬가지로 요즘 딥러닝의 영감은 수학, 통계, 컴퓨터 과학에서 같거나 더 많이 나옵니다.

## Summary
## 요약

* Key ingredients in a machine learning model are training data, a loss function, an optimization algorithm, and quite obviously, the model itself.
* Vectorizing makes everything better (mostly math) and faster (mostly code).
* Minimizing an objective function and performing maximum likelihood estimation can mean the same thing.
* Linear regression models are neural networks, too.

* 머신러닝의 주요 요소는 학습 데이터, 손실 함수, 최적화 알고리즘, 그리고 아주 당연하게도 모델 그 자체입니다.
* 벡터화는 모든 것을 좋게(대부분 수학) 그리고 빠르게(대부분 코드) 만듭니다.
* 오브젝티브 함수를 최소화하는 것과 최대 가능도 추정을 수행하는 것은 같은 것을 의미합니다.
* 선형 회귀 모델 역시 뉴럴 네트워크입니다.

## Exercises

1. Assume that we have some data $x_1, \ldots, x_n \in \mathbb{R}$. Our goal is to find a constant $b$ such that $\sum_i (x_i - b)^2$ is minimized.
    * Find a analytic solution for the optimal value of $b$.
    * How does this problem and its solution relate to the normal distribution?
1. Derive the analytic solution to the optimization problem for linear regression with squared error. To keep things simple, you can omit the bias $b$ from the problem (we can do this in principled fashion by adding one column to $\mathbf X$ consisting of all ones).
    * Write out the optimization problem in matrix and vector notation (treat all the data as a single matrix, and all the target values as a single vector).
    * Compute the gradient of the loss with respect to $w$.
    * Find the analytic solution by setting the gradient equal to zero and solving the matrix equation.
    * When might this be better than using stochastic gradient descent? When might this method break?
1. Assume that the noise model governing the additive noise $\epsilon$ is the exponential distribution. That is, $p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$.
    * Write out the negative log-likelihood of the data under the model $-\log P(\mathbf y \mid \mathbf X)$.
    * Can you find a closed form solution?
    * Suggest a stochastic gradient descent algorithm to solve this problem. What could possibly go wrong (hint: what happens near the stationary point as we keep on updating the parameters)? Can you fix this?

1. 데이터 $x_1, \ldots, x_n \in \mathbb{R}$가 있다고 가정하자. 목표는 $\sum_i (x_i - b)^2$ 최소화하는 상수 $b$를 찾는 것입니다.
    * 최적화 값 $b$를 구하는 분석적 해법을 찾아보세요.
    * 이 문제와 그 해법은 정규 분포화 어떻게 관련이 있나요?
1. 제곱 오류를 사용해서 선형 회귀를 최적화하는 문제의 분석적 해법을 도출하세요. 간단하게 하기 위해서, 문제에서 편향 $b$는 생략 할 수 있습니다다. (생략하지 않고, 모두 1로 구성된 $\mathbf X$ 에 열을 하나 추가하면 그대로 풀수도 있다)
    * 최적화 문제를 행렬과 벡터 표기로 적어봅니다.(모든 데이터를 하나의 행렬로 간주하고, 모든 타겟 값은 하나의 벡터로 간주하자)
    * $w$에 대해서 손실의 경사(gradient)를 계산합니다.
    * 경사(gradient)가 0인 행렬 방정식을 풀어서 분석적 해법을 구합니다.
    * 어떤 경우에 이 방법이 확률적 경사 하강법(stochastic gradient descent)보다 좋을까요? 어떤 경우에 이 방법이 실패할까요?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/40)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/258)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/259)
:end_tab:
