# Linear Regression Implementation from Scratch
# 선형 회귀를 처음부터 구현하기
:label:`sec_linear_scratch`

Now that you understand the key ideas behind linear regression,
we can begin to work through a hands-on implementation in code.
In this section, we will implement the entire method from scratch,
including the data pipeline, the model,
the loss function, and the minibatch stochastic gradient descent optimizer.
While modern deep learning frameworks can automate nearly all of this work,
implementing things from scratch is the only way
to make sure that you really know what you are doing.
Moreover, when it comes time to customize models,
defining our own layers or loss functions,
understanding how things work under the hood will prove handy.
In this section, we will rely only on tensors and auto differentiation.
Afterwards, we will introduce a more concise implementation,
taking advantage of bells and whistles of deep learning frameworks.

선행 회귀에 대한 주요 아이디어를 이해했으니, 코드로 직접 구현해 보겠습니다. 이 절에서 우리는 데이터 파이프라인, 모델, 손실 함수, 미니배치 확률적 경사 하강 최적화를 포함한 모든 것을 직접 구현할 것입니다. 최신 딥러닝 프레임워크들을 이 모든 것들을 거의 자동화해주지만, 직접 구현해보는 것이 우리가 무엇을 하는지 정확히 이해하는 유일한 방법입니다. 더군다나, 우리만의 층이나 손실 함수를 정의하면서 모델을 커스터마이즈해야 할 때, 이 것들이 어떻게 작동하는지를 이해하는 것이 도움이 될 것입니다. 이 절에서 우리는 텐서와 자동 미분만을 사용할 것입니다. 그리고 나서 딥러닝 프레임워크의 부속물들을 사용해서 더 간결한 구현을 소개하겠습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## Generating the Dataset
## 데이터셋 생성하기

To keep things simple, we will construct an artificial dataset
according to a linear model with additive noise.
Our task will be to recover this model's parameters
using the finite set of examples contained in our dataset.
We will keep the data low-dimensional so we can visualize it easily.
In the following code snippet, we generate a dataset
containing 1000 examples, each consisting of 2 features
sampled from a standard normal distribution.
Thus our synthetic dataset will be a matrix
$\mathbf{X}\in \mathbb{R}^{1000 \times 2}$.

간단하게 시작하기 위해서, 노이즈가 더해진 선형 모델에 대한 인위적인 데이터셋을 만들겠습니다. 우리가 할 일은 그 데이터셋에 포함된 유한개의 샘플들을 사용해서 모델의 파라미터들을 복원하는 것입니다. 데이터 시각화를 쉽게하기 위해서 데이터의 차원을 낮게하자. 아래 코드는 1000 개 샘플을 가진 데이터셋을 생성합니다. 각 샘플은 표준 정규 분포를 따르게 추출된 2개의 피처로 구성된다. 즉, 우리가 생성한 데이터셋은 행렬 $\mathbf{X}\in \mathbb{R}^{1000 \times 2}$이 됩니다.

The true parameters generating our dataset will be
$\mathbf{w} = [2, -3.4]^\top$ and $b = 4.2$,
and our synthetic labels will be assigned according
to the following linear model with the noise term $\epsilon$:

$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$

이 데이터셋을 만드는 실제 파라미터는 $\mathbf{w} = [2, -3.4]^\top$ 와 $b = 4.2$ 이고, 생성된 레이블은 다음 노이즈  $\epsilon$를 포함한 선형 모델을 따라 할당됩니다:

$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$

You could think of $\epsilon$ as capturing potential
measurement errors on the features and labels.
We will assume that the standard assumptions hold and thus
that $\epsilon$ obeys a normal distribution with mean of 0.
To make our problem easy, we will set its standard deviation to 0.01.
The following code generates our synthetic dataset.

$\epsilon$는 피처와 레이블의 잠재적인 오류로 생각할 수 있습니다. 여기에, 우리는 표준 가정을 따르고 $\epsilon$은 평균이 0인 정규 분포를 따른다고 가정합니다. 문제를 쉽게 만들기 위해서, 표준 편차를 0.01로 하겠습니다다. 합성 데이터셋을 만드는 코드는 다음과 같습니다.

```{.python .input}
#@tab mxnet, pytorch
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))
```

```{.python .input}
#@tab tensorflow
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = d2l.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = d2l.reshape(y, (-1, 1))
    return X, y
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

Note that each row in `features` consists of a 2-dimensional data point
and that each row in `labels` consists of a 1-dimensional label value (a scalar).

`features` 의 각 행은 2차 데이터 포인트,  `labels`의 각 행은 1차 레이블 값(스칼라)로 구성된다는 것을 기억하세요.

```{.python .input}
#@tab all
print('features:', features[0],'\nlabel:', labels[0])
```

By generating a scatter plot using the second feature `features[:, 1]` and `labels`,
we can clearly observe the linear correlation between the two.

두 번째 피처  `features[:, 1]` 와 `labels`를 사용해서 산점도(scatter plot)를 그려보면, 두 값들의 선형 상관관계를 명확하게 관찰할 수 있습니다.


```{.python .input}
#@tab all
d2l.set_figsize()
# The semicolon is for displaying the plot only
d2l.plt.scatter(d2l.numpy(features[:, 1]), d2l.numpy(labels), 1);
```

## Reading the Dataset
## 데이터셋 읽기

Recall that training models consists of
making multiple passes over the dataset,
grabbing one minibatch of examples at a time,
and using them to update our model.
Since this process is so fundamental
to training machine learning algorithms,
it is worth defining a utility function
to shuffle the dataset and access it in minibatches.

모델 학습은 데이터셋 전체를 여러번 반복해서 사용하는데, 샘플들의 미니배치를 한번에 하나씩 선택해서, 이를 모델 업데이트에 사용하는 것으로 구성되는 것을 떠올려 보세요. 이 과정은 머신러닝 알고리즘을 학습시키는데 아주 기본적인 절차이기 때문에, 데이터셋을 섞고 미니배치를 만드는 유틸리티 함수를 만드는 것이 가치가 있습니다.

In the following code, we define the `data_iter` function
to demonstrate one possible implementation of this functionality.
The function takes a batch size, a matrix of features,
and a vector of labels, yielding minibatches of the size `batch_size`.
Each minibatch consists of a tuple of features and labels.

아래 코드에서 이 기능을 제공하는 한 가지 방법을 구현하는  `data_iter` 함수를 정의합니다. 이 함수는 배치 크기, 피처들의 행렬, 그리고 레이블 벡터를 입력으로 받아서, `batch_size` 크기의 미니배치를 리턴한다. 각 미니배치는 피처와 레이블의 튜플로 구성됩니다.

```{.python .input}
#@tab mxnet, pytorch
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = d2l.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

```{.python .input}
#@tab tensorflow
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)
```

In general, note that we want to use reasonably sized minibatches
to take advantage of the GPU hardware,
which excels at parallelizing operations.
Because each example can be fed through our models in parallel
and the gradient of the loss function for each example can also be taken in parallel,
GPUs allow us to process hundreds of examples in scarcely more time
than it might take to process just a single example.

일반적으로 우리는 병렬 연산에 특화된 GPU 하드웨어의 이점을 살리기 위해서 합리적인 크기의 미니배치를 사용하기를 원합니다. 각 예제는 병렬로 모델에 입력될 수 있고, 각 예제의 손실 함수의 경사(gradient)도 병렬로 계산될 수 있기 때문에, GPU는 한 번에 한 예제만을 처리하는 것보다 수백개의 예제를 더 많은 시간에 처리할 수 있습니다.

To build some intuition, let us read and print
the first small batch of data examples.
The shape of the features in each minibatch tells us
both the minibatch size and the number of input features.
Likewise, our minibatch of labels will have a shape given by `batch_size`.

직관을 키우기 위해서, 데이터 샘플의 첫 번째 작은 배치를 읽고 출력해 보겠습니다. 각 미니배치의 샘플 모양은 미니배치 크기와 입력 피처들의 개수를 알려줍니다. 마찬가지로, 레이블 미니배치는 `batch_size` 모양을 갖습니다.

```{.python .input}
#@tab all
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

As we run the iteration, we obtain distinct minibatches
successively until the entire dataset has been exhausted (try this).
While the iteration implemented above is good for didactic purposes,
it is inefficient in ways that might get us in trouble on real problems.
For example, it requires that we load all the data in memory
and that we perform lots of random memory access.
The built-in iterators implemented in a deep learning framework
are considerably more efficient and they can deal
with both data stored in files and data fed via data streams.

반복을 수행하면서 우리는 전체 데이터가 모두 소진될 때까지 서로 다른 미니배치를 얻습니다. 위에서 구현한 반복 코드는 교육적인 목적으로는 충분하지만, 실제 문제를 다룰 때는 문제를 일이킬 만큼 비효율적입니다. 예를 들어, 모든 데이터를 메모리로 읽어들인 후, 랜덤 메모리 접근을 많이 수행해야 합니다. 딥러닝 프레임워크에 구현되어 있는 반복자(iterator)들은 상당히 더 효과적이고, 파일시스템에 저장된 데이터나 데이터 스트림을 통해서 얻어오는 데이터도 다룰 수 있습니다.

## Initializing Model Parameters
## 모델 파라미터들 초기화하기

Before we can begin optimizing our model's parameters by minibatch stochastic gradient descent,
we need to have some parameters in the first place.
In the following code, we initialize weights by sampling
random numbers from a normal distribution with mean 0
and a standard deviation of 0.01, and setting the bias to 0.

미니배치 확률적 경사 하강법을 사용해서 모델 파라이터들 최적화를 시작하기 전에, 우리는 우선 파라미터 값을 갖고 있어야 합니다. 아래 코드는 평균이 0이고 표준 편차가 0.01인 정규 분포를 따라 선택한 값으로 가중치를 초기화하고, 편향은 0으로 초기화합니다.

```{.python .input}
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)
```

After initializing our parameters,
our next task is to update them until
they fit our data sufficiently well.
Each update requires taking the gradient
of our loss function with respect to the parameters.
Given this gradient, we can update each parameter
in the direction that may reduce the loss.

파라미터들을 초기화한 후, 우리의 다음 과제는 이 파라미터들이 데이터를 충분히 잘 들어맞을 때까지 업데이트하는 것입니다. 각 업데이트를 하기 위해서는 파라미터에 대한 손실 함수의 경사값의 계산이 필요합니다. 경사값이 구해지면, 손실을 줄이는 방향으로 각 파라미터를 업데이트합니다.

Since nobody wants to compute gradients explicitly
(this is tedious and error prone),
we use automatic differentiation,
as introduced in :numref:`sec_autograd`, to compute the gradient.

누구도 경사값을 직접 계산하기를 원하지 않기 때문에 (이는 지루하고 오류가 발생하기 쉽습니다), 우리는 :numref:`sec_autograd`에서 소개된 자동 미분을 사용해서 경사값을 계산합니다.

## Defining the Model
## 모델 정의하기

Next, we must define our model,
relating its inputs and parameters to its outputs.
Recall that to calculate the output of the linear model,
we simply take the matrix-vector dot product
of the input features $\mathbf{X}$ and the model weights $\mathbf{w}$,
and add the offset $b$ to each example.
Note that below $\mathbf{Xw}$  is a vector and $b$ is a scalar.
Recall the broadcasting mechanism as described in :numref:`subsec_broadcasting`.
When we add a vector and a scalar,
the scalar is added to each component of the vector.

다음으로 입력들과 파라미터들을 출력과 연관시키는 모델을 정의해야 합니다. 선형 모델의 출력을 얻기 위해서 입력 피처 행렬 $\mathbf{X}$과 모델 가중치 벡터 $\mathbf{w}$의 형렬-벡터 곱을 구한 후, 오프셋 $b$를 각 예제에 더한다는 것을 상기해 보세요. $\mathbf{Xw}$는 벡터이고, $b$는 스칼라 값이다. :numref:`subsec_broadcasting` 에서 설명한 브로드케스팅 메카니즘에 따라서, 벡터와 스칼라를 더할 때, 스칼라 값는 벡터의 각 컴포넌트에 더해집니다.

```{.python .input}
#@tab all
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return d2l.matmul(X, w) + b
```

## Defining the Loss Function
## 손실 함수 정의하기

Since updating our model requires taking
the gradient of our loss function,
we ought to define the loss function first.
Here we will use the squared loss function
as described in :numref:`sec_linear_regression`.
In the implementation, we need to transform the true value `y`
into the predicted value's shape `y_hat`.
The result returned by the following function
will also have the same shape as `y_hat`.

모델을 업데이트하는 것은 손실 함수의 미분을 구하는 것이기 때문에, 우리는 손실 함수를 우선 정의해야 합니다. 여기서 우리는 :numref:`sec_linear_regression`에서 소개한 제곱 손실 함수를 사용하겠습니다. 이를 구현할 때, 실제 값 `y`를 예상 값,`y_hat`,의 모양과 동일하게 변경할 필요가 있습니다. 다음 함수의 리턴 값은 `y_hat`와 같은 모양을 갖습니다.

```{.python .input}
#@tab all
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
```

## Defining the Optimization Algorithm
## 최적화 알고리즘 정의하기

As we discussed in :numref:`sec_linear_regression`,
linear regression has a closed-form solution.
However, this is not a book about linear regression:
it is a book about deep learning.
Since none of the other models that this book introduces
can be solved analytically, we will take this opportunity to introduce your first working example of 
minibatch stochastic gradient descent.

:numref:`sec_linear_regression`에서 설명한 것처럼 선형 회귀는 닫힌 형태의 해법이 있습니다. 하지만, 이 책은 선형 회귀에 대한 책이 아니라, 딥러닝에 대한 책입니다. 이 책에서 소개하는 어떤 모델도 분석인 방법으로 풀리지 않기 때문에, 이 기회에 미니배치 확률적 경사 하강법의 첫 번째 동작 예를 소개하겠습니다.

At each step, using one minibatch randomly drawn from our dataset,
we will estimate the gradient of the loss with respect to our parameters.
Next, we will update our parameters
in the direction that may reduce the loss.
The following code applies the minibatch stochastic gradient descent update,
given a set of parameters, a learning rate, and a batch size.
The size of the update step is determined by the learning rate `lr`.
Because our loss is calculated as a sum over the minibatch of examples,
we normalize our step size by the batch size (`batch_size`),
so that the magnitude of a typical step size
does not depend heavily on our choice of the batch size.

매 단계 마다 데이터셋에서 임의로 추출한 미니배치 한 개를 사용해서 파라미터에 대한 손실의 미분값을 추정할 것입니다. 그 다음, 손실을 줄이는 방향으로 파라미터를 업데이트합니다. 아래 코드는 파라미터, 학습 속도 그리고 배치 크기의 세트가 주어졌을 때, 미니배치 확률적 경사 하강법을 이용한 업데이트를 구현합니다. 업데이트 스탭의 크기는 학습 속도 `lr`에 의해서 정해집니다. 손실은 예제들의 미니배치에 대한 합으로 계산되기 때문에, 배치 크기(`batch_size`)로 스탭 크기를 정규화하는데, 이는 전형적인 스탭 크기의 정도가 배치 크기의 선택에 크게 영향받지 않도록 해줍니다.

```{.python .input}
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

```{.python .input}
#@tab pytorch
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param.data.sub_(lr*param.grad/batch_size)
        param.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)
```  param.assign_sub(lr*grad/batch_size)
```

## Training
## 학습

Now that we have all of the parts in place,
we are ready to implement the main training loop.
It is crucial that you understand this code
because you will see nearly identical training loops
over and over again throughout your career in deep learning.

자 이제 모든 것들이 준비되었으니, 메인 학습 룹을 구현할 차례입니다. 여러분의 딥러닝 경력을 걸처셔 거의 동일한 학습 룹을 계속해서 볼 것이기 때문에, 이 코드를 이해하는 것이 매우 중요합니다.

In each iteration, we will grab a minibatch of training examples,
and pass them through our model to obtain a set of predictions.
After calculating the loss, we initiate the backwards pass through the network,
storing the gradients with respect to each parameter.
Finally, we will call the optimization algorithm `sgd`
to update the model parameters.

매 반복 마다, 우리는 학습 샘플들의 미니배치를 얻고, 이를 모델에 입력해서 예측들을 얻습니다. 손실 값들을 계산하고, 네트워크에 걸쳐서 백워드 패스를 수행하면서 각 파라미터에 대한 미분값을 저장합니다. 마지막으로, 우리는 모델 파라미터들을 업데이트하기 위해서 최적화 알고리즘 `sgd`을 호출합니다.

In summary, we will execute the following loop:

* Initialize parameters $(\mathbf{w}, b)$
* Repeat until done
    * Compute gradient $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * Update parameters $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

요약하면, 다음 룹을 수행한다:

* 파라미터 $(\mathbf{w}, b)$를 초기화한다.
* 끝날 때까지 다음을 반복한다
    * 미분 계산, $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * 파라미터 업데이트, $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

In each *epoch*,
we will iterate through the entire dataset
(using the `data_iter` function) once
passing through every example in the training dataset
(assuming that the number of examples is divisible by the batch size).
The number of epochs `num_epochs` and the learning rate `lr` are both hyperparameters,
which we set here to 3 and 0.03, respectively.
Unfortunately, setting hyperparameters is tricky
and requires some adjustment by trial and error.
We elide these details for now but revise them
later in
:numref:`chap_optimization`.

매 *에폭(epoch)*마다, (`data_iter` 함수를 이용해서) 학습 데이터셋의 모든 예제를 한 번씩 사용(전체 예제들의 개수가 배치 크기의 배수라고 가정)해서 전체 데이터셋을 반복합니다. 에폭 수 `num_epochs`와 학습 속도  `lr`는 모두 하이퍼파라미터이며, 이들은 여기서 3과 0.03으로 각각 설정했습니다. 불행히도 하이퍼파라미터를 설정하는 것은 까다롭고, 시행 착오를 통한 조정이 필요합니다. 지금은 이것에 대한 자세한 설명은 하지 않고, :numref:`chap_optimization`에서 다루겠습니다.

```{.python .input}
#@tab all
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
```

```{.python .input}
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Because `l` has a shape (`batch_size`, 1) and is not a scalar
        # variable, the elements in `l` are added together to obtain a new
        # variable, on which gradients with respect to [`w`, `b`] are computed
        l.backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab pytorch
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab tensorflow
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on l with respect to [`w`, `b`]
        dw, db = g.gradient(l, [w, b])
        # Update parameters using their gradient
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')
```

In this case, because we synthesized the dataset ourselves,
we know precisely what the true parameters are.
Thus, we can evaluate our success in training
by comparing the true parameters
with those that we learned through our training loop.
Indeed they turn out to be very close to each other.

이 경우에 우리가 직접 데이터셋을 합성했기 때문에 진짜 파라미터가 무엇인지 정확하게 알고 있습니다. 따라서, 우리는 실제 파라미터값과 학습 룹을 통해서 배운 파라미터들을 비교해서 학습이 잘 되었는지를 평가할 수 있습니다. 실제로 학습된 파라미터와 실제 파라미터가 서로 매우 비슷합니다.

```{.python .input}
#@tab all
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
print(f'error in estimating b: {true_b - b}')
```

Note that we should not take it for granted
that we are able to recover the parameters perfectly.
However, in machine learning, we are typically less concerned
with recovering true underlying parameters,
and more concerned with parameters that lead to highly accurate prediction.
Fortunately, even on difficult optimization problems,
stochastic gradient descent can often find remarkably good solutions,
owing partly to the fact that, for deep networks,
there exist many configurations of the parameters
that lead to highly accurate prediction.

우리는 파라미터들을 완벽하게 복원할 수 있다는 것을 당연하게 여겨서는 안된다는 점에 유의하세요. 하지만, 머신러닝에서 우리는 잠재되어 있는 실제 파라미터를 복원하는 것에 관심이 크지 않고, 높은 정확도로 예측을 하는 파라미터들에 더 관심이 있습니다. 다행스럽게도 어려운 최적화 문제에서도 립러닝의 경우 매우 정확한 예측을 만들어내는 파라미터들의 조합이 많이 존재한다는 일부 사실 때문에 확률적 경사 하강법이 종종 훌륭한 솔루션을 찾아냅니다.

## Summary
## 요약

* We saw how a deep network can be implemented and optimized from scratch, using just tensors and auto differentiation, without any need for defining layers or fancy optimizers.
* This section only scratches the surface of what is possible. In the following sections, we will describe additional models based on the concepts that we have just introduced and learn how to implement them more concisely.

* 우리는 층을 정의하거나 멋진 최적화 방법이 필요없이 단지 텐서와 자동 미분만으로 딥러닝 네트워크를 처음부터 구현하고 최적화를 하는 방법을 알아봤습니다.
* 이 절은 가능한 것들의 일부만 다뤘습니다. 다음 절들에서 우리는 지금까지 소개한 개념들을 기반으로 추가적인 모델들을 설명하고, 간결하게 구현하는 방법을 알아보겠습다.

## Exercises
## 연습문제

1. What would happen if we were to initialize the weights to zero. Would the algorithm still work?
1. Assume that you are
   [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) trying to come up
   with a model between voltage and current. Can you use auto differentiation to learn the parameters of your model?
1. Can you use [Planck's Law](https://en.wikipedia.org/wiki/Planck%27s_law) to determine the temperature of an object using spectral energy density?
1. What are the problems you might encounter if you wanted to  compute the second derivatives? How would you fix them?
1.  Why is the `reshape` function needed in the `squared_loss` function?
1. Experiment using different learning rates to find out how fast the loss function value drops.
1. If the number of examples cannot be divided by the batch size, what happens to the `data_iter` function's behavior?


1. 가중치를 모두 0으로 초기화하면 어떻게 될까? 알고리즘이 동작할까요?
1. 여러분이 전압과 전류의 모델을 만들고자 하는 [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) 라고 가정합니다. 모델을 파라미터를 학습하기 위해서 자동 미분을 사용할 수 있나요?
1. 스팩트럼 에너지 밀도를 사용하여 물제의 온도를 결정하기 위해서 [Planck's Law](https://en.wikipedia.org/wiki/Planck%27s_law)를 사용할 수 있나요?
1. 2차 미문들 사용해야 하는 경우 만날 수 있는 문제가 무엇이 있을까? 어떻게 그 문제를 해결할 수 있나요?
1. `squared_loss` 함수에서 `reshape` 함수가 필요한가요?
1. 다른 학습 속도를 사용했을 때, 손실 함수 값이 얼마나 빠르게 떨어지는지 실험해 봅시다.
1. 만약 예제들의 개수가 배치 크기로 나눠 떨어지지 않을 경우, `data_iter`  함수의 동작에 어떤 일이 발생할까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:
