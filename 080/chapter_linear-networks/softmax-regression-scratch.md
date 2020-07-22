# Implementation of Softmax Regression from Scratch
# 소프트맥스 회귀를 처음부터 구현하기
:label:`sec_softmax_scratch`

Just as we implemented linear regression from scratch,
we believe that softmax regression
is similarly fundamental and you ought to know
the gory details of how to implement it yourself.

선형 회귀를 처음부터 구현했던 것처럼 우리는 소프트맥스 회귀도 기본적으로 동일하다고 생각하며, 이를 직접 구현하는 방법에 대한 세부적인 내용을 알아야합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
from IPython import display
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from IPython import display
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

We will work with the Fashion-MNIST dataset, just introduced in :numref:`sec_fashion_mnist`,
setting up a data iterator with batch size 256.

우리는 :numref:`sec_fashion_mnist`에서 소개한 Fashion-MNIST 데이터셋을 사용할 것이며, 데이터 반복자는 배치 크기가 256으로 설정할 것입니다.

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Initializing Model Parameters
## 모델 파라미터 초기화

As in our linear regression example,
each example here will be represented by a fixed-length vector.
Each example in the raw dataset is a $28 \times 28$ image.
In this section, we will flatten each image,
treating them as vectors of length 784.
In the future, we will talk about more sophisticated strategies
for exploiting the spatial structure in images,
but for now we treat each pixel location as just another feature.

선형 회귀 예제와 동일하게 각 예제는 고정된 길이의 벡터로 표현될 것입니다. 원시 데이터셋의 각 예제는 $28 \times 28$ 이미지입니다. 이 절에서 우리는 이미지를 평평하게 만들어서 길이가 784인 벡터로 다룰 것입니다. 향후에 우리는 이미지의 공간적인 특징을 활용하기 위한 보다 복잡한 전략을 논의하겠지만, 우선은 각 픽셀의 위치는 단지 다른 피처로 다루겠습니다.

Recall that in softmax regression,
we have as many outputs as there are classes.
Because our dataset has 10 classes,
our network will have an output dimension of 10.
Consequently, our weights will constitute a $784 \times 10$ matrix
and the biases will constitute a $1 \times 10$ row vector.
As with linear regression, we will initialize our weights `W`
with Gaussian noise and our biases to take the initial value 0.

소프트맥스 회귀에서 클래스의 개수만큼 출력이 있었습니다. 우리의 데이터셋은 10개의 클래스를 가지고 있으니, 네트워크의 출력 차원은 10입니다. 결과적으로 가중치는 $784 \times 10$ 행렬이고, 편향은 $1 \times 10$인 행-벡터로 구성됩니다. 선형 회귀처럼 우리는 가충치 `W`를 가우시안 노이즈를 따르는 값으로 초기화하고, 편향은 0으로 초기값을 설정합니다.

```{.python .input}
num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
num_inputs = 784
num_outputs = 10

W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                 mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))
```

## Defining the Softmax Operation
## 소프트맥스 연산 정의

Before implementing the softmax regression model,
let us briefly review how the sum operator work
along specific dimensions in a tensor,
as discussed in :numref:`subseq_lin-alg-reduction` and :numref:`subseq_lin-alg-non-reduction`.
Given a matrix `X` we can sum over all elements (by default) or only
over elements in the same axis, 
i.e., the same column (axis 0) or the same row (axis 1).
Note that if `X` is an tensor with shape (2, 3)
and we sum over the columns,
the result will be a vector with shape (3,).
When invoking the sum operator,
we can specify to keep the number of axes in the original tensor,
rather than collapsing out the dimension that we summed over.
This will result in a two-dimensional tensor with shape (1, 3).

소프트맥스 회귀 모델을 구현하기에 앞서서 :numref:`subseq_lin-alg-reduction`와 :numref:`subseq_lin-alg-non-reduction`에서 알아봤던 텐서의 특정 차원에 따라서 합 연산이 어떻게 동작하는지를 간단하게 복습해 보겠습니다. 행렬 `X`이 주어졌을 때, 우리는 (기본 설정) 모든 원소를 더하거나 같은 축(예를 들면 축 0은 같은 열, 축 1은 같은 행)에 있는 원소들만 더할 수 있습니다. 만약 `X`가 모양이 (2,3)인 텐서이고, 컬럼을 따라서 합을 한다면, 결과는 모양이 (3,)인 벡터가 됩니다. 합 연산을 호출할 때, 합을 통해서 차원을 줄이는 것이 아니라, 원래 텐서의 축의 개수를 유지하도록 설정할 수도 있습니다. 이 경우에는 결과가 모양이 (1, 3)인 2-차원 텐서가 됩니다.

```{.python .input}
X = np.array([[1, 2, 3], [4, 5, 6]])
X.sum(axis=0, keepdims=True), '\n', X.sum(axis=1, keepdims=True)
```

```{.python .input}
#@tab pytorch
X = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
torch.sum(X, dim=0, keepdim=True), torch.sum(X, dim=1, keepdim=True)
```

```{.python .input}
#@tab tensorflow
X = tf.constant([[1., 2., 3.], [4., 5., 6.]])
[tf.reduce_sum(X, axis=i, keepdims=True) for i in range(0, 1)]
```

We are now ready to implement the softmax operation.
Recall that softmax consists of three steps:
i) we exponentiate each term (using `exp`);
ii) we sum over each row (we have one row per example in the batch)
to get the normalization constant for each example;
iii) we divide each row by its normalization constant,
ensuring that the result sums to 1.
Before looking at the code, let us recall
how this looks expressed as an equation:

$$
\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.
$$

이제 우리는 소프트맥스 연산을 구현할 준비가 완료됬습니다. 소프트맥스는 3단계로 이뤄져있다는 것을 떠올리세요: i) (`exp`를 사용해서) 모든 항에 지수를 취합니다; ii) (배치의 각 항은 하나의 예제입니다) 행을 따라서 더해서, 각 예제의 정규화 상수를 얻습니다; iii) 각 행을 그 행의 정규화 상수로 나눠서, 합이 1이 되도록 합니다. 코드를 살펴보기 전에, 이것이 수식으로 어떻게 표현되는지 다시 살펴보겠습니다.

$$
\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.
$$

The denominator, or normalization constant,
is also sometimes called the *partition function*
(and its logarithm is called the log-partition function).
The origins of that name are in [statistical physics](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))
where a related equation models the distribution
over an ensemble of particles.

분모, 즉 정규화 상수는 때로 *파티션 함수*라고 합니다 (그리고 그것의 로그값은 로그-파티션 함수라고 합니다). 이 이름의 기원은 관련된 방정식이 입자 앙상블에 대한 분포를 모델링하는 [통계 물리학](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))입니다.

```{.python .input}
def softmax(X):
    X_exp = np.exp(X)
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

```{.python .input}
#@tab pytorch
def softmax(X):
    X_exp = torch.exp(X)
    partition = torch.sum(X_exp, dim=1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

```{.python .input}
#@tab tensorflow
def softmax(X):
    X_exp = tf.exp(X)
    partition = tf.reduce_sum(X_exp, -1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

As you can see, for any random input,
we turn each element into a non-negative number.
Moreover, each row sums up to 1,
as is required for a probability.

보시다시피, 임의의 입력에 대해서 각 원소는 음수가 아닌 수로 바뀝니다. 더군다나, 확률에 필요한 것처럼 각 행의 합은 1입니다. 

```{.python .input}
X = np.random.normal(size=(2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(axis=1)
```

```{.python .input}
#@tab pytorch
X = torch.normal(0, 1, size=(2, 5))
X_prob = softmax(X)
X_prob, torch.sum(X_prob, dim=1)
```

```{.python .input}
#@tab tensorflow
X = tf.random.normal(shape=(2, 5))
X_prob = softmax(X)
X_prob, tf.reduce_sum(X_prob, axis=1)
```

Note that while this looks correct mathematically,
we were a bit sloppy in our implementation
because we failed to take precautions against numerical overflow or underflow
due to large or very small elements of the matrix.

이것이 수학적으로는 올바른 것처럼 보이지만, 행렬에서 아주 큰 수 또는 아주 작은 수로 인해서 발생할 수 있는 수치적 오퍼플로우 또는 언더플로우에 대한 주의를 기울이지 않았기 때문에 우리의 구현은 약간 조잡합니다.

## Defining the Model
## 모델 정의하기

Now that we have defined the softmax operation,
we can implement the softmax regression model.
The below code defines how the input is mapped to the output through the network.
Note that we flatten each original image in the batch
into a vector using the `reshape` function
before passing the data through our model.

소프트맥스 연살을 정의했으니 이제 소프트맥스 회귀 모델을 구현할 수 있습니다. 아래 코드는 입력이 어떻게 네트워크를 통해서 출력으로 매핑되는지를 정의합니다. 데이터를 모델에 전달하기 전에, 배치의 원본 이미지를 `reshape` 함수를 이용해서 평평하게 만들어 벡터로 만들었다는 것을 주의하세요.

```{.python .input}
def net(X):
    return softmax(np.dot(X.reshape(-1, W.shape[0]), W) + b)
```

```{.python .input}
#@tab pytorch
def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)
```

```{.python .input}
#@tab tensorflow
def net(X):
    return softmax(tf.matmul(tf.reshape(X, shape=(-1, W.shape[0])), W) + b)
```

## Defining the Loss Function
## 손실 함수 정의하기

Next, we need to implement the cross-entropy loss function,
as introduced in :numref:`sec_softmax`.
This may be the most common loss function
in all of deep learning because, at the moment,
classification problems far outnumber regression problems.

다음으로 :numref:`sec_softmax`에서 소개한 크로스-엔트로피 손실 함수를 구현해야 합니다. 지금까지는 회귀 문제보다 분류 문제의 수가 훨씬 더 많기 때문에, 이 함수는 딥러닝에서 가장 일반적인 손실 함수일 것입니다.

Recall that cross-entropy takes the negative log-likelihood
of the predicted probability assigned to the true label.
Rather than iterating over the predictions with a Python for-loop
(which tends to be inefficient),
we can pick all elements by a single operator.
Below, we create a toy data `y_hat`
with 2 examples of predicted probabilities over 3 classes.
Then we pick the probability of the first class in the first example
and the probability of the third class in the second example.

크로스-엔트로피는 실제 레이블에 대한 예측된 확률의 음의 로그-가능도(negative log-likelihood)라는 것을 상기하세요. Python의 for-룹으로 예측값을 하나씩 반복하는 것이 아니라 (이 방법은 비효율적입니다), 단일 연산을 사용해서 모든 원소를 선택할 수 있습니다. 아래에서 우리는 3개 클래스들에 대한 예측된 확률을 갖는 2개 예제로 구성된 장난감 데이터 `y_hat`을 만듭니다. 그리고 첫 번째 예제에서는 첫 번째 클래스의 확률값을 두 번째 예제에서는 세 번째 클래스의 확률값을 선택합니다.

```{.python .input}
y_hat = np.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = np.array([0, 2])
y_hat[[0, 1], y]
```

```{.python .input}
#@tab pytorch
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
#@tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

Now we can implement the cross-entropy loss function efficiently with just one line of code.

이제 우리는 단 한 줄의 코드로 크로스-엔트로피 손실 함수를 구현할 수 있습니다.

```{.python .input}
def cross_entropy(y_hat, y):
    return - np.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```

```{.python .input}
#@tab pytorch
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

cross_entropy(y_hat, y)
```

## Classification Accuracy
## 분류 정확도

Given the predicted probability distribution `y_hat`,
we typically choose the class with the highest predicted probability
whenever we must output a hard prediction.
Indeed, many applications require that we make a choice.
Gmail must categorize an email into "Primary", "Social", "Updates", or "Forums".
It might estimate probabilities internally,
but at the end of the day it has to choose one among the classes.

예측된 확률 분포 `y_hat`가 주어졌을 때, 하드 예측을 해야할 때마다 보통은 가장 높게 예측된 확률을 갖는 클래스를 선택합니다. 실제로 많은 어플리케이션들은 우리가 선택하기를 요구합니다. Gmail은 메일을 "Primary", "Social", "Updates" 또는 "Forums" 카테고리로 분류해야 합니다. 내부족으로는 확률들을 예측할 것이지만, 결국에는 클래들 중 하나를 선택해야하는 것입니다.

When predictions are consistent with the label class `y`, they are correct.
The classification accuracy is the fraction of all predictions that are correct.
Although it can be difficult to optimize accuracy directly (it is not differentiable),
it is often the performance measure that we care most about,
and we will nearly always report it when training classifiers.

예측이 레이블 클래스  `y`와 일치하면, 그 예측은 정확합니다. 분류 정확도는 정확한 예측의 비율입니다. 정확도(미분가능하지 않음)는 직접 최적화하기는 어렵지만, 종종 정확도가 우리가 대부분 관심을 갖는 성능 지표이며 분류기를 학습시킬 때 늘 이 지표를 리포트할 것입니다.

To compute accuracy we do the following.
First, if `y_hat` is a matrix,
we assume that the second dimension stores prediction scores for each class.
We use `argmax` to obtain the predicted class by the index for the largest entry in each row.
Then we compare the predicted class with the ground-truth `y` elementwise.
Since the equality operator `==` is sensitive to data types,
we convert `y_hat`'s data type to match that of `y`.
The result is a tensor containing entries of 0 (false) and 1 (true).
Taking the sum yields the number of correct predictions.

정확도 계산은 다음과 같이 합니다. 우선, `y_hat`가 행렬이라면, 행렬의 2번째 차원은 각 클래스에 대한 예측 점수를 가지고 있다고 가정합니다. 예측된 클래스는 `argmax`를 사용해서 각 행에서 가장 큰 값에 대한 인덱스를 얻습니다. 그 다음 예측된 클래스와 실제 클래스 `y` 를 원소별로 비교합니다. 평등 연산자 `==`는 데이터 타입에 민감하기 때문에,  `y_hat` 데이터 타입을 `y`의 데이터 타입과 같아지도록 바꿔줍니다. 비교 결과는 (예측이 틀린 경우) 0과 (예측이 맞은 경우) 1로 이뤄진 텐서가 됩니다. 이 텐서의 합은 정확한 예측의 개수입니다.

```{.python .input}
def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    return float((y_hat.astype(y.dtype) == y).sum())
```

```{.python .input}
#@tab pytorch
def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    return float((y_hat.type(y.dtype) == y).sum())
```

```{.python .input}
#@tab tensorflow
def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = tf.argmax(y_hat, axis=1)
    return float((tf.cast(y_hat, dtype=y.dtype) == y).numpy().sum())
```

We will continue to use the variables `y_hat` and `y`
defined before
as the predicted probability distributions and labels, respectively.
We can see that the first example's prediction class is 2
(the largest element of the row is 0.6 with the index 2),
which is inconsistent with the actual label, 0.
The second example's prediction class is 2
(the largest element of the row is 0.5 with the index of 2),
which is consistent with the actual label, 2.
Therefore, the classification accuracy rate for these two examples is 0.5.

이미 정의한 변수들 `y_hat`와 `y`를 예측 확률 분포와 레이블로 계속 사용합니다. 첫 번째 예제에 대한 예측 클래스는 2라는 것을 알 수 있는데 (1번째 행에서 가장 큰 원소는 0.6이고 그 원소의 인덱스는 2임), 이 값은 실제 레이블 0과 다릅니다. 두 번째 예제의 예측 클래스는 2 (2번째 행에서 가장 큰 원소는 0.5이고 그 원소의 인덱스는 2임)이고, 이는 실제 레이블과 일치합니다. 따라서, 이 두 예제에 대한 분류 정확도는 0.5가 됩니다.

```{.python .input}
#@tab all
accuracy(y_hat, y) / len(y)
```

Similarly, we can evaluate the accuracy for any model `net` on a dataset
that is accessed via the data iterator `data_iter`.

비슷하게, 우리는 데이터 반복자 `data_iter`를 통해서 얻을 수 있는 데이터셋에 대해서 임의의 모델 `net`의 정확도를 평가할 수 있습니다. 

```{.python .input}
#@tab all
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for _, (X, y) in enumerate(data_iter):
        metric.add(accuracy(net(X), y), sum(y.shape))
    return metric[0] / metric[1]
```

Here `Accumulator` is a utility class to accumulate sums over multiple variables.
In the above `evaluate_accuracy` function,
we create 2 variables in the `Accumulator` instance for storing both
the number of correct predictions and the number of predictions, respectively.
Both will be accumulated over time as we iterate over the dataset.

여기서 `Accumulator`는 유틸리티 클래스로 여러 변수에 걸친 합계를 누적합니다. 위 `evaluate_accuracy` 함수에서 정확한 예측 개수와 부정확한 예측 개수를 저장하기 위해서 `Accumulator` 인스턴스에 2개 변수를 생성합니다. 데이터셋을 반복하면서 이 두 값은 누적될 것입니다.

```{.python .input}
#@tab all
class Accumulator:  #@save
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

Because we initialized the `net` model with random weights,
the accuracy of this model should be close to random guessing,
i.e., 0.1 for 10 classes.

`net` 모델의 가중치를 임의의 수들로 초기화했기 때문에, 이 모델의 정확도는 랜덤 추측과 비슷합니다. 즉, 10개의 클래스에 대해서 각각 1.0.

```{.python .input}
#@tab all
evaluate_accuracy(net, test_iter)
```

## Training
## 학습하기

The training loop for softmax regression should look strikingly familiar
if you read through our implementation
of linear regression in :numref:`sec_linear_scratch`.
Here we refactor the implementation to make it reusable.
First, we define a function to train for one epoch.
Note that `updater` is a general function to update the model parameters,
which accepts the batch size as an argument.
It can be either a wrapper of the `d2l.sgd` function
or a framework's built-in optimization function.

:numref:`sec_linear_scratch`의 선형 회귀의 구현을 잘 읽었다면 소프트맥스 회귀의 학습 룹은 놀랍게도 친숙할 것입니다. 재사용성 고려해서 구현을 리팩터링합니다. 우선, 우리는 한 에폭을 학습하는 함수를 정의합니다. 배치 크기를 함수 인자로 받는 `updater`는 모델 파라미터를 업데이트하는 일반적인 함수임을 주목하세요. 이것은 `d2l.sgd` 함수 또는 프래임워크의 빌트인 최적화 함수의 래퍼입니다.

```{.python .input}
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Train a model within one epoch (defined in Chapter 3)."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab pytorch
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l)*len(y), accuracy(y_hat, y), y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.size().numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab tensorflow
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # tf.Keras' implementations for loss takes (labels, predictions)
            # instead of (predictions, labels) that users might implement
            # in this book, e.g. `cross_entropy()` that we implemented above
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Keras loss in default returns the average loss in a batch
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

Before showing the implementation of the training function,
we define a utility class that plot data in animation.
Again, it aims to simplify code in the rest of the book.

학습 함수의 구현을 보이기 앞서, 플롯 데이터를 애니매이션으로 보여주는 유틸리티 클래스를 정의합니다. 이는 이 책의 나머지 부분에 걸쳐서 코드를 단순화하기 위함입니다.

```{.python .input}
#@tab all
class Animator:  #@save
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

The following training function then 
trains a model `net` on a training dataset accessed via `train_iter`
for multiple epochs, which is specified by `num_epochs`.
At the end of each epoch,
the model is evaluated on a testing dataset accessed via `test_iter`.
We will leverage the `Animator` class to visualize
the training progress.

아래 학습 함수는 `num_epochs`로 정의된 에폭만큼 반복하면서 `train_iter`를 통해서 얻은 학습 데이터를 사용한 모델 `net`을 학습시킵니다. 각 에폭을 마치면 `test_iter`를 통해서 얻은 테스트 데이터셋을 사용해서 모델을 평가 합니다. 학습 진행 상황을 시각화하기 위해서 `Animator` 클래스를 사용할 것입니다.

```{.python .input}
#@tab all
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """Train a model (defined in Chapter 3)."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

As an implementation from scratch,
we use the minibatch stochastic gradient descent defined in :numref:`sec_linear_scratch`
to optimize the loss function of the model with a learning rate 0.1.

처음부터 구현은 모델의 손실 함수를 학습 속도 0.1로 최적화하기 위해서 :numref:`sec_linear_scratch`에서 정의한 미니배치 확률적 경사 하강법을 사용합니다.

```{.python .input}
#@tab mxnet, pytorch
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

```{.python .input}
#@tab tensorflow
class Updater():  #@save
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        d2l.sgd(self.params, grads, self.lr, batch_size)

updater = Updater([W, b], lr=0.1)
```

Now we train the model with 10 epochs.
Note that both the number of epochs (`num_epochs`),
and learning rate (`lr`) are adjustable hyperparameters.
By changing their values, we may be able
to increase the classification accuracy of the model.

10 에폭으로 모델을 학습합니다. 에폭수 (`num_epochs`)와 학습 속도(`lr`)는 조절이 가능한 하이퍼파라미터임을 주의하세요. 이들 값을 바꾸면 모델의 분류 정확도가 높아질 수도 있습니다.

```{.python .input}
#@tab all
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

## Prediction
## 예측

Now that training is complete,
our model is ready to classify some images.
Given a series of images,
we will compare their actual labels
(first line of text output)
and the predictions from the model
(second line of text output).

학습이 완료되면, 모델이 몇 개의 이미지를 분류할 준비가 되었습니다. 이미지들이 주어졌을 때, 실제 레이블(텍스트 출력의 첫 번째 줄)과 모델 예측 결과(텍스트 출력의 두 번째 줄)을 비교합니다.

```{.python .input}
#@tab mxnet, pytorch
def predict_ch3(net, test_iter, n=6):  #@save
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape(n, 28, 28), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

```{.python .input}
#@tab tensorflow
def predict_ch3(net, test_iter, n=6):  #@save
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(tf.argmax(net(X), axis=1))
    titles = [true+'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(tf.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

## Summary
## 요약

* With softmax regression, we can train models for multiclass classification.
* The training loop of softmax regression is very similar to that in linear regression: retrieve and read data, define models and loss functions, then train models using optimization algorithms. As you will soon find out, most common deep learning models have similar training procedures.

* 소프트맥스를 사용해서 우리는 멀리 클래스 분류 모델을 학습시킬 수 있습니다.
* 소프트맥스 회귀의 학습 룹은 선형 회귀와 비슷합니다: 데이터를 얻은 후, 읽고, 모델과 손실 학습 함수를 정의하고, 최적화 알고리즘을 이용해서 모델을 학습시킵니다. 앞으로 알게되겠지만, 대부분 일반적인 딥러닝 모델은 비슷한 학습 과정을 갖습니다.

## Exercises
## 연습문제

1. In this section, we directly implemented the softmax function based on the mathematical definition of the softmax operation. What problems might this cause? Hint: try to calculate the size of $\exp(50)$.
1. The function `cross_entropy` in this section was implemented according to the definition of the cross-entropy loss function.  What could be the problem with this implementation? Hint: consider the domain of the logarithm.
1. What solutions you can think of to fix the two problems above?
1. Is it always a good idea to return the most likely label? For example, would you do this for medical diagnosis?
1. Assume that we want to use softmax regression to predict the next word based on some features. What are some problems that might arise from a large vocabulary?

1. 이 절에서 소프트맥스 연산의 수학적인 정의에 기반해서 소프트맥스 함수를 구현했습니다. 이것이 어떤 문제를 야기할 수 있을까요? 힌트:  $\exp(50)$의 크기를 계산해보세요.
1. 이 절에서 `cross_entropy` 함수는 크로스-엔트로피 손실 함수의 정의에 따라서 구현되었습니다. 이 구현에 어떤 문제가 있을까요? 힌트: 로그의 정의역을 생각해보세요.
1. 이 두 문제를 해결하는 방법을 생각해 보세요.
1. 가장 가능성이 높은 레이블을 리턴하는 것이 항상 좋은 아이디어일까요? 예를 들어, 의학 분석의 경우도 이렇게 할 것입니까?
1. 어떤 피처들을 기반으로 다음 단어를 예측하기 위해서 소프트맥스 회귀를 사용하고 싶다고 가정합니다. 많은단어들이 있을 때 언떤 문제가 일어날 수 있을까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/225)
:end_tab:
