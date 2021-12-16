# 소프트맥스 리그레션 처음부터 구현
:label:`sec_softmax_scratch`

(**우리가 처음부터 선형 회귀를 구현했듯이**) softmax 회귀도 마찬가지로 근본적이며 (**당신은**) (~~softmax 회귀 ~~) 의 피투성이의 세부 사항과 직접 구현하는 방법을 알아야합니다.우리는 :numref:`sec_fashion_mnist`에서 방금 소개된 패션-MNIST 데이터세트로 작업하여 배치 크기가 256인 데이터 반복기를 설정할 것입니다.

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

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 모델 매개변수 초기화

선형 회귀 예제에서와 같이 각 예제는 고정 길이 벡터로 표현됩니다.원시 데이터셋의 각 예는 $28 \times 28$ 이미지입니다.이 섹션에서는 [**각 이미지를 평면화하여 길이 784의 벡터로 처리하겠습니다**] 앞으로는 이미지의 공간 구조를 활용하기 위한 보다 정교한 전략에 대해 이야기하겠습니다. 하지만 지금은 각 픽셀 위치를 또 다른 기능으로 취급합니다. 

소프트맥스 회귀에서는 클래스 수만큼의 출력값을 가집니다.(**데이터셋에 10개의 클래스가 있으므로 네트워크의 출력 차원은 10.**) 결과적으로 가중치는 $784 \times 10$ 행렬을 구성하고 편향은 $1 \times 10$ 행 벡터를 구성합니다.선형 회귀와 마찬가지로 가우스 잡음과 편향을 사용하여 가중치 `W`를 초기화하여 초기 값 0을 취합니다.

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

## 소프트맥스 연산 정의

소프트맥스 회귀 모델을 구현하기 전에 :numref:`subseq_lin-alg-reduction` 및 :numref:`subseq_lin-alg-non-reduction`에서 설명한 것처럼 합계 연산자가 텐서의 특정 차원을 따라 어떻게 작동하는지 간단히 검토해 보겠습니다.[**행렬 `X`이 주어지면 모든 요소 (기본적으로) 또는 동일한 축의 요소 (**]), 즉 동일한 열 (축 0) 또는 동일한 행 (축 1) 에 대해서만 합을 구할 수 있습니다.`X`이 모양 (2, 3) 을 가진 텐서이고 열에 대해 합산하면 결과는 모양 (3,) 을 가진 벡터가됩니다.sum 연산자를 호출 할 때 합산 한 차원을 축소하는 대신 원래 텐서의 축 수를 유지하도록 지정할 수 있습니다.그러면 모양 (1, 3) 의 2차원 텐서가 생성됩니다.

```{.python .input}
#@tab pytorch
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdim=True), d2l.reduce_sum(X, 1, keepdim=True)
```

```{.python .input}
#@tab mxnet, tensorflow
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

이제 (**소프트맥스 작업 구현**) 할 준비가 되었습니다.softmax는 세 단계로 구성됩니다. (i) 각 항을 지수 화합니다 (`exp` 사용). (ii) 각 예에 대한 정규화 상수를 얻기 위해 각 행에 대해 합산 (배치에 예제 당 하나의 행이 있음); (iii) 각 행을 정규화 상수로 나누어 결과의 합계가 1이되도록합니다.코드를 살펴보기 전에 이것이 방정식으로 어떻게 표현되는지 생각해 봅시다. 

(** $\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$
**)

분모 또는 정규화 상수는*파티션 함수*라고도 합니다 (그리고 그 로그를 로그 분할 함수라고 함).이 이름의 기원은 관련 방정식이 입자 앙상블에 대한 분포를 모델링하는 [통계 물리학](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)) 에 있습니다.

```{.python .input}
#@tab mxnet, tensorflow
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

```{.python .input}
#@tab pytorch
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

보시다시피 임의의 입력에 대해 [**각 요소를 음수가 아닌 숫자로 바꿉니다.또한 각 행의 합은 확률에 필요한 대로 최대 1, **] 입니다.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
#@tab tensorflow
X = tf.random.normal((2, 5), 0, 1)
X_prob = softmax(X)
X_prob, tf.reduce_sum(X_prob, 1)
```

이것은 수학적으로 정확해 보이지만 행렬의 크거나 매우 작은 요소로 인해 수치 오버 플로우 또는 언더 플로우에 대한 예방 조치를 취하지 못했기 때문에 구현에서 약간 엉성했습니다. 

## 모델 정의

이제 softmax 연산을 정의했으므로 [**softmax 회귀 모델을 구현할 수 있습니다.**] 아래 코드는 입력이 네트워크를 통해 출력에 매핑되는 방식을 정의합니다.모델을 통해 데이터를 전달하기 전에 `reshape` 함수를 사용하여 배치의 각 원본 이미지를 벡터로 평탄화합니다.

```{.python .input}
#@tab all
def net(X):
    return softmax(d2l.matmul(d2l.reshape(X, (-1, W.shape[0])), W) + b)
```

## 손실 함수 정의하기

다음으로 :numref:`sec_softmax`에 소개된 교차 엔트로피 손실 함수를 구현해야 합니다.이는 현재 분류 문제가 회귀 문제보다 훨씬 많기 때문에 모든 딥러닝에서 가장 일반적인 손실 함수일 수 있습니다. 

교차 엔트로피는 실제 레이블에 할당된 예측 확률의 음의 로그 우도를 취한다는 것을 상기하십시오.Python for 루프 (비효율적 인 경향이 있음) 로 예측을 반복하는 대신 단일 연산자로 모든 요소를 선택할 수 있습니다.아래에서는 [**3개의 클래스에 대한 예측 확률의 2가지 예와 해당 레이블 `y`.**를 사용하여 샘플 데이터 `y_hat`를 만듭니다.] `y`를 사용하면 첫 번째 예에서는 첫 번째 클래스가 올바른 예측이고 두 번째 예에서는 세 번째 클래스가 실측이라는 것을 알 수 있습니다.[**`y`를 `y_hat`의 확률 인덱스로 사용**] 첫 번째 예에서는 첫 번째 클래스의 확률을 선택하고 두 번째 예에서는 세 번째 클래스의 확률을 선택합니다.

```{.python .input}
#@tab mxnet, pytorch
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
#@tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

이제 단 한 줄의 코드로 효율적으로 (**교차 엔트로피 손실 함수 구현**) 할 수 있습니다.

```{.python .input}
#@tab mxnet, pytorch
def cross_entropy(y_hat, y):
    return - d2l.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

cross_entropy(y_hat, y)
```

## 분류 정확도

예측 확률 분포 `y_hat`가 주어지면 일반적으로 어려운 예측을 출력해야 할 때마다 예측 확률이 가장 높은 클래스를 선택합니다.실제로 많은 응용 프로그램에서 선택을 요구합니다.Gmail은 이메일을 '기본', '소셜', '업데이트' 또는 '포럼'으로 분류해야 합니다.내부적으로 확률을 추정할 수도 있지만 하루가 끝나면 클래스 중 하나를 선택해야 합니다. 

예측이 레이블 클래스 `y`와 일치하면 정확합니다.분류 정확도는 모든 예측에서 정확한 비율입니다.정확도를 직접 최적화하는 것은 어려울 수 있지만 (차별화할 수는 없음), 종종 우리가 가장 중요하게 생각하는 성능 척도이며 분류기를 훈련시킬 때 거의 항상 보고할 것입니다. 

정확도를 계산하기 위해 다음을 수행합니다.첫째, `y_hat`이 행렬인 경우 두 번째 차원이 각 클래스에 대한 예측 점수를 저장한다고 가정합니다.`argmax`을 사용하여 각 행에서 가장 큰 항목에 대한 인덱스로 예측된 클래스를 얻습니다.그런 다음 [**예측된 클래스를 실측 `y`와 요소별로 비교합니다.**] 등식 연산자 `==`는 데이터 유형에 민감하므로 `y_hat`의 데이터 유형을 `y`의 데이터 유형과 일치하도록 변환합니다.결과는 0 (false) 과 1 (true) 의 항목을 포함하는 텐서입니다.합계를 취하면 정확한 예측의 수가 산출됩니다.

```{.python .input}
#@tab all
def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))
```

앞서 정의한 변수 `y_hat` 및 `y`를 각각 예측 확률 분포 및 레이블로 계속 사용할 것입니다.첫 번째 예제의 예측 클래스가 2 (행의 가장 큰 요소는 인덱스 2의 0.6) 이며 실제 레이블 0과 일치하지 않음을 알 수 있습니다.두 번째 예제의 예측 클래스는 2 (행의 가장 큰 요소는 0.5이고 인덱스는 2) 이며, 이는 실제 레이블 2와 일치합니다.따라서 이 두 예에 대한 분류 정확도는 0.5입니다.

```{.python .input}
#@tab all
accuracy(y_hat, y) / len(y)
```

[**마찬가지로 데이터 반복기 `data_iter`를 통해 액세스되는 데이터 세트의 모든 모델 `net`**] 의 정확도를 평가할 수 있습니다.

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

여기서 `Accumulator`는 여러 변수에 대해 합계를 누적하는 유틸리티 클래스입니다.위의 `evaluate_accuracy` 함수에서 올바른 예측 수와 예측 수를 각각 저장하기 위해 `Accumulator` 인스턴스에 두 개의 변수를 만듭니다.데이터 세트를 반복하면서 시간이 지남에 따라 둘 다 누적됩니다.

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

[**랜덤 가중치를 사용하여 `net` 모델을 초기화했기 때문에 이 모델의 정확도는 무작위 추측에 가까워야 합니다. 즉, 10개의 클래스에 대해 0.1이어야 합니다.

```{.python .input}
#@tab all
evaluate_accuracy(net, test_iter)
```

## 트레이닝

:numref:`sec_linear_scratch`의 선형 회귀 구현을 살펴보면 소프트맥스 회귀에 대한 [**훈련 루프**] 가 매우 친숙해 보일 것입니다.여기서는 다시 사용할 수 있도록 구현을 리팩토링합니다.먼저, 한 epoch 동안 훈련할 함수를 정의합니다.`updater`는 배치 크기를 인수로 받아들이는 모델 매개 변수를 업데이트하는 일반적인 함수입니다.`d2l.sgd` 함수의 래퍼이거나 프레임워크의 내장 최적화 함수일 수 있습니다.

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
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            l.sum().backward()
            updater.step()
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
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
            # Keras implementations for loss takes (labels, predictions)
            # instead of (predictions, labels) that users might implement
            # in this book, e.g. `cross_entropy` that we implemented above
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
        # Keras loss by default returns the average loss in a batch
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

훈련 함수의 구현을 보여주기 전에 [**애니메이션에서 데이터를 플로팅하는 유틸리티 클래스**] 를 정의합니다. 다시 말하지만, 책의 나머지 부분에서 코드를 단순화하는 것을 목표로 합니다.

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

[~~훈련 함수~~] 다음 훈련 함수는 `num_epochs`에 의해 지정된 다중 에포크에 대해 `train_iter`를 통해 액세스한 훈련 데이터세트에 대해 모델 `net`을 훈련시킵니다.각 Epoch가 끝날 때 모델은 `test_iter`을 통해 액세스한 테스트 데이터 세트에서 평가됩니다.`Animator` 클래스를 활용하여 교육 진행 상황을 시각화합니다.

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

처음부터 구현하기 위해 :numref:`sec_linear_scratch`에 정의된 [**미니배치 확률적 기울기 하강**] 을 사용하여 학습률 0.1로 모델의 손실 함수를 최적화합니다.

```{.python .input}
#@tab mxnet, pytorch
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

```{.python .input}
#@tab tensorflow
class Updater():  #@save
    """For updating parameters using minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        d2l.sgd(self.params, grads, self.lr, batch_size)

updater = Updater([W, b], lr=0.1)
```

이제 우리는 [**10개의 에포크로 모델을 훈련시킵니다.**] 에포크 수 (`num_epochs`) 와 학습률 (`lr`) 은 모두 조정 가능한 하이퍼파라미터입니다.해당 값을 변경하면 모델의 분류 정확도를 높일 수 있습니다.

```{.python .input}
#@tab all
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

## 예측

이제 학습이 완료되었으므로 모델은 [**일부 이미지를 분류할 준비가 되었습니다.**] 일련의 이미지가 주어지면 실제 레이블 (텍스트 출력의 첫 번째 줄) 과 모델의 예측 (텍스트 출력의 두 번째 줄) 을 비교합니다.

```{.python .input}
#@tab all
def predict_ch3(net, test_iter, n=6):  #@save
    """Predict labels (defined in Chapter 3)."""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

## 요약

* 소프트맥스 회귀를 사용하면 다중 클래스 분류를 위한 모델을 훈련시킬 수 있습니다.
* 소프트맥스 회귀의 훈련 루프는 선형 회귀의 훈련 루프와 매우 유사합니다. 즉, 데이터 검색 및 읽기, 모델 및 손실 함수 정의, 최적화 알고리즘을 사용하여 모델 훈련입니다.곧 알게 되겠지만, 대부분의 일반적인 딥러닝 모델은 비슷한 훈련 절차를 가지고 있습니다.

## 연습문제

1. 이 섹션에서는 softmax 연산의 수학적 정의를 기반으로 softmax 함수를 직접 구현했습니다.이로 인해 어떤 문제가 발생할 수 있습니까?힌트: $\exp(50)$의 크기를 계산해 보십시오.
1. 이 섹션의 함수 `cross_entropy`는 교차 엔트로피 손실 함수의 정의에 따라 구현되었습니다.이 구현의 문제점은 무엇일까요?힌트: 로그의 영역을 고려합니다.
1. 위의 두 가지 문제를 해결하기 위해 어떤 해결책을 생각해 볼 수 있습니까?
1. 가장 가능성이 높은 라벨을 반품하는 것이 항상 좋은 생각입니까?예를 들어, 의료 진단을 위해 이렇게 하시겠습니까?
1. 일부 기능을 기반으로 다음 단어를 예측하기 위해 softmax 회귀를 사용한다고 가정합니다.큰 어휘에서 발생할 수 있는 몇 가지 문제는 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/225)
:end_tab:
