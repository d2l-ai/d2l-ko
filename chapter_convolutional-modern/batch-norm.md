# 배치 정규화
:label:`sec_batch_norm`

심층 신경망을 훈련시키는 것은 어렵습니다.그리고 합리적인 시간 내에 수렴하는 것은 까다로울 수 있습니다.이 섹션에서는 심층 네트워크 :cite:`Ioffe.Szegedy.2015`의 수렴을 지속적으로 가속화하는 인기 있고 효과적인 기술인*배치 정규화*에 대해 설명합니다.나중에 :numref:`sec_resnet`에서 다루는 잔차 블록과 함께 배치 정규화를 통해 실무자는 100개 이상의 계층으로 네트워크를 일상적으로 훈련시킬 수 있었습니다. 

## 심층 네트워크 훈련

배치 정규화에 동기를 부여하기 위해 특히 기계 학습 모델 및 신경망을 훈련시킬 때 발생하는 몇 가지 실질적인 문제를 검토해 보겠습니다. 

첫째, 데이터 전처리에 대한 선택은 종종 최종 결과에 큰 차이를 만듭니다.주택 가격 예측에 MLP를 적용한 것을 상기하십시오 (:numref:`sec_kaggle_house`).실제 데이터로 작업할 때 첫 번째 단계는 입력 피처를 표준화하여 각각의 평균이 0이고 분산이 1이 되도록 하는 것이었습니다.직관적으로 이 표준화는 파라미터*a priori*를 비슷한 규모로 배치하기 때문에 옵티마이저와 잘 어울립니다. 

둘째, 일반적인 MLP 또는 CNN의 경우 훈련 할 때 중간 계층의 변수 (예: MLP의 아핀 변환 출력) 는 입력에서 출력까지의 레이어를 따라, 동일한 계층의 단위에 걸쳐, 그리고 시간이 지남에 따라 매우 다양한 크기의 값을 취할 수 있습니다. 모델 업데이트로 인해매개 변수.배치 정규화의 발명자들은 그러한 변수의 분포에서 이러한 드리프트가 네트워크의 수렴을 방해 할 수 있다고 비공식적으로 가정했습니다.직관적으로 한 계층에 다른 계층의 100배인 변수 값이 있는 경우 학습률의 보상 조정이 필요할 수 있다고 추측할 수 있습니다. 

셋째, 더 깊은 네트워크는 복잡하고 쉽게 과적합할 수 있습니다.즉, 정규화가 더욱 중요해집니다. 

배치 정규화는 개별 계층 (선택적으로 모든 계층) 에 적용되며 다음과 같이 작동합니다. 각 훈련 반복에서 먼저 평균을 빼고 표준 편차로 나누어 (배치 정규화의) 입력을 정규화합니다.현재 미니배치.다음으로 스케일 계수와 스케일 오프셋을 적용합니다.*배치 정규화*가 그 이름을 파생시키는 것은*배치* 통계를 기반으로하는*정규화* 때문입니다. 

크기가 1인 미니 배치로 배치 정규화를 적용하려고 하면 아무것도 배울 수 없습니다.그 이유는 평균을 빼면 숨겨진 각 단위의 값이 0이 되기 때문입니다!짐작할 수 있듯이, 미니 배치가 충분히 큰 배치 정규화에 전체 섹션을 할애하고 있기 때문에 접근 방식이 효과적이고 안정적입니다.여기서 한 가지 요점은 배치 정규화를 적용할 때 배치 정규화를 사용하지 않는 것보다 배치 크기를 선택하는 것이 훨씬 더 중요할 수 있다는 것입니다. 

공식적으로, 미니배치 $\mathcal{B}$의 배치 정규화 ($\mathrm{BN}$) 에 대한 입력을 $\mathbf{x} \in \mathcal{B}$으로 나타내면 배치 정규화는 다음 표현식에 따라 $\mathbf{x}$를 변환합니다. 

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

:eqref:`eq_batchnorm`에서 $\hat{\boldsymbol{\mu}}_\mathcal{B}$는 표본 평균이고 $\hat{\boldsymbol{\sigma}}_\mathcal{B}$은 미니배치 $\mathcal{B}$의 표본 표준 편차입니다.표준화를 적용한 후 결과로 생성되는 미니배치는 평균과 단위 분산이 0입니다.단위 분산 (다른 매직 넘버와 비교) 의 선택은 임의의 선택이기 때문에 일반적으로 요소별로
*스케일 파라미터* $\boldsymbol{\gamma}$ 및*시프트 파라미터* $\boldsymbol{\beta}$
$\mathbf{x}$와 같은 모양을 가지고 있습니다.$\boldsymbol{\gamma}$ 및 $\boldsymbol{\beta}$은 다른 모델 매개 변수와 공동으로 학습해야 하는 매개 변수입니다. 

결과적으로 배치 정규화가 능동적으로 중심을 맞추고 주어진 평균과 크기로 다시 스케일링하기 때문에 중간 계층의 가변 크기는 훈련 중에 발산할 수 없습니다 ($\hat{\boldsymbol{\mu}}_\mathcal{B}$ 및 ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$를 통해).실무자의 직감이나 지혜의 한 부분은 배치 정규화가 더 공격적인 학습 속도를 허용하는 것처럼 보인다는 것입니다. 

공식적으로 다음과 같이 :eqref:`eq_batchnorm`에서 $\hat{\boldsymbol{\mu}}_\mathcal{B}$와 ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$을 계산합니다. 

$$\begin{aligned} \hat{\boldsymbol{\mu}}_\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x},\\
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.\end{aligned}$$

경험적 분산 추정치가 사라질 수 있는 경우에도 0으로 나누기를 시도하지 않도록 분산 추정치에 작은 상수 $\epsilon > 0$를 추가합니다.추정치 $\hat{\boldsymbol{\mu}}_\mathcal{B}$ 및 ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$은 평균과 분산에 대한 잡음이 있는 추정치를 사용하여 스케일링 문제에 대응합니다.이 잡음이 문제가 될 것이라고 생각할 수도 있습니다.결과적으로 이것은 실제로 유익합니다. 

이는 딥 러닝에서 반복되는 주제로 판명되었습니다.이론적으로 아직 잘 특성화되지 않은 이유로 최적화의 다양한 노이즈 원인은 종종 훈련 속도가 빨라지고 과적합이 줄어 듭니다. 이러한 변형은 정규화의 한 형태로 작용하는 것으로 보입니다.일부 예비 연구에서 :cite:`Teye.Azizpour.Smith.2018` 및 :cite:`Luo.Wang.Shao.ea.2018`는 배치 정규화의 특성을 각각 베이지안 사전 및 벌칙과 관련시킵니다.특히 이것은 배치 정규화가 $50 \sim 100$ 범위의 중간 미니 배치 크기에 가장 적합한 이유에 대한 퍼즐을 밝힙니다. 

훈련된 모델을 수정하면 평균과 분산을 추정하기 위해 전체 데이터셋을 사용하는 것이 더 좋다고 생각할 수 있습니다.훈련이 완료되면 동일한 이미지가 상주하는 배치에 따라 다르게 분류되기를 원하는 이유는 무엇입니까?훈련 중에는 모델을 업데이트할 때마다 모든 데이터 예제의 중간 변수가 변경되므로 이러한 정확한 계산은 불가능합니다.그러나 모델이 훈련되면 전체 데이터셋을 기반으로 각 계층 변수의 평균과 분산을 계산할 수 있습니다.실제로 이것은 배치 정규화를 사용하는 모델에 대한 표준 관행이므로 배치 정규화 계층은*훈련 모드* (미니배치 통계량으로 정규화) 와*예측 모드* (데이터셋 통계량으로 정규화) 에서 다르게 작동합니다. 

이제 배치 정규화가 실제로 어떻게 작동하는지 살펴볼 준비가 되었습니다. 

## 배치 정규화 계층

완전 연결 계층과 컨벌루션 계층에 대한 배치 정규화 구현은 약간 다릅니다.아래에서 두 가지 사례에 대해 논의합니다.배치 정규화와 다른 계층 간의 주요 차이점 중 하나는 배치 정규화가 한 번에 전체 미니 배치에서 작동하기 때문에 다른 계층을 도입할 때 이전처럼 배치 차원을 무시할 수 없다는 것입니다. 

### 완전 연결 레이어

완전 연결 계층에 배치 정규화를 적용할 때 원본 논문에서는 아핀 변환 후와 비선형 활성화 함수 앞에 배치 정규화를 삽입합니다 (이후 응용 프로그램에서는 활성화 함수 바로 다음에 배치 정규화를 삽입할 수 있음) :cite:`Ioffe.Szegedy.2015`.완전히 연결된 계층에 대한 입력을 $\mathbf{x}$으로 표시하고, 아핀 변환을 $\mathbf{W}\mathbf{x} + \mathbf{b}$ (가중치 매개 변수 $\mathbf{W}$ 및 바이어스 매개 변수 $\mathbf{b}$) 으로 표시하고 활성화 함수를 $\phi$로 표시하면 배치 정규화가 가능하고 완전히 연결된 계층 출력의 계산을 표현할 수 있습니다.$\mathbf{h}$는 다음과 같습니다. 

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

평균과 분산은 변환이 적용되는*동일한* 미니배치에서 계산됩니다. 

### 컨벌루션 계층

마찬가지로 컨벌루션 계층을 사용하면 컨벌루션 뒤와 비선형 활성화 함수 앞에 배치 정규화를 적용 할 수 있습니다.컨볼 루션에 여러 개의 출력 채널이있는 경우 이러한 채널의 출력의*각*에 대해 배치 정규화를 수행해야하며 각 채널에는 자체 스케일 및 시프트 매개 변수가 있으며 둘 다 스칼라입니다.미니 배치에 $m$의 예가 포함되어 있고 각 채널에 대해 컨벌루션의 출력의 높이가 $p$이고 너비가 $q$라고 가정합니다.컨벌루션 계층의 경우 출력 채널당 $m \cdot p \cdot q$ 요소에 대해 각 배치 정규화를 동시에 수행합니다.따라서 평균과 분산을 계산할 때 모든 공간 위치에 대한 값을 수집하고 결과적으로 주어진 채널 내에서 동일한 평균과 분산을 적용하여 각 공간 위치에서 값을 정규화합니다. 

### 예측 중 배치 정규화

앞서 언급했듯이 배치 정규화는 일반적으로 훈련 모드와 예측 모드에서 다르게 동작합니다.첫째, 모델을 훈련시킨 후에는 표본 평균의 잡음과 미니 배치에서 각각을 추정하여 발생하는 표본 분산이 더 이상 바람직하지 않습니다.둘째, 배치당 정규화 통계를 계산하는 사치가 없을 수도 있습니다.예를 들어 한 번에 하나씩 예측하기 위해 모델을 적용해야 할 수 있습니다. 

일반적으로 훈련 후에는 전체 데이터셋을 사용하여 변수 통계량의 안정적인 추정치를 계산한 다음 예측 시점에 수정합니다.따라서 배치 정규화는 훈련 중과 테스트 시 다르게 동작합니다.드롭 아웃도 이러한 특성을 나타냅니다. 

## (**처음부터 구현**)

아래에서는 처음부터 텐서를 사용하여 배치 정규화 계층을 구현합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `autograd` to determine whether the current mode is training mode or
    # prediction mode
    if not autograd.is_training():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `is_grad_enabled` to determine whether the current mode is training
    # mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

이제 [**적절한 `BatchNorm` 레이어를 생성할 수 있습니다.**] 레이어가 스케일 `gamma` 및 시프트 `beta`에 대한 적절한 파라미터를 유지하며, 이 두 가지 모두 훈련 과정에서 업데이트됩니다.또한 계층은 모델 예측 중에 후속 사용을 위해 평균 및 분산의 이동 평균을 유지합니다. 

알고리즘의 세부 사항을 제외하고 레이어 구현의 기초가 되는 디자인 패턴에 주목하십시오.일반적으로 수학은 `batch_norm`와 같은 별도의 함수로 정의합니다.그런 다음 이 기능을 사용자 지정 계층에 통합합니다. 이 계층은 데이터를 올바른 장치 컨텍스트로 이동하고, 필요한 변수를 할당 및 초기화하고, 이동 평균 (여기서는 평균 및 분산) 을 추적하는 등의 부기 문제를 주로 다룹니다.이 패턴을 사용하면 상용구 코드에서 수학을 깔끔하게 분리할 수 있습니다.또한 편의를 위해 입력 셰이프를 자동으로 추론하는 것에 대해 걱정하지 않았으므로 전체 피처 수를 지정해야 합니다.걱정하지 마세요. 딥러닝 프레임워크의 상위 수준 배치 정규화 API가 이 문제를 해결해 줄 것이며 나중에 설명하겠습니다.

```{.python .input}
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

```{.python .input}
#@tab pytorch
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

```{.python .input}
#@tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

## [**LeNet에서 배치 정규화 적용**]

컨텍스트에서 `BatchNorm`를 적용하는 방법을 보려면 아래에서 기존 LeNet 모델 (:numref:`sec_lenet`) 에 적용합니다.배치 정규화는 컨벌루션 계층 또는 완전 연결 계층 다음에 적용되지만 해당 활성화 함수 앞에 적용됩니다.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
# Recall that this has to be a function that will be passed to `d2l.train_ch6`
# so that model building or compiling need to be within `strategy.scope()` in
# order to utilize the CPU/GPU devices that we have
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]
    )
```

이전과 마찬가지로 [**패션-MNIST 데이터세트에 대한 네트워크 교육**] 을 하겠습니다.이 코드는 LeNet (:numref:`sec_lenet`) 을 처음 학습했을 때의 코드와 거의 동일합니다.가장 큰 차이점은 학습률이 높다는 것입니다.

```{.python .input}
#@tab mxnet, pytorch
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

첫 번째 배치 정규화 계층에서 학습한 스케일 파라미터 `gamma`와 시프트 파라미터 `beta`**] 를 살펴보겠습니다.

```{.python .input}
net[1].gamma.data().reshape(-1,), net[1].beta.data().reshape(-1,)
```

```{.python .input}
#@tab pytorch
net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))
```

```{.python .input}
#@tab tensorflow
tf.reshape(net.layers[1].gamma, (-1,)), tf.reshape(net.layers[1].beta, (-1,))
```

## [**간결한 구현**]

방금 정의한 `BatchNorm` 클래스와 비교할 때 딥 러닝 프레임워크에서 상위 수준 API에 정의된 `BatchNorm` 클래스를 직접 사용할 수 있습니다.코드는 위의 구현과 거의 동일하게 보입니다.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])
```

아래에서는 [**동일한 하이퍼파라미터를 사용하여 모델을 학습시키십시오.**] 평소와 같이 고수준 API 변형은 코드가 C++ 또는 CUDA로 컴파일되었지만 사용자 지정 구현은 Python에 의해 해석되어야 하기 때문에 훨씬 빠르게 실행됩니다.

```{.python .input}
#@tab all
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 논란

직관적으로 배치 정규화는 최적화 환경을 더 매끄럽게 만드는 것으로 생각됩니다.그러나 심층 모델을 훈련 할 때 관찰하는 현상에 대한 투기 적 직관과 진정한 설명을 구분해야합니다.더 단순한 심층 신경망 (MLP 및 기존 CNN) 이 처음부터 잘 일반화되는 이유조차 알지 못합니다.드롭 아웃 및 체중 감소에도 불구하고 기존의 학습 이론적 일반화 보증을 통해 보이지 않는 데이터로 일반화하는 능력을 설명 할 수 없을 정도로 유연합니다. 

배치 정규화를 제안하는 원본 논문에서 저자는 강력하고 유용한 도구를 소개하는 것 외에도 내부 공변량 이동*을 줄임으로써 작동 이유에 대한 설명을 제공했습니다.아마도*내부 공변량 이동*에 의해 저자는 위에 표현된 직관, 즉 훈련 과정에서 변수 값의 분포가 변한다는 개념과 같은 것을 의미했을 것입니다.그러나이 설명에는 두 가지 문제가있었습니다. i) 이 드리프트는*공변량 이동*과 매우 다르므로 이름을 잘못된 명칭으로 만듭니다. ii) 설명은 지정되지 않은 직관을 제공하지만*이 기술이 정확히 작동하는 이유*에 대한 질문을 엄격한 설명을 원하는 열린 질문입니다..이 책 전체에서 우리는 실무자가 심층 신경망 개발을 안내하는 데 사용하는 직관을 전달하는 것을 목표로합니다.그러나 우리는 이러한 지침 직관을 확립 된 과학적 사실과 분리하는 것이 중요하다고 생각합니다.결국, 이 자료를 숙달하고 자신의 연구 논문을 작성하기 시작할 때 기술적 주장과 직감을 명확하게 묘사하고 싶을 것입니다. 

배치 정규화의 성공에 이어, *내부 공변량 이동*에 대한 설명은 기술 문헌의 논쟁과 기계 학습 연구를 제시하는 방법에 대한 광범위한 담론에서 반복적으로 드러났습니다.2017 NeuRips 컨퍼런스에서 Test of Time Award를 수락하면서 기억에 남는 연설에서 Ali Rahimi는 딥 러닝의 현대 관행을 연금술에 비유하는 논쟁에서*내부 공변량 이동*을 초점으로 사용했습니다.그 후 기계 학습 :cite:`Lipton.Steinhardt.2018`의 문제가되는 추세를 요약 한 직책 논문에서 예제를 자세히 다시 검토했습니다.다른 저자들은 배치 정규화의 성공에 대한 대체 설명을 제안했으며, 일부는 원본 논문 :cite:`Santurkar.Tsipras.Ilyas.ea.2018`에서 주장한 것과 반대되는 동작을 보였음에도 불구하고 배치 정규화의 성공이 발생한다고 주장합니다. 

*내부 공변량 이동*은 기술 기계 학습 문헌에서 매년 수천 건의 유사한 모호한 주장보다 더 이상 비판의 가치가 없다는 점에 주목합니다.아마도 이러한 논쟁의 초점으로서의 공명은 대상 청중에 대한 광범위한 인식 가능성 때문일 것입니다.배치 정규화는 거의 모든 배포된 이미지 분류기에 적용되는 필수 방법임이 입증되어 수만 건의 인용 기법을 도입한 논문을 얻었습니다. 

## 요약

* 모델 훈련 중에 배치 정규화는 미니 배치의 평균과 표준편차를 활용하여 신경망의 중간 출력을 지속적으로 조정하므로 신경망 전체의 각 계층에 있는 중간 출력값의 값이 더 안정적입니다.
* 완전 연결 계층과 컨벌루션 계층의 배치 정규화 방법은 약간 다릅니다.
* 드롭아웃 계층과 마찬가지로 배치 정규화 계층은 훈련 모드와 예측 모드에서 서로 다른 계산 결과를 갖습니다.
* 배치 정규화에는 주로 정규화와 같은 여러 가지 유익한 부작용이 있습니다.반면에 내부 공변량 이동을 줄이려는 원래 동기는 유효한 설명이 아닌 것 같습니다.

## 연습문제

1. 배치 정규화 전에 완전 연결 계층 또는 컨벌루션 계층에서 편향 파라미터를 제거할 수 있습니까?왜요?
1. 배치 정규화가 있는 경우와 없는 LeNet의 학습률을 비교합니다.
    1. 훈련 및 테스트 정확도의 증가를 플로팅합니다.
    1. 학습률을 얼마나 높일 수 있습니까?
1. 모든 계층에서 배치 정규화가 필요한가요?실험 해 보시겠습니까?
1. 드롭아웃을 배치 정규화로 대체할 수 있습니까?행동은 어떻게 변하는가?
1. 매개 변수 `beta` 및 `gamma`를 수정하고 결과를 관찰하고 분석합니다.
1. 상위 수준 API의 `BatchNorm`에 대한 온라인 설명서를 검토하여 배치 정규화를 위한 다른 응용 프로그램을 확인하십시오.
1. 연구 아이디어: 적용할 수 있는 다른 정규화 변환을 생각하시나요?확률 적분 변환을 적용할 수 있습니까?완전 순위 공분산 추정치는 어떻습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/330)
:end_tab:
