# 아담
:label:`sec_adam`

이 섹션을 앞둔 논의에서 효율적인 최적화를 위한 여러 가지 기법을 접했습니다.여기에서 자세히 정리해 보겠습니다. 

* :numref:`sec_sgd`는 중복 데이터에 대한 고유의 복원력 때문에 최적화 문제를 해결할 때 경사 하강법보다 더 효과적이라는 것을 확인했습니다. 
* :numref:`sec_minibatch_sgd`가 하나의 미니배치에서 더 큰 관측치 세트를 사용하여 벡터화로 인해 발생하는 상당한 추가 효율성을 제공한다는 것을 확인했습니다.이것이 효율적인 다중 머신, 다중 GPU 및 전체 병렬 처리의 핵심입니다. 
* :numref:`sec_momentum`는 수렴을 가속화하기 위해 과거 그래디언트의 기록을 집계하는 메커니즘을 추가했습니다.
* :numref:`sec_adagrad`는 계산적으로 효율적인 선조건자를 허용하기 위해 좌표당 스케일링을 사용했습니다. 
* :numref:`sec_rmsprop` 학습 속도 조정에서 좌표별 스케일링을 분리했습니다. 

Adam :cite:`Kingma.Ba.2014`는 이러한 모든 기술을 하나의 효율적인 학습 알고리즘으로 결합합니다.예상대로 이 알고리즘은 딥 러닝에 사용하기에 더 강력하고 효과적인 최적화 알고리즘 중 하나로 널리 사용되고 있습니다.하지만 문제가 없는 것은 아닙니다.특히 :cite:`Reddi.Kale.Kumar.2019`는 분산 제어가 좋지 않아 아담이 갈라질 수 있는 상황이 있음을 보여줍니다.후속 작업에서 :cite:`Zaheer.Reddi.Sachan.ea.2018`은 이러한 문제를 해결하는 Yogi라는 Adam에게 핫픽스를 제안했습니다.이에 대해서는 나중에 자세히 설명합니다.이제 Adam 알고리즘을 검토해 보겠습니다.  

## 알고리즘

Adam의 주요 구성 요소 중 하나는 지수 가중 이동 평균 (누설 평균화라고도 함) 을 사용하여 모멘텀과 기울기의 두 번째 모멘트를 모두 추정한다는 것입니다.즉, 상태 변수를 사용합니다. 

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

여기서 $\beta_1$ 및 $\beta_2$는 음이 아닌 가중치 매개 변수입니다.이들을 위한 일반적인 선택은 $\beta_1 = 0.9$과 $\beta_2 = 0.999$입니다.즉, 분산 추정치가 모멘텀 항보다 훨씬 느리게* 움직입니다.$\mathbf{v}_0 = \mathbf{s}_0 = 0$를 초기화하면 처음에는 더 작은 값에 상당한 양의 편향이 생깁니다.이 문제는 $\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$이라는 사실을 사용하여 항을 다시 정규화함으로써 해결할 수 있습니다.이에 따라 정규화된 상태 변수는 다음과 같이 지정됩니다.  

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

적절한 추정치로 무장하여 이제 업데이트 방정식을 작성할 수 있습니다.먼저, RMSProp과 매우 유사한 방식으로 그래디언트를 다시 스케일링하여 

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

RMSProp과 달리 업데이트는 그래디언트 자체가 아닌 모멘텀 $\hat{\mathbf{v}}_t$를 사용합니다.또한 스케일링이 $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$ 대신 $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$을 사용하여 발생하므로 약간의 외관상의 차이가 있습니다.전자는 실제로 약간 더 잘 작동하므로 RMSProp과의 편차가 있습니다.일반적으로 수치 안정성과 충실도 간의 적절한 절충을 위해 $\epsilon = 10^{-6}$을 선택합니다.  

이제 업데이트를 계산할 모든 요소가 준비되었습니다.이것은 약간 항변성이며 양식을 간단하게 업데이트했습니다. 

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

Adam의 디자인을 검토하면서 영감을 얻은 것은 분명합니다.모멘텀과 스케일은 상태 변수에서 명확하게 볼 수 있습니다.다소 독특한 정의는 용어를 디바이어스하도록 강요합니다 (초기화 및 업데이트 조건이 약간 다르면 해결할 수 있습니다).둘째, RMSProp을 고려할 때 두 용어의 조합은 매우 간단합니다.마지막으로 명시적 학습률 $\eta$를 사용하면 수렴 문제를 해결하기 위해 스텝 길이를 제어할 수 있습니다.  

## 구현 

Adam을 처음부터 구현하는 것은 그리 어려운 일이 아닙니다.편의를 위해 시간 단계 카운터 $t$를 `hyperparams` 사전에 저장합니다.그 외에도 모든 것이 간단합니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

Adam을 사용하여 모델을 훈련시킬 준비가 되었습니다.우리는 $\eta = 0.01$의 학습률을 사용합니다.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

`adam`는 Gluon `trainer` 최적화 라이브러리의 일부로 제공되는 알고리즘 중 하나이므로 보다 간결한 구현은 간단합니다.따라서 Gluon에서 구현에 대한 구성 매개 변수 만 전달하면 됩니다.

```{.python .input}
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## 요가 수행자

아담의 문제점 중 하나는 $\mathbf{s}_t$의 두 번째 모멘트 추정치가 폭파되면 볼록한 설정에서도 수렴하지 못할 수 있다는 것입니다.수정 사항으로 :cite:`Zaheer.Reddi.Sachan.ea.2018`는 $\mathbf{s}_t$에 대한 세련된 업데이트 (및 초기화) 를 제안했습니다.무슨 일이 일어나고 있는지 이해하기 위해 다음과 같이 Adam 업데이트를 다시 작성해 보겠습니다. 

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

$\mathbf{g}_t^2$의 분산이 높거나 업데이트가 희소할 때마다 $\mathbf{s}_t$은 과거 값을 너무 빨리 잊어버릴 수 있습니다.이 문제를 해결할 수 있는 방법은 $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$를 $\mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$로 교체하는 것입니다.이제 업데이트의 크기는 더 이상 편차의 양에 의존하지 않습니다.이렇게 하면 Yogi 업데이트가 생성됩니다. 

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

또한 저자는 초기 점별 추정치가 아닌 더 큰 초기 배치에서 모멘텀을 초기화하는 것이 좋습니다.세부 사항은 토론에 중요하지 않으며 이러한 수렴이 없어도 꽤 좋기 때문에 세부 사항을 생략합니다.

```{.python .input}
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## 요약

* Adam은 많은 최적화 알고리즘의 기능을 상당히 강력한 업데이트 규칙으로 결합합니다. 
* RMSProp을 기반으로 만들어진 Adam은 미니배치 확률적 기울기에도 EWMA를 사용합니다.
* Adam은 모멘텀과 두 번째 모멘트를 추정 할 때 바이어스 보정을 사용하여 느린 시작을 조정합니다. 
* 분산이 유의한 그라디언트의 경우 수렴 문제가 발생할 수 있습니다.더 큰 미니 배치를 사용하거나 $\mathbf{s}_t$에 대한 향상된 추정치로 전환하여 수정할 수 있습니다.요기는 이러한 대안을 제공합니다. 

## 연습문제

1. 학습률을 조정하고 실험 결과를 관찰하고 분석합니다.
1. 편향 보정이 필요하지 않도록 모멘텀과 세컨드 모멘트 업데이트를 다시 작성할 수 있습니까?
1. 수렴할 때 학습률 $\eta$를 줄여야 하는 이유는 무엇입니까?
1. 아담이 갈라지고 요기가 수렴하는 사례를 만들어보십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1078)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1079)
:end_tab:
