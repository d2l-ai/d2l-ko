# RMS 프롭
:label:`sec_rmsprop`

:numref:`sec_adagrad`의 주요 문제 중 하나는 효과적으로 $\mathcal{O}(t^{-\frac{1}{2}})$의 사전 정의된 일정에 따라 학습률이 감소한다는 것입니다.이 방법은 일반적으로 볼록 문제에 적합하지만 딥러닝에서 발생하는 것과 같이 볼록하지 않은 문제에는 적합하지 않을 수 있습니다.그러나 Adagrad의 좌표 별 적응성은 선조건자로서 매우 바람직합니다. 

:cite:`Tieleman.Hinton.2012`는 속도 스케줄링을 좌표 적응 학습 속도에서 분리하는 간단한 수정으로 RMSProp 알고리즘을 제안했습니다.문제는 아다그라드가 기울기 $\mathbf{g}_t$의 제곱을 상태 벡터 $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$에 누적한다는 것입니다.결과적으로 $\mathbf{s}_t$는 정규화 부족으로 인해 한계 없이 계속 성장하고 있으며, 알고리즘이 수렴함에 따라 기본적으로 선형적으로 증가합니다. 

이 문제를 해결하는 한 가지 방법은 $\mathbf{s}_t / t$를 사용하는 것입니다.$\mathbf{g}_t$의 합리적인 분포의 경우 이 값이 수렴됩니다.안타깝게도 절차에서 값의 전체 궤적을 기억하기 때문에 한계 동작이 중요해지기 시작할 때까지는 매우 오랜 시간이 걸릴 수 있습니다.대안은 모멘텀 방법에서 사용한 것과 동일한 방식으로 누설 평균을 사용하는 것입니다 (예: 일부 매개 변수 $\gamma > 0$에 대해 $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$).다른 모든 부분을 변경하지 않고 유지하면 RMSProp이 생성됩니다 

## 알고리즘

방정식을 자세히 작성해 보겠습니다. 

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

상수 $\epsilon > 0$는 일반적으로 $10^{-6}$으로 설정되어 0 또는 지나치게 큰 스텝 크기로 나누지 않습니다.이 확장을 감안할 때 이제 좌표별로 적용되는 스케일링과 독립적으로 학습률 $\eta$를 자유롭게 제어 할 수 있습니다.누설 평균의 경우 모멘텀 방법의 경우 이전에 적용된 것과 동일한 추론을 적용 할 수 있습니다.$\mathbf{s}_t$ 수익률의 정의 확대 

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

이전과 마찬가지로 :numref:`sec_momentum`에서는 $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$를 사용합니다.따라서 가중치의 합은 관측치의 반감기 시간이 $\gamma^{-1}$인 $1$로 정규화됩니다.$\gamma$의 다양한 선택 항목에 대한 지난 40개 시간 스텝의 가중치를 시각화해 보겠습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## 처음부터 구현

이전과 마찬가지로 이차 함수 $f(\mathbf{x})=0.1x_1^2+2x_2^2$를 사용하여 RMSProp의 궤적을 관찰합니다.:numref:`sec_adagrad`에서 학습률이 0.4 인 Adagrad를 사용했을 때 학습 속도가 너무 빨리 감소했기 때문에 알고리즘의 후반 단계에서 변수가 매우 느리게 움직였습니다.$\eta$은 별도로 제어되므로 RMSProp에서는 이런 일이 발생하지 않습니다.

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

다음으로 심층 네트워크에서 사용할 RMSProp을 구현합니다.이것은 똑같이 간단합니다.

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

초기 학습률을 0.01로 설정하고 가중치 용어 $\gamma$을 0.9로 설정했습니다.즉, $\mathbf{s}$는 제곱 기울기의 지난 $1/(1-\gamma) = 10$개 관측치에 대한 평균적으로 집계됩니다.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## 간결한 구현

RMSProp은 널리 사용되는 알고리즘이므로 `Trainer` 인스턴스에서도 사용할 수 있습니다.`rmsprop`라는 알고리즘을 사용하여 인스턴스화하고 $\gamma$을 매개 변수 `gamma1`에 할당하기만 하면 됩니다.

```{.python .input}
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## 요약

* RMSProp은 두 가지 모두 그래디언트의 제곱을 사용하여 계수를 스케일링한다는 점에서 Adagrad와 매우 유사합니다.
* RMSProp은 누설 평균을 모멘텀과 공유합니다.그러나 RMSProp은 이 기법을 사용하여 계수별 선조건자를 조정합니다.
* 학습률은 실제로 실험자가 계획해야 합니다.
* 계수 $\gamma$는 좌표별 척도를 조정할 때 이력이 얼마나 오래 지속되는지를 결정합니다.

## 연습문제

1. $\gamma = 1$를 설정하면 실험적으로 어떤 일이 발생합니까?왜요?
1. 최적화 문제를 회전하여 $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$를 최소화합니다.컨버전스는 어떻게 되나요?
1. 패션-MNIST에 대한 교육과 같은 실제 기계 학습 문제에 대해 RMSPprop에 어떤 일이 발생하는지 시험해 보십시오.학습률을 조정하기 위해 다양한 선택 사항을 실험해 보십시오.
1. 최적화가 진행됨에 따라 $\gamma$를 조정하시겠습니까?RMSProp은 이 문제에 얼마나 민감한가요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1075)
:end_tab:
