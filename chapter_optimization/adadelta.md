# 아다델타
:label:`sec_adadelta`

아다델타는 아다그라드의 또 다른 변종입니다 (:numref:`sec_adagrad`).가장 큰 차이점은 학습 속도가 좌표에 적응하는 양을 줄인다는 사실에 있습니다.또한 전통적으로 미래의 변화에 대한 교정으로 변화의 양 자체를 사용하기 때문에 학습률이 없다고 불렀습니다.이 알고리즘은 :cite:`Zeiler.2012`에서 제안되었습니다.지금까지 이전 알고리즘에 대한 논의를 감안할 때 매우 간단합니다.  

## 알고리즘

간단히 말해서 Adadelta는 두 가지 상태 변수 $\mathbf{s}_t$를 사용하여 기울기의 두 번째 모멘트의 누수 평균을 저장하고 $\Delta\mathbf{x}_t$를 사용하여 모델 자체에 매개 변수 변경의 두 번째 모멘트의 누수 평균을 저장합니다.다른 출판물 및 구현과의 호환성을 위해 저자의 원래 표기법과 이름을 사용합니다 (모멘텀, Adagrad, RMSProp 및 Adadelta에서 동일한 목적을 제공하는 매개 변수를 나타 내기 위해 다른 그리스어 변수를 사용해야하는 다른 실제 이유는 없습니다).  

다음은 Adadelta의 기술적 세부 사항입니다.매개 변수 뒤 주르가 $\rho$인 경우 :numref:`sec_rmsprop`와 유사하게 다음과 같은 누출 업데이트가 발생합니다. 

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

:numref:`sec_rmsprop`와의 차이점은 배율 조정된 그래디언트 $\mathbf{g}_t'$로 업데이트를 수행한다는 것입니다. 즉, 

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

그렇다면 스케일링된 그래디언트 $\mathbf{g}_t'$는 무엇입니까?다음과 같이 계산할 수 있습니다. 

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

여기서 $\Delta \mathbf{x}_{t-1}$는 배율 조정된 기울기 제곱 $\mathbf{g}_t'$의 누수 평균입니다.$\Delta \mathbf{x}_{0}$을 $0$으로 초기화하고 각 단계에서 $\mathbf{g}_t'$로 업데이트합니다. 즉, 

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

수치 안정성을 유지하기 위해 $\epsilon$ (예: $10^{-5}$와 같은 작은 값) 가 추가됩니다. 

## 구현

아다델타는 각 변수에 대해 $\mathbf{s}_t$와 $\Delta\mathbf{x}_t$라는 두 개의 상태 변수를 유지해야 합니다.이렇게 하면 다음과 같은 구현이 생성됩니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

$\rho = 0.9$를 선택하면 각 매개 변수 업데이트에 대해 반감기 시간이 10입니다.이것은 꽤 잘 작동하는 경향이 있습니다.다음과 같은 동작이 발생합니다.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

간결한 구현을 위해 `Trainer` 클래스의 `adadelta` 알고리즘을 사용하기만 하면 됩니다.이렇게 하면 훨씬 더 간결한 호출을 위해 다음과 같은 단일 줄기가 생성됩니다.

```{.python .input}
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadelta is not converging at default learning rate
# but it's converging at lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## 요약

* Adadelta에는 학습률 매개 변수가 없습니다.대신 매개 변수 자체의 변화율을 사용하여 학습률을 조정합니다. 
* Adadelta는 기울기의 두 번째 모멘트와 매개 변수의 변화를 저장하기 위해 두 개의 상태 변수가 필요합니다. 
* Adadelta는 누설 평균을 사용하여 적절한 통계량의 실행 추정치를 유지합니다. 

## 연습문제

1. $\rho$의 값을 조정합니다.어떻게 되나요?
1. $\mathbf{g}_t'$를 사용하지 않고 알고리즘을 구현하는 방법을 보여줍니다.왜 이게 좋은 생각일까요?
1. Adadelta는 실제로 학습 속도가 무료입니까?Adadelta를 깨뜨리는 최적화 문제를 찾을 수 있습니까?
1. Adadelta를 Adagrad 및 RMS 소품과 비교하여 수렴 동작에 대해 논의합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1076)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1077)
:end_tab:
