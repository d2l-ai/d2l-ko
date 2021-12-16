# 주의 풀링: 나다라야-왓슨 커널 회귀
:label:`sec_nadaraya-watson`

이제 :numref:`fig_qkv`의 프레임 워크 하에서주의 메커니즘의 주요 구성 요소를 알게되었습니다.요약하면 쿼리 (의지 큐) 와 키 (비자발적 큐) 간의 상호 작용은*주의 풀링*을 생성합니다.주의력 풀링은 값 (감각 입력) 을 선택적으로 집계하여 출력을 생성합니다.이 섹션에서는 주의력 풀링에 대해 자세히 설명하여 주의력 메커니즘이 실제로 어떻게 작동하는지에 대한 높은 수준의 뷰를 제공합니다.특히, 1964년에 제안된 Nadaraya-Watson 커널 회귀 모델은 주의력 메커니즘으로 기계 학습을 시연하는 간단하면서도 완전한 예입니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
tf.random.set_seed(seed=1322)
```

## [**데이터세트 생성**]

간단하게 유지하기 위해 다음과 같은 회귀 문제를 고려해 보겠습니다. 입력-출력 쌍 $\{(x_1, y_1), \ldots, (x_n, y_n)\}$의 데이터 세트가 주어지면 $f$을 학습하여 새 입력 $x$에 대한 출력 $\hat{y} = f(x)$을 예측하는 방법은 무엇입니까? 

여기서는 잡음 항 $\epsilon$를 사용하여 다음과 같은 비선형 함수에 따라 인공 데이터 세트를 생성합니다. 

$$y_i = 2\sin(x_i) + x_i^{0.8} + \epsilon,$$

여기서 $\epsilon$는 평균이 0이고 표준 편차가 0.5인 정규 분포를 따릅니다.50개의 교육 예제와 50개의 테스트 예제가 모두 생성됩니다.나중에 주의 패턴을 더 잘 시각화하기 위해 훈련 입력값이 정렬됩니다.

```{.python .input}
n_train = 50  # No. of training examples
x_train = np.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab pytorch
n_train = 50  # No. of training examples
x_train, _ = torch.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab tensorflow
n_train = 50
x_train = tf.sort(tf.random.uniform(shape=(n_train,), maxval=5))
```

```{.python .input}
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

```{.python .input}
#@tab pytorch
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

```{.python .input}
#@tab tensorflow
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal((n_train,), 0.0, 0.5)  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

다음 함수는 모든 훈련 예제 (원으로 표시됨), 잡음 항이 없는 지상 실측 데이터 생성 함수 `f` (“Truth”로 레이블 지정) 및 학습된 예측 함수 (“Pred”로 레이블 지정) 를 플로팅합니다.

```{.python .input}
#@tab all
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
```

## 평균 풀링

이 회귀 문제에 대한 세계에서 가장 멍청한 추정기로 시작합니다. 평균 풀링을 사용하여 모든 훈련 출력에서 평균을 구합니다. 

$$f(x) = \frac{1}{n}\sum_{i=1}^n y_i,$$
:eqlabel:`eq_avg-pooling`

아래에 그려져 있습니다.보시다시피, 이 추정기는 실제로 그렇게 똑똑하지 않습니다.

```{.python .input}
y_hat = y_train.mean().repeat(n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
y_hat = tf.repeat(tf.reduce_mean(y_train), repeats=n_test)
plot_kernel_reg(y_hat)
```

## [**비모수적 주의 풀링**]

분명히 평균 풀링은 입력 $x_i$을 생략합니다.나다라야 :cite:`Nadaraya.1964`와 왓슨 :cite:`Watson.1964`는 입력 위치에 따라 출력 $y_i$의 무게를 측정하기 위해 더 나은 아이디어를 제안했습니다. 

$$f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i,$$
:eqlabel:`eq_nadaraya-watson`

여기서 $K$은*커널*입니다.:eqref:`eq_nadaraya-watson`의 추정기를*나다라야-왓슨 커널 회귀*라고 합니다.여기서는 커널에 대해 자세히 설명하지 않겠습니다.:numref:`fig_qkv`에서 주의력 메커니즘의 틀을 상기하십시오.주의의 관점에서 :eqref:`eq_nadaraya-watson`를 보다 일반화된 형태의*주의력 풀링*으로 다시 작성할 수 있습니다. 

$$f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i,$$
:eqlabel:`eq_attn-pooling`

여기서 $x$는 쿼리이고 $(x_i, y_i)$은 키-값 쌍입니다.:eqref:`eq_attn-pooling`와 :eqref:`eq_avg-pooling`를 비교해 보면, 여기서 주의 풀링은 값 $y_i$의 가중 평균입니다.:eqref:`eq_attn-pooling`에서*주의 가중치* $\alpha(x, x_i)$은 쿼리 $x$와 $\alpha$에 의해 모델링된 키 $x_i$ 간의 상호 작용을 기반으로 해당 값 $y_i$에 할당됩니다.모든 쿼리에서 모든 키-값 쌍에 대한 주의 가중치는 유효한 확률 분포입니다. 즉, 음수가 아니며 합계가 최대 1입니다. 

주의력 풀링에 대한 직관을 얻으려면 다음과 같이 정의된*가우스 커널*을 고려하십시오. 

$$
K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2}).
$$

가우스 커널을 :eqref:`eq_attn-pooling` 및 :eqref:`eq_nadaraya-watson`에 연결하면 다음과 같은 이점을 얻을 수 있습니다. 

$$\begin{aligned} f(x) &=\sum_{i=1}^n \alpha(x, x_i) y_i\\ &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i. \end{aligned}$$
:eqlabel:`eq_nadaraya-watson-gaussian`

:eqref:`eq_nadaraya-watson-gaussian`에서 지정된 쿼리 $x$에 더 가까운 키 $x_i$는 다음과 같이 지정됩니다.
*키의 해당 값 $y_i$에 할당되는*더 큰 주의 가중치*를 통해 더 많은 주의*

특히 나다라야-왓슨 커널 회귀는 비모수적 모델이므로 :eqref:`eq_nadaraya-watson-gaussian`는*비모수적 주의 풀링*의 예입니다.다음에서는 이 비모수적 주의 모델을 기반으로 예측을 플로팅합니다.예측된 선은 평균 풀링으로 생성된 선보다 매끄럽고 실측 실수에 더 가깝습니다.

```{.python .input}
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = npx.softmax(-(X_repeat - x_train)**2 / 2)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat_interleave(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the 
# same testing inputs (i.e., same queries)
X_repeat = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = tf.nn.softmax(-(X_repeat - tf.expand_dims(x_train, axis=1))**2/2, axis=1)
# Each element of `y_hat` is weighted average of values, where weights are attention weights
y_hat = tf.matmul(attention_weights, tf.expand_dims(y_train, axis=1))
plot_kernel_reg(y_hat)
```

이제 [**주의 가중치**] 를 살펴보겠습니다.여기서 테스트 입력은 쿼리이고 훈련 입력은 키입니다.두 입력이 모두 정렬되어 있으므로 쿼리 키 쌍이 가까울수록 주의력 풀링에서 관심 가중치가 더 높다는 것을 알 수 있습니다.

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab tensorflow
d2l.show_heatmaps(tf.expand_dims(tf.expand_dims(attention_weights, axis=0), axis=0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## **파라메트릭 주의 풀링**

비모수적 Nadaraya-Watson 커널 회귀는*일관성*의 이점을 누립니다. 충분한 데이터가 주어지면 이 모델이 최적의 해로 수렴됩니다.그럼에도 불구하고 학습 가능한 매개 변수를 주의력 풀링에 쉽게 통합 할 수 있습니다. 

예를 들어, :eqref:`eq_nadaraya-watson-gaussian`와 약간 다른 경우, 다음에서는 쿼리 $x$과 키 $x_i$ 사이의 거리에 학습 가능한 매개 변수 $w$을 곱합니다. 

$$\begin{aligned}f(x) &= \sum_{i=1}^n \alpha(x, x_i) y_i \\&= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}((x - x_j)w)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i.\end{aligned}$$
:eqlabel:`eq_nadaraya-watson-gaussian-para`

나머지 섹션에서는 :eqref:`eq_nadaraya-watson-gaussian-para`에서 주의력 풀링의 매개 변수를 학습하여 이 모델을 학습합니다. 

### 배치 행렬 곱셈
:label:`subsec_batch_dot`

미니 배치에 대한 주의력을 보다 효율적으로 계산하기 위해 딥 러닝 프레임워크에서 제공하는 배치 행렬 곱셈 유틸리티를 활용할 수 있습니다. 

첫 번째 미니배치에는 형상 $a\times b$의 $n$ 행렬 $\mathbf{X}_1, \ldots, \mathbf{X}_n$이 있고 두 번째 미니배치에는 형상 $b\times c$의 $n$ 행렬 $\mathbf{Y}_1, \ldots, \mathbf{Y}_n$이 포함되어 있다고 가정합니다.배치 행렬 곱셈은 형상 $a\times c$의 $n$ 행렬 $\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$을 생성합니다.따라서 [**두 개의 텐서 형태 ($n$, $a$, $b$) 와 ($n$, $b$, $c$) 가 주어지면 배치 행렬 곱셈 출력의 모양은 ($n$, $a$, $c$) 입니다.**]

```{.python .input}
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
npx.batch_dot(X, Y).shape
```

```{.python .input}
#@tab pytorch
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
torch.bmm(X, Y).shape
```

```{.python .input}
#@tab tensorflow
X = tf.ones((2, 1, 4))
Y = tf.ones((2, 4, 6))
tf.matmul(X, Y).shape
```

주의 메커니즘의 맥락에서 [**미니배치 행렬 곱셈을 사용하여 미니배치에서 값의 가중 평균을 계산할 수 있습니다.**]

```{.python .input}
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20), (2, 10))
npx.batch_dot(np.expand_dims(weights, 1), np.expand_dims(values, -1))
```

```{.python .input}
#@tab pytorch
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20.0), (2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
```

```{.python .input}
#@tab tensorflow
weights = tf.ones((2, 10)) * 0.1
values = tf.reshape(tf.range(20.0), shape = (2, 10))
tf.matmul(tf.expand_dims(weights, axis=1), tf.expand_dims(values, axis=-1)).numpy()
```

### 모델 정의

아래에서는 미니배치 행렬 곱셈을 사용하여 :eqref:`eq_nadaraya-watson-gaussian-para`의 [**모수적 주의 풀링**] 을 기반으로 Nadaraya-Watson 커널 회귀의 파라메트릭 버전을 정의합니다.

```{.python .input}
class NWKernelRegression(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = self.params.get('w', shape=(1,))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = npx.softmax(
            -((queries - keys) * self.w.data())**2 / 2)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return npx.batch_dot(np.expand_dims(self.attention_weights, 1),
                             np.expand_dims(values, -1)).reshape(-1)
```

```{.python .input}
#@tab pytorch
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat_interleave(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
```

```{.python .input}
#@tab tensorflow
class NWKernelRegression(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(initial_value=tf.random.uniform(shape=(1,)))
        
    def call(self, queries, keys, values, **kwargs):
        # For training queries are `x_train`. Keys are distance of taining data for each point. Values are `y_train`.
        # Shape of the output `queries` and `attention_weights`: (no. of queries, no. of key-value pairs)
        queries = tf.repeat(tf.expand_dims(queries, axis=1), repeats=keys.shape[1], axis=1)
        self.attention_weights = tf.nn.softmax(-((queries - keys) * self.w)**2 /2, axis =1)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return tf.squeeze(tf.matmul(tf.expand_dims(self.attention_weights, axis=1), tf.expand_dims(values, axis=-1)))
```

### 트레이닝

다음에서는 주의 모델을 훈련시키기 위해 [**훈련 데이터 세트를 키와 값으로 변환**] 합니다.모수적 주의력 풀링에서 모든 훈련 입력은 출력을 예측하기 위해 자체를 제외한 모든 훈련 예제에서 키-값 쌍을 가져옵니다.

```{.python .input}
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = np.tile(x_train, (n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = np.tile(y_train, (n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).astype('bool')],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).astype('bool')],
                     (n_train, -1))
```

```{.python .input}
#@tab pytorch
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = x_train.repeat((n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = y_train.repeat((n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                     (n_train, -1))
```

```{.python .input}
#@tab tensorflow
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_train, axis=0)
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = tf.reshape(X_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = tf.reshape(Y_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
```

제곱 손실과 확률적 기울기 하강을 사용하여 [**파라메트릭 주의 모델을 훈련**] 합니다.

```{.python .input}
net = NWKernelRegression()
net.initialize()
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    with autograd.record():
        l = loss(net(x_train, keys, values), y_train)
    l.backward()
    trainer.step(1)
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
#@tab pytorch
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
#@tab tensorflow
net = NWKernelRegression()
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])


for epoch in range(5):
    with tf.GradientTape() as t:
        loss = loss_object(y_train, net(x_train, keys, values)) * len(y_train)
    grads = t.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    print(f'epoch {epoch + 1}, loss {float(loss):.6f}')
    animator.add(epoch + 1, float(loss))
```

모수적 주의 모델을 훈련시킨 후 [**예측을 플로팅**] 할 수 있습니다.훈련 데이터셋에 잡음을 맞추려고 할 때 예측된 선은 이전에 플로팅된 비모수적 대응 선보다 덜 매끄럽습니다.

```{.python .input}
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = np.tile(x_train, (n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = np.tile(y_train, (n_test, 1))
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = x_train.repeat((n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_test, axis=0)
# Shape of `value`: (`n_test`, `n_train`)
values = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_test, axis=0)
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

비모수적 주의력 풀링과 비교할 때, 학습 가능한 파라메트릭 설정에서 [**주의 가중치가 큰 영역이 더 선명해집니다**].

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(net.attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab tensorflow
d2l.show_heatmaps(tf.expand_dims(tf.expand_dims(net.attention_weights, axis=0), axis=0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## 요약

* 나다라야-왓슨 커널 회귀는 주의력 메커니즘을 사용하는 기계 학습의 예입니다.
* 나다라야-왓슨 커널 회귀의 주의력 풀링은 훈련 출력의 가중 평균입니다.주의 관점에서 주의 가중치는 쿼리의 함수 및 값과 쌍을 이루는 키를 기반으로 하는 값에 할당됩니다.
* 주의 풀링은 비모수적이거나 모수적일 수 있습니다.

## 연습문제

1. 훈련 예제의 수를 늘립니다.비모수 나다라야-왓슨 커널 회귀를 더 잘 배울 수 있습니까?
1. 모수 주의력 풀링 실험에서 학습한 $w$의 가치는 무엇입니까?주의력 가중치를 시각화할 때 가중 영역이 더 선명하게 되는 이유는 무엇입니까?
1. 더 나은 예측을 위해 비모수적 Nadaraya-Watson 커널 회귀에 초모수를 추가하려면 어떻게 해야 할까요?
1. 이 섹션의 커널 회귀를 위한 또 다른 모수적 주의 풀링을 설계합니다.이 새 모델을 훈련시키고 주의력 가중치를 시각화합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1599)
:end_tab:
