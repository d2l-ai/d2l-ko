# 학습 속도 스케줄링
:label:`sec_scheduler`

지금까지 우리는 가중치 벡터가 업데이트되는*rate*보다는 가중치 벡터를 업데이트하는 방법에 대한 최적화*알고리즘*에 중점을 두었습니다.그럼에도 불구하고 학습률을 조정하는 것은 실제 알고리즘만큼 중요한 경우가 많습니다.고려해야 할 몇 가지 측면이 있습니다. 

* 가장 명백하게 학습률의*크기*가 중요합니다.너무 크면 최적화가 갈라지고, 너무 작 으면 훈련하는 데 너무 오래 걸리거나 최적이 아닌 결과로 끝납니다.이전에 문제의 상태 번호가 중요하다는 것을 확인했습니다 (예: 자세한 내용은 :numref:`sec_momentum` 참조).직관적으로 가장 민감하지 않은 방향의 변화량과 가장 민감한 방향의 변화량의 비율입니다.
* 둘째, 부패 속도도 마찬가지로 중요합니다.학습 속도가 여전히 크면 최소값 주변에서 튀어 나와 최적성에 도달하지 못할 수 있습니다. :numref:`sec_minibatch_sgd`는 이에 대해 자세히 논의했으며 :numref:`sec_sgd`에서 성능 보증을 분석했습니다.요컨대, 우리는 비율이 하락하기를 원하지만 아마도 $\mathcal{O}(t^{-\frac{1}{2}})$보다 더 느릴 것입니다. 이는 볼록 문제에 좋은 선택이 될 것입니다.
* 마찬가지로 중요한 또 다른 측면은*초기화*입니다.이는 매개변수가 초기에 설정되는 방식 (자세한 내용은 :numref:`sec_numerical_stability` 검토) 과 초기에 어떻게 진화하는지와 관련이 있습니다.이것은*워밍업*이라는 모니 커 아래에 있습니다. 즉, 처음에 솔루션을 향해 얼마나 빨리 이동하기 시작했는지입니다.처음에 큰 스텝은 유용하지 않을 수 있습니다. 특히 초기 파라미터 세트가 랜덤이기 때문입니다.초기 업데이트 방향도 의미가 없을 수 있습니다.
* 마지막으로, 주기적 학습률 조정을 수행하는 여러 가지 최적화 변형이 있습니다.이것은 현재 장의 범위를 벗어납니다.독자는 :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`의 세부 사항 (예: 매개 변수의 전체*경로*에 대한 평균을 계산하여 더 나은 솔루션을 얻는 방법) 을 검토하는 것이 좋습니다.

학습률을 관리하는 데 많은 세부 정보가 필요하다는 사실을 감안할 때 대부분의 딥 러닝 프레임워크에는 이를 자동으로 처리할 수 있는 도구가 있습니다.현재 장에서는 서로 다른 일정이 정확성에 미치는 영향을 검토하고*학습률 스케줄러*를 통해 이를 효율적으로 관리할 수 있는 방법을 보여줍니다. 

## 장난감 문제

우리는 쉽게 계산할 수 있을 만큼 저렴하지만 몇 가지 주요 측면을 설명하기에는 충분히 중요하지 않은 장난감 문제로 시작합니다.이를 위해 패션-MNIST에 적용되는 약간 현대화 된 버전의 LeNet (`sigmoid` 활성화 대신 `relu`, 평균 풀링이 아닌 최대 풀링) 을 선택합니다.또한 성능을 위해 네트워크를 하이브리드화합니다.대부분의 코드는 표준이므로 더 자세한 설명 없이 기본 사항을 소개하기만 하면 됩니다.필요에 따라 재교육을 받으려면 :numref:`chap_cnn`를 참조하십시오.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    
        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0, 
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

학습률 $0.3$와 같은 기본 설정으로 이 알고리즘을 호출하고 $30$ 반복을 위해 훈련하면 어떤 일이 발생하는지 살펴보겠습니다.테스트 정확도 측면에서 진행이 한 지점을 넘어서 멈추는 동안 훈련 정확도가 계속 증가하는 방법에 유의하십시오.두 곡선 사이의 간격은 과적합을 나타냅니다.

```{.python .input}
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## 스케줄러

학습률을 조정하는 한 가지 방법은 각 단계에서 학습률을 명시적으로 설정하는 것입니다.이는 `set_learning_rate` 방법으로 편리하게 달성됩니다.예를 들어 최적화가 진행되는 방식에 따라 동적으로 매 Epoch 이후 (또는 모든 미니 배치 후) 를 아래쪽으로 조정할 수 있습니다.

```{.python .input}
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

좀 더 일반적으로 스케줄러를 정의하려고 합니다.업데이트 횟수와 함께 호출되면 적절한 학습률 값을 반환합니다.학습률을 $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$로 설정하는 간단한 것을 정의해 보겠습니다.

```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

값의 범위에 대한 동작을 플로팅해 보겠습니다.

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

이제 패션 MNIST에 대한 교육을 위해 이것이 어떻게 진행되는지 살펴 보겠습니다.훈련 알고리즘에 대한 추가 인수로 스케줄러를 제공하기만 하면 됩니다.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

이것은 이전보다 훨씬 더 잘 작동했습니다.두 가지가 두드러집니다. 곡선이 이전보다 훨씬 매끄 럽습니다.둘째, 과적합이 적었습니다.안타깝게도 특정 전략이*이론*에서 과적합을 줄이는 이유에 대해서는 잘 해결된 질문이 아닙니다.단계 크기가 작을수록 매개 변수가 0에 가까워지고 따라서 더 간단해진다는 주장이 있습니다.그러나 이것은 우리가 실제로 일찍 멈추지 않고 단순히 학습 속도를 부드럽게 낮추기 때문에 현상을 완전히 설명하지는 않습니다. 

## 정책

다양한 학습률 스케줄러를 다룰 수는 없지만 아래에서 인기 있는 정책에 대한 간략한 개요를 제공하려고 합니다.일반적으로 다항식 감쇠와 조각별 상수 일정을 선택할 수 있습니다.그 외에도 코사인 학습률 일정은 일부 문제에서 경험적으로 잘 작동하는 것으로 밝혀졌습니다.마지막으로, 일부 문제의 경우 큰 학습률을 사용하기 전에 옵티마이저를 예열하는 것이 좋습니다. 

### 팩터 스케줄러

다항식 붕괴에 대한 한 가지 대안은 승법, 즉 $\alpha \in (0, 1)$의 경우 $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$입니다.학습률이 합리적인 하한을 초과하여 감소하는 것을 방지하기 위해 업데이트 방정식은 종종 $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$으로 수정됩니다.

```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

이 작업은 `lr_scheduler.FactorScheduler` 개체를 통해 MXNet의 기본 제공 스케줄러로도 수행할 수 있습니다.워밍업 기간, 워밍업 모드 (선형 또는 상수), 원하는 업데이트의 최대 수 등과 같은 몇 가지 매개 변수가 더 필요합니다. 앞으로는 내장 스케줄러를 적절하게 사용하고 여기에서만 기능을 설명합니다.설명된 대로 필요한 경우 자체 스케줄러를 구축하는 것은 매우 간단합니다. 

### 멀티 팩터 스케줄러

심층 네트워크를 훈련시키는 일반적인 전략은 학습률을 조각별로 일정하게 유지하고 주어진 양만큼 자주 줄이는 것입니다.즉, $s = \{5, 10, 20\}$이 $t \in s$가 될 때마다 $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$를 줄이는 것과 같이 비율을 낮출 때가 주어지면 됩니다.각 단계에서 값이 절반으로 줄어든다고 가정하면 다음과 같이 구현할 수 있습니다.

```{.python .input}
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler) 
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr
  
    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

이 조각별 상수 학습률 스케줄의 직관은 가중치 벡터의 분포 측면에서 고정점에 도달할 때까지 최적화를 진행할 수 있다는 것입니다.그런 다음 (그리고 나서야) 더 높은 품질의 프록시를 좋은 지역 최소값으로 얻는 것과 같은 비율을 낮춥니다.아래 예제는 이것이 어떻게 조금 더 나은 솔루션을 만들 수 있는지를 보여줍니다.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### 코사인 스케줄러

:cite:`Loshchilov.Hutter.2016`에 의해 다소 난처한 휴리스틱이 제안되었습니다.처음에는 학습률을 너무 크게 낮추고 싶지 않을 수도 있고, 또한 매우 작은 학습률을 사용하여 결국 솔루션을 “개선”하고 싶을 수도 있다는 관찰에 의존합니다.그 결과 $t \in [0, T]$ 범위의 학습률에 대해 다음과 같은 기능적 형식을 가진 코사인과 같은 일정이 생성됩니다. 

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$

여기서 $\eta_0$는 초기 학습률이고, $\eta_T$은 시간 $T$의 목표 속도입니다.또한 $t > T$의 경우 값을 다시 늘리지 않고 $\eta_T$에 고정하기만 하면 됩니다.다음 예에서는 최대 업데이트 단계 $T = 20$을 설정합니다.

```{.python .input}
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps
  
    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

컴퓨터 비전의 맥락에서 이 일정은*할 수 있습니다* 향상된 결과를 가져올 수 있습니다.그러나 이러한 개선은 보장되지 않습니다 (아래에서 볼 수 있음).

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### 워밍업

경우에 따라 매개 변수를 초기화하는 것만으로는 좋은 솔루션을 보장할 수 없습니다.이는 특히 불안정한 최적화 문제를 야기할 수 있는 일부 고급 네트워크 설계에서 문제가 됩니다.처음에는 발산을 방지하기 위해 충분히 작은 학습률을 선택함으로써 이 문제를 해결할 수 있습니다.안타깝게도 이는 진행 속도가 느리다는 것을 의미합니다.반대로, 학습률이 크면 처음에는 발산이 발생합니다. 

이 딜레마에 대한 간단한 해결책은 학습률이 초기 최대값으로*증가하는* 워밍업 기간을 사용하고 최적화 프로세스가 끝날 때까지 속도를 낮추는 것입니다.단순화를 위해 일반적으로 이러한 목적으로 선형 증가를 사용합니다.이로 인해 아래 표시된 양식의 일정이 표시됩니다.

```{.python .input}
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

처음에는 네트워크가 더 잘 수렴된다는 점에 유의하십시오 (특히 처음 5 epoch 동안의 성능을 관찰하십시오).

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

워밍업은 코사인뿐만 아니라 모든 스케줄러에 적용할 수 있습니다.학습률 일정 및 더 많은 실험에 대한 자세한 내용은 :cite:`Gotmare.Keskar.Xiong.ea.2018`를 참조하십시오.특히 워밍업 단계가 매우 깊은 네트워크에서 매개 변수의 발산 양을 제한한다는 것을 발견했습니다.처음에 진행하는 데 가장 많은 시간이 걸리는 네트워크 부분에서 무작위 초기화로 인해 상당한 차이가 발생할 것으로 예상되므로 직관적으로 의미가 있습니다. 

## 요약

* 훈련 중에 학습률을 줄이면 정확도가 향상되고 (가장 당혹 스럽지만) 모델의 과적합이 줄어들 수 있습니다.
* 실제로 진행률이 정체 될 때마다 학습률을 조각별로 낮추는 것이 효과적입니다.기본적으로 이를 통해 적절한 해에 효율적으로 수렴하고 학습률을 줄임으로써 매개 변수의 고유 분산을 줄일 수 있습니다.
* 코사인 스케줄러는 일부 컴퓨터 비전 문제에 널리 사용됩니다.이러한 스케줄러에 대한 자세한 내용은 예를 들어 [GluonCV](http://gluon-cv.mxnet.io)를 참조하십시오.
* 최적화 전의 워밍업 기간은 발산을 방지할 수 있습니다.
* 최적화는 딥 러닝에서 다양한 용도로 사용됩니다.훈련 목표를 최소화하는 것 외에도 최적화 알고리즘과 학습률 스케줄링을 다르게 선택하면 테스트 세트에 대한 일반화 및 과적합의 양이 다소 다를 수 있습니다 (동일한 양의 훈련 오차에 대해).

## 연습문제

1. 주어진 고정 학습률에 대한 최적화 동작을 실험합니다.이런 식으로 얻을 수 있는 가장 좋은 모델은 무엇입니까?
1. 학습률 감소의 지수를 변경하면 수렴은 어떻게 변합니까?실험의 편의를 위해 `PolyScheduler`를 사용하십시오.
1. 코사인 스케줄러를 대규모 컴퓨터 시각 문제 (예: ImageNet 훈련) 에 적용합니다.다른 스케줄러에 비해 성능에 어떤 영향을 미칩니 까?
1. 워밍업은 얼마나 오래 지속되어야 합니까?
1. 최적화와 샘플링을 연결할 수 있습니까?스토캐스틱 그라데이션 랑게빈 역학에 대한 :cite:`Welling.Teh.2011`의 결과를 사용하여 시작합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/359)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1080)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1081)
:end_tab:
