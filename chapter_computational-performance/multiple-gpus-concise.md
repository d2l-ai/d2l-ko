# 여러 GPU를 위한 간결한 구현
:label:`sec_multi_gpu_concise`

모든 새 모델에 대해 처음부터 병렬 처리를 구현하는 것은 재미가 없습니다.또한 고성능을 위해 동기화 도구를 최적화하면 상당한 이점이 있습니다.다음에서는 딥러닝 프레임워크의 상위 수준 API를 사용하여 이 작업을 수행하는 방법을 보여줍니다.수학과 알고리즘은 :numref:`sec_multi_gpu`와 동일합니다.당연히 이 섹션의 코드를 실행하려면 최소 두 개의 GPU가 필요합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**장난감 네트워크**]

:numref:`sec_multi_gpu`의 LeNet보다 약간 더 의미 있는 네트워크를 사용하겠습니다. 이 네트워크는 여전히 충분히 쉽고 빠르게 훈련할 수 있습니다.우리는 레스넷-18 변종 :cite:`He.Zhang.Ren.ea.2016`를 선택합니다.입력 이미지가 작기 때문에 약간 수정합니다.특히 :numref:`sec_resnet`와의 차이점은 처음에 더 작은 컨볼루션 커널, 스트라이드 및 패딩을 사용한다는 것입니다.또한 최댓값 풀링 계층을 제거합니다.

```{.python .input}
#@save
def resnet18(num_classes):
    """A slightly modified ResNet-18 model."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## 네트워크 초기화

:begin_tab:`mxnet`
`initialize` 함수를 사용하면 선택한 장치에서 매개 변수를 초기화 할 수 있습니다.초기화 방법에 대한 자세한 내용은 :numref:`sec_numerical_stability`를 참조하십시오.*여러* 장치에서 동시에 네트워크를 초기화 할 수 있다는 점이 특히 편리합니다.이것이 실제로 어떻게 작동하는지 시도해 보겠습니다.
:end_tab:

:begin_tab:`pytorch`
훈련 루프 내에서 네트워크를 초기화합니다.초기화 방법에 대한 자세한 내용은 :numref:`sec_numerical_stability`를 참조하십시오.
:end_tab:

```{.python .input}
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# Initialize all the parameters of the network
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```

:begin_tab:`mxnet`
:numref:`sec_multi_gpu`에 도입된 `split_and_load` 함수를 사용하여 데이터의 미니 배치를 나누고 `devices` 변수가 제공하는 장치 목록에 부분을 복사할 수 있습니다.네트워크 인스턴스는*자동으로* 적절한 GPU를 사용하여 순방향 전파 값을 계산합니다.여기서는 4개의 관측값을 생성하고 GPU로 분할합니다.
:end_tab:

```{.python .input}
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
데이터가 네트워크를 통과하면 해당 매개 변수가 초기화됩니다*데이터가 통과 한 장치에서*.즉, 초기화는 장치별로 수행됩니다.초기화를 위해 GPU 0과 GPU 1을 선택했기 때문에 네트워크는 CPU가 아닌 거기에서만 초기화됩니다.실제로 매개 변수는 CPU에도 존재하지 않습니다.매개변수를 인쇄하고 발생할 수 있는 오류를 관찰하여 이를 확인할 수 있습니다.
:end_tab:

```{.python .input}
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
다음으로 [**정확도 평가**] 코드를 작동하는 코드 (**여러 장치에서 병렬로 사용**) 로 바꾸겠습니다.이는 :numref:`sec_lenet`에서 `evaluate_accuracy_gpu` 함수를 대체하는 역할을 합니다.가장 큰 차이점은 네트워크를 호출하기 전에 미니 배치를 분할한다는 것입니다.다른 모든 것은 본질적으로 동일합니다.
:end_tab:

```{.python .input}
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    # Query the list of devices
    devices = list(net.collect_params().values())[0].list_ctx()
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**교육**]

이전과 마찬가지로 훈련 코드는 효율적인 병렬 처리를 위해 몇 가지 기본 기능을 수행해야 합니다. 

* 모든 디바이스에서 네트워크 파라미터를 초기화해야 합니다.
* 데이터 세트를 반복하는 동안 미니 배치는 모든 장치로 분할됩니다.
* 장치 간에 손실과 기울기를 병렬로 계산합니다.
* 그라디언트가 집계되고 그에 따라 매개 변수가 업데이트됩니다.

결국 네트워크의 최종 성능을 보고하기 위해 정확도 (다시 병렬로) 를 계산합니다.훈련 루틴은 데이터를 분할하고 집계해야 한다는 점을 제외하면 이전 장의 구현과 매우 유사합니다.

```{.python .input}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    # Set the model on multiple GPUs
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

이것이 실제로 어떻게 작동하는지 봅시다.워밍업으로 [**단일 GPU에서 네트워크를 훈련시키십시오.**]

```{.python .input}
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

다음으로 [**교육에 2개의 GPU를 사용**] 합니다.:numref:`sec_multi_gpu`에서 평가된 LeNet과 비교할 때 ResNet-18에 대한 모델은 훨씬 더 복잡합니다.여기서 병렬화가 이점을 보여줍니다.계산 시간은 매개 변수를 동기화하는 시간보다 훨씬 큽니다.이렇게 하면 병렬화에 대한 오버헤드가 덜 관련되므로 확장성이 향상됩니다.

```{.python .input}
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## 요약

:begin_tab:`mxnet`
* Gluon은 컨텍스트 목록을 제공하여 여러 장치에서 모델 초기화를 위한 프리미티브를 제공합니다.
:end_tab:

* 데이터는 데이터를 찾을 수 있는 장치에서 자동으로 평가됩니다.
* 해당 장치의 매개 변수에 액세스하기 전에 각 장치의 네트워크를 초기화해야 합니다.그렇지 않으면 오류가 발생합니다.
* 최적화 알고리즘은 여러 GPU에 걸쳐 자동으로 집계됩니다.

## 연습문제

:begin_tab:`mxnet`
1. 이 섹션에서는 ResNet-18을 사용합니다.다양한 Epoch, 배치 크기 및 학습률을 시도해 보십시오.계산에 더 많은 GPU를 사용합니다.16개의 GPU (예: AWS p2.16xlarge 인스턴스) 로 이 작업을 시도하면 어떻게 됩니까?
1. 경우에 따라 장치마다 다른 컴퓨팅 성능을 제공합니다.GPU와 CPU를 동시에 사용할 수 있습니다.작업을 어떻게 나누어야 할까요?노력할만한 가치가 있니?왜요?왜 안 돼?
1. `npx.waitall()`를 떨어뜨리면 어떻게 될까요?병렬화를 위해 최대 두 단계가 겹치도록 훈련을 어떻게 수정하시겠습니까?
:end_tab:

:begin_tab:`pytorch`
1. 이 섹션에서는 ResNet-18을 사용합니다.다양한 Epoch, 배치 크기 및 학습률을 시도해 보십시오.계산에 더 많은 GPU를 사용합니다.16개의 GPU (예: AWS p2.16xlarge 인스턴스) 로 이 작업을 시도하면 어떻게 됩니까?
1. 경우에 따라 장치마다 다른 컴퓨팅 성능을 제공합니다.GPU와 CPU를 동시에 사용할 수 있습니다.작업을 어떻게 나누어야 할까요?노력할만한 가치가 있니?왜요?왜 안 돼?
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1403)
:end_tab:
