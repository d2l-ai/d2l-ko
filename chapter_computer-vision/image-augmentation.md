# 이미지 증강
:label:`sec_image_augmentation`

:numref:`sec_alexnet`에서는 대규모 데이터 세트가 다양한 응용 분야에서 심층 신경망의 성공을 위한 전제 조건이라고 언급했습니다.
*이미지 증대* 
는 훈련 영상의 일련의 무작위 변경 후 유사하지만 뚜렷한 훈련 예제를 생성하여 훈련 세트의 크기를 확장합니다.또는 훈련 예제의 무작위 조정으로 인해 모델이 특정 속성에 덜 의존하여 일반화 능력을 향상시킬 수 있다는 사실로 인해 이미지 증대가 동기를 부여 할 수 있습니다.예를 들어, 관심있는 객체가 다른 위치에 나타나도록 이미지를 여러 가지 방법으로 자르면 객체의 위치에 대한 모델의 의존성을 줄일 수 있습니다.또한 밝기 및 색상과 같은 요소를 조정하여 모델의 색상 민감도를 줄일 수 있습니다.당시 AlexNet의 성공을 위해 이미지 증대가 필수 불가결 한 것은 사실 일 것입니다.이 섹션에서는 컴퓨터 비전에서 널리 사용되는 기술에 대해 설명합니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
```

## 일반적인 이미지 증강 방법

일반적인 이미지 증강 방법을 조사 할 때 다음 $400\times 500$ 이미지를 예로 사용합니다.

```{.python .input}
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

대부분의 이미지 증강 방법에는 어느 정도의 임의성이 있습니다.이미지 증대의 효과를 더 쉽게 관찰 할 수 있도록 다음으로 보조 함수 `apply`를 정의합니다.이 함수는 입력 이미지 `img`에 대해 영상 증강 방법 `aug`를 여러 번 실행하고 모든 결과를 표시합니다.

```{.python .input}
#@tab all
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### 뒤집기 및 자르기

:begin_tab:`mxnet`
[**이미지를 좌우로 뒤집기**] 는 일반적으로 개체의 범주를 변경하지 않습니다.이것은 가장 초기의 가장 널리 사용되는 이미지 증강 방법 중 하나입니다.다음으로 `transforms` 모듈을 사용하여 이미지를 50% 확률로 좌우로 뒤집는 `RandomFlipLeftRight` 인스턴스를 만듭니다.
:end_tab:

:begin_tab:`pytorch`
[**이미지를 좌우로 뒤집기**] 는 일반적으로 개체의 범주를 변경하지 않습니다.이것은 가장 초기의 가장 널리 사용되는 이미지 증강 방법 중 하나입니다.다음으로 `transforms` 모듈을 사용하여 이미지를 50% 확률로 좌우로 뒤집는 `RandomHorizontalFlip` 인스턴스를 만듭니다.
:end_tab:

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

:begin_tab:`mxnet`
[**위아래로 뒤집기**] 는 좌우로 뒤집는 것만큼 일반적이지 않습니다.그러나 적어도 이 예제 이미지의 경우 위아래로 뒤집어도 인식이 방해되지는 않습니다.다음으로 `RandomFlipTopBottom` 인스턴스를 만들어 50% 확률로 이미지를 위아래로 뒤집습니다.
:end_tab:

:begin_tab:`pytorch`
[**위아래로 뒤집기**] 는 좌우로 뒤집는 것만큼 일반적이지 않습니다.그러나 적어도 이 예제 이미지의 경우 위아래로 뒤집어도 인식이 방해되지는 않습니다.다음으로 `RandomVerticalFlip` 인스턴스를 만들어 50% 확률로 이미지를 위아래로 뒤집습니다.
:end_tab:

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```

사용한 예제 이미지에서 고양이는 이미지 중간에 있지만 일반적으로 그렇지 않을 수 있습니다.:numref:`sec_pooling`에서는 풀링 계층이 대상 위치에 대한 컨벌루션 계층의 민감도를 줄일 수 있다고 설명했습니다.또한 이미지를 무작위로 자르면 물체가 이미지의 다른 위치에 다른 배율로 나타나게하여 대상 위치에 대한 모델의 민감도를 줄일 수도 있습니다. 

아래 코드에서는 면적이 $10 \sim 100$ of the original area each time, and the ratio of width to height of this area is randomly selected from $0.5 \sim 2$인 영역을 [**무작위로 자르기**] 합니다.그런 다음 영역의 너비와 높이가 모두 200픽셀로 조정됩니다.달리 명시되지 않는 한, 이 섹션에서 $a$과 $b$ 사이의 난수는 구간 $[a, b]$에서 랜덤하고 균일한 샘플링으로 얻은 연속형 값을 나타냅니다.

```{.python .input}
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### 색상 변경

또 다른 증강 방법은 색상을 변경하는 것입니다.이미지 색상의 네 가지 측면 (밝기, 대비, 채도 및 색조) 을 변경할 수 있습니다.아래 예에서는 이미지의 밝기를 원본 이미지의 50% ($1-0.5$) 에서 150% ($1+0.5$) 사이의 값으로 [**무작위로 변경**] 합니다.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

마찬가지로 이미지의 [**임의로 색조를 변경**] 할 수 있습니다.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

또한 `RandomColorJitter` 인스턴스를 만들고 [**이미지의 `brightness`, `contrast`, `saturation` 및 `hue`를 동시에 무작위로 변경**] 하는 방법을 설정할 수도 있습니다.

```{.python .input}
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### 여러 이미지 증강 방법 결합

실제로 우리는 [**여러 이미지 증강 방법을 결합**] 할 것입니다.예를 들어 위에서 정의한 다양한 이미지 증강 방법을 결합하여 `Compose` 인스턴스를 통해 각 이미지에 적용할 수 있습니다.

```{.python .input}
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## [**이미지 증대를 사용한 훈련**]

이미지 증대를 사용하여 모델을 훈련시켜 보겠습니다.여기서는 이전에 사용했던 패션-MNIST 데이터세트 대신 CIFAR-10 데이터세트를 사용합니다.이는 Fashion-MNIST 데이터셋에 있는 객체의 위치와 크기가 정규화된 반면, CIFAR-10 데이터셋에 있는 객체의 색상과 크기에는 더 큰 차이가 있기 때문입니다.CIFAR-10 데이터셋의 처음 32개의 훈련 이미지가 아래에 나와 있습니다.

```{.python .input}
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[0:32][0], 4, 8, scale=0.8);
```

```{.python .input}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

예측 중에 확실한 결과를 얻기 위해 일반적으로 훈련 예제에만 이미지 증대를 적용하고 예측 중에는 임의 연산과 함께 이미지 증대를 사용하지 않습니다.[**여기서는 가장 간단한 무작위 좌우 뒤집기 방법**] 만 사용합니다.또한 `ToTensor` 인스턴스를 사용하여 이미지의 미니 배치를 딥 러닝 프레임워크에 필요한 형식, 즉 (배치 크기, 채널 수, 높이, 너비) 모양의 0에서 1 사이의 32비트 부동 소수점 숫자로 변환합니다.

```{.python .input}
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

:begin_tab:`mxnet`
다음으로 이미지 읽기 및 이미지 증강 적용을 용이하게하는 보조 함수를 정의합니다.Gluon의 데이터 세트에서 제공하는 `transform_first` 함수는 각 훈련 예제 (이미지 및 레이블) 의 첫 번째 요소, 즉 이미지에 이미지 증대를 적용합니다.`DataLoader`에 대한 자세한 소개는 :numref:`sec_fashion_mnist`를 참조하십시오.
:end_tab:

:begin_tab:`pytorch`
다음으로, [**이미지 읽기 및 이미지 증강 적용을 용이하게 하는 보조 함수를 정의합니다**].파이토치의 데이터셋에서 제공하는 `transform` 인수는 증대를 적용하여 이미지를 변환합니다.`DataLoader`에 대한 자세한 소개는 :numref:`sec_fashion_mnist`를 참조하십시오.
:end_tab:

```{.python .input}
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

### 다중 GPU 교육

CIFAR-10 데이터세트에서 :numref:`sec_resnet`의 레스넷-18 모델을 훈련시킵니다.:numref:`sec_multi_gpu_concise`에서 다중 GPU 교육에 대한 소개를 상기하십시오.다음에서는 [**여러 GPU를 사용하여 모델을 훈련시키고 평가하는 함수를 정의합니다**].

```{.python .input}
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13)."""
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # The `True` flag allows parameters with stale gradients, which is useful
    # later (e.g., in fine-tuning BERT)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13)."""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    """Train a model with mutiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with mutiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

이제 [**`train_with_data_aug` 함수를 정의하여 이미지 증대를 사용하여 모델을 훈련**] 할 수 있습니다.이 함수는 사용 가능한 모든 GPU를 가져오고, Adam을 최적화 알고리즘으로 사용하며, 훈련 데이터 세트에 이미지 증대를 적용하고, 마지막으로 방금 정의한 `train_ch13` 함수를 호출하여 모델을 훈련시키고 평가합니다.

```{.python .input}
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

무작위 좌우 뒤집기를 기반으로 한 이미지 증대를 사용하여 [**모델 훈련**] 을 하겠습니다.

```{.python .input}
#@tab all
train_with_data_aug(train_augs, test_augs, net)
```

## 요약

* 이미지 증대는 기존 훈련 데이터를 기반으로 무작위 이미지를 생성하여 모델의 일반화 능력을 향상시킵니다.
* 예측 중에 확실한 결과를 얻기 위해 일반적으로 훈련 예제에만 이미지 증대를 적용하고 예측 중에는 임의 연산과 함께 이미지 증대를 사용하지 않습니다.
* 딥러닝 프레임워크는 동시에 적용할 수 있는 다양한 이미지 증강 방법을 제공합니다.

## 연습문제

1. 이미지 증대를 사용하지 않고 모델을 훈련시킵니다: `train_with_data_aug(test_augs, test_augs)`.이미지 증대를 사용하거나 사용하지 않을 때 훈련 및 테스트 정확도를 비교합니다.이 비교 실험이 이미지 증대가 과적합을 완화할 수 있다는 주장을 뒷받침할 수 있을까요?왜요?
1. CIFAR-10 데이터세트에 대한 모델 학습에서 다양한 이미지 증강 방법을 결합할 수 있습니다.테스트 정확도가 향상됩니까? 
1. 딥러닝 프레임워크의 온라인 설명서를 참조하십시오.다른 이미지 증강 방법도 제공합니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/367)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1404)
:end_tab:
