# 카글의 이미지 분류 (CIFAR-10)
:label:`sec_kaggle_cifar10`

지금까지 딥 러닝 프레임워크의 상위 수준 API를 사용하여 텐서 형식의 이미지 데이터 세트를 직접 얻었습니다.그러나 사용자 지정 이미지 데이터셋은 이미지 파일 형태로 제공되는 경우가 많습니다.이 섹션에서는 원시 이미지 파일부터 시작하여 단계별로 구성, 읽기 및 텐서 형식으로 변환합니다. 

우리는 컴퓨터 비전에서 중요한 데이터셋인 :numref:`sec_image_augmentation`의 CIFAR-10 데이터세트를 실험했습니다.이 섹션에서는 이전 섹션에서 배운 지식을 적용하여 CIFAR-10 이미지 분류의 Kaggle 경쟁을 연습합니다.(**대회의 웹 주소는 https://www.kaggle.com/c/cifar-10 **입니다) 

:numref:`fig_kaggle_cifar10`는 대회 웹 페이지의 정보를 보여줍니다.결과를 제출하려면 Kaggle 계정을 등록해야 합니다. 

![CIFAR-10 image classification competition webpage information. The competition dataset can be obtained by clicking the "Data" tab.](../img/kaggle-cifar10.png)
:width:`600px`
:label:`fig_kaggle_cifar10`

```{.python .input}
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, init, npx
from mxnet.gluon import nn
import os
import pandas as pd
import shutil

npx.set_np()
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import torchvision
from torch import nn
import os
import pandas as pd
import shutil
```

## 데이터세트 획득 및 구성

경쟁 데이터셋은 훈련 세트와 테스트 세트로 나뉘며, 여기에는 각각 50000개 및 300,000개의 이미지가 포함되어 있습니다.테스트 세트에서는 10000개의 이미지가 평가에 사용되고 나머지 290000 개의 이미지는 평가되지 않습니다. 속임수를 쓰기 어렵게 만들기 위해 포함됩니다.
*테스트 세트의 결과가 수동으로* 표시됩니다.
이 데이터셋의 이미지는 모두 png 색상 (RGB 채널) 이미지 파일이며 높이와 너비는 모두 32픽셀입니다.이미지는 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 보트 및 트럭과 같은 총 10 개의 범주를 포함합니다.:numref:`fig_kaggle_cifar10`의 왼쪽 상단 모서리에는 데이터셋에 있는 비행기, 자동차 및 새의 일부 이미지가 표시됩니다. 

### 데이터세트 다운로드

Kaggle에 로그인 한 후 :numref:`fig_kaggle_cifar10`에 표시된 CIFAR-10 이미지 분류 경쟁 웹 페이지에서 “데이터”탭을 클릭하고 “모두 다운로드”버튼을 클릭하여 데이터 세트를 다운로드 할 수 있습니다.`../data`에서 다운로드한 파일의 압축을 풀고 그 안에 `train.7z` 및 `test.7z`의 압축을 풀면 다음 경로에서 전체 데이터 세트를 찾을 수 있습니다. 

* `../data/cifar-10/train/[1-50000].png`
* `../data/cifar-10/test/[1-300000].png`
* `../data/cifar-10/trainLabels.csv`
* `../data/cifar-10/sampleSubmission.csv`

여기서 `train` 및 `test` 디렉토리에는 각각 교육 및 테스트 이미지가 포함되어 있습니다. `trainLabels.csv`은 교육 이미지에 대한 레이블을 제공하고 `sample_submission.csv`은 샘플 제출 파일입니다. 

더 쉽게 시작할 수 있도록 [**처음 1000개의 훈련 이미지와 5개의 무작위 테스트 이미지가 포함된 데이터세트의 소규모 샘플을 제공합니다.**] Kaggle 경쟁의 전체 데이터세트를 사용하려면 다음 `demo` 변수를 `False`로 설정해야 합니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# If you use the full dataset downloaded for the Kaggle competition, set
# `demo` to False
demo = True

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../data/cifar-10/'
```

### [**데이터세트 구성**]

모델 학습과 테스트를 용이하게 하기 위해 데이터세트를 구성해야 합니다.먼저 csv 파일에서 레이블을 읽어 보겠습니다.다음 함수는 파일 이름의 확장자가 아닌 부분을 레이블에 매핑하는 사전을 반환합니다.

```{.python .input}
#@tab all
#@save
def read_csv_labels(fname):
    """Read `fname` to return a filename to label dictionary."""
    with open(fname, 'r') as f:
        # Skip the file header line (column name)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# training examples:', len(labels))
print('# classes:', len(set(labels.values())))
```

다음으로 `reorg_train_valid` 함수를 정의하여 [**원래 훈련 세트에서 검증 세트를 분할합니다.**] 이 함수의 인수 `valid_ratio`은 유효성 검사 세트의 예제 수와 원래 훈련 세트의 예제 수의 비율입니다.좀 더 구체적으로 말하자면, $n$을 예제가 가장 적은 클래스의 이미지 수이고 $r$를 비율로 지정합니다.유효성 검사 세트는 각 클래스에 대해 $\max(\lfloor nr\rfloor,1)$개의 이미지를 분할합니다.`valid_ratio=0.1`를 예로 들어 보겠습니다.원래 훈련 세트에는 50000개의 이미지가 있으므로 경로 `train_valid_test/train`에는 훈련에 45000개의 이미지가 사용되고 나머지 5000개의 이미지는 경로 `train_valid_test/valid`의 유효성 검사 세트로 분할됩니다.데이터세트를 구성하면 같은 클래스의 이미지가 같은 폴더 아래에 배치됩니다.

```{.python .input}
#@tab all
#@save
def copyfile(filename, target_dir):
    """Copy a file into a target directory."""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """Split the validation set out of the original training set."""
    # The number of examples of the class that has the fewest examples in the
    # training dataset
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # The number of examples per class for the validation set
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label
```

아래 `reorg_test` 함수 [**예측 중 데이터 로드에 대한 테스트 세트를 구성합니다.**]

```{.python .input}
#@tab all
#@save
def reorg_test(data_dir):
    """Organize the testing set for data loading during prediction."""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
```

마지막으로 함수를 사용하여 `read_csv_labels`, `reorg_train_valid` 및 `reorg_test` (**위에서 정의한 함수**) 을 [**호출**] 합니다.

```{.python .input}
#@tab all
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
```

여기서는 데이터셋의 소규모 샘플에 대해서만 배치 크기를 32로 설정했습니다.Kaggle 대회의 전체 데이터세트를 훈련하고 테스트할 때는 `batch_size`를 128과 같이 더 큰 정수로 설정해야 합니다.훈련 예제의 10% 를 하이퍼파라미터 튜닝을 위한 검증 세트로 분할했습니다.

```{.python .input}
#@tab all
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
```

## [**이미지 증대**]

과적합을 해결하기 위해 이미지 증대를 사용합니다.예를 들어, 훈련 중에 이미지를 무작위로 가로로 뒤집을 수 있습니다.컬러 이미지의 RGB 채널 3개에 대한 표준화도 수행할 수 있습니다.아래에는 조정할 수 있는 이러한 작업 중 일부가 나열되어 있습니다.

```{.python .input}
transform_train = gluon.data.vision.transforms.Compose([
    # Scale the image up to a square of 40 pixels in both height and width
    gluon.data.vision.transforms.Resize(40),
    # Randomly crop a square image of 40 pixels in both height and width to
    # produce a small square of 0.64 to 1 times the area of the original
    # image, and then scale it to a square of 32 pixels in both height and
    # width
    gluon.data.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    # Standardize each channel of the image
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # Scale the image up to a square of 40 pixels in both height and width
    torchvision.transforms.Resize(40),
    # Randomly crop a square image of 40 pixels in both height and width to
    # produce a small square of 0.64 to 1 times the area of the original
    # image, and then scale it to a square of 32 pixels in both height and
    # width
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # Standardize each channel of the image
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

테스트 중에는 평가 결과에서 임의성을 제거하기 위해 이미지에 대해서만 표준화를 수행합니다.

```{.python .input}
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

## 데이터세트 읽기

다음으로 [**원시 이미지 파일로 구성된 구성된 데이터 세트를 읽습니다**].각 예제에는 이미지와 레이블이 포함되어 있습니다.

```{.python .input}
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ['train', 'valid', 'train_valid', 'test']]
```

```{.python .input}
#@tab pytorch
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

훈련 중에 [**위에서 정의한 모든 영상 증강 연산을 지정**] 해야 합니다.하이퍼파라미터 조정 중에 모델 평가에 검증 세트를 사용하는 경우 이미지 증대로 인한 임의성이 도입되지 않아야 합니다.최종 예측 전에 레이블이 지정된 모든 데이터를 최대한 활용할 수 있도록 결합된 훈련 세트와 검증 세트에 대해 모델을 훈련시킵니다.

```{.python .input}
train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True,
    last_batch='discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='keep')
```

```{.python .input}
#@tab pytorch
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

## [**모델**] 정의

:begin_tab:`mxnet`
여기서는 `HybridBlock` 클래스를 기반으로 잔차 블록을 빌드합니다. 이 클래스는 :numref:`sec_resnet`에 설명된 구현과 약간 다릅니다.이는 계산 효율성을 개선하기 위한 것입니다.
:end_tab:

```{.python .input}
class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y = F.npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.npx.relu(Y + X)
```

:begin_tab:`mxnet`
다음으로 ResNet-18 모델을 정의합니다.
:end_tab:

```{.python .input}
def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

:begin_tab:`mxnet`
교육을 시작하기 전에 :numref:`subsec_xavier`에 설명된 Xavier 초기화를 사용합니다.
:end_tab:

:begin_tab:`pytorch`
우리는 :numref:`sec_resnet`에 설명된 레스넷-18 모델을 정의합니다.
:end_tab:

```{.python .input}
def get_net(devices):
    num_classes = 10
    net = resnet18(num_classes)
    net.initialize(ctx=devices, init=init.Xavier())
    return net

loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")
```

## [**훈련 기능**] 정의

모델을 선택하고 검증 세트에서 모델의 성능에 따라 하이퍼파라미터를 조정합니다.다음에서는 모델 훈련 함수 `train`를 정의합니다.

```{.python .input}
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels.astype('float32'), loss, trainer,
                devices, d2l.split_batch)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpus(net, valid_iter,
                                                   d2l.split_batch)
            animator.add(epoch + 1, (None, None, valid_acc))
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**모델 교육 및 검증**]

이제 모델을 훈련하고 검증할 수 있습니다.다음의 모든 하이퍼파라미터를 조정할 수 있습니다.예를 들어, epoch의 수를 늘릴 수 있습니다.`lr_period` 및 `lr_decay`가 각각 4와 0.9로 설정된 경우 최적화 알고리즘의 학습률은 4시대마다 0.9를 곱합니다.시연을 쉽게하기 위해 여기서는 20 신기원 만 훈련합니다.

```{.python .input}
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 0.02, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**테스트 세트 분류**] 및 Kaggle에 결과 제출

하이퍼 파라미터가 있는 유망한 모델을 얻은 후 레이블이 지정된 모든 데이터 (검증 세트 포함) 를 사용하여 모델을 다시 학습하고 테스트 세트를 분류합니다.

```{.python .input}
net, preds = get_net(devices), []
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.as_in_ctx(devices[0]))
    preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)
```

```{.python .input}
#@tab pytorch
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
```

위의 코드는 `submission.csv` 파일을 생성하며, 이 파일의 형식은 Kaggle 경쟁사의 요구 사항을 충족합니다.결과를 Kaggle에 제출하는 방법은 :numref:`sec_kaggle_house`의 방법과 유사합니다. 

## 요약

* 원시 이미지 파일을 필요한 형식으로 구성한 후 데이터 세트를 읽을 수 있습니다.

:begin_tab:`mxnet`
* 이미지 분류 경쟁에서 컨벌루션 신경망, 이미지 증강 및 하이브리드 프로그래밍을 사용할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
* 이미지 분류 경쟁에서 컨벌루션 신경망과 이미지 증대를 사용할 수 있습니다.
:end_tab:

## 연습문제

1. 이 Kaggle 대회에 완전한 CIFAR-10 데이터세트를 사용하십시오.하이퍼파라미터를 `batch_size = 128`, `num_epochs = 100`, `lr = 0.1`, `lr_period = 50` 및 `lr_decay = 0.1`로 설정합니다.이 대회에서 어떤 정확도와 순위를 달성할 수 있는지 확인하십시오.좀 더 개선할 수 있을까요?
1. 이미지 증대를 사용하지 않을 때 어떤 정확도를 얻을 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/379)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1479)
:end_tab:
