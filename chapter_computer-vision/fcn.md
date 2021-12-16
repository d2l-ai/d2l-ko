# 완전 컨벌루션 네트워크
:label:`sec_fcn`

:numref:`sec_semantic_segmentation`에서 설명한 것처럼 시맨틱 분할은 이미지를 픽셀 수준으로 분류합니다.완전 컨벌루션 네트워크 (FCN) 는 컨벌루션 신경망을 사용하여 이미지 픽셀을 픽셀 클래스 :cite:`Long.Shelhamer.Darrell.2015`으로 변환한다.이전에 이미지 분류 또는 객체 감지를 위해 접한 CNN과 달리 완전 컨벌루션 네트워크는 중간 특징 맵의 높이와 너비를 입력 이미지의 높이와 너비로 다시 변환합니다. 이는 :numref:`sec_transposed_conv`에 도입된 전치된 컨벌루션 계층에 의해 달성됩니다.결과적으로 분류 출력과 입력 이미지는 픽셀 레벨에서 일대일 대응을 갖습니다. 즉, 출력 픽셀의 채널 차원은 동일한 공간 위치에 있는 입력 픽셀에 대한 분류 결과를 보유합니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
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
from torch.nn import functional as F
```

## 더 모델

여기서는 완전 컨벌루션 네트워크 모델의 기본 설계에 대해 설명합니다.:numref:`fig_fcn`에서 볼 수 있듯이 이 모델은 먼저 CNN을 사용하여 이미지 특징을 추출한 다음 $1\times 1$ 컨벌루션 계층을 통해 채널 수를 클래스 수로 변환한 다음 마지막으로 도입된 전치된 컨벌루션을 통해 특징 맵의 높이와 너비를 입력 이미지의 높이와 너비로 변환합니다.:numref:`sec_transposed_conv`년에 있습니다.결과적으로 모델 출력값은 입력 영상과 높이와 너비가 동일하며, 여기서 출력 채널에는 동일한 공간 위치에 있는 입력 픽셀에 대한 예측된 클래스가 포함됩니다.

![Fully convolutional network.](../img/fcn.svg)
:label:`fig_fcn`

아래에서는 [**ImageNet 데이터 세트에서 사전 훈련된 ResNet-18 모델을 사용하여 이미지 특징을 추출**] 하고 모델 인스턴스를 `pretrained_net`로 나타냅니다.이 모델의 마지막 몇 계층에는 전역 평균 풀링 계층과 완전 연결 계층이 포함됩니다. 완전 컨벌루션 네트워크에서는 필요하지 않습니다.

```{.python .input}
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-3:], pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
```

다음으로 [**완전 컨벌루션 네트워크 인스턴스 `net`**] 를 만듭니다.최종 전역 평균 풀링 계층과 출력값에 가장 가까운 완전 연결 계층을 제외하고 ResNet-18의 모든 사전 훈련된 계층을 복사합니다.

```{.python .input}
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

높이와 너비가 각각 320과 480인 입력이 주어지면 `net`의 순방향 전파는 입력 높이와 너비를 원본의 1/32, 즉 10과 15로 줄입니다.

```{.python .input}
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
```

다음으로 출력 채널 수를 Pascal VOC2012 데이터 세트의 클래스 수 (21) 로 변환하기 위해 $1\times 1$ 컨벌루션 계층을 사용합니다. 마지막으로, 입력 이미지의 높이와 너비로 다시 변경하려면 (**특징 맵의 높이와 너비를 32 배 증가**) 해야합니다.:numref:`sec_padding`에서 컨벌루션 계층의 출력 형상을 계산하는 방법을 생각해 보십시오.$(320-64+16\times2+32)/32=10$과 $(480-64+16\times2+32)/32=15$부터 보폭이 $32$인 전치된 컨벌루션 계층을 구성하여 커널의 높이와 너비를 $64$로 설정하고 패딩을 $16$로 설정합니다.일반적으로 스트라이드 $s$, 패딩 $s/2$ ($s/2$이 정수라고 가정) 및 커널 $2s$의 높이와 너비에 대해 전치된 컨벌루션이 입력값의 높이와 너비를 $s$배 증가시킨다는 것을 알 수 있습니다.

```{.python .input}
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

```{.python .input}
#@tab pytorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

## [**전치된 컨벌루션 레이어 초기화**]

우리는 전치된 컨벌루션 계층이 특징 맵의 높이와 너비를 증가시킬 수 있다는 것을 이미 알고 있습니다.이미지 처리에서 이미지를 확장해야 할 수도 있습니다 (예: *업샘플링*).
*쌍선형 보간*
일반적으로 사용되는 업샘플링 기법 중 하나입니다.또한 전치된 컨벌루션 계층을 초기화하는 데에도 자주 사용됩니다.

쌍선형 보간을 설명하기 위해 입력 영상이 주어지면 업샘플링된 출력 영상의 각 픽셀을 계산하려고 한다고 가정해 보겠습니다.좌표 $(x, y)$에서 출력 이미지의 픽셀을 계산하려면 먼저 $(x, y)$를 입력 이미지에 좌표 $(x', y')$로 매핑합니다. 예를 들어 입력 크기와 출력 크기의 비율에 따라 매핑합니다.매핑된 $x'$ and $y'$는 실수입니다.그런 다음 입력 영상에서 좌표 $(x', y')$에 가장 가까운 4개의 픽셀을 찾습니다.마지막으로, 좌표 $(x, y)$에서 출력 이미지의 픽셀은 입력 이미지 상의 이들 4개의 가장 가까운 픽셀과 $(x', y')$로부터의 상대적 거리에 기초하여 계산된다.

쌍선형 보간의 업샘플링은 다음 `bilinear_kernel` 함수로 구성된 커널을 사용하여 전치된 컨벌루션 계층에 의해 구현될 수 있습니다.공간 제약으로 인해 알고리즘 설계에 대한 논의 없이 아래 `bilinear_kernel` 함수의 구현만 제공합니다.

```{.python .input}
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```{.python .input}
#@tab pytorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

전치된 컨벌루션 계층에 의해 구현되는 [**쌍선형 보간의 업샘플링 실험**] 을 살펴보겠습니다.높이와 무게를 두 배로 늘리는 전치 된 컨벌루션 계층을 구성하고 `bilinear_kernel` 함수로 커널을 초기화합니다.

```{.python .input}
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

```{.python .input}
#@tab pytorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
```

이미지 `X`를 읽고 업샘플링 출력을 `Y`에 할당합니다.이미지를 인쇄하려면 채널 치수의 위치를 조정해야 합니다.

```{.python .input}
img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

```{.python .input}
#@tab pytorch
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
```

보시다시피, 전치된 컨벌루션 계층은 이미지의 높이와 너비를 2배 증가시킵니다.좌표의 다른 배율을 제외하고 쌍선형 보간에 의해 확대된 이미지와 :numref:`sec_bbox`로 인쇄된 원본 이미지는 동일하게 보입니다.

```{.python .input}
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

완전 컨벌루션 네트워크에서는 쌍선형 보간의 업샘플링을 사용하여 전치된 컨벌루션 계층을 초기화합니다.$1\times 1$ 컨벌루션 계층의 경우 자비에르 초기화를 사용합니다.

```{.python .input}
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```{.python .input}
#@tab pytorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```

## [**데이터세트 읽기**]

:numref:`sec_semantic_segmentation`에 소개된 의미론적 세분화 데이터세트를 읽었습니다.임의 자르기의 출력 이미지 모양은 $320\times 480$로 지정됩니다. 높이와 너비를 모두 $32$으로 나눌 수 있습니다.

```{.python .input}
#@tab all
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

## [**교육**]

이제 구성된 완전 컨벌루션 네트워크를 훈련시킬 수 있습니다.여기서 손실 함수와 정확도 계산은 이전 장의 이미지 분류와 본질적으로 다르지 않습니다.전치된 컨벌루션 계층의 출력 채널을 사용하여 각 픽셀의 클래스를 예측하므로 손실 계산에 채널 차원이 지정됩니다.또한 정확도는 모든 픽셀에 대해 예측된 클래스의 정확성을 기반으로 계산됩니다.

```{.python .input}
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## [**예측**]

예측할 때 각 채널의 입력 이미지를 표준화하고 이미지를 CNN에서 요구하는 4 차원 입력 형식으로 변환해야합니다.

```{.python .input}
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input}
#@tab pytorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

각 픽셀의 [**예측 클래스를 시각화**] 하기 위해 예측된 클래스를 데이터셋의 레이블 색상에 다시 매핑합니다.

```{.python .input}
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```{.python .input}
#@tab pytorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```

테스트 데이터셋의 이미지는 크기와 모양이 다양합니다.이 모델은 스트라이드가 32인 전치된 컨벌루션 계층을 사용하므로 입력 영상의 높이 또는 너비를 32로 나눌 수 없는 경우 전치된 컨벌루션 계층의 출력 높이 또는 너비는 입력 영상의 모양에서 벗어납니다.이 문제를 해결하기 위해 이미지에서 높이와 너비가 32의 정수 배수 인 여러 직사각형 영역을 자르고 이러한 영역의 픽셀에 대해 개별적으로 순방향 전파를 수행 할 수 있습니다.이러한 직사각형 영역의 합집합은 입력 이미지를 완전히 덮어야 합니다.픽셀이 여러 직사각형 영역으로 덮여 있는 경우 동일한 픽셀에 대해 별도의 영역에 있는 전치된 컨벌루션 출력의 평균을 softmax 연산에 입력하여 클래스를 예측할 수 있습니다.

간단하게하기 위해 몇 개의 큰 테스트 이미지만 읽고 이미지의 왼쪽 상단에서 시작하여 예측을 위해 $320\times480$ 영역을 자릅니다.이러한 테스트 이미지의 경우 잘린 영역, 예측 결과 및 지상 진실을 행별로 인쇄합니다.

```{.python .input}
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input}
#@tab pytorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

## 요약

* 완전 컨벌루션 네트워크는 먼저 CNN을 사용하여 이미지 특징을 추출한 다음 $1\times 1$ 컨벌루션 계층을 통해 채널 수를 클래스 수로 변환하고 마지막으로 특성 맵의 높이와 너비를 전치된 컨벌루션을 통해 입력 이미지의 높이와 너비를 변환합니다.
* 완전 컨벌루션 네트워크에서는 쌍선형 보간의 업 샘플링을 사용하여 전치 된 컨벌루션 계층을 초기화 할 수 있습니다.

## 연습문제

1. 실험에서 전치 된 컨벌루션 계층에 Xavier 초기화를 사용하면 결과가 어떻게 변경됩니까?
1. 초모수를 조정하여 모델의 정확도를 더 향상시킬 수 있습니까?
1. 테스트 영상에서 모든 픽셀의 클래스를 예측합니다.
1. 원래의 완전 컨벌루션 네트워크 백서는 일부 중간 CNN 계층 :cite:`Long.Shelhamer.Darrell.2015`의 출력도 사용합니다.이 아이디어를 구현하십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/377)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1582)
:end_tab:
