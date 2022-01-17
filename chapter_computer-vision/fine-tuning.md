# 미세 조정
:label:`sec_fine_tuning`

이전 장에서는 60000개의 이미지만으로 Fashion-MNIST 훈련 데이터세트에서 모델을 훈련시키는 방법에 대해 논의했습니다.또한 학계에서 가장 널리 사용되는 대규모 이미지 데이터 세트인 ImageNet에 대해서도 설명했습니다. ImageNet에는 천만 개 이상의 이미지와 1000개의 객체가 있습니다.그러나 일반적으로 접하는 데이터세트의 크기는 두 데이터셋의 크기 사이입니다. 

이미지에서 다양한 유형의 의자를 인식 한 다음 사용자에게 구매 링크를 권장한다고 가정합니다.한 가지 가능한 방법은 먼저 100개의 공통 의자를 식별하고 각 의자에 대해 서로 다른 각도의 1000개의 이미지를 촬영한 다음 수집된 이미지 데이터셋에 대한 분류 모델을 학습시키는 것입니다.이 의자 데이터 세트는 패션-MNIST 데이터 세트보다 클 수 있지만 ImageNet의 예제 수는 여전히 1/10 미만입니다.이로 인해 이 의자 데이터셋에서 ImageNet에 적합한 복잡한 모델이 과적합될 수 있습니다.또한 훈련 예제의 양이 제한되어 있기 때문에 훈련된 모델의 정확도가 실제 요구 사항을 충족하지 못할 수 있습니다. 

위의 문제를 해결하기 위해 확실한 해결책은 더 많은 데이터를 수집하는 것입니다.그러나 데이터를 수집하고 레이블을 지정하는 데는 많은 시간과 비용이 소요될 수 있습니다.예를 들어, ImageNet 데이터 세트를 수집하기 위해 연구원들은 연구 자금으로 수백만 달러를 지출했습니다.현재 데이터 수집 비용이 크게 감소했지만 이 비용은 여전히 무시할 수 없습니다. 

또 다른 해결책은*전이 학습*을 적용하여*소스 데이터 세트*에서 배운 지식을*대상 데이터 세트*로 전송하는 것입니다.예를 들어 ImageNet 데이터 세트의 대부분의 이미지는 의자와 관련이 없지만 이 데이터 세트에서 학습된 모델은 보다 일반적인 이미지 특징을 추출하여 가장자리, 텍스처, 모양 및 객체 구성을 식별하는 데 도움이 될 수 있습니다.이와 유사한 기능은 의자를 인식하는 데에도 효과적일 수 있습니다. 

## 단계

이 섹션에서는 전이 학습: *fine-tuning*. As shown in :numref:`fig_finetune`의 일반적인 기술을 소개합니다. 미세 조정은 다음 네 단계로 구성됩니다. 

1. 소스 데이터셋 (예: ImageNet 데이터셋) 에서 신경망 모델 (예: *소스 모델*) 을 사전 학습합니다.
1. 새로운 신경망 모델, 즉*대상 모델*을 만듭니다.이렇게 하면 출력 레이어를 제외한 모든 모델 설계와 해당 파라미터가 소스 모델에 복사됩니다.이러한 모델 매개 변수에는 소스 데이터 세트에서 배운 지식이 포함되어 있으며이 지식은 대상 데이터 세트에도 적용 할 수 있다고 가정합니다.또한 원본 모델의 출력 레이어가 원본 데이터셋의 레이블과 밀접한 관련이 있다고 가정하므로 대상 모델에서는 사용되지 않습니다.
1. 대상 모델에 출력 레이어를 추가합니다. 대상 모델의 출력 개수는 대상 데이터셋의 범주 수입니다.그런 다음 이 계층의 모델 파라미터를 랜덤하게 초기화합니다.
1. 의자 데이터세트와 같은 대상 데이터세트에서 대상 모델을 훈련시킵니다.출력 레이어는 처음부터 학습되며 다른 모든 레이어의 파라미터는 소스 모델의 파라미터에 따라 미세 조정됩니다.

![Fine tuning.](../img/finetune.svg)
:label:`fig_finetune`

대상 데이터 세트가 소스 데이터 세트보다 훨씬 작은 경우 미세 조정을 통해 모델의 일반화 능력을 향상시킬 수 있습니다. 

## 핫도그 인식

핫도그 인식이라는 구체적인 사례를 통해 미세 조정을 시연해 보겠습니다.ImageNet 데이터 세트에서 사전 학습된 작은 데이터 세트에서 ResNet 모델을 미세 조정합니다.이 작은 데이터셋은 핫도그를 포함하거나 포함하지 않은 수천 개의 이미지로 구성됩니다.미세 조정 모델을 사용하여 이미지에서 핫도그를 인식합니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from torch import nn
import torch
import torchvision
import os
```

### 데이터세트 읽기

[**우리가 사용하는 핫도그 데이터 세트는 온라인 이미지에서 가져온 것입니다**].이 데이터셋은 핫도그가 포함된 1400개의 포지티브 클래스 이미지와 다른 음식이 포함된 네거티브 클래스 이미지로 구성됩니다. 두 클래스의 이미지 1000개가 모두 훈련에 사용되고 나머지는 테스트용으로 사용됩니다. 

다운로드한 데이터세트의 압축을 풀면 `hotdog/train` 및 `hotdog/test`이라는 두 개의 폴더를 얻습니다.두 폴더에는 모두 `hotdog` 및 `not-hotdog` 하위 폴더가 있으며 두 폴더 중 하나에는 해당 클래스의 이미지가 포함되어 있습니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
```

훈련 데이터 세트와 테스트 데이터 세트의 모든 이미지 파일을 각각 읽는 두 개의 인스턴스를 만듭니다.

```{.python .input}
train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))
```

```{.python .input}
#@tab pytorch
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
```

처음 8개의 긍정적인 예와 마지막 8개의 네거티브 이미지가 아래에 나와 있습니다.보시다시피 [**이미지는 크기와 종횡비가 다양합니다**].

```{.python .input}
#@tab all
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
```

훈련 중에 먼저 이미지에서 임의의 크기와 임의의 종횡비의 임의 영역을 자른 다음 이 영역을 $224 \times 224$ 입력 영상으로 스케일링합니다.테스트하는 동안 이미지의 높이와 너비를 256픽셀로 조정한 다음 중앙 $224 \times 224$ 영역을 입력으로 자릅니다.또한 세 개의 RGB (빨강, 녹색 및 파랑) 색상 채널에 대해 채널별로*표준화* 합니다.구체적으로, 채널의 평균값을 해당 채널의 각 값에서 뺀 다음 그 결과를 해당 채널의 표준 편차로 나눕니다. 

[~~데이터 증강~~]

```{.python .input}
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])
```

```{.python .input}
#@tab pytorch
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
```

### [**모델 정의 및 초기화**]

이미지넷 데이터셋에서 사전 학습된 ResNet-18을 소스 모델로 사용합니다.여기서는 사전 훈련된 모델 파라미터를 자동으로 다운로드하도록 `pretrained=True`를 지정합니다.이 모델을 처음 사용하는 경우 다운로드를 위해 인터넷 연결이 필요합니다.

```{.python .input}
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
```

:begin_tab:`mxnet`
사전 훈련된 원본 모델 인스턴스에는 `features`와 `output`라는 두 개의 멤버 변수가 포함되어 있습니다.전자는 출력 레이어를 제외한 모델의 모든 레이어를 포함하고 후자는 모델의 출력 레이어입니다.이 분할의 주요 목적은 출력 레이어를 제외한 모든 레이어의 모델 매개 변수의 미세 조정을 용이하게하는 것입니다.다음은 소스 모델의 멤버 변수 `output`입니다.
:end_tab:

:begin_tab:`pytorch`
사전 훈련된 소스 모델 인스턴스에는 여러 피처 레이어와 출력 레이어 `fc`가 포함되어 있습니다.이 분할의 주요 목적은 출력 레이어를 제외한 모든 레이어의 모델 매개 변수의 미세 조정을 용이하게하는 것입니다.소스 모델의 멤버 변수 `fc`는 다음과 같습니다.
:end_tab:

```{.python .input}
pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net.fc
```

완전 연결 계층으로서 ResNet의 최종 전역 평균 풀링 출력을 ImageNet 데이터 세트의 1000 클래스 출력으로 변환합니다.그런 다음 새 신경망을 대상 모델로 구성합니다.최종 레이어의 출력값 수가 대상 데이터셋의 클래스 수 (1000이 아닌) 로 설정된다는 점을 제외하면 사전 훈련된 소스 모델과 동일한 방식으로 정의됩니다. 

다음 코드에서는 대상 모델 인스턴스 finetune_net의 멤버 변수 피쳐에 있는 모델 매개변수가 원본 모델의 해당 레이어의 모델 매개변수로 초기화됩니다.특징의 모델 파라미터는 ImageNet 데이터 세트에 대해 사전 훈련되고 충분하기 때문에 일반적으로 이러한 파라미터를 미세 조정하는 데 약간의 학습 속도만 필요합니다.  

멤버 변수 출력의 모델 모수는 랜덤하게 초기화되며 일반적으로 처음부터 훈련하려면 더 큰 학습률이 필요합니다.Trainer 인스턴스의 학습률이 η라고 가정하면 반복에서 멤버 변수 출력의 모델 매개 변수의 학습률을 10η로 설정합니다. 

아래 코드에서는 대상 모델 인스턴스 `finetune_net`의 출력 계층 이전의 모델 매개 변수가 원본 모델에서 해당 계층의 모델 매개 변수로 초기화됩니다.이러한 모델 파라미터는 ImageNet에서 사전 학습을 통해 획득되었으므로 효과적입니다.따라서 이러한 사전 훈련된 파라미터를*미세 조정*하기 위해 작은 학습률만 사용할 수 있습니다.반대로 출력 계층의 모델 파라미터는 무작위로 초기화되며 일반적으로 처음부터 학습하려면 더 큰 학습률이 필요합니다.기본 학습률을 $\eta$으로 설정하고 학습률 $10\eta$를 사용하여 출력 계층에서 모델 매개 변수를 반복합니다.

```{.python .input}
finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# The model parameters in the output layer will be iterated using a learning
# rate ten times greater
finetune_net.output.collect_params().setattr('lr_mult', 10)
```

```{.python .input}
#@tab pytorch
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);
```

### [**모델 미세 조정**]

먼저 여러 번 호출할 수 있도록 미세 조정을 사용하는 훈련 함수 `train_fine_tuning`를 정의합니다.

```{.python .input}
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    devices = d2l.try_all_gpus()
    net.collect_params().reset_ctx(devices)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

```{.python .input}
#@tab pytorch
# If `param_group=True`, the model parameters in the output layer will be
# updated using a learning rate ten times greater
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

사전 학습을 통해 얻은 모델 매개 변수를*미세 조정*하기 위해 [**기본 학습률을 작은 값으로 설정**] 합니다.이전 설정을 기반으로 10배 더 큰 학습률을 사용하여 대상 모델의 출력 계층 파라미터를 처음부터 훈련합니다.

```{.python .input}
train_fine_tuning(finetune_net, 0.01)
```

```{.python .input}
#@tab pytorch
train_fine_tuning(finetune_net, 5e-5)
```

[**비교를 위해**] 동일한 모델을 정의하지만 (**모든 모델 매개 변수를 임의의 값으로 초기화**).전체 모델을 처음부터 학습해야 하므로 더 큰 학습률을 사용할 수 있습니다.

```{.python .input}
scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)
```

```{.python .input}
#@tab pytorch
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

보시다시피, 미세 조정된 모델은 초기 파라미터 값이 더 효과적이기 때문에 동일한 시대에 더 나은 성능을 보이는 경향이 있습니다. 

## 요약

* 전이 학습은 원본 데이터세트에서 학습한 지식을 대상 데이터세트로 전송합니다.미세 조정은 전이 학습을 위한 일반적인 기법입니다.
* 대상 모델은 출력 레이어를 제외한 원본 모델의 파라미터와 함께 모든 모델 설계를 복사하고 대상 데이터셋을 기반으로 이러한 파라미터를 미세 조정합니다.반대로 대상 모델의 출력 계층은 처음부터 훈련해야 합니다.
* 일반적으로 파라미터를 미세 조정하면 더 작은 학습률을 사용하는 반면, 출력 계층을 처음부터 훈련시킬 때는 더 큰 학습률을 사용할 수 있습니다.

## 연습문제

1. `finetune_net`의 학습률을 계속 높이십시오.모델의 정확도는 어떻게 변합니까?
2. 비교 실험에서 `finetune_net` 및 `scratch_net`의 초모수를 추가로 조정합니다.여전히 정확도가 다른가요?
3. `finetune_net`의 출력 계층 앞에 있는 파라미터를 소스 모델의 파라미터로 설정하고 훈련 중에는 업데이트하지 마십시오*.모델의 정확도는 어떻게 변합니까?다음 코드를 사용할 수 있습니다.

```{.python .input}
finetune_net.features.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
for param in finetune_net.parameters():
    param.requires_grad = False
```

4. 실제로 `ImageNet` 데이터 세트에는 “핫도그” 클래스가 있습니다.출력 레이어의 해당 가중치 매개 변수는 다음 코드를 통해 얻을 수 있습니다.이 가중치 매개 변수를 어떻게 활용할 수 있을까요?

```{.python .input}
weight = pretrained_net.output.weight
hotdog_w = np.split(weight.data(), 1000, axis=0)[713]
hotdog_w.shape
```

```{.python .input}
#@tab pytorch
weight = pretrained_net.fc.weight
hotdog_w = torch.split(weight.data, 1, dim=0)[934]
hotdog_w.shape
```

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/368)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1439)
:end_tab:
