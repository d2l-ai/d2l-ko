# 뉴럴 스타일 전달

사진 애호가라면 필터에 익숙할 것입니다.풍경 사진이 더 선명 해지거나 인물 사진의 스킨이 희게 보이도록 사진의 색상 스타일을 변경할 수 있습니다.그러나 일반적으로 한 필터는 사진의 한 측면만 변경합니다.사진에 이상적인 스타일을 적용하려면 다양한 필터 조합을 시도해야 할 것입니다.이 프로세스는 모델의 하이퍼파라미터를 조정하는 것만큼 복잡합니다. 

이 섹션에서는 CNN의 레이어별 표현을 활용하여 한 이미지의 스타일을 다른 이미지에 자동으로 적용합니다 (예: *스타일 전송* :cite:`Gatys.Ecker.Bethge.2016`).이 작업에는 두 개의 입력 이미지가 필요합니다. 하나는*콘텐츠 이미지*이고 다른 하나는*스타일 이미지*입니다.신경망을 사용하여 콘텐츠 이미지를 수정하여 스타일 이미지에 가깝게 만듭니다.예를 들어 :numref:`fig_style_transfer`의 콘텐츠 이미지는 시애틀 교외의 레이니어 산 국립 공원에서 찍은 풍경 사진이고 스타일 이미지는 가을 떡갈 나무를 테마로 한 유화입니다.합성된 출력 이미지에서는 스타일 이미지의 오일 브러시 획이 적용되어 내용 이미지에 있는 개체의 기본 모양을 유지하면서 보다 선명한 색상을 얻을 수 있습니다. 

![Given content and style images, style transfer outputs a synthesized image.](../img/style-transfer.svg)
:label:`fig_style_transfer`

## 메소드

:numref:`fig_style_transfer_model`는 단순화된 예제와 함께 CNN 기반 스타일 전송 방법을 보여줍니다.먼저 합성된 이미지 (예: 콘텐츠 이미지) 를 초기화합니다.이 합성 이미지는 스타일 전송 프로세스 중에 업데이트해야 하는 유일한 변수입니다. 즉, 훈련 중에 업데이트할 모델 매개 변수입니다.그런 다음 사전 훈련된 CNN을 선택하여 이미지 특징을 추출하고 훈련 중에 모델 파라미터를 고정합니다.이 심층 CNN은 여러 계층을 사용하여 이미지의 계층적 특징을 추출합니다.이러한 레이어 중 일부의 출력을 콘텐츠 기능 또는 스타일 기능으로 선택할 수 있습니다.:numref:`fig_style_transfer_model`를 예로 들어 보겠습니다.여기서 사전 훈련된 신경망에는 3개의 컨벌루션 계층이 있으며, 두 번째 계층은 콘텐츠 특징을 출력하고, 첫 번째 계층과 세 번째 계층은 스타일 특징을 출력합니다. 

![CNN-based style transfer process. Solid lines show the direction of forward propagation and dotted lines show backward propagation. ](../img/neural-style.svg)
:label:`fig_style_transfer_model`

다음으로 정방향 전파 (실선 화살표 방향) 를 통해 스타일 전달의 손실 함수를 계산하고 역 전파 (파선 화살표 방향) 를 통해 모델 매개 변수 (출력용 합성 이미지) 를 업데이트합니다.스타일 전송에 일반적으로 사용되는 손실 함수는 세 부분으로 구성됩니다. (i) * 콘텐츠 손실*은 합성 된 이미지와 콘텐츠 이미지를 콘텐츠 기능에 가깝게 만듭니다. (ii) * 스타일 손실*은 합성 된 이미지와 스타일 이미지를 스타일 기능에 가깝게 만듭니다.합성된 이미지의 노이즈.마지막으로 모델 학습이 끝나면 스타일 전달의 모델 매개 변수를 출력하여 최종 합성 이미지를 생성합니다. 

다음에서는 구체적인 실험을 통해 스타일 전달의 기술적 세부 사항을 설명합니다. 

## [**콘텐츠 및 스타일 이미지 읽기**]

먼저 콘텐츠와 스타일 이미지를 읽습니다.인쇄된 좌표축에서 이러한 이미지의 크기가 다르다는 것을 알 수 있습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

d2l.set_figsize()
content_img = image.imread('../img/rainier.jpg')
d2l.plt.imshow(content_img.asnumpy());
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn

d2l.set_figsize()
content_img = d2l.Image.open('../img/rainier.jpg')
d2l.plt.imshow(content_img);
```

```{.python .input}
style_img = image.imread('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img.asnumpy());
```

```{.python .input}
#@tab pytorch
style_img = d2l.Image.open('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img);
```

## [**전처리 및 후처리**]

아래에서는 이미지 전처리 및 후처리를 위한 두 가지 함수를 정의합니다.`preprocess` 함수는 입력 영상의 3개의 RGB 채널 각각을 표준화하고 결과를 CNN 입력 형식으로 변환합니다.`postprocess` 함수는 표준화 전에 출력 이미지의 픽셀 값을 원래 값으로 복원합니다.이미지 인쇄 기능에서는 각 픽셀에 0에서 1까지의 부동 소수점 값이 있어야 하므로 0보다 작거나 1보다 큰 값은 각각 0 또는 1로 바꿉니다.

```{.python .input}
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)

def postprocess(img):
    img = img[0].as_in_ctx(rgb_std.ctx)
    return (img.transpose(1, 2, 0) * rgb_std + rgb_mean).clip(0, 1)
```

```{.python .input}
#@tab pytorch
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
```

## [**기능 추출**]

이미지넷 데이터셋에서 사전 훈련된 VGG-19 모델을 사용하여 이미지 특징 :cite:`Gatys.Ecker.Bethge.2016`를 추출합니다.

```{.python .input}
pretrained_net = gluon.model_zoo.vision.vgg19(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.vgg19(pretrained=True)
```

이미지의 콘텐츠 특징과 스타일 특징을 추출하기 위해 VGG 네트워크에서 특정 레이어의 출력을 선택할 수 있습니다.일반적으로 입력 레이어에 가까울수록 이미지의 세부 정보를 추출하기 쉽고 그 반대의 경우도 이미지의 전역 정보를 더 쉽게 추출할 수 있습니다.합성 된 이미지에서 콘텐츠 이미지의 세부 사항이 과도하게 유지되는 것을 방지하기 위해 출력에 더 가까운 VGG 레이어를*콘텐츠 레이어*로 선택하여 이미지의 콘텐츠 기능을 출력합니다.또한 로컬 및 전역 스타일 피처를 추출하기 위해 다양한 VGG 레이어의 출력을 선택합니다.이러한 레이어는*스타일 레이어*라고도 합니다.:numref:`sec_vgg`에서 언급했듯이 VGG 네트워크는 5개의 컨벌루션 블록을 사용합니다.실험에서는 네 번째 컨벌루션 블록의 마지막 컨벌루션 계층을 콘텐츠 계층으로 선택하고 각 컨벌루션 블록의 첫 번째 컨벌루션 계층을 스타일 계층으로 선택합니다.이러한 레이어의 인덱스는 `pretrained_net` 인스턴스를 인쇄하여 얻을 수 있습니다.

```{.python .input}
#@tab all
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
```

VGG 레이어를 사용하여 피처를 추출하는 경우 입력 레이어에서 출력 레이어와 가장 가까운 콘텐츠 레이어 또는 스타일 레이어까지 모든 피처를 사용하면 됩니다.특징 추출에 사용할 모든 VGG 계층만 유지하는 새 네트워크 인스턴스 `net`를 구성해 보겠습니다.

```{.python .input}
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])
```

입력 `X`가 주어지면 단순히 순방향 전파 `net(X)`를 호출하면 마지막 계층의 출력만 얻을 수 있습니다.중간 레이어의 출력값도 필요하므로 레이어별 계산을 수행하고 콘텐츠 및 스타일 레이어 출력을 유지해야 합니다.

```{.python .input}
#@tab all
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles
```

아래에 두 가지 함수가 정의되어 있습니다. `get_contents` 함수는 콘텐츠 이미지에서 콘텐츠 특징을 추출하고 `get_styles` 함수는 스타일 이미지에서 스타일 특징을 추출합니다.훈련 중에 사전 훈련된 VGG의 모델 파라미터를 업데이트할 필요가 없으므로 훈련을 시작하기 전에도 콘텐츠와 스타일 특징을 추출할 수 있습니다.합성 이미지는 스타일 전송을 위해 업데이트 할 모델 매개 변수 집합이므로 훈련 중에 `extract_features` 함수를 호출하여 합성 이미지의 내용과 스타일 특성만 추출 할 수 있습니다.

```{.python .input}
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).copyto(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).copyto(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

```{.python .input}
#@tab pytorch
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

## [**손실 함수 정의**]

이제 스타일 전송의 손실 함수에 대해 설명하겠습니다.손실 함수는 콘텐츠 손실, 스타일 손실 및 총 변형 손실로 구성됩니다. 

### 콘텐츠 손실

선형 회귀의 손실 함수와 마찬가지로 콘텐츠 손실은 제곱 손실 함수를 통해 합성 이미지와 콘텐츠 이미지 간의 콘텐츠 특징 차이를 측정합니다.제곱 손실 함수의 두 입력값은 모두 `extract_features` 함수로 계산된 콘텐츠 계층의 출력값입니다.

```{.python .input}
def content_loss(Y_hat, Y):
    return np.square(Y_hat - Y).mean()
```

```{.python .input}
#@tab pytorch
def content_loss(Y_hat, Y):
    # We detach the target content from the tree used to dynamically compute
    # the gradient: this is a stated value, not a variable. Otherwise the loss
    # will throw an error.
    return torch.square(Y_hat - Y.detach()).mean()
```

### 스타일 손실

콘텐츠 손실과 유사한 스타일 손실도 제곱 손실 함수를 사용하여 합성된 이미지와 스타일 이미지 간의 스타일 차이를 측정합니다.스타일 레이어의 스타일 출력을 표현하기 위해 먼저 `extract_features` 함수를 사용하여 스타일 레이어 출력을 계산합니다.출력에 1개의 예제, $c$ 채널, 높이 $h$ 및 너비 $w$가 있다고 가정하면 이 출력을 $c$ 행과 $hw$ 열이 있는 행렬 $\mathbf{X}$로 변환할 수 있습니다.이 행렬은 각각 길이가 $hw$인 $c$ 벡터 $\mathbf{x}_1, \ldots, \mathbf{x}_c$의 연결로 생각할 수 있습니다.여기서 벡터 $\mathbf{x}_i$은 채널 $i$의 스타일 기능을 나타냅니다. 

이러한 벡터 $\mathbf{X}\mathbf{X}^\top \in \mathbb{R}^{c \times c}$의*그램 행렬*에서 행 $i$에 있는 요소 $x_{ij}$와 열 $j$은 벡터 $\mathbf{x}_i$ 및 $\mathbf{x}_j$의 내적입니다.채널 $i$ 및 $j$의 스타일 기능의 상관 관계를 나타냅니다.이 그램 행렬을 사용하여 모든 스타일 레이어의 스타일 출력을 나타냅니다.$hw$의 값이 더 크면 그람 행렬의 값이 더 커질 수 있습니다.또한 그램 행렬의 높이와 너비는 모두 채널 수 $c$입니다.스타일 손실이 이러한 값의 영향을 받지 않도록 하기 위해 아래의 `gram` 함수는 그램 행렬을 요소의 수, 즉 $chw$로 나눕니다.

```{.python .input}
#@tab all
def gram(X):
    num_channels, n = X.shape[1], d2l.size(X) // X.shape[1]
    X = d2l.reshape(X, (num_channels, n))
    return d2l.matmul(X, X.T) / (num_channels * n)
```

스타일 손실에 대한 제곱 손실 함수의 두 그람 행렬 입력값은 합성된 이미지와 스타일 이미지에 대한 스타일 레이어 출력을 기반으로 합니다.여기서는 스타일 이미지를 기반으로 한 그램 행렬 `gram_Y`가 미리 계산되었다고 가정합니다.

```{.python .input}
def style_loss(Y_hat, gram_Y):
    return np.square(gram(Y_hat) - gram_Y).mean()
```

```{.python .input}
#@tab pytorch
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()
```

### 총 변동 손실

때로는 학습된 합성 이미지에 고주파 노이즈, 즉 특히 밝거나 어두운 픽셀이 많이 있습니다.일반적인 소음 감소 방법 중 하나는
*총 편차 노이즈 제거*.
좌표 $(i, j)$의 픽셀 값을 $x_{i, j}$로 나타냅니다.총 변동 손실 감소 

$$\sum_{i, j} \left|x_{i, j} - x_{i+1, j}\right| + \left|x_{i, j} - x_{i, j+1}\right|$$

합성된 이미지의 인접 픽셀 값을 더 가깝게 만듭니다.

```{.python .input}
#@tab all
def tv_loss(Y_hat):
    return 0.5 * (d2l.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  d2l.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
```

### 손실 함수

[**스타일 전송의 손실 함수는 콘텐츠 손실, 스타일 손실 및 총 변형 손실의 가중 합계입니다**].이러한 가중치 하이퍼파라미터를 조정하여 합성된 이미지의 콘텐츠 유지, 스타일 전송 및 노이즈 감소 간에 균형을 맞출 수 있습니다.

```{.python .input}
#@tab all
content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # Calculate the content, style, and total variance losses respectively
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # Add up all the losses
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l
```

## [**합성 이미지 초기화**]

스타일 전송에서 합성 이미지는 훈련 중에 업데이트해야 하는 유일한 변수입니다.따라서 간단한 모델 `SynthesizedImage`를 정의하고 합성 이미지를 모델 매개 변수로 처리 할 수 있습니다.이 모델에서 순방향 전파는 모델 매개 변수만 반환합니다.

```{.python .input}
class SynthesizedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()
```

```{.python .input}
#@tab pytorch
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
```

다음으로 `get_inits` 함수를 정의합니다.이 함수는 합성된 이미지 모델 인스턴스를 만들고 이미지 `X`로 초기화합니다.다양한 스타일 계층 `styles_Y_gram`의 스타일 이미지에 대한 그램 행렬은 훈련 전에 계산됩니다.

```{.python .input}
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=device, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

```{.python .input}
#@tab pytorch
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

## [**교육**]

스타일 전송을 위해 모델을 훈련 할 때 합성 이미지의 콘텐츠 특징과 스타일 특징을 지속적으로 추출하고 손실 함수를 계산합니다.아래는 훈련 루프를 정의합니다.

```{.python .input}
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs], ylim=[0, 20],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        if (epoch + 1) % lr_decay_epoch == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.8)
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X).asnumpy())
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

```{.python .input}
#@tab pytorch
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

이제 [**모델 훈련을 시작합니다**].콘텐츠 및 스타일 이미지의 높이와 너비를 300x450픽셀로 다시 조정합니다.콘텐츠 이미지를 사용하여 합성 이미지를 초기화합니다.

```{.python .input}
device, image_shape = d2l.try_gpu(), (450, 300)
net.collect_params().reset_ctx(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.9, 500, 50)
```

```{.python .input}
#@tab pytorch
device, image_shape = d2l.try_gpu(), (300, 450)  # PIL Image (h, w)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
```

합성 된 이미지는 콘텐츠 이미지의 풍경과 대상을 유지하고 동시에 스타일 이미지의 색상을 전송하는 것을 볼 수 있습니다.예를 들어 합성된 이미지에는 스타일 이미지와 같은 색상 블록이 있습니다.이러한 블록 중 일부는 브러시 획의 미묘한 질감을 가지고 있습니다. 

## 요약

* 스타일 전송에 일반적으로 사용되는 손실 함수는 세 부분으로 구성됩니다. (i) 콘텐츠 손실은 합성 이미지와 콘텐츠 이미지를 콘텐츠 기능에 가깝게 만듭니다. (ii) 스타일 손실은 합성 된 이미지와 스타일 이미지를 스타일 기능에 가깝게 만듭니다.합성된 이미지입니다.
* 사전 훈련 된 CNN을 사용하여 이미지 특징을 추출하고 손실 함수를 최소화하여 훈련 중에 합성 이미지를 모델 매개 변수로 지속적으로 업데이트 할 수 있습니다.
* 그램 행렬을 사용하여 스타일 레이어의 스타일 출력을 나타냅니다.

## 연습문제

1. 다른 콘텐츠 및 스타일 레이어를 선택하면 출력이 어떻게 변경됩니까?
1. 손실 함수에서 가중치 하이퍼파라미터를 조정합니다.출력에 더 많은 내용이 유지되거나 노이즈가 적습니까?
1. 다양한 콘텐츠 및 스타일 이미지를 사용합니다.더 흥미로운 합성 이미지를 만들 수 있습니까?
1. 텍스트에 스타일 전송을 적용할 수 있나요?힌트: you may refer to the survey paper by Hu et al. :cite:`Hu.Lee.Aggarwal.ea.2020`.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/378)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1476)
:end_tab:
