# 인수 분해 기계

2010년 스테펜 렌들이 제안한 인수 분해 기계 (FM) :cite:`Rendle.2010`는 분류, 회귀 및 순위 지정 작업에 사용할 수 있는 감독 알고리즘입니다.빠르게 주목하여 예측 및 권장 사항을 만드는 데 인기 있고 영향력있는 방법이되었습니다.특히 선형 회귀 모델과 행렬 인수 분해 모델의 일반화입니다.또한 다항식 커널이 있는 서포트 벡터 머신을 연상시킵니다.선형 회귀 및 행렬 분해에 비해 인수 분해 기계의 장점은 다음과 같습니다. (1) $\chi$차 변수 교호작용을 모형화할 수 있습니다. 여기서 $\chi$는 다항식 차수의 수이며 일반적으로 2로 설정됩니다. (2) 인수 분해 기계와 관련된 빠른 최적화 알고리즘은다항식 계산 시간을 선형 복잡성으로 변환하므로 특히 고차원 희소 입력값에 매우 효율적입니다.이러한 이유로 인수 분해 기계는 현대 광고 및 제품 권장 사항에 널리 사용됩니다.기술 세부 사항 및 구현은 아래에 설명되어 있습니다. 

## 2-웨이 인수 분해 기계

공식적으로 $x \in \mathbb{R}^d$는 한 표본의 특징 벡터를 나타내고 $y$는 이진 클래스 “클릭/비 클릭”과 같은 실수 값 레이블 또는 클래스 레이블이 될 수있는 해당 레이블을 나타냅니다.차수가 2인 인수 분해 기계의 모델은 다음과 같이 정의됩니다. 

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

여기서 $\mathbf{w}_0 \in \mathbb{R}$는 전역 치우침이고, $\mathbf{w} \in \mathbb{R}^d$은 i번째 변수의 가중치를 나타내고, $\mathbf{V} \in \mathbb{R}^{d\times k}$은 특징 임베딩을 나타내고, $\mathbf{v}_i$는 $\mathbf{V}$의 $i^\mathrm{th}$ 행을 나타내고, $k$은 잠재 요인의 차원성, $\langle\cdot, \cdot \rangle$는 두 벡터의 내적입니다. $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ 상호 작용을 모델링합니다.$i^\mathrm{th}$와 $j^\mathrm{th}$ 기능 사이에 있습니다.일부 기능 상호 작용은 전문가가 설계할 수 있도록 쉽게 이해할 수 있습니다.그러나 대부분의 다른 특징 상호 작용은 데이터에 숨겨져 있으며 식별하기 어렵습니다.따라서 피쳐 상호 작용을 자동으로 모델링하면 피쳐 엔지니어링의 노력을 크게 줄일 수 있습니다.처음 두 항은 선형 회귀 모형에 해당하고 마지막 항은 행렬 인수 분해 모형의 확장입니다.기능 $i$이 항목을 나타내고 기능 $j$가 사용자를 나타내는 경우 세 번째 용어는 사용자와 항목 포함 사이의 내적입니다.FM이 더 높은 주문 (도 > 2) 으로 일반화 할 수도 있다는 점은 주목할 가치가 있습니다.그럼에도 불구하고 수치적 안정성은 일반화를 약화시킬 수 있습니다. 

## 효율적인 최적화 기준

간단한 방법으로 분해 기계를 최적화하면 모든 쌍별 교호작용을 계산해야 하므로 $\mathcal{O}(kd^2)$의 복잡성이 발생합니다.이러한 비효율성 문제를 해결하기 위해 FM의 세 번째 용어를 재구성하여 계산 비용을 크게 줄여 선형 시간 복잡성 ($\mathcal{O}(kd)$) 을 초래할 수 있습니다.데이터 쌍들의 교호작용 항의 재공식화는 다음과 같습니다. 

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

이러한 재구성을 통해 모델 복잡성이 크게 감소합니다.또한 희소 피쳐의 경우 전체 복잡도가 0이 아닌 피쳐 수에 선형이되도록 0이 아닌 요소만 계산해야 합니다. 

FM 모델을 배우기 위해 회귀 작업에 MSE 손실, 분류 작업에 대한 교차 엔트로피 손실, 순위 작업에 BPR 손실을 사용할 수 있습니다.확률적 경사 하강법 및 Adam과 같은 표준 옵티마이저는 최적화에 실행 가능합니다.

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## 모델 구현 다음 코드는 인수 분해 기계를 구현합니다.FM이 선형 회귀 블록과 효율적인 기능 상호 작용 블록으로 구성되어 있음을 알 수 있습니다.CTR 예측을 분류 작업으로 취급하기 때문에 최종 점수에 시그모이드 함수를 적용합니다.

```{.python .input  n=2}
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x
```

## 광고 데이터 세트 로드 마지막 섹션의 CTR 데이터 래퍼를 사용하여 온라인 광고 데이터 세트를 로드합니다.

```{.python .input  n=3}
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## 모델 훈련 후에 모델을 훈련시킵니다.학습률은 0.02로 설정되고 임베딩 크기는 기본적으로 20으로 설정됩니다.`Adam` 옵티마이저와 `SigmoidBinaryCrossEntropyLoss` 손실은 모델 트레이닝에 사용됩니다.

```{.python .input  n=5}
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## 요약

* FM은 회귀, 분류 및 순위와 같은 다양한 작업에 적용 할 수있는 일반적인 프레임 워크입니다.
* 특징 상호 작용/교차는 예측 작업에 중요하며 FM을 사용하여 양방향 상호 작용을 효율적으로 모델링 할 수 있습니다.

## 연습문제

* 아바즈, 무비렌즈, 크리테오 데이터세트와 같은 다른 데이터셋에서 FM을 테스트할 수 있나요?
* 임베딩 크기를 변경하여 성능에 미치는 영향을 확인합니다. 행렬 분해의 패턴과 유사한 패턴을 관찰할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/406)
:end_tab:
