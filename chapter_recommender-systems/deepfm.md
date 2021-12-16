# 심층 인수 분해 기계

효과적인 기능 조합을 학습하는 것은 클릭률 예측 작업의 성공에 매우 중요합니다.인수 분해 기계는 선형 패러다임으로 특징 상호 작용 (예: 쌍선형 교호작용) 을 모델링합니다.고유한 형상 교차 구조가 일반적으로 매우 복잡하고 비선형인 실제 데이터에는 충분하지 않은 경우가 많습니다.더 나쁜 것은 실제로 인수 분해 기계에서 일반적으로 2차 특징 교호작용이 사용된다는 것입니다.이론적으로는 인수 분해 기계를 사용하여 더 높은 수준의 특징 조합을 모델링할 수 있지만 수치적 불안정성과 높은 계산 복잡성으로 인해 일반적으로 채택되지 않습니다. 

효과적인 솔루션 중 하나는 심층 신경망을 사용하는 것입니다.심층 신경망은 특징 표현 학습에서 강력하며 정교한 특징 상호 작용을 학습할 수 있는 잠재력을 가지고 있습니다.따라서 심층 신경망을 인수 분해 기계에 통합하는 것은 당연합니다.인수 분해 기계에 비선형 변환 계층을 추가하면 저차 특징 조합과 고차 특징 조합을 모두 모형화할 수 있습니다.또한 입력의 비선형 고유 구조도 심층 신경망으로 캡처할 수 있습니다.이 섹션에서는 FM과 심층 신경망을 결합한 심층 인수 분해 기계 (DeepFM) :cite:`Guo.Tang.Ye.ea.2017`라는 대표 모델을 소개합니다. 

## 모델 아키텍처

DeepFM은 FM 구성 요소와 병렬 구조로 통합 된 심층 구성 요소로 구성됩니다.FM 성분은 저차 특징 교호작용을 모형화하는 데 사용되는 2원 분해 기계와 동일합니다.심층 성분은 고차 특징 교호작용과 비선형성을 캡처하는 데 사용되는 MLP입니다.이 두 구성 요소는 동일한 입력/임베딩을 공유하며 해당 출력은 최종 예측으로 요약됩니다.DeepFM의 정신은 암기와 일반화를 모두 포착 할 수있는 Wide\ & Deep 아키텍처의 정신과 유사하다는 점을 지적 할 가치가 있습니다.Wide\ & Deep 모델에 비해 DeepFM의 장점은 기능 조합을 자동으로 식별하여 수작업으로 제작 된 기능 엔지니어링의 노력을 줄인다는 것입니다. 

간결성을 위해 FM 구성 요소에 대한 설명을 생략하고 출력을 $\hat{y}^{(FM)}$으로 표시합니다.자세한 내용은 마지막 섹션을 참조하십시오.$\mathbf{e}_i \in \mathbb{R}^{k}$가 $i^\mathrm{th}$ 필드의 잠재 특징 벡터를 나타낸다고 합시다.deep 컴포넌트의 입력은 다음과 같이 표시되는 희소 범주형 특징 입력으로 조회되는 모든 필드의 덴스 임베딩을 결합한 것입니다. 

$$
\mathbf{z}^{(0)}  = [\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_f],
$$

여기서 $f$는 필드의 수입니다.그런 다음 다음 신경망으로 공급됩니다. 

$$
\mathbf{z}^{(l)}  = \alpha(\mathbf{W}^{(l)}\mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}),
$$

여기서 $\alpha$은 활성화 함수입니다. $\mathbf{W}_{l}$ 및 $\mathbf{b}_{l}$은 $l^\mathrm{th}$ 계층의 가중치와 치우침입니다.$y_{DNN}$이 예측의 출력을 나타낸다고 합니다.DeepFM의 궁극적 인 예측은 FM과 DNN의 출력을 합한 것입니다.그래서 우리는: 

$$
\hat{y} = \sigma(\hat{y}^{(FM)} + \hat{y}^{(DNN)}),
$$

여기서 $\sigma$는 시그모이드 함수입니다.DeepFM의 아키텍처는 아래에 설명되어 있습니다.![Illustration of the DeepFM model](../img/rec-deepfm.svg) 

DeepFM이 심층 신경망과 FM을 결합하는 유일한 방법은 아니라는 점은 주목할 가치가 있습니다.피처 상호 작용 :cite:`He.Chua.2017`에 비선형 계층을 추가할 수도 있습니다.

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## 딥FM 구현 딥FM의 구현은 FM의 구현과 유사합니다.FM 부분을 변경하지 않고 활성화 기능으로 `relu`가있는 MLP 블록을 사용합니다.드롭아웃은 모델을 정규화하는 데에도 사용됩니다.MLP의 뉴런 수는 `mlp_dims` 하이퍼파라미터로 조정할 수 있습니다.

```{.python .input  n=2}
class DeepFM(nn.Block):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)
        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
            self.mlp.add(nn.Dropout(rate=drop_rate))
            input_dim = dim
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))

    def forward(self, x):
        embed_x = self.embedding(x)
        square_of_sum = np.sum(embed_x, axis=1) ** 2
        sum_of_square = np.sum(embed_x ** 2, axis=1)
        inputs = np.reshape(embed_x, (-1, self.embed_output_dim))
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) \
            + self.mlp(inputs)
        x = npx.sigmoid(x)
        return x
```

## 모델 훈련 및 평가 데이터 로딩 프로세스는 FM의 프로세스와 동일합니다.DeepFM의 MLP 구성 요소를 피라미드 구조 (30-20-10) 의 3 계층 고밀도 네트워크로 설정했습니다.다른 모든 하이퍼파라미터는 FM과 동일하게 유지됩니다.

```{.python .input  n=4}
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
field_dims = train_data.field_dims
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
devices = d2l.try_all_gpus()
net = DeepFM(field_dims, num_factors=10, mlp_dims=[30, 20, 10])
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.01, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

FM과 비교하여 DeepFM은 더 빠르게 수렴하고 더 나은 성능을 달성합니다. 

## 요약

* 신경망을 FM에 통합하면 복잡하고 고차 상호 작용을 모델링할 수 있습니다.
* DeepFM은 광고 데이터 세트에서 원래 FM보다 성능이 뛰어납니다.

## 연습문제

* MLP의 구조를 변경하여 모델 성능에 미치는 영향을 확인합니다.
* 데이터세트를 Criteo로 변경하고 원래 FM 모델과 비교합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/407)
:end_tab:
