# AutoRec: 오토인코더를 사용한 등급 예측

행렬 인수 분해 모델은 등급 예측 작업에서 적절한 성능을 달성하지만 본질적으로 선형 모델입니다.따라서 이러한 모델은 사용자의 선호도를 예측할 수 있는 복잡한 비선형적이고 복잡한 관계를 포착할 수 없습니다.이 섹션에서는 비선형 신경망 협업 필터링 모델인 AutoREC :cite:`Sedhain.Menon.Sanner.ea.2015`를 소개합니다.오토인코더 아키텍처로 협업 필터링 (CF) 을 식별하고 명시적 피드백을 기반으로 비선형 변환을 CF에 통합하는 것을 목표로 합니다.신경망은 모든 연속 함수를 근사화할 수 있는 것으로 입증되어 행렬 분해의 한계를 해결하고 행렬 분해의 표현력을 강화하는 데 적합합니다. 

한편, AutoREC는 입력 계층, 은닉 계층 및 재구성 (출력) 계층으로 구성된 오토인코더와 동일한 구조를 가지고 있습니다.오토인코더는 입력값을 은닉 (그리고 일반적으로 저차원) 표현으로 코딩하기 위해 입력값을 출력으로 복사하는 법을 학습하는 신경망입니다.AutoREC에서는 사용자/항목을 저차원 공간에 명시적으로 포함시키는 대신 상호 작용 행렬의 열/행을 입력으로 사용한 다음 출력 계층에서 상호 작용 행렬을 재구성합니다. 

반면 AutoRec은 기존 오토인코더와 다릅니다. AutoREC는 숨겨진 표현을 학습하는 대신 출력 계층을 학습/재구성하는 데 중점을 둡니다.부분적으로 관측된 교호작용 행렬을 입력으로 사용하여 완성된 등급 행렬을 재구성합니다.그 동안 입력의 누락된 항목은 추천을 목적으로 재구성을 통해 출력 계층에 채워집니다.  

AutoRec에는 사용자 기반과 항목 기반의 두 가지 변형이 있습니다.간단히 말씀드리자면 여기서는 항목 기반 AutoRec만 소개합니다.이에 따라 사용자 기반 AutoRec을 파생할 수 있습니다. 

## 모델

$\mathbf{R}_{*i}$는 등급 행렬의 $i^\mathrm{th}$ 열을 나타냅니다. 여기서 알 수 없는 등급은 기본적으로 0으로 설정됩니다.신경 아키텍처는 다음과 같이 정의됩니다. 

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

여기서 $f(\cdot)$ 및 $g(\cdot)$는 활성화 함수를 나타내고, $\mathbf{W}$ 및 $\mathbf{V}$은 가중치 행렬이고, $\mu$ 및 $b$은 편향입니다.$h( \cdot )$는 자동 녹의 전체 네트워크를 나타냅니다.출력값 $h(\mathbf{R}_{*i})$은 등급 행렬의 $i^\mathrm{th}$ 열을 재구성한 것입니다. 

다음 목적 함수는 재구성 오류를 최소화하는 것을 목표로 합니다. 

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

여기서 $\| \cdot \|_{\mathcal{O}}$는 관측된 등급의 기여도만 고려됨을 의미합니다. 즉, 관측된 입력과 관련된 가중치만 역 전파 중에 업데이트됩니다.

```{.python .input  n=3}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## 모델 구현

일반적인 오토인코더는 인코더와 디코더로 구성됩니다.인코더는 입력을 은닉 표현으로 투영하고 디코더는 숨겨진 레이어를 재구성 계층에 매핑합니다.이 실습에 따라 밀도가 높은 계층으로 인코더와 디코더를 만듭니다.인코더의 활성화는 기본적으로 `sigmoid`로 설정되며 디코더에는 활성화가 적용되지 않습니다.오버피팅을 줄이기 위해 인코딩 변환 후 드롭아웃이 포함됩니다.관찰되지 않은 입력값의 기울기는 관찰된 등급만 모델 학습 과정에 기여하도록 마스킹됩니다.

```{.python .input  n=2}
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## 평가자 다시 구현

입력과 출력이 변경되었으므로 RMSE를 정확도 측정으로 사용하는 동안 평가 함수를 다시 구현해야 합니다.

```{.python .input  n=3}
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## 모델 훈련 및 평가

이제 MovieLens 데이터 세트에서 AutoRec을 훈련시키고 평가해 보겠습니다.테스트 RMSE가 행렬 인수 분해 모델보다 낮아 등급 예측 작업에서 신경망의 효과를 확인할 수 있음을 분명히 알 수 있습니다.

```{.python .input  n=4}
devices = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

## 요약

* 비선형 계층과 드롭아웃 정규화를 통합하면서 행렬 인수 분해 알고리즘을 오토인코더로 구성할 수 있습니다. 
* MovieLens 100K 데이터세트에 대한 실험은 AutoRec이 행렬 분해보다 우수한 성능을 달성한다는 것을 보여줍니다.

## 연습문제

* AutoRec의 숨겨진 치수를 변경하여 모델 성능에 미치는 영향을 확인합니다.
* 숨겨진 레이어를 더 추가해 봅니다.모델 성능을 개선하는 것이 도움이 됩니까?
* 디코더와 인코더 활성화 기능의 더 나은 조합을 찾을 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/401)
:end_tab:
