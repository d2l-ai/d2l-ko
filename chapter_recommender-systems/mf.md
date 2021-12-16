# 행렬 인수 분해

행렬 인수 분해 :cite:`Koren.Bell.Volinsky.2009`는 권장 시스템 문헌에서 잘 정립 된 알고리즘입니다.행렬 인수 분해 모델의 첫 번째 버전은 Simon Funk가 유명한 [블로그 게시물](https://sifter.org/~simon/journal/20061211.html) 에서 제안하여 상호 작용 행렬을 분해하는 아이디어를 설명했습니다.그 후 2006년에 개최된 넷플릭스 콘테스트로 인해 널리 알려지게 되었습니다.당시 미디어 스트리밍 및 비디오 대여 회사 인 Netflix는 추천자 시스템 성능을 개선하기위한 콘테스트를 발표했습니다.넷플릭스 기준 (예: 시네매치) 을 10% 개선할 수 있는 최고의 팀은 백만 달러의 상금을 받을 것입니다.따라서 이번 콘테스트는 추천자 시스템 연구 분야에 많은 관심을 끌었습니다.그 후 BellKor, 실용 이론 및 BigChaos의 결합 된 팀 인 BellKor의 실용 카오스 팀이 최우수상을 수상했습니다 (지금은 이러한 알고리즘에 대해 걱정할 필요가 없습니다).최종 점수는 앙상블 해 (즉, 여러 알고리즘의 조합) 의 결과였지만 행렬 인수 분해 알고리즘은 최종 혼합에서 중요한 역할을 했습니다.넷플릭스 대상 솔루션 :cite:`Toscher.Jahrer.Bell.2009`의 기술 보고서는 채택된 모델에 대한 자세한 소개를 제공합니다.이 섹션에서는 행렬 인수 분해 모델과 그 구현에 대해 자세히 설명합니다. 

## 행렬 인수 분해 모델

행렬 분해는 협업 필터링 모델의 한 종류입니다.구체적으로, 모델은 사용자-항목 상호작용 행렬 (예를 들어, 등급 행렬) 을 2개의 하위 랭크 행렬의 곱으로 분해하여, 사용자-항목 상호작용의 낮은 랭크 구조를 캡처한다. 

$\mathbf{R} \in \mathbb{R}^{m \times n}$은 $m$명의 사용자와 $n$개의 항목이 있는 교호작용 행렬을 나타내고 $\mathbf{R}$의 값은 명시적 등급을 나타냅니다.사용자-아이템 상호작용은 사용자 잠재 행렬 $\mathbf{P} \in \mathbb{R}^{m \times k}$ 및 항목 잠재 행렬 $\mathbf{Q} \in \mathbb{R}^{n \times k}$로 분해될 것이다. 여기서 $k \ll m, n$은 잠재 인자 크기이다.$\mathbf{p}_u$은 $\mathbf{P}$의 $u^\mathrm{th}$ 행을 나타내고 $\mathbf{q}_i$은 $\mathbf{Q}$의 $i^\mathrm{th}$ 행을 나타냅니다.주어진 항목 $i$에 대해 $\mathbf{q}_i$의 요소는 해당 항목이 영화의 장르 및 언어와 같은 특성을 소유하는 정도를 측정합니다.지정된 사용자 $u$에 대해 $\mathbf{p}_u$의 요소는 항목의 해당 특성에 대한 사용자의 관심 범위를 측정합니다.이러한 잠재 요인은 이러한 예에서 언급한 대로 명백한 차원을 측정하거나 완전히 해석할 수 없습니다.예상 등급은 다음과 같이 추정할 수 있습니다. 

$$\hat{\mathbf{R}} = \mathbf{PQ}^\top$$

여기서 $\hat{\mathbf{R}}\in \mathbb{R}^{m \times n}$은 $\mathbf{R}$과 모양이 같은 예측된 등급 행렬입니다.이 예측 규칙의 주요 문제점 중 하나는 사용자/항목 편향을 모델링할 수 없다는 것입니다.예를 들어, 일부 사용자는 더 높은 평점을 주는 경향이 있거나 일부 항목은 품질이 좋지 않아 항상 낮은 평점을 받습니다.이러한 편향은 실제 애플리케이션에서 흔히 볼 수 있습니다.이러한 편향을 포착하기 위해 사용자별 및 항목별 편향 용어가 도입되었습니다.구체적으로, 사용자 $u$가 항목 $i$에 부여하는 예측 등급은 다음과 같이 계산됩니다. 

$$
\hat{\mathbf{R}}_{ui} = \mathbf{p}_u\mathbf{q}^\top_i + b_u + b_i
$$

그런 다음 예측 등급 점수와 실제 등급 점수 사이의 평균 제곱 오차를 최소화하여 행렬 인수 분해 모델을 훈련시킵니다.목적 함수는 다음과 같이 정의됩니다. 

$$
\underset{\mathbf{P}, \mathbf{Q}, b}{\mathrm{argmin}} \sum_{(u, i) \in \mathcal{K}} \| \mathbf{R}_{ui} -
\hat{\mathbf{R}}_{ui} \|^2 + \lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )
$$

여기서 $\lambda$는 정규화 비율을 나타냅니다.정규화 용어 $\ 람다 (\ |\ mathbf {P}\ |^2_F +\ |\ mathbf {Q}\ |^2_F + b_u^2 + b_i^2) $ is used to avoid over-fitting by penalizing the magnitude of the parameters. The $ (u, i) $ pairs for which $\ mathbf {R} _ {ui} $는 $\mathcal{K}=\{(u, i) \mid \mathbf{R}_{ui} \text{ is known}\}$ 세트에 저장되어 있습니다.모델 매개변수는 확률적 경사하강법 및 Adam과 같은 최적화 알고리즘을 사용하여 학습할 수 있습니다. 

행렬 인수 분해 모델의 직관적인 그림이 아래에 나와 있습니다. 

![Illustration of matrix factorization model](../img/rec-mf.svg)

이 섹션의 나머지 부분에서는 행렬 인수 분해의 구현에 대해 설명하고 MovieLens 데이터 세트에서 모델을 학습시킵니다.

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
npx.set_np()
```

## 모델 구현

먼저 위에서 설명한 행렬 인수 분해 모델을 구현합니다.`nn.Embedding`를 사용하여 사용자 및 항목 잠재 요인을 생성할 수 있습니다.`input_dim`은 항목/사용자의 수이고 (`output_dim`) 은 잠재 요인 ($k$) 의 차원입니다.`nn.Embedding`를 사용하여 `output_dim`를 1로 설정하여 사용자/항목 편향을 만들 수도 있습니다.`forward` 함수에서 사용자 및 항목 ID는 임베딩을 조회하는 데 사용됩니다.

```{.python .input  n=4}
class MF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(input_dim=num_users, output_dim=num_factors)
        self.Q = nn.Embedding(input_dim=num_items, output_dim=num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
        return outputs.flatten()
```

## 평가 조치

그런 다음 모델에서 예측한 등급 점수와 실제로 관찰된 등급 (실측) :cite:`Gunawardana.Shani.2015` 간의 차이를 측정하는 데 일반적으로 사용되는 RMSE (제곱평균제곱근 오차) 측정을 구현합니다.RMSE는 다음과 같이 정의됩니다. 

$$
\mathrm{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u, i) \in \mathcal{T}}(\mathbf{R}_{ui} -\hat{\mathbf{R}}_{ui})^2}
$$

여기서 $\mathcal{T}$는 평가할 사용자 및 항목 쌍으로 구성된 집합입니다. $|\mathcal{T}|$은 이 집합의 크기입니다.`mx.metric`에서 제공하는 RMSE 함수를 사용할 수 있습니다.

```{.python .input  n=3}
def evaluator(net, test_iter, devices):
    rmse = mx.metric.RMSE()  # Get the RMSE
    rmse_list = []
    for idx, (users, items, ratings) in enumerate(test_iter):
        u = gluon.utils.split_and_load(users, devices, even_split=False)
        i = gluon.utils.split_and_load(items, devices, even_split=False)
        r_ui = gluon.utils.split_and_load(ratings, devices, even_split=False)
        r_hat = [net(u, i) for u, i in zip(u, i)]
        rmse.update(labels=r_ui, preds=r_hat)
        rmse_list.append(rmse.get()[1])
    return float(np.mean(np.array(rmse_list)))
```

## 모델 훈련 및 평가

훈련 기능에서는 체중 감소와 함께 $L_2$ 손실을 채택합니다.중량 감쇠 메커니즘은 $L_2$ 정규화와 동일한 효과를 나타냅니다.

```{.python .input  n=4}
#@save
def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices=d2l.try_all_gpus(), evaluator=None,
                        **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            timer.start()
            input_data = []
            values = values if isinstance(values, list) else [values]
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            train_feat = input_data[0:-1] if len(values) > 1 else input_data
            train_label = input_data[-1]
            with autograd.record():
                preds = [net(*t) for t in zip(*train_feat)]
                ls = [loss(p, s) for p, s in zip(preds, train_label)]
            [l.backward() for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean() / len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        if len(kwargs) > 0:  # It will be used in section AutoRec
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'],
                                  devices)
        else:
            test_rmse = evaluator(net, test_iter, devices)
        train_l = l / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

마지막으로 모든 것을 한데 모아 모델을 훈련시켜 보겠습니다.여기서는 잠재 요인 차원을 30으로 설정합니다.

```{.python .input  n=5}
devices = d2l.try_all_gpus()
num_users, num_items, train_iter, test_iter = d2l.split_and_load_ml100k(
    test_ratio=0.1, batch_size=512)
net = MF(30, num_users, num_items)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                    devices, evaluator)
```

아래에서는 학습된 모델을 사용하여 사용자 (ID 20) 가 항목에 부여할 수 있는 등급 (ID 30) 을 예측합니다.

```{.python .input  n=6}
scores = net(np.array([20], dtype='int', ctx=devices[0]),
             np.array([30], dtype='int', ctx=devices[0]))
scores
```

## 요약

* 행렬 인수 분해 모델은 추천 시스템에서 널리 사용됩니다.사용자가 항목에 부여할 수 있는 등급을 예측하는 데 사용할 수 있습니다.
* 추천자 시스템에 대한 행렬 분해를 구현하고 훈련시킬 수 있습니다.

## 연습문제

* 잠재 요인의 크기를 변경합니다.잠재 요인의 크기가 모델 성능에 어떤 영향을 미칩니 까?
* 다양한 옵티마이저, 학습률 및 체중 감소율을 사용해 보십시오.
* 특정 영화에 대한 다른 사용자의 예상 평점 점수를 확인합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/400)
:end_tab:
