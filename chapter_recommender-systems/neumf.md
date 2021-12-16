# 개인화된 순위를 위한 신경 협업 필터링

이 섹션에서는 명시적 피드백을 넘어 암시적 피드백과 함께 추천을 위한 NCF (신경 협업 필터링) 프레임워크를 소개합니다.암시적 피드백은 추천자 시스템에 널리 퍼져 있습니다.클릭, 구매 및 시계와 같은 작업은 수집하기 쉽고 사용자의 선호도를 나타내는 일반적인 암시적 피드백입니다.신경 행렬 분해의 약자 인 NeumF :cite:`He.Liao.Zhang.ea.2017`라는 제목의 모델은 암시적 피드백으로 개인화 된 순위 작업을 해결하는 것을 목표로합니다.이 모델은 신경망의 유연성과 비선형성을 활용하여 행렬 분해의 내적을 대체하여 모델 표현력 향상을 목표로 합니다.특히 이 모델은 일반화 행렬 인수 분해 (GMF) 및 MLP를 포함한 두 개의 하위 네트워크로 구성되며 단순한 내적 대신 두 경로의 교호작용을 모델링합니다.이 두 네트워크의 출력은 최종 예측 점수 계산을 위해 연결됩니다.AutoRec의 등급 예측 작업과 달리 이 모델은 암시적 피드백을 기반으로 각 사용자에게 순위가 매겨진 추천 목록을 생성합니다.이 모델을 교육하기 위해 마지막 섹션에서 소개한 개인화된 순위 손실을 사용할 것입니다. 

## 노이MF 모델

앞서 언급한 바와 같이, NeuMF는 두 개의 서브네트워크를 융합합니다.GMF는 행렬 분해의 일반적인 신경망 버전으로, 입력값은 사용자 및 항목 잠재 요인의 요소별 곱입니다.두 개의 신경층으로 구성됩니다. 

$$
\mathbf{x} = \mathbf{p}_u \odot \mathbf{q}_i \\
\hat{y}_{ui} = \alpha(\mathbf{h}^\top \mathbf{x}),
$$

여기서 $\odot$는 벡터의 하다마르드 곱을 나타냅니다. $\mathbf{P} \in \mathbb{R}^{m \times k}$ 및 $\mathbf{Q} \in \mathbb{R}^{n \times k}$은 각각 사용자 및 항목 잠재 행렬에 해당합니다. $\mathbf{p}_u \in \mathbb{R}^{ k}$은 $P$의 $u^\mathrm{th}$ 행이고 $\mathbf{q}_i \in \mathbb{R}^{ k}$은 $Q$의 $i^\mathrm{th}$ 행입니다. $\alpha$ 및 $h$은 활성화 함수를 나타냅니다. 출력 레이어의 가중치. $\hat{y}_{ui}$은 $u$이 항목 $i$에 부여할 수 있는 사용자의 예측 점수입니다. 

이 모델의 또 다른 구성 요소는 MLP입니다.모델 유연성을 높이기 위해 MLP 하위 네트워크는 사용자 및 항목 임베딩을 GMF와 공유하지 않습니다.사용자 및 항목 임베딩의 연결을 입력으로 사용합니다.복잡한 연결과 비선형 변환을 통해 사용자와 항목 간의 복잡한 상호 작용을 예측할 수 있습니다.보다 정확하게는 MLP 서브네트워크는 다음과 같이 정의됩니다. 

$$
\begin{aligned}
z^{(1)} &= \phi_1(\mathbf{U}_u, \mathbf{V}_i) = \left[ \mathbf{U}_u, \mathbf{V}_i \right] \\
\phi^{(2)}(z^{(1)})  &= \alpha^1(\mathbf{W}^{(2)} z^{(1)} + b^{(2)}) \\
&... \\
\phi^{(L)}(z^{(L-1)}) &= \alpha^L(\mathbf{W}^{(L)} z^{(L-1)} + b^{(L)})) \\
\hat{y}_{ui} &= \alpha(\mathbf{h}^\top\phi^L(z^{(L-1)}))
\end{aligned}
$$

여기서 $\mathbf{W}^*, \mathbf{b}^*$ 및 $\alpha^*$는 가중치 행렬, 편향 벡터 및 활성화 함수를 나타냅니다. $\phi^*$은 해당 계층의 함수를 나타냅니다. $\mathbf{z}^*$는 해당 계층의 출력을 나타냅니다. 

GMF와 MLP의 결과를 융합하기 위해 NeuMF는 두 개의 서브네트워크의 두 번째 마지막 계층을 연결하여 추가 계층으로 전달할 수 있는 특징 벡터를 만듭니다.그 후 출력은 행렬 $\mathbf{h}$와 시그모이드 활성화 함수로 투영됩니다.예측 계층은 다음과 같이 공식화됩니다. $$\ hat {y} _ {ui} =\ 시그마 (\ mathbf {h} ^\ top [\ mathbf {x},\ phi^L (z^ {(L-1)})]).$$ 

다음 그림은 NeuMF의 모델 아키텍처를 보여줍니다. 

![Illustration of the NeuMF model](../img/rec-neumf.svg)

```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## 모델 구현 다음 코드는 NeuMF 모델을 구현합니다.일반화 행렬 인수 분해 모델과 사용자 및 항목 임베딩 벡터가 서로 다른 MLP로 구성됩니다.MLP의 구조는 매개 변수 `nums_hiddens`로 제어됩니다.ReLU는 기본 활성화 함수로 사용됩니다.

```{.python .input  n=2}
class NeuMF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, nums_hiddens,
                 **kwargs):
        super(NeuMF, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.U = nn.Embedding(num_users, num_factors)
        self.V = nn.Embedding(num_items, num_factors)
        self.mlp = nn.Sequential()
        for num_hiddens in nums_hiddens:
            self.mlp.add(nn.Dense(num_hiddens, activation='relu',
                                  use_bias=True))
        self.prediction_layer = nn.Dense(1, activation='sigmoid', use_bias=False)

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(np.concatenate([p_mlp, q_mlp], axis=1))
        con_res = np.concatenate([gmf, mlp], axis=1)
        return self.prediction_layer(con_res)
```

## 네거티브 샘플링을 사용한 맞춤형

데이터 쌍들의 순위 손실의 경우 중요한 단계는 음의 표본 추출입니다.각 사용자에 대해 사용자가 상호 작용하지 않은 항목은 후보 항목 (관찰되지 않은 항목) 입니다.다음 함수는 사용자의 ID 및 후보 항목을 입력으로 받아 해당 사용자의 후보 집합에서 각 사용자에 대해 임의로 음수 항목을 샘플링합니다.훈련 단계에서 모델은 사용자가 싫어하는 항목이나 상호 작용하지 않은 항목보다 순위가 높은 항목을 확인합니다.

```{.python .input  n=3}
class PRDataset(gluon.data.Dataset):
    def __init__(self, users, items, candidates, num_items):
        self.users = users
        self.items = items
        self.cand = candidates
        self.all = set([i for i in range(num_items)])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        neg_items = list(self.all - set(self.cand[int(self.users[idx])]))
        indices = random.randint(0, len(neg_items) - 1)
        return self.users[idx], self.items[idx], neg_items[indices]
```

## Evaluator 이 섹션에서는 훈련 세트와 테스트 세트를 구성하기 위해 시간별 분할 전략을 채택합니다.주어진 절단 $\ell$ ($\ text {Hit} @\ ell$) 에서의 적중률과 ROC 곡선 아래 영역 (AUC) 을 포함한 두 가지 평가 측정을 사용하여 모델 효과를 평가합니다.각 사용자에 대한 주어진 위치 $\ell$의 적중률은 권장 항목이 상위 $\ell$ 순위 목록에 포함되어 있는지 여부를 나타냅니다.공식적인 정의는 다음과 같습니다. 

$$
\text{Hit}@\ell = \frac{1}{m} \sum_{u \in \mathcal{U}} \textbf{1}(rank_{u, g_u} <= \ell),
$$

여기서 $\textbf{1}$은 지상 진리 항목이 상위 $\ell$ 목록에 순위가 매겨진 경우 1과 동일한 표시기 함수를 나타내며, 그렇지 않으면 0과 같습니다. $rank_{u, g_u}$은 추천 목록에서 사용자 $u$의 지상 진리 항목 $g_u$의 순위를 나타냅니다 (이상적인 순위는 1). $m$은 다음과 같습니다.사용자 수입니다. $\mathcal{U}$가 사용자 집합입니다. 

AUC의 정의는 다음과 같습니다. 

$$
\text{AUC} = \frac{1}{m} \sum_{u \in \mathcal{U}} \frac{1}{|\mathcal{I} \backslash S_u|} \sum_{j \in I \backslash S_u} \textbf{1}(rank_{u, g_u} < rank_{u, j}),
$$

여기서 $\mathcal{I}$는 항목 집합입니다. $S_u$은 사용자 $u$의 후보 항목입니다.정밀도, 리콜 및 정규화된 할인 누적 이득 (NDCG) 과 같은 다른 많은 평가 프로토콜도 사용할 수 있습니다. 

다음 함수는 각 사용자에 대한 적중 횟수와 AUC를 계산합니다.

```{.python .input  n=4}
#@save
def hit_and_auc(rankedlist, test_matrix, k):
    hits_k = [(idx, val) for idx, val in enumerate(rankedlist[:k])
              if val in set(test_matrix)]
    hits_all = [(idx, val) for idx, val in enumerate(rankedlist)
                if val in set(test_matrix)]
    max = len(rankedlist) - 1
    auc = 1.0 * (max - hits_all[0][0]) / max if len(hits_all) > 0 else 0
    return len(hits_k), auc
```

그런 다음 전체 적중률과 AUC가 다음과 같이 계산됩니다.

```{.python .input  n=5}
#@save
def evaluate_ranking(net, test_input, seq, candidates, num_users, num_items,
                     devices):
    ranked_list, ranked_items, hit_rate, auc = {}, {}, [], []
    all_items = set([i for i in range(num_users)])
    for u in range(num_users):
        neg_items = list(all_items - set(candidates[int(u)]))
        user_ids, item_ids, x, scores = [], [], [], []
        [item_ids.append(i) for i in neg_items]
        [user_ids.append(u) for _ in neg_items]
        x.extend([np.array(user_ids)])
        if seq is not None:
            x.append(seq[user_ids, :])
        x.extend([np.array(item_ids)])
        test_data_iter = gluon.data.DataLoader(
            gluon.data.ArrayDataset(*x), shuffle=False, last_batch="keep",
            batch_size=1024)
        for index, values in enumerate(test_data_iter):
            x = [gluon.utils.split_and_load(v, devices, even_split=False)
                 for v in values]
            scores.extend([list(net(*t).asnumpy()) for t in zip(*x)])
        scores = [item for sublist in scores for item in sublist]
        item_scores = list(zip(item_ids, scores))
        ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
        ranked_items[u] = [r[0] for r in ranked_list[u]]
        temp = hit_and_auc(ranked_items[u], test_input[u], 50)
        hit_rate.append(temp[0])
        auc.append(temp[1])
    return np.mean(np.array(hit_rate)), np.mean(np.array(auc))
```

## 모델 훈련 및 평가

훈련 기능은 아래에 정의되어 있습니다.모델을 쌍별 방식으로 훈련시킵니다.

```{.python .input  n=6}
#@save
def train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, devices, evaluator,
                  candidates, eval_step=1):
    timer, hit_rate, auc = d2l.Timer(), 0, 0
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['test hit rate', 'test AUC'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            input_data = []
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            with autograd.record():
                p_pos = [net(*t) for t in zip(*input_data[0:-1])]
                p_neg = [net(*t) for t in zip(*input_data[0:-2],
                                              input_data[-1])]
                ls = [loss(p, n) for p, n in zip(p_pos, p_neg)]
            [l.backward(retain_graph=False) for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean()/len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        with autograd.predict_mode():
            if (epoch + 1) % eval_step == 0:
                hit_rate, auc = evaluator(net, test_iter, test_seq_iter,
                                          candidates, num_users, num_items,
                                          devices)
                animator.add(epoch + 1, (hit_rate, auc))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test hit rate {float(hit_rate):.3f}, test AUC {float(auc):.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

이제 MovieLens 100k 데이터세트를 로드하고 모델을 훈련시킬 수 있습니다.MovieLens 데이터 세트에는 등급만 있고 정확도가 약간 손실되므로 이러한 등급을 0과 1로 이진화합니다.사용자가 항목에 등급을 매기면 암시적 피드백을 1로 간주하고 그렇지 않으면 0으로 간주합니다.항목에 등급을 매기는 작업은 암시적 피드백을 제공하는 형태로 취급할 수 있습니다.여기서는 사용자의 최근 상호작용한 항목이 테스트를 위해 제외되는 `seq-aware` 모드에서 데이터세트를 분할합니다.

```{.python .input  n=11}
batch_size = 1024
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_iter = gluon.data.DataLoader(
    PRDataset(users_train, items_train, candidates, num_items ), batch_size,
    True, last_batch="rollover", num_workers=d2l.get_dataloader_workers())
```

그런 다음 모델을 만들고 초기화합니다. 숨겨진 크기가 10 인 3 레이어 MLP를 사용합니다.

```{.python .input  n=8}
devices = d2l.try_all_gpus()
net = NeuMF(10, num_users, num_items, nums_hiddens=[10, 10, 10])
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
```

다음 코드는 모델을 훈련시킵니다.

```{.python .input  n=12}
lr, num_epochs, wd, optimizer = 0.01, 10, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_ranking(net, train_iter, test_iter, loss, trainer, None, num_users,
              num_items, num_epochs, devices, evaluate_ranking, candidates)
```

## 요약

* 행렬 인수 분해 모델에 비선형성을 추가하면 모형 공정 능력 및 효과를 개선하는 데 유용합니다.
* NeuMF는 행렬 분해와 다층 퍼셉트론의 조합입니다.다중 레이어 퍼셉트론은 사용자 및 항목 임베딩의 연결을 입력으로 사용합니다.

## 연습문제

* 잠재 요인의 크기를 변경합니다.잠재 요인의 크기가 모델 성능에 어떤 영향을 미칩니 까?
* MLP의 아키텍처 (예: 레이어 수, 각 레이어의 뉴런 수) 를 변경하여 MLP가 성능에 미치는 영향을 확인합니다.
* 다양한 옵티마이저, 학습률 및 체중 감소율을 사용해 보십시오.
* 마지막 섹션에서 정의한 힌지 손실을 사용하여 이 모델을 최적화하십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/403)
:end_tab:
