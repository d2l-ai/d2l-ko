#  무비렌즈 데이터세트

추천 조사에 사용할 수 있는 많은 데이터셋이 있습니다.그 중에서도 [MovieLens](https://movielens.org/) 데이터셋은 아마도 가장 인기 있는 데이터셋 중 하나일 것입니다.MovieLens는 비상업적 웹 기반 영화 추천 시스템입니다.연구 목적으로 영화 등급 데이터를 수집하기 위해 1997년에 제작되었으며 미네소타 대학의 연구소인 GroupLens가 운영합니다.MovieLens 데이터는 맞춤형 추천 및 사회 심리학을 포함한 여러 연구 연구에 중요했습니다. 

## 데이터 얻기

무비렌즈 데이터세트는 [GroupLens](https://://grouplens.org/datasets/movielens/) website. Several versions are available. We will use the MovieLens 100K dataset :cite:`Herlocker.Konstan.Borchers.ea.1999`) 에 의해 호스팅됩니다.이 데이터세트는 1682개 영화의 사용자 943명에서 별 1개에서 5개에 이르는 $100,000$개의 등급으로 구성되어 있습니다.각 사용자가 최소 20편의 영화를 평가하도록 정리되었습니다.연령, 성별, 사용자 및 항목에 대한 장르와 같은 간단한 인구 통계 정보도 사용할 수 있습니다.[ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip)을 다운로드하고 csv 형식의 모든 $100,000$ 등급이 포함 된 `u.data` 파일을 추출 할 수 있습니다.폴더에는 다른 많은 파일이 있습니다. 각 파일에 대한 자세한 설명은 데이터 세트의 [README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) 파일에서 찾을 수 있습니다. 

먼저 이 섹션의 실험을 실행하는 데 필요한 패키지를 임포트해 보겠습니다.

```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
import pandas as pd
```

그런 다음 무비렌즈 100k 데이터세트를 다운로드하고 상호작용을 `DataFrame`로 로드합니다.

```{.python .input  n=2}
#@save
d2l.DATA_HUB['ml-100k'] = (
    'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

#@save
def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names,
                       engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
```

## 데이터세트 통계

데이터를 로드하고 처음 5개 레코드를 수동으로 검사해 보겠습니다.데이터 구조를 학습하고 제대로 로드되었는지 확인하는 효과적인 방법입니다.

```{.python .input  n=3}
data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))
```

각 줄은 “사용자 ID” 1-943, “항목 ID” 1-1682, “등급” 1-5 및 “타임 스탬프”를 포함하여 네 개의 열로 구성되어 있음을 알 수 있습니다.크기가 $n \times m$인 교호작용 행렬을 구성할 수 있습니다. 여기서 $n$과 $m$은 각각 사용자 수와 항목 수입니다.이 데이터셋은 기존 등급만 기록하므로 평가 행렬이라고도 할 수 있으며 이 행렬의 값이 정확한 등급을 나타내는 경우 상호 작용 행렬과 등급 행렬을 서로 바꿔서 사용합니다.사용자가 대부분의 영화를 평가하지 않았기 때문에 등급 매트릭스의 대부분의 값은 알 수 없습니다.이 데이터셋의 희소성도 보여줍니다.희소성은 `1 - number of nonzero entries / ( number of users * number of items)`로 정의됩니다.분명히 교호작용 행렬은 매우 희박합니다 (즉, 희소성 = 93.695%).실제 데이터 세트는 더 큰 희소성으로 인해 어려움을 겪을 수 있으며 추천자 시스템을 구축하는 데 오랜 도전이었습니다.실행 가능한 해결책은 사용자/항목 기능과 같은 추가 측면 정보를 사용하여 희소성을 완화하는 것입니다. 

그런 다음 서로 다른 등급의 카운트 분포를 플로팅합니다.예상대로 이 분포는 정규 분포인 것으로 보이며 대부분의 평점은 3-4입니다.

```{.python .input  n=4}
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()
```

## 데이터세트 분할

데이터세트를 훈련 세트와 테스트 세트로 분할합니다.다음 함수는 `random` 및 `seq-aware`를 포함한 두 가지 분할 모드를 제공합니다.`random` 모드에서 이 함수는 타임스탬프를 고려하지 않고 100k 교호작용을 무작위로 분할하고 기본적으로 데이터의 90% 를 훈련 표본으로 사용하고 나머지 10% 를 검정 표본으로 사용합니다.`seq-aware` 모드에서는 사용자가 테스트에 대해 가장 최근에 평가한 항목과 사용자의 과거 상호 작용을 교육 세트로 제외합니다.사용자 기록 상호 작용은 타임스탬프를 기준으로 가장 오래된 것부터 최신 항목 순으로 정렬됩니다.이 모드는 시퀀스 인식 권장 사항 섹션에서 사용됩니다.

```{.python .input  n=5}
#@save
def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

테스트 세트만 제외하고 실제로 검증 세트를 사용하는 것이 좋습니다.그러나 간결성을 위해 생략합니다.이 경우 테스트 세트는 보류된 유효성 검사 세트로 간주될 수 있습니다. 

## 데이터 로드

데이터 세트 분할 후 편의를 위해 훈련 세트와 테스트 세트를 목록과 사전/행렬로 변환합니다.다음 함수는 데이터 프레임을 한 줄씩 읽고 0부터 시작하는 사용자/항목의 인덱스를 열거합니다.그런 다음 함수는 사용자, 항목, 등급 및 상호 작용을 기록하는 사전/행렬 목록을 반환합니다.피드백 유형을 `explicit` 또는 `implicit`로 지정할 수 있습니다.

```{.python .input  n=6}
#@save
def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

그런 다음 위의 단계를 종합하면 다음 섹션에서 사용할 것입니다.결과는 `Dataset` 및 `DataLoader`로 래핑됩니다.훈련 데이터에 대한 `DataLoader` 중 `last_batch`은 `rollover` 모드로 설정되고 (나머지 샘플은 다음 시대로 롤오버됨) 주문이 섞입니다.

```{.python .input  n=7}
#@save
def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_iter = gluon.data.DataLoader(
        train_set, shuffle=True, last_batch='rollover',
        batch_size=batch_size)
    test_iter = gluon.data.DataLoader(
        test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
```

## 요약

* MovieLens 데이터 세트는 추천 연구에 널리 사용됩니다.공개적으로 사용할 수 있으며 무료로 사용할 수 있습니다.
* 이후 섹션에서 추가로 사용할 수 있도록 MovieLens 100k 데이터 세트를 다운로드하고 전처리하는 함수를 정의합니다.

## 연습문제

* 다른 유사한 추천 데이터셋은 무엇입니까?
* 무비렌즈에 대한 자세한 내용은 [https://movielens.org/](https://movielens.org/) 사이트를 참조하십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/399)
:end_tab:
