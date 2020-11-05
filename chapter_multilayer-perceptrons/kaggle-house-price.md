# 카글에 대한 주택 가격 예측
:label:`sec_kaggle_house`

이제 우리는 깊은 네트워크를 구축 및 훈련하고 체중 감량과 드롭아웃을 포함한 기술로 정규화하기위한 몇 가지 기본 도구를 도입했으므로 Kaggle 대회에 참가하여 이러한 모든 지식을 실천할 준비가 되었습니다.주택 가격 예측 경쟁은 시작하기에 좋은 장소입니다.데이터는 매우 일반적이며 특수 모델이 필요할 수있는 이국적인 구조를 나타내지 않습니다 (오디오 또는 비디오가 될 수 있음).2011년 바트 드 콕이 수집한 이 데이터셋은 2011년 :cite:`De-Cock.2011`에서 2006년부터 2010년까지 아이오와 에임스의 주택 가격을 다루고 있습니다.해리슨과 루빈펠드 (1978) 의 유명한 [Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) 보다 훨씬 넓으며, 더 많은 예제와 더 많은 기능을 자랑합니다.

이 섹션에서는 데이터 전처리, 모델 설계 및 하이퍼 매개 변수 선택에 대한 세부 사항을 안내합니다.우리는 실습 방식을 통해 데이터 과학자로서 경력을 쌓을 수 있는 몇 가지 직관을 얻을 수 있기를 바랍니다.

## 데이터 세트 다운로드 및 캐싱

이 책 전체에서 우리는 다운로드 한 다양한 데이터 세트에 대한 모델을 교육하고 테스트합니다.여기서는 데이터 다운로드를 용이하게하기 위해 몇 가지 유틸리티 기능을 구현합니다.먼저 문자열 (데이터 집합의*이름*) 을 데이터 집합을 찾는 URL과 파일의 무결성을 확인하는 SHA-1 키를 모두 포함하는 튜플에 매핑하는 사전 `DATA_HUB`를 유지합니다.이러한 모든 데이터 세트는 주소가 `DATA_URL` 인 사이트에서 호스팅됩니다.

```{.python .input}
#@tab all
import os
import requests
import zipfile
import tarfile
import hashlib

DATA_HUB = dict()  #@save
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'  #@save
```

다음 `download` 함수는 데이터 세트를 다운로드하여 로컬 디렉토리 (기본적으로 `../data`) 에 캐싱하고 다운로드한 파일의 이름을 반환합니다.이 데이터 세트에 해당하는 파일이 캐시 디렉토리에 이미 있고 SHA-1이 `DATA_HUB`에 저장된 파일과 일치하는 경우, 우리의 코드는 중복 다운로드로 인해 인터넷이 막히지 않도록 캐시 된 파일을 사용합니다.

```{.python .input}
#@tab all
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    d2l.mkdir_if_not_exist(cache_dir)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```

우리는 또한 두 가지 추가 유틸리티 기능을 구현합니다. 하나는 zip 또는 tar 파일을 다운로드하고 추출하고 다른 하나는 `DATA_HUB`에서이 책에 사용 된 모든 데이터 세트를 캐시 디렉토리로 다운로드하는 것입니다.

```{.python .input}
#@tab all
def download_extract(name, folder=None):  #@save
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """Download all files in the DATA_HUB."""
    for name in DATA_HUB:
        download(name)
```

## 카글

[Kaggle](https://www.kaggle.com)는 기계 학습 대회를 주최하는 인기있는 플랫폼입니다.데이터 세트의 각 경쟁 센터와 많은 사람들이 승리 솔루션에 상을 제공하는 이해 관계자에 의해 후원된다.이 플랫폼은 사용자가 포럼과 공유 코드를 통해 상호 작용할 수 있도록 지원하여 협업과 경쟁을 촉진합니다.리더보드를 쫓는 것은 종종 통제에서 벗어나는 경향이 있지만 연구원은 근본적인 질문을 던지기보다는 전처리 단계에 근적으로 초점을 맞추고 있지만 경쟁 접근법과 코드 간의 직접적인 정량적 비교를 용이하게하는 플랫폼의 객관성에는 엄청난 가치가 있습니다.모든 사람이 한 일을하지 않았다 무엇을 배울 수 있도록 공유.Kaggle 대회에 참가하려면 먼저 계정을 등록해야합니다 (:numref:`fig_kaggle` 참조).

![The Kaggle website.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

주택 가격 예측 경쟁 페이지에서 :numref:`fig_house_pricing`에 나와 있는 것처럼 데이터 집합 (“데이터” 탭 아래) 을 찾아 예측을 제출하고 순위를 확인할 수 있습니다. URL은 바로 여기에 있습니다.

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![The house price prediction competition page.](../img/house_pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## 데이터 집합 액세스 및 읽기

경쟁 데이터는 교육 및 테스트 세트로 구분됩니다.각 레코드는 주택의 속성 값과 같은 거리 유형, 건설 연도, 지붕 유형, 지하실 조건 등의 속성이 다양한 데이터 유형으로 구성되어 포함되어 있습니다.예를 들어, 구성 연도는 정수로, 지붕 유형은 불연속 범주형 할당값으로, 기타 피쳐는 부동 소수점 숫자로 표시됩니다.그리고 현실이 상황을 복잡하게 만드는 곳이 있습니다. 몇 가지 예를 들어, 일부 데이터가 누락되어 누락 된 값이 단순히 “na”로 표시됩니다.각 주택의 가격은 교육 세트에만 포함됩니다 (결국 경쟁입니다).학습 집합을 분할하여 유효성 검사 집합을 만들려고하지만 Kaggle에 예측을 업로드 한 후 공식 테스트 세트에서 모델을 평가하면됩니다.:numref:`fig_house_pricing`의 경쟁 탭에 있는 “데이터” 탭에는 데이터를 다운로드할 수 있는 링크가 있습니다.

시작하려면 :numref:`sec_pandas`에 도입 된 `pandas`를 사용하여 데이터를 읽고 처리합니다.따라서 더 진행하기 전에 `pandas`가 설치되어 있는지 확인해야 합니다.다행히도 Jupyter에서 읽는 경우 노트북을 떠나지 않고도 팬더를 설치할 수 있습니다.

```{.python .input}
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input}
#@tab pytorch
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import torch as d2l
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
```

```{.python .input}
#@tab tensorflow
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
import numpy as np
```

편의를 위해 위에서 정의한 스크립트를 사용하여 Kaggle 주택 데이터 세트를 다운로드하고 캐시 할 수 있습니다.

```{.python .input}
#@tab all
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

우리는 `pandas`를 사용하여 각각 교육 및 테스트 데이터를 포함하는 두 개의 CSV 파일을로드합니다.

```{.python .input}
#@tab all
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```

교육 데이터 집합에는 1460개의 예제, 80개의 기능 및 1개의 레이블이 포함되어 있으며 테스트 데이터에는 1459개의 예와 80개의 기능이 포함되어 있습니다.

```{.python .input}
#@tab all
print(train_data.shape)
print(test_data.shape)
```

우리가 처음 네 마지막 두 기능뿐만 아니라 처음 네 예에서 레이블 (SalePrice를) 살펴 보자.

```{.python .input}
#@tab all
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

각 예제에서 첫 번째 기능은 ID임을 알 수 있습니다.이렇게 하면 모델이 각 학습 예제를 식별하는 데 도움이 됩니다.이 기능은 편리하지만 예측 목적으로 정보를 전달하지 않습니다.따라서 데이터를 모델에 공급하기 전에 데이터 집합에서 제거합니다.

```{.python .input}
#@tab all
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## 데이터 사전 처리

전술 한 바와 같이, 우리는 데이터 유형의 다양한있다.모델링을 시작하기 전에 데이터를 사전 처리해야 합니다.숫자 기능부터 시작하겠습니다.먼저 모든 누락 된 값을 해당 기능의 평균으로 대체하여 휴리스틱을 적용합니다.그런 다음 모든 피처를 공통 척도에 배치하기 위해 피처를 제로 평균 및 단위 분산으로 재조정하여 데이터를 표준화* 합니다.

$$x \leftarrow \frac{x - \mu}{\sigma}.$$

이것이 실제로 우리의 기능 (변수) 을 변환하여 평균 및 단위 분산이 0인지 확인하려면 $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$와 $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$에 유의하십시오.직관적으로, 우리는 두 가지 이유로 데이터를 표준화합니다.첫째, 최적화에 편리하다는 것을 증명합니다.둘째, 어떤 피처가 관련되는*우선 순위*를 알지 못하기 때문에 한 피처에 할당 된 계수를 다른 피처보다 더 많이 처벌하고 싶지 않습니다.

```{.python .input}
#@tab all
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# After standardizing the data all means vanish, hence we can set missing
# values to 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

다음으로 우리는 이산 값을 다룬다.여기에는 “MSZoning”과 같은 기능이 포함됩니다.이전에 다중 클래스 레이블을 벡터로 변환 한 것과 같은 방식으로 단일 핫 인코딩으로 대체합니다 (:numref:`subsec_classification-problem` 참조).예를 들어, “MSZoning”은 “RL” 및 “RM” 값을 가정합니다.“MS조닝” 기능을 삭제하면 두 개의 새로운 표시기 기능 “Mszoning_rl” 및 “Mszoning_rm” 값이 0 또는 1인 만들어집니다.단일 핫 인코딩에 따르면 “MSzoning”의 원래 값이 “RL”이면 “MSzoning_rl”은 1이고 “MSzoning_rm”은 0입니다.`pandas` 패키지는 자동으로이 작업을 수행합니다.

```{.python .input}
#@tab all
# `Dummy_na=True` considers "na" (missing value) as a valid feature value, and
# creates an indicator feature for it
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

이 변환은 피처 수를 79에서 331으로 늘리는 것을 알 수 있습니다.마지막으로 `values` 속성을 통해 `pandas` 형식에서 NumPy 형식을 추출하여 훈련을 위해 텐서 표현으로 변환 할 수 있습니다.

```{.python .input}
#@tab all
n_train = train_data.shape[0]
train_features = d2l.tensor(all_features[:n_train].values, dtype=d2l.float32)
test_features = d2l.tensor(all_features[n_train:].values, dtype=d2l.float32)
train_labels = d2l.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=d2l.float32)
```

## 교육

시작하기 위해 우리는 제곱 손실을 가진 선형 모델을 훈련시킵니다.당연히 우리의 선형 모델은 경쟁 우위를 차지하는 제출으로 이어지지는 않지만 데이터에 의미 있는 정보가 있는지 확인하기 위해 온 전성 검사를 제공합니다.여기서 무작위 추측보다 더 잘 할 수 없다면 데이터 처리 버그가있을 가능성이 있습니다.그리고 일들이 작동한다면, 선형 모델은 단순한 모델이 가장 잘 보고된 모델에 얼마나 가까운지에 대한 직관을 제공하는 기준선 역할을 할 것입니다. 더 멋진 모델들로부터 얼마나 많은 이득을 기대해야하는지 알 수 있습니다.

```{.python .input}
loss = gluon.loss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()

def get_net():
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    return net
```

주택 가격은 주식 가격과 마찬가지로 절대 수량보다 상대적인 수량을 중요시합니다.따라서 우리는 절대 오류 $y - \hat{y}$보다 상대 오류 7323615에 대해 더 신경 쓰는 경향이 있습니다.예를 들어, 전형적인 집의 가치가 125,000달러인 오하이오 농촌의 주택 가격을 추정할 때 우리의 예측이 미화 100,000달러로 떨어진다면, 우리는 아마도 끔찍한 일을 하고 있을 것입니다.반면에, 캘리포니아 로스 알토스 힐스 (Los Altos Hills) 에서이 금액을 실수한다면, 이것은 놀랄만큼 정확한 예측을 나타낼 수 있습니다. (저기 중앙값은 4백만 달러를 초과합니다).

이 문제를 해결하는 한 가지 방법은 가격 견적의 대수의 불일치를 측정하는 것입니다.사실, 이것은 또한 제출의 품질을 평가하기 위해 경쟁에서 사용하는 공식적인 오류 조치입니다.결국, $|\log y - \log \hat{y}| \leq \delta$에 대한 작은 값 $\delta$은 $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$로 변환됩니다.이로 인해 예측 된 가격의 대수와 레이블 가격의 로그 사이에 다음과 같은 근본 평균 제곱 오차가 발생합니다.

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input}
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = np.clip(net(features), 1, float('inf'))
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())
```

```{.python .input}
#@tab pytorch
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(torch.mean(loss(torch.log(clipped_preds),
                                       torch.log(labels))))
    return rmse.item()
```

```{.python .input}
#@tab tensorflow
def log_rmse(y_true, y_pred):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = tf.clip_by_value(y_pred, 1, float('inf'))
    return tf.sqrt(tf.reduce_mean(loss(
        tf.math.log(y_true), tf.math.log(clipped_preds))))
```

이전 섹션과 달리 교육 기능은 Adam 최적화 프로그램에 의존합니다 (나중에 자세히 설명하겠습니다).이 옵티 마이저의 가장 큰 매력은 하이퍼 매개 변수 최적화를위한 무제한 리소스를 고려할 때 더 나은 (때로는 더 나쁜) 작업을 수행하지 못함에도 불구하고 사람들은 초기 학습 속도에 크게 덜 민감하다는 것을 발견하는 경향이 있다는 것입니다.

```{.python .input}
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

```{.python .input}
#@tab pytorch
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

```{.python .input}
#@tab tensorflow
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    net.compile(loss=loss, optimizer=optimizer)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = loss(y, y_hat)
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
        train_ls.append(log_rmse(train_labels, net(train_features)))
        if test_labels is not None:
            test_ls.append(log_rmse(test_labels, net(test_features)))
    return train_ls, test_ls
```

## $K$ 접힘 교차 검증

모델 선택 (:numref:`sec_model_selection`) 을 처리하는 방법을 설명한 섹션에서 $K$ 배 교차 유효성 검사를 도입한 것을 기억할 수 있습니다.우리는 모델 디자인을 선택하고 하이퍼 매개 변수를 조정하는 데 유용하게 사용할 것입니다.먼저 $K$ 배 교차 유효성 검사 절차에서 데이터의 $i^\mathrm{th}$ 배를 반환하는 함수가 필요합니다.$i^\mathrm{th}$ 세그먼트를 유효성 검사 데이터로 분할하고 나머지는 교육 데이터로 반환하여 진행됩니다.이것은 데이터를 처리하는 가장 효율적인 방법은 아니며 데이터 세트가 상당히 큰 경우 훨씬 더 똑똑한 것을 할 것입니다.그러나이 추가 된 복잡성으로 인해 코드를 불필요하게 난독 할 수 있으므로 문제의 단순성 때문에 여기에서 안전하게 생략 할 수 있습니다.

```{.python .input}
#@tab all
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = d2l.concat([X_train, X_part], 0)
            y_train = d2l.concat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

$K$ 배 교차 검증에서 $K$ 번 훈련하면 교육 및 검증 오류 평균이 반환됩니다.

```{.python .input}
#@tab all
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs+1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse',
                     legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

## 모델 선택

이 예에서는 조정되지 않은 하이퍼 매개 변수 집합을 선택하고 모델을 개선하기 위해 독자에게 맡깁니다.좋은 선택을 찾는 것은 최적화 된 변수의 수에 따라 시간이 걸릴 수 있습니다.충분히 큰 데이터 세트와 정상적인 종류의 하이퍼 매개 변수를 사용하면 $K$ 배 교차 유효성 검사는 여러 테스트에 대해 합리적으로 탄력적입니다.그러나 부당하게 많은 수의 옵션을 시도하면 운이 좋을 수 있으며 유효성 검사 성능이 더 이상 실제 오류를 나타내지 않는다는 것을 알 수 있습니다.

```{.python .input}
#@tab all
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, '
      f'avg valid log rmse: {float(valid_l):f}')
```

$K$ 배 교차 검증의 오류 수가 상당히 많더라도 하이퍼파라미터 집합에 대한 학습 오류 수가 매우 낮을 수 있습니다.이것은 우리가 과대 적합하고 있음을 나타냅니다.교육을 통해 두 숫자를 모두 모니터링하려고합니다.과대 적합도가 낮으면 데이터가 더 강력한 모형을 지원할 수 있음을 나타낼 수 있습니다.대규모 오버 피팅은 우리가 정규화 기술을 통합하여 얻을 수 있음을 제안 할 수 있습니다.

##  Kaggle에서 예측 제출

이제 하이퍼 매개 변수의 좋은 선택이 무엇인지 알았으므로 교차 유효성 검사 슬라이스에 사용되는 데이터의 $1-1/K$ 대신 모든 데이터를 사용하여 학습할 수 있습니다.이 방법으로 얻은 모델을 테스트 세트에 적용 할 수 있습니다.예측을 csv 파일로 저장하면 결과를 Kaggle에 간단하게 업로드할 수 있습니다.

```{.python .input}
#@tab all
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', yscale='log')
    print(f'train log rmse {float(train_ls[-1]):f}')
    # Apply the network to the test set
    preds = d2l.numpy(net(test_features))
    # Reformat it to export to Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

좋은 온 전성 검사 중 하나는 테스트 세트의 예측이 $K$ 배 교차 유효성 검사 프로세스의 예측과 유사한지 확인하는 것입니다.만약 그렇게 한다면, Kaggle에 업로드해야 할 때입니다.다음 코드는 `submission.csv`라는 파일을 생성합니다.

```{.python .input}
#@tab all
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

다음으로 :numref:`fig_kaggle_submit2`에서 알 수 있듯이 Kaggle에 대한 예측을 제출하고 테스트 세트의 실제 주택 가격 (라벨) 과 어떻게 비교하는지 확인할 수 있습니다.단계는 매우 간단합니다.

* Kaggle 웹 사이트에 로그인하여 주택 가격 예측 경쟁 페이지를 방문하십시오.
* “예측 제출” 또는 “늦게 제출” 버튼을 클릭합니다 (이 글을 쓰는 현재 버튼은 오른쪽에 있습니다).
* 페이지 하단의 파선 상자에서 “제출 서류 파일 업로드” 버튼을 클릭하고 업로드할 예상 파일을 선택합니다.
* 페이지 하단의 “제출하기” 버튼을 클릭하여 결과를 확인합니다.

![Submitting data to Kaggle](../img/kaggle_submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## 요약

* 실제 데이터에는 종종 다양한 데이터 유형이 혼합되어 있으므로 사전 처리해야 합니다.
* 실제 값 데이터를 평균 및 단위 분산으로 다시 스케일링하는 것이 좋은 기본값입니다.따라서 누락 된 값을 평균으로 대체하고 있습니다.
* 범주 기능을 표시기 기능으로 변환하면 한 핫 벡터처럼 처리 할 수 있습니다.
* $K$ 배 교차 검증을 사용하여 모델을 선택하고 하이퍼파라미터를 조정할 수 있습니다.
* 로그는 상대 오류에 유용합니다.

## 연습 문제

1. 이 섹션에 대한 예상 검색어를 Kaggle에 제출합니다.당신의 예측은 얼마나 좋습니까?
1. 가격의 로그를 직접 최소화하여 모델을 향상시킬 수 있습니까?가격이 아닌 가격의 로그를 예측하려고 하면 어떻게 됩니까?
1. 누락 된 값을 평균으로 대체하는 것이 항상 좋은 생각입니까?힌트: 값이 무작위로 누락되지 않은 상황을 구성 할 수 있습니까?
1. $K$ 배 교차 유효성 검사를 통해 하이퍼 매개 변수를 조정하여 Kaggle의 점수를 향상시킵니다.
1. 모델 (예: 레이어, 체중 감소 및 드롭아웃) 을 개선하여 점수를 향상시킵니다.
1. 우리는이 섹션에서 수행 한 것과 같은 연속 수치 기능을 표준화하지 않으면 어떻게됩니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/237)
:end_tab:
