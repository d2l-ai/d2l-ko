# Kaggle의 주택 가격 예측
:label:`sec_kaggle_house`

이제 심층 네트워크를 구축 및 훈련하고 체중 감소 및 드롭아웃을 포함한 기술로 정규화하기 위한 몇 가지 기본 도구를 도입했으므로 Kaggle 대회에 참가하여 이러한 모든 지식을 실천할 준비가 되었습니다.주택 가격 예측 경쟁은 시작하기에 좋은 곳입니다.데이터는 매우 일반적이며 특수한 모델 (오디오 또는 비디오) 이 필요할 수 있는 이국적인 구조를 나타내지 않습니다.2011년 :cite:`De-Cock.2011`년에 바트 드 콕이 수집한 이 데이터세트는 2006년부터 2010년까지 아이오와 주 에임스의 주택 가격을 다룹니다.해리슨과 루빈펠드 (1978) 의 유명한 [Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) (1978) 보다 훨씬 크며 더 많은 예와 더 많은 기능을 자랑합니다.

이 섹션에서는 데이터 전처리, 모델 설계 및 하이퍼파라미터 선택에 대해 자세히 설명합니다.실습 접근 방식을 통해 데이터 과학자로서 경력을 쌓을 수 있는 몇 가지 직관을 얻을 수 있기를 바랍니다.

## 데이터세트 다운로드 및 캐싱

책 전체에서 다운로드된 다양한 데이터 세트에 대한 모델을 학습하고 테스트합니다.여기서는 (**데이터 다운로드를 용이하게 하기 위해 몇 가지 유틸리티 함수를 구현합니다**).먼저 문자열 (데이터 세트의*이름*) 을 데이터 세트를 찾는 URL과 파일의 무결성을 확인하는 SHA-1 키를 모두 포함하는 튜플에 매핑하는 사전 `DATA_HUB`를 유지합니다.이러한 모든 데이터 세트는 주소가 `DATA_URL`인 사이트에서 호스팅됩니다.

```{.python .input}
#@tab all
import os
import requests
import zipfile
import tarfile
import hashlib

#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
```

다음 `download` 함수는 데이터 세트를 다운로드하고 로컬 디렉토리 (기본값 `../data`) 에 캐시한 다음 다운로드한 파일의 이름을 반환합니다.이 데이터 세트에 해당하는 파일이 캐시 디렉토리에 이미 있고 해당 SHA-1 이 `DATA_HUB`에 저장된 파일과 일치하는 경우, 코드는 중복 다운로드로 인해 인터넷이 막히는 것을 방지하기 위해 캐시된 파일을 사용합니다.

```{.python .input}
#@tab all
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
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

또한 두 가지 추가 유틸리티 함수를 구현합니다. 하나는 zip 또는 tar 파일을 다운로드하여 추출하는 것이고 다른 하나는 이 책에 사용된 모든 데이터셋을 `DATA_HUB`에서 캐시 디렉터리로 다운로드하는 것입니다.

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

## Kaggle

[Kaggle](https://www.kaggle.com)는 기계 학습 대회를 주최하는 인기 있는 플랫폼입니다.각 경쟁은 데이터 세트를 중심으로하며 많은 경쟁은 수상 솔루션에 상을 제공하는 이해 관계자의 후원을 받습니다.이 플랫폼은 사용자가 포럼 및 공유 코드를 통해 상호 작용할 수 있도록 지원하여 협업과 경쟁을 촉진합니다.리더 보드 추격은 종종 통제 불능 상태이지만 연구자들은 근본적인 질문을하지 않고 전처리 단계에 근시적으로 초점을 맞추고 있지만, 경쟁 접근 방식과 코드 간의 직접적인 정량적 비교를 용이하게하는 플랫폼의 객관성에는 엄청난 가치가 있습니다.공유하여 모든 사람이 효과가 있었던 것과 작동하지 않는 것을 배울 수 있습니다.Kaggle 대회에 참가하려면 먼저 계정을 등록해야 합니다 (:numref:`fig_kaggle` 참조).

![The Kaggle website.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

:numref:`fig_house_pricing`에 설명된 대로 주택 가격 예측 경쟁 페이지에서 데이터 세트 (“데이터” 탭 아래) 를 찾고 예측을 제출하고 순위를 확인할 수 있습니다. URL은 바로 여기에 있습니다.

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![The house price prediction competition page.](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## 데이터세트 액세스 및 읽기

경쟁 데이터는 훈련 세트와 테스트 세트로 구분됩니다.각 레코드에는 주택의 속성 값과 거리 유형, 건축 연도, 지붕 유형, 지하실 상태 등과 같은 속성이 포함됩니다. 이 기능은 다양한 데이터 유형으로 구성됩니다.예를 들어, 건축 연도는 정수로, 지붕 유형은 불연속 범주형 할당으로, 기타 피쳐는 부동 소수점 숫자로 표시됩니다.그리고 현실이 상황을 복잡하게 만드는 부분이 있습니다. 예를 들어, 일부 데이터는 단순히 “na”로 표시된 누락 된 값으로 완전히 누락되었습니다.각 주택의 가격은 교육 세트에만 포함됩니다 (결국 경쟁입니다).학습 세트를 분할하여 검증 세트를 만들고 싶지만 Kaggle에 예측을 업로드한 후에만 공식 테스트 세트에서 모델을 평가할 수 있습니다.:numref:`fig_house_pricing`의 경쟁 탭에 있는 “데이터” 탭에는 데이터를 다운로드할 수 있는 링크가 있습니다.

시작하기 위해 :numref:`sec_pandas`에서 소개한 [**`pandas`를 사용하여 데이터를 읽고 처리**] 할 것입니다.따라서 계속 진행하기 전에 `pandas`가 설치되어 있는지 확인해야 합니다.다행히도 Jupyter에서 책을 읽고 있다면 노트북을 떠나지 않고도 팬더를 설치할 수 있습니다.

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
from torch import nn
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

편의를 위해 위에서 정의한 스크립트를 사용하여 Kaggle 하우징 데이터 세트를 다운로드하고 캐시할 수 있습니다.

```{.python .input}
#@tab all
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

`pandas`를 사용하여 훈련 및 테스트 데이터가 포함된 두 개의 csv 파일을 각각 로드합니다.

```{.python .input}
#@tab all
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```

훈련 데이터셋에는 1460개의 예제, 80개의 특징, 1개의 레이블이 포함되어 있으며, 테스트 데이터에는 1459개의 예제와 80개의 특징이 포함되어 있습니다.

```{.python .input}
#@tab all
print(train_data.shape)
print(test_data.shape)
```

처음 네 가지 예에서 [**처음 네 가지 기능과 마지막 두 가지 기능과 레이블 (SalePrice) **] 을 살펴 보겠습니다.

```{.python .input}
#@tab all
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

각 예에서 (**첫 번째 기능은 ID**) 모델이 각 훈련 예제를 식별하는 데 도움이 된다는 것을 알 수 있습니다.편리하지만 예측 목적으로 정보를 전달하지는 않습니다.따라서 모델에 데이터를 공급하기 전에 (**데이터 세트에서 제거**) 합니다.

```{.python .input}
#@tab all
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## 데이터 전처리

위에서 언급한 바와 같이 다양한 데이터 유형이 있습니다.모델링을 시작하기 전에 데이터를 전처리해야 합니다.수치 적 특징부터 시작하겠습니다.먼저 휴리스틱 (**모든 누락된 값을 해당 기능의 평균으로 대체 을 적용한 다음, 모든 특징을 공통 척도로 배치하기 위해 특징을 0 평균과 단위 분산으로 다시 스케일링하여 데이터를 표준화**) 합니다.

$$x \leftarrow \frac{x - \mu}{\sigma},$$

여기서 $\mu$와 $\sigma$은 각각 평균과 표준 편차를 나타냅니다.이것이 평균과 단위 분산이 0이되도록 기능 (변수) 을 실제로 변환하는지 확인하려면 $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$와 $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$에 유의하십시오.직관적으로 데이터를 표준화하는 이유는 두 가지입니다.첫째, 최적화에 편리함을 입증합니다.둘째, 어떤 기능이 관련성이 있는*선험적*을 알지 못하기 때문에 한 특성에 할당된 계수에 다른 특성보다 더 많은 불이익을 주고 싶지 않습니다.

```{.python .input}
#@tab all
# If test data were inaccessible, mean and standard deviation could be
# calculated from training data
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# After standardizing the data all means vanish, hence we can set missing
# values to 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

[**다음으로 불연속 값을 다루겠습니다.**] 여기에는 “MSZoning”과 같은 기능이 포함됩니다.이전에 멀티클래스 레이블을 벡터로 변환한 것과 같은 방식으로 (**원-핫 인코딩으로 대체**) 합니다 (:numref:`subsec_classification-problem` 참조).예를 들어, “MSZoning”은 “RL”과 “RM” 값을 가정합니다.“MSZoning” 기능을 삭제하면 값이 0 또는 1인 두 개의 새로운 표시기 기능 “MSZoning_rl”과 “MSZoning_RM”이 생성됩니다.원-핫 인코딩에 따르면 “MSZoning”의 원래 값이 “RL”이면 “MSZoning_RL”은 1이고 “MSZoning_RM”은 0입니다.`pandas` 패키지는 이 작업을 자동으로 수행합니다.

```{.python .input}
#@tab all
# `Dummy_na=True` considers "na" (missing value) as a valid feature value, and
# creates an indicator feature for it
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

이 변환으로 피처 수가 79개에서 331로 증가한다는 것을 알 수 있습니다.마지막으로 `values` 속성을 통해 [**`pandas` 형식에서 NumPy 형식을 추출하여 훈련을 위해 텐서**] 표현으로 변환 할 수 있습니다.

```{.python .input}
#@tab all
n_train = train_data.shape[0]
train_features = d2l.tensor(all_features[:n_train].values.astype(np.float32), dtype=d2l.float32)
test_features = d2l.tensor(all_features[n_train:].values.astype(np.float32), dtype=d2l.float32)
train_labels = d2l.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=d2l.float32)
```

## [**교육**]

시작하기 위해 손실 제곱으로 선형 모델을 훈련시킵니다.당연히 우리의 선형 모델은 경쟁에서 우승 한 제출로 이어지지는 않지만 데이터에 의미있는 정보가 있는지 확인하기 위해 온전성 검사를 제공합니다.여기서 무작위로 추측하는 것보다 더 잘할 수 없다면 데이터 처리 버그가 발생할 가능성이 높습니다.그리고 일이 작동한다면 선형 모델은 단순한 모델이 가장 잘 보고된 모델에 얼마나 가까워지는지에 대한 직관을 제공하는 기준선 역할을 하여 더 멋진 모델에서 얼마나 많은 이득을 기대해야하는지 알 수 있습니다.

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

주택 가격은 주가와 마찬가지로 절대 수량보다 상대 수량을 더 중요하게 생각합니다.따라서 [**우리는 절대 오차 $y - \hat{y}$보다 상대 오차 $\frac{y - \hat{y}}{y}$**] 에 더 신경을 쓰는 경향이 있습니다.예를 들어, 전형적인 주택의 가치가 125,000 달러인 오하이오 시골의 주택 가격을 추정 할 때 예측이 100,000 달러 할인된다면 우리는 아마도 끔찍한 일을하고 있을 것입니다.반면에 캘리포니아 로스 알토스 힐스 (Los Altos Hills) 에서이 금액으로 잘못하면 놀랍도록 정확한 예측을 나타낼 수 있습니다 (평균 주택 가격이 400 만 달러를 초과 함).

(**이 문제를 해결하는 한 가지 방법은 가격 견적 로그의 불일치를 측정하는 것입니다.**) 실제로 이는 경쟁 업체가 제출 품질을 평가하기 위해 사용하는 공식 오류 측정치이기도합니다.결국 $|\log y - \log \hat{y}| \leq \delta$에 대한 작은 값 $\delta$는 $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$로 변환됩니다.이로 인해 예측 가격의 로그와 라벨 가격의 로그 사이에 다음과 같은 평균 제곱근 오차가 발생합니다.

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
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
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

이전 섹션과 달리 [**교육 함수는 Adam 옵티마이저에 의존합니다 (나중에 자세히 설명하겠습니다) **].이 옵티마이저의 주된 매력은 하이퍼파라미터 최적화를 위한 무제한 리소스가 주어지더라도 초기 학습 속도에 훨씬 덜 민감하다는 것을 사람들이 발견하는 경향이 있다는 것입니다.

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

## $K$겹 교차 검증

모델 선택을 처리하는 방법 (:numref:`sec_model_selection`) 에 대해 논의한 섹션에서 [**$K$겹 교차 검증**] 을 도입했음을 기억하실 것입니다.모델 설계를 선택하고 초모수를 조정하는 데 이 방법을 유용하게 사용할 것입니다.먼저 $K$겹 교차 검증 절차에서 데이터의 $i^\mathrm{th}$배를 반환하는 함수가 필요합니다.$i^\mathrm{th}$ 세그먼트를 검증 데이터로 분할하고 나머지는 훈련 데이터로 반환하는 방식으로 진행됩니다.이것은 데이터를 처리하는 가장 효율적인 방법은 아니며 데이터 세트가 상당히 크면 훨씬 더 현명한 작업을 수행 할 것입니다.그러나 이렇게 추가된 복잡성으로 인해 코드가 불필요하게 난독 화될 수 있으므로 문제의 단순성 때문에 여기서 코드를 안전하게 생략할 수 있습니다.

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

$K$겹 교차 검증에서 $K$번 훈련하면 [**훈련 및 검증 오차 평균이 반환됩니다**].

```{.python .input}
#@tab all
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

## [**모델 선택**]

이 예제에서는 튜닝되지 않은 하이퍼파라미터 세트를 선택하고 모델을 개선하기 위해 독자에게 맡깁니다.최적화하는 변수의 수에 따라 적절한 선택을 찾는 데 시간이 걸릴 수 있습니다.데이터셋이 충분히 크고 일반적인 종류의 하이퍼파라미터가 있는 경우 $K$겹 교차 검증은 다중 테스트에 대해 상당히 탄력적인 경향이 있습니다.그러나 부당하게 많은 수의 옵션을 시도하면 운이 좋으면 유효성 검사 성능이 더 이상 실제 오류를 나타내지 않는다는 것을 알게 될 것입니다.

```{.python .input}
#@tab all
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, '
      f'avg valid log rmse: {float(valid_l):f}')
```

$K$겹 교차 검증의 오차 수가 상당히 많더라도 하이퍼파라미터 집합에 대한 훈련 오차 수가 매우 적을 수 있는 경우가 있습니다.이는 우리가 과적합되고 있음을 나타냅니다.훈련 내내 두 수치를 모두 모니터링하는 것이 좋습니다.과적합이 적으면 데이터가 더 강력한 모형을 지원할 수 있다는 것을 나타낼 수 있습니다.대규모 과적합은 정규화 기술을 통합하여 얻을 수 있음을 시사 할 수 있습니다.

##  [**Kaggle에 대한 예측 제출**]

이제 하이퍼파라미터의 좋은 선택이 무엇인지 알았으므로 교차 검증 슬라이스에 사용되는 데이터의 $1-1/K$만 사용하는 것이 아니라 모든 데이터를 사용하여 해당 하이퍼파라미터를 학습시킬 수도 있습니다.이렇게 얻은 모델을 테스트 세트에 적용할 수 있습니다.예측을 csv 파일에 저장하면 결과를 Kaggle에 업로드하는 작업이 간소화됩니다.

```{.python .input}
#@tab all
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'train log rmse {float(train_ls[-1]):f}')
    # Apply the network to the test set
    preds = d2l.numpy(net(test_features))
    # Reformat it to export to Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

한 가지 좋은 온전성 검사는 검정 세트의 예측이 $K$겹 교차 검증 과정의 예측과 유사한지 확인하는 것입니다.그렇다면 Kaggle에 업로드할 차례입니다.다음 코드는 `submission.csv`라는 파일을 생성합니다.

```{.python .input}
#@tab all
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

다음으로 :numref:`fig_kaggle_submit2`에서 설명한 것처럼 Kaggle에 대한 예측을 제출하고 테스트 세트의 실제 주택 가격 (라벨) 과 어떻게 비교되는지 확인할 수 있습니다.단계는 매우 간단합니다.

* Kaggle 웹 사이트에 로그인하고 주택 가격 예측 경쟁 페이지를 방문하십시오.
* “예측 제출” 또는 “제출 지연” 버튼을 클릭합니다 (이 글을 쓰는 시점에서 버튼은 오른쪽에 있습니다).
* 페이지 하단의 파선 상자에서 “제출 파일 업로드” 버튼을 클릭하고 업로드하려는 예측 파일을 선택합니다.
* 페이지 하단의 “제출하기” 버튼을 클릭하여 결과를 확인합니다.

![Submitting data to Kaggle](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## 요약

* 실제 데이터에는 종종 서로 다른 데이터 유형이 혼합되어 있으므로 사전 처리해야 합니다.
* 실수 값 데이터를 평균 0 및 단위 분산으로 다시 스케일링하는 것이 좋습니다.결측값을 평균으로 바꾸는 것도 마찬가지입니다.
* 범주형 특징을 지표 특징으로 변환하면 이를 원핫 벡터처럼 취급할 수 있습니다.
* $K$겹 교차 검증을 사용하여 모델을 선택하고 초모수를 조정할 수 있습니다.
* 로그는 상대 오차에 유용합니다.

## 연습문제

1. 이 섹션에 대한 예측을 Kaggle에 제출하십시오.예측이 얼마나 좋은가요?
1. 가격 로그를 직접 최소화하여 모델을 개선할 수 있습니까?가격이 아닌 가격의 로그를 예측하려고 하면 어떻게 될까요?
1. 결측값을 평균으로 바꾸는 것이 항상 좋은 생각입니까?힌트: 값이 무작위로 누락되지 않는 상황을 만들 수 있습니까?
1. $K$겹 교차 검증을 통해 초모수를 조정하여 Kaggle의 점수를 향상시킵니다.
1. 모델을 개선하여 점수를 높입니다 (예: 레이어, 가중치 감소, 드롭아웃).
1. 이 섹션에서 수행한 것과 같은 연속 수치 특성을 표준화하지 않으면 어떻게 될까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/237)
:end_tab:
