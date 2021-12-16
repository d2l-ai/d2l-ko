# 데이터 전처리
:label:`sec_pandas`

지금까지 텐서에 이미 저장된 데이터를 조작하는 다양한 기술을 도입했습니다.실제 문제를 해결하는 데 딥 러닝을 적용하기 위해 텐서 형식으로 잘 준비된 데이터가 아닌 원시 데이터를 전처리하는 것으로 시작하는 경우가 많습니다.파이썬에서 널리 사용되는 데이터 분석 도구 중에는 `pandas` 패키지가 일반적으로 사용됩니다.방대한 파이썬 생태계의 다른 많은 확장 패키지와 마찬가지로 `pandas`는 텐서와 함께 작동할 수 있습니다.따라서 `pandas`로 원시 데이터를 전처리하고 텐서 형식으로 변환하는 단계를 간략하게 살펴 보겠습니다.이후 장에서 더 많은 데이터 전처리 기술을 다룰 것입니다. 

## 데이터세트 읽기

예를 들어, (**csv (쉼표로 구분된 값) 파일에 저장된 인공 데이터 세트 만들기**) `../data/house_tiny.csv`로 시작합니다.다른 형식으로 저장된 데이터는 유사한 방식으로 처리될 수 있습니다. 

아래에서는 데이터 세트를 행별로 csv 파일에 씁니다.

```{.python .input}
#@tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

[**생성된 csv 파일에서 원시 데이터세트를 로드**] 하기 위해 `pandas` 패키지를 가져오고 `read_csv` 함수를 호출합니다.이 데이터셋에는 4개의 행과 3개의 열이 있으며, 각 행은 주택의 객실 수 (“NumRooms”), 골목 유형 (“골목”) 및 가격 (“가격”) 을 설명합니다.

```{.python .input}
#@tab all
# If pandas is not installed, just uncomment the following line:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## 누락된 데이터 처리

“NaN” 항목은 누락된 값입니다.누락된 데이터를 처리하기 위한 일반적인 방법에는*전치* 및*삭제*가 있습니다. 대치 방법은 결측값을 치환된 값으로 대체하고 삭제는 결측값을 무시합니다.여기서 우리는 대가를 고려할 것입니다. 

정수 위치 기반 인덱싱 (`iloc`) 을 통해 `data`를 `inputs` 및 `outputs`으로 분할합니다. 여기서 전자는 처음 두 열을 가져오고 후자는 마지막 열만 유지합니다.`inputs`에서 누락된 숫자 값의 경우 [**“ NaN” 항목을 동일한 열의 평균값으로 대체합니다.**]

```{.python .input}
#@tab all
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

[**`inputs`의 범주형 또는 이산형 값의 경우 “NaN”을 범주로 간주합니다.**] “골목” 열은 두 가지 유형의 범주형 값 “Pave”와 “NaN”만 사용하므로 `pandas`는 이 열을 “골목길”과 “골목_NaN”의 두 가지 열로 자동 변환할 수 있습니다.골목 유형이 “포장”인 행은 “골목_포장” 및 “골목_난”의 값을 1과 0으로 설정합니다.골목 유형이 누락된 행의 값은 0과 1로 설정됩니다.

```{.python .input}
#@tab all
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

## 텐서 형식으로 변환

이제 [**`inputs` 및 `outputs`의 모든 항목이 숫자이므로 텐서 형식으로 변환할 수 있습니다.**] 데이터가 이 형식이면 :numref:`sec_ndarray`에서 도입한 텐서 기능을 사용하여 추가로 조작할 수 있습니다.

```{.python .input}
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X, y
```

```{.python .input}
#@tab pytorch
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

X, y = tf.constant(inputs.values), tf.constant(outputs.values)
X, y
```

## 요약

* 방대한 파이썬 생태계의 다른 많은 확장 패키지와 마찬가지로 `pandas`는 텐서와 함께 작동할 수 있습니다.
* 대체 및 삭제는 누락된 데이터를 처리하는 데 사용할 수 있습니다.

## 연습문제

행과 열이 더 많은 원시 데이터세트를 생성합니다. 

1. 결측값이 가장 많은 열을 삭제합니다.
2. 전처리된 데이터세트를 텐서 형식으로 변환합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/195)
:end_tab:
