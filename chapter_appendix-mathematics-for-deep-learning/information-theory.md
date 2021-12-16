# 정보 이론
:label:`sec_information_theory`

우주는 정보로 넘쳐납니다.정보는 셰익스피어의 소네트부터 코넬 아르Xiv에 관한 연구원 논문, 반 고흐의 인쇄 별이 빛나는 밤부터 베토벤의 음악 교향곡 5번, 최초의 프로그래밍 언어 Plankalkül에서 최첨단 기계 학습 알고리즘에 이르기까지 징계 균열 전반에 걸쳐 공통 언어를 제공합니다.형식에 관계없이 모든 것이 정보 이론의 규칙을 따라야합니다.정보 이론을 통해 서로 다른 신호에 존재하는 정보의 양을 측정하고 비교할 수 있습니다.이 섹션에서는 정보 이론의 기본 개념과 기계 학습에서 정보 이론의 적용에 대해 살펴 봅니다. 

시작하기 전에 머신 러닝과 정보 이론 간의 관계를 간략하게 설명하겠습니다.머신러닝은 데이터에서 흥미로운 신호를 추출하고 중요한 예측을 하는 것을 목표로 합니다.반면에 정보 이론은 정보의 인코딩, 디코딩, 전송 및 조작을 연구합니다.결과적으로 정보 이론은 기계 학습 시스템에서 정보 처리를 논의하기 위한 기본 언어를 제공합니다.예를 들어, 많은 기계 학습 응용 프로그램은 :numref:`sec_softmax`에 설명된 대로 교차 엔트로피 손실을 사용합니다.이러한 손실은 정보 이론적 고려 사항에서 직접 파생될 수 있습니다. 

## 인포메이션

정보 이론의 “영혼”, 즉 정보부터 시작하겠습니다.*정보*는 하나 이상의 인코딩 형식의 특정 시퀀스로 인코딩할 수 있습니다.우리가 정보의 개념을 정의하려고 노력한다고 가정해 봅시다.우리의 출발점은 무엇일까요? 

다음과 같은 사고 실험을 생각해 보십시오.한 벌의 카드를 가진 친구가 있습니다.그들은 덱을 섞고, 카드 몇 장을 뒤집고, 카드에 대한 진술을 알려줄 것입니다.우리는 각 진술의 정보 내용을 평가하려고 노력할 것입니다. 

먼저, 그들은 카드를 뒤집어 “카드가 보여요.” 라고 말합니다.이것은 우리에게 전혀 정보를 제공하지 않습니다.우리는 이미 이것이 사실이라고 확신했기 때문에 정보가 0이되기를 바랍니다. 

그런 다음 카드를 뒤집어 “심장이 보입니다.” 라고 말합니다.이것은 우리에게 몇 가지 정보를 제공하지만 실제로는 $4$ 개의 서로 다른 수트가 가능했으며 각각 똑같이 가능하므로이 결과에 놀라지 않습니다.정보의 척도가 무엇이든간에, 이 이벤트는 정보 내용이 적어야하기를 바랍니다. 

다음으로, 그들은 카드를 뒤집어 “이것은 스페이드의 $3$입니다.” 라고 말합니다.자세한 내용은 다음과 같습니다.실제로 $52$의 결과가 똑같이 가능했고, 우리 친구는 그것이 어떤 결과인지 알려주었습니다.중간 정도의 정보여야 합니다. 

이것을 논리적 극단으로 가져 가자.마지막으로 덱에서 모든 카드를 뒤집어 뒤섞은 덱의 전체 순서를 읽는다고 가정합니다.52 달러가 있습니다!갑판에 $ 다른 주문, 다시 모두 똑같이 가능성이 있으므로 어떤 주문인지 알기 위해 많은 정보가 필요합니다. 

우리가 개발하는 정보에 대한 모든 개념은 이러한 직관에 부합해야 합니다.실제로 다음 섹션에서는 이러한 이벤트에 각각 $0\text{ bits}$, $2\text{ bits}$, $~5.7\ 텍스트 {비트} $, and $~225.6\ 텍스트 {비트} $의 정보가 있는지 계산하는 방법을 배웁니다. 

이러한 사고 실험을 통해 읽으면 자연스러운 아이디어를 볼 수 있습니다.출발점으로 지식에 관심을 가지기보다는 정보가 놀라움의 정도 또는 사건의 추상적 가능성을 나타낸다는 생각을 쌓을 수 있습니다.예를 들어 비정상적인 이벤트를 설명하려면 많은 정보가 필요합니다.일반적인 이벤트의 경우 많은 정보가 필요하지 않을 수 있습니다. 

1948년 클로드 이 섀넌은 정보 이론을 확립하는*수학적 의사 소통 이론* :cite:`Shannon.1948`를 출판했습니다.그의 기사에서 Shannon은 처음으로 정보 엔트로피의 개념을 소개했습니다.여기서 여정을 시작하겠습니다. 

### 셀프 정보

정보는 이벤트의 추상적 가능성을 구현하기 때문에 가능성을 비트 수에 어떻게 매핑합니까?섀넌은 원래 John Tukey가 만든 정보 단위로*bit*라는 용어를 도입했습니다.그렇다면 “비트”란 무엇이며 정보를 측정하는 데 사용하는 이유는 무엇입니까?역사적으로 골동품 송신기는 $0$ 및 $1$의 두 가지 유형의 코드만 보내거나 받을 수 있습니다.실제로 바이너리 인코딩은 모든 최신 디지털 컴퓨터에서 여전히 일반적으로 사용됩니다.이러한 방식으로 모든 정보는 일련의 $0$ 및 $1$로 인코딩됩니다.따라서 길이가 $n$인 일련의 이진수에는 $n$비트의 정보가 포함됩니다. 

이제 코드 계열에 대해 $0$ 또는 $1$가 각각 $\frac{1}{2}$의 확률로 발생한다고 가정합니다.따라서 길이가 $n$인 일련의 코드를 가진 이벤트 $X$은 $\frac{1}{2^n}$의 확률로 발생합니다.동시에 앞서 언급했듯이이 시리즈에는 $n$ 비트의 정보가 포함되어 있습니다.그렇다면 확률 $p$을 비트 수로 전달할 수 있는 수학 함수로 일반화할 수 있을까요?Shannon은 *자기 정보*를 정의하여 답을 제시했습니다. 

$$I(X) = - \log_2 (p),$$

이 이벤트 $X$에 대해 받은 정보의*비트*입니다.이 섹션에서는 항상 밑이 2 로그를 사용한다는 점에 유의하십시오.단순화하기 위해 이 섹션의 나머지 부분에서는 로그 표기법에서 첨자 2를 생략합니다. 즉, $\log(.)$는 항상 $\log_2(.)$을 나타냅니다.예를 들어 코드 “0010"에는 자체 정보가 있습니다. 

$$I(\text{"0010"}) = - \log (p(\text{"0010"})) = - \log \left( \frac{1}{2^4} \right) = 4 \text{ bits}.$$

아래와 같이 자기 정보를 계산할 수 있습니다.그 전에 먼저 이 섹션에서 필요한 패키지를 모두 가져오도록 하겠습니다.

```{.python .input}
from mxnet import np
from mxnet.metric import NegativeLogLikelihood
from mxnet.ndarray import nansum
import random

def self_information(p):
    return -np.log2(p)

self_information(1 / 64)
```

```{.python .input}
#@tab pytorch
import torch
from torch.nn import NLLLoss

def nansum(x):
    # Define nansum, as pytorch doesn't offer it inbuilt.
    return x[~torch.isnan(x)].sum()

def self_information(p):
    return -torch.log2(torch.tensor(p)).item()

self_information(1 / 64)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def log2(x):
    return tf.math.log(x) / tf.math.log(2.)

def nansum(x):
    return tf.reduce_sum(tf.where(tf.math.is_nan(
        x), tf.zeros_like(x), x), axis=-1)

def self_information(p):
    return -log2(tf.constant(p)).numpy()

self_information(1 / 64)
```

## 엔트로피

자체 정보는 단일 이산 사건의 정보만 측정하므로 이산 또는 연속 분포의 확률 변수에 대해 보다 일반화된 측정값이 필요합니다. 

### 동기 부여 엔트로피

우리가 원하는 것을 구체적으로 설명해 보겠습니다.이것은*Shannon 엔트로피의*공리*로 알려진 것에 대한 비공식적 인 진술이 될 것입니다.다음과 같은 상식적인 진술 모음은 우리에게 정보의 고유 한 정의를 강요한다는 것이 밝혀 질 것입니다.이러한 공리의 공식 버전은 여러 다른 공리와 함께 :cite:`Csiszar.2008`에서 찾을 수 있습니다. 

1.  랜덤 변수를 관찰하여 얻는 정보는 요소라고 부르는 것 또는 확률이 0인 추가 요소의 존재 여부에 의존하지 않습니다.
2.  두 랜덤 변수를 관찰하여 얻는 정보는 개별적으로 관찰하여 얻은 정보의 합에 지나지 않습니다.그것들이 독립적이라면 정확히 합계입니다.
3.  (거의) 특정 사건을 관찰할 때 얻은 정보는 (거의) 0입니다.

이 사실이 우리 본문의 범위를 벗어난다는 것을 증명하지만, 이것이 엔트로피가 취해야 할 형태를 독특하게 결정한다는 것을 아는 것이 중요합니다.이것이 허용하는 유일한 모호성은 기본 단위를 선택하는 것입니다. 기본 단위는 단일 공정한 동전 플립으로 제공되는 정보가 1 비트이기 전에 보았던 선택을함으로써 가장 자주 정규화됩니다. 

### 정의

확률 밀도 함수 (p.d.f.) 또는 확률 질량 함수 (p.m.f.) $p(x)$와 함께 확률 분포 $P$를 따르는 확률 변수 $X$의 경우*엔트로피* (또는*섀넌 엔트로피*) 를 통해 예상되는 정보량을 측정합니다. 

$$H(X) = - E_{x \sim P} [\log p(x)].$$
:eqlabel:`eq_ent_def`

구체적으로 말하자면, $X$가 이산형인 경우 $H(X) = - \sum_i p_i \log p_i \text{, where } p_i = P(X_i).$달러입니다. 

그렇지 않으면 $X$가 연속이면 엔트로피를*차등 엔트로피*라고도합니다. 

$$H(X) = - \int_x p(x) \log p(x) \; dx.$$

엔트로피를 아래와 같이 정의할 수 있습니다.

```{.python .input}
def entropy(p):
    entropy = - p * np.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(entropy.as_nd_ndarray())
    return out

entropy(np.array([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab pytorch
def entropy(p):
    entropy = - p * torch.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(entropy)
    return out

entropy(torch.tensor([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab tensorflow
def entropy(p):
    return nansum(- p * log2(p))

entropy(tf.constant([0.1, 0.5, 0.1, 0.3]))
```

### 해석

궁금할 수도 있습니다. : in the entropy definition :eqref:`eq_ent_def`, 왜 우리는 음수 로그에 대한 기대치를 사용합니까?여기에 몇 가지 직관이 있습니다. 

먼저, *로그* 함수 $\log$을 사용하는 이유는 무엇입니까?$p(x) = f_1(x) f_2(x) \ldots, f_n(x)$라고 가정합니다. 여기서 각 성분 함수 $f_i(x)$는 서로 독립적입니다.즉, 각 $f_i(x)$는 $p(x)$에서 얻은 전체 정보에 독립적으로 기여한다는 것을 의미합니다.위에서 논의한 바와 같이 엔트로피 공식을 독립 랜덤 변수보다 가산하기를 원합니다.다행히 $\log$은 확률 분포의 곱을 개별 항의 합으로 자연스럽게 바꿀 수 있습니다. 

다음으로*네거티브* $\log$를 사용하는 이유는 무엇입니까?직관적으로 더 빈번한 이벤트는 일반적인 이벤트보다 비정상적인 사례에서 더 많은 정보를 얻는 경우가 많기 때문에 덜 일반적인 이벤트보다 적은 정보를 포함해야합니다.그러나 $\log$는 확률에 따라 단조롭게 증가하고 $[0, 1]$의 모든 값에 대해 실제로 음수입니다.우리는 사건의 확률과 엔트로피 사이에 단조롭게 감소하는 관계를 구성해야하는데, 이는 이상적으로는 항상 긍정적입니다 (우리가 관찰하는 것은 우리가 알고있는 것을 잊도록 강요해서는 안되기 때문입니다).따라서 $\log$ 함수 앞에 음수 부호를 추가합니다. 

마지막으로, *기대* 함수의 출처는 어디입니까?랜덤 변수 $X$을 가정해 보겠습니다.우리는 자기 정보 ($-\log(p)$) 를 특정 결과를 볼 때 겪는*놀라움*의 양으로 해석할 수 있습니다.실제로 확률이 0에 가까워지면 놀라움은 무한해집니다.마찬가지로 엔트로피를 $X$을 관찰한 결과 평균 놀라움으로 해석할 수 있습니다.예를 들어, 슬롯 머신 시스템이 각각 확률 ${p_1, \ldots, p_k}$을 가진 통계적 독립 기호 ${s_1, \ldots, s_k}$를 방출한다고 가정합니다.그런 다음이 시스템의 엔트로피는 각 출력을 관찰 한 평균 자기 정보와 같습니다. 

$$H(S) = \sum_i {p_i \cdot I(s_i)} = - \sum_i {p_i \cdot \log p_i}.$$

### 엔트로피의 특성

위의 예와 해석을 통해 엔트로피 :eqref:`eq_ent_def`의 다음 속성을 도출 할 수 있습니다.여기서는 $X$을 사건으로, $P$를 $X$의 확률 분포라고 합니다. 

* H (X)\ 게크 0$ for all discrete $X$ (entropy can be negative for continuous $X$).

* 만약 $X \sim P$이 P.D.F. 또는 오후 $p(x)$인 경우, 우리는 새로운 확률 분포 $Q$로 $P$을 추산하려고 합니다.

* $X \sim P$인 경우 $x$는 가능한 모든 결과에 균등하게 분산되는 경우 최대 정보량을 전달합니다.특히, 확률 분포 $P$가 $k$-클래스 $\{p_1, \ldots, p_k \}$과 이산인 경우, 해당 구간에 대한 균등 분포인 경우, $$H(X) \leq \log(k), \text{ with equality if and only if } p_i = \frac{1}{k}, \forall i.$$ If $P$ If $P$ If $P$ If $P$는 엔트로피가 가장 높습니다.

## 상호 정보

이전에는 단일 랜덤 변수 $X$의 엔트로피를 정의했습니다. 한 쌍의 랜덤 변수 $(X, Y)$의 엔트로피는 어떻습니까?우리는 이러한 기술을 다음과 같은 유형의 질문에 답하려는 것으로 생각할 수 있습니다. “$X$와 $Y$에 포함 된 정보는 각각 별도로 비교됩니까?중복된 정보가 있는가, 아니면 모두 고유한 정보인가?” 

다음 논의에서는 항상 $(X, Y)$를 p.d.f. 또는 오후 $p_{X, Y}(x, y)$과 함께 공동 확률 분포 $P$를 따르는 확률 변수 쌍으로 사용하는 반면, $X$ 및 $Y$은 각각 확률 분포 $p_X(x)$ 및 $p_Y(y)$을 따릅니다. 

### 관절 엔트로피

단일 랜덤 변수 :eqref:`eq_ent_def`의 엔트로피와 유사하게, 한 쌍의 랜덤 변수 $(X, Y)$의*관절 엔트로피* $H(X, Y)$을 다음과 같이 정의합니다. 

$$H(X, Y) = −E_{(x, y) \sim P} [\log p_{X, Y}(x, y)]. $$
:eqlabel:`eq_joint_ent_def`

정확히 한편으로 $(X, Y)$가 한 쌍의 이산 확률 변수라면 

$$H(X, Y) = - \sum_{x} \sum_{y} p_{X, Y}(x, y) \log p_{X, Y}(x, y).$$

반면에 $(X, Y)$가 연속 확률 변수 쌍인 경우*미분 관절 엔트로피*를 다음과 같이 정의합니다. 

$$H(X, Y) = - \int_{x, y} p_{X, Y}(x, y) \ \log p_{X, Y}(x, y) \;dx \;dy.$$

:eqref:`eq_joint_ent_def`는 확률 변수 쌍의 총 임의성을 알려주는 것으로 생각할 수 있습니다.극단 쌍으로서 $X = Y$이 두 개의 동일한 확률 변수인 경우 쌍의 정보는 정확히 하나의 정보이며 $H(X, Y) = H(X) = H(Y)$를 갖습니다.다른 극단에서 $X$과 $Y$이 독립적이라면 $H(X, Y) = H(X) + H(Y)$입니다.실제로 우리는 한 쌍의 랜덤 변수에 포함 된 정보가 랜덤 변수의 엔트로피보다 작지 않고 둘 다의 합보다 크지 않다는 것을 항상 알게 될 것입니다. 

$$
H(X), H(Y) \le H(X, Y) \le H(X) + H(Y).
$$

조인트 엔트로피를 처음부터 구현해 보겠습니다.

```{.python .input}
def joint_entropy(p_xy):
    joint_ent = -p_xy * np.log2(p_xy)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(joint_ent.as_nd_ndarray())
    return out

joint_entropy(np.array([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab pytorch
def joint_entropy(p_xy):
    joint_ent = -p_xy * torch.log2(p_xy)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(joint_ent)
    return out

joint_entropy(torch.tensor([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab tensorflow
def joint_entropy(p_xy):
    joint_ent = -p_xy * log2(p_xy)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(joint_ent)
    return out

joint_entropy(tf.constant([[0.1, 0.5], [0.1, 0.3]]))
```

이것은 이전과 동일한*코드*이지만, 이제는 두 확률 변수의 합동 분포에 대해 작업하는 것과 다르게 해석합니다. 

### 조건부 엔트로피

한 쌍의 랜덤 변수에 포함된 정보의 양 위에 정의된 관절 엔트로피입니다.이것은 유용하지만 종종 우리가 신경 쓰지 않는 경우가 있습니다.머신 러닝의 설정을 고려해 보십시오.$X$를 이미지의 픽셀 값을 설명하는 랜덤 변수 (또는 랜덤 변수의 벡터) 로, $Y$를 클래스 레이블 인 랜덤 변수로 사용하겠습니다. $X$에는 실질적인 정보가 포함되어야합니다.그러나 이미지가 표시된 후 $Y$에 포함된 정보는 낮아야 합니다.실제로 숫자 이미지에는 숫자를 읽을 수 없는 경우가 아니면 숫자가 어떤 숫자인지에 대한 정보가 이미 포함되어 있어야 합니다.따라서 정보 이론의 어휘를 계속 확장하려면 다른 조건부 랜덤 변수의 정보 내용에 대해 추론 할 수 있어야합니다. 

확률 이론에서 변수 간의 관계를 측정하기 위해*조건부 확률*의 정의를 보았습니다.이제*조건부 엔트로피* $H(Y \mid X)$를 유사하게 정의하려고 합니다.이것을 다음과 같이 쓸 수 있습니다. 

$$ H(Y \mid X) = - E_{(x, y) \sim P} [\log p(y \mid x)],$$
:eqlabel:`eq_cond_ent_def`

여기서 $p(y \mid x) = \frac{p_{X, Y}(x, y)}{p_X(x)}$는 조건부 확률입니다.구체적으로, $(X, Y)$가 이산 확률 변수의 쌍인 경우 

$$H(Y \mid X) = - \sum_{x} \sum_{y} p(x, y) \log p(y \mid x).$$

$(X, Y)$가 연속 확률 변수 쌍인 경우*미분 조건부 엔트로피*는 다음과 같이 유사하게 정의됩니다. 

$$H(Y \mid X) = - \int_x \int_y p(x, y) \ \log p(y \mid x) \;dx \;dy.$$

*조건부 엔트로피* $H(Y \mid X)$는 엔트로피 $H(X)$ 및 조인트 엔트로피 $H(X, Y)$과 어떤 관련이 있는지 묻는 것이 당연합니다.위의 정의를 사용하면 다음과 같이 깔끔하게 표현할 수 있습니다. 

$$H(Y \mid X) = H(X, Y) - H(X).$$

이것은 직관적인 해석을 가지고 있습니다. $X$ ($H(Y \mid X)$) 에 주어진 $Y$의 정보는 $X$와 $Y$의 정보 ($H(X, Y)$) 에서 $X$에 이미 포함된 정보를 뺀 것과 동일합니다.이것은 $Y$의 정보를 제공하며, 이는 $X$에도 표시되지 않습니다. 

이제 조건부 엔트로피 :eqref:`eq_cond_ent_def`를 처음부터 구현해 보겠습니다.

```{.python .input}
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * np.log2(p_y_given_x)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(cond_ent.as_nd_ndarray())
    return out

conditional_entropy(np.array([[0.1, 0.5], [0.2, 0.3]]), np.array([0.2, 0.8]))
```

```{.python .input}
#@tab pytorch
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * torch.log2(p_y_given_x)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(cond_ent)
    return out

conditional_entropy(torch.tensor([[0.1, 0.5], [0.2, 0.3]]),
                    torch.tensor([0.2, 0.8]))
```

```{.python .input}
#@tab tensorflow
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * log2(p_y_given_x)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(cond_ent)
    return out

conditional_entropy(tf.constant([[0.1, 0.5], [0.2, 0.3]]),
                    tf.constant([0.2, 0.8]))
```

### 상호 정보

랜덤 변수 $(X, Y)$의 이전 설정을 감안할 때 다음과 같이 궁금 할 것입니다. “이제 $Y$에는 포함되어 있지만 $X$에는 포함되지 않은 정보의 양을 알았으므로 $X$와 $Y$ 사이에 얼마나 많은 정보가 공유되는지 비슷하게 물어볼 수 있습니까?”답은 $(X, Y)$의*상호 정보*가 될 것이며, 우리는 $I(X, Y)$로 쓸 것입니다. 

공식적인 정의로 곧바로 뛰어 들기보다는 먼저 이전에 구성한 용어를 기반으로 상호 정보에 대한 표현을 도출하려고 노력하여 직관을 연습합시다.두 랜덤 변수 간에 공유되는 정보를 찾고 싶습니다.이 작업을 시도할 수 있는 한 가지 방법은 $X$와 $Y$에 포함된 모든 정보로 시작한 다음 공유되지 않는 부분을 제거하는 것입니다.$X$와 $Y$에 모두 포함된 정보는 $H(X, Y)$로 기록됩니다.우리는 $X$에 포함되어 있지만 $Y$에는 포함되지 않은 정보와 $Y$에는 포함되어 있지만 $X$에는 포함되지 않은 정보를 빼고 싶습니다.이전 섹션에서 보았 듯이 이것은 각각 $H(X \mid Y)$과 $H(Y \mid X)$에 의해 제공됩니다.따라서 우리는 상호 정보가 

$$
I(X, Y) = H(X, Y) - H(Y \mid X) − H(X \mid Y).
$$

실제로 이것은 상호 정보에 대한 유효한 정의입니다.이 용어의 정의를 확장하여 결합하면 약간의 대수학이 다음과 같음을 보여줍니다. 

$$I(X, Y) = E_{x} E_{y} \left\{ p_{X, Y}(x, y) \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)} \right\}. $$
:eqlabel:`eq_mut_ent_def`

이미지 :numref:`fig_mutual_information`에서 이러한 모든 관계를 요약할 수 있습니다.다음 진술이 모두 $I(X, Y)$와 동일한 이유를 확인하는 것은 직관에 대한 훌륭한 테스트입니다. 

* $ H (X) - H (X\ 중간 Y) $
* $ H (Y) - H (Y\ 중간 X) $
* $H (X) +H (Y) - H (X, Y) $

![Mutual information's relationship with joint entropy and conditional entropy.](../img/mutual-information.svg)
:label:`fig_mutual_information`

여러 면에서 우리는 상호 정보 :eqref:`eq_mut_ent_def`를 :numref:`sec_random_variables`에서 본 상관 계수의 원칙적인 확장으로 생각할 수 있습니다.이를 통해 변수 간의 선형 관계뿐만 아니라 모든 종류의 두 확률 변수 간에 공유되는 최대 정보를 요청할 수 있습니다. 

이제 상호 정보를 처음부터 구현해 보겠습니다.

```{.python .input}
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * np.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(mutual.as_nd_ndarray())
    return out

mutual_information(np.array([[0.1, 0.5], [0.1, 0.3]]),
                   np.array([0.2, 0.8]), np.array([[0.75, 0.25]]))
```

```{.python .input}
#@tab pytorch
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * torch.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(mutual)
    return out

mutual_information(torch.tensor([[0.1, 0.5], [0.1, 0.3]]),
                   torch.tensor([0.2, 0.8]), torch.tensor([[0.75, 0.25]]))
```

```{.python .input}
#@tab tensorflow
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(mutual)
    return out

mutual_information(tf.constant([[0.1, 0.5], [0.1, 0.3]]),
                   tf.constant([0.2, 0.8]), tf.constant([[0.75, 0.25]]))
```

### 상호 정보의 등록 정보

상호 정보 :eqref:`eq_mut_ent_def`의 정의를 암기하는 대신 주목할만한 속성 만 명심하면됩니다. 

* 상호 정보는 대칭입니다. 즉, $I(X, Y) = I(Y, X)$입니다.
* 상호 정보는 음수가 아닙니다. 즉, $I(X, Y) \geq 0$입니다.
* $I(X, Y) = 0$는 $X$와 $Y$이 독립적인 경우에만 해당됩니다.예를 들어, $X$와 $Y$이 독립적 인 경우 $Y$을 알면 $X$에 대한 정보가 제공되지 않으며 그 반대의 경우도 마찬가지이므로 상호 정보는 0입니다.
* 또는 $X$가 $Y$의 반전 가능한 함수인 경우 $Y$과 $X$가 모든 정보를 공유하고 $I(X, Y) = H(Y) = H(X).$달러를 공유합니다.

### 포인트 와이즈 상호 정보

이 장의 시작 부분에서 엔트로피로 작업했을 때 $-\log(p_X(x))$에 대한 해석을 특정 결과에 얼마나 놀랐는지* 설명 할 수있었습니다.상호 정보의 로그 용어와 유사한 해석을 할 수 있으며, 이를 종종*점별 상호 정보*라고합니다. 

$$\mathrm{pmi}(x, y) = \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)}.$$
:eqlabel:`eq_pmi_def`

:eqref:`eq_pmi_def`는 $x$과 $y$의 특정 결과 조합이 독립적인 무작위 결과에 대해 기대하는 것과 비교될 가능성이 얼마나 높거나 적은지 측정하는 것으로 생각할 수 있습니다.크고 긍정적이라면이 두 가지 특정 결과는 무작위 확률에 비해 훨씬 더 자주 발생합니다 (* 참고*: 분모는 $p_X(x) p_Y(y)$이며 두 결과가 독립적 일 확률 임). 크고 음수이면 멀리 발생하는 두 결과를 나타냅니다.우연히 우리가 기대하는 것보다 적습니다. 

이를 통해 상호 정보 :eqref:`eq_mut_ent_def`를 독립적일 때 기대하는 것과 비교하여 두 가지 결과가 함께 발생하는 것을보고 놀랐던 평균 금액으로 해석 할 수 있습니다. 

### 상호 정보의 적용

상호 정보는 순수한 정의에서 약간 추상적일 수 있습니다. 그렇다면 기계 학습과 어떤 관련이 있습니까?자연어 처리에서 가장 어려운 문제 중 하나는*모호성 해결* 또는 문맥에서 단어의 의미가 명확하지 않은 문제입니다.예를 들어, 최근 뉴스의 헤드 라인에 “아마존이 불타고 있습니다”라고 보도했습니다.아마존에 건물이 불타고 있는지 아니면 아마존 열대 우림에 불이 났는지 궁금 할 것입니다. 

이 경우 상호 정보는 이러한 모호성을 해결하는 데 도움이 될 수 있습니다.먼저 전자 상거래, 기술 및 온라인과 같이 Amazon과 상호 정보가 상대적으로 큰 단어 그룹을 찾습니다.둘째, 비, 숲 및 열대와 같이 아마존 열대 우림과 상대적으로 큰 상호 정보를 가진 또 다른 단어 그룹을 찾습니다.“Amazon”을 명확하게해야 할 때 Amazon이라는 단어의 맥락에서 어떤 그룹이 더 많이 발생하는지 비교할 수 있습니다.이 경우 기사는 숲에 대해 설명하고 맥락을 명확하게 설명합니다. 

## 쿨백-라이블러 다이버전스

:numref:`sec_linear-algebra`에서 논의한 바와 같이 규범을 사용하여 모든 차원의 공간에서 두 점 사이의 거리를 측정 할 수 있습니다.확률 분포로 비슷한 작업을 할 수 있기를 바랍니다.이 문제를 해결하는 방법에는 여러 가지가 있지만 정보 이론은 가장 좋은 방법 중 하나를 제공합니다.이제 두 분포가 서로 가깝는지 여부를 측정하는 방법을 제공하는*Kullback-Leibler (KL) 발산*을 살펴봅니다. 

### 정의

확률 분포 $P$을 p.d.f. 또는 오후 $p(x)$와 함께 따르는 확률 변수 $X$가 주어지고, 우리는 $P$을 p.d.f. 또는 오후 $q(x)$을 가진 다른 확률 분포 $Q$로 추정합니다.그런 다음 $P$과 $Q$ 사이의*쿨백-라이블러 (KL) 발산* (또는*상대 엔트로피*) 은 다음과 같습니다. 

$$D_{\mathrm{KL}}(P\|Q) = E_{x \sim P} \left[ \log \frac{p(x)}{q(x)} \right].$$
:eqlabel:`eq_kl_def`

점별 상호 정보 :eqref:`eq_pmi_def`와 마찬가지로 로그 용어에 대한 해석을 다시 제공 할 수 있습니다. $-\log \frac{q(x)}{p(x)} = -\log(q(x)) - (-\log(p(x)))$는 $Q$에서 기대하는 것보다 $P$에서 훨씬 더 자주 $x$을 볼 경우 크고 긍정적이며 결과가 예상보다 훨씬 적으면 크고 부정적입니다.이런 식으로 우리는 기준 분포에서 관찰하는 것이 얼마나 놀랐는지에 비해 결과를 관찰하는 데*상대적* 놀라움으로 해석 할 수 있습니다. 

KL 발산을 처음부터 구현해 보겠습니다.

```{.python .input}
def kl_divergence(p, q):
    kl = p * np.log2(p / q)
    out = nansum(kl.as_nd_ndarray())
    return out.abs().asscalar()
```

```{.python .input}
#@tab pytorch
def kl_divergence(p, q):
    kl = p * torch.log2(p / q)
    out = nansum(kl)
    return out.abs().item()
```

```{.python .input}
#@tab tensorflow
def kl_divergence(p, q):
    kl = p * log2(p / q)
    out = nansum(kl)
    return tf.abs(out).numpy()
```

### KL 다이버전스 등록

KL 발산 :eqref:`eq_kl_def`의 몇 가지 특성을 살펴 보겠습니다. 

* KL 발산은 대칭이 아닙니다. 즉, $P,Q$달러가 되는 $P,Q$달러가 있습니다.
* KL 발산은 음수가 아닙니다. 즉, $D_{\mathrm{KL}}(P\|Q) \geq 0.$$ Note that the equality holds only when $P = Q$입니다.
* $p(x) > 0$과 $q(x) = 0$과 같은 $x$가 존재한다면 $D_{\mathrm{KL}}(P\|Q) = \infty$가 있습니다.
* KL 발산과 상호 정보 사이에는 밀접한 관계가 있습니다.:numref:`fig_mutual_information`에 표시된 관계 외에도 $I(X, Y)$는 다음 용어와 수치적으로 동일합니다.
    1. $D_{\mathrm{KL}}(P(X, Y)  \ \| \ P(X)P(Y))$;
    1. $E_Y \{ D_{\mathrm{KL}}(P(X \mid Y) \ \| \ P(X)) \}$;
    1. $E_X \{ D_{\mathrm{KL}}(P(Y \mid X) \ \| \ P(Y)) \}$.

  첫 번째 항의 경우 상호 정보를 $P(X, Y)$와 $P(X)$ 및 $P(Y)$의 곱 사이의 KL 발산으로 해석하므로 접합 분포가 독립적일 경우 분포와 얼마나 다른지를 측정한 것입니다.두 번째 용어의 경우 상호 정보는 $X$ 분포의 가치를 학습하여 발생하는 $Y$에 대한 불확실성의 평균 감소를 알려줍니다.세 번째 학기와 비슷합니다. 

### 예시

비대칭성을 명시적으로 보기 위해 장난감 예제를 살펴보겠습니다. 

먼저 길이가 $10,000$인 세 개의 텐서를 생성하고 정렬해 보겠습니다. 즉, 정규 분포 $N(0, 1)$을 따르는 객관적인 텐서 $p$과 각각 정규 분포 $N(-1, 1)$ 및 $N(1, 1)$를 따르는 두 개의 후보 텐서 $q_1$ 및 $q_2$입니다.

```{.python .input}
random.seed(1)

nd_len = 10000
p = np.random.normal(loc=0, scale=1, size=(nd_len, ))
q1 = np.random.normal(loc=-1, scale=1, size=(nd_len, ))
q2 = np.random.normal(loc=1, scale=1, size=(nd_len, ))

p = np.array(sorted(p.asnumpy()))
q1 = np.array(sorted(q1.asnumpy()))
q2 = np.array(sorted(q2.asnumpy()))
```

```{.python .input}
#@tab pytorch
torch.manual_seed(1)

tensor_len = 10000
p = torch.normal(0, 1, (tensor_len, ))
q1 = torch.normal(-1, 1, (tensor_len, ))
q2 = torch.normal(1, 1, (tensor_len, ))

p = torch.sort(p)[0]
q1 = torch.sort(q1)[0]
q2 = torch.sort(q2)[0]
```

```{.python .input}
#@tab tensorflow
tensor_len = 10000
p = tf.random.normal((tensor_len, ), 0, 1)
q1 = tf.random.normal((tensor_len, ), -1, 1)
q2 = tf.random.normal((tensor_len, ), 1, 1)

p = tf.sort(p)
q1 = tf.sort(q1)
q2 = tf.sort(q2)
```

$q_1$과 $q_2$는 y축 (즉, $x=0$) 에 대해 대칭이기 때문에 $D_{\mathrm{KL}}(p\|q_1)$과 $D_{\mathrm{KL}}(p\|q_2)$ 사이에서 유사한 KL 발산 값이 나타날 것으로 예상됩니다.아래에서 볼 수 있듯이 $D_{\mathrm{KL}}(p\|q_1)$과 $D_{\mathrm{KL}}(p\|q_2)$ 사이에는 3% 미만의 할인이 있습니다.

```{.python .input}
#@tab all
kl_pq1 = kl_divergence(p, q1)
kl_pq2 = kl_divergence(p, q2)
similar_percentage = abs(kl_pq1 - kl_pq2) / ((kl_pq1 + kl_pq2) / 2) * 100

kl_pq1, kl_pq2, similar_percentage
```

반대로 $D_{\mathrm{KL}}(q_2 \|p)$ 및 $D_{\mathrm{KL}}(p \| q_2)$가 많이 할인되었으며 아래 그림과 같이 약 40% 할인되었음을 알 수 있습니다.

```{.python .input}
#@tab all
kl_q2p = kl_divergence(q2, p)
differ_percentage = abs(kl_q2p - kl_pq2) / ((kl_q2p + kl_pq2) / 2) * 100

kl_q2p, differ_percentage
```

## 크로스 엔트로피

딥 러닝에서 정보 이론의 적용에 대해 궁금한 점이 있다면 여기에 간단한 예가 있습니다.확률 분포 $p(x)$를 사용하여 실제 분포 $P$를 정의하고 확률 분포 $q(x)$을 사용하여 추정된 분포 $Q$을 정의하고 이 섹션의 나머지 부분에서 사용할 것입니다. 

주어진 $n$ 데이터 예제 {$x_1, \ldots, x_n$} 를 기반으로 이진 분류 문제를 풀어야 한다고 가정해 보겠습니다.$1$ 및 $0$을 각각 양수 및 음수 클래스 레이블 $y_i$로 인코딩하고 신경망이 $\theta$에 의해 매개 변수화된다고 가정합니다.$\hat{y}_i= p_{\theta}(y_i \mid x_i)$이 되도록 최상의 $\theta$를 찾는 것을 목표로 한다면 :numref:`sec_maximum_likelihood`에서 볼 수 있듯이 최대 로그 우도 접근법을 적용하는 것이 당연합니다.구체적으로 말하면, 실제 레이블 $y_i$ 및 예측값 $\hat{y}_i= p_{\theta}(y_i \mid x_i)$의 경우 양수로 분류될 확률은 $\pi_i= p_{\theta}(y_i = 1 \mid x_i)$입니다.따라서 로그 우도 함수는 다음과 같습니다. 

$$
\begin{aligned}
l(\theta) &= \log L(\theta) \\
  &= \log \prod_{i=1}^n \pi_i^{y_i} (1 - \pi_i)^{1 - y_i} \\
  &= \sum_{i=1}^n y_i \log(\pi_i) + (1 - y_i) \log (1 - \pi_i). \\
\end{aligned}
$$

로그 우도 함수 $l(\theta)$을 최대화하는 것은 $- l(\theta)$를 최소화하는 것과 동일하므로 여기에서 최상의 $\theta$을 찾을 수 있습니다.위의 손실을 모든 분포에 일반화하기 위해 $-l(\theta)$을*교차 엔트로피 손실* $\mathrm{CE}(y, \hat{y})$라고도 합니다. 여기서 $y$는 실제 분포 $P$를 따르고 $\hat{y}$은 추정 분포 $Q$을 따릅니다. 

이 모든 것은 최대 가능성의 관점에서 작업함으로써 파생되었습니다.그러나 자세히 살펴보면 $\log(\pi_i)$와 같은 용어가 계산에 입력되었음을 알 수 있습니다. 이는 정보 이론적 관점에서 표현을 이해할 수 있다는 확실한 표시입니다. 

### 공식 정의

KL 발산과 마찬가지로 랜덤 변수 $X$의 경우*교차 엔트로피*를 통해 추정 분포 $Q$와 실제 분포 $P$ 사이의 발산을 측정 할 수도 있습니다. 

$$\mathrm{CE}(P, Q) = - E_{x \sim P} [\log(q(x))].$$
:eqlabel:`eq_ce_def`

위에서 논의한 엔트로피의 특성을 사용함으로써, 우리는 또한 엔트로피 $H(P)$와 $P$와 $Q$ 사이의 KL 발산의 합으로 해석 할 수 있습니다. 

$$\mathrm{CE} (P, Q) = H(P) + D_{\mathrm{KL}}(P\|Q).$$

다음과 같이 교차 엔트로피 손실을 구현할 수 있습니다.

```{.python .input}
def cross_entropy(y_hat, y):
    ce = -np.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab pytorch
def cross_entropy(y_hat, y):
    ce = -torch.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    # `tf.gather_nd` is used to select specific indices of a tensor.
    ce = -tf.math.log(tf.gather_nd(y_hat, indices = [[i, j] for i, j in zip(
        range(len(y_hat)), y)]))
    return tf.reduce_mean(ce).numpy()
```

이제 레이블과 예측에 대해 두 개의 텐서를 정의하고 이들의 교차 엔트로피 손실을 계산합니다.

```{.python .input}
labels = np.array([0, 2])
preds = np.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab pytorch
labels = torch.tensor([0, 2])
preds = torch.tensor([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab tensorflow
labels = tf.constant([0, 2])
preds = tf.constant([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

### 등록 정보

이 섹션의 시작 부분에서 언급했듯이 교차 엔트로피 :eqref:`eq_ce_def`를 사용하여 최적화 문제에서 손실 함수를 정의할 수 있습니다.다음은 동일하다는 것이 밝혀졌습니다. 

1. 분포 $P$에 대한 예측 확률 $Q$를 최대화합니다 (즉, $E_ {x).
\ 심 P} [\ 로그 (q (x))] $);
1. 교차 엔트로피 $\mathrm{CE} (P, Q)$를 최소화합니다.
1. KL 다이버전스 $D_{\mathrm{KL}}(P\|Q)$를 최소화합니다.

교차 엔트로피의 정의는 실제 데이터 $H(P)$의 엔트로피가 일정하다면 목적 2와 목적 3 사이의 동등한 관계를 간접적으로 증명합니다. 

### 다중 클래스 분류의 목적 함수로서의 교차 엔트로피

교차 엔트로피 손실 $\mathrm{CE}$를 갖는 분류 목적 함수에 대해 자세히 살펴보면 $\mathrm{CE}$를 최소화하는 것이 로그 우도 함수 $L$를 최대화하는 것과 동일하다는 것을 알 수 있습니다. 

먼저 $n$ 예제가 포함된 데이터셋이 제공되고 $k$-클래스로 분류할 수 있다고 가정합니다.각 데이터 예제 $i$에 대해 $k$ 클래스 레이블 $\mathbf{y}_i = (y_{i1}, \ldots, y_{ik})$를*원핫 인코딩*으로 나타냅니다.구체적으로 말하자면, 예제 $i$가 클래스 $j$에 속하는 경우 $j$번째 항목을 $1$으로 설정하고 다른 모든 구성 요소를 $0$로 설정합니다. 

$$ y_{ij} = \begin{cases}1 & j \in J; \\ 0 &\text{otherwise.}\end{cases}$$

예를 들어, 다중 클래스 분류 문제에 $A$, $B$ 및 $C$의 세 가지 클래스가 포함되어 있는 경우 레이블 $\mathbf{y}_i$를 {$A: (1, 0, 0); B: (0, 1, 0); C: (0, 0, 1)$} 으로 인코딩할 수 있습니다. 

신경망이 $\theta$에 의해 매개변수화되어 있다고 가정합니다.실제 레이블 벡터 $\mathbf{y}_i$ 및 예측의 경우 $\hat{\mathbf{y}}_i= p_{\theta}(\mathbf{y}_i \mid \mathbf{x}_i) = \sum_{j=1}^k y_{ij} p_{\theta} (y_{ij}  \mid  \mathbf{x}_i).$달러 

따라서*교차 엔트로피 손실*은 

$$
\mathrm{CE}(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{i=1}^n \mathbf{y}_i \log \hat{\mathbf{y}}_i
 = - \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)}.\\
$$

반면에 최대우도 추정을 통해 문제에 접근할 수도 있습니다.먼저 $k$급 멀티울리 분포를 빠르게 소개하겠습니다.Bernoulli 분포를 이진 클래스에서 다중 클래스로 확장 한 것입니다.랜덤 변수 $\mathbf{z} = (z_{1}, \ldots, z_{k})$이 확률이 $\mathbf{p} =$ ($p_{1}, \ldots, p_{k}$) 인 $k$-클래스*멀티울리 분포*를 따르는 경우, 즉 $p(\mathbf{z}) = p(z_1, \ldots, z_k) = \mathrm{Multi} (p_1, \ldots, p_k), \text{ where } \sum_{i=1}^k p_i = 1,$$ then the joint probability mass function(p.m.f.) of $\ mathbf {z} $는 $\mathbf{p}^\mathbf{z} = \prod_{j=1}^k p_{j}^{z_{j}}.$달러입니다. 

각 데이터 예제의 레이블인 $\mathbf{y}_i$은 확률이 $\boldsymbol{\pi} =$ ($\pi_{1}, \ldots, \pi_{k}$) 인 $k$ 클래스 멀티울리 분포를 따르고 있음을 알 수 있습니다.따라서 각 데이터 예 $\mathbf{y}_i$의 합동 p.m.f. 는 $\mathbf{\pi}^{\mathbf{y}_i} = \prod_{j=1}^k \pi_{j}^{y_{ij}}.$입니다. 따라서 로그 우도 함수는 다음과 같습니다. 

$$
\begin{aligned}
l(\theta)
 = \log L(\theta)
 = \log \prod_{i=1}^n \boldsymbol{\pi}^{\mathbf{y}_i}
 = \log \prod_{i=1}^n \prod_{j=1}^k \pi_{j}^{y_{ij}}
 = \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{\pi_{j}}.\\
\end{aligned}
$$

최대 가능도 추정에서 $\pi_{j} = p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)$을 사용하여 목적 함수 $l(\theta)$를 최대화합니다.따라서 모든 다중 클래스 분류에 대해 위의 로그 우도 함수 $l(\theta)$를 최대화하는 것은 CE 손실 $\mathrm{CE}(y, \hat{y})$를 최소화하는 것과 같습니다. 

위의 증거를 테스트하기 위해 내장 측정값 `NegativeLogLikelihood`를 적용해 보겠습니다.이전 예제와 동일한 `labels` 및 `preds`를 사용하면 이전 예제와 동일한 수치 손실을 소수점 이하 5자리까지 얻을 수 있습니다.

```{.python .input}
nll_loss = NegativeLogLikelihood()
nll_loss.update(labels.as_nd_ndarray(), preds.as_nd_ndarray())
nll_loss.get()
```

```{.python .input}
#@tab pytorch
# Implementation of cross-entropy loss in PyTorch combines `nn.LogSoftmax()`
# and `nn.NLLLoss()`
nll_loss = NLLLoss()
loss = nll_loss(torch.log(preds), labels)
loss
```

```{.python .input}
#@tab tensorflow
def nll_loss(y_hat, y):
    # Convert labels to one-hot vectors.
    y = tf.keras.utils.to_categorical(y, num_classes= y_hat.shape[1])
    # We will not calculate negative log-likelihood from the definition.
    # Rather, we will follow a circular argument. Because NLL is same as
    # `cross_entropy`, if we calculate cross_entropy that would give us NLL
    cross_entropy = tf.keras.losses.CategoricalCrossentropy(
        from_logits = True, reduction = tf.keras.losses.Reduction.NONE)
    return tf.reduce_mean(cross_entropy(y, y_hat)).numpy()

loss = nll_loss(tf.math.log(preds), labels)
loss
```

## 요약

* 정보 이론은 정보의 인코딩, 디코딩, 전송 및 조작에 관한 연구 분야입니다.
* 엔트로피는 서로 다른 신호에 표시되는 정보의 양을 측정하는 단위입니다.
* KL 발산은 두 분포 간의 발산을 측정할 수도 있습니다.
* 교차 엔트로피는 다중 클래스 분류의 목적 함수로 볼 수 있습니다.교차 엔트로피 손실을 최소화하는 것은 로그 우도 함수를 최대화하는 것과 같습니다.

## 연습문제

1. 첫 번째 섹션의 카드 예제에 실제로 청구된 엔트로피가 있는지 확인합니다.
1. KL 발산 $D(p\|q)$이 모든 분포 $p$ 및 $q$에 대해 음수가 아님을 보여줍니다.힌트: 젠슨의 부등식을 사용하십시오. 즉, $-\log x$가 볼록 함수라는 사실을 사용하십시오.
1. 몇 가지 데이터 소스에서 엔트로피를 계산해 보겠습니다.
    * 타자기에서 원숭이가 생성한 출력을 보고 있다고 가정해 보겠습니다.원숭이는 타자기의 $44$ 키 중 하나를 무작위로 누릅니다 (아직 특수 키나 Shift 키를 발견하지 못했다고 가정 할 수 있음).캐릭터당 몇 비트의 임의성을 관찰하십니까?
    * 원숭이가 불만스러워서 술에 취한 조판자로 바꿨습니다.일관적이지는 않지만 단어를 생성 할 수 있습니다.대신 $2,000$단어의 어휘 중에서 임의의 단어를 선택합니다.단어의 평균 길이가 영어로 $4.5$자라고 가정해 보겠습니다.캐릭터당 몇 비트의 임의성을 지금 관찰하고 있습니까?
    * 여전히 결과에 만족하지 못하면 조판자를 고품질 언어 모델로 대체합니다.언어 모델은 현재 단어당 $15$포인트의 낮은 난처함을 얻을 수 있습니다.언어 모델의 문자*perplexity*는 확률 집합의 기하 평균의 역으로 정의되며, 각 확률은 단어의 문자에 해당합니다.구체적으로 말하면, 주어진 단어의 길이가 $l$이면 $\mathrm{PPL}(\text{word}) = \left[\prod_i p(\text{character}_i)\right]^{ -\frac{1}{l}} = \exp \left[ - \frac{1}{l} \sum_i{\log p(\text{character}_i)} \right].$ 테스트 단어에 4.5 글자가 있다고 가정하면 문자당 몇 비트의 임의성을 지금 관찰합니까?
1. $I(X, Y) = H(X) - H(X|Y)$의 이유를 직관적으로 설명하십시오.그런 다음 접합 분포와 관련하여 양쪽을 기대로 표현하여 이것이 사실임을 보여줍니다.
1. 두 가우스 분포 $\mathcal{N}(\mu_1, \sigma_1^2)$와 $\mathcal{N}(\mu_2, \sigma_2^2)$ 사이의 KL 다이버전스는 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/420)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1105)
:end_tab:
