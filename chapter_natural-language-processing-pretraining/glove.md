# 글로벌 벡터를 사용한 단어 임베딩 (GLOVE)
:label:`sec_glove`

컨텍스트 창 내에서 단어-단어 동시 발생은 풍부한 의미 정보를 전달할 수 있습니다.예를 들어, 큰 코퍼스 단어에서 “고체”는 “증기”보다 “얼음”과 함께 발생할 가능성이 더 높지만 “가스”라는 단어는 “얼음”보다 “증기”와 더 자주 발생합니다.또한 이러한 동시 발생에 대한 글로벌 코퍼스 통계를 미리 계산할 수 있습니다. 이로 인해 더 효율적인 훈련이 이루어질 수 있습니다.단어 임베딩을 위해 전체 코퍼스의 통계 정보를 활용하려면 먼저 :numref:`subsec_skip-gram`의 skip-gram 모델을 다시 살펴보고 동시 발생 횟수와 같은 글로벌 코퍼스 통계를 사용하여 해석해 보겠습니다. 

## 글로벌 코퍼스 통계가 포함된 스킵 그램
:label:`subsec_skipgram-global`

스킵 그램 모델에서 단어 $w_i$이 주어진 단어 $w_j$의 조건부 확률 $P(w_j\mid w_i)$을 $q_{ij}$로 나타내면 다음과 같습니다. 

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \text{exp}(\mathbf{u}_k^\top \mathbf{v}_i)},$$

여기서 모든 인덱스에 대해 $i$ 벡터 $\mathbf{v}_i$ 및 $\mathbf{u}_i$은 단어 $w_i$를 각각 중심 단어 및 컨텍스트 단어로 나타내고 $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$은 어휘의 인덱스 집합입니다. 

코퍼스에서 여러 번 발생할 수 있는 단어 $w_i$를 생각해 보십시오.전체 코퍼스에서 $w_i$를 중심 단어로 사용하는 모든 컨텍스트 단어는 동일한 요소의 여러 인스턴스를 허용하는*다중 집합* $\mathcal{C}_i$의 단어 인덱스를 형성합니다.모든 요소에 대해 인스턴스 수를*다중성*이라고 합니다.예를 들어, 단어 $w_i$가 코퍼스에서 두 번 발생하고 두 컨텍스트 창에서 $w_i$를 중심 단어로 사용하는 컨텍스트 단어의 인덱스가 $k, j, m, k$ 및 $k, l, k, j$라고 가정합니다.따라서 멀티세트 $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$이며, 여기서 요소 $j, k, l, m$의 다중도는 각각 2, 4, 1, 1입니다. 

이제 멀티세트 $\mathcal{C}_i$에서 요소 $j$의 다중도를 $x_{ij}$로 표시해 보겠습니다.전체 코퍼스의 동일한 컨텍스트 창에 있는 단어 $w_j$ (컨텍스트 단어) 와 단어 $w_i$ (가운데 단어) 의 전역 동시 발생 횟수입니다.이러한 전역 코퍼스 통계량을 사용하면 skip-gram 모델의 손실 함수는 다음과 같습니다. 

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-x_ij`

또한 $x_i$로 컨텍스트 창의 모든 컨텍스트 단어의 수를 나타냅니다. 여기서 $w_i$은 중심 단어로 발생하며 이는 $|\mathcal{C}_i|$와 같습니다.주어진 중심 단어 $w_i$, :eqref:`eq_skipgram-x_ij`가 주어진 컨텍스트 단어 $w_j$를 생성하기 위한 조건부 확률 $x_{ij}/x_i$으로 지정하면 다음과 같이 다시 작성할 수 있습니다. 

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-p_ij`

:eqref:`eq_skipgram-p_ij`에서 $-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$은 글로벌 코퍼스 통계량의 조건부 분포 $p_{ij}$과 모델 예측의 조건부 분포 $q_{ij}$의 교차 엔트로피를 계산합니다.이 손실은 위에서 설명한 것처럼 $x_i$에 의해 가중치가 부여됩니다.:eqref:`eq_skipgram-p_ij`에서 손실 함수를 최소화하면 예측된 조건부 분포가 전역 코퍼스 통계량의 조건부 분포에 가까워질 수 있습니다. 

확률 분포 사이의 거리를 측정하는 데 일반적으로 사용되지만 교차 엔트로피 손실 함수는 여기서는 좋은 선택이 아닐 수 있습니다.한편으로 :numref:`sec_approx_train`에서 언급했듯이 $q_{ij}$를 올바르게 정규화하는 데 드는 비용은 전체 어휘에 대한 합계를 산출하므로 계산 비용이 많이들 수 있습니다.반면에 큰 코퍼스에서 발생하는 많은 희귀 사건은 종종 너무 많은 가중치로 할당되는 교차 엔트로피 손실로 모델링됩니다. 

## 글러브 모델

이를 고려하여 *장갑* 모델은 제곱 손실 :cite:`Pennington.Socher.Manning.2014`를 기준으로 스킵그램 모델을 세 가지 변경합니다. 

1. 변수 $p'_{ij}=x_{ij}$ 및 $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$를 사용하십시오. 
확률 분포가 아니며 두 분포의 로그를 취하므로 손실 제곱 항은 $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$입니다.
2. 각 단어 $w_i$에 대해 두 개의 스칼라 모델 모수, 즉 중심 단어 편향 $b_i$과 문맥 단어 편향 $c_i$를 추가합니다.
3. 각 손실 항의 가중치를 가중치 함수 $h(x_{ij})$로 바꿉니다. 여기서 $h(x)$은 $[0, 1]$의 구간에서 증가하고 있습니다.

모든 것을 종합하여 GLOve를 훈련하면 다음과 같은 손실 기능을 최소화하는 것입니다. 

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$
:eqlabel:`eq_glove-loss`

가중치 함수의 경우 권장되는 선택 사항은 $h(x) = (x/c) ^\alpha$ (예: $\alpha = 0.75$) 인 경우 $h(x) = (x/c) ^\alpha$ (예: $\alpha = 0.75$) 이고, 그렇지 않으면 $h(x) = 1$입니다.이 경우 $h(0)=0$이므로 계산 효율성을 위해 $x_{ij}=0$에 대한 손실 제곱 항을 생략할 수 있습니다.예를 들어, 훈련에 미니배치 확률적 경사하강법을 사용하는 경우, 각 반복에서*0이 아닌* $x_{ij}$의 미니배치를 무작위로 샘플링하여 기울기를 계산하고 모델 파라미터를 업데이트합니다.이러한 0이 아닌 $x_{ij}$는 미리 계산된 글로벌 코퍼스 통계입니다. 따라서 이 모델은*글로벌 벡터*에 대해 GloVE라고 합니다. 

단어 $w_i$이 단어 $w_j$의 컨텍스트 창에 나타나면*그 반대*라는 점을 강조해야 합니다.따라서 $x_{ij}=x_{ji}$입니다.비대칭 조건부 확률 $p_{ij}$에 맞는 워드2벡과 달리 글로브는 대칭 $\log \, x_{ij}$을 적합합니다.따라서 모든 단어의 중심 단어 벡터와 문맥 단어 벡터는 GLOVE 모델에서 수학적으로 동일합니다.그러나 실제로는 초기화 값이 다르기 때문에 훈련 후에도 동일한 단어가 여전히 이 두 벡터에서 다른 값을 얻을 수 있습니다. GLOve는 이를 출력 벡터로 합산합니다. 

## 동시 발생 확률의 비율에서 GLOVE 해석하기

GLOve 모델을 다른 관점에서 해석할 수도 있습니다.:numref:`subsec_skipgram-global`에서 동일한 표기법을 사용하여 $p_{ij} \stackrel{\mathrm{def}}{=} P(w_j \mid w_i)$을 코퍼스의 중심 단어로 주어진 컨텍스트 단어 $w_j$을 생성할 조건부 확률로 설정합니다. :numref:`tab_glove`는 “얼음”과 “증기”라는 단어가 주어진 몇 가지 동시 발생 확률과 큰 코퍼스의 통계를 기반으로 한 비율을 나열합니다. 

:Word-word co-occurrence probabilities and their ratios from a large corpus (adapted from Table 1 in :cite:`Pennington.Socher.Manning.2014`:) 

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \text{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\text{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|
:label:`tab_glove`

:numref:`tab_glove`에서 다음을 관찰 할 수 있습니다. 

* $w_k=\text{solid}$와 같이 “얼음”과 관련이 있지만 “증기”와 관련이 없는 단어 $w_k$의 경우 8.9와 같이 공발생 확률의 비율이 더 클 것으로 예상됩니다.
* $w_k=\text{gas}$와 같이 “증기”와 관련이 있지만 “얼음”과 관련이 없는 단어 $w_k$의 경우 0.085와 같이 공발생 확률의 비율이 더 작을 것으로 예상됩니다.
* $w_k=\text{water}$와 같이 “얼음”과 “증기”와 관련된 단어 $w_k$의 경우 1.36과 같이 1에 가까운 동시 발생 확률의 비율이 예상됩니다.
* $w_k=\text{fashion}$와 같이 “얼음”과 “증기”와 관련이 없는 단어 $w_k$의 경우 0.96과 같이 1에 가까운 동시 발생 확률의 비율이 예상됩니다.

동시 발생 확률의 비율은 단어 간의 관계를 직관적으로 표현할 수 있음을 알 수 있습니다.따라서 이 비율에 맞게 세 개의 워드 벡터로 구성된 함수를 설계할 수 있습니다.$w_i$이 중심 단어이고 $w_j$ 및 $w_k$가 문맥 단어인 동시 발생 확률 ${p_{ij}}/{p_{ik}}$의 비율에 대해 일부 함수 $f$을 사용하여 이 비율을 피팅하려고 합니다. 

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$
:eqlabel:`eq_glove-f`

$f$에 대해 가능한 많은 설계 중에서 다음에서만 합리적인 선택을 선택합니다.동시 발생 확률의 비율이 스칼라이므로 $f$이 스칼라 함수 (예: $f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$) 여야 합니다.:eqref:`eq_glove-f`에서 단어 인덱스 $j$ 및 $k$을 전환하면 $f(x)f(-x)=1$를 보유해야하므로 한 가지 가능성은 $f(x)=\exp(x)$입니다. 즉,  

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

이제 $\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$를 선택하겠습니다. 여기서 $\alpha$는 상수입니다.$p_{ij}=x_{ij}/x_i$ 이후 양쪽에서 로그를 취한 후 $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$을 얻습니다.중앙 단어 편향 $b_i$ 및 문맥 단어 편향 $c_j$와 같은 추가 편향 용어를 $- \log\, \alpha + \log\, x_i$에 맞출 수 있습니다. 

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$
:eqlabel:`eq_glove-square`

가중치로 :eqref:`eq_glove-square`의 제곱 오차를 측정하면 :eqref:`eq_glove-loss`의 글러브 손실 함수가 얻어집니다. 

## 요약

* 스킵 그램 모델은 단어-단어 동시 발생 횟수와 같은 글로벌 코퍼스 통계를 사용하여 해석할 수 있습니다.
* 교차 엔트로피 손실은 특히 큰 코퍼스의 경우 두 확률 분포의 차이를 측정하는 데 적합하지 않을 수 있습니다.GloVe는 손실 제곱을 사용하여 미리 계산된 글로벌 코퍼스 통계를 피팅합니다.
* 중심 단어 벡터와 문맥 단어 벡터는 GLOve의 모든 단어에 대해 수학적으로 동일합니다.
* GLOve는 단어-단어 동시 발생 확률의 비율로 해석할 수 있습니다.

## 연습문제

1. 단어 $w_i$과 $w_j$가 동일한 컨텍스트 창에서 함께 발생하는 경우 텍스트 시퀀스에서 거리를 사용하여 조건부 확률 $p_{ij}$을 계산하는 방법을 어떻게 다시 설계 할 수 있습니까?힌트: see Section 4.2 of the GloVe paper :cite:`Pennington.Socher.Manning.2014`.
1. 어떤 단어에서든 중심 단어 바이어스와 컨텍스트 단어 바이어스가 GLOV에서 수학적으로 동일합니까?왜요?

[Discussions](https://discuss.d2l.ai/t/385)
