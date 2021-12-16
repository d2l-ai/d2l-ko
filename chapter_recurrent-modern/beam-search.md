# 빔 검색
:label:`sec_beam-search`

:numref:`sec_seq2seq`에서는 특수한 시퀀스 끝 “<eos>" 토큰이 예측될 때까지 토큰별로 출력 시퀀스 토큰을 예측했습니다.이 섹션에서는 이*욕심 많은 검색* 전략을 공식화하고 문제를 살펴본 다음 이 전략을 다른 대안과 비교합니다.
*철저한 검색* 및*빔 검색*.

탐욕스러운 검색을 공식적으로 소개하기 전에 :numref:`sec_seq2seq`와 동일한 수학 표기법을 사용하여 검색 문제를 공식화하겠습니다.임의의 시간 스텝 ($t'$) 에서, 디코더 출력 ($y_{t'}$) 의 확률은 $t'$ 이전의 출력 서브시퀀스 ($y_1, \ldots, y_{t'-1}$) 및 입력 시퀀스의 정보를 인코딩하는 컨텍스트 변수 ($\mathbf{c}$) 에 조건부이다.계산 비용을 정량화하려면 <eos>출력 어휘를 $\mathcal{Y}$ (“포함) 으로 표시하십시오.따라서 이 어휘 집합의 카디널리티 $\left|\mathcal{Y}\right|$은 어휘 크기입니다.출력 시퀀스의 최대 토큰 수를 $T'$로 지정해 보겠습니다.결과적으로 우리의 목표는 가능한 모든 $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ 출력 시퀀스에서 이상적인 출력을 검색하는 것입니다.물론 이러한 모든 출력 시퀀스의 경우 “<eos>“을 포함한 부분과 뒤의 부분은 실제 출력에서 삭제됩니다. 

## 욕심 검색

먼저 간단한 전략, *욕심 많은 검색*을 살펴보겠습니다.이 전략은 :numref:`sec_seq2seq`에서 시퀀스를 예측하는 데 사용되었습니다.욕심 많은 검색에서 출력 시퀀스의 $t'$ 단계 언제든지 $\mathcal{Y}$에서 조건부 확률이 가장 높은 토큰을 검색합니다. 즉,  

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$

출력으로 사용할 수 있습니다.“<eos>“이 출력되거나 출력 시퀀스가 최대 길이 $T'$에 도달하면 출력 시퀀스가 완료됩니다. 

그렇다면 욕심 많은 검색으로 무엇이 잘못 될 수 있을까요?실제로*최적 시퀀스*는 최대 $\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$의 출력 시퀀스여야 하며, 이는 입력 시퀀스를 기반으로 출력 시퀀스를 생성할 조건부 확률입니다.불행히도 욕심 많은 검색으로 최적의 순서를 얻을 수 있다는 보장은 없습니다. 

![At each time step, greedy search selects the token with the highest conditional probability.](../img/s2s-prob1.svg)
:label:`fig_s2s-prob1`

예를 들어 설명해 보겠습니다.<eos>출력 딕셔너리에 토큰 “A”, “B”, “C” 및 “가 네 개 있다고 가정합니다.:numref:`fig_s2s-prob1`에서 각 시간 스텝 아래의 네 숫자는 해당 시간 <eos>스텝에서 각각 “A”, “B”, “C” 및 “를 생성할 조건부 확률을 나타냅니다.각 시간 단계에서 욕심 검색은 조건부 확률이 가장 높은 토큰을 선택합니다.따라서 출력 시퀀스 “A”, “B”, “C” 및 ““는 <eos>:numref:`fig_s2s-prob1`에서 예측됩니다.이 출력 시퀀스의 조건부 확률은 $0.5\times0.4\times0.4\times0.6 = 0.048$입니다. 

![The four numbers under each time step represent the conditional probabilities of generating "A", "B", "C", and "&lt;eos&gt;" at that time step.  At time step 2, the token "C", which has the second highest conditional probability, is selected.](../img/s2s-prob2.svg)
:label:`fig_s2s-prob2`

다음으로 :numref:`fig_s2s-prob2`의 또 다른 예를 살펴 보겠습니다.:numref:`fig_s2s-prob1`와 달리 시간 단계 2에서는 :numref:`fig_s2s-prob2`에서 토큰 “C”를 선택합니다. 이 토큰은*두 번째*가 가장 높은 조건부 확률을 갖습니다.시간 단계 3의 기반이 되는 시간 단계 1과 2의 출력 하위 시퀀스가 :numref:`fig_s2s-prob1`의 “A”와 “B”에서 :numref:`fig_s2s-prob2`의 “A”와 “C”로 변경되었으므로 시간 단계 3에서 각 토큰의 조건부 확률도 :numref:`fig_s2s-prob2`에서 변경되었습니다.3단계에서 토큰 “B”를 선택한다고 가정합니다.이제 시간 단계 4는 처음 세 개의 시간 단계 “A”, “C” 및 “B”의 출력 하위 시퀀스에 대한 조건부입니다. 이는 :numref:`fig_s2s-prob1`의 “A”, “B” 및 “C”와 다릅니다.따라서 :numref:`fig_s2s-prob2`의 시간 단계 4에서 각 토큰을 생성할 조건부 확률도 :numref:`fig_s2s-prob1`의 조건부 확률과 다릅니다.결과적으로 <eos>:numref:`fig_s2s-prob2`에서 출력 시퀀스 “A”, “C”, “B” 및 ““의 조건부 확률은 $0.5\times0.3 \times0.6\times0.6=0.054$이며, 이는 :numref:`fig_s2s-prob1`의 욕심 많은 검색보다 큽니다.이 예에서는 <eos>욕심 검색에 의해 얻어진 출력 시퀀스 “A”, “B”, “C” 및 "“가 최적의 시퀀스가 아닙니다. 

## 철저한 검색

최적의 시퀀스를 얻는 것이 목표라면*철저한 검색*을 사용하는 것을 고려할 수 있습니다. 가능한 모든 출력 시퀀스를 조건부 확률로 철저히 열거 한 다음 조건부 확률이 가장 높은 시퀀스를 출력합니다. 

철저한 검색을 사용하여 최적의 시퀀스를 얻을 수 있지만 계산 비용 $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$는 지나치게 높을 수 있습니다.예를 들어, $|\mathcal{Y}|=10000$ 및 $T'=10$인 경우 $10000^{10} = 10^{40}$ 시퀀스를 평가해야 합니다.이것은 불가능 옆에 있습니다!반면에 탐욕스러운 검색의 계산 비용은 $\mathcal{O}(\left|\mathcal{Y}\right|T')$입니다. 일반적으로 전체 검색보다 훨씬 적습니다.예를 들어, $|\mathcal{Y}|=10000$ 및 $T'=10$인 경우 $10000\times10=10^5$ 시퀀스만 평가하면 됩니다. 

## 빔 검색

시퀀스 검색 전략에 대한 결정은 스펙트럼에 있으며 어느 쪽이든 쉬운 질문입니다.정확도만 중요하다면 어떻게 될까요?분명히 철저한 검색입니다.계산 비용만 중요하다면 어떻게 될까요?분명히 욕심 많은 검색입니다.실제 응용 프로그램은 일반적으로 이 두 극단 사이의 복잡한 질문을 합니다. 

*빔 검색*은 욕심 검색의 향상된 버전입니다.*빔 크기*, $k$라는 이름의 하이퍼파라미터가 있습니다. 
시간 단계 1에서는 조건부 확률이 가장 높은 $k$개의 토큰을 선택합니다.각각은 각각 $k$ 후보 출력 시퀀스의 첫 번째 토큰이 될 것입니다.각 후속 시간 단계에서 이전 시간 단계의 $k$ 후보 출력 시퀀스를 기반으로 $k\left|\mathcal{Y}\right|$ 가능한 선택 항목 중에서 조건부 확률이 가장 높은 $k$ 후보 출력 시퀀스를 계속 선택합니다. 

![The process of beam search (beam size: 2, maximum length of an output sequence: 3). The candidate output sequences are $A$, $C$, $AB$, $CE$, $ABD$, and $CED$.](../img/beam-search.svg)
:label:`fig_beam-search`

:numref:`fig_beam-search`는 예를 들어 빔 검색 프로세스를 보여줍니다.출력 어휘에 $\mathcal{Y} = \{A, B, C, D, E\}$이라는 다섯 가지 요소만 포함되어 있다고 가정합니다. 여기서 그 중 하나는 “<eos>" 입니다.빔 크기를 2로 설정하고 출력 시퀀스의 최대 길이를 3으로 설정합니다.시간 단계 1에서 조건부 확률이 $P(y_1 \mid \mathbf{c})$이 가장 높은 토큰이 $A$ 및 $C$라고 가정합니다.시간 단계 2에서 모든 $y_2 \in \mathcal{Y},$에 대해 계산합니다.  

$$\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}$$  

이 10개의 값 중에서 가장 큰 두 값을 선택합니다. 예를 들어 $P(A, B \mid \mathbf{c})$과 $P(C, E \mid \mathbf{c})$입니다.그런 다음 시간 단계 3에서 모든 $y_3 \in \mathcal{Y}$에 대해 다음을 계산합니다.  

$$\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}$$ 

이 10개의 값 중에서 가장 큰 두 값을 선택합니다. 예를 들어 $P(A, B, D \mid \mathbf{c})$ 및 $P(C, E, D \mid  \mathbf{c}).$과 같이 6개의 후보 출력 시퀀스를 얻을 수 있습니다. (i) $A$; (ii) $C$; (iii) $A$; (iv) $C$; (v) $E$; (v) $A$, $B$, $D$; 및 (vi) $C$, $E$, $D$.  

결국, 이 여섯 개의 시퀀스를 기반으로 최종 후보 출력 시퀀스 세트를 얻습니다 (예: “<eos>” 를 포함한 및 뒤의 부분 폐기).그런 다음 다음 점수 중 가장 높은 시퀀스를 출력 시퀀스로 선택합니다. 

$$ \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}\mid \mathbf{c}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$
:eqlabel:`eq_beam-search-score`

여기서 $L$은 최종 후보 시퀀스의 길이이고 $\alpha$은 일반적으로 0.75로 설정됩니다.더 긴 시퀀스는 :eqref:`eq_beam-search-score`의 합계에서 더 많은 로그 항을 갖기 때문에 분모의 용어 $L^\alpha$는 긴 시퀀스에 불이익을 줍니다. 

빔 검색의 계산 비용은 $\mathcal{O}(k\left|\mathcal{Y}\right|T')$입니다.이 결과는 탐욕스러운 검색과 철저한 검색 사이에 있습니다.실제로 욕심 검색은 빔 크기가 1인 특수 유형의 빔 검색으로 취급할 수 있습니다.빔 크기를 유연하게 선택할 수 있으므로 빔 검색은 정확도와 계산 비용 간의 균형을 제공합니다. 

## 요약

* 시퀀스 검색 전략에는 욕심 많은 검색, 철저한 검색 및 빔 검색이 포함됩니다.
* 빔 검색은 빔 크기의 유연한 선택을 통해 정확도와 계산 비용 간의 균형을 제공합니다.

## 연습문제

1. 철저한 검색을 특수 유형의 빔 검색으로 간주할 수 있습니까?왜, 왜 안되니?
1. :numref:`sec_seq2seq`에서 기계 번역 문제에 빔 검색을 적용합니다.빔 크기가 변환 결과와 예측 속도에 어떤 영향을 미칩니 까?
1. :numref:`sec_rnn_scratch`에서 사용자가 제공한 접두사를 따르는 텍스트를 생성하기 위해 언어 모델링을 사용했습니다.어떤 종류의 검색 전략을 사용합니까?개선할 수 있을까요?

[Discussions](https://discuss.d2l.ai/t/338)
