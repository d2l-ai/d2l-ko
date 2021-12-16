# 시간을 통한 역전파
:label:`sec_bptt`

지금까지 우리는 다음과 같은 것을 반복해서 언급했습니다.
*폭발하는 그라디언트*
*배니싱 그라디언트*
그리고 그 필요성
*RNN의 그래디언트*를 분리합니다.
예를 들어, :numref:`sec_rnn_scratch`에서는 시퀀스에 대해 `detach` 함수를 호출했습니다.모델을 빠르게 구축하고 작동 방식을 볼 수 있다는 점에서 이 중 어느 것도 완전히 설명되지 않았습니다.이 섹션에서는 시퀀스 모델에 대한 역전파의 세부 사항과 수학이 작동하는 이유 (및 방법) 에 대해 좀 더 자세히 살펴볼 것입니다. 

RNN (:numref:`sec_rnn_scratch`) 을 처음 구현했을 때 기울기 폭발의 효과 중 일부가 발생했습니다.특히 연습을 해결했다면 적절한 수렴을 보장하기 위해 그래디언트 클리핑이 필수적이라는 것을 알았을 것입니다.이 문제를 더 잘 이해할 수 있도록 이 섹션에서는 시퀀스 모델에 대한 그래디언트가 계산되는 방식을 검토합니다.작동 방식에는 개념적으로 새로운 것이 없습니다.결국 우리는 여전히 그래디언트를 계산하기 위해 연쇄 규칙을 적용하기만 합니다.그럼에도 불구하고 역전파 (:numref:`sec_backprop`) 를 다시 검토하는 것이 좋습니다. 

우리는 :numref:`sec_backprop`에서 MLP에서 정방향 및 역방향 전파 및 계산 그래프를 설명했습니다.RNN의 순방향 전파는 비교적 간단합니다.
*시간*을 통한 역전파는 실제로 구체적입니다.
RNN :cite:`Werbos.1990`에서 역전파의 적용.모델 변수와 파라미터 간의 종속성을 얻기 위해 RNN의 계산 그래프를 한 번에 한 번에 하나씩 확장해야 합니다.그런 다음 연쇄 규칙에 따라 역전파를 적용하여 그라디언트를 계산하고 저장합니다.시퀀스는 다소 길어질 수 있으므로 종속성이 다소 길어질 수 있습니다.예를 들어, 1000자 시퀀스의 경우 첫 번째 토큰은 최종 위치에서 토큰에 상당한 영향을 미칠 수 있습니다.이것은 실제로 계산적으로 가능하지 않으며 (너무 오래 걸리고 너무 많은 메모리가 필요함) 매우 어려운 그래디언트에 도달하기 전에 1000 개 이상의 행렬 곱이 필요합니다.이것은 계산 및 통계적 불확실성으로 가득 찬 과정입니다.다음에서는 실제로 발생하는 상황과이를 해결하는 방법을 설명합니다. 

## RNN의 기울기 분석
:label:`subsec_bptt_analysis`

RNN의 작동 방식에 대한 단순화된 모델부터 시작합니다.이 모델은 숨겨진 상태의 세부 사항 및 업데이트 방법에 대한 세부 정보를 무시합니다.여기서 수학적 표기법은 스칼라, 벡터 및 행렬을 예전처럼 명시적으로 구분하지 않습니다.이러한 세부 사항은 분석에 중요하지 않으며 이 하위 섹션의 표기법을 어지럽히는 데만 사용됩니다. 

이 단순화된 모델에서는 $h_t$를 숨겨진 상태로, $x_t$을 입력으로, $o_t$을 시간 단계 $t$에서 출력으로 나타냅니다.:numref:`subsec_rnn_w_hidden_states`에서 입력과 은닉 상태를 연결하여 은닉 레이어에서 가중치 변수 하나를 곱할 수 있다는 논의를 상기하십시오.따라서 $w_h$ 및 $w_o$을 사용하여 은닉 레이어와 출력 레이어의 가중치를 각각 나타냅니다.결과적으로 각 시간 단계의 숨겨진 상태와 출력은 다음과 같이 설명 할 수 있습니다. 

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_h),\\o_t &= g(h_t, w_o),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

여기서 $f$ 및 $g$은 각각 은닉 레이어와 출력 레이어의 변환입니다.따라서 반복 계산을 통해 서로 의존하는 값 $\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$의 체인이 있습니다.순방향 전파는 매우 간단합니다.우리에게 필요한 것은 $(x_t, h_t, o_t)$ 트리플을 한 번에 한 번에 하나씩 반복하는 것뿐입니다.그런 다음 출력 $o_t$과 원하는 레이블 ($y_t$) 간의 불일치는 모든 $T$ 시간 단계에서 목적 함수에 의해 다음과 같이 평가됩니다. 

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_h, w_o) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$

역전파의 경우, 특히 목적 함수 $L$의 파라미터 $w_h$와 관련하여 기울기를 계산할 때 문제가 좀 더 까다롭습니다.구체적으로 말하자면, 체인 규칙에 따라 

$$\begin{aligned}\frac{\partial L}{\partial w_h}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_h}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_o)}{\partial h_t}  \frac{\partial h_t}{\partial w_h}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

:eqref:`eq_bptt_partial_L_wh`에서 제품의 첫 번째 요소와 두 번째 요소는 쉽게 계산할 수 있습니다.세 번째 요소 $\partial h_t/\partial w_h$은 $h_t$에서 매개 변수 $w_h$의 효과를 반복적으로 계산해야 하기 때문에 상황이 까다로워지는 부분입니다.:eqref:`eq_bptt_ht_ot`의 반복 계산에 따르면 $h_t$는 $h_{t-1}$과 $w_h$에 따라 달라지며, 여기서 $h_{t-1}$의 계산도 $w_h$에 따라 달라집니다.따라서 체인 규칙을 사용하면 

$$\frac{\partial h_t}{\partial w_h}= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`

위의 기울기를 도출하기 위해 $t=1, 2,\ldots$에 대해 $a_{0}=0$과 $a_{t}=b_{t}+c_{t}a_{t-1}$를 충족하는 세 개의 시퀀스 $\{a_{t}\},\{b_{t}\},\{c_{t}\}$가 있다고 가정합니다.그런 다음 $t\geq 1$의 경우 쉽게 보여줄 수 있습니다. 

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

다음에 따라 $a_t$, $b_t$ 및 $c_t$를 대체하여 

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_h},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}},\end{aligned}$$

:eqref:`eq_bptt_partial_ht_wh_recur`의 기울기 계산은 $a_{t}=b_{t}+c_{t}a_{t-1}$을 충족합니다.따라서 :eqref:`eq_bptt_at`에 따라 다음과 같이 :eqref:`eq_bptt_partial_ht_wh_recur`에서 반복 계산을 제거 할 수 있습니다. 

$$\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_h)}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_h)}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

연쇄 규칙을 사용하여 $\partial h_t/\partial w_h$를 재귀적으로 계산할 수 있지만 $t$가 클 때마다 이 체인은 매우 길어질 수 있습니다.이 문제를 해결하기위한 여러 가지 전략에 대해 논의하겠습니다. 

### 전체 계산 ### 

분명히 :eqref:`eq_bptt_partial_ht_wh_gen`에서 전체 합계를 계산할 수 있습니다.그러나 초기 조건의 미묘한 변화가 결과에 많은 영향을 미칠 수 있기 때문에 이것은 매우 느리고 그라디언트가 터질 수 있습니다.즉, 초기 조건에서 최소한의 변화가 결과에 불균형적인 변화를 가져 오는 나비 효과와 유사한 것을 볼 수 있습니다.이것은 우리가 추정하고자 하는 모형의 관점에서 실제로 매우 바람직하지 않습니다.결국 우리는 잘 일반화되는 강력한 추정기를 찾고 있습니다.따라서이 전략은 실제로 거의 사용되지 않습니다. 

### 자르기 시간 단계 ###

또는 $\tau$단계 후에 :eqref:`eq_bptt_partial_ht_wh_gen`에서 합계를 잘라낼 수 있습니다.이것은 :numref:`sec_rnn_scratch`에서 그라디언트를 분리했을 때와 같이 지금까지 논의한 내용입니다.이렇게 하면 $\partial h_{t-\tau}/\partial w_h$에서 합계를 종료하기만 하면 실제 기울기의*근사치*가 됩니다.실제로 이것은 아주 잘 작동합니다.일반적으로 시간 :cite:`Jaeger.2002`를 통해 잘린 역 전파라고 합니다.그 결과 중 하나는 모델이 장기적인 결과보다는 단기적인 영향력에 주로 초점을 맞추고 있다는 것입니다.이는 추정치를 더 간단하고 안정적인 모델로 편향시키기 때문에 실제로*바람직합니다*. 

### 무작위 잘림 ### 

마지막으로 $\partial h_t/\partial w_h$을 기대치는 정확하지만 순서를 자르는 랜덤 변수로 바꿀 수 있습니다.이는 미리 정의된 $0 \leq \pi_t \leq 1$과 함께 $\xi_t$의 시퀀스를 사용하여 달성됩니다. 여기서 $P(\xi_t = 0) = 1-\pi_t$ 및 $P(\xi_t = \pi_t^{-1}) = \pi_t$, 따라서 $E[\xi_t] = 1$입니다.이를 사용하여 :eqref:`eq_bptt_partial_ht_wh_recur`의 그라디언트 $\partial h_t/\partial w_h$을 다음과 같이 대체합니다. 

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$

그것은 $\xi_t$의 정의에서 $E[z_t] = \partial h_t/\partial w_h$이라는 것을 따릅니다.$\xi_t = 0$이 될 때마다 반복 계산은 해당 시간 단계 $t$에서 종료됩니다.이로 인해 긴 시퀀스가 드물지만 적절하게 과가중되는 다양한 길이의 시퀀스의 가중 합이 생성됩니다.이 아이디어는 탈렉과 올리비에 :cite:`Tallec.Ollivier.2017`에 의해 제안되었습니다. 

### 전략 비교

![Comparing strategies for computing gradients in RNNs. From top to bottom: randomized truncation, regular truncation, and full computation.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`

:numref:`fig_truncated_bptt`는 RNN에 대한 시간 경과에 따른 역전파를 사용하여*타임 머신* 책의 처음 몇 문자를 분석할 때의 세 가지 전략을 보여줍니다. 

* 첫 번째 행은 텍스트를 다양한 길이의 세그먼트로 분할하는 무작위 잘림입니다.
* 두 번째 행은 텍스트를 동일한 길이의 하위 시퀀스로 나누는 일반 잘림입니다.이것이 우리가 RNN 실험에서 해왔던 것입니다.
* 세 번째 행은 계산적으로 실현 불가능한 표현식으로 이어지는 시간 경과에 따른 전체 역전파입니다.

안타깝게도 이론적으로는 매력적이지만 무작위 절단은 일반 절단보다 훨씬 잘 작동하지 않습니다. 대부분 여러 요인으로 인해 발생합니다.첫째, 과거로의 여러 역전파 단계 이후 관찰의 효과는 실제로 종속성을 포착하기에 충분합니다.둘째, 분산이 증가하면 스텝이 많을수록 기울기가 더 정확하다는 사실을 상쇄합니다.셋째, 상호 작용 범위가 짧은 모델을 실제로*원합니다*.따라서 시간이 지남에 따라 규칙적으로 잘린 역 전파는 바람직 할 수있는 약간의 정규화 효과가 있습니다. 

## 시간을 통한 역전파 세부 정보

일반적인 원칙을 논의한 후 시간에 따른 역전파에 대해 자세히 논의하겠습니다.:numref:`subsec_bptt_analysis`의 분석과 달리 다음에서는 분해된 모든 모델 파라미터에 대해 목적 함수의 기울기를 계산하는 방법을 보여줍니다.단순하게 유지하기 위해 바이어스 파라미터가 없는 RNN을 고려합니다. 이 RNN은 은닉 계층의 활성화 함수가 아이덴티티 매핑 ($\phi(x)=x$) 을 사용합니다.시간 단계 $t$의 경우 단일 예제 입력과 레이블을 각각 $\mathbf{x}_t \in \mathbb{R}^d$ 및 $y_t$로 지정합니다.숨겨진 상태 ($\mathbf{h}_t \in \mathbb{R}^h$) 및 출력 $\mathbf{o}_t \in \mathbb{R}^q$은 다음과 같이 계산됩니다. 

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_{qh} \mathbf{h}_{t},\end{aligned}$$

여기서 $\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$, $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ 및 $\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$은 가중치 매개변수입니다.시간 단계 $t$에서의 손실을 $l(\mathbf{o}_t, y_t)$로 나타냅니다.우리의 목적 함수, 시퀀스의 시작부터 $T$ 시간 스텝을 초과하는 손실은 다음과 같습니다. 

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$

RNN을 계산하는 동안 모델 변수와 매개 변수 간의 종속성을 시각화하기 위해 :numref:`fig_rnn_bptt`와 같이 모델에 대한 계산 그래프를 그릴 수 있습니다.예를 들어, 시간 단계 3, $\mathbf{h}_3$의 숨겨진 상태 계산은 모델 매개변수 $\mathbf{W}_{hx}$ 및 $\mathbf{W}_{hh}$, 마지막 시간 스텝 $\mathbf{h}_2$의 숨겨진 상태 및 현재 시간 스텝 $\mathbf{x}_3$의 입력에 따라 달라집니다. 

![Computational graph showing dependencies for an RNN model with three time steps. Boxes represent variables (not shaded) or parameters (shaded) and circles represent operators.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

방금 언급한 바와 같이 :numref:`fig_rnn_bptt`의 모델 매개변수는 $\mathbf{W}_{hx}$, $\mathbf{W}_{hh}$ 및 $\mathbf{W}_{qh}$입니다.일반적으로 이 모델을 훈련하려면 이러한 매개변수 $\partial L/\partial \mathbf{W}_{hx}$, $\partial L/\partial \mathbf{W}_{hh}$ 및 $\partial L/\partial \mathbf{W}_{qh}$에 대한 기울기 계산이 필요합니다.:numref:`fig_rnn_bptt`의 종속성에 따라 화살표의 반대 방향으로 이동하여 그라디언트를 차례로 계산하고 저장할 수 있습니다.연쇄 규칙에서 서로 다른 모양의 행렬, 벡터 및 스칼라의 곱셈을 유연하게 표현하기 위해 :numref:`sec_backprop`에 설명된 대로 $\text{prod}$ 연산자를 계속 사용합니다. 

우선, 언제든지 $t$ 단계에서 모델 출력과 관련하여 목적 함수를 구별하는 것은 매우 간단합니다. 

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

이제 출력 계층 $\partial L/\partial \mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$에서 매개 변수 $\mathbf{W}_{qh}$에 대한 목적 함수의 기울기를 계산할 수 있습니다.:numref:`fig_rnn_bptt`를 기준으로, 목적 함수 $L$은 $\mathbf{o}_1, \ldots, \mathbf{o}_T$을 통해 $\mathbf{W}_{qh}$에 종속됩니다.체인 규칙을 사용하면 

$$
\frac{\partial L}{\partial \mathbf{W}_{qh}}
= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

여기서 $\partial L/\partial \mathbf{o}_t$는 :eqref:`eq_bptt_partial_L_ot`에 의해 주어집니다. 

다음으로, :numref:`fig_rnn_bptt`에 나타낸 바와 같이, 최종 시간 스텝 $T$에서 목적 함수 $L$은 $\mathbf{o}_T$를 통해서만 은닉 상태 ($\mathbf{h}_T$) 에 의존한다.따라서 체인 규칙을 사용하여 기울기 $\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$을 쉽게 찾을 수 있습니다. 

$$\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

목적 함수 $L$은 $\mathbf{h}_{t+1}$ 및 $\mathbf{o}_t$를 통해 $\mathbf{h}_t$에 의존하는 모든 시간 단계 $t < T$에 대해 더 까다로워집니다.체인 규칙에 따르면, 임의의 시간 스텝 $t < T$에서 은닉 상태 ($\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$) 의 기울기는 다음과 같이 반복적으로 계산될 수 있다. 

$$\frac{\partial L}{\partial \mathbf{h}_t} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

분석을 위해 $1 \leq t \leq T$가 제공하는 시간 단계에 대한 반복 계산을 확장하면 

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

:eqref:`eq_bptt_partial_L_ht`에서 이 간단한 선형 예제가 이미 긴 시퀀스 모델의 몇 가지 주요 문제를 보여주고 있음을 알 수 있습니다. 잠재적으로 $\mathbf{W}_{hh}^\top$의 매우 큰 거듭제곱을 포함합니다.여기에서 1보다 작은 고유값은 사라지고 1보다 큰 고유값은 발산합니다.이것은 수치적으로 불안정하며 사라지거나 폭발하는 그라디언트의 형태로 나타납니다.이 문제를 해결하는 한 가지 방법은 :numref:`subsec_bptt_analysis`에서 설명한 대로 계산적으로 편리한 크기로 시간 단계를 잘라내는 것입니다.실제로 이 잘림은 주어진 시간 단계 수 후에 그래디언트를 분리하여 수행됩니다.나중에 장기 단기 기억과 같은 더 정교한 시퀀스 모델이 어떻게 이것을 더 완화시킬 수 있는지 살펴볼 것입니다.  

마지막으로, :numref:`fig_rnn_bptt`는 목적 함수 $L$이 은닉 상태 $\mathbf{h}_1, \ldots, \mathbf{h}_T$을 통해 은닉 레이어의 모델 파라미터 $\mathbf{W}_{hx}$ 및 $\mathbf{W}_{hh}$에 종속된다는 것을 보여줍니다.이러한 매개 변수 $\partial L / \partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$ 및 $\partial L / \partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$과 관련하여 기울기를 계산하기 위해 다음과 같은 체인 규칙을 적용합니다. 

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

여기서 :eqref:`eq_bptt_partial_L_hT_final_step` 및 :eqref:`eq_bptt_partial_L_ht_recur`에 의해 반복적으로 계산되는 $\partial L/\partial \mathbf{h}_t$은 수치 안정성에 영향을 미치는 주요 수량입니다. 

:numref:`sec_backprop`에서 설명했듯이 시간을 통한 역 전파는 RNN에서 역 전파를 적용하는 것이기 때문에 훈련 RNN은 시간을 통한 역 전파와 순방향 전파를 번갈아 가며 수행합니다.또한 시간을 통한 역전파는 위의 그라디언트를 차례로 계산하고 저장합니다.구체적으로, 저장된 중간 값은 $\partial L / \partial \mathbf{W}_{hx}$ 및 $\partial L / \partial \mathbf{W}_{hh}$의 계산에 사용될 $\partial L/\partial \mathbf{h}_t$를 저장하는 것과 같은 중복 계산을 피하기 위해 재사용된다. 

## 요약

* 시간에 따른 역전파는 단순히 숨겨진 상태의 시퀀스 모델에 역전파를 적용하는 것입니다.
* 잘림은 계산 편의성과 수치 안정성 (예: 일반 잘림 및 무작위 잘림) 을 위해 필요합니다.
* 행렬의 거듭제곱이 높으면 고유값이 분산되거나 사라질 수 있습니다.이것은 폭발하거나 사라지는 그라디언트의 형태로 나타납니다.
* 효율적인 계산을 위해 중간 값은 시간에 따라 역전파되는 동안 캐시됩니다.

## 연습문제

1. 고유값 $\lambda_i$를 갖는 대칭 행렬 $\mathbf{M} \in \mathbb{R}^{n \times n}$이 있다고 가정합니다. 이 행렬의 해당 고유 벡터는 $\mathbf{v}_i$ ($i = 1, \ldots, n$) 입니다.일반성을 잃지 않고 $|\lambda_i| \geq |\lambda_{i+1}|$ 순서로 주문되었다고 가정합니다. 
   1. $\mathbf{M}^k$에 고유값 $\lambda_i^k$가 있음을 보여줍니다.
   1. 확률 벡터 $\mathbf{x} \in \mathbb{R}^n$에 대해 확률이 높은 $\mathbf{M}^k \mathbf{x}$가 고유 벡터 $\mathbf{v}_1$과 매우 많이 정렬된다는 것을 증명합니다. 
그 중 $\mathbf{M}$입니다.이 진술을 공식화하십시오.
   1. 위의 결과는 RNN의 기울기에 대해 무엇을 의미합니까?
1. 그래디언트 클리핑 외에도 순환 신경망에서 그래디언트 폭발에 대처하는 다른 방법을 생각해 볼 수 있습니까?

[Discussions](https://discuss.d2l.ai/t/334)
