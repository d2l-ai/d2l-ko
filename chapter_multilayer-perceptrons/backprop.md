# 순방향 전파, 역방향 전파 및 계산 그래프
:label:`sec_backprop`

지금까지 우리는 미니배치 확률적 경사하강법으로 모델을 훈련시켰습니다.그러나 알고리즘을 구현할 때 모델을 통해*순방향 전파*와 관련된 계산에 대해서만 걱정했습니다.그래디언트를 계산할 때가 되면 딥 러닝 프레임워크에서 제공하는 역전파 함수를 호출했습니다. 

그래디언트 자동 계산 (자동 미분) 은 딥 러닝 알고리즘의 구현을 크게 단순화합니다.자동 미분 전에는 복잡한 모델을 조금만 변경하더라도 복잡한 도함수를 수동으로 다시 계산해야 했습니다.놀랍게도 학술 논문은 업데이트 규칙을 도출하기 위해 수많은 페이지를 할당해야 하는 경우가 많았습니다.흥미로운 부분에 집중할 수 있도록 자동 차별화에 계속 의존해야하지만 딥 러닝에 대한 얕은 이해를 넘어서려면 이러한 기울기가 어떻게 계산되는지 알아야합니다. 

이 섹션에서는*역방향 전파* (더 일반적으로*역전파*라고 함) 에 대해 자세히 알아봅니다.기술과 구현 모두에 대한 통찰력을 전달하기 위해 몇 가지 기본 수학 및 계산 그래프에 의존합니다.우선, 우리는 체중 감소 ($L_2$ 정규화) 가있는 단일 숨겨진 레이어 MLP에 박람회를 집중합니다. 

## 순방향 전파

*순방향 전파* (또는*정방향 패스*) 는 계산 및 저장을 나타냅니다.
입력 계층에서 출력 계층까지 순서대로 신경망에 대한 중간 변수 (출력값 포함)이제 하나의 은닉 레이어가 있는 신경망의 메카닉을 단계별로 살펴봅니다.지루해 보일지 모르지만 펑크 거장 제임스 브라운 (James Brown) 의 영원한 말로 “보스가되기 위해 비용을 지불”해야합니다. 

간단하게하기 위해 입력 예제가 $\mathbf{x}\in \mathbb{R}^d$이고 숨겨진 계층에 편향 항이 포함되어 있지 않다고 가정해 보겠습니다.여기서 중간 변수는 다음과 같습니다. 

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

여기서 $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$은 은닉 레이어의 가중치 매개변수입니다.활성화 함수 $\phi$를 통해 중간 변수 $\mathbf{z}\in \mathbb{R}^h$을 실행한 후 길이 $h$의 숨겨진 활성화 벡터를 얻습니다. 

$$\mathbf{h}= \phi (\mathbf{z}).$$

숨겨진 변수 $\mathbf{h}$도 중간 변수입니다.출력 계층의 파라미터가 $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$의 가중치만 가지고 있다고 가정하면 길이가 $q$인 벡터를 갖는 출력 계층 변수를 얻을 수 있습니다. 

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

손실 함수가 $l$이고 예제 레이블이 $y$라고 가정하면 단일 데이터 예에 대한 손실 항을 계산할 수 있습니다. 

$$L = l(\mathbf{o}, y).$$

$L_2$ 정규화의 정의에 따르면 하이퍼 파라미터 $\lambda$가 주어지면 정규화 용어는 다음과 같습니다. 

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$
:eqlabel:`eq_forward-s`

여기서 행렬의 프로베니우스 노름은 단순히 행렬을 벡터로 평탄화 한 후 적용된 $L_2$ 노름입니다.마지막으로, 주어진 데이터 예에서 모델의 정규화된 손실은 다음과 같습니다. 

$$J = L + s.$$

다음 논의에서는 $J$를*목적 함수*라고 합니다. 

## 순방향 전파의 계산 그래프

*계산 그래프*를 플로팅하면 계산 내에서 연산자와 변수의 종속성을 시각화할 수 있습니다. :numref:`fig_forward`에는 위에서 설명한 간단한 네트워크와 관련된 그래프가 포함되어 있습니다. 여기서 사각형은 변수를 나타내고 원은 연산자를 나타냅니다.왼쪽 아래 모서리는 입력을 나타내고 오른쪽 위 모서리는 출력을 나타냅니다.화살표의 방향 (데이터 흐름을 나타냄) 은 주로 오른쪽과 위쪽입니다. 

![Computational graph of forward propagation.](../img/forward.svg)
:label:`fig_forward`

## 역전파

*역전파*는 계산 방법을 나타냅니다.
신경망 파라미터의 기울기.간단히 말해, 이 방법은 미적분의*체인 규칙*에 따라 출력값에서 입력 계층까지 역순으로 네트워크를 횡단합니다.이 알고리즘은 일부 파라미터에 대한 기울기를 계산하는 동안 필요한 중간 변수 (편도함수) 를 저장합니다.함수 $\mathsf{Y}=f(\mathsf{X})$ 및 $\mathsf{Z}=g(\mathsf{Y})$가 있다고 가정합니다. 이 함수에서 입력과 출력 $\mathsf{X}, \mathsf{Y}, \mathsf{Z}$은 임의의 모양의 텐서입니다.체인 규칙을 사용하여 다음을 통해 $\mathsf{X}$에 대한 $\mathsf{Z}$의 도함수를 계산할 수 있습니다. 

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

여기서는 조옮김 및 입력 위치 교환과 같은 필요한 작업을 수행 한 후 $\text{prod}$ 연산자를 사용하여 인수를 곱합니다.벡터의 경우 이는 간단합니다. 단순히 행렬-행렬 곱셈입니다.더 높은 차원 텐서의 경우 적절한 텐서를 사용합니다.연산자 $\text{prod}$는 모든 표기법 오버헤드를 숨깁니다. 

계산 그래프가 :numref:`fig_forward`에 있는 하나의 숨겨진 계층을 가진 단순 네트워크의 매개 변수는 $\mathbf{W}^{(1)}$과 $\mathbf{W}^{(2)}$이라는 것을 상기하십시오.역전파의 목적은 기울기 $\partial J/\partial \mathbf{W}^{(1)}$ 및 $\partial J/\partial \mathbf{W}^{(2)}$을 계산하는 것입니다.이를 위해 체인 규칙을 적용하고 각 중간 변수 및 매개 변수의 기울기를 차례로 계산합니다.계산 순서는 순방향 전파에서 수행되는 순방향 전파와 비교하여 반전됩니다. 계산 그래프의 결과부터 시작하여 매개 변수를 향해 나아갈 필요가 있기 때문입니다.첫 번째 단계는 손실 항 $L$ 및 정규화 항 $s$에 대한 목적 함수 $J=L+s$의 기울기를 계산하는 것입니다. 

$$\frac{\partial J}{\partial L} = 1 \; \text{and} \; \frac{\partial J}{\partial s} = 1.$$

다음으로 연쇄 규칙에 따라 출력 계층 $\mathbf{o}$의 변수에 대한 목적 함수의 기울기를 계산합니다. 

$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

다음으로 두 매개 변수에 대한 정규화 항의 기울기를 계산합니다. 

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

이제 출력 계층에 가장 가까운 모델 매개 변수의 기울기 $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$를 계산할 수 있습니다.체인 규칙을 사용하면 다음과 같은 결과가 나타납니다. 

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
:eqlabel:`eq_backprop-J-h`

$\mathbf{W}^{(1)}$에 대한 그래디언트를 얻으려면 출력 레이어를 따라 은닉 레이어로 역전파를 계속해야 합니다.은닉 레이어의 출력값 $\partial J/\partial \mathbf{h} \in \mathbb{R}^h$에 대한 기울기는 다음과 같이 지정됩니다. 

$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

활성화 함수 $\phi$는 요소별로 적용되므로 중간 변수 $\mathbf{z}$의 기울기 $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$을 계산하려면 요소별 곱셈 연산자를 사용해야 합니다. 이 연산자는 $\odot$로 표시됩니다. 

$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

마지막으로 입력 계층에 가장 가까운 모델 매개 변수의 기울기 $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$를 얻을 수 있습니다.연쇄 규칙에 따르면 

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$

## 신경망 훈련

신경망을 훈련시킬 때 순방향 및 역방향 전파는 서로 의존합니다.특히 순방향 전파를 위해 계산 그래프를 종속성 방향으로 이동하고 경로의 모든 변수를 계산합니다.그런 다음 그래프의 계산 순서가 반전되는 역전파에 사용됩니다. 

앞서 언급한 간단한 네트워크를 예로 들어 설명해 보겠습니다.한편으로 순방향 전파 중에 정규화 용어 :eqref:`eq_forward-s`를 계산하는 것은 모델 매개변수 $\mathbf{W}^{(1)}$ 및 $\mathbf{W}^{(2)}$의 현재 값에 따라 달라집니다.최신 반복에서 역전파에 따라 최적화 알고리즘에 의해 제공됩니다.반면에 역전파 중 매개변수 :eqref:`eq_backprop-J-h`에 대한 기울기 계산은 순방향 전파에 의해 주어진 숨겨진 변수 $\mathbf{h}$의 현재 값에 따라 달라집니다. 

따라서 신경망을 훈련시킬 때 모델 매개 변수가 초기화 된 후 역 전파로 순방향 전파를 번갈아 가며 역 전파로 주어진 기울기를 사용하여 모델 매개 변수를 업데이트합니다.역전파는 중복 계산을 방지하기 위해 순방향 전달에서 저장된 중간 값을 재사용합니다.그 결과 중 하나는 역전파가 완료될 때까지 중간 값을 유지해야 한다는 것입니다.이것은 또한 훈련에 일반 예측보다 훨씬 더 많은 메모리가 필요한 이유 중 하나입니다.또한 이러한 중간 값의 크기는 네트워크 계층의 수와 배치 크기에 거의 비례합니다.따라서 더 큰 배치 크기를 사용하여 심층 네트워크를 훈련시키면 더 쉽게 메모리 부족* 오류가 발생합니다. 

## 요약

* 순방향 전파는 신경망에 의해 정의된 계산 그래프 내에 중간 변수를 순차적으로 계산하고 저장합니다.입력에서 출력 계층으로 진행됩니다.
* 역전파는 신경망 내 중간 변수 및 파라미터의 기울기를 역순으로 순차적으로 계산하고 저장합니다.
* 딥러닝 모델을 학습시킬 때 순방향 전파와 역 전파는 상호 의존적입니다.
* 훈련에는 예측보다 훨씬 더 많은 메모리가 필요합니다.

## 연습문제

1. 일부 스칼라 함수 $f$에 대한 입력값 $\mathbf{X}$이 $n \times m$ 행렬이라고 가정합니다.$\mathbf{X}$에 대한 $f$의 기울기의 차원은 무엇입니까?
1. 이 섹션에서 설명하는 모델의 은닉 계층에 치우침을 추가합니다 (정규화 항에 치우침을 포함할 필요는 없음).
    1. 해당하는 계산 그래프를 그립니다.
    1. 순방향 및 역방향 전파 방정식을 도출합니다.
1. 이 섹션에서 설명하는 모델에서 훈련 및 예측을 위한 메모리 공간을 계산합니다.
1. 2차 도함수를 계산하려고 한다고 가정합니다.계산 그래프는 어떻게 되나요?계산에 얼마나 오래 걸릴 것으로 예상하십니까?
1. 계산 그래프가 GPU에 비해 너무 크다고 가정합니다.
    1. 두 개 이상의 GPU로 파티셔닝할 수 있습니까?
    1. 더 작은 미니 배치에서 훈련하는 것에 비해 장점과 단점은 무엇입니까?

[Discussions](https://discuss.d2l.ai/t/102)
