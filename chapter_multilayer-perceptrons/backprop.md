# 전방 전파, 역방향 전파 및 계산 그래프
:label:`sec_backprop`

지금까지 우리는 미니 배치 확률 적 그라데이션 강하로 모델을 교육했습니다.그러나 알고리즘을 구현했을 때 모델을 통해*전달 전파*와 관련된 계산에 대해서만 걱정했습니다.그래디언트를 계산할 시간이 왔을 때 딥 러닝 프레임워크에서 제공하는 역 전파 함수를 호출했습니다.

그래디언트의 자동 계산 (자동 차별화) 은 딥 러닝 알고리즘의 구현을 매우 단순화합니다.자동 차별화 전에 복잡한 모델을 조금이라도 변경해도 복잡한 파생 상품을 손으로 다시 계산해야 했습니다.놀랍게도 종종 학술 논문은 업데이트 규칙을 도출하기 위해 수많은 페이지를 할당해야했습니다.흥미로운 부분에 집중할 수 있도록 자동 차별화에 계속 의존해야 하지만 딥 러닝에 대한 얕은 이해를 넘고 싶다면 이러한 그라데이션이 어떻게 계산되는지 알아야 합니다.

이 섹션에서는, 우리는*역전파* (더 일반적으로*역전파*라고 함) 의 세부 사항에 깊은 다이빙을.기술과 구현 모두에 대한 통찰력을 전달하기 위해 몇 가지 기본 수학 및 계산 그래프에 의존합니다.시작하기 위해, 우리는 체중 감량 ($L_2$ 정규화) 이있는 하나의 숨겨진 레이어 MLP에 우리의 박람회 초점을 맞 춥니 다.

## 전달 전파

*전달 전파* (또는*전달 패스*) 는 계산 및 저장을 나타냅니다.
입력 계층에서 출력 계층까지 순서대로 신경망에 대한 중간 변수 (출력 포함) 의.우리는 이제 하나의 숨겨진 레이어가있는 신경망의 메커니즘을 단계별로 작업합니다.이것은 지루한 것처럼 보일 수 있지만 펑크 거장 제임스 브라운의 영원한 말에, 당신은 “보스가 될 비용을 지불”해야합니다.

단순화를 위해 입력 예제가 $\mathbf{x}\in \mathbb{R}^d$이고 숨겨진 레이어에 바이어스 용어가 포함되어 있지 않다고 가정합시다.여기서 중간 변수는 다음과 같습니다.

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

여기서 $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$는 숨겨진 레이어의 가중치 매개변수입니다.활성화 함수 $\phi$을 통해 중간 변수 $\mathbf{z}\in \mathbb{R}^h$를 실행 한 후 길이 $h$의 숨겨진 활성화 벡터를 얻고,

$$\mathbf{h}= \phi (\mathbf{z}).$$

숨겨진 변수 $\mathbf{h}$도 중간 변수입니다.출력 레이어의 매개 변수가 $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$의 가중치 만 가지고 있다고 가정하면 길이가 $q$인 벡터를 사용하여 출력 레이어 변수를 얻을 수 있습니다.

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

손실 함수가 $l$이고 예제 레이블이 $y$라고 가정하면 단일 데이터 예에 대한 손실 기간을 계산할 수 있습니다.

$$L = l(\mathbf{o}, y).$$

$L_2$ 정규화의 정의에 따르면, 하이퍼 매개 변수 $\lambda$가 주어지면 정규화 용어는

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$
:eqlabel:`eq_forward-s`

행렬의 Frobenius 규범은 행렬을 벡터로 평평하게 한 후에 적용되는 $L_2$ 표준입니다.마지막으로 주어진 데이터 예제에 대한 모델의 정규화 된 손실은 다음과 같습니다.

$$J = L + s.$$

다음 설명에서 $J$를 *목표 함수*라고 합니다.

## 전방 전파의 계산 그래프

*계산 그래프*를 플롯하면 계산 내에서 연산자와 변수의 종속성을 시각화하는 데 도움이 됩니다. :numref:`fig_forward`에는 위에서 설명한 단순 네트워크와 관련된 그래프가 포함되어 있습니다. 여기서 사각형은 변수를 나타내고 원은 연산자를 나타냅니다.왼쪽 아래 모서리는 입력을 나타내고 오른쪽 위 모서리는 출력입니다.화살표의 방향 (데이터 흐름을 나타냄) 은 주로 오른쪽 및 위쪽입니다.

![Computational graph of forward propagation.](../img/forward.svg)
:label:`fig_forward`

## 역전파

*역 전파*는 계산 방법을 나타냅니다.
신경망 매개 변수의 기울기.즉, 이 방법은 미적분학의* 체인 규칙*에 따라 출력에서 입력 레이어로 역순으로 네트워크를 통과합니다.알고리즘은 일부 매개 변수에 대한 그라데이션을 계산하는 동안 필요한 중간 변수 (부분 파생 상품) 를 저장합니다.$\mathsf{Y}=f(\mathsf{X})$ 및 $\mathsf{Z}=g(\mathsf{Y})$ 함수가 있다고 가정합니다. 입력 및 출력 $\mathsf{X}, \mathsf{Y}, \mathsf{Z}$는 임의의 모양의 텐서입니다.체인 규칙을 사용하여, 우리는 통해 $\mathsf{X}$에 대한 7323618의 유도체를 계산할 수 있습니다

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

여기서 우리는 $\text{prod}$ 연산자를 사용하여 전위 및 입력 위치를 교환과 같은 필요한 연산이 수행 된 후 인수를 곱합니다.벡터의 경우 이것은 간단합니다. 단순히 행렬 행렬 곱셈입니다.고차원 텐서의 경우 적절한 텐서를 사용합니다.연산자 $\text{prod}$는 모든 표기법 오버 헤드를 숨 깁니다.

계산 그래프가 :numref:`fig_forward`에있는 하나의 숨겨진 레이어가있는 간단한 네트워크의 매개 변수는 $\mathbf{W}^{(1)}$ 및 $\mathbf{W}^{(2)}$입니다.역 전파의 목적은 그라디언트 $\partial J/\partial \mathbf{W}^{(1)}$ 및 $\partial J/\partial \mathbf{W}^{(2)}$을 계산하는 것입니다.이를 위해 체인 규칙을 적용하고 각 중간 변수와 매개변수의 그라데이션을 차례로 계산합니다.계산 그래프의 결과부터 시작하여 매개 변수쪽으로 나아갈 필요가 있기 때문에 계산 순서는 전방 전파에서 수행 된 것과 비교하여 반대됩니다.첫 번째 단계는 손실 항 $L$ 및 정규화 용어 $s$과 관련하여 객관적 함수 $J=L+s$의 그라디언트를 계산하는 것입니다.

$$\frac{\partial J}{\partial L} = 1 \; \text{and} \; \frac{\partial J}{\partial s} = 1.$$

다음으로 체인 규칙에 따라 출력 계층 $\mathbf{o}$의 변수에 대해 객관적인 함수의 그라데이션을 계산합니다.

$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

다음으로 두 매개 변수에 대해 정규화 용어의 그라디언트를 계산합니다.

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

이제 출력 레이어에 가장 가까운 모델 매개 변수의 그라디언트 $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$를 계산할 수 있습니다.체인 규칙을 사용하면 다음과 같은 결과가 생성됩니다.

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
:eqlabel:`eq_backprop-J-h`

$\mathbf{W}^{(1)}$에 대한 그라데이션을 얻으려면 출력 레이어를 따라 숨겨진 레이어로 역 전파를 계속해야합니다.숨겨진 레이어의 출력 $\partial J/\partial \mathbf{h} \in \mathbb{R}^h$에 대한 그라데이션은

$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

활성화 함수 $\phi$은 요소 단위로 적용되기 때문에 중간 변수 $\mathbf{z}$의 그라디언트 7323614를 계산하려면 요소 별 곱셈 연산자를 사용해야합니다. 이 연산자는 $\odot$로 표시됩니다.

$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

마지막으로 입력 레이어에 가장 가까운 모델 매개 변수의 그라디언트 $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$를 얻을 수 있습니다.체인 규칙에 따르면, 우리는

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$

## 신경망 훈련

신경망을 훈련 할 때 전방 및 후방 전파는 서로에 의존합니다.특히, 앞으로 전파를 위해, 우리는 종속성의 방향으로 계산 그래프를 통과하고 그 경로에있는 모든 변수를 계산합니다.그런 다음 그래프의 계산 순서가 반전되는 역 전파에 사용됩니다.

설명하기 위해 앞서 언급 한 간단한 네트워크를 예로 들어 보겠습니다.한편으로는 전파 중 정규화 용어 :eqref:`eq_forward-s`를 계산하는 것은 모형 매개변수 $\mathbf{W}^{(1)}$ 및 $\mathbf{W}^{(2)}$의 현재 값에 따라 달라집니다.그들은 최신 반복에서 역 전파에 따라 최적화 알고리즘에 의해 주어진다.반면에 역 전파 중 매개변수 `eq_backprop-J-h`에 대한 그라데이션 계산은 정방향 전달에 의해 제공되는 숨겨진 변수 $\mathbf{h}$의 현재 값에 따라 달라집니다.

따라서 신경망을 훈련 할 때 모델 매개 변수가 초기화 된 후 역 전파로 전파를 대체하여 역 전파를 통해 주어진 그라디언트를 사용하여 모델 매개 변수를 업데이트합니다.역 전달은 중복 계산을 피하기 위해 정방향 전달에서 저장된 중간 값을 재사용합니다.결과 중 하나는 역 전파가 완료 될 때까지 중간 값을 유지해야한다는 것입니다.이것은 또한 훈련이 일반 예측보다 훨씬 더 많은 기억을 필요로하는 이유 중 하나입니다.게다가 이러한 중간 값의 크기는 네트워크 계층의 수와 배치 크기에 대략 비례합니다.따라서 더 큰 배치 크기를 사용하여 더 깊은 네트워크를 훈련하는 것이 더 쉽게*메모리 부족* 오류가 발생합니다.

## 요약

* 순방향 전파는 신경망에 의해 정의된 계산 그래프 내에 중간 변수를 순차적으로 계산하고 저장합니다.입력에서 출력 레이어로 진행됩니다.
* 역 전파 순차적으로 신경망 내의 중간 변수 및 매개 변수의 그라디언트를 역순으로 계산하고 저장합니다.
* 딥 러닝 모델을 학습할 때 전방 전파 및 후방 전파는 상호 의존적입니다.
* 훈련은 예측보다 훨씬 더 많은 기억이 필요합니다.

## 연습 문제

1. 일부 스칼라 함수 $f$에 대한 입력 $\mathbf{X}$이 $n \times m$ 행렬이라고 가정합니다.$\mathbf{X}$과 관련하여 $f$의 그라디언트의 차원은 얼마입니까?
1. 이 섹션에서 설명하는 모델의 숨겨진 레이어에 바이어스를 추가합니다.
    * 해당 계산 그래프를 그립니다.
    * 전방 및 후방 전파 방정식을 파생시킵니다.
1. 이 섹션에서 설명하는 모델에서 학습 및 예측을 위한 메모리 사용 공간을 계산합니다.
1. 두 번째 파생 상품을 계산한다고 가정합니다.계산 그래프는 어떻게됩니까?계산이 얼마나 오래 걸릴 것으로 예상됩니까?
1. 계산 그래프가 GPU에 비해 너무 크다고 가정합니다.
    * 하나 이상의 GPU를 통해 분할 할 수 있습니까?
    * 소규모 미니배치에 대한 교육에 비해 장점과 단점은 무엇입니까?

[Discussions](https://discuss.d2l.ai/t/102)
