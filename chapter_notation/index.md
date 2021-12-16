# 표기법
:label:`chap_notation`

이 책 전체에서 우리는 다음과 같은 표기법 규칙을 고수합니다.이러한 기호 중 일부는 자리 표시자이고 다른 기호는 특정 개체를 나타냅니다.일반적으로 부정 관사 “a”는 기호가 자리 표시자이고 비슷한 형식의 기호가 동일한 유형의 다른 객체를 나타낼 수 있음을 나타냅니다.예를 들어, “$x$: 스칼라”는 일반적으로 소문자가 스칼라 값을 나타낸다는 의미입니다. 

## 숫자 오브젝트

* $x$: 스칼라
* $\mathbf{x}$: 벡터입니다.
* $\mathbf{X}$: 매트릭스
* $\mathsf{X}$: 일반 텐서
* $\mathbf{I}$: 모든 대각선 항목에 $1$가 있고 모든 비대각선에 $0$가 있는 단위 행렬 - 정사각형
* $x_i$, $[\mathbf{x}]_i$: 벡터 $\mathbf{x}$의 $i^\mathrm{th}$ 요소입니다.
* $x_{ij}$, $x_{i,j}$,$[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$: 행 $i$와 열 $j$에 있는 행렬 $\mathbf{X}$의 요소입니다.

## 집합 이론

* $\mathcal{X}$: 한 세트
* $\mathbb{Z}$: 정수의 집합입니다.
* $\mathbb{Z}^+$: 양의 정수의 집합입니다.
* $\mathbb{R}$: 실수의 집합입니다.
* $\mathbb{R}^n$: 실수로 구성된 $n$차원 벡터의 집합
* $\mathbb{R}^{a\times b}$:$a$개의 행과 $b$개의 열이 있는 실수로 구성된 행렬 집합입니다.
* $|\mathcal{X}|$: 세트의 카디널리티 (요소 수) $\mathcal{X}$
* $\mathcal{A}\cup\mathcal{B}$: 세트 $\mathcal{A}$와 $\mathcal{B}$의 조합
* $\mathcal{A}\cap\mathcal{B}$: 세트 $\mathcal{A}$와 $\mathcal{B}$의 교차점
* $\mathcal{A}\setminus\mathcal{B}$:$\mathcal{A}$에서 $\mathcal{B}$의 빼기를 설정합니다 ($\mathcal{B}$에 속하지 않는 $\mathcal{A}$의 요소만 포함)

## 함수 및 연산자

* $f(\cdot)$: 하나의 함수
* $\log(\cdot)$: 자연 로그 (기본 $e$)
* $\log_2(\cdot)$: 밑이 있는 로그 $2$
* $\exp(\cdot)$: 지수 함수
* $\mathbf{1}(\cdot)$: 표시기 함수로, 부울 인수가 참이면 $1$으로 평가되고 그렇지 않으면 $0$로 평가됩니다.
* $\mathbf{1}_{\mathcal{X}}(z)$: 세트 멤버십 표시기 함수는 요소 $z$이 세트 $\mathcal{X}$ 및 $0$에 속하는 경우 $1$로 평가됩니다. 그렇지 않으면 $0$로 평가됩니다.
* $\mathbf{(\cdot)}^\top$: 벡터 또는 행렬의 전치
* $\mathbf{X}^{-1}$: 행렬의 역함수 $\mathbf{X}$
* $\odot$: 하다마르 (요소별) 제품
* $[\cdot, \cdot]$: 연결
* $\|\cdot\|_p$:$L_p$ 표준
* $\|\cdot\|$:$L_2$ 표준
* $\langle \mathbf{x}, \mathbf{y} \rangle$: 벡터의 내적 $\mathbf{x}$ 및 $\mathbf{y}$의 내적
* #$\sum$: 요소 모음에 대한 요약
* $\prod$: 요소 컬렉션을 통한 제품
* $\stackrel{\mathrm{def}}{=}$: 왼쪽에 있는 기호의 정의로 주장되는 평등

## 미적분

* $\frac{dy}{dx}$:$x$와 관련하여 $y$의 파생물
* $\frac{\partial y}{\partial x}$:$x$와 관련하여 $y$의 부분 도함수
* $\nabla_{\mathbf{x}} y$:$\mathbf{x}$에 대한 $y$의 기울기
* $\int_a^b f(x) \;dx$:$x$과 관련하여 $a$에서 $b$까지 $f$의 명확한 적분
* $\int f(x) \;dx$:$x$와 관련하여 $f$의 무기한 적분

## 확률 및 정보 이론

* $X$: 랜덤 변수
* $P$: 확률 분포
* $X \sim P$: 랜덤 변수 $X$에는 분포 $P$가 있습니다.
* $P(X=x)$: 랜덤 변수 $X$이 값 $x$를 갖는 사건에 지정된 확률입니다.
* $P(X \mid Y)$:$Y$가 주어진 $X$의 조건부 확률 분포
* $p(\cdot)$: 분포 P와 연관된 확률 밀도 함수 (PDF)
* ${E}[X]$: 랜덤 변수에 대한 기대치 $X$
* $X \perp Y$: 랜덤 변수 $X$ 및 $Y$는 독립적입니다.
* $X \perp Y \mid Z$: 랜덤 변수 $X$ 및 $Y$은 $Z$가 주어지면 조건부로 독립적입니다.
* $\sigma_X$: 랜덤 변수 $X$의 표준 편차
* $\mathrm{Var}(X)$: 랜덤 변수 $X$의 분산, $\sigma^2_X$와 같음
* $\mathrm{Cov}(X, Y)$: 랜덤 변수 $X$ 및 $Y$의 공분산
* $\rho(X, Y)$:$X$과 $Y$ 사이의 피어슨 상관 계수는 $\frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$와 같습니다.
* $H(X)$: 랜덤 변수의 엔트로피 $X$
* $D_{\mathrm{KL}}(P\|Q)$: 분포 $Q$에서 분포 $P$로의 KL 발산 (또는 상대 엔트로피)

[Discussions](https://discuss.d2l.ai/t/25)
