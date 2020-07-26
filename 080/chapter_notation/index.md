# Notation
# 표기법



:label:`chap_notation`

The notation used throughout this book is summarized below.

이 책에서 사용되는 표기법을 아래에 요약했습니다.



## Numbers

## 수

* $x$: A scalar , 스칼라
* $\mathbf{x}$: A vector , 벡터
* $\mathbf{X}$: A matrix , 행렬
* $\mathsf{X}$: A tensor , 텐서
* $\mathbf{I}$: An identity matrix , 항등 행렬
* $x_i$, $[\mathbf{x}]_i$: The $i^\mathrm{th}$ element of vector $\mathbf{x}$ , 벡터 $\mathbf{x}$ 의 $i^\mathrm{th}$ 번째 원소
* $x_{ij}$, $[\mathbf{X}]_{ij}$: The element of matrix $\mathbf{X}$ at row $i$ and column $j$ , 행렬 $\mathbf{X}$ 의 $i$ 행, $j$ 열 원소



## Set Theory


## 집합론


* $\mathcal{X}$: A set , 집합
* $\mathbb{Z}$: The set of integers , 정수 집합
* $\mathbb{R}$: The set of real numbers , 실수 집합
* $\mathbb{R}^n$: The set of $n$-dimensional vectors of real numbers , $n$-차원 실수 벡터 집합
* $\mathbb{R}^{a\times b}$: The set of matrices of real numbers with $a$ rows and $b$ columns , $a$개의 행과 $b$개의 열을 가진 실수 행렬 집합
* $\mathcal{A}\cup\mathcal{B}$: Union of sets $\mathcal{A}$ and $\mathcal{B}$ , 집합 $\mathcal{A}$ 와 $\mathcal{B}$ 의 합집합
* $\mathcal{A}\cap\mathcal{B}$: Intersection of sets $\mathcal{A}$ and $\mathcal{B}$ , 집합 $\mathcal{A}$ 와 $\mathcal{B}$ 의 교집합
* $\mathcal{A}\setminus\mathcal{B}$: Subtraction of set $\mathcal{B}$ from set $\mathcal{A}$ , 집합 $\mathcal{A}$ 와 $\mathcal{B}$ 의 차집합



## Functions and Operators

## 함수와 연산자


* $f(\cdot)$: A function , 함수
* $\log(\cdot)$: The natural logarithm , 자연 로그
* $\exp(\cdot)$: The exponential function , 지수 함수 (로그 함수의 역함수)
* $\mathbf{1}_\mathcal{X}$: The indicator function , [지시 함수](https://ko.wikipedia.org/wiki/%EC%A7%80%EC%8B%9C_%ED%95%A8%EC%88%98) 
* $\mathbf{(\cdot)}^\top$: Transpose of a vector or a matrix , 전치 벡터 또는 [전치 행렬](https://ko.wikipedia.org/wiki/%EC%A0%84%EC%B9%98%ED%96%89%EB%A0%AC)
* $\mathbf{X}^{-1}$: Inverse of matrix $\mathbf{X}$ , 행렬 $\mathbf{X}$ 의 역행렬
* $\odot$: Hadamard (elementwise) product , 원소별 [아다마르 곱](https://ko.wikipedia.org/wiki/%EC%95%84%EB%8B%A4%EB%A7%88%EB%A5%B4_%EA%B3%B1)
* $[\cdot, \cdot]$: Concatenation , 연결
* $\lvert \mathcal{X} \rvert$: Cardinality of set $\mathcal{X}$ , 집합 $\mathcal{X}$ 의 [크기](https://ko.wikipedia.org/wiki/%EC%A7%91%ED%95%A9%EC%9D%98_%ED%81%AC%EA%B8%B0)
* $\|\cdot\|_p$: $L_p$ norm , $L_p$ [노름](https://ko.wikipedia.org/wiki/%EB%85%B8%EB%A6%84_%EA%B3%B5%EA%B0%84)
* $\|\cdot\|$: $L_2$ norm , $L_2$ [노름](https://ko.wikipedia.org/wiki/%EB%85%B8%EB%A6%84_%EA%B3%B5%EA%B0%84)
* $\langle \mathbf{x}, \mathbf{y} \rangle$: Dot product of vectors $\mathbf{x}$ and $\mathbf{y}$ , 벡터 $\mathbf{x}$ 와 $\mathbf{y}$ 의 내적
* $\sum$: Series addition , (수열의) 합
* $\prod$: Series multiplication , (수열의) 곱
* $\stackrel{\mathrm{def}}{=}$: Definition, 정의



## Calculus


## 미적분학

* $\frac{dy}{dx}$: Derivative of $y$ with respect to $x$ , $y$ 의 $x$ 에 대한 미분
* $\frac{\partial y}{\partial x}$: Partial derivative of $y$ with respect to $x$ , $y$ 의 $x$ 에 대한 편미분
* $\nabla_{\mathbf{x}} y$: Gradient of $y$ with respect to $\mathbf{x}$ , $y$ 의 $x$ 에 대한 그래디언트
* $\int_a^b f(x) \;dx$: Definite integral of $f$ from $a$ to $b$ with respect to $x$ , 구간 $[a, b]$ 에서 $f$ 의 $x$ 에 대한 정적분
* $\int f(x) \;dx$: Indefinite integral of $f$ with respect to $x$ , $f$ 의 $x$ 에 대한 부정적분



## Probability and Information Theory

## 확률 및 정보 이론

* $P(\cdot)$: Probability distribution , 확률 분포
* $z \sim P$: Random variable $z$ has probability distribution $P$ , 확률 변수 $z$ 는 확률 분포 $P$를 따른다.
* $P(X \mid Y)$: Conditional probability of $X \mid Y$ , $X \mid Y$ 의 조건부 확률
* $p(x)$: Probability density function , 확률 밀도 함수
* ${E}_{x} [f(x)]$: Expectation of $f$ with respect to $x$ , $x$ 에 대한 $f$ 의 [기댓값](https://ko.wikipedia.org/wiki/%EA%B8%B0%EB%8C%93%EA%B0%92)
* $X \perp Y$: Random variables $X$ and $Y$ are independent , 확률 변수 $X$ 와 $Y$ 가 [독립](https://ko.wikipedia.org/wiki/%EB%8F%85%EB%A6%BD_(%ED%99%95%EB%A5%A0%EB%A1%A0))이다.
* $X \perp Y \mid Z$: Random variables  $X$  and  $Y$  are conditionally independent given random variable $Z$ , 확률 변수 $X$ 와 $Y$ 가 주어진 확률 변수 $Z$ 에 대해 조건부 [독립](https://ko.wikipedia.org/wiki/%EB%8F%85%EB%A6%BD_(%ED%99%95%EB%A5%A0%EB%A1%A0))이다.
* $\mathrm{Var}(X)$: Variance of random variable $X$ , 확률 변수 $X$ 의 분산
* $\sigma_X$: Standard deviation of random variable $X$ , 확률 변수 $X$ 의 표준 편차
* $\mathrm{Cov}(X, Y)$: Covariance of random variables $X$ and $Y$ , 확률 변수 $X$ 와 $Y$ 의 [공분산](https://ko.wikipedia.org/wiki/%EA%B3%B5%EB%B6%84%EC%82%B0)
* $\rho(X, Y)$: Correlation of random variables $X$ and $Y$ , 확률 변수 $X$ 와 $Y$ 의 [상관관계](https://ko.wikipedia.org/wiki/%EC%83%81%EA%B4%80_%EB%B6%84%EC%84%9D)
* $H(X)$: Entropy of random variable $X$ , 확률 변수 $X$ 의 엔트로피
* $D_{\mathrm{KL}}(P\|Q)$: KL-divergence of distributions $P$ and $Q$ , 분포 $P$ 와 $Q$ 의 [쿨백-라이블러 발산](https://ko.wikipedia.org/wiki/%EC%BF%A8%EB%B0%B1-%EB%9D%BC%EC%9D%B4%EB%B8%94%EB%9F%AC_%EB%B0%9C%EC%82%B0)



## Complexity

## 복잡도

* $\mathcal{O}$: Big O notation , 대문자 O 표기법

[Discussions](https://discuss.d2l.ai/t/25)