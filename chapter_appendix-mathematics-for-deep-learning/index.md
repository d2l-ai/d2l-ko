# 부록: 딥러닝을 위한 수학
:label:`chap_appendix_math`

**브렌트 베르네스** (*아마존*), **레이첼 후** (*아마존*) 및 이 책의 저자

현대 딥 러닝의 놀라운 부분 중 하나는 그 아래의 수학에 대한 완전한 이해 없이도 그 대부분을 이해하고 사용할 수 있다는 사실입니다.이것은 필드가 성숙하고 있다는 신호입니다.대부분의 소프트웨어 개발자가 계산 가능한 함수 이론에 대해 더 이상 걱정할 필요가 없는 것처럼 딥 러닝 실무자도 최대 가능성 학습의 이론적 토대에 대해 걱정할 필요가 없습니다. 

하지만 우리는 아직 거기에 있지 않습니다. 

실제로 아키텍처 선택이 기울기 흐름에 어떤 영향을 미치는지 또는 특정 손실 함수로 훈련하여 내세우는 암시적 가정을 이해해야 할 때가 있습니다.엔트로피 측정 기준이 무엇인지, 모델에서 문자당 비트가 무엇을 의미하는지 정확히 이해하는 데 어떻게 도움이 되는지 알아야 할 수도 있습니다.이 모든 것은 더 깊은 수학적 이해가 필요합니다. 

이 부록은 최신 딥 러닝의 핵심 이론을 이해하는 데 필요한 수학적 배경을 제공하는 것을 목표로 하지만, 완전한 것은 아닙니다.선형 대수를 더 깊이 조사하는 것으로 시작하겠습니다.우리는 데이터에 대한 다양한 변환의 영향을 시각화할 수 있도록 모든 일반적인 선형 대수 객체와 연산에 대한 기하학적 이해를 개발합니다.핵심 요소는 고유 분해의 기본 사항을 개발하는 것입니다. 

다음으로 우리는 기울기가 가장 가파른 하강의 방향인 이유와 역 전파가 그 형태를 취하는 이유를 완전히 이해할 수있는 지점까지 미분 미적분 이론을 개발합니다.그런 다음 다음 주제 인 확률 이론을 뒷받침하는 데 필요한 정도까지 적분 미적분에 대해 논의합니다. 

실제로 자주 발생하는 문제는 확실하지 않으므로 불확실한 것에 대해 말할 언어가 필요합니다.확률 변수 이론과 가장 일반적으로 발생하는 분포를 검토하여 모델을 확률적으로 논의 할 수 있습니다.이는 확률적 분류 기법인 나이브 베이즈 분류기의 기초를 제공합니다. 

확률 이론과 밀접한 관련이있는 것은 통계 연구입니다.통계는 짧은 섹션에서 정의를 수행하기에는 너무 큰 분야이지만 모든 기계 학습 실무자가 알아야 할 기본 개념, 특히 추정기 평가 및 비교, 가설 검정 수행 및 신뢰 구간 구성을 소개합니다. 

마지막으로 정보 저장 및 전송에 대한 수학적 연구 인 정보 이론의 주제로 넘어갑니다.이것은 모델이 담론의 영역에서 얼마나 많은 정보를 보유하고 있는지 정량적으로 논의 할 수있는 핵심 언어를 제공합니다. 

이를 종합하면 딥 러닝에 대한 깊은 이해를 향한 길을 시작하는 데 필요한 수학적 개념의 핵심을 형성합니다.

```toc
:maxdepth: 2

geometry-linear-algebraic-ops
eigendecomposition
single-variable-calculus
multivariable-calculus
integral-calculus
random-variables
maximum-likelihood
distributions
naive-bayes
statistics
information-theory
```