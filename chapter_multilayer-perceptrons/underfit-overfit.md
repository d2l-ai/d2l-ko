# 모델 선택, 언더피팅 및 과적합
:label:`sec_model_selection`

머신 러닝 과학자로서 우리의 목표는*패턴*을 발견하는 것입니다.하지만 단순히 데이터를 암기하는 것이 아니라*일반적인* 패턴을 진정으로 발견했음을 어떻게 확신할 수 있을까요?예를 들어, 환자를 치매 상태와 연결하는 유전자 마커 중에서 패턴을 찾고 싶다고 상상해보십시오. 여기서 라벨은 $\{\text{dementia}, \text{mild cognitive impairment}, \text{healthy}\}$ 세트에서 추출됩니다.각 사람의 유전자는 유전자를 고유하게 식별하기 때문에 (동일한 형제를 무시함) 전체 데이터 세트를 기억할 수 있습니다. 

우리는 모델이 말하는 것을 원하지 않습니다.
*“저건 밥이야!그를 기억해요!치매가 있어요!”*
그 이유는 간단합니다.향후 모델을 배포하면 이전에는 볼 수 없었던 환자를 만나게 될 것입니다.예측은 모델이 진정으로*일반* 패턴을 발견한 경우에만 유용합니다. 

좀 더 공식적으로 요약하면, 우리의 목표는 훈련 세트가 그려진 기본 인구의 규칙성을 포착하는 패턴을 발견하는 것입니다.이러한 노력에 성공하면 이전에 만난 적이없는 개인에 대해서도 위험을 성공적으로 평가할 수 있습니다.이 문제, 즉 일반화* 패턴을 발견하는 방법은 기계 학습의 근본적인 문제입니다. 

위험은 모델을 훈련시킬 때 작은 데이터 샘플에만 액세스한다는 것입니다.가장 큰 공개 이미지 데이터 세트에는 약 백만 개의 이미지가 포함되어 있습니다.더 자주, 우리는 수천 또는 수만 개의 데이터 예제를 통해서만 배워야 합니다.대규모 병원 시스템에서는 수십만 건의 의료 기록에 접근할 수 있습니다.유한 샘플로 작업 할 때 더 많은 데이터를 수집 할 때 견디지 못하는 명백한 연관성을 발견 할 위험이 있습니다. 

기본 분포에 맞는 것보다 훈련 데이터를 더 가깝게 맞추는 현상을*과적합*이라고 하며 과적합을 방지하는 데 사용되는 기술을*정규화*라고 합니다.이전 섹션에서는 Fashion-MNIST 데이터세트를 실험하면서 이 효과를 관찰했을 수 있습니다.실험 중에 모델 구조 또는 초모수를 변경한 경우 뉴런, 계층 및 훈련 시대가 충분하면 테스트 데이터의 정확도가 떨어지더라도 모델이 결국 훈련 세트에서 완벽한 정확도에 도달할 수 있다는 것을 알았을 것입니다. 

## 훈련 오류 및 일반화 오류

이 현상을 좀 더 공식적으로 논의하려면 훈련 오류와 일반화 오류를 구별해야 합니다.*훈련 오류*는 훈련 데이터 세트에서 계산 된 모델의 오류이며, *일반화 오류*는 원래 샘플과 동일한 기본 데이터 분포에서 가져온 추가 데이터 예제의 무한 스트림에 적용 할 때 모델 오류의 예상입니다. 

문제는 일반화 오차를 정확히 계산할 수 없다는 것입니다.무한 데이터 스트림이 허수 객체이기 때문입니다.실제로 훈련 세트에서 보류된 데이터 예제의 무작위 선택으로 구성된 독립적인 테스트 세트에 모델을 적용하여 일반화 오차를*추정*해야 합니다. 

다음 세 가지 사고 실험은 이러한 상황을 더 잘 설명하는 데 도움이 될 것입니다.대학생이 최종 시험을 준비하려고 한다고 가정해 보십시오.부지런한 학생은 전년도 시험을 통해 잘 연습하고 자신의 능력을 시험하기 위해 노력할 것입니다.그럼에도 불구하고 과거 시험에서 잘 수행한다고해서 그가 중요 할 때 탁월하다는 보장은 없습니다.예를 들어, 학생은 시험 문제에 대한 답을 기계적 학습으로 준비하려고 할 수 있습니다.이를 위해서는 학생이 많은 것을 암기해야 합니다.그녀는 과거 시험에 대한 답을 완벽하게 기억할 수도 있습니다.다른 학생은 특정 답을 제공하는 이유를 이해하려고 노력하여 준비할 수 있습니다.대부분의 경우 후자의 학생이 훨씬 더 잘할 것입니다. 

마찬가지로 단순히 룩업 테이블을 사용하여 질문에 답하는 모델을 생각해 보십시오.허용 가능한 입력값 집합이 이산적이고 상당히 작다면 아마도*많은* 훈련 예제를 본 후에 이 접근법이 잘 수행될 것입니다.아직도이 모델은 이전에 본 적이없는 예제에 직면했을 때 무작위 추측보다 더 잘 수행 할 수있는 능력이 없습니다.실제로 입력 공간이 너무 커서 생각할 수있는 모든 입력에 해당하는 답변을 기억할 수 없습니다.예를 들어, 흑백 $28\times28$ 이미지가 있다고 가정해 보겠습니다.각 픽셀이 $256$개의 회색조 값 중 하나를 취할 수 있는 경우 $256^{784}$개의 가능한 이미지가 있습니다.즉, 우주에 있는 원자보다 해상도가 낮은 회색조 썸네일 크기의 이미지가 훨씬 더 많습니다.이러한 데이터를 접할 수 있더라도 룩업 테이블을 저장할 여유가 없습니다. 

마지막으로, 사용 가능한 몇 가지 상황에 맞는 기능을 기반으로 동전 던지기 (클래스 0: 머리, 클래스 1: 꼬리) 의 결과를 분류하려는 문제를 고려하십시오.동전이 공정하다고 가정합니다.어떤 알고리즘을 생각해 내더라도 일반화 오류는 항상 $\frac{1}{2}$입니다.그러나 대부분의 알고리즘에서는 기능이 없더라도 무승부의 운에 따라 훈련 오차가 상당히 줄어들 것으로 예상해야 합니다!데이터세트 {0, 1, 1, 1, 0, 1} 을 고려합니다.기능이 없는 알고리즘은 제한된 샘플에서*1*로 나타나는*다수 클래스*를 항상 예측하는 것으로 되돌아 가야 합니다.이 경우 항상 클래스 1을 예측하는 모델은 일반화 오류보다 훨씬 나은 $\frac{1}{3}$의 오류를 발생시킵니다.데이터 양을 늘리면 머리 부분이 $\frac{1}{2}$에서 크게 벗어날 확률이 줄어들고 훈련 오류가 일반화 오차와 일치합니다. 

### 통계 학습 이론

일반화는 기계 학습의 근본적인 문제이기 때문에 많은 수학자와 이론가들이 이러한 현상을 설명하는 공식 이론을 개발하는 데 평생을 바쳤다는 사실에 놀라지 않을 것입니다.[시조 정리](https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_theorem) 에서 글리벤코와 칸텔리는 훈련 오류가 일반화 오류로 수렴되는 속도를 도출했습니다.일련의 정액 논문에서 [Vapnik과 체르보넨키스](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory) 는이 이론을 좀 더 일반적인 함수 클래스로 확장했습니다.이 연구는 통계 학습 이론의 토대를 마련했습니다. 

지금까지 다루었으며 이 책의 대부분을 고수할 표준 지도 학습 환경에서는 훈련 데이터와 테스트 데이터가 모두*동일한* 분포에서*독립적으로* 그려진다고 가정합니다.이를 일반적으로*i.i.d. 가정*이라고 하는데, 이는 데이터를 샘플링하는 프로세스에 메모리가 없음을 의미합니다.즉, 그려진 두 번째 예제와 세 번째 그려진 예제는 그려진 두 번째 샘플과 200 만 번째 샘플보다 더 이상 상관 관계가 없습니다. 

훌륭한 기계 학습 과학자가 되려면 비판적으로 사고해야 하며, 가정이 실패하는 일반적인 경우를 생각해 내면서 이미 이 가정에 구멍을 뚫고 있어야 합니다.UCSF Medical Center의 환자로부터 수집한 데이터에 대한 사망 위험 예측 변수를 교육하고 매사추세츠 종합 병원의 환자에게 적용하면 어떻게 될까요?이러한 분포는 단순히 동일하지 않습니다.또한 무승부는 시간에 따라 상관 관계가 있을 수 있습니다.트윗의 주제를 분류하는 경우 어떻게 해야 하나요?뉴스 사이클은 논의되는 주제에 시간적 의존성을 유발하여 독립성에 대한 가정을 위반합니다. 

때때로 우리는 i.i.d. 가정에 대한 사소한 위반으로 벗어날 수 있으며 모델은 계속해서 현저하게 잘 작동 할 것입니다.결국 거의 모든 실제 응용 프로그램에는 최소한 i.id 가정에 대한 사소한 위반이 포함되지만 얼굴 인식, 음성 인식 및 언어 번역과 같은 다양한 응용 프로그램에 유용한 도구가 많이 있습니다. 

다른 위반으로 인해 문제가 발생할 수 있습니다.예를 들어, 대학생에게만 얼굴 인식 시스템을 교육하여 훈련 한 다음 요양원 인구의 노인병을 모니터링하는 도구로 배포하려는 경우를 상상해보십시오.대학생들은 노인과 상당히 다르게 보이는 경향이 있기 때문에 잘 작동하지 않을 것입니다. 

다음 장에서는 i.i.d. 가정 위반으로 인해 발생하는 문제에 대해 설명합니다.현재로서는 i.i.d. 가정을 당연하게 여긴다고 하더라도 일반화를 이해하는 것은 엄청난 문제입니다.더욱이, 심층 신경망이 일반화되고 일반화되는 이유를 설명 할 수있는 정확한 이론적 토대를 밝히는 것은 학습 이론에서 가장 위대한 마음을 계속 괴롭 히고 있습니다. 

모델을 훈련시킬 때 가능한 한 훈련 데이터에 맞는 함수를 검색하려고 시도합니다.함수가 너무 유연하여 실제 연관성만큼 쉽게 스퓨리어스 패턴을 포착할 수 있다면 보이지 않는 데이터를 잘 일반화하는 모델을 생성하지 않고*너무 잘* 수행할 수 있습니다.이것이 바로 우리가 피하거나 최소한 통제하고 싶은 것입니다.딥 러닝의 많은 기법은 과적합을 방지하기 위한 휴리스틱과 트릭입니다. 

### 모델 복잡성

간단한 모델과 풍부한 데이터가 있는 경우 일반화 오류가 훈련 오류와 비슷할 것으로 예상됩니다.더 복잡한 모델과 더 적은 예제로 작업할 때 훈련 오류는 줄어들지만 일반화 격차는 커질 것으로 예상됩니다.모델 복잡성을 정확하게 구성하는 것은 복잡한 문제입니다.모델이 잘 일반화되는지 여부는 많은 요소가 좌우됩니다.예를 들어 모수가 더 많은 모델은 더 복잡한 것으로 간주될 수 있습니다.매개변수가 더 넓은 범위의 값을 취할 수 있는 모델은 더 복잡할 수 있습니다.신경망에서는 종종 훈련 반복을 더 많이 취하는 모델이 더 복잡하고*조기 중단* (훈련 반복 횟수 감소) 이 적용되는 모델은 덜 복잡하다고 생각합니다. 

실질적으로 다른 모델 클래스 (예: 의사 결정 트리와 신경망) 의 구성원 간의 복잡성을 비교하는 것은 어려울 수 있습니다.현재로서는 간단한 경험 법칙이 매우 유용합니다. 임의의 사실을 쉽게 설명 할 수있는 모델은 통계학자가 복잡하다고 생각하는 반면 표현력은 제한적이지만 여전히 데이터를 잘 설명하는 모델은 진실에 더 가까울 것입니다.철학에서 이것은 Popper의 과학 이론의 위조 가능성 기준과 밀접한 관련이 있습니다. 이론은 데이터에 적합하고 반증하는 데 사용할 수있는 특정 테스트가 있으면 이론이 좋습니다.이는 모든 통계적 추정치가 다음과 같으므로 중요합니다.
*포스트호*,
즉, 사실을 관찰 한 후 추정하므로 관련 오류에 취약합니다.지금은 철학을 제쳐두고 더 실질적인 문제를 고수 할 것입니다. 

이 섹션에서는 몇 가지 직관을 제공하기 위해 모델 클래스의 일반화에 영향을 미치는 몇 가지 요소에 중점을 둘 것입니다. 

1. 조정 가능한 매개변수의 수입니다.*자유도*라고도 하는 조정 가능한 매개변수의 수가 많으면 모델이 과적합에 더 취약한 경향이 있습니다.
1. 매개 변수에서 가져온 값입니다.가중치가 더 넓은 범위의 값을 취할 수 있는 경우 모델은 과적합에 더 취약할 수 있습니다.
1. 교육 예제의 수입니다.모델이 단순하더라도 하나 또는 두 개의 예만 포함된 데이터세트를 과적합하는 것은 매우 쉽습니다.그러나 수백만 개의 예제로 데이터세트를 과적합하려면 매우 유연한 모델이 필요합니다.

## 모델 선택

머신 러닝에서는 일반적으로 여러 후보 모델을 평가한 후 최종 모델을 선택합니다.이 프로세스를*모델 선택*이라고 합니다.때때로 비교 대상 모델은 본질적으로 근본적으로 다릅니다 (예: 의사 결정 트리와 선형 모델).다른 경우에는 서로 다른 하이퍼파라미터 설정으로 훈련된 동일한 모델 클래스의 멤버를 비교합니다. 

예를 들어 MLP를 사용하면 은닉 레이어 수, 은닉 유닛 수, 각 히든 레이어에 적용된 다양한 활성화 기능 선택을 가진 모델을 비교할 수 있습니다.후보 모델 중에서 가장 적합한 모델을 결정하기 위해 일반적으로 검증 데이터 세트를 사용합니다. 

### 검증 데이터세트

원칙적으로 모든 초매개변수를 선택할 때까지 테스트 세트를 건드리지 않아야 합니다.모델 선택 과정에서 검정 데이터를 사용하면 검정 데이터를 과적합시킬 위험이 있습니다.그러면 우리는 심각한 곤경에 처할 것입니다.훈련 데이터를 과도하게 맞추면 정직하게 유지하기 위해 항상 테스트 데이터에 대한 평가가 있습니다.하지만 테스트 데이터를 과적합하면 어떻게 알 수 있을까요? 

따라서 모델 선택을 위해 테스트 데이터에 의존해서는 안됩니다.그러나 모델을 훈련시키는 데 사용하는 바로 그 데이터에 대한 일반화 오차를 추정 할 수 없기 때문에 모델 선택을 위해 훈련 데이터에만 의존 할 수는 없습니다. 

실제 응용 프로그램에서는 그림이 더 흐려집니다.가장 적합한 모델을 평가하거나 소수의 모델을 서로 비교하기 위해 테스트 데이터를 한 번만 터치하는 것이 이상적이지만 실제 테스트 데이터는 한 번만 사용한 후에 폐기되는 경우는 거의 없습니다.각 실험 라운드마다 새로운 테스트 세트를 구입할 여유가 거의 없습니다. 

이 문제를 해결하는 일반적인 관행은 훈련 및 테스트 데이터 세트 외에*검증 데이터 세트* (또는*검증 세트*) 를 통합하는 세 가지 방법으로 데이터를 분할하는 것입니다.그 결과 검증과 테스트 데이터 간의 경계가 걱정스럽게 모호해지는 어두운 관행이 탄생했습니다.명시적으로 달리 명시하지 않는 한, 이 책의 실험에서 우리는 실제 테스트 세트가 없는 훈련 데이터 및 검증 데이터라고 올바르게 불려야 하는 것에 대해 실제로 작업하고 있습니다.따라서 책의 각 실험에서 보고된 정확도는 실제로 검증 정확도이며 실제 테스트 세트 정확도가 아닙니다. 

### $K$겹 교차 검증

훈련 데이터가 부족한 경우 적절한 검증 세트를 구성하기에 충분한 데이터를 보유하지 못할 수도 있습니다.이 문제에 대한 일반적인 해결책 중 하나는 $K$*-겹 교차 검증*을 사용하는 것입니다.여기서 원래 훈련 데이터는 $K$개의 겹치지 않는 부분 집합으로 분할됩니다.그런 다음 모델 학습 및 검증이 $K$ 번 실행되며, 매번 $K-1$ 하위 집합에 대해 학습하고 다른 하위 집합 (해당 라운드에서 훈련에 사용되지 않은 하위 집합) 에서 유효성을 검사합니다.마지막으로 훈련 및 검증 오류는 $K$ 실험 결과에 대한 평균을 계산하여 추정합니다. 

## 언더피팅 또는 과적합?

훈련 오류와 검증 오류를 비교할 때 두 가지 일반적인 상황을 염두에 두고 싶습니다.먼저 훈련 오류와 검증 오류가 모두 상당하지만 둘 사이에 약간의 차이가 있는 경우를 주의해야 합니다.모델이 훈련 오류를 줄일 수 없다면 모델이 너무 단순하여 (즉, 표현력이 부족함) 모델링하려는 패턴을 포착할 수 없음을 의미할 수 있습니다.또한 훈련 오류와 검증 오류 사이의*일반화 격차*가 작기 때문에 더 복잡한 모델에서 벗어날 수 있다고 믿을만한 이유가 있습니다.이 현상을*언더피팅*이라고 합니다. 

반면에 위에서 논의한 것처럼 훈련 오류가 검증 오류보다 현저히 낮아 심각한*과적합*을 나타내는 경우를 주의하고 싶습니다.과적합이 항상 나쁜 것은 아닙니다.특히 딥러닝에서는 최상의 예측 모델이 홀드아웃 데이터보다 훈련 데이터에서 훨씬 더 나은 성능을 발휘한다는 것이 잘 알려져 있습니다.궁극적으로 우리는 일반적으로 훈련 오류와 검증 오류 사이의 간격보다 검증 오차에 더 관심이 있습니다. 

과적합이든 언더핏이든 모델의 복잡성과 사용 가능한 교육 데이터 세트의 크기에 따라 달라질 수 있습니다. 아래에서 논의하는 두 가지 주제입니다. 

### 모델 복잡성

과적합과 모델 복잡성에 대한 몇 가지 고전적 직관을 설명하기 위해 다항식을 사용한 예제를 제공합니다.단일 특징 $x$와 해당 실수 값 레이블 $y$으로 구성된 훈련 데이터가 주어지면 차수 $d$의 다항식을 찾으려고 시도합니다. 

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

레이블 $y$를 추정합니다.이것은 우리의 특징이 $x$의 거듭제곱에 의해 주어지고, 모델의 가중치는 $w_i$로 주어지며, 모든 $x$에 대해 $x^0 = 1$ 이후 $w_0$에 의해 치우침이 주어지는 선형 회귀 문제입니다.이것은 단지 선형 회귀 문제이므로 제곱 오차를 손실 함수로 사용할 수 있습니다. 

고차 다항식에는 더 많은 매개 변수가 있고 모델 함수의 선택 범위가 더 넓기 때문에 고차 다항식 함수는 저차 다항식 함수보다 복잡합니다.훈련 데이터 세트를 고정하면 고차 다항식 함수는 항상 낮은 차수 다항식에 비해 더 낮은 (최악의 경우 동일한) 훈련 오차를 달성해야합니다.실제로 데이터 예제의 고유한 값이 $x$일 때마다 차수가 데이터 예제의 수와 같은 다항식 함수가 훈련 세트를 완벽하게 적합할 수 있습니다.:numref:`fig_capacity_vs_error`에서 다항식 차수와 과소 적합 대 과적합 간의 관계를 시각화합니다. 

![Influence of model complexity on underfitting and overfitting](../img/capacity-vs-error.svg)
:label:`fig_capacity_vs_error`

### 데이터세트 크기

명심해야 할 또 다른 중요한 사항은 데이터세트 크기입니다.모델을 수정하면 훈련 데이터셋에 있는 샘플이 적을수록 과적합이 발생할 가능성이 더 커지고 더 심각해집니다.훈련 데이터의 양을 늘리면 일반적으로 일반화 오차가 감소합니다.또한 일반적으로 더 많은 데이터가 손상되지 않습니다.고정 작업 및 데이터 배포의 경우 일반적으로 모델 복잡성과 데이터세트 크기 간에 관계가 있습니다.더 많은 데이터가 주어지면 더 복잡한 모델을 적합하게 시도할 수 있습니다.충분한 데이터가 없으면 더 간단한 모델을 이길 수 있습니다.많은 작업에서 딥러닝은 수천 개의 훈련 예제를 사용할 수 있을 때 선형 모델보다 성능이 뛰어납니다.딥 러닝의 현재 성공은 인터넷 회사, 저렴한 스토리지, 연결된 장치 및 광범위한 경제 디지털화로 인해 현재 방대한 데이터 세트가 풍부하기 때문입니다. 

## 다항식 회귀

이제 (**다항식을 데이터에 피팅하여 이러한 개념을 대화식으로 탐색**) 할 수 있습니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import math
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np
import math
```

### 데이터세트 생성

먼저 데이터가 필요합니다.$x$가 주어지면 훈련 및 테스트 데이터에 대해 [**다음 3차 다항식을 사용하여 레이블을 생성**] 할 것입니다. 

(**$y = 5 + 1.2x - 3.4\ 프락 {x^2} {2!}+ 5.6\ 프락 {x^3} {3!}+\ 엡실론\ 텍스트 {여기서}\ 엡실론\ 심\ 수학 {N} (0, 0.1^2) .$$**) 

잡음 항 $\epsilon$는 평균이 0이고 표준 편차가 0.1인 정규 분포를 따릅니다.최적화를 위해 일반적으로 매우 큰 기울기 또는 손실 값을 피하려고 합니다.그렇기 때문에*기능*이 $x^i$에서 $\ frac {x^i} {i!} 로 다시 스케일링됩니다.$.이렇게 하면 큰 지수 $i$에 대해 매우 큰 값을 피할 수 있습니다.훈련 세트와 테스트 세트에 대해 각각 100개의 샘플을 합성합니다.

```{.python .input}
#@tab all
max_degree = 20  # Maximum degree of the polynomial
n_train, n_test = 100, 100  # Training and test dataset sizes
true_w = np.zeros(max_degree)  # Allocate lots of empty space
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # `gamma(n)` = (n-1)!
# Shape of `labels`: (`n_train` + `n_test`,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
```

다시 말하지만, `poly_features`에 저장된 단항식은 감마 함수에 의해 다시 스케일링됩니다. 여기서 $\ 감마 (n) = (n-1)!$.생성된 데이터셋에서 [**처음 2개의 샘플을 살펴보세요**].값 1은 기술적으로 특징, 즉 편향에 해당하는 상수 특성입니다.

```{.python .input}
#@tab pytorch, tensorflow
# Convert from NumPy ndarrays to tensors
true_w, features, poly_features, labels = [d2l.tensor(x, dtype=
    d2l.float32) for x in [true_w, features, poly_features, labels]]
```

```{.python .input}
#@tab all
features[:2], poly_features[:2, :], labels[:2]
```

### 모델 교육 및 테스트

먼저 [**주어진 데이터 세트에 대한 손실을 평가하는 함수를 구현**] 하겠습니다.

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_loss(net, data_iter, loss):  #@save
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        l = loss(net(X), y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_loss(net, data_iter, loss):  #@save
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = d2l.reshape(y, out.shape)
        l = loss(out, y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]
```

이제 [**훈련 함수를 정의**] 합니다.

```{.python .input}
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = gluon.loss.L2Loss()
    net = nn.Sequential()
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net.add(nn.Dense(1, use_bias=False))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size,
                               is_train=False)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data().asnumpy())
```

```{.python .input}
#@tab pytorch
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
```

```{.python .input}
#@tab tensorflow
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = tf.losses.MeanSquaredError()
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1, use_bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size,
                               is_train=False)
    trainer = tf.keras.optimizers.SGD(learning_rate=.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net.get_weights()[0].T)
```

### [**3차 다항식 함수 피팅 (정규) **]

먼저 데이터 생성 함수와 동일한 순서인 3차 다항식 함수를 사용하는 것으로 시작하겠습니다.결과는 이 모델의 훈련 및 테스트 손실을 모두 효과적으로 줄일 수 있음을 보여줍니다.학습된 모델 매개변수도 실제 값 $w = [5, 1.2, -3.4, 5.6]$에 가깝습니다.

```{.python .input}
#@tab all
# Pick the first four dimensions, i.e., 1, x, x^2/2!, x^3/3! from the
# polynomial features
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
```

### [**선형 함수 피팅 (언더피팅) **]

선형 함수 피팅에 대해 다시 살펴 보겠습니다.초기 시대가 감소한 후에는 이 모델의 훈련 손실을 더 줄이기가 어려워졌습니다.마지막 epoch 반복이 완료된 후에도 훈련 손실은 여전히 높습니다.비선형 패턴 (예: 3차 다항식 함수) 을 피팅하는 데 사용되는 경우 선형 모델은 과소적합되기 쉽습니다.

```{.python .input}
#@tab all
# Pick the first two dimensions, i.e., 1, x, from the polynomial features
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
```

### [**고차 다항식 함수 피팅 (과적합) **]

이제 차수가 너무 높은 다항식을 사용하여 모델을 훈련해 보겠습니다.여기서는 차수가 높은 계수가 0에 가까운 값을 가져야 한다는 것을 알기에 충분한 데이터가 없습니다.결과적으로 지나치게 복잡한 모델은 매우 취약하여 훈련 데이터의 노이즈의 영향을 받고 있습니다.훈련 손실은 효과적으로 줄일 수 있지만 테스트 손실은 여전히 훨씬 높습니다.복잡한 모형이 데이터를 과적합한다는 것을 보여줍니다.

```{.python .input}
#@tab all
# Pick all the dimensions from the polynomial features
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)
```

이후 섹션에서는 과적합 문제와 중량 감소 및 드롭아웃과 같은 문제를 처리하는 방법에 대해 계속 논의할 것입니다. 

## 요약

* 훈련 오차를 기준으로 일반화 오차를 추정할 수 없기 때문에 단순히 훈련 오류를 최소화한다고 해서 반드시 일반화 오차가 줄어드는 것은 아닙니다.기계 학습 모델은 일반화 오류를 최소화하기 위해 과적합을 방지하기 위해 주의를 기울여야 합니다.
* 검증 세트는 너무 많이 사용되지 않는 경우 모델 선택에 사용할 수 있습니다.
* 과소적합은 모델이 훈련 오차를 줄일 수 없음을 의미합니다.훈련 오차가 검증 오차보다 훨씬 낮은 경우 과적합이 발생합니다.
* 적절하게 복잡한 모델을 선택하고 훈련 샘플을 불충분하게 사용하지 않아야 합니다.

## 연습문제

1. 다항식 회귀 문제를 정확히 풀 수 있습니까?힌트: 선형 대수를 사용합니다.
1. 다항식의 모델 선택을 고려하십시오.
    1. 훈련 손실 대 모델 복잡도 (다항식의 차수) 를 플로팅합니다.무엇을 관찰하시나요?훈련 손실을 0으로 줄이려면 어느 정도의 다항식이 필요합니까?
    1. 이 경우 테스트 손실을 플로팅합니다.
    1. 데이터 양의 함수와 동일한 플롯을 생성합니다.
1. 정규화를 삭제하면 어떻게 되나요 ($1/i!$) of the polynomial features $x^i$?이 문제를 다른 방법으로 해결할 수 있습니까?
1. 일반화 오류가 전혀 없을 것으로 예상할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:
