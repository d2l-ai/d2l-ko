# 모델 선택, 언더피팅 및 오버피팅
:label:`sec_model_selection`

기계 학습 과학자로서 우리의 목표는 *패턴*을 발견하는 것입니다.그러나 단순히 데이터를 암기하지 않고* 일반* 패턴을 진정으로 발견했는지 어떻게 확신 할 수 있습니까?예를 들어, 우리는 환자의 치매 상태를 연결하는 유전자 표지자 중 패턴을 찾고 싶다고 상상해보십시오. $\{\text{dementia}, \text{mild cognitive impairment}, \text{healthy}\}$ 세트에서 레이블이 그려집니다.각 사람의 유전자가 고유하게 식별하기 때문에 (동일한 형제를 무시하고), 전체 데이터 세트를 암기 할 수있다.

우리는 우리의 모델이 말하기를 원하지 않습니다.
*“밥이야!난 그를 기억해!그는 치매를 가지고 있습니다!”*
그 이유는 간단합니다.우리가 미래에 모델을 배포하면, 우리는 그 모델이 전에 보지 못했던 환자들을 만날 것입니다.우리의 예측은 우리 모델이 진정으로 * 일반* 패턴을 발견 한 경우에만 유용 할 것입니다.

보다 공식적으로 다시 되풀이하기 위해, 우리의 목표는 우리의 훈련 세트가 그려진 기본 인구의 규칙성을 포착하는 패턴을 발견하는 것입니다.우리가 이 노력에 성공한다면, 우리는 이전에 만난 적이없는 개인에 대해서도 위험을 성공적으로 평가할 수 있습니다.이 문제, 즉 일반화*하는 패턴을 발견하는 방법은 기계 학습의 근본적인 문제입니다.

위험은 모델을 훈련 할 때 작은 데이터 샘플에 액세스한다는 것입니다.가장 큰 공용 이미지 데이터 세트에는 약 백만 개의 이미지가 포함되어 있습니다.종종 수천 또는 수만 개의 데이터 포인트에서만 배워야합니다.대규모 병원 시스템에서 우리는 수십만 개의 의료 기록에 접근할 수 있습니다.유한 샘플로 작업할 때 더 많은 데이터를 수집할 때 보류하지 않는 명백한 연관성을 발견할 수 있는 위험을 감수합니다.

우리의 훈련 데이터를 기본 분포에 적합시키는 것보다 더 밀접하게 맞추는 현상을 *overfitting*이라고 하며, 과잉 적합에 대처하기 위해 사용되는 기술을*정규화*라고합니다.이전 섹션에서는 Fashion-Mnist 데이터 집합을 실험하는 동안 이 효과를 관찰했을 수 있습니다.실험 중에 모델 구조 또는 하이퍼파라미터를 변경한 경우 충분한 뉴런, 레이어 및 학습 신기구가 있으면 테스트 데이터의 정확도가 저하되더라도 모형이 학습 집합에서 완벽한 정확도에 도달할 수 있다는 것을 알 수 있습니다.

## 교육 오류 및 일반화 오류

보다 공식적으로이 현상을 논의하기 위해, 우리는 훈련 오류와 일반화 오류를 구별해야합니다.* 훈련 오류*는 교육 데이터 집합에서 계산 된 우리 모델의 오류입니다.* 일반화 오류*는 우리 모델의 오류에 대한 기대는 동일한 기본 데이터 배포에서 가져온 추가 데이터 포인트의 무한 스트림에 적용 할 것입니다 우리의 원래 샘플입니다.

문제적으로, 우리는 일반화 오류를 정확하게 계산할 수 없습니다.무한 데이터의 스트림이 허상의 객체이기 때문입니다.실제로, 우리는 우리의 훈련 세트에서 원천 징수 된 데이터 포인트의 임의의 선택으로 구성 된 독립적 인 테스트 세트에 우리의 모델을 적용하여 일반화 오류를*추정* 해야합니다.

다음 세 가지 생각 실험은이 상황을 더 잘 설명하는 데 도움이됩니다.최종 시험을 준비하는 대학생을 생각해 보십시오.부지런한 학생은 지난 몇 년 동안의 시험을 사용하여 잘 연습하고 자신의 능력을 시험하기 위해 노력할 것입니다.그럼에도 불구하고, 과거 시험에서 잘하는 것은 그것이 중요 할 때 그녀가 뛰어날 것이라는 보장은 없습니다.예를 들어, 학생은 시험 문항에 대한 답을 학습하여 로테로 준비하려고 할 수 있습니다.이것은 학생이 많은 것을 암기해야합니다.그녀는 심지어 과거 시험에 대한 답을 완벽하게 기억할 수도 있습니다.다른 학생은 특정 답변을 제공하는 이유를 이해하려고 노력함으로써 준비 할 수 있습니다.대부분의 경우, 후자의 학생은 훨씬 더 잘할 것입니다.

마찬가지로 단순히 조회 테이블을 사용하여 질문에 대답하는 모델을 고려하십시오.허용 입력 집합이 이산적이고 합리적으로 작 으면* 많은* 교육 예제를 본 후에도이 방법이 잘 수행됩니다.여전히이 모델은 이전에 본 적이없는 예제에 직면 할 때 무작위 추측보다 더 잘 할 수있는 능력이 없습니다.실제로 입력 공간은 생각할 수있는 모든 입력에 해당하는 답변을 암기하기에는 너무 큽니다.예를 들어, 흑백 $28\times28$ 이미지를 생각해 보십시오.각 픽셀이 $256$ 회색 음영 값 중 하나를 취할 수 있다면 $256^{784}$ 가능한 이미지가 있습니다.즉, 우주에 원자가있는 것보다 훨씬 더 낮은 해상도의 그레이 스케일 썸네일 크기의 이미지가 있다는 것을 의미합니다.이러한 데이터가 발생할 수 있더라도 조회 테이블을 저장할 여유가 없습니다.

마지막으로, 동전 던지기 (클래스 0: 머리, 클래스 1: 꼬리) 의 결과를 사용할 수있는 상황에 맞는 기능을 기반으로 분류하는 문제를 고려하십시오.동전이 공정하다고 가정 해보십시오.우리가 어떤 알고리즘을 생각해 내더라도 일반화 오류는 항상 $\frac{1}{2}$입니다.그러나 대부분의 알고리즘에서는 기능이 없더라도 무승부의 운에 따라 교육 오류가 상당히 낮을 것으로 예상됩니다.데이터 세트 {0, 1, 1, 1, 1, 0, 1} 를 고려하십시오.우리의 기능이없는 알고리즘은 제한된 샘플에서* 1*로 나타나는 * 과반수 클래스*를 항상 예측해야합니다.이 경우 클래스 1을 항상 예측하는 모델에는 일반화 오류보다 훨씬 나은 $\frac{1}{3}$의 오류가 발생합니다.데이터 양을 늘리면 머리 부분이 $\frac{1}{2}$에서 크게 벗어날 확률이 줄어들고 교육 오류가 일반화 오류와 일치하게됩니다.

### 통계 학습 이론

일반화가 기계 학습의 근본적인 문제이기 때문에 많은 수학자와 이론가가이 현상을 설명하기 위해 공식적인 이론을 개발하는 데 자신의 삶을 바쳤다는 것을 알게되면 놀라지 않을 것입니다.[시조 정리](https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_theorem) 에서 Glivenko와 Cantelli는 훈련 오류가 일반화 오류에 수렴하는 비율을 도출했습니다.일련의 정액 논문에서, [Vapnik과 Chervonenkis](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory) 보다 일반적인 기능 클래스에이 이론을 확장했습니다.이 작품은 통계 학습 이론의 기초를 마련했다.

지금까지 다루어 왔으며이 책의 대부분을 고수 할 표준 감독 학습 설정에서 교육 데이터와 테스트 데이터가*동일한* 배포판에서*독립적으로* 그려진다고 가정합니다.이를 일반적으로*i.d. 가정*이라고하며, 이는 데이터를 샘플링하는 프로세스에 메모리가 없음을 의미합니다.즉, 그려진 두 번째 예와 세 번째 그려진 것은 그려진 두 번째 샘플과 2 백만 번째 샘플보다 더 이상 상관 관계가 없습니다.

좋은 기계 학습 과학자가된다는 것은 비판적으로 생각해야하며 이미이 가정에 구멍을 뚫고 가정이 실패한 일반적인 경우를 생각해 내야합니다.UCSF 메디컬 센터 (Medical Center) 의 환자로부터 수집한 데이터에 대한 사망률 예측 변수를 훈련시키고 매사추세츠 종합병원 환자들에게 적용한다면 어떨까요?이러한 분포는 단순히 동일하지 않습니다.또한, 무승부 시간에 상관 될 수 있습니다.트윗의 주제를 분류하면 어떻게 될까요?뉴스 사이클은 논의되는 주제에 시간적 의존성을 만들어 독립성에 대한 가정을 위반합니다.

때때로 우리는 가정의 사소한 위반으로 도망 갈 수 있으며 우리의 모델은 계속해서 현저하게 잘 작동 할 것입니다.결국, 거의 모든 실제 응용 프로그램은 즉 가정의 적어도 약간의 사소한 위반을 포함하지만, 우리는 얼굴 인식, 음성 인식 및 언어 번역과 같은 다양한 응용 프로그램에 대한 많은 유용한 도구를 가지고 있습니다.

다른 위반은 문제를 일으킬 수 있습니다.예를 들어, 대학생들에게 독점적으로 교육하여 얼굴 인식 시스템을 훈련시키고 요양원 인구의 노화학을 모니터링하기 위한 도구로 배포하려는 경우를 상상해보십시오.대학생은 노인과 상당히 다른 경향이 있기 때문에 이것은 잘 작동하지 않을 수 있습니다.

다음 장에서, 우리는 i.i.d. 가정의 위반에서 발생하는 문제를 설명합니다.당연하게 가정하는 것조차도 일반화를 이해하는 것은 엄청난 문제입니다.더욱이, 깊은 신경망이 일반화되는 이유를 설명 할 수있는 정확한 이론적 토대를 설명하면 학습 이론에서 가장 큰 마음을 계속 얻게됩니다.

모델을 학습할 때 가능한 한 교육 데이터에 맞는 함수를 검색하려고 시도합니다.함수가 너무 유연하여 진정한 연관성처럼 쉽게 가짜 패턴을 잡을 수 있다면 보이지 않는 데이터로 잘 일반화되는 모델을 생성하지 않고*도 잘 수행 할 수 있습니다.이것은 정확하게 우리가 피하거나 적어도 통제하기를 원하는 것입니다.딥 러닝의 많은 기술은 과잉 피팅을 방지하기 위한 휴리스틱과 트릭입니다.

### 모델 복잡성

우리는 간단한 모델과 풍부한 데이터가있을 때, 우리는 일반화 오류가 훈련 오류와 유사 할 것으로 예상된다.보다 복잡한 모델과 적은 예제를 사용하여 작업 할 때 교육 오류는 줄어들지만 일반화 격차는 커질 것으로 예상됩니다.모델 복잡성을 정확하게 구성하는 것은 복잡한 문제입니다.모형이 잘 일반화되는지 여부가 많은 요인이 적용됩니다.예를 들어 매개변수가 더 많은 모형은 더 복잡한 것으로 간주될 수 있습니다.매개변수가 더 넓은 범위의 값을 사용할 수 있는 모형은 더 복잡할 수 있습니다.종종 신경망을 사용하면 더 많은 훈련 반복을 더 복잡하게 취하는 모델을 생각하고*조기 중단* (훈련 반복 횟수가 적음) 의 대상이 덜 복잡하다고 생각합니다.

실질적으로 다른 모델 클래스 (예: 의사 결정 트리 대 신경망) 의 멤버 간의 복잡성을 비교하는 것은 어려울 수 있습니다.지금은 간단한 경험 법칙이 매우 유용합니다. 임의의 사실을 쉽게 설명 할 수있는 모델은 통계학자가 복잡한 것으로 보는 반면 제한된 표현력 만 가지고 있지만 데이터를 잘 설명하는 모델은 아마도 진실에 더 가깝습니다.철학에서 이것은 과학 이론의 위조 가능성에 대한 포퍼의 기준과 밀접한 관련이 있습니다. 데이터에 적합하고이를 반증하는 데 사용할 수있는 특정 테스트가있는 경우 이론이 좋습니다.모든 통계 추정 때문에 이것은 중요하다
*포스트 호크*,
우리가 관련 오류에 따라서 취약한 사실을 관찰 한 후 즉, 우리는 추정한다.지금은 철학을 제쳐두고 더 확실한 문제에 고수할 것입니다.

이 섹션에서는, 당신에게 몇 가지 직관을 제공하기 위해, 우리는 모델 클래스의 일반화에 영향을 미치는 경향이있는 몇 가지 요인에 초점을 맞출 것이다:

1. 조정 가능한 매개 변수의 수입니다.*자유도*라고도 하는 조정 가능한 모수의 수가 클 경우 모형은 과대 적합에 더 취약합니다.
1. 매개 변수에 의해 취해진 값.가중치가 더 넓은 범위의 값을 사용할 수 있는 경우 모형이 과대 적합에 더 민감할 수 있습니다.
1. 교육 예제 수입니다.모델이 단순하더라도 하나 또는 두 개의 예제 만 포함하는 데이터 집합을 과도하게 맞추는 것은 쉽습니다.그러나 수백만 개의 예제로 데이터 세트를 과도하게 맞추려면 매우 유연한 모델이 필요합니다.

## 모델 선택

기계 학습에서 우리는 일반적으로 여러 후보 모델을 평가 한 후 최종 모델을 선택합니다.이 프로세스를 *모델 선택*이라고 합니다.때로는 비교가 필요한 모델이 본질적으로 다릅니다 (예: 의사 결정 트리 대 선형 모델).다른 시간에, 우리는 다른 하이퍼 매개 변수 설정으로 훈련 된 모델의 동일한 클래스의 멤버를 비교하고 있습니다.

예를 들어 MLP를 사용하면 숨겨진 레이어 수가 서로 다른 모델, 숨겨진 단위 수, 숨겨진 각 레이어에 적용된 다양한 활성화 기능 선택 항목을 비교할 수 있습니다.후보 모델 중 최고를 결정하기 위해 일반적으로 유효성 검사 데이터 세트를 사용합니다.

### 유효성 검사 데이터 세트

원칙적으로 우리는 우리의 모든 하이퍼 매개 변수를 선택한 후까지 우리의 테스트 세트를 만지지 않아야합니다.모델 선택 프로세스에서 테스트 데이터를 사용한다면 테스트 데이터를 과도하게 사용할 위험이 있습니다.그러면 우리는 심각한 문제가 될 것입니다.우리가 우리의 교육 데이터를 과도하게 사용한다면, 우리를 정직하게 유지하기 위해 항상 테스트 데이터에 대한 평가가 있습니다.하지만 만약 우리가 테스트 데이터를 과도하게 사용한다면, 우리는 어떻게 알 수 있을까요?

따라서 모델 선택을위한 테스트 데이터에 의존해서는 안됩니다.그러나 모델을 학습하는 데 사용하는 데이터에 대한 일반화 오류를 예측할 수 없기 때문에 모델 선택을위한 교육 데이터에만 의존 할 수는 없습니다.

실용적인 응용 프로그램에서는 그림이 더 진흙 투성이됩니다.이상적으로 우리는 테스트 데이터를 한 번만 만지면 최상의 모델을 평가하거나 소수의 모델을 서로 비교하기 위해 실제 테스트 데이터는 한 번만 사용하면 거의 폐기되지 않습니다.우리는 실험의 각 라운드에 대한 새로운 테스트 세트를 거의 감당할 수 없습니다.

이 문제를 해결하는 일반적인 방법은 교육 및 테스트 데이터 세트 외에*유효성 검사 데이터세트* (또는*검증 세트*) 를 통합하는 세 가지 방법으로 데이터를 분할하는 것입니다.그 결과 유효성 검사와 테스트 데이터 간의 경계가 걱정스럽게 모호하는 어두운 관행입니다.명시 적으로 달리 명시되지 않는 한, 이 책의 실험에서 우리는 진정한 테스트 세트없이 교육 데이터 및 유효성 검사 데이터라고 부르는 것을 실제로 연구하고 있습니다.따라서 책의 각 실험에서보고 된 정확도는 실제로 검증 정확도이며 실제 테스트 세트 정확도가 아닙니다.

### $K$ 접힘 교차 검증

교육 데이터가 부족한 경우 적절한 유효성 검사 집합을 구성하기에 충분한 데이터를 보유하지 못할 수도 있습니다.이 문제에 대한 인기있는 해결책 중 하나는 $K$*배 교차 검증*을 사용하는 것입니다.여기서 원래 학습 데이터는 $K$ 겹치지 않는 하위 집합으로 분할됩니다.그런 다음 $K-1$ 하위 집합을 학습하고 다른 하위 집합 (해당 라운드에서 교육에 사용되지 않은 부분) 에서 유효성을 검사 할 때마다 모델 교육 및 유효성 검사가 $K$ 번 실행됩니다.마지막으로 $K$ 실험의 결과를 평균화하여 교육 및 검증 오류를 추정합니다.

## 언더피팅 또는 오버피팅?

우리는 교육 및 검증 오류를 비교할 때, 우리는 두 가지 일반적인 상황을 염두에 두기를 원합니다.첫째, 교육 오류 및 유효성 검사 오류가 모두 상당하지만 그 사이에 약간의 차이가있는 경우를 조심하고 싶습니다.모델이 학습 오류를 줄일 수 없다면 모델이 모델링하려는 패턴을 캡처하기에는 너무 단순하다는 것을 의미 할 수 있습니다 (즉, 표현력이 부족함).또한 교육과 검증 오류 사이의*일반화 간격*이 적기 때문에 더 복잡한 모델을 벗어날 수 있다고 믿을만한 이유가 있습니다.이러한 현상은 *밑받침*으로 알려져 있습니다.

반면에 위에서 설명한 것처럼 교육 오류가 검증 오류보다 현저히 낮은 경우 심각한*overfitting*을 나타내는 경우를 조심하고 싶습니다.과대 적합이 항상 나쁜 것은 아닙니다.특히 딥 러닝의 경우 최고의 예측 모델이 홀드아웃 데이터보다 교육 데이터에서 훨씬 더 잘 수행된다는 것이 잘 알려져 있습니다.궁극적으로, 우리는 일반적으로 교육 및 유효성 검사 오류 사이의 간격보다 유효성 검사 오류에 대해 더 신경 써야합니다.

과부적합 또는 부적합 여부는 모형의 복잡성과 사용 가능한 교육 데이터 집합의 크기에 따라 달라질 수 있습니다. 아래에서 설명하는 두 가지 주제입니다.

### 모델 복잡성

과잉 적합과 모델 복잡성에 대한 몇 가지 고전적 직관을 설명하기 위해 다항식을 사용하는 예제를 제공합니다.단일 기능 $x$ 및 해당 실제 가치 레이블 $y$으로 구성된 교육 데이터를 감안할 때, 우리는 학위 $d$의 다항식을 찾으려고 노력합니다.

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

를 사용하여 레이블 $y$을 추정할 수 있습니다.이것은 우리의 기능이 $x$의 힘에 의해 주어지는 선형 회귀 문제 일 뿐이며, 모델의 가중치는 $w_i$에 의해 주어지며, 모든 $x$에 대한 $w_0$부터 $w_0$에 의해 주어집니다.이것은 단지 선형 회귀 문제이기 때문에, 우리는 손실 함수로 제곱 오류를 사용할 수 있습니다.

고차 다항식 함수는 더 많은 매개 변수를 가지며 모델 함수의 선택 범위가 넓기 때문에 하위 다항식 함수보다 더 복잡합니다.훈련 데이터 세트를 고정하면 고차 다항식 함수는 항상 낮은 차수의 다항식을 기준으로 낮은 (최악의 경우 동등한) 훈련 오류를 달성해야합니다.실제로 데이터 포인트가 각각 $x$의 고유 값을 가질 때마다 데이터 포인트의 수와 같은 정도의 다항식 함수가 학습 세트에 완벽하게 맞을 수 있습니다.:numref:`fig_capacity_vs_error`에서 다항식 차수와 언더피팅 대 과대 피팅 사이의 관계를 시각화합니다.

![Influence of model complexity on underfitting and overfitting](../img/capacity_vs_error.svg)
:label:`fig_capacity_vs_error`

### 데이터 세트 크기

염두에 두어야 할 또 다른 큰 고려 사항은 데이터 세트 크기입니다.모델을 수정하면 교육 데이터 세트에 있는 샘플이 적을수록 과잉 적합을 접할 가능성이 높아집니다.학습 데이터의 양이 증가함에 따라 일반화 오류는 일반적으로 감소합니다.또한 일반적으로 더 많은 데이터가 결코 손상되지 않습니다.고정 작업 및 데이터 배포의 경우 일반적으로 모델 복잡성과 데이터 집합 크기 간에 관계가 있습니다.더 많은 데이터를 감안할 때 더 복잡한 모형을 수익성 있게 적합시키려고 할 수 있습니다.충분한 데이터가 없으면 간단한 모델이 이길 수 있습니다.많은 작업에서 딥 러닝은 수천 개의 교육 예제를 사용할 수 있는 경우에만 선형 모델보다 성능이 뛰어납니다.부분적으로, 현재 딥 러닝의 성공은 인터넷 회사, 저렴한 스토리지, 커넥티드 디바이스, 광범위한 경제의 디지털화로 인해 현재 방대한 데이터셋이 풍부해졌기 때문입니다.

## 다항식 회귀 분석

이제 다항식을 데이터에 맞추어 대화식으로 이러한 개념을 탐색할 수 있습니다.

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

### 데이터 세트 생성

먼저 데이터가 필요합니다.$x$를 감안할 때 다음 입방 다항식을 사용하여 학습 및 테스트 데이터에 대한 레이블을 생성합니다.

$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1^2).$$

잡음 항 $\epsilon$는 평균이 0이고 표준 편차가 0.1인 정규 분포를 따릅니다.최적화를 위해 일반적으로 매우 큰 그라디언트 또는 손실 값을 피하려고합니다.*기능*이 $x^i$에서 $\ FRAC {X ^ i} {i!} 로 다시 조정되는 이유입니다.$.그것은 우리가 큰 지수 $i$에 대한 매우 큰 값을 피할 수 있습니다.교육 세트 및 테스트 세트에 대해 각각 100개의 샘플을 합성합니다.

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

다시 말하지만, `poly_features`에 저장된 단일식은 감마 함수에 의해 재조정됩니다. 여기서 $\ 감마 (n) = (n-1)!$.생성된 데이터셋에서 처음 2개의 샘플을 살펴봅니다.값 1은 기술적으로 형상, 즉 치우침에 해당하는 상수 피쳐입니다.

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

우리가 먼저 주어진 데이터 세트의 손실을 평가하는 함수를 구현하자.

```{.python .input}
#@tab all
def evaluate_loss(net, data_iter, loss):  #@save
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        l = loss(net(X), y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]
```

이제 훈련 기능을 정의하십시오.

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
    loss = nn.MSELoss()
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

### 3차 다항식 함수 피팅 (일반)

우리는 먼저 데이터 생성 함수와 동일한 순서 인 3 차 다항식 함수를 사용하여 시작합니다.결과는 이 모델의 교육 및 테스트 손실을 모두 효과적으로 줄일 수 있음을 보여줍니다.학습 된 모델 매개 변수는 실제 값 $w = [5, 1.2, -3.4, 5.6]$에 가깝습니다.

```{.python .input}
#@tab all
# Pick the first four dimensions, i.e., 1, x, x^2/2!, x^3/3! from the
# polynomial features
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
```

### 선형 함수 피팅 (언더피팅)

우리가 선형 함수 피팅을 또 다른 살펴 보자.초기 시대에서의 감소 후, 이 모델의 훈련 손실을 더 감소시키기가 어려워진다.마지막 신기원 반복이 완료된 후에도 훈련 손실은 여전히 높습니다.비선형 패턴 (여기서의 3차 다항식 함수와 같은) 을 맞추는 데 사용되는 경우 선형 모형은 미달 적합에 대한 책임이 있습니다.

```{.python .input}
#@tab all
# Pick the first two dimensions, i.e., 1, x, from the polynomial features
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
```

### 고차 다항식 함수 피팅 (오버 피팅)

이제 우리가 너무 높은 수준의 다항식을 사용하여 모델을 훈련하려고 할 수 있습니다.여기서는 높은 차수의 계수가 0에 가까운 값을 가져야 한다는 것을 알 수 있는 데이터가 부족합니다.결과적으로 지나치게 복잡한 모델은 너무 민감하여 훈련 데이터의 노이즈에 영향을 받습니다.훈련 손실이 효과적으로 감소 될 수 있지만, 시험 손실은 여전히 훨씬 더 높습니다.복잡한 모형이 데이터를 과도하게 적합시킨다는 것을 보여줍니다.

```{.python .input}
#@tab all
# Pick all the dimensions from the polynomial features
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)
```

후속 섹션에서, 우리는 체중 감퇴 및 드롭아웃과 같은 그들을 처리하기위한 과잉 피팅 문제와 방법에 대해 논의 할 것입니다.

## 요약

* 일반화 오류는 학습 오류를 기반으로 추정할 수 없으므로 단순히 학습 오류를 최소화한다고 해서 일반화 오류가 감소되는 것은 아닙니다.머신 러닝 모델은 일반화 오류를 최소화하기 위해 과잉 피팅을 방지하도록 주의해야 합니다.
* 검증 세트는 너무 자유롭게 사용되지 않는 경우 모델 선택에 사용할 수 있습니다.
* 언더부속은 모형이 학습 오류를 줄일 수 없음을 의미합니다.교육 오류가 검증 오류보다 훨씬 낮은 경우, 초과 적합이 있습니다.
* 우리는 적절하게 복잡한 모델을 선택하고 불충분 한 교육 샘플을 사용하지 않아야합니다.

## 연습 문제

1. 다항식 회귀 문제를 정확하게 해결할 수 있습니까?힌트: 선형 대수학을 사용하십시오.
1. 다항식의 간결한 모델 선택:
    * 훈련 손실 대 모델 복잡성 (다항식의 정도) 을 플롯합니다.당신은 무엇을 관찰합니까?훈련 손실을 0으로 줄이려면 어느 정도의 다항식을 필요로합니까?
    * 이 경우 테스트 손실을 플롯합니다.
    * 데이터 양의 함수와 동일한 그림을 생성합니다.
1. 정규화를 떨어 뜨리면 어떻게됩니까 ($1/i!나는 $ 나는 $?다른 방법으로이 문제를 해결할 수 있습니까?
1. 일반화 오류가 전혀 없을 것으로 예상 할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:
