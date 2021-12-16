# 심층 컨벌루션 신경망 (AlexNet)
:label:`sec_alexnet`

CNN은 LeNet이 도입 된 후 컴퓨터 비전 및 기계 학습 커뮤니티에서 잘 알려져 있었지만 현장을 즉시 지배하지는 못했습니다.LeNet은 초기의 소규모 데이터 세트에서 좋은 결과를 얻었지만 더 크고 현실적인 데이터 세트에 대해 CNN을 훈련시키는 성능과 타당성은 아직 확립되지 않았습니다.실제로 1990 년대 초반과 2012 년 유역 결과 사이의 많은 시간 동안 신경망은 지원 벡터 기계와 같은 다른 기계 학습 방법보다 능가하는 경우가 많습니다. 

컴퓨터 비전의 경우 이러한 비교는 공정하지 않을 수 있습니다.즉, 컨벌루션 네트워크에 대한 입력은 원시 또는 가볍게 처리된 (예: 센터링) 픽셀 값으로 구성되지만 실무자는 원시 픽셀을 기존 모델에 공급하지 않습니다.대신 일반적인 컴퓨터 비전 파이프라인은 기능 추출 파이프라인을 수동으로 엔지니어링하는 것으로 구성되었습니다.* 기능을 배우기* 대신 기능은*제작되었습니다*.대부분의 발전은 기능에 대한 더 영리한 아이디어를 얻는 데서 비롯되었으며 학습 알고리즘은 종종 나중에 고려되는 것으로 강등되었습니다. 

1990 년대에 일부 신경망 가속기를 사용할 수 있었지만 아직 많은 매개 변수를 가진 심층 다중 채널 다층 CNN을 만들기에는 충분히 강력하지 않았습니다.게다가 데이터 세트는 여전히 상대적으로 작았습니다.이러한 장애물에 더해 파라미터 초기화 휴리스틱, 확률적 경사 하강법의 영리한 변형, 비스쿼싱 활성화 함수 및 효과적인 정규화 기술을 포함한 신경망 훈련을 위한 핵심 트릭이 여전히 누락되었습니다. 

따라서*엔드-투-엔드* (픽셀-분류) 시스템을 훈련시키는 대신 클래식 파이프라인은 다음과 같이 보입니다. 

1. 흥미로운 데이터세트를 얻습니다.초기에는 이러한 데이터 세트에 값비싼 센서가 필요했습니다 (당시 1메가픽셀 이미지는 최첨단 이미지였습니다).
2. 광학, 기하학, 기타 분석 도구에 대한 지식과 때로는 운이 좋은 대학원생의 우연한 발견을 기반으로 수작업으로 만든 기능으로 데이터 세트를 전처리하십시오.
3. SIFT (스케일 불변 피쳐 변환) :cite:`Lowe.2004`, SURF (강력한 기능 속도 향상) :cite:`Bay.Tuytelaars.Van-Gool.2006` 또는 기타 여러 수동 조정 파이프라인과 같은 표준 기능 추출기 세트를 통해 데이터를 공급합니다.
4. 결과 표현을 선호하는 분류자 (예: 선형 모델 또는 커널 방법) 에 덤프하여 분류기를 훈련시킵니다.

머신 러닝 연구원들과 이야기를 나눴다면, 그들은 머신 러닝이 중요하고 아름답다고 믿었습니다.우아한 이론은 다양한 분류 자의 특성을 입증했습니다.기계 학습 분야는 번성하고 엄격하며 매우 유용했습니다.그러나 컴퓨터 비전 연구원과 이야기를 나누면 매우 다른 이야기를 듣게 될 것입니다.이미지 인식의 더러운 진실은 학습 알고리즘이 아닌 기능이 발전을 주도했다는 것입니다.컴퓨터 비전 연구원들은 약간 더 크거나 깨끗한 데이터 세트 또는 약간 개선 된 특징 추출 파이프 라인이 어떤 학습 알고리즘보다 최종 정확도에 훨씬 더 중요하다고 정당하게 믿었습니다. 

## 학습 표현

상황을 캐스팅하는 또 다른 방법은 파이프 라인의 가장 중요한 부분이 표현이라는 것입니다.그리고 2012년까지 표현은 기계적으로 계산되었습니다.실제로 새로운 기능 기능 세트를 엔지니어링하고 결과를 개선하며 방법을 작성하는 것이 종이 장르에서 두드러졌습니다.엄밀히 :cite:`Lowe.2004`, 서프 :cite:`Bay.Tuytelaars.Van-Gool.2006`, 호그 (배향 그래디언트의 히스토그램) :cite:`Dalal.Triggs.2005`, [bags of visual words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision) 및 이와 유사한 특징 추출기가 보금자리를 지배했습니다. 

얀 르쿤, 제프 힌튼, 요슈아 벵지오, 앤드류 응, 순이치 아마리, 유르겐 슈미드후버를 포함한 또 다른 연구원 그룹은 다른 계획을 가지고 있었다.그들은 특징 자체를 배워야 한다고 믿었습니다.또한 합리적으로 복잡하기 위해서는 특징을 학습 가능한 매개 변수를 가진 여러 공동으로 학습된 계층으로 계층적으로 구성되어야 한다고 믿었습니다.이미지의 경우 가장 낮은 계층이 가장자리, 색상 및 질감을 감지하기 위해 올 수 있습니다.실제로 알렉스 크리제프스키, 일리아 수츠케버, 제프 힌튼은 CNN의 새로운 변종을 제안했습니다.
*알렉스넷*,
2012 이미지넷 챌린지에서 탁월한 성능을 달성했습니다.알렉스넷은 획기적인 이미지넷 분류 논문 :cite:`Krizhevsky.Sutskever.Hinton.2012`의 첫 번째 저자인 알렉스 크리제프스키의 이름을 따서 명명되었습니다. 

흥미롭게도 네트워크의 가장 낮은 계층에서 모델은 일부 기존 필터와 유사한 특징 추출기를 학습했습니다. :numref:`fig_filters`는 AlexNet 논문 :cite:`Krizhevsky.Sutskever.Hinton.2012`에서 재현되었으며 하위 수준 이미지 설명자를 설명합니다. 

![Image filters learned by the first layer of AlexNet.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

네트워크의 상위 계층은 눈, 코, 풀잎 등과 같은 더 큰 구조를 나타내기 위해 이러한 표현을 기반으로 구축될 수 있습니다.더 높은 레이어라도 사람, 비행기, 개 또는 프리스비와 같은 전체 개체를 나타낼 수 있습니다.궁극적으로 최종 은닉 상태는 서로 다른 카테고리에 속하는 데이터를 쉽게 분리 할 수 있도록 내용을 요약하는 이미지의 간결한 표현을 학습합니다. 

2012년에 여러 계층화된 CNN의 궁극적인 돌파구가 나왔지만, 핵심 연구원 그룹은 수년 동안 시각적 데이터의 계층적 표현을 배우려고 시도하면서 이 아이디어에 전념했습니다.2012년의 궁극적인 돌파구는 두 가지 핵심 요소에 기인할 수 있습니다. 

### 누락된 성분: 데이터

계층이 많은 심층 모델은 볼록 최적화 (예: 선형 및 커널 방법) 를 기반으로 한 기존 방법보다 훨씬 뛰어난 성능을 발휘하는 영역에 들어가기 위해 많은 양의 데이터가 필요합니다.그러나 컴퓨터의 제한된 저장 용량, 센서의 상대적 비용 및 1990 년대의 비교적 엄격한 연구 예산을 고려할 때 대부분의 연구는 작은 데이터 세트에 의존했습니다.수많은 논문에서 UCI 데이터 세트 수집을 다루었으며, 그 중 다수에는 저해상도의 부자연스러운 설정에서 캡처된 수백 또는 수천 개의 이미지만 포함되어 있습니다. 

2009년에는 ImageNet 데이터세트가 발표되어 연구원들은 1000개의 서로 다른 개체 범주에서 각각 1000개의 1백만 개의 예제에서 모델을 학습하도록 도전했습니다.이 데이터 세트를 도입 한 Fei-Fei Li가 이끄는 연구원들은 Google 이미지 검색을 활용하여 각 카테고리에 대한 대규모 후보 세트를 사전 필터링하고 Amazon Mechanical Turk 크라우드 소싱 파이프 라인을 사용하여 각 이미지에 대해 관련 카테고리에 속하는지 확인했습니다.이 규모는 전례가 없었습니다.ImageNet Challenge라고 불리는 관련 경쟁은 컴퓨터 비전 및 머신 러닝 연구를 추진하여 연구자들이 이전에 고려했던 것보다 더 큰 규모로 가장 잘 수행 된 모델을 식별하도록 도전했습니다. 

### 누락된 성분: 하드웨어

딥 러닝 모델은 컴퓨팅 사이클의 열렬한 소비자입니다.훈련에는 수백 개의 epoch가 필요할 수 있으며, 각 반복에는 계산 비용이 많이 드는 선형 대수 연산의 여러 계층을 통해 데이터를 전달해야 합니다.이것이 1990 년대와 2000 년대 초반에보다 효율적으로 최적화 된 볼록 대물렌즈를 기반으로 한 간단한 알고리즘이 선호되는 주된 이유 중 하나입니다. 

*그래픽 처리 장치* (GPU) 는 게임 체인저로 입증되었습니다.
딥 러닝을 실현 가능하게 만드는 것입니다.이 칩은 컴퓨터 게임에 도움이되도록 그래픽 처리를 가속화하기 위해 오랫동안 개발되었습니다.특히 많은 컴퓨터 그래픽 작업에 필요한 높은 처리량의 $4 \times 4$ 매트릭스-벡터 제품에 최적화되었습니다.다행히도 이 수학은 컨벌루션 계층을 계산하는 데 필요한 수학과 매우 유사합니다.그 무렵 NVIDIA와 ATI는 일반 컴퓨팅 작업을 위해 GPU를 최적화하기 시작했으며, 이를 범용 GPU* (GPGPU) 로 마케팅하기 시작했습니다. 

직감을 제공하기 위해 최신 마이크로 프로세서 (CPU) 의 코어를 고려하십시오.각 코어는 높은 클럭 주파수에서 실행되며 대용량 캐시 (최대 수 메가 바이트의 L3) 를 자랑합니다.각 코어는 분기 예측자, 심층 파이프라인 및 다양한 프로그램을 실행할 수 있는 기타 종소리와 휘파람을 사용하여 광범위한 명령을 실행하는 데 적합합니다.그러나 이러한 명백한 강도는 아킬레스 건이기도합니다. 범용 코어는 제작 비용이 매우 비쌉니다.많은 칩 영역, 정교한 지원 구조 (메모리 인터페이스, 코어 간 캐싱 로직, 고속 상호 연결 등) 가 필요하며 단일 작업에서 비교적 좋지 않습니다.최신 랩톱에는 최대 4 개의 코어가 있으며 고급 서버도 비용 효율적이지 않기 때문에 64 코어를 초과하는 경우는 거의 없습니다. 

이에 비해 GPU는 $100 \sim 1000$ 개의 작은 처리 요소 (NVIDIA, ATI, ARM 및 기타 칩 공급 업체마다 세부 사항이 다소 다름) 로 구성되며 종종 더 큰 그룹으로 그룹화됩니다 (NVIDIA는 워프라고 부름).각 코어는 상대적으로 약하고 때로는 1GHz 이하의 클록 주파수에서 실행되기도하지만 CPU보다 GPU를 훨씬 빠르게 만드는 것은 이러한 코어의 총 수입니다.예를 들어 NVIDIA의 최근 Volta 세대는 특수 명령을 위해 칩당 최대 120 TFlops를 제공하며 (보다 일반적인 용도의 경우 최대 24 TFlops) CPU의 부동 소수점 성능은 현재까지 1 TFlop을 초과하지 않았습니다.이것이 가능한 이유는 실제로 매우 간단합니다. 첫째, 전력 소비는 클럭 주파수에 따라*2차적* 증가하는 경향이 있습니다.따라서 4배 더 빠르게 실행되는 CPU 코어의 전력 예산 (일반적인 숫자) 의 경우 $1/4$의 속도로 16개의 GPU 코어를 사용할 수 있으므로 성능이 $16 \times 1/4 = 4$배입니다.또한 GPU 코어는 훨씬 단순하여 (실제로 오랫동안 범용 코드를 실행할 수 없었습니다*) 에너지 효율성이 향상됩니다.마지막으로 딥러닝의 많은 작업에는 높은 메모리 대역폭이 필요합니다.다시 말하지만 GPU는 CPU의 너비가 최소 10배 이상인 버스로 빛을 발합니다. 

2012년으로 거슬러 올라갑니다.알렉스 크리제프스키와 일리아 수츠케버가 GPU 하드웨어에서 실행할 수 있는 심층 CNN을 구현했을 때 큰 돌파구가 생겼습니다.그들은 CNN, 컨벌루션 및 행렬 곱셈의 계산 병목 현상이 모두 하드웨어에서 병렬화 될 수있는 작업이라는 것을 깨달았습니다.3GB 메모리를 갖춘 두 개의 NVIDIA GTX 580을 사용하여 빠른 컨벌루션을 구현했습니다.코드 [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/)는 몇 년 동안 업계 표준이었고 딥 러닝 붐의 첫 몇 년을 구동하기에 충분했습니다. 

## 알렉스넷

8 레이어 CNN을 채택한 AlexNet은 놀랍도록 큰 마진으로 이미지넷 대규모 시각 인식 챌린지 2012에서 우승했습니다.이 네트워크는 학습을 통해 얻은 기능이 수동으로 설계된 기능을 초월하여 컴퓨터 비전의 이전 패러다임을 깨뜨릴 수 있음을 처음으로 보여주었습니다. 

알렉스넷과 르넷의 아키텍처는 :numref:`fig_alexnet`에서 알 수 있듯이 매우 유사합니다.이 모델을 두 개의 작은 GPU에 맞추기 위해 2012년에 필요했던 몇 가지 디자인 단점을 제거한 약간 간소화된 AlexNet 버전을 제공합니다. 

![From LeNet (left) to AlexNet (right).](../img/alexnet.svg)
:label:`fig_alexnet`

AlexNet과 LeNet의 디자인 철학은 매우 유사하지만 중요한 차이점도 있습니다.첫째, 알렉스넷은 비교적 작은 LeNet5보다 훨씬 깊습니다.AlexNet은 8개의 계층으로 구성되어 있습니다. 5개의 컨벌루션 계층, 2개의 완전 연결 은닉 계층, 그리고 1개의 완전 연결 출력 계층입니다.둘째, AlexNet은 시그모이드 대신 ReLU를 활성화 함수로 사용했습니다.아래에서 자세히 살펴 보겠습니다. 

### 아키텍처

알렉스넷의 첫 번째 계층에서 컨벌루션 윈도우 모양은 $11\times11$입니다.ImageNet에 있는 대부분의 이미지는 MNIST 이미지보다 10배 이상 높고 넓기 때문에 ImageNet 데이터의 객체는 더 많은 픽셀을 차지하는 경향이 있습니다.따라서 객체를 캡처하려면 더 큰 컨볼루션 윈도우가 필요합니다.두 번째 레이어의 컨볼루션 윈도우 모양은 $5\times5$으로 축소되고 그 다음에는 $3\times3$가 이어집니다.또한 첫 번째, 두 번째 및 다섯 번째 컨벌루션 계층 다음에 윈도우 모양이 $3\times3$이고 스트라이드가 2인 최대 풀링 계층을 추가합니다.또한 AlexNet은 LeNet보다 10배 더 많은 컨볼루션 채널을 가지고 있습니다. 

마지막 컨벌루션 계층 뒤에는 4096개의 출력값을 갖는 두 개의 완전 연결 계층이 있습니다.이 두 개의 거대한 완전 연결 계층은 거의 1GB의 모델 파라미터를 생성합니다.초기 GPU의 제한된 메모리로 인해 원래 AlexNet은 이중 데이터 스트림 설계를 사용했기 때문에 두 GPU 각각이 모델의 절반만 저장하고 계산할 수 있습니다.다행히도 GPU 메모리는 현재 비교적 풍부하기 때문에 요즘 GPU에서 모델을 분리 할 필요가 거의 없습니다 (이 측면에서 AlexNet 모델 버전은 원본 논문에서 벗어납니다). 

### 활성화 함수

게다가 AlexNet은 시그모이드 활성화 함수를 더 간단한 ReLU 활성화 함수로 변경했습니다.한편으로는 ReLU 활성화 함수의 계산이 더 간단합니다.예를 들어, 시그모이드 활성화 함수에서 지수 연산을 찾을 수 없습니다.반면에 ReLU 활성화 기능은 다른 매개 변수 초기화 방법을 사용할 때 모델 학습을 더 쉽게 만듭니다.시그모이드 활성화 함수의 출력값이 0 또는 1에 매우 가까우면 이러한 영역의 기울기가 거의 0이므로 역전파가 일부 모델 파라미터를 계속 업데이트할 수 없기 때문입니다.반면 양의 구간에서 ReLU 활성화 함수의 기울기는 항상 1입니다.따라서 모델 모수가 제대로 초기화되지 않으면 sigmoid 함수가 양의 구간에서 거의 0의 기울기를 얻을 수 있으므로 모델을 효과적으로 훈련시킬 수 없습니다. 

### 용량 제어 및 전처리

AlexNet은 드롭아웃 (:numref:`sec_dropout`) 을 통해 완전 연결 계층의 모델 복잡성을 제어하는 반면, LeNet은 가중치 감소만 사용합니다.데이터를 더욱 보강하기 위해 AlexNet의 훈련 루프는 뒤집기, 클리핑 및 색상 변경과 같은 많은 이미지 증대를 추가했습니다.이렇게 하면 모형이 더욱 견고해지고 표본 크기가 클수록 과적합이 효과적으로 감소합니다.:numref:`sec_image_augmentation`에서 데이터 증대에 대해 자세히 설명하겠습니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
# Here, we use a larger 11 x 11 window to capture objects. At the same time,
# we use a stride of 4 to greatly reduce the height and width of the output.
# Here, the number of output channels is much larger than that in LeNet
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(
    # Here, we use a larger 11 x 11 window to capture objects. At the same
    # time, we use a stride of 4 to greatly reduce the height and width of the
    # output. Here, the number of output channels is much larger than that in
    # LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Make the convolution window smaller, set padding to 2 for consistent
    # height and width across the input and output, and increase the number of
    # output channels
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Use three successive convolutional layers and a smaller convolution
    # window. Except for the final convolutional layer, the number of output
    # channels is further increased. Pooling layers are not used to reduce the
    # height and width of input after the first two convolutional layers
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # Here, the number of outputs of the fully-connected layer is several
    # times larger than that in LeNet. Use the dropout layer to mitigate
    # overfitting
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # Output layer. Since we are using Fashion-MNIST, the number of classes is
    # 10, instead of 1000 as in the paper
    nn.Linear(4096, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        # Here, we use a larger 11 x 11 window to capture objects. At the same
        # time, we use a stride of 4 to greatly reduce the height and width of
        # the output. Here, the number of output channels is much larger than
        # that in LeNet
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        tf.keras.layers.Dense(10)
    ])
```

높이와 너비가 224인 [**단일 채널 데이터 예제**] 를 구성합니다 (**각 레이어의 출력 형태를 관찰하기 위해**).:numref:`fig_alexnet`의 알렉스넷 아키텍처와 일치합니다.

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

## 데이터세트 읽기

AlexNet은 이 백서에서 ImageNet에 대한 교육을 받았지만, 최신 GPU에서도 ImageNet 모델을 수렴하도록 훈련시키는 데 몇 시간 또는 며칠이 걸릴 수 있기 때문에 여기서는 Fashion-MNIST를 사용합니다.AlexNet을 [**Fashion-MNist**] 에 직접 적용할 때의 문제점 중 하나는 (**이미지의 해상도가 낮습니다**) ($28 \times 28$ 픽셀) (**ImageNet 이미지보다**) 작업을 수행하기 위해 (**$224 \times 224$**로 업샘플링) (일반적으로 현명한 관행은 아니지만 AlexNet에 충실하기 위해 여기서 수행합니다.아키텍처).`d2l.load_data_fashion_mnist` 함수에서 `resize` 인수를 사용하여 이 크기 조정을 수행합니다.

```{.python .input}
#@tab all
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

## 트레이닝

이제 우리는 [**AlexNet의 훈련을 시작합니다.**] :numref:`sec_lenet`의 LeNet과 비교할 때, 여기서 가장 큰 변화는 더 깊고 넓은 네트워크, 더 높은 이미지 해상도 및 더 많은 비용이 드는 컨벌루션으로 인해 더 작은 학습률과 훨씬 느린 훈련을 사용한다는 것입니다.

```{.python .input}
#@tab all
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 요약

* AlexNet은 LeNet의 구조와 유사하지만 대규모 ImageNet 데이터 세트에 적합하기 위해 더 많은 컨벌루션 계층과 더 큰 매개 변수 공간을 사용합니다.
* 오늘날 AlexNet은 훨씬 더 효과적인 아키텍처를 능가했지만 오늘날 사용되는 얕은 네트워크에서 심층 네트워크까지 핵심 단계입니다.
* AlexNet의 구현에는 LeNet보다 몇 줄이 더 많은 것 같지만 학계에서 이러한 개념적 변화를 수용하고 우수한 실험 결과를 활용하는 데 수년이 걸렸습니다.이는 효율적인 계산 도구가 부족하기 때문이기도 합니다.
* 드롭아웃, ReLU 및 전처리는 컴퓨터 비전 작업에서 뛰어난 성능을 달성하는 또 다른 주요 단계였습니다.

## 연습문제

1. 에포크 수를 늘려 보십시오.LeNet과 비교할 때 결과는 어떻게 다릅니 까?왜요?
1. AlexNet은 패션-MNIST 데이터세트에 비해 너무 복잡할 수 있습니다.
    1. 모델을 단순화하여 훈련 속도를 높이고 정확도가 크게 떨어지지 않도록 하십시오.
    1. $28 \times 28$ 이미지에서 직접 작동하는 더 나은 모델을 설계합니다.
1. 배치 크기를 수정하고 정확도와 GPU 메모리의 변화를 관찰합니다.
1. AlexNet의 계산 성능을 분석합니다.
    1. AlexNet의 메모리 공간에서 가장 중요한 부분은 무엇입니까?
    1. AlexNet에서 계산에서 가장 중요한 부분은 무엇입니까?
    1. 결과를 계산할 때 메모리 대역폭은 어떻습니까?
1. 드롭아웃 및 ReLU를 LeNet-5에 적용합니다.개선됩니까?전처리는 어떻습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/276)
:end_tab:
