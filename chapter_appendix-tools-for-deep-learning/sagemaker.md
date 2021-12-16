# 아마존 세이지메이커 사용
:label:`sec_sagemaker`

많은 딥 러닝 애플리케이션에는 상당한 양의 계산이 필요합니다.로컬 시스템의 속도가 너무 느려 적절한 시간 내에 이러한 문제를 해결할 수 없습니다.클라우드 컴퓨팅 서비스를 사용하면 이 설명서의 GPU 집약적인 부분을 실행할 수 있는 더 강력한 컴퓨터에 액세스할 수 있습니다.이 자습서에서는 이 책을 쉽게 실행할 수 있는 서비스인 Amazon SageMaker를 안내합니다. 

## 등록 및 로그인

먼저 https://aws.amazon.com/ 에 계정을 등록해야 합니다.추가 보안을 위해 이중 인증을 사용하는 것이 좋습니다.실행 중인 인스턴스를 중지하는 것을 잊은 경우 예상치 못한 예기치 않은 상황이 발생하지 않도록 세부 결제 및 지출 알림을 설정하는 것도 좋습니다.신용카드가 필요합니다.AWS 계정에 로그인한 후 [console](http://console.aws.amazon.com/)로 이동하여 “세이지메이커” (:numref:`fig_sagemaker` 참조) 를 검색한 다음 클릭하여 세이지메이커 패널을 엽니다. 

![Open the SageMaker panel.](../img/sagemaker.png)
:width:`300px`
:label:`fig_sagemaker`

## 세이지메이커 인스턴스 생성

다음으로 :numref:`fig_sagemaker-create`에 설명된 대로 노트북 인스턴스를 만들어 보겠습니다. 

![Create a SageMaker instance.](../img/sagemaker-create.png)
:width:`400px`
:label:`fig_sagemaker-create`

세이지메이커는 다양한 계산 능력과 가격의 여러 [instance types](https://aws.amazon.com/sagemaker/pricing/instance-types/)을 제공합니다.인스턴스를 생성할 때 인스턴스 이름을 지정하고 유형을 선택할 수 있습니다.:numref:`fig_sagemaker-create-2`에서는 `ml.p3.2xlarge`를 선택합니다.하나의 Tesla V100 GPU와 8코어 CPU를 갖춘 이 인스턴스는 대부분의 챕터에서 충분히 강력합니다. 

![Choose the instance type.](../img/sagemaker-create-2.png)
:width:`400px`
:label:`fig_sagemaker-create-2`

:begin_tab:`mxnet`
SageMaker를 장착하기 위한 이 책의 주피터 노트북 버전은 https_://github.com/d2l-ai/d2l-en-sagemaker. We can specify this GitHub repository URL to let SageMaker clone this repository during instance creation, as shown in :numref:`fig_sagemaker-create-3`에서 구할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
SageMaker를 장착하기 위한 이 책의 주피터 노트북 버전은 https_://github.com/d2l-ai/d2l-pytorch-sagemaker. We can specify this GitHub repository URL to let SageMaker clone this repository during instance creation, as shown in :numref:`fig_sagemaker-create-3`에서 구할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
SageMaker를 장착하기 위한 이 책의 주피터 노트북 버전은 https_://github.com/d2l-ai/d2l-tensorflow-sagemaker. We can specify this GitHub repository URL to let SageMaker clone this repository during instance creation, as shown in :numref:`fig_sagemaker-create-3`에서 구할 수 있습니다.
:end_tab:

![Specify the GitHub repository.](../img/sagemaker-create-3.png)
:width:`400px`
:label:`fig_sagemaker-create-3`

## 인스턴스 실행 및 중지

인스턴스가 준비되기까지 몇 분 정도 걸릴 수 있습니다.준비가되면 :numref:`fig_sagemaker-open`와 같이 “목성 열기”링크를 클릭 할 수 있습니다. 

![Open Jupyter on the created SageMaker instance.](../img/sagemaker-open.png)
:width:`400px`
:label:`fig_sagemaker-open`

그런 다음 :numref:`fig_sagemaker-jupyter`와 같이 이 인스턴스에서 실행 중인 Jupyter 서버를 탐색할 수 있습니다. 

![The Jupyter server running on the SageMaker instance.](../img/sagemaker-jupyter.png)
:width:`400px`
:label:`fig_sagemaker-jupyter`

SageMaker 인스턴스에서 주피터 노트북을 실행하고 편집하는 것은 :numref:`sec_jupyter`에서 논의한 것과 유사합니다.작업을 마친 후에는 :numref:`fig_sagemaker-stop`와 같이 추가 충전을 피하기 위해 인스턴스를 중지하는 것을 잊지 마십시오. 

![Stop a SageMaker instance.](../img/sagemaker-stop.png)
:width:`300px`
:label:`fig_sagemaker-stop`

## 노트북 업데이트

:begin_tab:`mxnet`
우리는 [d2l-ai/d2l-en-sagemaker](https://github.com/d2l-ai/d2l-en-sagemaker) GitHub 리포지토리의 노트북을 정기적으로 업데이트할 것입니다.`git pull` 명령을 사용하여 최신 버전으로 업데이트하기만 하면 됩니다.
:end_tab:

:begin_tab:`pytorch`
우리는 [d2l-ai/d2l-pytorch-sagemaker](https://github.com/d2l-ai/d2l-pytorch-sagemaker) GitHub 리포지토리의 노트북을 정기적으로 업데이트할 것입니다.`git pull` 명령을 사용하여 최신 버전으로 업데이트하기만 하면 됩니다.
:end_tab:

:begin_tab:`tensorflow`
우리는 [d2l-ai/d2l-tensorflow-sagemaker](https://github.com/d2l-ai/d2l-tensorflow-sagemaker) GitHub 리포지토리의 노트북을 정기적으로 업데이트할 것입니다.`git pull` 명령을 사용하여 최신 버전으로 업데이트하기만 하면 됩니다.
:end_tab:

먼저 :numref:`fig_sagemaker-terminal`와 같이 터미널을 열어야 합니다. 

![Open a terminal on the SageMaker instance.](../img/sagemaker-terminal.png)
:width:`300px`
:label:`fig_sagemaker-terminal`

업데이트를 가져오기 전에 로컬 변경 사항을 커밋할 수 있습니다.또는 터미널에서 다음 명령을 사용하여 모든 로컬 변경 사항을 무시할 수 있습니다.

:begin_tab:`mxnet`
```bash
cd SageMaker/d2l-en-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`pytorch`
```bash
cd SageMaker/d2l-pytorch-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`tensorflow`
```bash
cd SageMaker/d2l-tensorflow-sagemaker/
git reset --hard
git pull
```
:end_tab:

## 요약

* 이 책을 실행하기 위해 Amazon SageMaker를 통해 Jupyter 서버를 시작하고 중지할 수 있습니다.
* Amazon SageMaker 인스턴스의 터미널을 통해 노트북을 업데이트할 수 있습니다.

## 연습문제

1. Amazon SageMaker를 사용하여 이 책의 코드를 편집하고 실행해 보십시오.
1. 터미널을 통해 소스 코드 디렉토리에 액세스합니다.

[Discussions](https://discuss.d2l.ai/t/422)
