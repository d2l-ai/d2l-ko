# 설치
:label:`chap_installation`

실습 학습 경험을 시작하고 실행하려면 Python, Jupyter 노트북, 관련 라이브러리 및 책 자체를 실행하는 데 필요한 코드를 실행할 수 있는 환경을 설정해야 합니다. 

## 미니콘다 설치

가장 간단한 방법은 [Miniconda](https://conda.io/en/latest/miniconda.html)를 설치하는 것입니다.파이썬 3.x 버전이 필요합니다.컴퓨터에 이미 conda가 설치되어 있는 경우 다음 단계를 건너뛸 수 있습니다. 

Miniconda 웹 사이트를 방문하여 Python 3.x 버전 및 머신 아키텍처를 기반으로 시스템에 적합한 버전을 결정하십시오.예를 들어, macOS 및 Python 3.x를 사용하는 경우 이름에 “Miniconda3"및 “macOSX” 문자열이 포함된 bash 스크립트를 다운로드하고 다운로드 위치로 이동한 다음 다음과 같이 설치를 실행합니다.

```bash
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

파이썬 3.x를 사용하는 리눅스 사용자는 이름에 “Miniconda3"과 “Linux” 문자열이 포함된 파일을 다운로드하고 다운로드 위치에서 다음을 실행합니다.

```bash
sh Miniconda3-latest-Linux-x86_64.sh -b
```

다음으로 `conda`를 직접 실행할 수 있도록 셸을 초기화합니다.

```bash
~/miniconda3/bin/conda init
```

이제 현재 셸을 닫았다가 다시 엽니다.다음과 같이 새 환경을 만들 수 있어야 합니다.

```bash
conda create --name d2l python=3.8 -y
```

## D2L 노트북 다운로드

다음으로 이 책의 코드를 다운로드해야 합니다.HTML 페이지 상단의 “모든 노트북” 탭을 클릭하여 코드를 다운로드하고 압축을 풀 수 있습니다.또는 `unzip` (그렇지 않으면 `sudo apt install unzip`를 실행) 를 사용할 수 있는 경우 다음을 수행합니다.

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

이제 `d2l` 환경을 활성화할 수 있습니다.

```bash
conda activate d2l
```

## 프레임워크 및 `d2l` 패키지 설치

딥 러닝 프레임워크를 설치하기 전에 먼저 컴퓨터에 적절한 GPU가 있는지 확인하십시오 (표준 노트북의 디스플레이에 전원을 공급하는 GPU는 당사의 목적과 관련이 없습니다).GPU 서버에서 작업하는 경우 관련 라이브러리의 GPU 호환 버전을 설치하는 방법에 대한 지침을 보려면 :ref:`subsec_gpu`로 이동하십시오. 

컴퓨터에 GPU가 없어도 아직 걱정할 필요가 없습니다.CPU는 처음 몇 장을 통과하기에 충분한 마력을 제공합니다.더 큰 모델을 실행하기 전에 GPU에 액세스해야 한다는 점을 기억하세요.CPU 버전을 설치하려면 다음 명령을 실행합니다.

:begin_tab:`mxnet`
```bash
pip install mxnet==1.7.0.post1
```
:end_tab:

:begin_tab:`pytorch`
```bash
pip install torch torchvision
```
:end_tab:

:begin_tab:`tensorflow`
다음과 같이 CPU와 GPU를 모두 지원하는 텐서플로우를 설치할 수 있습니다.

```bash
pip install tensorflow tensorflow-probability
```
:end_tab:

다음 단계는 이 책에서 자주 사용하는 함수와 클래스를 캡슐화하기 위해 개발한 `d2l` 패키지를 설치하는 것입니다.

```bash
# -U: Upgrade all packages to the newest available version
pip install -U d2l
```

이 설치 단계를 완료하면 다음을 실행하여 Jupyter 노트북 서버를 실행할 수 있습니다.

```bash
jupyter notebook
```

이 시점에서 웹 브라우저에서 http://localhost:8888 (이미 자동으로 열렸을 수 있음) 을 열 수 있습니다.그런 다음 책의 각 섹션에 대해 코드를 실행할 수 있습니다.책의 코드를 실행하거나 딥 러닝 프레임워크 또는 `d2l` 패키지를 업데이트하기 전에 항상 `conda activate d2l`을 실행하여 런타임 환경을 활성화하십시오.환경을 종료하려면 `conda deactivate`를 실행합니다. 

## GPU 지원
:label:`subsec_gpu`

:begin_tab:`mxnet`
기본적으로 MXNet은 GPU를 지원하지 않고 설치되어 모든 컴퓨터 (대부분의 랩톱 포함) 에서 실행되도록 합니다.이 책의 일부에서는 GPU를 실행해야 하거나 실행하는 것이 좋습니다.컴퓨터에 NVIDIA 그래픽 카드가 있고 [CUDA](https://developer.nvidia.com/cuda-downloads)가 설치되어 있는 경우 GPU 지원 버전을 설치해야 합니다.CPU 전용 버전을 설치한 경우 먼저 다음을 실행하여 제거해야 할 수 있습니다.

```bash
pip uninstall mxnet
```

이제 어떤 버전의 CUDA를 설치했는지 확인해야 합니다.`nvcc --version` 또는 `cat /usr/local/cuda/version.txt`를 실행하여 이를 확인할 수 있습니다.CUDA 10.1을 설치했다고 가정하면 다음 명령을 사용하여 설치할 수 있습니다.

```bash
# For Windows users
pip install mxnet-cu101==1.7.0 -f https://dist.mxnet.io/python

# For Linux and macOS users
pip install mxnet-cu101==1.7.0
```

CUDA 버전에 따라 마지막 숫자를 변경할 수 있습니다 (예: CUDA 10.0의 경우 `cu100`, 쿠다 9.0의 경우 `cu90`).
:end_tab:

:begin_tab:`pytorch,tensorflow`
기본적으로 딥러닝 프레임워크는 GPU를 지원하는 상태로 설치됩니다.컴퓨터에 NVIDIA GPU가 있고 [CUDA](https://developer.nvidia.com/cuda-downloads)가 설치되어 있다면 모든 준비가 완료된 것입니다.
:end_tab:

## 연습문제

1. 책의 코드를 다운로드하고 런타임 환경을 설치합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/436)
:end_tab:
