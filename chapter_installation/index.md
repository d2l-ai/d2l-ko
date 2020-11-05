# 설치
:label:`chap_installation`

실습 학습 경험을 위해 Python, Jupyter 노트북, 관련 라이브러리 및 책 자체를 실행하는 데 필요한 코드를 실행할 수 있는 환경을 구축해야 합니다.

## 미니콘다 설치

가장 간단한 방법은 [Miniconda](https://conda.io/en/latest/miniconda.html)를 설치하는 것입니다.파이썬 3.x 버전이 필요합니다.conda가 이미 설치된 경우 다음 단계를 건너뛸 수 있습니다.웹 사이트에서 해당 Miniconda sh 파일을 다운로드 한 다음 `sh <FILENAME> -b`를 사용하여 명령 줄에서 설치를 실행하십시오.macOS 사용자의 경우:

```bash
# The file name is subject to changes
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

Linux 사용자의 경우:

```bash
# The file name is subject to changes
sh Miniconda3-latest-Linux-x86_64.sh -b
```

다음으로 `conda`를 직접 실행할 수 있도록 쉘을 초기화합니다.

```bash
~/miniconda3/bin/conda init
```

이제 현재 셸을 닫고 다시 엽니 다.다음과 같이 새 환경을 만들 수 있어야합니다.

```bash
conda create --name d2l -y
```

## D2L 노트북 다운로드

다음으로이 책의 코드를 다운로드해야합니다.HTML 페이지 상단의 “모든 노트북” 탭을 클릭하여 코드를 다운로드하고 압축을 풀 수 있습니다.또는 `unzip` (그렇지 않으면 `sudo apt install unzip` 실행) 를 사용할 수있는 경우:

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

이제 `d2l` 환경을 활성화하고 `pip`를 설치하려고합니다.이 명령 뒤에 오는 질의에 대해 `y`를 입력합니다.

```bash
conda activate d2l
conda install python=3.7 pip -y
```

## 프레임워크 및 `d2l` 패키지 설치

:begin_tab:`mxnet,pytorch`
딥 러닝 프레임워크를 설치하기 전에 먼저 컴퓨터에 적절한 GPU가 있는지 확인하십시오 (표준 노트북의 디스플레이에 전원을 공급하는 GPU는 용도에 포함되지 않습니다).GPU 서버에 설치하는 경우 :ref:`subsec_gpu`로 이동하여 GPU 지원 버전을 설치하는 방법을 확인하십시오.

그렇지 않으면 CPU 버전을 설치할 수 있습니다.이는 처음 몇 장을 통해 당신을 얻을 수있는 충분한 마력 이상이지만 더 큰 모델을 실행하기 전에 GPU에 액세스하려고합니다.
:end_tab:

:begin_tab:`mxnet`
```bash
pip install mxnet==1.6.0
```
:end_tab:

:begin_tab:`pytorch`
```bash
pip install torch==1.5.1 torchvision -f https://download.pytorch.org/whl/torch_stable.html
```
:end_tab:

:begin_tab:`tensorflow`
다음을 통해 CPU와 GPU를 모두 지원하는 TensorFlow를 설치할 수 있습니다.

```bash
pip install tensorflow==2.2.0 tensorflow-probability==0.10.0
```
:end_tab:

또한 이 책에서 자주 사용하는 함수와 클래스를 캡슐화하는 `d2l` 패키지도 설치합니다.

```bash
pip install -U d2l
```

설치가 완료되면 다음을 실행하여 Jupyter 노트북을 엽니 다.

```bash
jupyter notebook
```

이때 웹 브라우저에서 http://localhost:8888 (일반적으로 자동으로 열림) 을 열 수 있습니다.그런 다음 책의 각 섹션에 대한 코드를 실행할 수 있습니다.책의 코드를 실행하거나 딥 러닝 프레임워크 또는 `d2l` 패키지를 업데이트하기 전에 항상 `conda activate d2l`를 실행하여 런타임 환경을 활성화하십시오.환경을 종료하려면 `conda deactivate`을 실행합니다.

## GPU 지원
:label:`subsec_gpu`

:begin_tab:`mxnet,pytorch`
기본적으로 딥 러닝 프레임워크는 GPU를 지원하지 않고 설치되어 모든 컴퓨터 (대부분의 랩톱 포함) 에서 실행됩니다.이 책의 일부는 GPU로 실행하거나 실행하는 것이 좋습니다.컴퓨터에 NVIDIA 그래픽 카드가 있고 [CUDA](https://developer.nvidia.com/cuda-downloads)가 설치된 경우 GPU 지원 버전을 설치해야 합니다.CPU 전용 버전을 설치한 경우 먼저 다음을 실행하여 제거해야 할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
기본적으로 TensorFlow는 GPU 지원과 함께 설치됩니다.컴퓨터에 NVIDIA 그래픽 카드가 있고 [CUDA](https://developer.nvidia.com/cuda-downloads)가 설치된 경우 모두 설정된 것입니다.
:end_tab:

:begin_tab:`mxnet`
```bash
pip uninstall mxnet
```
:end_tab:

:begin_tab:`pytorch`
```bash
pip uninstall torch
```
:end_tab:

:begin_tab:`mxnet,pytorch`
그런 다음 설치 한 CUDA 버전을 찾아야합니다.당신은 `nvcc --version` 또는 `cat /usr/local/cuda/version.txt`를 통해 확인할 수 있습니다.CUDA 10.1을 설치했다고 가정하면 다음 명령을 사용하여 설치할 수 있습니다.
:end_tab:

:begin_tab:`mxnet`
```bash
# For Windows users
pip install mxnet-cu101==1.6.0b20190926

# For Linux and macOS users
pip install mxnet-cu101==1.6.0
```
:end_tab:

:begin_tab:`pytorch`
```bash
pip install torch==1.5.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
:end_tab:

:begin_tab:`mxnet,pytorch`
CUDA 버전에 따라 마지막 숫자를 변경할 수 있습니다 (예: CUDA 10.0의 경우 `cu100`, CUDA 9.0의 경우 `cu90`).
:end_tab:

## 연습 문제

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
