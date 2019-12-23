# Installation
# 설치
:label:`chap_installation`

In order to get you up and running for hands-on learning experience,
we need to set you up with an environment for running Python,
Jupyter notebooks, the relevant libraries,
and the code needed to run the book itself.

실습을 통해서 배우는 경험을 위해서 우선 Python, Jupyter 노트북, 관련 라이브러리들, 그리고 이 책에 포함된 코드들을 위한 환경을 구성해야합니다.

## Installing Miniconda
## Miniconda 설치하기

The simplest way to get going will be to install
[Miniconda](https://conda.io/en/latest/miniconda.html). The Python 3.x version
is recommended. You can skip the following steps if conda has already been installed.
Download the corresponding Miniconda sh file from the website
and then execute the installation from the command line
using `sh <FILENAME> -b`. For macOS users:

실습 환경을 구성하는 가장 간단한 방법은 [Miniconda](https://conda.io/en/latest/miniconda.html)를 설치해서 사용하는 것이고, Python 3.x 버전을 권장합니다. 만약 이미 conda가 설치가 되어 있다면, 아래 단계들은 건너뛰세요. 웹 사이트에서 여러분의 시스템에 맞는 Miniconda sh 파일을 다운로드한 후, 명령창에서  `sh <FILENAME> -b` 를 입력해서 설치를 수행합니다. macOS 사용자는 아래 명령을 수행합니다.

```bash
# The file name is subject to changes
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

For Linux users:

리눅스 사용자는 다음 명령을 수행합니다.

```bash
# The file name is subject to changes
sh Miniconda3-latest-Linux-x86_64.sh -b
```

Next, initialize the shell so we can run `conda` directly.

다음으로는 `conda` 를 직접 수행할 수 있도록 쉘을 초기화합니다.

```bash
~/miniconda3/bin/conda init
```

Now close and re-open your current shell. You should be able to create a new
environment as following:

자 이제 열려있는 쉘을 닫고 다시 열어보세요. 다음 명령을 수행해서 새로운 환경을 만들 수 있습니다.

```bash
conda create --name d2l -y
```


## Downloading the D2L Notebooks
## D2L 노트북들 다운로드하기

Next, we need to download the code of this book. You can use the
[link](https://d2l.ai/d2l-en-0.7.1.zip) to download and unzip the code.
Alternatively, if you have `unzip` (otherwise run `sudo apt install unzip`) available:

이제 우리는 이 책에서 사용되는 코드들을 다운로드하겠습니다. 이 [링크](https://d2l.ai/d2l-en-0.7.1.zip) 를 통해서 파일을 다운로드한 후 압축을 풀어서 코드를 얻을 수 있습니다. 다른 방법으로는  `unzip` 툴이 있다면 (없을 경우  `sudo apt install unzip` 를 수행해서 설치할 수 있습니다.), 다음과 같이 다운로드 및 압축을 해제할 수 있습니다. 

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-0.7.1.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

Now we will want to activate the `d2l` environment and install `pip`.
Enter `y` for the queries that follow this command.

자 `d2l` 환경을 활성화하고 `pip`를 설치합니다. 이 명령을 수행한 후 나오는 질문에 모두 `y` 를 입력하세요.

```bash
conda activate d2l
conda install python=3.7 pip -y
```


## Installing MXNet and the `d2l` Package
## MXNet과 `d2l` 패키지 설치하기

Before installing MXNet, please first check
whether or not you have proper GPUs on your machine
(the GPUs that power the display on a standard laptop
do not count for our purposes).
If you are installing on a GPU server,
proceed to :ref:`subsec_gpu` for instructions
to install a GPU-supported MXNet.

MXNet을 설치하기 앞서 여러분의 컴퓨터에 GPU가 설치되어 있는지 여부를 확인합니다. (표준 랩탑의 화면 출력에 사용되는 GPU는 우리 목적에 해당하지 않습니다.) 만약 GPU 서버에 설치하는 중이라면, :ref:`subsec_gpu`을 참고해서 GPU를 지원하는 MXNet 설치합니다.

Otherwise, you can install the CPU version.
That will be more than enough horsepower to get you
through the first few chapters but you will want
to access GPUs before running larger models.

GPU가 없는 컴퓨터에 설치하는 경우라면 CPU 버전을 설치할 수 있습니다. CPU 버전 MXNet은 처음 몇 장들의 내용을 수행하기에 충분하지만, 이 후 큰 모델을 수행하기 위해서는 GPU가 필요합니다.

```bash
# For Windows users
pip install mxnet==1.6.0b20190926

# For Linux and macOS users
pip install mxnet==1.6.0b20191122
```

We also install the `d2l` package that encapsulates frequently used
functions and classes in this book.

이 책에서 자주 사용되는 함수들과 클래스들을 가지고 있는 `d2l` 패키지도 설치합니다.

```bash
pip install d2l==0.11.1
```

Once they are installed, we now open the Jupyter notebook by running:

이제 모든 설치가 끝났으면 아래 명령을 수행해서 Jupyter 노트북을 열어보세요.

```bash
jupyter notebook
```

At this point, you can open http://localhost:8888 (it usually opens automatically) in your Web browser. Then we can run the code for each section of the book.
Please always execute `conda activate d2l` to activate the runtime environment
before running the code of the book or updating MXNet or the `d2l` package.
To exit the environment, run `conda deactivate`.

이제 여러분의 웹 브라우에의 주소창에 http://localhost:8888 (보통은 자동으로 열립니다)를 입력해서 Jupyter 노트북을 열 수 있습니다. 이 노트북에서 이 책의 각 절에 포함된 코드들을 수행할 수 있습니다. 이 책의 코드를 수행하기 전에 그리고 MXNet 또는 `d2l` 패키지를 업데이트하기 전에 반드시  `conda activate d2l` 명령을 수행해서 실행 환경을 활성화하는 것을 기억하세요. 환경에서 나가는 명령은  `conda deactivate`입니다.


## Upgrading to a New Version
## 새로운 버전으로 업그래이드하기
Both this book and MXNet are keeping improving. Please check a new version from time to time.

1. The URL https://d2l.ai/d2l-en.zip always points to the latest contents.
2. Please upgrade the `d2l` package by `pip install d2l --upgrade`.
3. For the CPU version, MXNet can be upgraded by `pip install -U --pre mxnet`.

이 책과 MXNet 모두 지속적으로 향상되고 있으니, 자주 새로운 버전을 확인해주세요. 

1. https://d2l.ai/d2l-en.zip 는 항상 최신의 내용을 담고 있습니다.
2. `pip install d2l --upgrade` 명령으로 `d2l` 패키지를 업그래이드하세요.
3. CPU 버전의 MXNet은  `pip install -U --pre mxnet`로 업그래이드합니다.


## GPU Support
## GPU 지원
:label:`subsec_gpu`

By default, MXNet is installed without GPU support
to ensure that it will run on any computer (including most laptops).
Part of this book requires or recommends running with GPU.
If your computer has NVIDIA graphics cards and has installed [CUDA](https://developer.nvidia.com/cuda-downloads),
then you should install a GPU-enabled MXNet.
If you have installed the CPU-only version,
you may need to remove it first by running:

기본적으로 MXNet은 (대부분의 랩탑 컴퓨터를 포함한) 여러 컴퓨터에서 수행될 수 있도록 하기 위해서 GPU 지원을 포함하지 않고 설치됩니다. 이 책의 일부는 GPU가 필요하거나 사용을 권장합니다. 만약 여러분의 컴퓨터에 NVIDIA 그래픽 카드와 [CUDA](https://developer.nvidia.com/cuda-downloads)가 설치되어 있다면, GPU를 지원하는 MXNet을 설치하세요. 만약 CPU만 지원하는 MXNet 버전을 이미 설치했다면, 아래 명령을 통해서 우선 제거를 해야합니다

```bash
pip uninstall mxnet
```

Then we need to find the CUDA version you installed.
You may check it through `nvcc --version` or `cat /usr/local/cuda/version.txt`.
Assume that you have installed CUDA 10.1,
then you can install MXNet
with the following command:

그 다음으로 설치된 CUDA 버전을 확인하세요. CUDA 버전은 `nvcc --version` 또는  `cat /usr/local/cuda/version.txt` 명령으로 확인할 수 있습니다. 만약 여러분의 컴퓨터에서 설치된 CUDA 버전이 10.1이라고 가정한다묜, 아래와 같이 MXNet을 설치할 수 있습니다.

```bash
# For Windows users
pip install mxnet-cu101==1.6.0b20190926

# For Linux and macOS users
pip install mxnet-cu101==1.6.0b20191122
```

Like the CPU version, the GPU-enabled MXNet can be upgraded by
`pip install -U --pre mxnet-cu101`.
You may change the last digits according to your CUDA version,
e.g., `cu100` for CUDA 10.0 and `cu90` for CUDA 9.0.
You can find all available MXNet versions via `pip search mxnet`.

CPU 버전과 같이 GPU 지원 MXNet도 `pip install -U --pre mxnet-cu101`를 통해서 업그래이드를 할 수 있습니다. 마지막 숫자는 여러분의 CUDA 버전에 맞게 바꿔주세요. 예를 들어 CUDA 10.0인 경우에는 `cu100`을 CUDA 9.0인 경우에는 `cu90`로 바꾸면 됩니다.


## Exercises
## 연습문제

1. Download the code for the book and install the runtime environment.
1. 이 책의 코드를 다운로드하고 실행 환경을 설치하세요.

## [Discussions](https://discuss.mxnet.io/t/2315)
## [논의](https://discuss.mxnet.io/t/2315)

![](../img/qr_install.svg)
