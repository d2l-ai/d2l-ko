# AWS EC2 인스턴스 사용
:label:`sec_aws`

이 섹션에서는 원시 Linux 시스템에 모든 라이브러리를 설치하는 방법을 보여줍니다.:numref:`sec_sagemaker`에서는 Amazon SageMaker를 사용하는 방법에 대해 논의한 반면, 인스턴스를 직접 빌드하는 방법은 AWS에서 더 저렴하다는 점을 기억하십시오.이 연습에는 다음과 같은 여러 단계가 포함됩니다. 

1. AWS EC2에서 GPU 리눅스 인스턴스에 대한 요청입니다.
1. 선택 사항: CUDA를 설치하거나 CUDA가 사전 설치된 AMI를 사용합니다.
1. 해당 MXNet GPU 버전을 설정합니다.

이 프로세스는 약간의 수정이 있었음에도 불구하고 다른 인스턴스 (및 다른 클라우드) 에도 적용됩니다.계속 진행하기 전에 AWS 계정을 만들어야 합니다. 자세한 내용은 :numref:`sec_sagemaker`를 참조하십시오. 

## EC2 인스턴스 생성 및 실행

AWS 계정에 로그인한 후 “EC2" (:numref:`fig_aws`에서 빨간색 상자로 표시) 를 클릭하여 EC2 패널로 이동합니다. 

![Open the EC2 console.](../img/aws.png)
:width:`400px`
:label:`fig_aws`

:numref:`fig_ec2`는 민감한 계정 정보가 회색으로 표시된 EC2 패널을 보여 줍니다. 

![EC2 panel.](../img/ec2.png)
:width:`700px`
:label:`fig_ec2`

### 위치 사전 설정 대기 시간을 줄이려면 가까운 데이터 센터를 선택합니다 (예: “Oregon” (:numref:`fig_ec2`의 오른쪽 상단에 빨간색 상자로 표시).중국에 거주하는 경우 서울 또는 도쿄와 같은 인근 아시아 태평양 지역을 선택할 수 있습니다.일부 데이터 센터에는 GPU 인스턴스가 없을 수 있습니다. 

### 한도 증가 인스턴스를 선택하기 전에 :numref:`fig_ec2`와 같이 왼쪽 막대의 “제한” 레이블을 클릭하여 수량 제한이 있는지 확인하십시오. :numref:`fig_limits`에서는 이러한 제한의 예를 보여 줍니다.계정은 현재 리전별로 “p2.xlarge” 인스턴스를 열 수 없습니다.하나 이상의 인스턴스를 열어야 하는 경우 “요청 한도 증가” 링크를 클릭하여 더 높은 인스턴스 할당량을 신청하십시오.일반적으로 신청서를 처리하는 데 영업일 기준 1일이 소요됩니다. 

![Instance quantity restrictions.](../img/limits.png)
:width:`700px`
:label:`fig_limits`

### 인스턴스 시작 다음으로 :numref:`fig_ec2`에서 빨간색 상자로 표시된 “인스턴스 시작” 버튼을 클릭하여 인스턴스를 시작합니다. 

먼저 적합한 AMI (AWS 머신 이미지) 를 선택합니다.검색 상자에 “우분투”를 입력합니다 (:numref:`fig_ubuntu`에서 빨간색 상자로 표시). 

![Choose an operating system.](../img/ubuntu-new.png)
:width:`700px`
:label:`fig_ubuntu`

EC2는 선택할 수 있는 다양한 인스턴스 구성을 제공합니다.초보자에게는 때때로 압도적으로 느껴질 수 있습니다.다음은 적합한 기계 표입니다. 

| Name | GPU         | Notes                         |
|------|-------------|-------------------------------|
| g2   | Grid K520   | ancient                       |
| p2   | Kepler K80  | old but often cheap as spot   |
| g3   | Maxwell M60 | good trade-off                |
| p3   | Volta V100  | high performance for FP16     |
| g4   | Turing T4   | inference optimized FP16/INT8 |

위의 모든 서버는 사용되는 GPU 수를 나타내는 여러 가지 형태로 제공됩니다.예를 들어, p2.xlarge에는 1개의 GPU가 있고 p2.16xlarge에는 16개의 GPU와 더 많은 메모리가 있습니다.자세한 내용은 [AWS EC2 documentation](https732293614)를 참조하십시오. 

**참고: ** 적합한 드라이버와 GPU가 활성화된 MXNet 버전이 있는 GPU 지원 인스턴스를 사용해야 합니다.그렇지 않으면 GPU를 사용해도 아무런 이점이 없습니다.

![Choose an instance.](../img/p2x.png)
:width:`700px`
:label:`fig_p2x`

지금까지 :numref:`fig_disk`의 맨 위에 표시된 대로 EC2 인스턴스를 시작하는 7단계 중 처음 두 단계를 완료했습니다.이 예에서는 “3 단계에 대한 기본 구성을 유지합니다.인스턴스 구성”, “5.태그 추가”, “6.보안 그룹 구성”.“4”를 탭합니다.스토리지를 추가하고 기본 하드 디스크 크기를 64GB (:numref:`fig_disk`의 빨간색 상자에 표시) 로 늘립니다.CUDA 자체는 이미 4GB를 차지합니다. 

![Modify instance hard disk size.](../img/disk.png)
:width:`700px`
:label:`fig_disk`

마지막으로 “7”로 이동합니다.검토"를 클릭하고 “Launch”를 클릭하여 구성된 인스턴스를 시작합니다.이제 인스턴스에 액세스하는 데 사용된 키 페어를 선택하라는 메시지가 표시됩니다.키 페어가 없는 경우 :numref:`fig_keypair`의 첫 번째 드롭다운 메뉴에서 “새 키 쌍 만들기”를 선택하여 키 페어를 생성합니다.그런 다음 이 메뉴에서 “기존 키 페어 선택”을 선택한 다음 이전에 생성된 키 페어를 선택할 수 있습니다.“인스턴스 시작”을 클릭하여 생성된 인스턴스를 시작합니다. 

![Select a key pair.](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`

키 페어를 새로 생성한 경우 키 페어를 다운로드하여 안전한 위치에 저장해야 합니다.SSH를 서버에 연결하는 유일한 방법입니다.:numref:`fig_launching`에 표시된 인스턴스 ID를 클릭하여 이 인스턴스의 상태를 확인합니다. 

![Click the instance ID.](../img/launching.png)
:width:`700px`
:label:`fig_launching`

### 인스턴스에 연결

:numref:`fig_connect`에서와 같이 인스턴스 상태가 녹색으로 바뀌면 인스턴스를 마우스 오른쪽 버튼으로 클릭하고 `Connect`를 선택하여 인스턴스 액세스 방법을 확인합니다. 

![View instance access and startup method.](../img/connect.png)
:width:`700px`
:label:`fig_connect`

새 키인 경우 SSH가 작동하려면 공개적으로 볼 수 없어야 합니다.`D2L_key.pem`를 저장한 폴더 (예: 다운로드 폴더) 로 이동하여 키를 공개적으로 볼 수 없는지 확인합니다.

```bash
cd /Downloads  ## if D2L_key.pem is stored in Downloads folder
chmod 400 D2L_key.pem
```

![View instance access and startup method.](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`

이제 :numref:`fig_chmod`의 아래쪽 빨간색 상자에 ssh 명령을 복사하여 명령줄에 붙여 넣습니다.

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```

명령행에 “연결을 계속하시겠습니까 (예/아니요)" 라는 메시지가 표시되면 “yes”를 입력하고 Enter 키를 눌러 인스턴스에 로그인합니다. 

이제 서버가 준비되었습니다. 

## CUDA 설치하기

CUDA를 설치하기 전에 인스턴스를 최신 드라이버로 업데이트해야 합니다.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```

여기서 CUDA 10.1을 다운로드합니다.NVIDIA의 [공식 저장소](https://://developer.nvidia.com/cuda-downloads) to find the download link of CUDA 10.1 as shown in :numref:`fig_cuda`) 를 방문하십시오. 

![Find the CUDA 10.1 download address.](../img/cuda101.png)
:width:`500px`
:label:`fig_cuda`

지침을 복사하여 터미널에 붙여 넣어 CUDA 10.1을 설치합니다.

```bash
## Paste the copied link from CUDA website
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

프로그램을 설치한 후 다음 명령을 실행하여 GPU를 확인합니다.

```bash
nvidia-smi
```

마지막으로 CUDA를 라이브러리 경로에 추가하여 다른 라이브러리에서 쉽게 찾을 수 있도록 합니다.

```bash
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> ~/.bashrc
```

## MXNet 설치 및 D2L 노트북 다운로드

먼저 설치를 단순화하려면 Linux용 [Miniconda](https://conda.io/en/latest/miniconda.html)를 설치해야 합니다.다운로드 링크 및 파일 이름은 변경될 수 있으므로 Miniconda 웹 사이트로 이동하여 :numref:`fig_miniconda`와 같이 “링크 주소 복사”를 클릭하십시오. 

![Download Miniconda.](../img/miniconda.png)
:width:`700px`
:label:`fig_miniconda`

```bash
# The link and file name are subject to changes
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b
```

미니콘다 설치 후 다음 명령을 실행하여 CUDA 및 conda를 활성화합니다.

```bash
~/miniconda3/bin/conda init
source ~/.bashrc
```

다음으로 이 책의 코드를 다운로드합니다.

```bash
sudo apt-get install unzip
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

그런 다음 콘다 `d2l` 환경을 만들고 `y`를 입력하여 설치를 계속합니다.

```bash
conda create --name d2l -y
```

`d2l` 환경을 만든 후 환경을 활성화하고 `pip`를 설치합니다.

```bash
conda activate d2l
conda install python=3.7 pip -y
```

마지막으로 MXNet과 `d2l` 패키지를 설치합니다.접미사 `cu101`는 이것이 CUDA 10.1 변형임을 의미합니다.CUDA 10.0이라고 말하는 다른 버전의 경우 대신 `cu100`를 선택하는 것이 좋습니다.

```bash
pip install mxnet-cu101==1.7.0
pip install git+https://github.com/d2l-ai/d2l-en
```

다음과 같이 모든 것이 잘 진행되었는지 빠르게 테스트 할 수 있습니다.

```
$ python
>>> from mxnet import np, npx
>>> np.zeros((1024, 1024), ctx=npx.gpu())
```

## 달리기 목성

Jupyter를 원격으로 실행하려면 SSH 포트 포워딩을 사용해야 합니다.결국 클라우드의 서버에는 모니터 나 키보드가 없습니다.이를 위해 다음과 같이 데스크톱 (또는 랩톱) 에서 서버에 로그인하십시오.

```
# This command must be run in the local command line
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
conda activate d2l
jupyter notebook
```

:numref:`fig_jupyter`는 주피터 노트북을 실행한 후 가능한 출력을 보여줍니다.마지막 행은 포트 8888에 대한 URL입니다. 

![Output after running Jupyter Notebook. The last row is the URL for port 8888.](../img/jupyter.png)
:width:`700px`
:label:`fig_jupyter`

포트 8889로 포트 포워딩을 사용했기 때문에 로컬 브라우저에서 URL을 열 때 포트 번호를 바꾸고 Jupyter가 제공한 암호를 사용해야 합니다. 

## 미사용 인스턴스 닫기

클라우드 서비스는 사용 시간에 따라 요금이 청구되므로 사용하지 않는 인스턴스는 닫아야 합니다.다른 방법이 있습니다. 인스턴스를 “중지”하면 인스턴스를 다시 시작할 수 있습니다.이는 일반 서버의 전원을 끄는 것과 비슷합니다.그러나 중지된 인스턴스에는 보존된 하드 디스크 공간에 대해 소액의 요금이 청구됩니다.“종료”는 연결된 모든 데이터를 삭제합니다.여기에는 디스크가 포함되므로 다시 시작할 수 없습니다.나중에 필요하지 않을 것이라는 것을 알고있는 경우에만 수행하십시오. 

인스턴스를 더 많은 인스턴스에 대한 템플릿으로 사용하려면 :numref:`fig_connect`의 예제를 마우스 오른쪽 버튼으로 클릭하고 “이미지” $\rightarrow$ “만들기”를 선택하여 인스턴스 이미지를 만듭니다.이 작업이 완료되면 “인스턴스 상태” $\rightarrow$ “종료”를 선택하여 인스턴스를 종료합니다.다음에 이 인스턴스를 사용하려는 경우 이 단원에 설명된 EC2 인스턴스 생성 및 실행 단계에 따라 저장된 이미지를 기반으로 인스턴스를 생성할 수 있습니다.유일한 차이점은 “1.:numref:`fig_ubuntu`에 표시된 “AMI”를 선택합니다. 왼쪽의 “내 AMI” 옵션을 사용하여 저장된 이미지를 선택해야 합니다.생성된 인스턴스는 이미지 하드 디스크에 저장된 정보를 유지합니다.예를 들어 CUDA 및 기타 런타임 환경을 다시 설치할 필요가 없습니다. 

## 요약

* 자체 컴퓨터를 구입하여 구축할 필요 없이 온디맨드로 인스턴스를 시작하고 중지할 수 있습니다.
* 사용하기 전에 적합한 GPU 드라이버를 설치해야 합니다.

## 연습문제

1. 클라우드는 편리함을 제공하지만 저렴하지는 않습니다.[spot instances](https://aws.amazon.com/ec2/spot/)를 출시하여 가격을 낮추는 방법을 알아보십시오.
1. 다양한 GPU 서버로 실험해 보세요.얼마나 빠릅니까?
1. 다중 GPU 서버로 실험해 보십시오.얼마나 잘 확장할 수 있을까요?

[Discussions](https://discuss.d2l.ai/t/423)
