# 파라미터 서버
:label:`sec_parameterserver`

단일 GPU에서 여러 GPU로 이동한 다음 여러 GPU가 포함된 여러 서버로 이동할 때 (아마도 모두 여러 랙과 네트워크 스위치에 분산되어 있을 수 있음) 분산 및 병렬 학습을 위한 알고리즘은 훨씬 더 정교해질 필요가 있습니다.상호 연결마다 대역폭이 매우 다르기 때문에 세부 사항이 중요합니다 (예: NVLink는 적절한 설정에서 6 개의 링크에서 최대 100Gb/s를 제공 할 수 있고 PCIe 4.0 (16 레인) 은 32Gb/s를 제공하며 고속 100GbE 이더넷도 10Gb/s에 불과합니다).동시에 통계 모델러가 네트워킹 및 시스템 전문가가 될 것으로 기대하는 것은 무리입니다. 

매개 변수 서버의 핵심 아이디어는 분산 잠재 변수 모델의 맥락에서 :cite:`Smola.Narayanamurthy.2010`에 도입되었습니다.푸시 앤 풀 의미에 대한 설명은 :cite:`Ahmed.Aly.Gonzalez.ea.2012`에서 이어졌고 시스템 및 오픈 소스 라이브러리에 대한 설명은 :cite:`Li.Andersen.Park.ea.2014`에서 이어졌습니다.다음에서는 효율성에 필요한 구성 요소에 동기를 부여합니다. 

## 데이터 병렬 교육

분산 훈련에 대한 데이터 병렬 훈련 접근법을 검토해 보겠습니다.실제로 구현하기가 훨씬 간단하기 때문에 이 섹션의 다른 모든 항목을 제외하는 데 사용합니다.현재 GPU에는 메모리가 충분하기 때문에 병렬 처리에 대한 다른 전략이 선호되는 사용 사례는 거의 없습니다 (그래프의 딥 러닝 제외). :numref:`fig_parameterserver`는 :numref:`sec_multi_gpu`에서 구현한 데이터 병렬 처리의 변형을 설명합니다.주요 측면은 업데이트된 매개 변수가 모든 GPU로 재브로드캐스트되기 전에 GPU 0에서 그래디언트 집계가 발생한다는 것입니다. 

![Left: single GPU training. Right: a variant of multi-GPU training: (1) we compute loss and gradient, (2) all gradients are aggregated on one GPU, (3) parameter update happens and the parameters are re-distributed to all GPUs.](../img/ps.svg)
:label:`fig_parameterserver`

돌이켜 보면 GPU 0에서 집계하기로 한 결정은 다소 임시로 보입니다.결국 CPU를 집계 할 수도 있습니다.실제로 한 GPU에서 일부 매개 변수를 집계하고 다른 GPU에서 일부 매개 변수를 집계하기로 결정할 수도 있습니다.최적화 알고리즘이 이를 지원한다면 우리가 할 수 없었던 실질적인 이유는 없습니다.예를 들어, 연관된 그래디언트 $\mathbf{g}_1, \ldots, \mathbf{g}_4$가 있는 4개의 파라미터 벡터가 있는 경우 각 $\mathbf{g}_i$ ($i = 1, \ldots, 4$) 에 대해 하나의 GPU에서 그래디언트를 집계할 수 있습니다. 

이 추론은 임의적이고 경솔한 것처럼 보입니다.결국 수학은 전체적으로 동일합니다.그러나 :numref:`sec_hardware`에서 설명한 것처럼 버스마다 대역폭이 다른 실제 하드웨어를 다루고 있습니다.:numref:`fig_bw_hierarchy`에 설명된 대로 실제 4방향 GPU 서버를 생각해 보십시오.특히 잘 연결되어 있으면 100GbE 네트워크 카드가 있을 수 있습니다.보다 일반적인 수치는 1-10GbE 범위에 있으며 유효 대역폭은 100MB/s에서 1GB/s입니다. CPU의 PCIe 레인이 너무 적어서 모든 GPU에 직접 연결할 수 없으므로 (예: 소비자 등급 인텔 CPU에는 24개의 레인이 있음) [multiplexer](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches)이 필요합니다.16x Gen3 링크의 CPU 대역폭은 16Gb/s이며, 이는 GPU의*각*이 스위치에 연결되는 속도이기도 합니다.즉, 장치 간 통신이 더 효과적입니다. 

![A 4-way GPU server.](../img/bw-hierarchy.svg)
:label:`fig_bw_hierarchy`

논증을 위해 그라디언트가 160MB라고 가정 해 보겠습니다.이 경우 나머지 3개의 GPU에서 네 번째 GPU로 그라디언트를 전송하는 데 30ms가 걸립니다 (각 전송에는 10ms = 160MB/16Gb/s가 소요됨).가중치 벡터를 다시 전송하기 위해 30ms를 더 추가하면 총 60ms에 도달합니다.모든 데이터를 CPU로 보내면 4개의 GPU의*각각*이 데이터를 CPU로 보내야 하므로 총 80ms의 벌금이 발생합니다.마지막으로 그라디언트를 각각 40MB의 네 부분으로 나눌 수 있다고 가정합니다.이제 PCIe 스위치가 모든 링크 간에 전체 대역폭 작업을 제공하므로 다른 GPU의 각 부분을 동시에* 집계 할 수 있습니다.30ms 대신 7.5ms가 걸리므로 동기화 작업에 총 15ms가 생성됩니다.간단히 말해 매개 변수를 동기화하는 방법에 따라 동일한 작업이 15ms에서 80ms까지 걸릴 수 있습니다. :numref:`fig_ps_distributed`는 매개 변수 교환을 위한 다양한 전략을 보여줍니다. 

![Parameter synchronization strategies.](../img/ps-distributed.svg)
:label:`fig_ps_distributed`

[Horovod](https://github.com/horovod/horovod)에서 이 작업을 수행하는 방법에 대한 자세한 내용은 성능 향상과 관련하여 : in a deep network it takes some time to compute all gradients from the top to the bottom. We can begin synchronizing gradients for some parameter groups even while we are still busy computing them for others. See e.g., :cite:`Sergeev.Del-Balso.2018`를 사용할 수 있는 또 다른 도구를 사용할 수 있습니다. 

## 링 동기화

최신 딥 러닝 하드웨어에서의 동기화와 관련하여 상당히 맞춤화된 네트워크 연결을 자주 접하게 됩니다.예를 들어, AWS p3.16xlarge와 엔비디아 DGX-2 인스턴스는 :numref:`fig_nvlink`의 연결 구조를 공유합니다.각 GPU는 최대 16Gb/s에서 작동하는 PCIe 링크를 통해 호스트 CPU에 연결됩니다. 또한 각 GPU에는 6 개의 NVLink 연결이 있으며 각 연결은 양방향으로 300Gbit/s를 전송할 수 있습니다.이는 방향당 링크당 약 18Gb/s입니다.요컨대, 총 NVLink 대역폭은 PCIe 대역폭보다 훨씬 높습니다.문제는 가장 효율적으로 사용하는 방법입니다. 

![NVLink connectivity on 8  V100 GPU servers (image courtesy of NVIDIA).](../img/nvlink.svg)
:label:`fig_nvlink`

최적의 동기화 전략은 네트워크를 두 개의 링으로 분해하고이를 사용하여 데이터를 직접 동기화하는 것입니다. :numref:`fig_nvlink_twoloop`는 네트워크가 이중 NVLink 대역폭을 갖는 하나의 링 (1-2-3-4-5-6-7-8-1) 으로 분해 될 수 있음을 보여줍니다 (1-4-6-3-5-8-2-7-1)일반 대역폭.이 경우 효율적인 동기화 프로토콜을 설계하는 것은 중요하지 않습니다. 

![Decomposition of the NVLink network into two rings.](../img/nvlink-twoloop.svg)
:label:`fig_nvlink_twoloop`

다음과 같은 사고 실험을 생각해 보십시오. $n$개의 컴퓨팅 노드 (또는 GPU) 의 링이 주어지면 첫 번째 노드에서 두 번째 노드로 그라디언트를 보낼 수 있습니다.로컬 그래디언트에 추가되고 세 번째 노드로 전송되는 식입니다.$n-1$ 단계 후에는 마지막으로 방문한 노드에서 집계 그래디언트를 찾을 수 있습니다.즉, 그래디언트를 집계하는 시간은 노드 수에 따라 선형적으로 증가합니다.하지만 이렇게 하면 알고리즘은 상당히 비효율적입니다.결국 언제든지 노드 중 하나만 통신합니다.그래디언트를 $n$ 청크로 나누고 노드 $i$에서 시작하여 청크 $i$를 동기화하기 시작하면 어떻게 될까요?각 청크의 크기는 $1/n$이므로 총 시간은 이제 $(n-1)/n \approx 1$입니다.즉, 링의 크기를 늘려도 그라디언트를 집계하는 데 소요되는 시간은*증가하지 않습니다*.이것은 매우 놀라운 결과입니다. :numref:`fig_ringsync`는 $n=4$ 노드에서 일련의 단계를 보여줍니다. 

![Ring synchronization across 4 nodes. Each node starts transmitting parts of gradients to its left neighbor until the assembled gradient can be found in its right neighbor.](../img/ringsync.svg)
:label:`fig_ringsync`

8개의 V100 GPU에서 160MB를 동기화하는 동일한 예제를 사용하면 약 $2 \cdot 160 \mathrm{MB} / (3 \cdot 18 \mathrm{GB/s}) \approx 6 \mathrm{ms}$에 도달합니다.현재 8개의 GPU를 사용하고 있지만 PCIe 버스를 사용하는 것보다 낫습니다.딥 러닝 프레임워크가 통신을 대규모 버스트 전송으로 결합하지 못하는 경우가 많기 때문에 실제로는 이러한 수치가 약간 더 나쁩니다.  

링 동기화가 다른 동기화 알고리즘과 근본적으로 다르다는 일반적인 오해가 있습니다.유일한 차이점은 간단한 트리와 비교할 때 동기화 경로가 다소 정교하다는 것입니다. 

## 다중 기계 교육

여러 머신에 대한 분산 교육은 또 다른 과제를 추가합니다. 비교적 낮은 대역폭 패브릭에서만 연결된 서버와 통신해야 하는데, 경우에 따라 몇 배 이상 느려질 수 있습니다.장치 간 동기화는 까다롭습니다.결국 훈련 코드를 실행하는 기계마다 속도가 미묘하게 다릅니다.따라서 동기식 분산 최적화를 사용하려면 동기화*동기화*해야 합니다. :numref:`fig_ps_multimachine`는 분산 병렬 훈련이 어떻게 발생하는지 보여줍니다. 

1. (다른) 데이터 배치가 각 머신에서 읽혀지고 여러 GPU로 분할되어 GPU 메모리로 전송됩니다.각 GPU 배치에서 개별적으로 예측과 기울기가 계산됩니다.
2. 모든 로컬 GPU의 그래디언트는 하나의 GPU에서 집계됩니다 (또는 일부가 다른 GPU에 걸쳐 집계됨).
3. 그래디언트가 CPU로 전송됩니다.
4. CPU는 모든 그라디언트를 집계하는 중앙 매개 변수 서버로 그라디언트를 보냅니다.
5. 그런 다음 집계 그라디언트를 사용하여 파라미터를 업데이트하고 업데이트된 파라미터는 개별 CPU로 다시 브로드캐스트됩니다.
6. 정보는 하나 (또는 여러) GPU로 전송됩니다.
7. 업데이트된 파라미터는 모든 GPU에 분산됩니다.

![Multi-machine multi-GPU distributed parallel training.](../img/ps-multimachine.svg)
:label:`fig_ps_multimachine`

이러한 각 작업은 다소 간단해 보입니다.그리고 실제로 단일 기계 내에서* 효율적으로 수행 할 수 있습니다.하지만 여러 머신을 살펴보면 중앙 파라미터 서버가 병목 현상이 발생한다는 것을 알 수 있습니다.결국 서버당 대역폭은 제한되어 있으므로 $m$ 작업자의 경우 모든 그래디언트를 서버로 보내는 데 걸리는 시간은 $\mathcal{O}(m)$입니다.서버 수를 $n$로 늘려 이 장벽을 극복할 수 있습니다.이 시점에서 각 서버는 매개 변수의 $\mathcal{O}(1/n)$만 저장하면 되므로 총 업데이트 및 최적화 시간은 $\mathcal{O}(m/n)$이 됩니다.두 숫자를 일치시키면 처리하는 워커 수에 관계없이 일정한 스케일링이 가능합니다.실제로는*동일한* 기계를 작업자와 서버 모두로 사용합니다. :numref:`fig_ps_multips`는 설계를 보여줍니다 (자세한 내용은 :cite:`Li.Andersen.Park.ea.2014` 참조).특히 여러 대의 기계가 불합리한 지연 없이 작동하도록 보장하는 것은 결코 쉬운 일이 아닙니다.장벽에 대한 자세한 내용은 생략하고 아래에서 동기 및 비동기 업데이트에 대해서만 간략하게 설명하겠습니다. 

![Top: a single parameter server is a bottleneck since its bandwidth is finite. Bottom: multiple parameter servers store parts of the parameters with aggregate bandwidth.](../img/ps-multips.svg)
:label:`fig_ps_multips`

## 키—값 스토어

분산 다중 GPU 훈련에 필요한 단계를 실제로 구현하는 것은 간단하지 않습니다.이것이 바로 일반적인 추상화, 즉 업데이트 시맨틱스가 재정의된*키-값 저장소*의 추상화를 사용하는 것이 좋은 이유입니다.  

많은 워커와 많은 GPU에서 기울기 $i$에 대한 계산은 다음과 같이 정의할 수 있습니다. 

$$\mathbf{g}_{i} = \sum_{k \in \text{workers}} \sum_{j \in \text{GPUs}} \mathbf{g}_{ijk},$$

여기서 $\mathbf{g}_{ijk}$은 작업자 $k$의 GPU $j$에서 분할된 그래디언트 $i$의 일부입니다.이 연산의 핵심 측면은*교환 환원*이라는 것입니다. 즉, 많은 벡터를 하나로 바꾸고 연산이 적용되는 순서는 중요하지 않습니다.그래디언트가 수신되는 시점을 세밀하게 제어할 필요가 없기 때문에 이것은 우리의 목적에 아주 좋습니다.게다가, 이 작업은 서로 다른 $i$ 사이에서 독립적입니다. 

이를 통해 다음 두 가지 작업을 정의할 수 있습니다. 그래디언트를 누적하는*push*와 집계 그래디언트를 검색하는*pull*입니다.다양한 그라디언트 세트가 있으므로 (결국 많은 레이어가 있음) 키 $i$로 그라디언트를 인덱싱해야합니다.Dynamo :cite:`DeCandia.Hastorun.Jampani.ea.2007`에 도입된 것과 같은 키-값 저장소와의 이러한 유사성은 우연이 아닙니다.또한 매개 변수를 여러 서버에 분산시킬 때 특히 유사한 특성을 많이 충족합니다. 

키-값 스토어에 대한 푸시 및 풀 작업은 다음과 같습니다. 

* **push (key, value) **는 워커에서 공통 스토리지로 특정 그래디언트 (값) 를 전송합니다.여기서 값은 집계됩니다 (예: 합산).
* **pull (key, value) **은 예를 들어 모든 워커의 그래디언트를 결합한 후 공통 저장소에서 집계 값을 검색합니다.

간단한 푸시 앤 풀 작업 뒤에 동기화에 대한 모든 복잡성을 숨김으로써 간단한 용어로 최적화를 표현하려는 통계 모델러와 분산 동기화에 내재된 복잡성을 처리해야 하는 시스템 엔지니어의 우려를 분리할 수 있습니다. 

## 요약

* 동기화는 서버 내의 특정 네트워크 인프라 및 연결에 매우 적합해야 합니다.이렇게 하면 동기화하는 데 걸리는 시간이 크게 달라질 수 있습니다.
* 링 동기화는 p3 및 DGX-2 서버에 최적일 수 있습니다.다른 사람들에게는 그렇게 많지 않을 수도 있습니다.
* 계층 동기화 전략은 대역폭을 늘리기 위해 여러 매개 변수 서버를 추가할 때 효과적입니다.

## 연습문제

1. 링 동기화를 더 늘릴 수 있습니까?힌트: 메시지를 양방향으로 보낼 수 있습니다.
1. (계산이 진행 중일 때) 비동기 통신을 허용할 수 있습니까?성능에 어떤 영향을 미칩니 까?
1. 장기 실행 계산 중에 서버가 손실되면 어떻게 될까요?계산을 완전히 다시 시작하지 않도록 내결함성* 메커니즘을 어떻게 설계할 수 있습니까?

[Discussions](https://discuss.d2l.ai/t/366)
