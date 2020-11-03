# GPUs
# GPU

0.15.0

:label:`sec_use_gpu`

In :numref:`tab_intro_decade`, we discussed the rapid growth
of computation over the past two decades.
In a nutshell, GPU performance has increased
by a factor of 1000 every decade since 2000.
This offers great opportunities but it also suggests
a significant need to provide such performance.

:numref:`tab_intro_decade` 에서 지난 20여년 동안 급격하게 발전한 연산능력에 대해서 이야기했습니다. 간단히 말해서 2000년 부터 10년 마다 GPU 성능이 1000배씩 향상되었습니다. 이는 커다란 기회를 가져왔는데, 하지만 이는 또한 그런 성능을 필요로 하는 요건도 가져왔습니다.

In this section, we begin to discuss how to harness
this computational performance for your research.
First by using single GPUs and at a later point,
how to use multiple GPUs and multiple servers (with multiple GPUs).

이 절에서 우리는 여러분의 연구에 이 연산 성능을 잘 사용하는 방법부터 논의하겠습니다. 우선 한 개의 GPU를 사용하고, 나중에는 여러 GPU들 그리고 여러 GPU를 갖은 여러 서버를 사용하는 방법을 봅니다.

Specifically, we will discuss how
to use a single NVIDIA GPU for calculations.
First, make sure you have at least one NVIDIA GPU installed.
Then, download the [NVIDIA driver and CUDA](https://developer.nvidia.com/cuda-downloads)
and follow the prompts to set the appropriate path.
Once these preparations are complete,
the `nvidia-smi` command can be used
to view the graphics card information.

특히 우리는 NVIDIA GPU 한 개를 연산에 어떻게 사용하는지 이야기하겠습니다. 우선, 적어도 한 개의 NVIDIA GPU가 설치되어 있는지 확인하세요. 그리고,  [NVIDIA 드라이버와 CUDA](https://developer.nvidia.com/cuda-downloads) 를 다운로드하고, 설치 안내에 따라서 적절한 경로를 설정하세요. 이 준비가 완료되면,  `nvidia-smi` 명령을 사용해서 그래픽 카드 정보를 확인하세요.

```{.python .input}
#@tab all
!nvidia-smi
```

:begin_tab:`mxnet`
You might have noticed that a MXNet tensor
looks almost identical to a NumPy `ndarray`.
But there are a few crucial differences.
One of the key features that distinguishes MXNet
from NumPy is its support for diverse hardware devices.

여러분은 아마도 MXNet 텐서가 NumPy의  `ndarray` 와 거의 유사하다는 것을 눈치챘을 것입니다. 하지만, 몇 가지 중요한 차이가 있습니다. NumPy와 달리 MXNet에서 제공하는 주요 특징 중에 하나는 다양한 하드웨어 장치를 지원한다는 것입니다.

In MXNet, every array has a context.
So far, by default, all variables
and associated computation
have been assigned to the CPU.
Typically, other contexts might be various GPUs.
Things can get even hairier when
we deploy jobs across multiple servers.
By assigning arrays to contexts intelligently,
we can minimize the time spent
transferring data between devices.
For example, when training neural networks on a server with a GPU,
we typically prefer for the model's parameters to live on the GPU.

MXNet에서 모든 배열은 컨텍스트(context)를 갖습니다. 지금까지 기본 설정에 따라 모든 변수들과 연관된 연산들은 CPU에 할당 되었습니다. 다른 컨텍스트는 다양한 GPU들입니다. 작업을 여러 서버들에 배포하는 경우에는 더 복잡해집니다. 배열들을 컨텍스트에 영리하게 할당함으로 우리는 디바이스들간의 데이터 전송에 소요되는 시간을 최소화 할 수 있습니다. 예를 들어, GPU 한 개를 갖는 서버에서 신경망을 학습시킬 때, 일반적으로우리는 모델의 파라미터들이 GPU에 머물러 있기를 원합니다.

Next, we need to confirm that
the GPU version of MXNet is installed.
If a CPU version of MXNet is already installed,
we need to uninstall it first.
For example, use the `pip uninstall mxnet` command,
then install the corresponding MXNet version
according to your CUDA version.
Assuming you have CUDA 10.0 installed,
you can install the MXNet version
that supports CUDA 10.0 via `pip install mxnet-cu100`.

다음으로 설치된 MXNet의 GPU 버전을 확인해야 합니다. 만약 CPU 버전의 MXNet이 이미 설치되어 있다면, 우선 그것을 삭제하세요. 예를 들어  `pip uninstall mxnet` 명령으로 삭제하고, 여러분의 CUDA 버전에 맞는 MXNet 버전을 설치하세요. 만약 CUDA 10.0이 설치되어 있다면, CUDA 10.0을 지원하는 MXNet 버전을  `pip install mxnet-cu100` 명령으로 설치할 수 있습니다.

:end_tab:

:begin_tab:`pytorch`
In PyTorch, every array has a device, we often refer it as a context.
So far, by default, all variables
and associated computation
have been assigned to the CPU.
Typically, other contexts might be various GPUs.
Things can get even hairier when
we deploy jobs across multiple servers.
By assigning arrays to contexts intelligently,
we can minimize the time spent
transferring data between devices.
For example, when training neural networks on a server with a GPU,
we typically prefer for the model's parameters to live on the GPU.

PyTorch에서 모든 배열은 컨텍스트(context)를 갖습니다. 지금까지 기본 설정에 따라 모든 변수들과 연관된 연산들은 CPU에 할당 되었습니다. 다른 컨텍스트는 다양한 GPU들입니다. 작업을 여러 서버들에 배포하는 경우에는 더 복잡해집니다. 배열들을 컨텍스트에 영리하게 할당함으로 우리는 디바이스들간의 데이터 전송에 소요되는 시간을 최소화 할 수 있습니다. 예를 들어, GPU 한 개를 갖는 서버에서 신경망을 학습시킬 때, 일반적으로우리는 모델의 파라미터들이 GPU에 머물러 있기를 원합니다.

Next, we need to confirm that
the GPU version of PyTorch is installed.
If a CPU version of PyTorch is already installed,
we need to uninstall it first.
For example, use the `pip uninstall torch` command,
then install the corresponding PyTorch version
according to your CUDA version.
Assuming you have CUDA 10.0 installed,
you can install the PyTorch version
that supports CUDA 10.0 via `pip install torch-cu100`.
:end_tab:

다음으로 설치된 PyTorch의 GPU 버전을 확인해야 합니다. 만약 CPU 버전의 PyTorch이 이미 설치되어 있다면, 우선 그것을 삭제하세요. 예를 들어  `pip uninstall torch` 명령으로 삭제하고, 여러분의 CUDA 버전에 맞는 PyTorch 버전을 설치하세요. 만약 CUDA 10.0이 설치되어 있다면, CUDA 10.0을 지원하는 PyTorch 버전을  `pip install torch-cu100` 명령으로 설치할 수 있습니다.

To run the programs in this section,
you need at least two GPUs.
Note that this might be extravagant for most desktop computers
but it is easily available in the cloud, e.g.,
by using the AWS EC2 multi-GPU instances.
Almost all other sections do *not* require multiple GPUs.
Instead, this is simply to illustrate
how data flow between different devices.

이 절의 프로그램들을 실행하기 위해서는 최소 두 개의 GPU가 필요합니다. 대부분의 데스크탑 컴퓨터에는 너무 높은 사양일 것이지만, AWS EC2 다중 GPU 인스턴스와 같은 것을 사용하면 쉽게 얻을 수 있습니다. 다른 대부분의 절들에서는 여러 GPU들이 필요 *없습니다*. 이 내용은 단지 여러 다른 디바이스들안에 데이터가 어떻게 이동하는지를 설명하기 위함입니다.

## Computing Devices
## 연산 디바이스들

We can specify devices, such as CPUs and GPUs,
for storage and calculation.
By default, tensors are created in the main memory
and then use the CPU to calculate it.

우리는 저장과 연산에 사용될 CPU와 GPU 같은 디바이스를 지정할 수 있습니다. 기본 설정으로 텐서는 주 메모리에 생성되고, CPU를 이용해서 연산을 수행합니다.

:begin_tab:`mxnet`
In MXNet, the CPU and GPU can be indicated by `cpu()` and `gpu()`.
It should be noted that `cpu()`
(or any integer in the parentheses)
means all physical CPUs and memory.
This means that MXNet's calculations
will try to use all CPU cores.
However, `gpu()` only represents one card
and the corresponding memory.
If there are multiple GPUs, we use `gpu(i)`
to represent the $i^\mathrm{th}$ GPU ($i$ starts from 0).
Also, `gpu(0)` and `gpu()` are equivalent.

MXNet에서 CPU와 GPU는 `cpu()` 와 `gpu()` 로 각각 지정됩니다.  `cpu()` (또는 인자로 전달되는 어떤 정수값이든지) 모든 물리 CPU 코어를 의미한다는 것을 주지해야합니다. 이는 MXNet의 연산은 모든 CPU 코어들을 사용하려 한다는 것을 의미합니다. 하지만,  `gpu()` 는 오직 한 개의 카드와 그것의 메모리를 의미합니다. 만약 여러개의 GPU가 있다면,  $i$ 번째 GPU를 지정하고 싶다면  `gpu(i)` 와 같이 지정합니다. ( $i$  는 0부터 시작합니다) 또한, `gpu(0)` 와 `gpu()` 는 같은 의미입니다.

:end_tab:

:begin_tab:`pytorch`
In PyTorch, the CPU and GPU can be indicated by `torch.device('cpu')` and `torch.cuda.device('cuda')`.
It should be noted that the `cpu` device
means all physical CPUs and memory.
This means that PyTorch's calculations
will try to use all CPU cores.
However, a `gpu` device only represents one card
and the corresponding memory.
If there are multiple GPUs, we use `torch.cuda.device(f'cuda:{i}')`
to represent the $i^\mathrm{th}$ GPU ($i$ starts from 0).
Also, `gpu:0` and `gpu` are equivalent.

PyTorch에서 CPU와 GPU는 `torch.device('cpu')` 와 `torch.cuda.device('cuda')` 로 각각 지정됩니다.  `cpu`  디바이스는 모든 물리 CPU 코어를 의미한다는 것을 주지해야합니다. 이는 PyTorch의 연산은 모든 CPU 코어들을 사용하려 한다는 것을 의미합니다. 하지만,  `cuda` 는 오직 한 개의 카드와 그것의 메모리를 의미합니다. 만약 여러개의 GPU가 있다면,  $i$ 번째 GPU를 지정하고 싶다면  `torch.cuda.device(f'cuda:{i}')` 와 같이 지정합니다. ( $i$  는 0부터 시작합니다) 또한,  `gpu:0` 와 `gpu` 는 같은 의미입니다.

:end_tab:

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

npx.cpu(), npx.gpu(), npx.gpu(1)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:1')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

tf.device('/CPU:0'), tf.device('/GPU:0'), tf.device('/GPU:1')
```

We can query the number of available GPUs.

가용한 GPU가 몇 개인지를 확인할 수 있습니다.

```{.python .input}
npx.num_gpus()
```

```{.python .input}
#@tab pytorch
torch.cuda.device_count()
```

```{.python .input}
#@tab tensorflow
len(tf.config.experimental.list_physical_devices('GPU'))
```

Now we define two convenient functions that allow us
to run code even if the requested GPUs do not exist.

자, 요청된 GPU가 존재하지 않아도 코드를 수행할 수 있도록 해주는 편리한 함수 두 개를 정의해 봅시다.

```{.python .input}
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    return npx.gpu(i) if npx.num_gpus() >= i + 1 else npx.cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu()] if no GPU exists."""
    devices = [npx.gpu(i) for i in range(npx.num_gpus())]
    return devices if devices else [npx.cpu()]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab pytorch
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab tensorflow
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    devices = [tf.device(f'/GPU:{i}') for i in range(num_gpus)]
    return devices if devices else [tf.device('/CPU:0')]

try_gpu(), try_gpu(10), try_all_gpus()
```

## Tensors and GPUs
## 텐서와 GPU

By default, tensors are created on the CPU.
We can query the device where the tensor is located.

기본 설정으로 텐서는 CPU에 생성됩니다. 텐서가 어떤 디바이스에 위치하고 있는지를 확인할 수 있습니다.

```{.python .input}
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

It is important to note that whenever we want
to operate on multiple terms,
they need to be on the same device.
For instance, if we sum two tensors,
we need to make sure that both arguments
live on the same device---otherwise the framework
would not know where to store the result
or even how to decide where to perform the computation.

여러 항목에 연산을 수행하고자 하면, 그것들은 모두 같은 디바이스에 있어야 한다는 것을 알아두는 것이 중요합니다. 예를 들어, 두 텐서의 합을 구하고자 한다면, 두 인자가 모두 같은 디바이스에 놓아야 합니다. 그렇지 않으며 프레임워크는 결과를 어느 디바이스에 저장할지 모르고, 심지어 어느 디바이스에서 연산을 수행해야 할지도 알 수가 없습니다.

### Storage on the GPU
### GPU의 저장소

There are several ways to store a tensor on the GPU.
For example, we can specify a storage device when creating a tensor.
Next, we create the tensor variable `X` on the first `gpu`.
The tensor created on a GPU only consumes the memory of this GPU.
We can use the `nvidia-smi` command to view GPU memory usage.
In general, we need to make sure that we do not create data that exceed the GPU memory limit.

텐서를 GPU에 저장하는 방법은 여러 가지가 있습니다. 예를 들면 텐서를 생성할 때 저장할 디바이스를 명시할 수 있습니다. 다음에서 우리는 첫 번째  `gpu`에 텐서 변수  `X` 를 생성합니다. GPU에 생성된 텐서는 그 GPU의 메모리만을 사용합니다. GPU 메모리 사용량은  `nvidia-smi` 명령으로 확인이 가능합니다. 일반적으로 우리는 GPU 메모리 제한을 넘는 데이터를 생성하지 않도록 해야 합니다.

```{.python .input}
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
#@tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
#@tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

Assuming that you have at least two GPUs, the following code will create a random tensor on the second GPU.

여러분이 최소 두 개의 GPU가 있다고 가정하면, 다음 코드는 두 번째 GPU에 임의의 값을 갖는 텐서를 생성합니다.

```{.python .input}
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
#@tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

### Copying
### 복사

If we want to compute `X + Y`,
we need to decide where to perform this operation.
For instance, as shown in :numref:`fig_copyto`,
we can transfer `X` to the second GPU
and perform the operation there.
*Do not* simply add `X` and `Y`,
since this will result in an exception.
The runtime engine would not know what to do:
it cannot find data on the same device and it fails.
Since `Y` lives on the second GPU,
we need to move `X` there before we can add the two.

 `X + Y` 를 계산하기 위해서 우리는 이 연산을 수행할 디바이스를 정해야 합니다. 예를 들어,  :numref:`fig_copyto` 처럼  `X` 를 두 번째 GPU로 이전한 후,  두 번째 GPU에서 연산을 수행할 수 있습니다. 단순하게  `X` 와 `Y` 를 더하지 *마세요*. 만약 그렇게 한다면, 오류가 발생할 것입니다. 즉, 런타임 엔진은 무엇을 해야 할지를 모릅니다. 같은 디바이스에서 데이터를 찾을 수 없고, 그로 인해서 실패합니다.  `Y` 는 두 번째 GPU에 있기 때문에, 두 값을 더하기 전에  `X` 를 두 번째 GPU로 옮겨야 합니다.

![Copy data to perform an operation on the same device.](../img/copyto.svg)
:label:`fig_copyto`



```{.python .input}
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
#@tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

Now that the data are on the same GPU
(both `Z` and `Y` are),
we can add them up.

자 이제 데이터들이 모두 같은 GPU( `Z` 와 `Y` )에 있으니, 우리는 이 둘을 더할 수 있습니다.

```{.python .input}
#@tab all
Y + Z
```

:begin_tab:`mxnet`
Imagine that your variable `Z` already lives on your second GPU.
What happens if we still call  `Z.copyto(gpu(1))`?
It will make a copy and allocate new memory,
even though that variable already lives on the desired device.
There are times where, depending on the environment our code is running in,
two variables may already live on the same device.
So we want to make a copy only if the variables
currently live in different devices.
In these cases, we can call `as_in_ctx`.
If the variable already live in the specified device
then this is a no-op.
Unless you specifically want to make a copy,
`as_in_ctx` is the method of choice.

변수 `Z` 가 이미 두 번째 GPU에 있다고 생각해 봅시다.  `Z.copyto(gpu(1))` 를 실행하면 어떤 일이 일어날까요? 원하는 디바이스에 이미 변수가 위치하고 있을 지라도, 새로운 메모리를 할당하고 복사를 수행합니다. 우리의 코드가 수행하는 환경에 따라서 두 변수가 이미 같은 디바이스에 있는 경우가 있습니다. 따라서 우리는 두 변수가 서로 다른 디바이스에 있을 때만 복사를 수행해야 합니다. 이 경우  `as_in_ctx` 를 호출합니다. 만약 그 변수가 이미 해당 디바이스에 위치해 있다면, 이 함수는 no-op이 될 것입니다. 복사를 하는 경우가 아니라면,  `as_in_ctx` 를 사용하세요.

:end_tab:

:begin_tab:`pytorch`
Imagine that your variable `Z` already lives on your second GPU.
What happens if we still call `Z.cuda(1)`?
It will return `Z` instead of making a copy and allocating new memory.

변수 `Z` 가 이미 두 번째 GPU에 있다고 생각해 봅시다.  `Z.cuda(1)` 를 실행하면 어떤 일이 일어날까요? 새로운 메모리를 할당해서 복사하는 대신 `Z`를 반환합니다.

:end_tab:

:begin_tab:`tensorflow`
Imagine that your variable `Z` already lives on your second GPU.
What happens if we still call `Z2 = Z` under the same device scope?
It will return `Z` instead of making a copy and allocating new memory.

변수 `Z` 가 이미 두 번째 GPU에 있다고 생각해 봅시다. 같은 디바이스 범위 아래에서  `Z2 = Z` 를 수행하면 어떤 일이 일어날까요? 새로운 메모리를 할당해서 복사하는 대신 `Z`를 반환합니다.

:end_tab:

```{.python .input}
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
#@tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

### Side Notes
### 사이드 노트

People use GPUs to do machine learning
because they expect them to be fast.
But transferring variables between devices is slow.
So we want you to be 100% certain
that you want to do something slow before we let you do it.
If the deep learning framework just did the copy automatically
without crashing then you might not realize
that you had written some slow code.
사람들은 빠른 속도록 기대하면서 머신 러닝에 GPU를 사용합니다. 하지만, 디바이스간의 변수 전송 속도는 느립니다. 그래서 우리는 여러분에게 어떤 것들을 하도록 시키기 전에, 그것들이 느리게 수행될 것이라는 것을 100% 확신시키고자 합니다. 만약에 딥러닝 프레임워크가 강제 종료되지 않고 자동으로 복사를 실행할 수 있었다면, 여러분은 느리게 동작하는 코드를 작성했다는 사실을 알아차리지 못 했을 것입니다.

Also, transferring data between devices (CPU, GPUs, and other machines)
is something that is much slower than computation.
It also makes parallelization a lot more difficult,
since we have to wait for data to be sent (or rather to be received)
before we can proceed with more operations.
This is why copy operations should be taken with great care.
As a rule of thumb, many small operations
are much worse than one big operation.
Moreover, several operations at a time
are much better than many single operations interspersed in the code
unless you know what you are doing.
This is the case since such operations can block if one device
has to wait for the other before it can do something else.
It is a bit like ordering your coffee in a queue
rather than pre-ordering it by phone
and finding out that it is ready when you are.

또한 디바이스들(CPU, GPU 또는 다른 머신) 사이의 데이터 전송은 연산에 비해 굉장히 느립니다. 그리고 다른 연산을 더 수행하기 앞서 데이터 보내기 (또는 받기)가 끝날 때까지 기달려야 하기 때문에, 이는 병렬화를 아주 어렵게 만듭니다. 이것이 복사 연산을 아주 조심하게 사용해야 하는 이유입니다. 경험에 따르면 많은 작은 연산들은 하나의 큰 연산보다 훨씬 나쁩니다. 또한, 여러분이 무엇을 하고 있는지 알지 못하는 경우라면, 여러 연산을 동시에 수행하는 것은 코드에 잠재된 여러 단일 연산들보다 훨씬 좋습니다. 이것은 디바이스가 어떤 것을 수행하기 전에 다른 연산의 수행이 끝나기를 기다려야한다면 병목이 발생하는 예입니다.

Last, when we print tensors or convert tensors to the NumPy format,
if the data is not in the main memory,
the framework will copy it to the main memory first,
resulting in additional transmission overhead.
Even worse, it is now subject to the dreaded global interpreter lock
that makes everything wait for Python to complete.

마지막으로, 텐서를 출력하거나 NumPy 형태로 바꾸는 경우에 데이터가 주메모리에 없다면 프레임워크는 먼서 텐서를 주메모리에 복사할 것이고, 그 결과 추가된 전송 오버해드가 발생합니다. 더 심한 것은, 이는 Python 수행이 종료될 때까지 모든 것이 멈추는 염려하던 글로벌 인터프린터 락을 적용 받을 수 있습니다.

## Neural Networks and GPUs
## 신경망과 GPU

Similarly, a neural network model can specify devices.
The following code puts the model parameters on the GPU.

비슷하게 신경망 모델은 디바이스를 지정할 수 있습니다. 아래 코드는 모델 파라미터들을 GPU에 놓습니다.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

```{.python .input}
#@tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

We will see many more examples of
how to run models on GPUs in the following chapters,
simply since they will become somewhat more computationally intensive.

우리는 다음 장들에서  더 많은 연산을 필요로 하는 모델들이 GPU에서 수행되는 더 많은 예제를 볼 예정입니다. 

When the input is a tensor on the GPU, the model will calculate the result on the same GPU.

입력이 GPU에 있는 텐서인 경우, 모델은 같은 GPU에서 결과를 계산할 것입니다.

```{.python .input}
#@tab all
net(X)
```

Let us confirm that the model parameters are stored on the same GPU.

모델 파라미터들이 같은 GPU에 있는지 확인해 보겠습니다.

```{.python .input}
net[0].weight.data().ctx
```

```{.python .input}
#@tab pytorch
net[0].weight.data.device
```

```{.python .input}
#@tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

In short, as long as all data and parameters are on the same device, we can learn models efficiently. In the following chapters we will see several such examples.

간략하게 말하면, 모든 데이터와 파라미터들이 같은 디바이스에 있다면 우리는 모델 학습을 효율적으로 할 수 있습니다. 다음 장들에서 우리는 그런 예제들을 볼 것입니다.

## Summary
## 요약

* We can specify devices for storage and calculation, such as the CPU or GPU.
  By default, data are created in the main memory
  and then use the CPU for calculations.
* The deep learning framework requires all input data for calculation
  to be on the same device,
  be it CPU or the same GPU.
* You can lose significant performance by moving data without care.
  A typical mistake is as follows: computing the loss
  for every minibatch on the GPU and reporting it back
  to the user on the command line (or logging it in a NumPy `ndarray`)
  will trigger a global interpreter lock which stalls all GPUs.
  It is much better to allocate memory
  for logging inside the GPU and only move larger logs.



- CPU 또는 GPU와 같이 저장 및 연산을 위한 디바이스를 지정할 수 있습니다. 기본 설정은 모든 데이터는 주메모리에 생성되고, 연산은 CPU를 사용합니다.
- 딥러닝 프레임워크의 연산에 사용되는 모든 입력 데이터가 동일한 디바이스에 있어야 합니다. 즉, CPU 또는 같은 GPU.
- 주의를 기울이지 않고 데이터를 이동하는 경우 심각한 성능 손실이 발생합니다. 전형적인 실수는 다음과 같습니다: GPU의 모든 미니배치마다 손실을 계산하고, 명령행의 사용자에게 리포팅하는 것 (또는 NumPy y `ndarray` 에 기록하는 것)은 글로벌 인터프린터 락을 유발하고, 이는 모든 GPU를 정지시킵니다. GPU에 로깅을 위한 메모리를 할당한 후, 더 큰 로그를 옮기는 것이 더 좋습니다.

## Exercises
## 연습문제
1. Try a larger computation task, such as the multiplication of large matrices,
   and see the difference in speed between the CPU and GPU.
   What about a task with a small amount of calculations?
1. How should we read and write model parameters on the GPU?
1. Measure the time it takes to compute 1000
   matrix-matrix multiplications of $100 \times 100$ matrices
   and log the Frobenius norm of the output matrix one result at a time
   vs. keeping a log on the GPU and transferring only the final result.
1. Measure how much time it takes to perform two matrix-matrix multiplications
   on two GPUs at the same time vs. in sequence
   on one GPU. Hint: you should see almost linear scaling.



1. 큰 행렬들의 곱과 같은 큰 연산을 시도해보고, CPU와 GPU의 속도를 비교해 보세요. 적은 수의 연산을 작는 작업은 어떤가요?
2. GPU의 모델 파라미터를 어떻게 읽고 적을 수 있나요?
3. $100 \times 100$ 행렬들을 사용한 행렬-행렬 곱 1000개를 수행하고 각 결과 행렬의 프로베니우스 놈(Frobenius norm)을 한번에 하나씩 저장하는 것과, 결과를 GPU에 저장하고 있다가 마지막에 최종 결과만 출력하는 것에 대한 시간을 측정하세요.
4. 동시에 두 개 GPU를 사용해서 2 개의 행렬-행렬 곱을 수행하는 것과 하나의 GPU를 사용해서 순서대로 수행하는 것의 시간을 측정해보세요. 힌트: 거의 선형적인 증가를 볼 것입니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/270)
:end_tab:
