# 인코더-디코더 아키텍처
:label:`sec_encoder-decoder`

:numref:`sec_machine_translation`에서 논의했듯이 기계 번역은 입력과 출력이 모두 가변 길이 시퀀스인 시퀀스 변환 모델의 주요 문제 영역입니다.이러한 유형의 입력과 출력을 처리하기 위해 두 가지 주요 구성 요소로 아키텍처를 설계할 수 있습니다.첫 번째 구성 요소는*encoder*입니다. 가변 길이 시퀀스를 입력으로 받아 고정 모양의 상태로 변환합니다.두 번째 구성 요소는*디코더*입니다. 고정 모양의 인코딩된 상태를 가변 길이 시퀀스에 매핑합니다.이를 :numref:`fig_encoder_decoder`에 묘사된*인코더-디코더* 아키텍처라고 합니다. 

![The encoder-decoder architecture.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

영어에서 프랑스어로 기계 번역을 예로 들어 보겠습니다.영어로 된 입력 시퀀스: “They”, “are”, “시청 중”,”.“, 이 인코더-디코더 아키텍처는 먼저 가변 길이 입력을 상태로 인코딩 한 다음 상태를 디코딩하여 변환된 시퀀스 토큰을 출력으로 토큰으로 생성합니다: “Ils”, “regardent”, “.”.인코더-디코더 아키텍처는 후속 섹션에서 서로 다른 시퀀스 변환 모델의 기초를 형성하므로 이 섹션에서는 이 아키텍처를 나중에 구현할 인터페이스로 변환합니다. 

## (**인코더**)

인코더 인터페이스에서는 인코더가 가변 길이 시퀀스를 입력 `X`로 사용하도록 지정합니다.구현은 이 기본 `Encoder` 클래스를 상속하는 모든 모델에서 제공됩니다.

```{.python .input}
from mxnet.gluon import nn

#@save
class Encoder(nn.Block):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
from torch import nn

#@save
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

#@save
class Encoder(tf.keras.layers.Layer):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def call(self, X, *args, **kwargs):
        raise NotImplementedError
```

## [**디코더**]

다음 디코더 인터페이스에서는 인코더 출력 (`enc_outputs`) 을 인코딩된 상태로 변환하는 추가 `init_state` 함수를 추가합니다.이 단계에서는 :numref:`subsec_mt_data_loading`에서 설명한 유효한 입력 길이와 같은 추가 입력이 필요할 수 있습니다.토큰에 의해 가변 길이 시퀀스 토큰을 생성하기 위해, 디코더는 입력 (예를 들어, 이전 시간 단계에서 생성된 토큰) 과 인코딩된 상태를 현재 시간 단계에서 출력 토큰으로 매핑할 수 있을 때마다.

```{.python .input}
#@save
class Decoder(nn.Block):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
#@save
class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
#@tab tensorflow
#@save
class Decoder(tf.keras.layers.Layer):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def call(self, X, state, **kwargs):
        raise NotImplementedError
```

## [**인코더와 디코더의 결합**]

결국 인코더-디코더 아키텍처에는 인코더와 디코더가 모두 포함되며 선택적으로 추가 인수가 있습니다.순방향 전파에서 인코더의 출력은 인코딩 된 상태를 생성하는 데 사용되며, 이 상태는 디코더에서 입력 중 하나로 추가로 사용됩니다.

```{.python .input}
#@save
class EncoderDecoder(nn.Block):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

```{.python .input}
#@tab pytorch
#@save
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

```{.python .input}
#@tab tensorflow
#@save
class EncoderDecoder(tf.keras.Model):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args, **kwargs):
        enc_outputs = self.encoder(enc_X, *args, **kwargs)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state, **kwargs)
```

인코더-디코더 아키텍처에서 “상태”라는 용어는 아마도 상태를 가진 신경망을 사용하여 이 아키텍처를 구현하도록 영감을 주었을 것입니다.다음 섹션에서는 이 인코더-디코더 아키텍처를 기반으로 시퀀스 변환 모델을 설계하기 위해 RNN을 적용하는 방법을 살펴보겠습니다. 

## 요약

* 인코더-디코더 아키텍처는 가변 길이 시퀀스인 입력 및 출력을 처리할 수 있으므로 기계 변환과 같은 시퀀스 변환 문제에 적합합니다.
* 인코더는 가변 길이 시퀀스를 입력으로 받아 모양이 고정 된 상태로 변환합니다.
* 디코더는 고정된 형상의 인코딩된 상태를 가변 길이 시퀀스에 매핑한다.

## 연습문제

1. 신경망을 사용하여 인코더-디코더 아키텍처를 구현한다고 가정합니다.인코더와 디코더가 동일한 유형의 신경망이어야 합니까?  
1. 기계 번역 외에도 인코더-디코더 아키텍처를 적용할 수 있는 다른 응용 분야를 생각해 볼 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1061)
:end_tab:
