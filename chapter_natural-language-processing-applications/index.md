# 자연어 처리: 응용 프로그램
:label:`chap_nlp_app`

우리는 :numref:`chap_nlp_pretrain`에서 텍스트 시퀀스로 토큰을 표현하고 그 표현을 훈련시키는 방법을 살펴보았습니다.이러한 사전 학습된 텍스트 표현은 다양한 다운스트림 자연어 처리 작업을 위해 다양한 모델에 제공될 수 있습니다. 

사실, 이전 장에서는 이미 몇 가지 자연어 처리 응용 프로그램에 대해 논의했습니다.
*사전 교육 없이*
딥 러닝 아키텍처를 설명하기 위한 것입니다.예를 들어, :numref:`chap_rnn`에서는 소설과 같은 텍스트를 생성하기 위해 언어 모델을 설계하기 위해 RNN에 의존했습니다.:numref:`chap_modern_rnn` 및 :numref:`chap_attention`에서는 기계 번역을 위한 RNN 및 주의 메커니즘을 기반으로 한 모델도 설계했습니다. 

그러나 이 책은 이러한 모든 응용 프로그램을 포괄적으로 다루지는 않습니다.대신, 우리는 자연어 처리 문제를 해결하기 위해 언어의 (심층) 표현 학습을 적용하는 방법*에 중점을 둡니다.사전 학습된 텍스트 표현을 고려할 때, 이 장에서는 두 가지 인기 있고 대표적인 다운스트림 자연어 처리 작업, 즉 감성 분석과 자연어 추론에 대해 살펴봅니다. 이 작업은 텍스트 쌍의 단일 텍스트와 관계를 각각 분석합니다. 

![Pretrained text representations can be fed to various deep learning architectures for different downstream natural language processing applications. This chapter focuses on how to design models for different downstream natural language processing applications.](../img/nlp-map-app.svg)
:label:`fig_nlp-map-app`

:numref:`fig_nlp-map-app`에서 설명했듯이 이 장에서는 MLP, CNN, RNN 및 주의력과 같은 다양한 유형의 딥 러닝 아키텍처를 사용하여 자연어 처리 모델을 설계하는 기본 아이디어를 설명합니다.:numref:`fig_nlp-map-app`에서는 사전 훈련된 텍스트 표현을 두 응용 프로그램의 아키텍처와 결합할 수 있지만 몇 가지 대표적인 조합을 선택합니다.특히 감성 분석을 위해 RNN 및 CNN을 기반으로 널리 사용되는 아키텍처를 살펴볼 것입니다.자연어 추론의 경우 텍스트 쌍을 분석하는 방법을 보여주기 위해 주의와 MLP를 선택합니다.마지막으로 시퀀스 수준 (단일 텍스트 분류 및 텍스트 쌍 분류) 및 토큰 수준 (텍스트 태그 지정 및 질문 응답) 과 같은 광범위한 자연어 처리 응용 프로그램에 대해 사전 훈련된 BERT 모델을 미세 조정하는 방법을 소개합니다.구체적인 경험적 사례로 자연어 추론을 위해 BERT를 미세 조정할 것입니다. 

:numref:`sec_bert`에서 소개한 것처럼 BERT는 광범위한 자연어 처리 응용 프로그램에 대해 최소한의 아키텍처 변경이 필요합니다.그러나 이러한 이점은 다운스트림 애플리케이션을 위해 많은 수의 BERT 파라미터를 미세 조정해야 하는 비용에서 비롯됩니다.공간이나 시간이 제한되면 MLP, CNN, RNN 및 주의를 기반으로 제작된 모델이 더 실현 가능합니다.다음에서는 감정 분석 애플리케이션부터 시작하여 각각 RNN과 CNN을 기반으로 한 모델 설계를 설명합니다.

```toc
:maxdepth: 2

sentiment-analysis-and-dataset
sentiment-analysis-rnn
sentiment-analysis-cnn
natural-language-inference-and-dataset
natural-language-inference-attention
finetuning-bert
natural-language-inference-bert
```
