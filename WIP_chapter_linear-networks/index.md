# 선형 뉴럴 네트워크(Linear Neural Networks)

딥러닝에 대한 자세한 내용으로 들어가기 전에, 뉴럴 네트워크 학습의 기본을 다룰 필요가 있습니다. 이 장에서 우리는 간단한 뉴럴 네트워크 정의, 데이터 다루기, 손실 함수(loss function) 저장하기 그리고 모델 학습시키기를 포함한 전체 학습 과정을 살펴볼 예정입니다. 여러분이 쉽게 이해할 수 있도록 가장 간단한 개념들부터 시작합니다. 다행히, 선형 회귀 및 로지스틱(logistic) 회귀 같은 전통적 통계적인 학습 기법들은 *얕은* 뉴럴 네트워크로 만들어질 수 있습니다. 이 전통적인 알고리즘들부터 시작해서, softmax 회귀 (이 장의 마지막에 소개됨)와 다층 퍼셉트론(multilayer perceptron)(다음 장에서 소개됨)과 같은 더 복잡한 기법들에 기반이되는 기본적인 내용을 설명하겠습니다.

```eval_rst

.. toctree::
   :maxdepth: 2

   linear-regression
   linear-regression-scratch
   linear-regression-gluon
   softmax-regression
   fashion-mnist
   softmax-regression-scratch
   softmax-regression-gluon
```
