# 다층 퍼셉트론(Multilayer Perceptron)

이 장에서 우리는 여러분의 첫번째로 진정한 *딥* 네트워크를 소개합니다. 가장 간단한 딥 네트워크는 다층 퍼셉트론(multilayer perceptron)이라고 불리는 것으로, 뉴론들로 이뤄진 여러 층(layer)들로 구성됩니다. 각 층의 뉴런들은 아래 층(입력으로 사용되는)과 위 층(영향을 받는)들과 완전히 연결됩니다. 고용량의 모델을 학습시키는 경우, 오버피팅(overfitting)에 대한 위험을 겪을 수 있습니다. 따라서, 오버피팅(overfitting), 언더피팅(underfitting) 그리고 용량 제어의 개념에 대한 엄밀한 소개할 필요가 있습니다. 이 문제들을 해결하기 것을 돕기 위해서 드롭아웃(dropout), 가중치 감쇠(weight decay) 같은 정칙화(regularization) 기법을 소개합니다. 딥러닝을 성공적으로 학습시키는데 중요한 역할을 하는 수치 안정성과 파라미터 초기화에 관련된 이슈들도 논의하겠습니다. 처음부터 끝까지 우리는 실제 데이터에 모델을 적용하는 것에 촛점을 마춰면서, 여러분이 개념을 이해하는 것뿐만 아니라 딥 네트워크의 실제 활용을 확실하게 이해할 수 있도록 하는 것이 우리의 목표입니다. 연산 성능, 확장성, 모델의 효율성은 다음 장들에서 다룹니다.

```eval_rst

.. toctree::
   :maxdepth: 2

   mlp
   mlp-scratch
   mlp-gluon
   underfit-overfit
   weight-decay
   dropout
   backprop
   numerical-stability-and-init
   environment
   kaggle-house-price
```
