# 딥러닝 기초

이 장에서는 딥러닝의 기본적인 내용들을 소개합니다. 네트워크 아키텍처, 데이터, 손실 함수(loss functino), 최적화, 그리고 용량 제어를 포함합니다. 이해를 돕기 위해서, 선형 함수, 선형 회귀, 그리고 확률적 경사 하강법(stochastic gradient descent)과 같은 간단한 개념부터 시작합니다. 이것들은 softmax나 다층 퍼셉트론(multilayer perceptron)와 같은 보다 복잡한 개념의 기초가 됩니다. 우리는 이미 상당히 강력한 네트워크를 디자인할 수 있지만, 필수적인 제어나 기교는 배우지 않았습니다. 이를 위해서, 용량 제어, 오버피팅(overfitting)과 언더피팅(underfitting)에 대한 개념을 이해할 필요가 있습니다. 드롭아웃(dropout), 수치 안정화(numerical stability), 그리고 초기화에 대한 설명으로 이 장을 마무리할 예정입니다. 우리는 실제 데이터에 모델을 적용하는 방법에 집중하겠습니다. 이를 통해서 여러분은 기본 개념 뿐만 아니라 딥 네트워크를 실제 문제에 적용할 수 있도록 할 예정입니다. 성능, 확장성 그리고 효율성은 다음 장들에서 다룹니다.

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
