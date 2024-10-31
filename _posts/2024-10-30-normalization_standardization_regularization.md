---
title: Normalization, Standardization, Regularization 정리
date: 2024-10-30 16:30:00 +09:00
categories: [인공지능 관련]
tags:
  [normalization, standardization, regularization]
toc: true
published: true
math: true
---

![alt text](../assets/img/posts/2024-10-30-normalization-stndardization-regularization/Spiderman-Pointing-Meme.jpg)

인공지능을 공부하다보면 3개의 정규화를 만나게 된다.
Normalization, Standardization, Regularization이 그것들이다.
한국어로는 셋 다 정규화여서 혼란을 피하기 위해 영어로 말하게 되는데 셋 다 음절이 길어서 말할때 좀 힘들다. 나만 그런가...?
아무튼 오늘은 이 셋에 대해 알아보도록 하자.

_이 셋 각각이 사용되는 범위가 넓지만 해당 포스트에서는 각자 대표적인 의미를 중심으로 정리하고자 한다._

## Normalization (Min-Max scaling)

> Wikipedia
> 
> Normalization of ratings means adjusting values measured on different scales to a notionally common scale.
{: .prompt-tip }

$$X_{norm} = {X-min(X) \over max(X) - min(X)}$$

**Normalization은 데이터를 공통된 스케일로 조정하여 인공지능 모델이 데이터의 다양한 특성을 일관되게 학습하도록 하여 학습의 안정성과 효율성을 높인다.**

이러한 작업을 수행하는 가장 대표적인 Norm이 Min-Max Scailing이고, 데이터 전처리 과정에서 사용된다. 스케일 된 $X_{norm}$는 [0,1]사이의 값을 갖게된다.

Norm을 하는 이유에 대해 좀 더 풀어서 설명하자면 다음과 같다. 

인공지능 모델은 다양한 출처나 단위로 측정된 데이터를 다뤄야 하므로, 서로 다른 스케일의 데이터를 공통된 스케일로 조정하는 것이 모델학습에 더 효율적이고 혼란을 줄일 있다. 
예시로 Norm을 하지않은 데이터는 과도하게 크거나 작은 값을 가질 수 있으며, 이는 모델 학습 시 가중치 업데이트에 불안정을 야기한다**.

Norm은 여기서 말한 데이터 전처리 뿐만아니라 여러 부분에서도 사용된다. 다른 여러 Norm에 대해서는 다른 포스트에서 정리할 계획이다.



## Standardization

> ChatGPT
> 
> Standardization는 데이터의 평균을 0, 표준편차를 1로 맞추어 데이터의 스케일을 조정하는 방법입니다. 표준화는 데이터의 단위를 통일하고 각 특징이 모델에 균등한 영향을 주도록 도와줍니다. 이 과정을 거친 데이터는 평균이 0, 표준편차가 1인 정규 분포 형태로 변환됩니다.
{: .prompt-tip }

$$ X_{std} = { {X-\mu} \over {\sigma} } $$

~~사실 ChatGPT가 다 말해주었다.~~

**데이터의 각 특징(변수)이 서로 다른 범위를 가지는 경우, 표준화를 통해 균일하게 조정하여 표준정규분포의 속성을 갖도록 재조정하는 것이다.**

Normalization과 같은거 아닌가? 라는 생각이 들어서 정리해 보았다.

| 구분     | Standardization                             | Normalization                        |
| -------- | ------------------------------------------- | ------------------------------------ |
| 목적     | 평균 0, 표준편차 1로 조정하여 스케일을 조정 | 데이터 범위를 조정                   |
| 사용     | 기울기 기반 최적화, 거리 기반 알고리즘      | 신경망 입력, 스케일 차이가 큰 데이터 |
| 결과분포 | 평균 0, 표준편차1인 정규분포를 따름.        | 특정 구간(0~1)에 데이터가 위치       |

**정규화(Normalization)**

 데이터가 정규 분포(Gaussian distribution)를 따르지 않는 경우 적합하며, 데이터의 분포가 불확실할 때 사용. 정규화는 데이터를 [0,1] 또는 [-1,1] 범위로 조정하며, 데이터에 이상치(outliers) 가 있을 경우 그 영향을 크게 받을 수 있음. 알고리즘이 데이터 분포에 대한 가정을 하지 않는 경우에 주로 사용됨.

- 데이터의 분포가 정규분포를 따르지 않거나 특정 값의 범위가 제한된 경우 (예: 이미지 데이터의 픽셀 값이 0에서 255 사이에 있을 때).
- 특히 신경망(Neural Networks)과 같이 입력 값의 범위가 정해져 있을 때 효과적. 입력 값이 0과 1 사이에 위치하면 모델 학습이 안정적으로 진행될 수 있음.
 

**표준화(Standardization)**

데이터가 정규 분포에 가깝거나 분포를 잘 알고 있는 경우 유용. 표준화는 평균을 0, 표준편차를 1로 맞추지만, 값의 범위에 제한이 없으며 이상치에 큰 영향을 받지 않음. 데이터 분포에 대한 가정을 바탕으로 예측을 수행하는 알고리즘에서 주로 사용됨.

- 데이터가 정규분포에 가깝거나 특징의 스케일 차이가 클 때 표준화를 통해 모델이 보다 효율적으로 학습할 수 있음.
- 선형 회귀, 서포트 벡터 머신(SVM), k-평균 클러스터링 등 거리 기반 알고리즘에서 표준화가 더 적합할 수 있음. 이는 각 특징이 동일한 스케일을 가짐으로써 모델이 특정 특징에 치우치지 않도록 하기 때문임.



## Regularization
> Wikipedia
> 
> The goal of regularization is to encourage models to learn the broader patterns within the data rather than memorizing it.
{: .prompt-tip }

Regularizaion은 학습과정에서 가중치가 너무 커지는 것을 억제한다**. **즉, 모델이 학습데이터를 외우는것(overfitting)이 아닌 데이터의 일반적인 패턴들을 학습하도록 하여 일반화 성능을 높이도록 하기 위해서이다.**

$$ 손실함수(L1\,Regularization) = Loss + \lambda \sum_{j=1}^{M}\left\vert w_j \right\vert $$

$$ 손실함수(L2\,Regularization) = Loss + \lambda \sum_{j=1}^{M}w_{j}^{2} $$

Loss에 가중치값에 대한 L1, L2값을 더해주어 가중치값이 너무 커지는 것을 막는다.


## 기타
위의 내용을 정리하며 가슴으로는 이해되지만 머리로는 잘 이해가 안되는 것들을 정리하였다. 

### 과도하게 크거나 작은 값이 학습을 불안정하게 하는 이유?

**Gradient Vanishing**
tanh 활성함수를 생각해보자. 입력이 매우 작거나 큰 경우 기울기가 0에 가까워지고, 역전파 과정에서 chain-rule에 의해 1보다 작은 값이 반복적으로 곱해져 결과적으로 vanishing 문제를 만든다. 


**Gradient Explosion**

$$ MSE=(y_{true} - y)^2 $$

$$ {\delta L \over \delta y} = 2(y_{true}-y) $$

$$ {\delta L \over \delta W} = {\delta L \over \delta y}*{\delta y \over \delta W} $$

신경망이 작은 값들을 주로 학습하여 가중치 w가 이에 맞게 조정된 상황에서, 갑자기 큰 입력값이 들어오면 최종 출력 y도 커질 가능성이 있다. 이로 인해 큰 Loss 값과 큰 기울기가 발생할 수 있다.

큰 입력값 X는 h(wx+b)를 무수히 여러번 하게 된다. 이 과정에서 당연히 입력값 x가 크면 최종 출력 y는 큰 값을 갖게 된다. 따라서 Loss도 큰 값을 만들게 되고,  Loss의 기울기도 보통 커지게 된다. chain rule에 의해 큰값의 Loss 기울기가 곱해지고 결국 기울기 값이 커지게된다.

정리하자면, 
큰 입력값 → 첫 층에서 큰 출력 발생
큰 출력이 각 층을 거쳐가며 증폭 → 최종적으로 큰 Loss 발생
큰 오차가 역전파 과정에서 연쇄적으로 기울기를 증폭시킴

기울기가 매우 크면, 가중치가 너무 큰 폭으로 조정되어 손실 함수가 발산하거나, 지나치게 크게 변동하여 학습이 불안정해질 수 있음.

**OverFitting**
입력값이 크면 모델의 출력값이 크게 변동할 가능성이 높아진다. 이로 인해 모델이 특정한 입력패턴에 대해 지나치게 큰 가중치를 학습할 수 있다. 따라서 모델이 특정 특성에 치우쳐 다른 특성을 충분히 반영하지 못하게 되어 일반화 성능이 저하됨.

### 가중치가 지나치게 커지면 왜 안되나?
가중치가 커진다는 것은 모델이 데이터의 작은 변화나 노이즈에 민감하게 반응하도록 학습된다는 것을 의미하기 때문에, 모델의 일반화 성능에 악영향을 준다.




