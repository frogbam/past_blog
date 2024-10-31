---
title: Normalization, Standardization, Regularization 코드 구현
date: 2024-10-31 14:30:00 +09:00
categories: [인공지능 구현]
tags:
  [normalization, standardization, regularization, 코드, 구현]
toc: true
published: true
math: true
---

오늘은 저번에 알아본 [Normalization, Standardization, Regularization](https://frogbam.github.io/posts/normalization-standardization-regularization/) 각각의 코드 구현에 대해 작성하고자 한다.

## Normalization

```python
import torch

data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
normalized_data = (data-data.min()) / (data.max() - data.min())

print("Normalized Data:")
print(normalized_data)
```

$$X_{norm} = {X-min(X) \over max(X) - min(X)}$$

기본적인 텐서에 대한 min-max normalization 코드 이다. 수식을 그대로 torch로 구현하면 된다.


```python
import torch
import torchvision.transforms as transforms
from PIL import Image

# 이미지 불러오기
image_path = "test.jpg"
image = Image.open(image_path) # Shape: [C, H, W]

# Transform 정의
transform = transforms.Compose([
    transforms.ToTensor(), # [0, 255] -> [0, 1] 범위의 텐서로 변환
])

# 이미지를 [0,1] 범위로 정규화 된 텐서로 변환
image_tensor = transform(image)
```

실제 이미지를 가지고 학습하는 모델 코드를 보면 위와 같이 전처리 단계에서 이미지를 텐서화 하고 transforms하는 과정에서 보통 normalization을 해주게 된다. 

> torchvision의 transforms.ToTensor()를 이용해 이미지를 텐서로 변환하면서 자동으로 [0, 1] 범위로 정규화 된 텐서를 반환해준다.
{: .prompt-tip }

## Standardization

```python
# 예제 텐서 생성
data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 텐서의 평균과 표준편차 계산
mean = data.mean()
std = data.std()

# 표준화 적용
standardized_data = (data - mean) / std

print("Standardized Tensor:", standardized_data)
print("Mean after standardization:", standardized_data.mean().item())
print("Standard deviation after standardization:", standardized_data.std().item())
```

$$ X_{std} = { {X-\mu} \over {\sigma} } $$

기본적인 텐서에 대한 standardization 코드이다. 역시 수식을 그대로 torch로 구현하면 된다.

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

# 이미지 불러오기
image_path = "test.jpg"
image = Image.open(image_path) # Shape: [C, H, W]

# Transform 정의
transform = transforms.Compose([
    transforms.ToTensor(),  # [0, 255] -> [0, 1] 범위의 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 표준화
])

# 이미지를 표준화 된 텐서로 변환
image_tensor = transform(image)
```

마찬가지로 torchvision의 transforms.Normalize를 이용해 RGB 이미지의 표준화를 간단하게 구현할 수 있다.

> 코드에서 사용된 mean값과 std값은 ImageNet 데이터셋에서 계산된 값이며 통상적으로 많이 사용된다.
{: .prompt-tip }

## Regularization

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.linear = nn.Linear(10,1)
  
  def forward(self, x):
    return self.linear(x)

inputs = torch.randn(64,10)
targets = torch.randn(64, 1)

model = Model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

l2_lambda = 0.001

for epoch in range(100):
  optimizer.zero_grad()
  outputs = model(inputs)
  loss = creterion(outputs, targets)

  # L2 Regularization 적용
  l2_reg = 0
  for param in model.parameters():
    l2_reg += torch.sum(param*param)
  loss = loss + l2_lambda * l2_reg 

  loss.backward()
  optimizer.step()
```

$$ 손실함수(L2\ Regularization) = Loss + \lambda \sum_{j=1}^{M}w_{j}^{2} $$

torch에서 모델을 학습할때 L2 Regularization을 구현한 코드이다. 이것도 그대로 수식을 코드로 구현하면 된다.

```python
# 앞은 동일

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

for epoch in range(100):
  optimizer.zero_grad()
  outputs = model(inputs)
  loss = creterion(outputs, targets)
  loss.backward()
  optimizer.step()

```

사실 L2 Regularization은 보통 위의 코드와 같이 구현한다. optimizer(여기서는 SGD)의 weight_decay값으로 lambda값만 넣어주면 해당 labmda값으로 L2 Regularization이 적용된다.

> 왜 Weight Decay라고 했을까?
> 
> Regularization은 보편적인 용어로, L1, L2 정규화뿐만 아니라 드롭아웃이나 조기 종료 등 다양한 형태의 규제 기법을 포함합니다.
> 
> Weight Decay는 L2 정규화를 적용하는 구체적인 방법 중 하나로, 주로 SGD와 같은 옵티마이저에 적용되는 방법을 뜻합니다.
> 즉, weight_decay는 L2 정규화의 구현 방식에 중점을 두고 사용된 용어로, 더 큰 범주인 "정규화 (regularization)"의 한 부분이라고 볼 수 있습니다.
>
> by ChatGPT
{: .prompt-tip }