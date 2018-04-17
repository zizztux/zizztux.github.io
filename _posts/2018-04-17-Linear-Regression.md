---
layout: post
title: Linear Regression
date: 2018-04-17 18:40:00 +0900
categories:
 - MachineLearning
---

# Linear Regression

Linear Regression은 주어진 training 데이터 셋에서 데이터(Feature)들과 이에 따른 종속변수(Label)를 선형관계(Hypothesis)로 모델링하는
방법입니다. training 데이터 셋에서 데이터와 종속변수의 관계를 잘 반영하도록 선형모델을 결정하면, 그 모델을 통해서 새로운 데이터에
대한 종속변수의 값을 예측할 수 있습니다. Linear Regression에서 사용되는 선형모델은 아래와 같습니다.

![Hypothesis for Linear Regression](/assets/images/2018-04-17-Linear-Regression/linear_regression_1.png)

> Features:  
$$\quad x_1, x_2, \cdots, x_n$$

> Parameters:  
$$\quad \theta_0, \theta_1, \cdots, \theta_n$$

> Hypothesis:  
$$\begin{align}
\quad h_\theta(x) &= \theta_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_nx_n \\
                  &= {\vec \theta}^T{\vec x}
\end{align} \qquad
\Biggl(\vec \theta = \left[\begin{matrix}
  \theta_0 \\
  \theta_1 \\
  \vdots \\
  \theta_n
\end{matrix}\right], \quad
\vec x = \left[\begin{matrix}
  1 \\
  x_1 \\
  \vdots \\
  x_n
\end{matrix}\right]\Biggr)$$

위에서 "데이터(Feature)와 종속변수(Label)의 관계를 잘 반영하도록 선형모델(Hypothesis)을 결정" 한다고 하였는데, 선형모델을 결정하는
것은 1차식의 Parameter 이므로 관계를 잘 반영하는 Parameter 값을 구해야하는 것입니다. 따라서, 모델이 얼마나 데이터들의 경향을 잘
반영하는지 나타내는 비용함수(Cost function)를 도입하고 이를 이용해서 Parameter 값을 결정하게 됩니다. Linear Regression에서는 보통
아래와 같이 MSE(Mean-Squared-Error)를 비용함수로 사용합니다.

> Cost function:  
$$\begin{align}
\quad J(\theta) &= \frac{1}{2m}\sum_{i=1}^{m}\Bigl(h_\theta({\vec x}^{(i)}) - y^{(i)}\Bigr)^2 \\
                &= \frac{1}{2m}\sum_{i=1}^{m}\Bigl({\vec \theta}^T{\vec x}^{(i)} - y^{(i)}\Bigr)^2
\end{align} \qquad
\Biggl(\vec \theta = \left[\begin{matrix}
  \theta_0 \\
  \theta_1 \\
  \vdots \\
  \theta_n
\end{matrix}\right], \quad
\vec x^{(i)} = \left[\begin{matrix}
  1 \\
  x_1^{(i)} \\
  \vdots \\
  x_n^{(i)}
\end{matrix}\right], \quad
\vec y = \left[\begin{matrix}
  y^{(1)} \\
  y^{(2)} \\
  \vdots \\
  y^{(m)} \\
\end{matrix}\right]\Biggr)$$


# 비용함수의 최소화

비용함수를 최소화하는 Parameter 값들을 찾는 것은 결국, 데이터들의 경향을 가장 잘 반영하는 모델(Hypothesis)을 찾는 것이 됩니다.
비용함수를 최소화하는 방법에는 여러가지가 있지만 여기서는 두 가지 방법을 알아보겠습니다.

#### Gradient Descent

임의의 초기 Parameter로 시작해서 비용함수를 줄이는 방향으로 Parameter를 조금씩 조정해나가는 방법입니다. Learning rate($\alpha$)는
적절한 값으로 선택되어야하는데, 너무 큰 값이 사용되면 convergence 하지 않게되고, 너무 작은 값이 사용되면 convergence 하기까지 너무
많은 시간이 걸리게 됩니다. Feature의 개수가 많은 경우에도 적용할 수 있지만 일반적으로 많은 시간이 걸리는 단점이 있습니다. 또한,
적절한 Learning rate($\alpha$)와 초기 Parameter 값을 결정해야 효율적으로 올바른 결과를 얻을 수 있습니다. 비용함수가 convex 형태의
함수여야 global minimum이 존재하고 궁극적으로 올바른 결과값을 얻을 수 있습니다.

> Gradient Descent:  
$$\quad repeat\; until\; convergence\; \{ \\
\begin{align}
\qquad \theta_j &:= \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta) \\
                &:= \theta_j - \alpha\frac{1}{2m}\frac{\partial}{\partial\theta_j}\sum_{i=1}^{m}\Bigl(h_\theta({\vec x}^{(i)}) - y^{(i)}\Bigr)^2 \\
                &:= \theta_j - \alpha\frac{1}{m}\sum_{i=1}^{m}\Bigl({\vec \theta}^T{\vec x}^{(i)} - y^{(i)}\Bigr)x_j^{(i)} \qquad (x_0^{(i)} = 1)
\end{align} \\
\quad \}$$

#### Normal Equation

Gradient Descent와는 달리 해석적으로 행렬식을 풀어 비용함수를 최소화하는 Parameter 값을 구하는 방법입니다. 따라서, 수치 계산이
필요없고 Learning rate($\alpha$)이나 Parameter 초기값을 결정할 필요도 없이 정확한 값을 구할 수 있습니다. 다만, 행렬식을 푸는
과정에서 역행렬을 구해야하므로 역행렬이 존재하지 않는 경우 이 방법을 사용할 수 없습니다. 또한, Feature의 개수가 많아질 경우
현실적으로 해석적 방법을 적용하기 어렵다는 단점도 있습니다.

> Feature vector & Sample matrix:  
$$\begin{align}
\quad \vec x = \left[\begin{matrix}
  1 \\
  x_1 \\
  \vdots \\
  x_n
\end{matrix}\right],
\quad X = \left[\begin{matrix}
  { {\vec x}^{(1)} }^T \\
  { {\vec x}^{(2)} }^T \\
  \vdots \\
  { {\vec x}^{(m)} }^T
\end{matrix}\right]
= \left[\begin{matrix}
  1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\
  1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\
  \vdots & \vdots & \vdots & \ddots & \vdots \\
  1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)}
\end{matrix}\right]
\end{align}$$

> Label vector:  
$$\begin{align}
\quad \vec y = \left[\begin{matrix}
  y^{(1)} \\
  y^{(2)} \\
  \vdots \\
  y^{(m)}
\end{matrix}\right]
\end{align}$$

> Cost function:  
$$\begin{align}
\quad J(\theta) &= \frac{1}{2m}\sum_{i=1}^{m}\Bigl((h_\theta({\vec x}^{(i)}) - y^{(i)}\Bigr)^2 \\
                &= \frac{1}{2m}\sum_{i=1}^{m}\Bigl(({\vec \theta}^T{\vec x}^{(i)} - y^{(i)}\Bigr)^2 \\
                &= \frac{1}{2m}(X{\vec \theta} - {\vec y})^T(X{\vec \theta} - {\vec y}) \\
                &= \frac{1}{2m}({\vec \theta}^TX^T - {\vec y}^T)(X{\vec \theta} - {\vec y}) \\
                &= \frac{1}{2m}({\vec \theta}^TX^TX{\vec \theta} - {\vec \theta}^TX^T{\vec y} - {\vec y}^TX{\vec \theta} +
                    {\vec y}^T{\vec y}) \\
                &= \frac{1}{2m}({\vec \theta}^TX^TX{\vec \theta} - {\vec \theta}^TX^T{\vec y} - (X{\vec \theta})^T{\vec y} +
                    {\vec y}^T{\vec y}) \\
                &= \frac{1}{2m}({\vec \theta}^TX^TX{\vec \theta} - {\vec \theta}^TX^T{\vec y} - {\vec \theta}^TX^T{\vec y} +
                    {\vec y}^T{\vec y}) \\
                &= \frac{1}{2m}({\vec \theta}^TX^TX{\vec \theta} - 2{\vec \theta}^TX^T{\vec y} + {\vec y}^T{\vec y})
\end{align}$$

> Gradient vector:  
$$\begin{align}
\quad \frac{\partial}{\partial\theta}J(\theta) &= \frac{1}{2m}\frac{\partial}{\partial\theta}({\vec \theta}^TX^TX{\vec \theta} -
                                                   2{\vec \theta}^TX^T{\vec y} + {\vec y}^T{\vec y}) \\
                                               &= \frac{1}{2m}(2X^TX{\vec \theta} - 2X^T{\vec y}) \\
                                               &= \frac{1}{m}(X^TX{\vec \theta} - X^T{\vec y})
\end{align}$$

따라서,

> Normal Equation:  
$$\begin{align}
\quad &\frac{\partial}{\partial\theta}J(\theta) = \frac{1}{m}(X^TX{\vec \theta} - X^T{\vec y}) = 0 \\
      &X^TX{\vec \theta} = X^T{\vec y} \\
      &{\vec \theta} = (X^TX)^{-1}X^T{\vec y}
\end{align}$$

위에 마지막 행렬식을 풀면 비용함수를 최소화하는 Parameter 값을 구할 수 있습니다.
