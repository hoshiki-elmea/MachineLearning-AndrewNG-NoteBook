# 五，正则化(Regularization)

当将这些未修正的算法应用到某些特定的机器学习应用时，会遇到**过拟合(over-fitting)**的问题，可能会导致它们效果很差。

![过拟合](https://raw.githubusercontent.com/hoshiki-elmea/MachineLearning-AndrewNG-NoteBook/main/Images/overfitting.jpg)

> 第一个模型是一个线性模型，欠拟合，不能很好地适应我们的训练集
>
> 我们将其称为**欠拟合(underfitting)**，或者叫作叫做**高偏差(bias)**。
>
> 第二张图，我们在中间加入一个二次项，也就是说对于这幅数据我们用二次函数去拟合。自然，可以拟合出一条曲线，事实也证明这个拟合效果很好。
>
> 第三个模型是一个四次方的模型，过于强调拟合原始数据，而丢失了算法的本质：预测新数据。
>
> 我们将其称为**过拟合(overfitting)**，也叫**高方差(variance)**。

>从第一印象上来说，如果我们拟合一个高阶多项式，那么这个函数能很好的拟合训练集（能拟合几乎所有的训练数据），但这也就面临函数可能太过庞大的问题，变量太多。
>
>**同时如果我们没有足够的数据集（训练集）去约束这个变量过多的模型，那么就会发生过拟合。**

为了度量拟合表现，引入：

- 偏差(bias)

  指模型的预测值与真实值的**偏离程度**。偏差越大，预测值偏离真实值越厉害。偏差低意味着能较好地反应训练集中的数据情况。

- 方差(Variance)

  指模型预测值的**离散程度或者变化范围**。方差越大，数据的分布越分散，函数波动越大，泛化能力越差。方差低意味着拟合曲线的稳定性高，波动小。

> 高偏差意味着欠拟合，高方差意味着过拟合。
>
> 我们应尽量使得拟合模型处于低方差（较好地拟合数据）状态且同时处于低偏差（较好地预测新值）的状态。

避免过拟合的方法有：

- 减少特征的数量
  - 手动选取需保留的特征
  - 使用模型选择算法来选取合适的特征(如 PCA 算法)
  - 减少特征的方式易丢失有用的特征信息
- 正则化(Regularization)
  - 可保留所有参数（许多有用的特征都能轻微影响结果）
  - 减少/惩罚各参数大小(magnitude)，以减轻各参数对模型的影响程度
  - 当有很多参数对于模型只有轻微影响时，正则化方法的表现很好

**（一）正则化(regurlarization)**

1,代价函数(CostFunction)

> 让我们考虑下面的假设，我们想要加上**惩罚项**，从而使参数<img src="http://latex.codecogs.com/svg.latex?\theta_3,\theta_4" title="http://latex.codecogs.com/svg.latex?\theta_3,theta_4" /> 足够的小。
>
> <img src="http://latex.codecogs.com/svg.latex?Suppose~we~penalize~and~make~\theta_3,\theta_4~really~small" title="http://latex.codecogs.com/svg.latex?Suppose~we~penalize~and~make~\theta_3,\theta_4~really~small" />
>
> <img  src="http://latex.codecogs.com/svg.latex?\min_{\theta}\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2"   title="http://latex.codecogs.com/svg.latex?\min_{\theta}\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2"  />
>
> <img src="http://latex.codecogs.com/svg.latex?\min_{\theta}\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2+1000~\theta_3^2+1000~\theta_4^2" title="http://latex.codecogs.com/svg.latex?\min_{\theta}\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2+1000~\theta_3^2+1000~\theta_4^2" />
>
> **1000 只是我随便写的某个较大的数字而已。**
>
> 如果我们要最小化这个函数，那么为了最小化这个新的代价函数，我们要让<img src="http://latex.codecogs.com/svg.latex?\theta_3,\theta_4" title="http://latex.codecogs.com/svg.latex?\theta_3,theta_4" />尽可能小。
>
> 因为，如果你**在原有代价函数的基础上加上 1000 乘以<img src="http://latex.codecogs.com/svg.latex?\theta_3" title="http://latex.codecogs.com/svg.latex?\theta_3" /> 这一项 ，那么这个新的代价函数将变得很大**
>
> 所以，**当我们最小化这个新的代价函数时， 我们将使 <img src="http://latex.codecogs.com/svg.latex?\theta_3" title="http://latex.codecogs.com/svg.latex?\theta_3" />的值接近于 0，同样<img src="http://latex.codecogs.com/svg.latex?\theta_4" title="http://latex.codecogs.com/svg.latex?\theta_4" /> 的值也接近于 0，就像我们忽略了这两个值一样。如果我们做到这一点（<img src="http://latex.codecogs.com/svg.latex?\theta_3" title="http://latex.codecogs.com/svg.latex?\theta_3" /> 和 <img src="http://latex.codecogs.com/svg.latex?\theta_4" title="http://latex.codecogs.com/svg.latex?\theta_4" />接近 0 ），那么我们将得到一个近似的二次函数。**

> 通过上述思路，我们可以得到这样一个代价函数

<img src="http://latex.codecogs.com/svg.latex?J(\theta)=\frac{1}{2m}[\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{j=1}^{n}\theta_j^2]" title="http://latex.codecogs.com/svg.latex?J(\theta)=\frac{1}{2m}[\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{j=1}^{n}\theta_j^2]" />

> 其中<img src="http://latex.codecogs.com/svg.latex?\lambda" title="http://latex.codecogs.com/svg.latex?\lambda" />又称为**正则化参数（Regularization Parameter）。** 
>
> <img src="http://latex.codecogs.com/svg.latex?\sum\limits_{j=1}^{n}" title="http://latex.codecogs.com/svg.latex?\sum\limits_{j=1}^{n}" />: 不惩罚基础参数 <img src="http://latex.codecogs.com/svg.latex?\theta_0" title="http://latex.codecogs.com/svg.latex?\theta_0" />
>
> <img src="http://latex.codecogs.com/svg.latex?\lambda\sum_{j=1}^{n}\theta_j^2" title="http://latex.codecogs.com/svg.latex?\lambda\sum_{j=1}^{n}\theta_j^2" />为正则化项。

<img src="http://latex.codecogs.com/svg.latex?\lambda" title="http://latex.codecogs.com/svg.latex?\lambda" /> 正则化参数类似于学习速率，也需要我们自行对其选择一个合适的值。

- 过大
  - 导致模型欠拟合(假设可能会变成近乎<img src="http://latex.codecogs.com/svg.latex?x = \theta_0" title="http://latex.codecogs.com/svg.latex?x = \theta_0" />的直线 )
  - 无法正常去过拟问题
  - 梯度下降可能无法收敛
- 过小
  - 无法避免过拟合（等于没有）

> 正则化符合奥卡姆剃刀(Occam's  razor)原理。在所有可能选择的模型中，能够很好地解释已知数据并且十分简单才是最好的模型，也就是应该选择的模型。从贝叶斯估计的角度来看，正则化项对应于模型的先验概率。可以假设复杂的模型有较大的先验概率，简单的模型有较小的先验概率。
>
> 正则化是结构风险最小化策略的实现，是去过拟合问题的典型方法，虽然看起来多了个一参数多了一重麻烦，后文会介绍自动选取正则化参数的方法。模型越复杂，正则化参数值就越大。比如，正则化项可以是模型参数向量的范数。

2,正则化线性回归(岭回归)(Regularization Linear Regression)

<img src="http://latex.codecogs.com/svg.latex?Regularization~Linear~Regression~CostFunction" title="http://latex.codecogs.com/svg.latex?Regularization~Linear~Regression~CostFunction" />

<img src="http://latex.codecogs.com/svg.latex?J(\theta)=\frac{1}{2m}[\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{j=1}^{n}\theta_j^2]" title="http://latex.codecogs.com/svg.latex?J(\theta)=\frac{1}{2m}[\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{j=1}^{n}\theta_j^2]" />

> 梯度下降
>
> <img src="http://latex.codecogs.com/svg.latex?\begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \  \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m  (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j  := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m  (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) +  \frac{\lambda}{m}\theta_j \right], \ \ \ j \in \lbrace  1,2...n\rbrace\newline & \rbrace \end{align*}" title="http://latex.codecogs.com/svg.latex?\begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \  \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m  (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j  := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m  (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) +  \frac{\lambda}{m}\theta_j \right], \ \ \ j \in \lbrace  1,2...n\rbrace\newline & \rbrace \end{align*}" />
>
> 也可以移项得到更新表达式的另一种表示形式
>
> <img src="http://latex.codecogs.com/svg.latex?\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}" title="http://latex.codecogs.com/svg.latex?\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}" />

> 正规方程
>
> <img src="http://latex.codecogs.com/svg.latex?X=\begin{bmatrix}(x^{(1)})^T\\.\\.\\.\\(x^{(m)})^T\end{bmatrix}~~~~~y=\begin{bmatrix}y^{(1)}\\.\\.\\.\\y^{(m)}\end{bmatrix}
> \\\min_\theta J(\theta)
> \frac{\partial}{\partial\theta_j}J(\theta)\overset{set}{=}0
> \\\Rightarrow\Theta=(X^TX-\lambda(L^{(n+1)}_{(n+1)}))
> \\L^{(n+1)}_{(n+1)}=\begin{bmatrix}0&  &  &  &  \\& 1&  &  &  \\&  & 1&  &  \\&  &  & 1&  \\&  &  &  & ...\end{bmatrix}" title="http://latex.codecogs.com/svg.latex?X=\begin{bmatrix}(x^{(1)})^T\\.\\.\\.\\(x^{(m)})^T\end{bmatrix}~~~~~y=\begin{bmatrix}y^{(1)}\\.\\.\\.\\y^{(m)}\end{bmatrix}
> \\\min_\theta J(\theta)
> \frac{\partial}{\partial\theta_j}J(\theta)\overset{set}{=}0
> \\\Rightarrow\Theta=(X^TX-\lambda(L^{(n+1)}_{(n+1)}))
> \\L^{(n+1)}_{(n+1)}=\begin{bmatrix}0&  &  &  &  \\& 1&  &  &  \\&  & 1&  &  \\&  &  & 1&  \\&  &  &  & ...\end{bmatrix}" />
>
> 
>
> <img src="http://latex.codecogs.com/svg.latex?eg~~~n=2~~~~~L=\begin{bmatrix} 0&  0&  0\\ 0&  1&  0\\ 0&  0&  1\\ \end{bmatrix}" title="http://latex.codecogs.com/svg.latex?eg~~~n=2~~~~~L=\begin{bmatrix} 0&  0&  0\\ 0&  1&  0\\ 0&  0&  1\\ \end{bmatrix}" />
>
> 前文提到正则化可以解决正规方程法中不可逆的问题，即增加了 <img src="http://latex.codecogs.com/svg.latex?\lambda \cdot L" title="http://latex.codecogs.com/svg.latex?\lambda \cdot L" />正则化项后，可以保证<img src="http://latex.codecogs.com/svg.latex?X^TX + \lambda \cdot L" title="http://latex.codecogs.com/svg.latex?X^TX + \lambda \cdot L" />可逆(invertible)，即便 <img src="http://latex.codecogs.com/svg.latex?X^TX" title="http://latex.codecogs.com/svg.latex?X^TX" /> 不可逆(non-invertible)。

3,逻辑回归正则化(Regularized Logistic Regression)

<img src="http://latex.codecogs.com/svg.latex?Regularization~Logistic~Regression~CostFunction" title="http://latex.codecogs.com/svg.latex?Regularization~Linear~Regression~CostFunction" />

<img src="http://latex.codecogs.com/svg.latex?J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2" title="http://latex.codecogs.com/svg.latex?J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2" />



> 梯度下降
>
> <img src="http://latex.codecogs.com/svg.latex?\begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \  \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m  (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j  := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m  (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) +  \frac{\lambda}{m}\theta_j \right], \ \ \ j \in \lbrace  1,2...n\rbrace\newline & \rbrace \end{align*}" title="http://latex.codecogs.com/svg.latex?\begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \  \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m  (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j  := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m  (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) +  \frac{\lambda}{m}\theta_j \right], \ \ \ j \in \lbrace  1,2...n\rbrace\newline & \rbrace \end{align*}" />

