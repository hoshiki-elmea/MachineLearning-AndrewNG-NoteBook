# 二，单变量 线性回归(Linear Regression with One Variable)

**（一）单变量线性回归模型**

<img  src="http://latex.codecogs.com/svg.latex?h_\theta(x)=\theta_{0}&plus;\theta_{1}x"   title="http://latex.codecogs.com/svg.latex?h_\theta(x)=\theta_{0}+\theta_{1}x"  />

> 训练集(Training Set)：用来**估计模型**，验证集用来确定网络结构或者**控制模型复杂程度的参数**

> <img src="http://latex.codecogs.com/svg.latex?m" title="http://latex.codecogs.com/svg.latex?m" />代表**训练集中实例的数量**
>
> <img src="http://latex.codecogs.com/svg.latex?x" title="http://latex.codecogs.com/svg.latex?x" /> 代表**特征/输入变量**
>
> <img src="http://latex.codecogs.com/svg.latex?y" title="http://latex.codecogs.com/svg.latex?y" /> 代表**目标变量/输出变量**
>
> <img src="http://latex.codecogs.com/svg.latex?(x,y)" title="http://latex.codecogs.com/svg.latex?(x,y)" /> 代表**训练集中的实例**
>
> <img  src="http://latex.codecogs.com/svg.latex?({{x}^{(i)}},{{y}^{(i)}})"  title="http://latex.codecogs.com/svg.latex?({{x}^{(i)}},{{y}^{(i)}})"  /> 代表**第$i$ 个观察实例**
>
> <img src="http://latex.codecogs.com/svg.latex?h" title="http://latex.codecogs.com/svg.latex?h" />代表**学习算法的解决方案或函数**也称为**假设(hypothesis)**

**（二）代价函数(Cost Function)**

<img  src="http://latex.codecogs.com/svg.latex?J(\theta_0,\theta_1)=\frac{1}{2m}\sum\limits_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^{2}"   title="http://latex.codecogs.com/svg.latex?J(\theta_0,\theta_1)=\frac{1}{2m}\sum\limits_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^{2}"  />

><img src="http://latex.codecogs.com/svg.latex?m" title="http://latex.codecogs.com/svg.latex?m" />代表**训练集中实例的数量**
>
><img  src="http://latex.codecogs.com/svg.latex?({{x}^{(i)}},{{y}^{(i)}})"  title="http://latex.codecogs.com/svg.latex?({{x}^{(i)}},{{y}^{(i)}})"  /> 代表**第$i$ 个观察实例**
>
><img src="http://latex.codecogs.com/svg.latex?\theta_0,\theta_1"  title="http://latex.codecogs.com/svg.latex?\theta_0,\theta_1" />为该回归模型的**参数(parameters)**
>
><img  src="http://latex.codecogs.com/svg.latex?(h_{\theta}(x^{(i)})-y^{(i)})^{2}"   title="http://latex.codecogs.com/svg.latex?(h_{\theta}(x^{(i)})-y^{(i)})^{2}"  />求**第<img src="http://latex.codecogs.com/svg.latex?i" title="http://latex.codecogs.com/svg.latex?i" />个观察实例的平方误差**

> <img src="http://latex.codecogs.com/svg.latex?J(\theta_0,\theta_1)"  title="http://latex.codecogs.com/svg.latex?J(\theta_0,\theta_1)" />实际求其平方误差的平均值，<img src="http://latex.codecogs.com/svg.latex?\times\frac{1}{2}"  title="http://latex.codecogs.com/svg.latex?\times\frac{1}{2}" />是为了求导

><img src="http://latex.codecogs.com/svg.latex?J(\theta_0,\theta_1)"  title="http://latex.codecogs.com/svg.latex?J(\theta_0,\theta_1)" />越小,模型所预测的值与训练集中实际值之间的差距**建模误差（modeling error）**越接近

> 目标为：<img  src="http://latex.codecogs.com/svg.latex?\min&space;J(\theta_0,\theta_1)"  title="http://latex.codecogs.com/svg.latex?\min J(\theta_0,\theta_1)"  />

**（三）梯度下降(Gradient Descent)**

> **梯度下降(Gradient Descent)**是一个用来**求函数最小值的算法**

> 梯度下降背后的思想是：开始时我们随机选择一个参数的组合<img  src="http://latex.codecogs.com/svg.latex?(\theta_{0},\theta_{1},......,\theta_{n})"   title="http://latex.codecogs.com/svg.latex?(\theta_{0},\theta_{1},......,\theta_{n})"  />，计算代价函数，然后我们寻找下一个能让代价函数值下降最多的参数组合。
>
> 我们持续这么做直到找到一个**局部最小值(local minimum)**，因为我们并没有尝试完所有的参数组合，所以不能确定我们得到的局部最小值是否便是**全局最小值(global minimum)**，选择不同的初始参数组合，可能会找到不同的局部最小值。

> **批量梯度下降(batch gradient descent)**

<img  src="http://latex.codecogs.com/svg.latex?{\theta_{j}}:={\theta_{j}}-\alpha&space;\frac{\partial&space;}{\partial&space;{\theta_{j}}}J\left(\theta&space;\right)"   title="http://latex.codecogs.com/svg.latex?{\theta_{j}}:={\theta_{j}}-\alpha  \frac{\partial }{\partial {\theta_{j}}}J\left(\theta \right)" />

> 其中<img src="http://latex.codecogs.com/svg.latex?\alpha&space;" title="http://latex.codecogs.com/svg.latex?\alpha " />是**学习率(learning rate)**，它决定了我们沿着能让代价函数下降程度最大的方向向下迈出的步子有多大,在批量梯度下降中，我们每一次都同时让所有的参数减去学习速率乘以代价函数的导数。

| <img src="http://latex.codecogs.com/svg.latex?\alpha&space;" title="http://latex.codecogs.com/svg.latex?\alpha " />太小 | 移动步伐小，移动次数多，梯度下降慢 |
| ------------------------------------------------------------ | ---------------------------------- |
| <img src="http://latex.codecogs.com/svg.latex?\alpha&space;" title="http://latex.codecogs.com/svg.latex?\alpha " />太大 | 移动步伐大，可能无法收敛，甚至发散 |

> 如果你要更新这个等式，你需要同时更新全体<img src="http://latex.codecogs.com/svg.latex?\theta" title="http://latex.codecogs.com/svg.latex?\theta" />值

> **如果你的参数已经处于局部最低点，那么梯度下降法更新其实什么都没做，它不会改变参数的值。这也解释了为什么即使学习速率<img src="http://latex.codecogs.com/svg.latex?\alpha&space;" title="http://latex.codecogs.com/svg.latex?\alpha " />保持不变时，梯度下降也可以收敛到局部最低点。**

> 探讨梯度下降算法的导数项：
>
> <img  src="http://latex.codecogs.com/svg.latex?\frac{\partial&space;}{\partial\theta_j}J(\theta_0,\theta_1)=\frac{\partial&space;}{\partial\theta_j}\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2\\=\frac{\partial}{\partial\theta_j}\frac{1}{2m}\sum_{i=1}^{m}(\theta_0&plus;\theta_1x^{(i)}-y^{(i)})^2"  title="http://latex.codecogs.com/svg.latex?\frac{\partial  }{\partial\theta_j}J(\theta_0,\theta_1)=\frac{\partial  }{\partial\theta_j}\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2\\=\frac{\partial}{\partial\theta_j}\frac{1}{2m}\sum_{i=1}^{m}(\theta_0+\theta_1x^{(i)}-y^{(i)})^2"  />
>
> <img  src="http://latex.codecogs.com/svg.latex?If~~~\theta_j=\theta_0,j=0:\frac{\partial}{\partial&space;\theta_0}J(\theta_0,\theta_1)=\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2"   title="http://latex.codecogs.com/svg.latex?If~~~\theta_j=\theta_0,j=0:\frac{\partial}{\partial   \theta_0}J(\theta_0,\theta_1)=\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2"  />
>
> <img  src="http://latex.codecogs.com/svg.latex?If~~~\theta_j=\theta_1,j=1:\frac{\partial}{\partial&space;\theta_1}J(\theta_0,\theta_1)=\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2\times x^{(i)}"   title="http://latex.codecogs.com/svg.latex?If~~~\theta_j=\theta_1,j=1:\frac{\partial}{\partial   \theta_1}J(\theta_0,\theta_1)=\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2\times x^{(i)}"  />

> 将上面的结果应用到梯度下降算法中，就得到了回归的梯度下降算法：
>
> <img  src="http://latex.codecogs.com/svg.latex?\theta_0:=\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x)^{(i)}-y^{(i)})^2(x^{(i)}_0)"   title="http://latex.codecogs.com/svg.latex?\theta_0:=\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x)^{(i)}-y^{(i)})^2(x^{(0)})"  />
>
> <img  src="http://latex.codecogs.com/svg.latex?\theta_1:=\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x)^{(i)}-y^{(i)})^2\times x^{(i)}"   title="http://latex.codecogs.com/svg.latex?\theta_0:=\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x)^{(i)}-y^{(i)})^2(x^{(1)})^2\times x^{(i)}"  />

