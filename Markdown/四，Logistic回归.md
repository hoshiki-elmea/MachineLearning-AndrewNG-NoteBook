# 四，Logistic回归

> 有时遇到一些分类问题，使用线性回归不太好解决

> 如果使用线性的方法来判断分类问题，就会出现图上的问题。我们需要人工的判断中间的分界点，这个很不容易判断；如果在很远的地方有样本点，那么中心点就会发生漂移，影响准确性。

**（一）Sigmoid函数**

> 如果我们想要结果总是在0到1之间，那么就可以使用sigmoid函数，它能保证数据在0-1之间。并且越趋近于无穷大，数据越趋近于1。

<img src="https://raw.githubusercontent.com/hoshiki-elmea/MachineLearning-AndrewNG-NoteBook/main/Images/sigfmoid-graph.jpg?raw=true" style="zoom:25%;" />

<img  src="http://latex.codecogs.com/svg.latex?SigmoidFunction-f(x)=\frac{1}{1&plus;e^{-x}}"   title="http://latex.codecogs.com/svg.latex?SigmoidFunction-f(x)=\frac{1}{1+e^{-x}}"  />

**（二）Logistic回归模型**

<img  src="http://latex.codecogs.com/svg.latex?h_\theta(x)=\frac{1}{1&plus;e^{-{\Theta^TX}}}"   title="http://latex.codecogs.com/svg.latex?h_\theta(x)=\frac{1}{1+e^{-{\Theta^TX}}}"  />

> <img src="http://latex.codecogs.com/svg.latex?h" title="http://latex.codecogs.com/svg.latex?h" />代表**学习算法的解决方案或函数**也称为**假设(hypothesis)**
>
> <img  src="http://latex.codecogs.com/svg.latex?\Theta=\begin{bmatrix}\theta_0\\&space;\theta_1\\&space;\theta_2\\&space;\theta_3\\...\\&space;\theta_n\end{bmatrix}"   title="http://latex.codecogs.com/svg.latex?\Theta=\begin{bmatrix}\theta_0\\  \theta_1\\ \theta_2\\ \theta_3\\...\\ \theta_n\end{bmatrix}" />为该回归模型的**参数(parameters)**
>
> <img  src="http://latex.codecogs.com/svg.latex?X=\begin{bmatrix}x_0\\x_1\\x_2\\x_3\\...\\x_n\end{bmatrix}"   title="http://latex.codecogs.com/svg.latex?X=\begin{bmatrix}x_0\\x_1\\x_2\\x_3\\...\\x_n\end{bmatrix}"  />代表**特征/输入变量**
>
> <img src="http://latex.codecogs.com/svg.latex?X_j^{(i)}" title="http://latex.codecogs.com/svg.latex?X_j^{(i)}" />代表特征矩阵中第i行的第j个特征

**（三）Logistic回归代价函数(对数损失函数)(Cost Function)**

> 使用均方误差，由于最终的值都是0和1，就会产生震荡，此时是无法进行求导的。

> 因此需要寻找一个方法，使得代价函数变成凸函数，从而易于求解。

<img  src="http://latex.codecogs.com/svg.latex?Cost(h_\theta(x),y)=\left\{\begin{matrix}-ln(h_\theta(x))&if~y=1\\-ln(1-h_\theta(x))&if~y=0\end{matrix}\right."   title="http://latex.codecogs.com/svg.latex?Cost(h_\theta(x),y)=\left\{\begin{matrix}-ln(h_\theta(x))&if~y=1\\-ln(1-h_\theta(x))&if~y=0\end{matrix}\right."  />

> <img src="http://latex.codecogs.com/svg.latex?y=1" title="http://latex.codecogs.com/svg.latex?y=1" />时,<img src="http://latex.codecogs.com/svg.latex?Cost(h_\theta(x),y)"  title="http://latex.codecogs.com/svg.latex?Cost(h_\theta(x),y)" />图像

<img src="https://raw.githubusercontent.com/hoshiki-elmea/MachineLearning-AndrewNG-NoteBook/main/Images/Cost_y%3D1-graph.jpg" style="zoom:25%;" />

><img src="http://latex.codecogs.com/svg.latex?y=0" title="http://latex.codecogs.com/svg.latex?y=0" />时,<img src="http://latex.codecogs.com/svg.latex?Cost(h_\theta(x),y)"  title="http://latex.codecogs.com/svg.latex?Cost(h_\theta(x),y)" />图像

<img src="https://raw.githubusercontent.com/hoshiki-elmea/MachineLearning-AndrewNG-NoteBook/main/Images/Cost_y%3D0-graph.jpg" style="zoom:25%;" />

> 如果把损失函数定义为上面的形式，当真实的值是1时，我们预测的值越靠近1，cost的值越小，误差越小。如果真实值是0，那么预测的值越靠近1，cost的值越大。完美的表达了损失的概念。

> 而且，由于0和1的概念，可以把上面的公式合并成下面统一的写法。

<img  src="http://latex.codecogs.com/svg.latex?J(\theta)=-\frac{1}{m}[\sum_{i=1}^{m}y^{i}\ln&space;h_\theta(x^{(i)})&plus;(1-y^{(i)}\ln(1-h_\theta))]"   title="http://latex.codecogs.com/svg.latex?J(\theta)=-\frac{1}{m}[\sum_{i=1}^{m}y^{i}\ln  h_\theta(x^{(i)})+(1-y^{(i)}\ln(1-h_\theta))]" />

><img src="http://latex.codecogs.com/svg.latex?m" title="http://latex.codecogs.com/svg.latex?m" />代表**训练集中实例的数量**
>
><img  src="http://latex.codecogs.com/svg.latex?({{x}^{(i)}},{{y}^{(i)}})"  title="http://latex.codecogs.com/svg.latex?({{x}^{(i)}},{{y}^{(i)}})"  /> 代表**第$i$ 个观察实例**
>
><img src="http://latex.codecogs.com/svg.latex?\theta_j=\Theta"  title="http://latex.codecogs.com/svg.latex?\theta_j=\Theta" />为该回归模型的**参数(parameters)**

**（四）梯度下降(Gradient Descent)**

<img  src="http://latex.codecogs.com/svg.latex?\theta_j:=\theta_j-\alpha\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j"   title="http://latex.codecogs.com/svg.latex?\theta_j:=\theta_j-\alpha\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j"  />

> <img  src="http://latex.codecogs.com/svg.latex?h_\theta(x)=\frac{1}{1&plus;e^{-{\Theta^TX}}}"   title="http://latex.codecogs.com/svg.latex?h_\theta(x)=\frac{1}{1+e^{-{\Theta^TX}}}"  />
>
> <img src="http://latex.codecogs.com/svg.latex?\alpha&space;" title="http://latex.codecogs.com/svg.latex?\alpha " />是**学习率(learning rate)**
>
> <img src="http://latex.codecogs.com/svg.latex?x^{(i)},X" title="http://latex.codecogs.com/svg.latex?x^{(i)},X" />中的行数据
>
> <img src="http://latex.codecogs.com/svg.latex?y^{(i)},Y" title="http://latex.codecogs.com/svg.latex?y^{(i)},Y" />中关于中关于X对应的数据
>
> <img src="http://latex.codecogs.com/svg.latex?\theta_j=\Theta"  title="http://latex.codecogs.com/svg.latex?\theta_j=\Theta" />为该回归模型的**参数(parameters)**

> 在求解最优化的问题时，不仅仅只有一种梯度下降Gradient descenet,还可以使用Conjugate gradient，BFGS，L-BFSGS。

> 多分类问题，可以理解为采用多个logistic分类器，进行分类。针对每个样本点进行一个预测，给出概率值，选择概率值最高的那个进行分类的标识。

<img src="https://raw.githubusercontent.com/hoshiki-elmea/MachineLearning-AndrewNG-NoteBook/main/Images/One_vs_all.jpg"  />

