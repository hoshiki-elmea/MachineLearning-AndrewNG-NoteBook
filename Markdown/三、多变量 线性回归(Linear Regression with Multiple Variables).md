# 四、多变量 线性回归(Linear Regression with Multiple Variables)

<img  src="http://latex.codecogs.com/svg.latex?h_{\theta}(x)=\Theta^TX"  title="http://latex.codecogs.com/svg.latex?h_{\theta}(x)=\Theta^TX"  />

> <img src="http://latex.codecogs.com/svg.latex?h" title="http://latex.codecogs.com/svg.latex?h" />代表**学习算法的解决方案或函数**也称为**假设(hypothesis)**
>
> <img  src="http://latex.codecogs.com/svg.latex?\Theta=\begin{bmatrix}\theta_0\\&space;\theta_1\\&space;\theta_2\\&space;\theta_3\\...\\&space;\theta_n\end{bmatrix}"   title="http://latex.codecogs.com/svg.latex?\Theta=\begin{bmatrix}\theta_0\\  \theta_1\\ \theta_2\\ \theta_3\\...\\ \theta_n\end{bmatrix}" />为该回归模型的**参数(parameters)**
>
> <img  src="http://latex.codecogs.com/svg.latex?X=\begin{bmatrix}x_0\\x_1\\x_2\\x_3\\...\\x_n\end{bmatrix}"   title="http://latex.codecogs.com/svg.latex?X=\begin{bmatrix}x_0\\x_1\\x_2\\x_3\\...\\x_n\end{bmatrix}"  />代表**特征/输入变量**
>
> <img src="http://latex.codecogs.com/svg.latex?X_j^{(i)}" title="http://latex.codecogs.com/svg.latex?X_j^{(i)}" />代表特征矩阵中第i行的第j个特征

> 此时模型中的参数是一个<img src="http://latex.codecogs.com/svg.latex?n&plus;1" title="http://latex.codecogs.com/svg.latex?n+1" />维的向量，任何一个训练实例也都是<img src="http://latex.codecogs.com/svg.latex?n&plus;1" title="http://latex.codecogs.com/svg.latex?n+1" />维的向量，特征矩阵$X$的维度是<img src="http://latex.codecogs.com/svg.latex?m\times(n&plus;1)"  title="http://latex.codecogs.com/svg.latex?m\times(n+1)" />

**（二）代价函数(Cost Function)**

<img  src="http://latex.codecogs.com/svg.latex?J(\theta_j)=\frac{1}{2m}\sum\limits_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^{2}"   title="http://latex.codecogs.com/svg.latex?J(\theta_j)=\frac{1}{2m}\sum\limits_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^{2}"  />

><img src="http://latex.codecogs.com/svg.latex?m" title="http://latex.codecogs.com/svg.latex?m" />代表**训练集中实例的数量**
>
><img  src="http://latex.codecogs.com/svg.latex?({{x}^{(i)}},{{y}^{(i)}})"  title="http://latex.codecogs.com/svg.latex?({{x}^{(i)}},{{y}^{(i)}})"  />代表**第$i$ 个观察实例**
>
><img src="http://latex.codecogs.com/svg.latex?\theta_j=\Theta"  title="http://latex.codecogs.com/svg.latex?\theta_j=\Theta" />为该回归模型的**参数(parameters)**
>
><img  src="http://latex.codecogs.com/svg.latex?(h_{\theta}(x^{(i)})-y^{(i)})^{2}"   title="http://latex.codecogs.com/svg.latex?(h_{\theta}(x^{(i)})-y^{(i)})^{2}"  />求**第<img src="http://latex.codecogs.com/svg.latex?i" title="http://latex.codecogs.com/svg.latex?i" />个观察实例的平方误差**

**（三）梯度下降(Gradient Descent)**

<img  src="http://latex.codecogs.com/svg.latex?\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta_j)"   title="http://latex.codecogs.com/svg.latex?\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta_j)"  />

<img  src="http://latex.codecogs.com/svg.latex?\theta_j:=\theta_j-\alpha\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j"   title="http://latex.codecogs.com/svg.latex?\theta_j:=\theta_j-\alpha\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j"  />

> <img src="http://latex.codecogs.com/svg.latex?\alpha&space;" title="http://latex.codecogs.com/svg.latex?\alpha " />是**学习率(learning rate)**
>
> <img src="http://latex.codecogs.com/svg.latex?x^{(i)},X" title="http://latex.codecogs.com/svg.latex?x^{(i)},X" />中的行数据
>
> <img src="http://latex.codecogs.com/svg.latex?y^{(i)},Y" title="http://latex.codecogs.com/svg.latex?y^{(i)},Y" />中关于<img src="http://latex.codecogs.com/svg.latex?X" title="http://latex.codecogs.com/svg.latex?X" />对应的数据
>
> <img src="http://latex.codecogs.com/svg.latex?\theta_j=\Theta"  title="http://latex.codecogs.com/svg.latex?\theta_j=\Theta" />为该回归模型的**参数(parameters)**

> 将上面的结果应用到梯度下降算法中，就得到了回归的梯度下降算法：
>
> <img  src="http://latex.codecogs.com/svg.latex?\theta_0:=\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x)^{(i)}-y^{(i)})\times x^{(i)}_j"   title="http://latex.codecogs.com/svg.latex?\theta_0:=\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x)^{(i)}-y^{(i)})\times x^{(i)}_j"  />
>
> <img  src="http://latex.codecogs.com/svg.latex?\theta_1:=\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x)^{(i)}-y^{(i)})\times x^{(i)}"   title="http://latex.codecogs.com/svg.latex?\theta_0:=\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x)^{(i)}-y^{(i)})^2(x^{(1)})\times x^{(i)}_1"  />
>
> <img  src="http://latex.codecogs.com/svg.latex?\theta_2:=\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x)^{(i)}-y^{(i)})\times x^{(i)}_2"   title="http://latex.codecogs.com/svg.latex?\theta_0:=\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x)^{(i)}-y^{(i)})^2(x^{(1)})\times x^{(i)}_2"  />
>
> ...