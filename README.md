# Data Science Algorithms
* Topics:
- [Data Science Algorithms](#data-science-algorithms)
	- [EDA](#eda)
	- [Supervised ML](#supervised-ml)
		- [**Linear Regression**](#linear-regression)
		- [**Ridge And Lasso Regression**](#ridge-and-lasso-regression)
			- [**Addressing Overfitting**:](#addressing-overfitting)
			- [**Lasso Regression (L1 Regularization)**:](#lasso-regression-l1-regularization)
			- [**Ridge Regression (L2 Regularization):**](#ridge-regression-l2-regularization)
			- [**Assumptions of Linear Regression**](#assumptions-of-linear-regression)
		- [Logistic Regression (Classification):](#logistic-regression-classification)

## **EDA**
  1. [Zomato Data EDA](EDA%20Zomato%20Dataset.ipynb)
  2. [Black Friday Dataset EDA and Feature Engineering](EDA%20and%20Feature%20Engineering%20-%20Black%20Friday%20Dataset.ipynb)
  3. [Flight Price Prediction EDA and Feature Engineering](EDA%20and%20Feature%20Engineering%20for%20Flight%20Price%20Prediction.ipynb)
   
 ****
## **Supervised ML**
### **Linear Regression**
   - Cost Function - Mean Squared Error

		![Gradient Descent](https://github.com/desai-nitin/DataScience/blob/main/readme_images/mse1.png)

		![Gradient Descent](https://github.com/desai-nitin/DataScience/blob/main/readme_images/mse2.png)

   - Gradient Descent Formula
		
		![Gradient Descent](https://github.com/desai-nitin/DataScience/blob/main/readme_images/gradient_descent.png)
	
   - Linear Regression Convergence Formula
		
		![Linear Regression Convergence](https://github.com/desai-nitin/DataScience/blob/main/readme_images/LRConvergence.png)	

   - Learning Rate:
		- Learning Rate is step size for each iteration while finding global minimum of the loss function.

			![Learning Rate](https://github.com/desai-nitin/DataScience/blob/main/readme_images/LearningRate1.png)

			![Learning Rate View](https://github.com/desai-nitin/DataScience/blob/main/readme_images/LearningRateView.png)
 ****
### **Ridge And Lasso Regression**

Problem In Linear Regression: 

| Overfitting     | Underfitting    | Generalized model |
| --------------- |:---------------:| -----------------:|
| Train Acc = 90% | Train Acc = 60% | Train Acc = 92%   |
| Test Acc = 80%  | Test Acc = 58% | Test Acc = 91%   |
| Low Bias        | High Bias       | Low Bias          |
| High Variance   | High Variance   | Low Variance      |


#### **Addressing Overfitting**:
  1. Reduce number of features
        - Manually select which features to be used
        - Feature selection
  2. Regularization
        - Keep all features but reduce magnitude of parameters theta 
        - Works well when we have a lot of features, each of which contributes a bit to predicting 
  
#### **Lasso Regression (L1 Regularization)**:
$${1\over2m} \sum_{i=1}^m (h_\theta (x)^i -y^i)^2 + \lambda \sum_{j=1}^n |\theta_j|$$
   
 - It adds "*aboslute value of magnitude*" of coefficents as penalty to loss function
  - Lasso shrinks the less important feature's coefficient to near zero thus, removing some feature.
  - Works as **feature selection** in case we have large number of features


#### **Ridge Regression (L2 Regularization):**
 $${1\over2m} \sum_{i=1}^m (h_\theta (x)^i -y^i)^2 + \lambda \sum_{j=1}^n \theta_j^2$$
  - It adds "*squared magnitute*" of coefficents as penalty to loss function
  - If lambda is very large then it will add too much weight and can lead to underfitting. Hence it's important how lambda is chosen.
  - This technique works very well to avoid **over-fitting**.

#### **Assumptions of Linear Regression**
  1. *Features follows Normal (Gaussian Distribution)*:
    		- As model will get trained well with normal distribution data
    		- If any feature does not follows it we should try to use feature transformation.
  2. *Standard Scale*:
    		- Scaling all features on same scale
    		- Usually we use Zscore for Scaling mu=0 and sigma=1
  3. *Linearity*:
    		- Data with Linear relation between Independent and Dependent variables  will give good results.
    		- If relation is not linear we can use non-linear model.
  4. *Multi Collinearity*:
    		- No feature with correlation with other feature (Independent Variable).
    		- Use Variance Inflation Factor (VIF) to select only important and non-multicolinear features.
  5. *Homoscedasticity (Same Variance)*:
    		- It means that the error is constant along the values of the dependent variable.
 ***
### Logistic Regression (Classification):

  - Why not Linear regression for classification?
  - For dynamic threshold we should use Regression models and for fixed  thresholds we should use classification models.
  - Regression values can go beyond (0,1) but in classification problems we focus on binary output y ∈ {0,1}.

#### Logistic Function / Sigmoid Function:
  - From Linear Regression we knew that
  	$$h_\theta(x) = \theta_0+\theta_1x$$
	Let, $z=\theta_0+\theta_1x$  
	$$h_\theta(x)=g(z)$$ 
	where $g(z)$ can be given as 
	$$h_\theta(x)={1\over1+e^{-z}}$$
	$$h_\theta(x) = g(\theta^Tx) = {1\over1+e^{-(\theta_0+\theta_1x)}}$$   
  	

	Here is plot showing $g(z)$

<p align="center">
	<img alt=Sigmoid src=https://github.com/desai-nitin/DataScience/blob/main/readme_images/Sigmoid.PNG>
</p>

  - Logistic Regression Cost Function:
	$$J(\theta)={1\over m}\sum_{i=1}^m cost(h_\theta(x^{(i)}),y^{(i)})$$
		
	$cost(h_\theta(x^{(i)}),y^{(i)}) = -y log(h_\theta(x^{(i)})) - (1-y) log(1-h_\theta(x^{(i)}))$

	![LogCost](https://github.com/desai-nitin/DataScience/blob/main/readme_images/LogCost.png)

	When $y=1$, $cost(h_\theta(x),1) = -log(h_\theta(x))$ 

	When $y=0$,	$cost(h_\theta(x),0) = -log(1-h_\theta(x))$ 

<p align="center">
	<img alt=y1 src=https://github.com/desai-nitin/DataScience/blob/main/readme_images/cost_y%3D1.PNG>
	<img alt=y0 src=https://github.com/desai-nitin/DataScience/blob/main/readme_images/cost_y%3D0.PNG>
</p>

