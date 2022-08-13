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
  1. [Zomato Data EDA](https://github.com/desai-nitin/DataScience/blob/main/EDA_Zomato_Dataset.ipynb)
  2. [Black Friday Dataset EDA and Feature Engineering](https://github.com/desai-nitin/DataScience/blob/main/EDA_And_Feature_Engineering_Black_Friday_Dataset.ipynb)
  3. [Flight Price Prediction EDA and Feature Engineering](https://github.com/desai-nitin/DataScience/blob/main/EDA_And_Feature_Engineering_For_Flight_Price_Prediction.ipynb)
   
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


	$$cost(h_\theta(x^{(i)}),y^{(i)}) = -y log(h_\theta(x^{(i)})) - (1-y) log(1-h_\theta(x^{(i)}))$$

<p align="center">
	<img alt=logcost style="width:400px;" src=https://github.com/desai-nitin/DataScience/blob/main/readme_images/LogCost.png >
</p>

<p align="center">
	<img alt=y1 style="width:400px;" src=https://github.com/desai-nitin/DataScience/blob/main/readme_images/cost_y%3D1.png>
	<img alt=y0 style="width:400px;" src=https://github.com/desai-nitin/DataScience/blob/main/readme_images/cost_y%3D0.png>
</p>

$$J(\theta_1)=-{1\over 2m}\sum_{i=1}^m (y^i log(h_\theta(x^{(i)})) + (1-y^i) log(1-h_\theta(x^{(i)})))$$

Repeat until Convergence:

{
$$\theta_j := \theta_j - \alpha {\delta(J(\theta))\over\delta\theta_j}
$$
}

#### Performance Matrix (Binary Classification):
<p align="center">
	<img alt=confusion_matrix src=https://github.com/desai-nitin/DataScience/blob/main/readme_images/confusion_matrix.png>
</p>

$$Precision = {True\;Positive\;(TP)\over{True\;Positive\;(TP) + False\;Positive\;(FP)}} = {True\;Positive\;(TP)\over{Total\;Positive\; Predictions}}$$

$$Recall = {True\;Positive\;(TP)\over{True\;Positive\;(TP) + False\;Negative\;(FN)}} = {True\;Positive\;(TP)\over{Total\;Actual\; Positives}}$$
Note- Recall is also called as Sensitivity or True Positive Rate
$$F-\beta = {(1+\beta^2)*Precision\;*\;Recall\over{\beta^2 * Precision\;+\;Recall} }$$

- When Focus is on both $FP$ and $FN$ we use $\beta =1$,
$$F1 = {2*Precision\;*\;Recall\over{Precision\;+\;Recall}}$$
(Harmonic Mean)

- When we want more weight on Precision, less weight on Recall we use $\mathbf{\beta <1}$,

	Let $\beta=0.5$
$$F0.5 = {(1+0.5^2)*Precision\;*\;Recall\over{0.5^2 * Precision\;+\;Recall}} = {1.25*Precision\;*\;Recall\over{0.25 * Precision\;+\;Recall}}$$

- When we want more weight on Recall, less weight on Precision we use $\mathbf{\beta >1}$,

	Let $\beta=2$
$$F2 = {(1+2^2)*Precision\;*\;Recall\over{2^2 * Precision\;+\;Recall}} = {5*Precision\;*\;Recall\over{4 * Precision\;+\;Recall}}$$

Note - *Above measures are used in Binary as well as Multi-Class Classification problems*

<a href=https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/ > Click </a> for more detailed measures for Imbalanced Classification.


### Naive Bayes Classifier
- Naive Bayes is a classification algorithm for binary (two-class) and multiclass classification problems.
	- It is called Naive Bayes or idiot Bayes because the calculations of the probabilities for each class.
  - Before we dive into Bayes theorem, let’s review marginal, joint, and conditional probability.
    - **Marginal Probability** : The probability of an event irrespective of the outcomes of other random variables. <br>e.g. P(A)= Fetching random card from deck of cards P(3 of diamond)=1/52
    - **Joint Probability**: Probability of two (or more) simultaneous events. <br>e.g. P(A and B) or P(A, B)= probability of picking up a card that is both red and 6 is P(6 ⋂ red) = 2/52 = 1/26
    - **Conditional Probability**: Probability of one (or more) event given the occurrence of another event. <br>e.g. P(A given B) or P(A | B)= probability that you get a 6, given that you drew a red card is P(6│red) = 2/26 = 1/13
      - The conditional probability can be calculated using the joint probability; for example:
		P(A | B) = P(A ⋂ B) / P(B)
  - **Bayes Theorem**: It is a way of calculating a conditional probability without the joint probability.
    - P(A ∣ B) = P(A ⋂ B) / P(B)  = P(A)⋅P(B ∣ A) / P(B)<br>
where:<br>
P(A)= The probability of A occurring<br>
P(B)= The probability of B occurring<br>
P(A∣B)=The probability of A given B<br>
P(B∣A)= The probability of B given A<br>
P(A⋂B))= The probability of both A and B occurring<br>
  <br>The terms in the Bayes Theorem equation are given names depending on the context where the equation is used.<br>
  -P(A) : **Prior probability**.<br>
  -P(B) : **Evidence**.<br>
  -P(A|B) : **Posterior probability**.<br>
  -P(B|A) : **Likelihood**.<br>
  -Bayes Theorom can be rewritten as: <br>**Posterior = Likelihood * Prior / Evidence**<br>Eg. What is the probability that there is fire given that there is smoke?<br>
Where P(Fire) is the Prior, P(Smoke|Fire) is the Likelihood, and P(Smoke) is the evidence:<br>
P(Fire|Smoke) = P(Smoke|Fire) * P(Fire) / P(Smoke)
​
