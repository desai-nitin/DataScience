## Here I will be practising Data Science tutorials
* Topics:
1. EDA
	1. Zomato Data EDA
	2. Black Friday Dataset EDA and Feature Engineering
	3. Flight Price Prediction EDA and Feature Engineering
2. Supervised ML
    1. Linear Regression
    	- Cost Function - Mean Squared Error
		
			![Gradient Descent](https://github.com/desai-nitin/DataScience/blob/main/readme_images/mse1.png)

			![Gradient Descent](https://github.com/desai-nitin/DataScience/blob/main/readme_images/mse2.png)

		- Gradient Descent Formula
		
			![Gradient Descent](https://github.com/desai-nitin/DataScience/blob/main/readme_images/gradient_descent.png)
	
		- Linear Regression Convergence Formula
		
			![Linear Regression Convergence](https://github.com/desai-nitin/DataScience/blob/main/readme_images/LRConvergence.png)		
		- Learning Rate:
			Learning Rate is step size for each iteration while finding global minimum of the loss function.

			![Learning Rate](https://github.com/desai-nitin/DataScience/blob/main/readme_images/LearningRate1.png)

			![Learning Rate View](https://github.com/desai-nitin/DataScience/blob/main/readme_images/LearningRateView.png)

    2. Ridge And Lasso Regression
		
		Problem In Linear Regression: 
		| Overfitting     | Underfitting    | Generalized model |
		| --------------- |:---------------:| -----------------:|
		| Train Acc = 90% | Train Acc = 60% | Train Acc = 92%   |
		| Test Acc = 80%  | Test Acc = 58% | Test Acc = 91%   |
		| Low Bias        | High Bias       | Low Bias          |
		| High Variance   | High Variance   | Low Variance      |

		Addressing Overfitting:
		1. Reduce number of features
			- Manually select which features to be used
			- Feature selection
		2. Regularization
			- Keep all features but reduce magnitude of parameters ![theta](https://github.com/desai-nitin/DataScience/blob/main/readme_images/thetaj.png) 
			- Works well when we have a lot of features, each of which contributes a bit to predicting y

		***
	-  **Lasso Regression (L1 Regularization):**
		$${1\over2m} \sum_{i=1}^m (h_\theta (x)^i -y^i)^2 + \lambda \sum_{j=1}^n |\theta_j|$$

		- It adds "*aboslute value of magnitude*" of coefficents as penalty to loss function
		- Lasso shrinks the less important feature's coefficient to near zero thus, removing some feature.
		- Works as **feature selection** in case we have large number of features
		

	- **Ridge Regression (L2 Regularization):**
		$${1\over2m} \sum_{i=1}^m (h_\theta (x)^i -y^i)^2 + \lambda \sum_{j=1}^n \theta_j^2$$
		
		- It adds "*squared magnitute*" of coefficents as penalty to loss function
		- If $$\lambda$$ is very large then it will add too much weight and can lead to underfitting. Hence it's important how $$\lambda$$ is chosen.
		- This technique works very well to avoid **over-fitting**.
	