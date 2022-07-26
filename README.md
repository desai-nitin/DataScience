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
		- Learning Rate

			![Learning Rate](https://github.com/desai-nitin/DataScience/blob/main/readme_images/LearningRate1.png)

			![Learning Rate View](https://github.com/desai-nitin/DataScience/blob/main/readme_images/LearningRateView.png)

    2. Ridge And Lasso Regression
		
		Problem In Linear Regression: 
		| Overfitting     | Underfitting    | Generalized model |
		| --------------- |:---------------:| -----------------:|
		| Train Acc = 90% | Train Acc = 60% | Train Acc = 92%   |
		| Train Acc = 80% | Train Acc = 58% | Train Acc = 91%   |
		| Low Bias        | High Bias       | Low Bias          |
		| High Variance   | High Variance   | Low Variance      |

		Addressing Overfitting:
		1. Reduce number of features
			- Manually select which features to be used
		2. Regularization
			- Keep all features but reduce magnitude of parameters ![theta](https://github.com/desai-nitin/DataScience/blob/main/readme_images/thetaj.png) 
