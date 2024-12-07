# regressionsAndMore

Q1: Linear regression
An implementation of batch linear regression was done on the dataset provided, detailed working of the program is inside the code in form of comments.
(x: linearX; y: linearY). Initial values of theta0, theta1, eta, stopping criteria was given.
a)
Learning rate is 0.1 (Such high learning rate was taken as the program was always able to converge).
Theta values were initialized to ‘5’ each (such high values were taken to show the gradient of learning on surface plot, i.e., how the theta values change with each iteration were visible on surface)
Stopping criteria is when MSE decreases to less than 6.4*10**-6 (earlier 6*10**-6 was taken but the function was not able to decrease to such low value)
Final set of values are: Number of iterations = 109, MSE = 6.3633313*10**-6
theta0 is: 0.9967116596272918 and theta1 is: 0.001454729773464114

Q2: Sampling and Stochastic Gradient Descent
a) Dataset of 1M values were formed using given function also adding noise in it and saving it.
b,c,d) Different batches of various sizes were made using a function “Batc” and different stopping criteria were defined for all cases i.e.
Batch size 1: stopping criteria = absolute value of change in theta values<2*10**-7
Also, the stopping criteria was made more robust by adding a counter, such that if the stopping criteria is met ‘5’ times only then stop the training process. This was done to reduce the effect of randomness and ensure the convergence is due to actual convergence of error function and not because of random chance
Eta = 0.001
Results = iterations: 266582, theta0: 3.007725469761903, theta1: 0.9794332168445327, theta2: 2.0312979818911483
error (MSE) for test data(q2test.csv) is: 2.103890208910105
error (MSE) for original hypothesis function is: 0.14414347962222854
![image](https://github.com/user-attachments/assets/465f295d-709d-4d1a-935f-fb6d3b0a9e1c)


Batch size 100: stopping criteria = absolute value of change in theta values<3*10**-5
Also, the stopping criteria was made more robust by adding a counter, such that if the stopping criteria is met ‘5’ times only then stop the training process. This was done to reduce the effect of randomness and ensure the convergence is due to actual convergence of error function and not because of random chance

Eta = 0.001
Results = iterations: 15409, theta0: 2.9527389936897785, theta1: 1.0101534128884329, theta2: 1.9997428100143075
error (MSE) for test data(q2test.csv) is: 1.9768359330108605
error (MSE) for original hypothesis function is: 0.01155108491791847

![image](https://github.com/user-attachments/assets/27a1aa11-378d-4fd7-a110-323232734428)



Batch size 10000: stopping criteria = absolute value of change in theta values<3*10**-5
Also, the stopping criteria was made more robust by adding a counter, such that if the stopping criteria is met ‘5’ times only then stop the training process. This was done to reduce the effect of randomness and ensure the convergence is due to actual convergence of error function and not because of random chance
Eta = 0.001
Results = theta0: 2.9148668790418713, theta1: 1.018206539422197, theta2: 1.99384770315497
error (MSE) for test data(q2test.csv) is: 2.0067070788682986
error (MSE) for original hypothesis function is: 0.040335223942400696
![image](https://github.com/user-attachments/assets/2ef35d99-f55f-498d-8aa1-e14aaae20764)


Batch size 1 M: stopping criteria = absolute value of change in theta values<3*10**-3
Also, the stopping criteria was made more robust by adding a counter, such that if the stopping criteria is met ‘5’ times only then stop the training process. This was done to reduce the effect of randomness and ensure the convergence is due to actual convergence of error function and not because of random chance
Eta = 0.1
Results = iterations: 132, theta0: 2.9230486039669787, theta1: 1.016180373357568, theta2: 1.9945392114671847
error (MSE) for test data(q2test.csv) is: 1.9983156650626748
error (MSE) for original hypothesis function is: 0.031982964564808945
![image](https://github.com/user-attachments/assets/ee60a1ae-427b-4da7-b86c-413f3871b316)


Q3 Logistic Regression
Logistic regression using Newton method was implemented on the given dataset with the following results. More detailed of program is present in code in form of comments.
a)
Result : The weights theta0, theta1, theta2 are : [-0.0050503 -0.33780772 0.38538395]


Q4 Gaussian Discriminant Analysis
a) mean of data of Alaska is :(-0.7552943279913608, 0.6850943055489279)
mean of data of Canada is :(0.7552943279913606, -0.6850943055489274)
Co-variance matrix sigma is : [[ 1. -0.53992012]
[-0.53992012 1.]]
