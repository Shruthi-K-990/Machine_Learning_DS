# Machine_Learning_DS
Machine Learning Codes for Data Science.
In this repository, you will find codes for ML models and algorithms.

1. ML Models.ipynb
  Dataset: This notebook uses the IRIS dataset, which includes 150 rows and 5 columns: sepal length, sepal width, petal length, petal width, and class.
  Visualization: The data is visualized using matplotlib and seaborn.
  Data Splitting: The dataset is split into 80% for training and 20% for testing.
  Modeling: Various machine learning models from sklearn are used, including:
  - LogisticRegression
  - LinearDiscriminantAnalysis
  - KNeighborsClassifier
  - DecisionTreeClassifier
  - GaussianNB
  - SVC

  Evaluation: The best-fit model for the IRIS dataset is identified based on the evaluation.
  
#######################################################################################################################################

2. Linear Regression.ipynb
   This file contains a simple example of implementing linear regression in Python using NumPy and Matplotlib. The code demonstrates how to estimate the coefficients of a linear regression   
   model and plot the regression line along with the data points.
Estimating Coefficients
The function estimate_coeff(x, y) calculates the coefficients for the linear regression model using the least squares method. The function takes two arguments, x and y, which are numpy arrays representing the independent and dependent variables, respectively.

Key Steps:
- Calculate the number of data points.
- Compute the mean of x and y.
- Calculate the covariance of x and y and the variance of x.
- Compute the slope (b1) and intercept (b0) of the regression line.
- Return the coefficients.
- 
Plotting the Regression Line
 The function plot_regression(x, y, b) plots the original data points and the regression line. It takes three arguments:

x: The independent variable (input data).
y: The dependent variable (output data).
b: A tuple containing the coefficients of the linear regression model (b0 and b1).
Key Steps:
Scatter plot the original data points.
Calculate the predicted y values using the regression coefficients.
Plot the regression line.
Label the axes and display the plot.

Example
An example is provided where:
x is an array representing the size of houses.
y is an array representing the cost of houses.
The coefficients are estimated using estimate_coeff(x, y).
The data points and regression line are plotted using plot_regression(x, y, b).

Output
The estimated coefficients and the plot showing the data points and the regression line.

##########################################################################################################################

3. Linear regression(Simple_&_Multiple)model.ipynb
    This code consists of two models simple and multiple where data of Boston is collected and trained and tested models.
    output: Root mean square of model 1 is greater than model 2. Hence model 2 is the best fit.
   
#####################################################################################################################

4. Logistic Regression.ipynb
This code evaluates a logistic regression model's performance using the ROC AUC score and ROC curve. It calculates the ROC AUC score to summarize the model's discrimination ability. The ROC curve is plotted to visualize the true positive rate against the false positive rate at various thresholds. A random guess reference line is added for comparison. Finally, the plot is labeled and displayed, with a legend indicating the ROC AUC score.
   
