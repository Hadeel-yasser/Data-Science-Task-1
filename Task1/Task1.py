
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

print("Step 1: import the Data")
 # store the csv file into a string then pass it to the function read_csv to load data

filePath=r'C:\Users\EGYPT_LAPTOP\Desktop\New Courses\GRIP-Data analysis\Task1_Linear Regression\student_scores.csv'
studentData= pd.read_csv(filePath)

 # the head function returns the first 5 values, passing a parameter can define how many values to return
new_StudentData=studentData.head(15)
print(new_StudentData)

print("Step 2: visualize the imported data")
 # plotting the imported data using matplotlib

#studentData.plot(x="Hours", y="Scores", style='x')
plt.title("Scores in respect to Hours")
plt.xlabel("Hours studied by students")
plt.ylabel("Scores Achieved")
plt.grid()
#plt.show()
print("\n")
 # determine the correlation between the dependent and independent variables

print(studentData.corr()) # output shows a positive correlation 

print("Step 3: Prepare the data")
 # divide the data into input data (independent) and output data (dependent)
 # store the x values (hours) 'independent' in a list, extract from csv. Same goes for the dependent values

independent= studentData.iloc[:,0:1].values # 0 is the start column, 1 end column to select values upon
print(independent)
dependent = studentData.iloc[:,1:2].values
print(dependent)

 # split data into training and testing data by using Scikit-Learn (because there is only 1 dataset to test the accuracy of the prediction)
 # (there doesn't exist another dataset that could be used for testing)

x_train, x_test, y_train, y_test= train_test_split(independent,dependent,train_size=None, test_size=0.25, random_state=0) 
print("\n")
print ("X_train: ", x_train,type( x_train)) 
print("\n")
print ("y_train: ", y_train)
print("\n")
print("X_test: ", x_test) #  testing data in hours
print("\n")
print ("y_test: ", y_test) # testing data in scores
print("\n")

print("Step 4: Training the algorithm")  
  # create linearRegression object that would store the to be fitted data
regressor= LinearRegression()

  # train the model using the training data sets
regressor.fit(x_train,y_train)
print("Training Complete")
print("\n")

print("Step 5: Visualizing the training model")
  # defining the regression line (coef is the x values, intercept as in Y intercept)
regressionLine= regressor.coef_*independent + regressor.intercept_

  # plotting the training dataset and the regression line
plt.scatter(x_train,y_train, color='blue')
plt.plot(independent,regressionLine,color='red')

  # Plotting testing data then displaying both train and test data
plt.scatter(x_test,y_test, color='green')
#plt.show()

print("Step 6: Data Prediction")
print("\n")
  # predict the scores based on the number of hours in the testing data
scoresPrediction= regressor.predict(x_test)
print(scoresPrediction)
  # comparing predicted scores with actual by using pandas

d1 = dict(enumerate(y_test.flatten(), 1)) # store the nparray in a dictionary
d2=dict(enumerate(scoresPrediction.flatten(), 1))
table= pd.DataFrame({'Actual': d1, 'Predicted': d2})
print(table)
  # test the model with your own 
newHours=9.25
#hours=hours.reshape(-1,1)
myPrediction= regressor.predict([[newHours]])
print("Number of Hours Studied =",newHours, "will result in a Score of", myPrediction[0])


print("Step 7: Model Evaluation")

 # Evaluate the effectivness of the Linear Regression Model
 # mean square error metric will be used as a measure of accuracy
error= metrics.mean_absolute_error(y_test,scoresPrediction)
print("The mean absolute error of this model is",error)

