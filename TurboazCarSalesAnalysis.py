# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:31:16 2020

@author: f.m.bayramova
"""
#imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
#from sklearn.metrics import mean_squared_errpr, r2_score

#reading data
data = pd.read_csv(r'C:\Users\f.m.bayramova\Desktop\data\uni\summer2020\Machine Learning\turboaz.csv')
#information about read data
data.info() #gives info about content of the csv file (rows & colums)
data.describe() #statistical description of csv file
data.head(10) #content and values in the csv file based on the number stated

#manipulations with contents of the colums
#the next line of code removes km from the string and replaces it with empty char variable
x = data['Yurush'].map(lambda x: x.rstrip('km').replace(' ', '')).map(int)

#assigning values of the string
y = data['Buraxilish ili']

#assigning values of the string with money sign manipulation
z = data['Qiymet'].map(lambda x: float(x.rstrip('$'))*1.7 if '$' in x else float(x.rstrip('AZN')))
#removing money sign and convertin dollar to azn

x_original = x
y_original = y
z_original = z

#I save the original values for future predictions
#Also the scaling of new values depends on original values

#Visualization starts here
plt.scatter(x, z)
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.show()
#as the mileage grows the price decreases
plt.scatter(y, z)
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()
#cars manufactured recently correspond to greater price

#3D Plot of x, y, z
shape = plt.figure()
axy = Axes3D(shape)
axy.scatter(x, y, z, color = '#e0198a')
plt.show()

#I use mean normalization so that the data appears in normalized way
x = (x - x.mean())/x.std()
y = (y - y.mean())/y.std()
z = (z - z.mean())/y.std()

l = len(x) #length of the dataset
x0 = np.ones(l) #calculating bias
X = np.array([x0, x, y]).T #calculating X
beta = np.array([0, 0, 0]) #calculating beta
alpha = 0.001 #learning rate

def cost_function(X, Y, B):
    l = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * l)
    return J

initial_cost = cost_function(X, z, beta)
initial_cost #calculating initial cost with initial parameters

def gradient_descent(X, Y, B, alpha, i):
    cost_history = [0] * i
    l = len(Y)
    for i in range (i):
        if i % 1000 == 0:
            print("#%d Iteration" % i)
            print(cost_function(X,Y, B))
            
            #predicted values
            predict = X.dot(B)
            lost = predict - Y
            #Predicted Value - Actual Values
            gradient = X.T.dot(lost) / l
            B = B - alpha * gradient
            
            #calculating new cost values
            cost = cost_function(X, Y, B)
            cost_history[i] = cost
            return B, cost_history
        
beta1, cost_history = gradient_descent(X, z, beta, alpha, 10000)
print(beta1) #printing new paramenters
print(cost_history[-1]) #printing last cost

#ploting the cost decrease
plt.plot(cost_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost per iteration')
plt.show()

plt.scatter(y, z)
plt.xlabel('Buraxilish ili')
plt.ylabel('Qiymet')
predict = beta1[2] * y + beta1[0] #predicting price based on the year of production
plt.plot(y, predict, c = 'r')
plt.show() 

plt.scatter(x, z)
plt.xlabel('Yurush')
plt.ylabel('Qiymet')
predict = beta1[1] * x + beta1[0] #predicting price based on the mileage
plt.plot(y, predict, c = 'r')
plt.show()         

shape = plt.figure()
axy = shape.gca(projection = '3d')
axy = Axes3D(shape)
axy.scatter(x, y, z, color = 'b')
predict = beta1[2] * y + beta1[1] * x + beta1[0] #predicting price based on mileage
axy.scatter(x, y, predict, color = 'r')

#Predicting with given examples

mileage = 240000
year = 2000
actual_price = 11500

#scaling the stated data
mileage = (mileage - x_original.mean())/x_original.std()
year = (year - y_original.mean())/y_original.std()
actual_price = (actual_price - z_original.mean())/y_original.std()

#predicting based on mileage
predicted_price = beta1[2] * year + beta1[1] * mileage + beta1[0]
#seeing prediction with comparison to actual price
print(predicted_price * z_original.std() + z_original.mean())
print(actual_price * z_original.std() + z_original.mean())

#second example
mileage = 415558
year = 1996
actual_price = 8800

#scaling the stated data
mileage = (mileage - x_original.mean())/x_original.std()
year = (year - y_original.mean())/y_original.std()
actual_price = (actual_price - z_original.mean())/y_original.std()
predicted_price = beta1[2] * year + beta1[1] * mileage + beta1[0]
#seeing predicted and actual result
#seeing prediction with comparison to actual price
print(predicted_price * z_original.std() + z_original.mean())
print(actual_price * z_original.std() + z_original.mean())

def gradient_descent_show_graph(X, Y, B, alpha, i, name):
    cost_history = [0] * i
    l = len(Y)
    for i in range(i):
        if i % 100 == 0:
            print("#&d Iteration" % i)
            if name == 'Yurush':
                plt.scatter(x, z)
                predict = B[1] * x + B[0]
                plt.plot(x, predict, c = 'r')
                plt.xlabel(name)
                plt.ylabel('Qiymet')
                plt.show()
                
            #values generated by prediction
                predict = X.dot(B)
                lost = predict - Y
                #predicted value - actual value
                gradient = X.T.dot(lost) / l #calculating gradient
                B = B - alpha * gradient
                #calculating new cost value
                cost = cost_function(X, Y, B)
                cost_history[i] = cost
                return B, cost_history
        beta = np.array([0,0,0])
        beta1, cost_history = gradient_descent_show_graph(X, z, beta, alpha, 10000, 'Yurush')
        
 #transforming data to a different format   
X_train = []
for i in range(len(x_original)):
    X_train.append(([x_original[i], y_original[i]]))
    
#GENERATING DATA TO BE USED IN PREDICTIONS    
X_test = [[240000, 2000], [41558, 1996]]
z_test = [11500, 8800]

regression = linear_model.LinearRegression()
regression.fit(X_train, z_original) #fitting data

z_predictions = regression.predict(X_test) #predicting values
print(z_predictions)
print('coefficients: \n', regression.coef_)

shape = plt.figure()
axy = shape.gca(projections = '3d')
axy = Axes3D(shape)
axy.scatter(x_original, y_original, z, color = 'b')
predictions = regression.coef_[1] * y_original + regression.coef_[0] * x_original + regression.intercept_ 
plt.show()

