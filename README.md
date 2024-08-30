 

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1.start step 2.Import the standard Libraries.

step 3.Set variables for assigning dataset values.

step 4.Import linear regression from sklearn.

step 5.Assign the points for representing in the graph.

step 6.Predict the regression for marks by using the representation of the graph.

step 7.Compare the graphs and hence we obtained the linear regression for the given datas.

step 8.stop

Program:

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: NARRA RAMYA

RegisterNumber: 212223040128  
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\admin\Downloads\student_scores.csv")
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
  

```

## Output:
## dataset:
![image](https://github.com/user-attachments/assets/0fe3d315-4f4c-4cb5-a4c1-231538c3689d)
## Hard values:
![image](https://github.com/user-attachments/assets/0c1c274f-df90-4b72-8f95-3c67cbecdf49)
## tail values:
![image](https://github.com/user-attachments/assets/d245685d-1976-4224-bfc6-1619e83ca360)
## X and Y values:
![image](https://github.com/user-attachments/assets/4699ad50-0722-452e-8d52-a27075408008)
## Prediction of X and Y:
![image](https://github.com/user-attachments/assets/4832a427-040c-442e-8240-870560a9f50d)
## MSE, MAE and RMSE:
![image](https://github.com/user-attachments/assets/983e9353-ca5c-4b39-ac8e-99d1ca9371ea)

## Training Set:

![image](https://github.com/user-attachments/assets/3931f789-7321-4234-9235-578aedf8bb4b)

![image](https://github.com/user-attachments/assets/b9d77b73-1815-43b1-8f4c-fad9e0c4f9e1)










 


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
