# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the 50 Startups dataset.
2. Select and scale the input features.
3. Initialize weights, bias, and learning rate.
4. Update parameters using Gradient Descent.
5. Predict and display the profit.

## Program:
```
/* Program to implement the linear regression using gradient descent.
Developed by: VARSHINI K M
RegisterNumber: 212225240179 */

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("50_Startups.csv")

X = data[["R&D Spend", "Administration", "Marketing Spend"]].values
y = data["Profit"].values

# Feature Scaling
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Parameters
m, n = X.shape
w = np.zeros(n)
b = 0.0
alpha = 0.01
epochs = 1000

losses = []

# Gradient Descent
for i in range(epochs):

    # Prediction
    y_hat = np.dot(X, w) + b

    # Loss (Mean Squared Error)
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    # Gradients
    dw = (2 / m) * np.dot(X.T, (y_hat - y))
    db = (2 / m) * np.sum(y_hat - y)

    # Update Parameters
    w = w - alpha * dw
    b = b - alpha * db

# Results
print("Final Weights:", np.round(w, 2))
print("Final Bias:", f"{b:.2f}")

# Predict Profit for a New City
rd = float(input("Enter R&D Spend: "))
admin = float(input("Enter Administration Spend: "))
marketing = float(input("Enter Marketing Spend: "))

new_city = np.array([rd, admin, marketing])

# Feature Scaling for New City
new_city = (new_city - X_mean) / X_std

# Predicted Profit
predicted_profit = np.dot(new_city, w) + b

print("Predicted Profit:", f"{predicted_profit:.2f}")

# Loss vs Iterations
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations (Multiple Linear Regression)")
plt.show()
```

## Output:
![Uploading Screenshot 2026-08-18 095441.png…]()



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
