import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load dataset
data = pd.read_csv("NairobiOfficePriceEx.csv")
x = data['SIZE'].values
y = data['PRICEgi'].values

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true- y_pred)**2)

def gradient_descent(x, y, m, c, learning_rate):
    n = len(y)
    y_pred = m*x + c
    #calculate the gradients
    d_m = (-2/n)*sum(x*(y-y_pred))
    d_c = (-2/n)*sum(y-y_pred)
    #update the weights
    m -= learning_rate*d_m
    c -= learning_rate*d_c
    return m, c

#Set the initial parameters
np.random.seed(42) 
m, c = np.random.randn(2)
learning_rate = 0.01
epochs = 10

#Train the model
for epoch in range(epochs):
   #predict the current values
    y_pred = m*x + c
    #calculate the error
    error= mean_squared_error(y, y_pred)
    print(f'Epoch: {epoch}, Error: {error}')
     # Update m and c using gradient descent
    m, c = gradient_descent(x, y, m, c, learning_rate)

    # Plotting
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, m * x + c, color='red', label='Line of Best Fit')
plt.xlabel('Office Size')
plt.ylabel('Office Price')
plt.legend()
plt.title('Linear Regression Line of Best Fit')
plt.show()


office_size = 100
predicted_price = m * office_size + c
print(f"Predicted Office Price for size 100 sq. ft: {predicted_price}")
