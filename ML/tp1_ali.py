import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt("salary.txt", delimiter=",", skiprows=1)
x_salary = data[:, 0]
y_salary = data[:, 1]

# Initialize parameters
w, bias = 0, 0
learning_rate = 0.0001  # Smaller learning rate to handle large x values
epochs = 1000  # More epochs to allow convergence
N = len(x_salary)

# Gradient Descent
for _ in range(epochs):
    y_pred = w * x_salary + bias
    dw = -2 / N * np.sum(x_salary * (y_salary - y_pred))
    db = -2 / N * np.sum(y_salary - y_pred)
    w -= learning_rate * dw
    bias -= learning_rate * db

# Generate values for plotting regression line
x_range = np.linspace(x_salary.min(), x_salary.max(), 100)
y_regression = w * x_range + bias

# Plot data points and regression line
plt.scatter(x_salary, y_salary, color="blue", label="Data Points")
plt.plot(x_range, y_regression, color="red", label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression: Experience vs Salary")
plt.legend()
plt.show()
