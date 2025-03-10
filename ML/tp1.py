import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary.txt')
x_full = df['YearsExperience'].values
y_full = df['Salary'].values

x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=1/3, random_state=42)

# Reshape x for scikit-learn (needs 2D array)
x_train_reshaped = x_train.reshape(-1, 1)
x_test_reshaped = x_test.reshape(-1, 1)

# Create and train scikit-learn model
sklearn_model = LinearRegression()
sklearn_model.fit(x_train_reshaped, y_train)

# Get sklearn model parameters
sklearn_w = sklearn_model.coef_[0]
sklearn_b = sklearn_model.intercept_

# Get predictions from sklearn model
sklearn_train_pred = sklearn_model.predict(x_train_reshaped)
sklearn_test_pred = sklearn_model.predict(x_test_reshaped)

# Calculate losses for sklearn model
sklearn_train_loss = np.mean((y_train - sklearn_train_pred) ** 2)
sklearn_test_loss = np.mean((y_test - sklearn_test_pred) ** 2)

# Original gradient descent implementation
w = 0
b = 0
learning_rate = 0.01
epochs = 400

train_loss_history = []
test_loss_history = []
w_history = []
b_history = []

for epoch in range(epochs):
    n = len(x_train)
    dldw = 0.0
    dldb = 0.0
    
    for xi, yi in zip(x_train, y_train):
        dldw += -2 * xi * (yi - (w * xi + b))
        dldb += -2 * (yi - (w * xi + b))
    
    w = w - learning_rate * (1/n) * dldw
    b = b - learning_rate * (1/n) * dldb
    
    y_train_pred = w * x_train + b
    y_test_pred = w * x_test + b
    
    train_loss = np.mean((y_train - y_train_pred) ** 2)
    test_loss = np.mean((y_test - y_test_pred) ** 2)
    
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    w_history.append(w)
    b_history.append(b)
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch}: Train loss: {train_loss:.2f}, Test loss: {test_loss:.2f}, w: {w:.4f}, b: {b:.2f}')

print(f'Final results - Gradient Descent: w: {w:.4f}, b: {b:.2f}')
print(f'Training loss: {train_loss:.2f}')
print(f'Testing loss: {test_loss:.2f}')

print(f'Scikit-learn results: w: {sklearn_w:.4f}, b: {sklearn_b:.2f}')
print(f'Scikit-learn Training loss: {sklearn_train_loss:.2f}')
print(f'Scikit-learn Testing loss: {sklearn_test_loss:.2f}')

# Final predictions after training
final_train_predictions = w * x_train + b
final_test_predictions = w * x_test + b

plt.figure(figsize=(18, 15))

# Create equation text for both models
gd_equation_text = f'y = {w:.2f}x + {b:.2f}'
sklearn_equation_text = f'y = {sklearn_w:.2f}x + {sklearn_b:.2f}'

# Plot 1: Data with both regression lines
plt.subplot(3, 2, 1)
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.scatter(x_test, y_test, color='green', label='Testing Data')
plt.plot(x_train, w * x_train + b, color='red', label=f'Gradient Descent: {gd_equation_text}')
plt.plot(x_train, sklearn_train_pred, color='purple', linestyle='--', 
         label=f'Scikit-learn: {sklearn_equation_text}')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience with Both Regression Models')
plt.legend()
plt.grid(True)

# Plot 2: Loss over training
plt.subplot(3, 2, 2)
plt.plot(range(epochs), train_loss_history, label='GD Training Loss', color='blue')
plt.plot(range(epochs), test_loss_history, label='GD Testing Loss', color='green')
plt.axhline(y=sklearn_train_loss, color='purple', linestyle='--', 
           label=f'Sklearn Train Loss: {sklearn_train_loss:.2f}')
plt.axhline(y=sklearn_test_loss, color='orange', linestyle='--', 
           label=f'Sklearn Test Loss: {sklearn_test_loss:.2f}')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Loss Comparison')
plt.legend()
plt.grid(True)

# Plot 3: Weight evolution
plt.subplot(3, 2, 3)
plt.plot(range(epochs), w_history, color='blue', label='Gradient Descent w')
plt.axhline(y=sklearn_w, color='purple', linestyle='--', 
           label=f'Sklearn w: {sklearn_w:.2f}')
plt.xlabel('Epoch')
plt.ylabel('Weight (w)')
plt.title('Weight Evolution Comparison')
plt.legend()
plt.grid(True)

# Plot 4: Bias evolution
plt.subplot(3, 2, 4)
plt.plot(range(epochs), b_history, color='blue', label='Gradient Descent b')
plt.axhline(y=sklearn_b, color='purple', linestyle='--', 
           label=f'Sklearn b: {sklearn_b:.2f}')
plt.xlabel('Epoch')
plt.ylabel('Bias (b)')
plt.title('Bias Evolution Comparison')
plt.legend()
plt.grid(True)

# Plot 5: Gradient Descent vs Actual (test data)
plt.subplot(3, 2, 5)
plt.scatter(y_test, final_test_predictions, color='red', label='Gradient Descent')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', label='Perfect Predictions')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Test Data: Gradient Descent Predictions')
plt.text(min(y_test) + 5000, max(y_test) - 10000, f'GD Model: {gd_equation_text}', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.7))
plt.legend()
plt.grid(True)

# Plot 6: Sklearn vs Actual (test data)
plt.subplot(3, 2, 6)
plt.scatter(y_test, sklearn_test_pred, color='purple', label='Scikit-learn')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', label='Perfect Predictions')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Test Data: Scikit-learn Predictions')
plt.text(min(y_test) + 5000, max(y_test) - 10000, f'Sklearn Model: {sklearn_equation_text}', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.7))
plt.legend()
plt.grid(True)

plt.tight_layout(pad=3.0)
plt.show()