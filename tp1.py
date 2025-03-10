import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('salary.txt')
x_full = df['YearsExperience'].values
y_full = df['Salary'].values

x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=1/3, random_state=42)

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

print(f'Final results - w: {w:.4f}, b: {b:.2f}')
print(f'Training loss: {train_loss:.2f}')
print(f'Testing loss: {test_loss:.2f}')

# Final predictions after training
final_train_predictions = w * x_train + b
final_test_predictions = w * x_test + b

plt.figure(figsize=(16, 12))

# Create equation text for the model
equation_text = f'y = {w:.2f}x + {b:.2f}'

plt.subplot(3, 2, 1)
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.scatter(x_test, y_test, color='green', label='Testing Data')
plt.plot(x_train, w * x_train + b, color='red', label=f'Regression Line: {equation_text}')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience with Regression Line (Trained Model)')
plt.legend()
plt.grid(True)
# Add the equation text directly on the plot
plt.text(min(x_full) + 0.5, max(y_full) - 10000, equation_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.subplot(3, 2, 2)
plt.plot(range(epochs), train_loss_history, label='Training Loss', color='blue')
plt.plot(range(epochs), test_loss_history, label='Testing Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Loss over Training')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(range(epochs), w_history, color='purple')
plt.xlabel('Epoch')
plt.ylabel('Weight (w)')
plt.title(f'Weight Evolution during Training (Final w = {w:.2f})')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(range(epochs), b_history, color='orange')
plt.xlabel('Epoch')
plt.ylabel('Bias (b)')
plt.title(f'Bias Evolution during Training (Final b = {b:.2f})')
plt.grid(True)

# Adding the comparison between actual and predicted values for training data
plt.subplot(3, 2, 5)
plt.scatter(y_train, final_train_predictions, color='blue')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', label='Perfect Predictions ')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Training: Actual vs Predicted Salary')
plt.text(min(y_train) + 5000, max(y_train) - 10000, f'Model: {equation_text}', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.7))
plt.legend()
plt.grid(True)

# Adding the comparison between actual and predicted values for testing data
plt.subplot(3, 2, 6)
plt.scatter(y_test, final_test_predictions, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect Predictions ')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Testing: Actual vs Predicted Salary')
plt.text(min(y_test) + 5000, max(y_test) - 10000, f'Model: {equation_text}', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.7))
plt.legend()
plt.grid(True)

plt.tight_layout(pad=3.0)
plt.show()