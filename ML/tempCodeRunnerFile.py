# Adding the comparison between actual and predicted values for training data
plt.subplot(3, 2, 5)
plt.scatter(y_train, final_train_predictions, color='blue')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Training: Actual vs Predicted Salary')
plt.grid(True)

# Adding the comparison between actual and predicted values for testing data
plt.subplot(3, 2, 6)
plt.scatter(y_test, final_test_predictions, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Testing: Actual vs Predicted Salary')
plt.grid(True)