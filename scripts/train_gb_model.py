import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import ROOT


# Load training and testing data from pickle files
with open('training_data_500_all_targets.pkl', 'rb') as f:
    data = pickle.load(f)

with open('testing_data_50_all_targets.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Training data from the pickle file
X_train = data[0]  # Shape (500, 14)
y_train = data[1]  # Shape (500, 14, 7)

# Flatten the target to shape (500, 14)
y_train = y_train.reshape(y_train.shape[0], -1)

#X_train, X_test, y_train, y_test  = train_test_split(X, y_flattened, test_size=0.1, random_state=1)

# Test data from another file
X_test = test_data[0] 
y_test = test_data[1]

y_test_flattened = y_test.reshape(y_test.shape[0], -1)

# Create lists to hold flattened data from ROOT files
root_data_flat = []

# Initialize the gradient boosting regression model
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, min_samples_leaf=2, min_samples_split=10,random_state=1)

# Train the model using MultiOutputRegressor
multi_target_regressor = MultiOutputRegressor(gbr)
multi_target_regressor.fit(X_train, y_train)

"""
# Load trained model
model_filename = 'trained_gb_model.pkl'
with open(model_filename, 'rb') as file:
    multi_target_regressor = pickle.load(file)
"""

# Predict and reshape the output
y_pred_flat = multi_target_regressor.predict(X_test)
y_pred = y_pred_flat.reshape(y_pred_flat.shape[0], 14, 7)

# Evaluate the model
mse = mean_squared_error(y_test_flattened, y_pred_flat)
rmse = np.sqrt(mse)
print(f"Mean squared error: {mse}")
print(f"Root mean squared error: {rmse}")

# Print the first prediction
print("First prediction:")
print(y_pred[0])


# Save trained model
model_filename = 'trained_gb_model_all_targets.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(multi_target_regressor, file)


# Scatterplot of the predicted and actual values
plt.figure(figsize=(8, 8))
plt.scatter(y_test.flatten(), y_pred_flat.flatten(), alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Predicted vs actual values")
plt.savefig("../figs/pred_vs_actual_values_all.pdf")



