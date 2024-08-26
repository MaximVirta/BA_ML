import numpy as np
import matplotlib.pyplot as plt
import pickle
import ROOT


# Load test data
with open('testing_data_50.pkl', 'rb') as f:
    test_data = pickle.load(f)

X_test = test_data[0] 
y_test = test_data[1]

y_test_flat = y_test.reshape(y_test.shape[0], -1)

# Load trained model
model_filename = 'trained_gb_model.pkl'
with open(model_filename, 'rb') as file:
    multi_target_regressor = pickle.load(file)


# Predict and reshape the output
y_pred_flat = multi_target_regressor.predict(X_test)
y_pred = y_pred_flat.reshape(y_pred_flat.shape[0], 2, 7)

# List for the flattened data from ROOT files
root_data_flat = []

n_samples = len(y_test)

for i in range(n_samples):

    root_filename = f'../gaussian_emulator/valid_all_{i:02}.root'  
    root_file = ROOT.TFile(root_filename, 'READ')

    # Extract y data from the ROOT files for both variables
    dNch_deta_charged_graph = root_file.Get("dNch_deta_charged")
    n_points = dNch_deta_charged_graph.GetN()
    dNch_deta_charged = np.array([dNch_deta_charged_graph.GetY()[i] for i in range(n_points)])

    vnk_2_graph = root_file.Get("vnk_2")
    n_points1 = vnk_2_graph.GetN()
    vnk_2 = np.array([vnk_2_graph.GetY()[i] for i in range(n_points1)])

    # Combine the values into 2x7 matrix
    combined_matrix = np.vstack([dNch_deta_charged, vnk_2])

    # Add the flattened matrix to the list
    root_data_flat.append(combined_matrix.flatten())

    root_file.Close()

# Convert lists to numpy arrays 
root_data_flat = np.array(root_data_flat)
print("Flattened:",root_data_flat.shape)

# Plot the actual, predicted, and emulator on the same plot
plt.figure(figsize=(8, 8))  

# Scatter plot for actual vs. predicted values
plt.scatter(y_test_flat, y_pred_flat, alpha=0.5, c='blue', marker='o', label='Predicted')

# Scatter plot for actual vs. emulator
plt.scatter(y_test_flat, root_data_flat, alpha=0.5, c='green', marker='x', label='Emulator')

# Actual values
plt.plot([y_test_flat.min(), y_test_flat.max()], [y_test_flat.min(), y_test_flat.max()], 'r--')

plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.legend(prop={'size': 16})

plt.tight_layout()
plt.savefig("../figs/pred_vs_actual_vs_emulator.png")
plt.show()
