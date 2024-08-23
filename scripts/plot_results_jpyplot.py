import pickle
import numpy as np
import ROOT
import sys
sys.path.append("../../JPyPlotRatio")
import JPyPlotRatio


# Function to convert ROOT TGraph to NumPy arrays
def TGraphToArray(graph):
    n_points = graph.GetN()
    x = np.array([graph.GetX()[i] for i in range(n_points)])
    y = np.array([graph.GetY()[i] for i in range(n_points)])
    return x, y

# Load test data
with open('new_testing_data_50.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Extract test data
X_test = test_data[0]
y_test = test_data[1]

# Load trained model
model_filename = 'trained_gb_model_new_targets.pkl'
with open(model_filename, 'rb') as file:
    multi_target_regressor = pickle.load(file)

# Predict and reshape the output
y_pred_flat = multi_target_regressor.predict(X_test)
y_pred = y_pred_flat.reshape(y_pred_flat.shape[0], 8, 7)

# Centrality values 
centrality = np.array([2.5, 7.5, 15, 25, 35, 45, 55])

n_samples = y_test.shape[0]

# File path for emulator data
root_files = {
    f"test_case_{i}": ROOT.TFile(f'../gaussian_emulator/valid_all_{i:02}.root', 'READ')
    for i in range(n_samples)
}

# Data dictionary and variable names for emulator data
data = {}
names_list = ["vnk_2", "vnk_3", "vnk_4", "vnk_5", "charged", "pion", "kaon", "proton"]

for i in range(n_samples):
    root_file = root_files[f"test_case_{i}"]

    vnk_2 = TGraphToArray(root_file.Get("vnk_2"))[1]
    vnk_3 = TGraphToArray(root_file.Get("vnk_3"))[1]
    vnk_4 = TGraphToArray(root_file.Get("vnk_4"))[1]
    vnk_5 = TGraphToArray(root_file.Get("vnk_5"))[1]
    charged = TGraphToArray(root_file.Get("dNch_deta_charged"))[1]
    pion = TGraphToArray(root_file.Get("dNch_deta_pion"))[1]
    kaon = TGraphToArray(root_file.Get("dNch_deta_kaon"))[1]
    proton = TGraphToArray(root_file.Get("dNch_deta_proton"))[1]

    # Store data in dictionary
    data[f"test_case_{i}"] = {
            "vnk_2": vnk_2,
            "vnk_3": vnk_3,
            "vnk_4": vnk_4,
            "vnk_5": vnk_5,
            "charged": charged,
            "pion" : pion,
            "kaon" : kaon,
            "proton": proton
        }   


# Plot parameters
plotParams = {
    "Actual": {"fmt": "o", "color": "b", "label": "Actual"},
    "Predicted": {"fmt": "x", "color": "r", "label": "Predicted"},
    "Emulator": {"fmt": "s", "color": "g", "label": "Emulator"}
}

# Loop over each test case and accumulate data
for j in range(n_samples):
    plot = JPyPlotRatio.JPyPlotRatio(
        panels=(2,4), 
        xlabel="Centrality (%)",
        ylabel={0:"$v_n$",1:"$dN_{ch}/d\eta$"},
        panelScaling={2:2.0,3:4.0,6:2.0,7:3.0}, 
        panelLabel={0: "$v_2$", 1: "$v_3$", 2: "$v_4$", 3: "$v_5$", 4: "Charged", 5: "Pion", 6: "Kaon", 7: "Proton"},
        panelLabelLoc=(0.7, 0.9),
        panelLabelSize=12,
        legendLoc=(0.7, 0.5)
    )

    # Create the plots
    plots={}
    for i in range(8):
        plots[0,i]=plot.Add(i, arrays=(centrality, y_test[j][i]), **plotParams["Actual"])
        plots[1,i]=plot.Add(i, arrays=(centrality, y_pred[j][i]), **plotParams["Predicted"])
        plots[2,i]=plot.Add(i,arrays=(centrality, data[f"test_case_{j}"][names_list[i]]), **plotParams["Emulator"] )
        plot.Ratio(plots[1,i],plots[0,i])
        plot.Ratio(plots[2,i],plots[0,i])

    plot.Plot()
    plot.Save(f"../figs/plot_test_case_{j+1}_8_targets.pdf")
  


