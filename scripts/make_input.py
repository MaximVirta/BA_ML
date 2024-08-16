import numpy as np
import os
import ROOT
import sys
sys.path.append("../../JPyPlotRatio")
import JPyPlotRatio as JPPR

#
params_fp="../scripts/training_data_500.csv"    #path to csv file containing parameter combinations
results_dir="../jfluc_results"                  #path to directory with respective results
n_pts=7                                         #assumed number of points per observable

#Load parameters and results
params=np.loadtxt(params_fp, delimiter=",", skiprows=0)
param_indices=params[:,0]
params=np.delete(params,0,1)
result_tfiles=[ROOT.TFile(os.path.join(results_dir, f"results-{int(params_idx):03}.root")) for params_idx in param_indices]

#Each parameter vector points to a 2x7 matrix with y vals of "gr_mult_Charged" on first row and "gr_v2_QC" on the second row 
results=np.zeros((np.shape(params)[0], 2, n_pts))              
for params_idx, result_tfile in enumerate(result_tfiles):
    tgrapherrors_mc=result_tfile.Get("gr_mult_Charged")
    x_mc, y_mc,xerr_mc, yerr_mc=JPPR.TGraphErrorsToNumpy(tgrapherrors_mc)
    results[params_idx,0,:]=y_mc
    tgrapherrors_v2QC=result_tfile.Get("gr_v2_QC")
    x_v2QC, y_v2QC,xerr_v2QC, yerr_v2QC=JPPR.TGraphErrorsToNumpy(tgrapherrors_v2QC)
    results[params_idx,1,:]=y_v2QC

#Final input arr
#combined=np.concatenate((params_reshaped, results), axis=1)
#combined=np.stack((params[:,:np.newaxis], results), axis=2)
#print(np.shape(combined))

print(np.shape(results))
print(np.shape(params))

#palapelin palat ei sovi yhteen :(
np.save("results.npy", results)         #shape (500, 2, 7)  First indices match
np.save("param_vectors.npy", params)    #shape (500, 14)

# param_names=[
#     "norm", 
#     "p", 
#     "fluc", 
#     "nucl_w", 
#     "t_fs", 
#     "eta_s_hrg", 
#     "eta_s_min", 
#     "eta_slope", 
#     "eta_s_curv", 
#     "zeta_s_max", 
#     "zeta_s_w", 
#     "zeta_s_t0", 
#     "t_switch"]