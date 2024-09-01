import numpy as np
import os
import ROOT
import sys
sys.path.append("../../JPyPlotRatio")
import JPyPlotRatio as JPPR
import pickle


validation = False;
if validation:
    params_fp="../scripts/parameters_validation_50.csv"    #path to csv file containing parameter combinations
    results_dir="../jfluc_results/testing"                  #path to directory with respective results
else:
    params_fp="../scripts/parameters_500.csv"    #path to csv file containing parameter combinations
    results_dir="../jfluc_results"                  #path to directory with respective results
n_pts=7                                         #assumed number of points per observable

#Load parameters and results
params=np.loadtxt(params_fp, delimiter=",", skiprows=0)
param_indices=params[:,0]
params=np.delete(params,0,1)
result_tfiles=[ROOT.TFile(os.path.join(results_dir, f"results-{int(params_idx):02}.root" if validation else f"results-{int(params_idx):03}.root")) for params_idx in param_indices]

#Each parameter vector points to a 2x7 matrix with y vals of "gr_mult_Charged" on first row and "gr_v2_QC" on the second row 
results=np.zeros((np.shape(params)[0], 14, n_pts))              
for params_idx, result_tfile in enumerate(result_tfiles):
    tgrapherrors_v2QC=result_tfile.Get("gr_v2_QC")
    x_v2QC, y_v2QC,xerr_v2QC, yerr_v2QC=JPPR.TGraphErrorsToNumpy(tgrapherrors_v2QC)
    results[params_idx,0,:]=y_v2QC
    tgrapherrors_v3QC=result_tfile.Get("gr_v3_QC")
    x_v3QC, y_v3QC,xerr_v3QC, yerr_v3QC=JPPR.TGraphErrorsToNumpy(tgrapherrors_v3QC)
    results[params_idx,1,:]=y_v3QC
    tgrapherrors_v4QC=result_tfile.Get("gr_v4_QC")
    x_v4QC, y_v4QC,xerr_v4QC, yerr_v4QC=JPPR.TGraphErrorsToNumpy(tgrapherrors_v4QC)
    results[params_idx,2,:]=y_v4QC
    tgrapherrors_v5QC=result_tfile.Get("gr_v5_QC")
    x_v5QC, y_v5QC,xerr_v5QC, yerr_v5QC=JPPR.TGraphErrorsToNumpy(tgrapherrors_v5QC)
    results[params_idx,3,:]=y_v5QC
    tgrapherrors_mc=result_tfile.Get("gr_mult_Charged")
    x_mc, y_mc,xerr_mc, yerr_mc=JPPR.TGraphErrorsToNumpy(tgrapherrors_mc)
    results[params_idx,4,:]=y_mc
    tgrapherrors_pion=result_tfile.Get("gr_mult_rap_Pion")
    x_pion, y_pion,xerr_pion, yerr_pion=JPPR.TGraphErrorsToNumpy(tgrapherrors_pion)
    results[params_idx,5,:]=y_pion
    tgrapherrors_kaon=result_tfile.Get("gr_mult_rap_Kaon")
    x_kaon, y_kaon,xerr_kaon, yerr_kaon=JPPR.TGraphErrorsToNumpy(tgrapherrors_kaon)
    results[params_idx,6,:]=y_kaon
    tgrapherrors_proton=result_tfile.Get("gr_mult_rap_Proton")
    x_proton, y_proton,xerr_proton, yerr_proton=JPPR.TGraphErrorsToNumpy(tgrapherrors_proton)
    results[params_idx,7,:]=y_proton
    tgrapherrors_ptpion=result_tfile.Get("gr_pTmean_Pion")
    x_ptpion, y_ptpion,xerr_ptpion, yerr_ptpion=JPPR.TGraphErrorsToNumpy(tgrapherrors_ptpion)
    results[params_idx,8,:]=y_ptpion
    tgrapherrors_ptkaon=result_tfile.Get("gr_pTmean_Kaon")
    x_ptkaon, y_ptkaon,xerr_ptkaon, yerr_ptkaon=JPPR.TGraphErrorsToNumpy(tgrapherrors_ptkaon)
    results[params_idx,9,:]=y_ptkaon
    tgrapherrors_ptproton=result_tfile.Get("gr_pTmean_Proton")
    x_ptproton, y_ptproton,xerr_ptproton, yerr_ptproton=JPPR.TGraphErrorsToNumpy(tgrapherrors_ptproton)
    results[params_idx,10,:]=y_ptproton
    tgrapherrors_sc32=result_tfile.Get("gr_sc32N_QC")
    x_sc32, y_sc32,xerr_sc32, yerr_sc32=JPPR.TGraphErrorsToNumpy(tgrapherrors_sc32)
    results[params_idx,11,:]=y_sc32
    tgrapherrors_sc42=result_tfile.Get("gr_sc42N_QC")
    x_sc42, y_sc42,xerr_sc42, yerr_sc42=JPPR.TGraphErrorsToNumpy(tgrapherrors_sc42)
    results[params_idx,12,:]=y_sc42
    tgrapherrors_sc43=result_tfile.Get("gr_sc43N_QC")
    x_sc43, y_sc43,xerr_sc43, yerr_sc43=JPPR.TGraphErrorsToNumpy(tgrapherrors_sc43)
    results[params_idx,13,:]=y_sc43
    


#Final input arr
#combined=np.concatenate((params_reshaped, results), axis=1)
#combined=np.stack((params[:,:np.newaxis], results), axis=2)
#print(np.shape(combined))

print(np.shape(results))
print(np.shape(params))

#palapelin palat ei sovi yhteen :(
np.save("results_all_targets.npy", results)         #shape (500, 2, 7)  First indices match
np.save("param_vectors_all_targets.npy", params)    #shape (500, 14)

outputList = [params, results]

with open("testing_data_50_all_targets.pkl" if validation else "training_data_500_all_targets.pkl", "wb") as f:
    pickle.dump(outputList, f);



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