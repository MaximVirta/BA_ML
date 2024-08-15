import numpy as np
import ROOT
import sys
import re
import csv

designPath = sys.argv[1];
nparams = int(sys.argv[2]);

info = "#ind, norm, p, fluc, nucl_w, t_fs, eta_s_hrg, eta_s_min, eta_slope, eta_s_curv, zeta_s_max, zeta_s_w, zeta_s_t0, t_switch"

def createArray(inputFile, ind):
  with open(inputFile) as f:
    isArgs=f.readline();
    nuckleonW=f.readline().split()[-1];
    taufs=f.readline().split()[-1];
    hydroArgs=f.readline();
    tswitch=f.readline().split()[-1];
    norm=isArgs.split()[7];
    p=isArgs.split()[9];
    fluc=isArgs.split()[11];
    d=float(isArgs.split()[13]);
    allhargs=re.split("=|\s", hydroArgs);
    etashrg=allhargs[4];
    etasmin=allhargs[6];
    etaslope=allhargs[8];
    etascurv=allhargs[10];
    zetasmax=allhargs[12];
    zetasw=allhargs[14];
    zetast0=allhargs[16];
    print(norm, p, fluc, d, nuckleonW, taufs, etashrg, etasmin, etaslope, etascurv, zetasmax, zetasw, zetast0, tswitch);
    return np.array([ind, norm, p, nuckleonW, fluc, d*d*d, taufs, etashrg, etasmin, etaslope, etascurv, zetast0, zetasmax, zetasw, tswitch])

f = open(f"training_data_{nparams}.csv", "w");
f.write(f"{info}\n");
writer = csv.writer(f);

for x in range(nparams):
  writer.writerow(createArray(f"{designPath}/{x:03d}.txt", x))

f.close();