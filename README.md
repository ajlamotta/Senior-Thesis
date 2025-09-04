# Senior-Thesis
This repository contains all relevant code and CSV files for my Harvard undergraduate senior math thesis. I had assistance from Lev Kruglyak and ChatGPT.

Python Files:
1. bin_packing.py generates data to plot $\mathrm{Var}(B)$ in Fig 9, pg 22 (Bin Packing);
    creates csv file binpacking.csv in format: n,var_b 
2. tw_clt.py generates data to plot 3 ECDFs in Fig 11, pg 33 (Tracy-Widom CLT);
    creates csv file tw_clt.csv in format: x,ecdf_n10,ecdf_n100,ecdf_n1000
3. wigner.py generates data to plot histogram in Fig 4, pg 14 (Semicircular Law);
    creates single column csv file wigner.csv

Mathematica:
1. sphere_concentration.nb computes the surface densities in Fig 8, pg 19

Rust files for SK simulations will be uploaded ASAP. These simulations create the following csv files:
1. gaussian_sk.csv used in Fig 3, pg 11 (Gaussian SK);
    format beta,N,variance_over_N
2. orthog_var.csv used in Fig 5, pg 15 (Orthogonal SK, Variance);
    format N,beta,max_variance
3. orthog_mean.csv used in Fig 6, pg 16 (Orthogonal SK, Mean);
    format N,beta,max_mean
