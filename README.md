# Senior-Thesis
This repository contains all relevant code and CSV files for my Harvard undergraduate senior math thesis. I had assistance from Lev Kruglyak and ChatGPT. A brief explanation of each file is as follows, along with where in my thesis the file is used.
  
<ins>CSV files</ins>
- **Fig 3** (pg 11)
  *gaussian_sk.csv* stores data (beta, N, variance_over_N) for plotting $\mathrm{Var}(F_{N,\beta})/N$ against $N$, with $\beta \in \\{0.1,1,10\\}$
- **Fig 4** (pg 14)
  *wigner.csv* stores single column data for the histogram
- **Fig 5** (pg 15)
  *orthog_var.csv* stores data (N, beta, max_variance) for plotting $\mathrm{Var}(F_{N,\beta})/N$ against $N$, with $\beta \in \\{0.1,1,10\\}$
- **Fig 6** (pg 16)
  *orthog_mean.csv* stores data (N, beta, max_mean) for plotting $\mathbb{E}[F_{N,\beta}]$ against $N$, with $\beta \in \\{0.1,1,10\\}$
- **Fig 9** (pg 22)
  *binpacking.csv* stores data (n, var_b) for plotting $\mathrm{Var}(B)$ against $N$
- **Fig 11** (pg 33)
  *tw_clt.csv* stores data (x, ecdf_n10, ecdf_n100, ecdf_n1000) for plotting the 3 ECDFs, with $n \in \\{10,100,1000\\}$

<ins>Simulation files</ins>
- *gaussian_sk* generates *gaussian_sk.csv*
- *orthog* generates *orthog_var.csv* and *orthog_mean.csv*
- *bin_packing.py* generates *bin_packing.csv*
- *tw_clt.py* generates *tw_clt.csv*
- *wigner.py* generates *wigner.csv*

<ins>Other</ins>
- **Fig 8** (pg 19)
  *sphere_concentration.nb* contains Mathematica code for computing surface densities $\sigma_n(E^\varepsilon)$ of equatorial bands, with $n \in \\{2,20,200,2000,20000,200000\\}$ and $\varepsilon \in \\{0.5, 0.05, 0.005\\}$
