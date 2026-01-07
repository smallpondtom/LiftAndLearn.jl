"""
Cost analysis of Streaming-OpInf experiments
"""
##
using Printf

## Burgers' 
n = 128
K = 10001
r = 14
d = r + r * (r+1) / 2 + 1
q = 4*r + 1
ζ = 8
svd_cost = n * (10*K + r)
ls_cost = d * (K + d)
original_total = svd_cost + ls_cost
isvd_cost = n * (q + ζ)
rls_cost = d^2 
stream_total = isvd_cost + rls_cost
@printf("Batch SVD cost: %.4e\n", svd_cost)
@printf("Batch LS cost: %.4e\n", ls_cost)
@printf("iSVD cost: %.4e\n", isvd_cost)
@printf("RLS cost: %.4e\n", rls_cost)
@printf("SVD cost reduction: %.4f%%\n", 100 - isvd_cost / svd_cost * 100)
@printf("LS cost reduction: %.4f%%\n", 100 - rls_cost / ls_cost * 100)
@printf("Total cost reduction: %.4f%%\n", 100 - stream_total / original_total * 100)

## KSE 
n = 512
K = 3000
r = 24
d = r + r * (r+1) / 2
q = 4*r + 1
ζ = 8
svd_cost = n * (9*K + r)
ls_cost = d * (K + d)
original_total = svd_cost + ls_cost
isvd_cost = n * r 
rls_cost = d^2 
stream_total = isvd_cost + rls_cost
@printf("Batch SVD cost: %.4e\n", svd_cost)
@printf("Batch LS cost: %.4e\n", ls_cost)
@printf("iSVD cost: %.4e\n", isvd_cost)
@printf("RLS cost: %.4e\n", rls_cost)
@printf("SVD cost reduction: %.4f%%\n", 100 - isvd_cost / svd_cost * 100)
@printf("LS cost reduction: %.4f%%\n", 100 - rls_cost / ls_cost * 100)
@printf("Total cost reduction: %.4f%%\n", 100 - stream_total / original_total * 100)

## Channel flow 
n = 384 * 192 * 32
K = 8000
r = 300
d = r + r * (r+1) / 2 + 1 
q = 4*r + 1
ζ = 8
svd_cost = n * (K + r)
ls_cost = d * (K + d)
original_total = svd_cost + ls_cost
isvd_cost = n * (q + ζ)
rls_cost = d * (K + d)
stream_total = isvd_cost + rls_cost
@printf("Batch SVD cost: %.4e\n", svd_cost)
@printf("Batch LS cost: %.4e\n", ls_cost)
@printf("iSVD cost: %.4e\n", isvd_cost)
@printf("RLS cost: %.4e\n", rls_cost)
@printf("SVD cost reduction: %.4f%%\n", 100 - isvd_cost / svd_cost * 100)
@printf("LS cost reduction: %.4f%%\n", 100 - rls_cost / ls_cost * 100)
@printf("Total cost reduction: %.4f%%\n", 100 - stream_total / original_total * 100)


#===================================================#
## Print so that I can copy-paste into latex table ##
#===================================================#
println("\n=== LaTeX Table ===")
println("\\begin{tabular}{lccccccc}")
println("\\hline")
println("Experiments & Batch SVD cost & Streaming SVD cost & Batch LS cost & Compact/Recursive LS cost & SVD cost reduction & LS cost reduction & Total cost reduction \\\\")
println("\\hline")

# Burgers'
n = 128
K = 10001
r = 14
d = r + r * (r+1) / 2 + 1
q = 4*r + 1
ζ = 8
svd_cost = n * (10*K + r)
ls_cost = d * (K + d)
isvd_cost = n * (q + ζ)
rls_cost = d^2
svd_reduction = 100 - isvd_cost / svd_cost * 100
ls_reduction = 100 - rls_cost / ls_cost * 100
total_reduction = 100 - (isvd_cost + rls_cost) / (svd_cost + ls_cost) * 100
@printf("Burgers' & %.2e & %.2e & %.2e & %.2e & %.2f\\%% & %.2f\\%% & %.2f\\%% \\\\\n",
        svd_cost, isvd_cost, ls_cost, rls_cost, svd_reduction, ls_reduction, total_reduction)

# KSE
n = 512
K = 3000
r = 24
d = r + r * (r+1) / 2
q = 4*r + 1
ζ = 8
svd_cost = n * (9*K + r)
ls_cost = d * (K + d)
isvd_cost = n * r
rls_cost = d^2
svd_reduction = 100 - isvd_cost / svd_cost * 100
ls_reduction = 100 - rls_cost / ls_cost * 100
total_reduction = 100 - (isvd_cost + rls_cost) / (svd_cost + ls_cost) * 100
@printf("KSE & %.2e & %.2e & %.2e & %.2e & %.2f\\%% & %.2f\\%% & %.2f\\%% \\\\\n",
        svd_cost, isvd_cost, ls_cost, rls_cost, svd_reduction, ls_reduction, total_reduction)

# Channel flow
n = 384 * 192 * 32
K = 8000
r_svd = 500
r = 300
d = r + r * (r+1) / 2 + 1
q = 4*r_svd + 1
ζ = 8
svd_cost = n * (K + r_svd)
ls_cost = d * (K + d)
isvd_cost = n * (q + ζ)
rls_cost = d * (K + d)
svd_reduction = 100 - isvd_cost / svd_cost * 100
ls_reduction = 100 - rls_cost / ls_cost * 100
total_reduction = 100 - (isvd_cost + rls_cost) / (svd_cost + ls_cost) * 100
@printf("Channel flow & %.2e & %.2e & %.2e & %.2e & %.2f\\%% & %.2f\\%% & %.2f\\%% \\\\\n",
        svd_cost, isvd_cost, ls_cost, rls_cost, svd_reduction, ls_reduction, total_reduction)

println("\\hline")
println("\\end{tabular}")