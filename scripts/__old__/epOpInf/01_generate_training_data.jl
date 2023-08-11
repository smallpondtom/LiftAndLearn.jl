if !(@isdefined LnL)
    include("00_settings.jl")
end

@info "Generate the FOM system matrices and training data."
# Generate the FOM system matrices 
A, F = burger.generateEPmatrix(burger, μ)
op_fom_tr = LnL.operators(A=A, F=F)
Xref = burger.semiImplicitEuler(A, F, burger.t, burger.IC)

## training data for OpInf
Xall = Vector{Matrix{Float64}}(undef, num_ICs)
Xdotall = Vector{Matrix{Float64}}(undef, num_ICs)
for j in 1:num_ICs
    states = burger.semiImplicitEuler(A, F, burger.t, ic_a * burger.IC)
    tmp = states[:, 2:end]
    Xall[j] = tmp[:, 1:DS:end]  # downsample data
    tmp = (states[:, 2:end] - states[:, 1:end-1]) / burger.Δt
    Xdotall[j] = tmp[:, 1:DS:end]  # downsample data
end

Xtr::Matrix{Float64} = reduce(hcat, Xall)
Rtr::Matrix{Float64} = reduce(hcat, Xdotall)
@info "Done."

# Compute the POD basis from the training data
tmp = svd(Xtr)
Vrmax::Matrix{Float64} = tmp.U[:, 1:rmax]

# Save data
@info "Save data"
save(
    "scripts/data/epOpInf_periodic_data.jld2", 
    Dict("Xref" => Xref, "Rtr" => Rtr, "Xtr" => Xtr, "Vrmax" => Vrmax, "op_fom_tr" => op_fom_tr)
)
@info "Done."

