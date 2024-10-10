#================#
## Load packages
#================#
import HSL_jll

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"EP-OpInf/") : joinpath(pwd(), "scripts/EP-OpInf/")

#===========#
## Settings
#===========#
KSE_system = LnL.SystemStructure(
    state=[1,2],
)
KSE_vars = LnL.VariableStructure(
    N=1,
)
KSE_data = LnL.DataStructure(
    Δt=KSE.Δt,
    DS=DS,  # keep every 100th data point
)
KSE_optim = LnL.OptimizationSetting(
    verbose=false,
    initial_guess=false,
    max_iter=1000,
    reproject=false,
    SIGE=false,
    HSL_lib_path=HSL_jll.libhsl_path,
    linear_solver="ma97",
)

options = LnL.LSOpInfOption(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
)

#==================================#
## Obtain standard OpInf operators
#==================================#
@info "Compute the least-squares solution"
options = LnL.LSOpInfOption(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
)

# Store values
op_LS = Array{LnL.Operators}(undef, KSE.param_dim)

@showprogress for i in eachindex(KSE.diffusion_coeffs)
    op_LS[i] = LnL.opinf(Xtr[i], Vr[i][:, 1:ro[end]], options; Xdot=Rtr[i])
end

#==============================#
## Compute intrusive operators
#==============================#
@info "Compute the intrusive model"

# Store values
op_int = Array{LnL.Operators}(undef, KSE.param_dim)

@showprogress for i in eachindex(KSE.diffusion_coeffs)
    op_int[i] = LnL.pod(op_fom_tr[i], Vr[i][:, 1:ro[end]], options.system)
end

#===============================#
## Compute EPHEC-OpInf operators
#===============================#
@info "Compute the EPHEC model"

options = LnL.EPHECOpInfOption(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
    linear_operator_bounds=(-1000.0, 1000.0),
    quad_operator_bounds=(-100.0, 100.0),
)

# Store values
op_ephec =  Array{LnL.Operators}(undef, KSE.param_dim)

@showprogress for i in eachindex(KSE.diffusion_coeffs)
    op_ephec[i] = LnL.epopinf(Xtr[i], Vr[i][:, 1:ro[end]], options; Xdot=Rtr[i])
end

#===============================#
## Compute EPSIC-OpInf operators
#===============================#
@info "Compute the EPSIC model"

options = LnL.EPSICOpInfOption(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
    ϵ = 1e-3,
    linear_operator_bounds=(-1000.0, 1000.0),
    quad_operator_bounds=(-100.0, 100.0),
)

# Store values
op_epsic = Array{LnL.Operators}(undef, KSE.param_dim)

@showprogress for i in eachindex(KSE.diffusion_coeffs)
    op_epsic[i] = LnL.epopinf(Xtr[i], Vr[i][:, 1:ro[end]], options; Xdot=Rtr[i])
end

#==============================#
## Compute EPP-OpInf operators
#==============================#
@info "Compute the EPP OpInf."

options = LnL.EPPOpInfOption(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
    α=1e6,
    linear_operator_bounds=(-1000.0, 1000.0),
    quad_operator_bounds=(-100.0, 100.0),
)

# Store values
op_epp =  Array{LnL.Operators}(undef, KSE.param_dim)

@showprogress for i in eachindex(KSE.diffusion_coeffs)
    op_epp[i] = LnL.epopinf(Xtr[i], Vr[i][:, 1:ro[end]], options; Xdot=Rtr[i])
end

#=================#
## Save operators
#=================#
@info "Save the operators"
tmp = joinpath(FILEPATH, "data/kse_epopinf_ops.jld2")
save(tmp, "OPS",
    Dict("op_LS" => op_LS, "op_int" => op_int, "op_ephec" => op_ephec,
         "op_epsic" => op_epsic, "op_epp" => op_epp)
)
@info "Done. Saved the operators to $(tmp)"