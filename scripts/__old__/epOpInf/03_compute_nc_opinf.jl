if !(@isdefined LnL)
    include("00_settings.jl")
end

options.opt_verbose = true
options.optimization = "NC"
options.initial_guess_for_opt = true
options.max_iter = 5000

# Load the data dictionary
if !(@isdefined Data)
    @info "Load the data dictionary"
    Data = load("scripts/data/epOpInf_periodic_data.jld2")
    @info "Data loaded."
end

@info "Compute the non-constrained OpInf."
Vrmax::Matrix{Float64} = Data["Vrmax"]

# Compute non-constrained OpInf
op_nc_opinf = LnL.inferOp(Data["Xtr"], zeros(1,1), zeros(1,1), Vrmax, Vrmax' * Data["Rtr"], options, Data["op_ls_opinf"])
@info "The non-constrained OpInf computed."
    
# Save to data
@info "Save the inferred operator"
Data["op_nc_opinf"] = op_nc_opinf
save("scripts/data/epOpInf_periodic_data.jld2", Data)
@info "Done."