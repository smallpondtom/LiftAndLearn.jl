if !(@isdefined LnL)
    include("00_settings.jl")
end

# Load the data dictionary
if !(@isdefined Data)
    @info "Load the data dictionary"
    Data = load("scripts/data/epOpInf_periodic_data.jld2")
    @info "Data loaded."
end

@info "Compute the EPSIC OpInf."
# Switch optimization scheme
options.optimization = "EPSIC"
options.ϵ_ep = 0.1
options.max_iter = 5000

Vrmax::Matrix{Float64} = Data["Vrmax"]

# Compute non-constrained OpInf (Give the initial guess the non-constrained operator inference operators)
op_epsic_opinf = LnL.inferOp(Data["Xtr"], zeros(1,1), zeros(1,1), Vrmax, Vrmax' * Data["Rtr"], options, Data["op_int"])
@info "The EPSIC OpInf computed."

# Save to data
@info "Save the inferred operator"
Data["op_epsic_opinf"] = op_epsic_opinf
save("scripts/data/epOpInf_periodic_data.jld2", Data)
@info "Done."