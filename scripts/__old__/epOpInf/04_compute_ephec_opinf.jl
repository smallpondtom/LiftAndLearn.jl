if !(@isdefined LnL)
    include("00_settings.jl")
end

# Switch optimization scheme
options.optimization = "EPHEC"

if !(@isdefined Data)
    @info "Load the data dictionary"
    Data = load("scripts/data/epOpInf_periodic_data.jld2")
    @info "Data loaded."
end

@info "Compute the EPHEC OpInf."

Vrmax_::Matrix{Float64} = Data["Vrmax"]
# Compute non-constrained OpInf (the initial guess is the non-constrained operator inference operators)
op_ephec_opinf = LnL.inferOp(Data["Xtr"], zeros(1,1), zeros(1,1), Vrmax, Vrmax' * Data["Rtr"], options, Data["op_int"])
@info "The EPHEC OpInf computed."
    
# Save to data
@info "Save the inferred operator"
Data["op_ephec_opinf"] = op_ephec_opinf
save("scripts/data/epOpInf_periodic_data.jld2", Data)
@info "Done."