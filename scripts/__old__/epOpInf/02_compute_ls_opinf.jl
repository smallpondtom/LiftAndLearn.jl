if !(@isdefined LnL)
    include("00_settings.jl")
end

# Load the data dictionary
if !(@isdefined Data)
    @info "Load the data dictionary"
    Data = load("scripts/data/epOpInf_periodic_data.jld2")
    @info "Data loaded."
end

Vrmax::Matrix{Float64} = Data["Vrmax"]
op_fom_tr = Data["op_fom_tr"]

@info "Compute the values for the intrusive model from the basis of the training data"
# Compute the values for the intrusive model from the basis of the training data
op_int = LnL.intrusiveMR(op_fom_tr, Vrmax, options)
@info "op_int computed."

@info "Compute the inferred operators from the training data"
# Compute the inferred operators from the training data
if options.reproject 
    op_inf_LS = LnL.inferOp(Data["Xtr"], zeros(1,1), zeros(1,1), Vrmax, op_fom_tr, options)  # Using Reprojection
else
    op_inf_LS = LnL.inferOp(Data["Xtr"], zeros(1,1), zeros(1,1), Vrmax, Vrmax' * Data["Rtr"], options)
end
@info "op_inf_LS computed."

@info "Save the inferred operator"
Data["op_ls_opinf"] = op_inf_LS
Data["op_int"] = op_int
save("scripts/data/epOpInf_periodic_data.jld2", Data)
@info "Data saved."