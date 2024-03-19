Base.@kwdef struct DataSetting
    N::Int
    num_ic::Int
    ti::Real
    tf::Real
    dt::Real
    DS::Real
    x0_bnds::Tuple{<:Real, <:Real}
    model_type::String
end

# Integrate the model
function integrate_model(ops::AbstractArray{<:AbstractArray{<:Real}}, x0::Array{<:Real}, 
                        ti::Real, tf::Real, dt::Real, type::String, dim::Int)
    if type == "Q"
        prob = ODEProblem(lin_quad_model!, x0, (ti, tf), ops)
    elseif type == "C"
        prob = ODEProblem(lin_cubic_model!, x0, (ti, tf), ops)
    elseif type == "QC"
        prob = ODEProblem(lin_quad_cubic_model!, x0, (ti, tf), ops)
    else
        error("Invalid type")
    end

    if dim == 2
        sol = solve(prob, RK4(); dt=dt, adaptive=false)
        data = sol[1:2,:]
        ddata = sol(ti:dt:tf, Val{1})[1:2,:]
    elseif dim == 3
        sol = solve(prob, RK4(); dt=dt, adaptive=false)
        data = sol[1:3,:]
        ddata = sol(ti:dt:tf, Val{1})[1:3,:]
    else
        error("Invalid dimension")
    end
    return data, ddata, Symbol(sol.retcode)
end


function generate_data(datasetting::DataSetting, ops::LnL.operators)
    X = []
    Xdot = []
    lb, ub = datasetting.x0_bnds
    ct = 0
    while ct < datasetting.num_ic
        x0 = (ub - lb)*rand(datasetting.N) .+ lb

        if datasetting.model_type == "Q"
            model_params = [ops.A, ops.F]
        elseif datasetting.model_type == "C"
            model_params = [ops.A, ops.E]
        elseif datasetting.model_type == "QC"
            model_params = [ops.A, ops.F, ops.E]
        else
            error("Invalid model type")
        end

        data, ddata, retcode = integrate_model(
            model_params, x0, datasetting.ti, datasetting.tf, 
            datasetting.dt, datasetting.model_type, datasetting.N
        )

        if retcode in (:Unstable, :Terminated, :Failure)
            continue
        elseif retcode == :Success
            ct += 1
        else
            error("Invalid retcode: $retcode")
        end
        push!(X, data[:,1:datasetting.DS:end])
        push!(Xdot, ddata[:,1:datasetting.DS:end])
    end
    X = reduce(hcat, X)
    Xdot = reduce(hcat, Xdot)

    return X, Xdot
end