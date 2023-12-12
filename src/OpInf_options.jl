export Abstract_Options, sys_struct, vars, data, opt_settings
export λtik, LS_options, NC_options, EPHEC_options, EPSIC_options, EPP_options

""" 
    sys_struct(is_lin::Bool, has_control::Bool, 
        has_output::Bool, is_quad::Bool, is_bilin::Bool, 
        has_const::Bool, has_funcOp::Bool, is_lifted::Bool)

Structure of the given system.

## Fields
- `is_lin::Bool`: the system is linear
- `is_quad::Bool`: the system is quadratic
- `is_bilin::Bool`: the system is bilinear
- `has_control::Bool`: the system has control inputs
- `has_output::Bool`: the system has output
- `has_const::Bool`: the system has a constant operator
- `has_funcOp::Bool`: the system has a functional operator for ODE
- `is_lifted::Bool`: the system is lifted
"""
@with_kw mutable struct sys_struct 
    is_lin::Bool = true
    has_control::Bool = false
    has_output::Bool = false
    is_quad::Bool = false
    is_bilin::Bool = false
    has_const::Bool = false
    has_funcOp::Bool = false
    is_lifted::Bool = false

    @assert !(has_control==false && is_bilin==true) "Bilinear system must have control input. Check the option 'has_control'."
end


"""
    vars(N::Int64, N_lift::Int64)

Information about the system variables.

## Fields
- `N::Int64`: the number of system variables
- `N_lift::Int64`: the number of lifted system variables
"""
@with_kw mutable struct vars
    N::Int64 = 0
    N_lift::Int64 = N
end


"""
    data(Δt::Float64, DS::Int64, deriv_type::String)

Information about the data.

## Fields
- `Δt::Float64`: the time step or temporal discretization
- `DS::Int64`: the downsampling rate
- `deriv_type::String`: the derivative scheme, e.g. "F"orward "E"uler
"""
@with_kw mutable struct data
    Δt::Float64 = 0.01
    DS::Int64 = 0
    deriv_type::String = "FE"
end


"""
    opt_settings(verbose::Bool, initial_guess::Bool, max_iter::Int64, 
        which_quad_term::String, reproject::Bool, SIGE::Bool, 
        provide_reduced_orders::Bool)

Information about the optimization.

## Fields
- `verbose::Bool`: enable the verbose output for optimization
- `initial_guess::Bool`: use initial guesses for optimization
- `max_iter::Int64`: the maximum number of iterations for the optimization
- `which_quad_term::String`: choose main quadratic operator (H or F) to use for computation
- `reproject::Bool`: use reprojection method for derivative data
- `SIGE::Bool`: use successive initial guess estimation
- `with_bnds::Bool`: add bounds to the variables
- `linear_solver::String`: the linear solver to use for optimization
- `HSL_lib_path::String`: the path to the HSL library
"""
@with_kw mutable struct opt_settings
    verbose::Bool = false
    initial_guess::Bool = false
    max_iter::Int64 = 3000
    which_quad_term::String = "F"
    reproject::Bool = false
    SIGE::Bool = false  # Successive Initial Guess Estimation
    with_bnds::Bool = false  # add bounds to the variables
    linear_solver::String = "none"
    HSL_lib_path::String = "none"
end


"""
    λtik(lin::Union{Real, AbstractArray{Real}}, quad::Real, ctrl::Real, bilin::Real)

Tikhonov regularization parameters.

## Fields
- `lin::Float64`: the Tikhonov regularization parameter for linear state operator
- `quad::Float64`: the Tikhonov regularization parameter for quadratic state operator
- `ctrl::Float64`: the Tikhonov regularization parameter for control operator
- `bilin::Float64`: the Tikhonov regularization parameter for bilinear state operator
"""
@with_kw struct λtik
    lin::Union{Real, AbstractArray{Real}} = 0.0
    quad::Real = 0.0
    ctrl::Real = 0.0
    bilin::Real = 0.0
end


"""
    LS_options(method::String, system::sys_struct, vars::vars, data::data, 
        optim::opt_settings, λ::λtik, pinv_tol::Real)

Standard Operator Inference.

## Fields
- `method::String`: the name of the method
- `system::sys_struct`: the system structure
- `vars::vars`: the system variables
- `data::data`: the data
- `optim::opt_settings`: the optimization settings
- `λ::λtik`: the Tikhonov regularization parameters
- `with_tol::Bool`: the option to use tolerance for the least square pseudo inverse
- `with_reg::Bool`: the option to use Tikhonov regularization
- `pinv_tol::Real`: the tolerance for the least square pseudo inverse
"""
@with_kw mutable struct LS_options <: Abstract_Options
    method::String = "LS"
    system::sys_struct = sys_struct()
    vars::vars = vars()
    data::data = data()
    optim::opt_settings = opt_settings()
    λ::λtik = λtik()
    with_tol::Bool = false  # This options makes it way slower
    with_reg::Bool = false  # tikhonov regularization
    pinv_tol::Real = 1e-6
end


"""
    NC_options(method::String, system::sys_struct, vars::vars, data::data, 
        optim::opt_settings, λ_lin::Real, λ_quad::Real)

Non-Constrained Operator Inference.

## Fields
- `method::String`: the name of the method
- `system::sys_struct`: the system structure
- `vars::vars`: the system variables
- `data::data`: the data
- `optim::opt_settings`: the optimization settings
- `λ_lin::Real`: the Tikhonov regularization parameter for linear state operator
- `λ_quad::Real`: the Tikhonov regularization parameter for quadratic state operator
"""
@with_kw mutable struct NC_options <: Abstract_Options
    method::String = "NC"
    system::sys_struct = sys_struct()
    vars::vars = vars()
    data::data = data()
    optim::opt_settings = opt_settings()
    λ_lin::Real = 0
    λ_quad::Real = 0
end


"""
    EPHEC_options(method::String, system::sys_struct, vars::vars, data::data, 
        optim::opt_settings, λ_lin::Real, λ_quad::Real)

Energy-Preserving Hard Equality Constraint Operator Inference.

## Fields
- `method::String`: the name of the method
- `system::sys_struct`: the system structure
- `vars::vars`: the system variables
- `data::data`: the data
- `optim::opt_settings`: the optimization settings
- `λ_lin::Real`: the Tikhonov regularization parameter for linear state operator
- `λ_quad::Real`: the Tikhonov regularization parameter for quadratic state operator
- `A_bnds::Tuple{Float64, Float64}`: the bounds for the linear operator
- `ForH_bnds::Tuple{Float64, Float64}`: the bounds for the quadratic operator (F or H)
"""
@with_kw mutable struct EPHEC_options <: Abstract_Options
    method::String = "EPHEC"
    system::sys_struct = sys_struct()
    vars::vars = vars()
    data::data = data()
    optim::opt_settings = opt_settings()
    λ_lin::Real = 0
    λ_quad::Real = 0
    A_bnds::Tuple{Float64, Float64} = (0.0, 0.0)
    ForH_bnds::Tuple{Float64, Float64} = (0.0, 0.0)
end


"""
    EPSIC_options(method::String, system::sys_struct, vars::vars, data::data, 
        optim::opt_settings, λ_lin::Real, λ_quad::Real, ϵ::Real)

Energy-Preserving Soft Inequality Constraint Operator Inference.

## Fields
- `method::String`: the name of the method
- `system::sys_struct`: the system structure
- `vars::vars`: the system variables
- `data::data`: the data
- `optim::opt_settings`: the optimization settings
- `λ_lin::Real`: the Tikhonov regularization parameter for linear state operator
- `λ_quad::Real`: the Tikhonov regularization parameter for quadratic state operator
- `ϵ::Real`: soft constraint radius
- `A_bnds::Tuple{Float64, Float64}`: the bounds for the linear operator
- `ForH_bnds::Tuple{Float64, Float64}`: the bounds for the quadratic operator (F or H)
"""
@with_kw mutable struct EPSIC_options <: Abstract_Options
    method::String = "EPSIC"
    system::sys_struct = sys_struct()
    vars::vars = vars()
    data::data = data()
    optim::opt_settings = opt_settings()
    λ_lin::Real = 0
    λ_quad::Real = 0
    ϵ::Real = 0.1
    A_bnds::Tuple{Float64, Float64} = (0.0, 0.0)
    ForH_bnds::Tuple{Float64, Float64} = (0.0, 0.0)
end


"""
    EPP_options(method::String, system::sys_struct, vars::vars, data::data, 
        optim::opt_settings, λ_lin::Real, λ_quad::Real, α::Float64)

Energy-Preserving Penalty Operator Inference.

## Fields
- `method::String`: the name of the method
- `system::sys_struct`: the system structure
- `vars::vars`: the system variables
- `data::data`: the data
- `optim::opt_settings`: the optimization settings
- `λ_lin::Real`: the Tikhonov regularization parameter for linear state operator
- `λ_quad::Real`: the Tikhonov regularization parameter for quadratic state operator
- `α::Float64`: the weight for the energy-preserving term in the cost function
- `A_bnds::Tuple{Float64, Float64}`: the bounds for the linear operator
- `ForH_bnds::Tuple{Float64, Float64}`: the bounds for the quadratic operator (F or H)
"""
@with_kw mutable struct EPP_options <: Abstract_Options
    method::String = "EPP"
    system::sys_struct = sys_struct()
    vars::vars = vars()
    data::data = data()
    optim::opt_settings = opt_settings()
    λ_lin::Real = 0
    λ_quad::Real = 0
    α::Float64 = 1.0
    A_bnds::Tuple{Float64, Float64} = (0.0, 0.0)
    ForH_bnds::Tuple{Float64, Float64} = (0.0, 0.0)
end
