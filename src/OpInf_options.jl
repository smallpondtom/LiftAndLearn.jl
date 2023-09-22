abstract type Abstract_Options end


""" 
Structure of the given system.

# Fields
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
Information about the system variables.

# Fields
- `N::Int64`: the number of system variables
- `N_lift::Int64`: the number of lifted system variables
"""
@with_kw mutable struct vars
    N::Int64 = 0
    N_lift::Int64 = N
end


"""
Information about the data.

# Fields
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
Information about the optimization.

# Fields
- `verbose::Bool`: enable the verbose output for optimization
- `initial_guess::Bool`: use initial guesses for optimization
- `max_iter::Int64`: the maximum number of iterations for the optimization
- `which_quad_term::String`: choose main quadratic operator (H or F) to use for computation
- `reproject::Bool`: use reprojection method for derivative data
- `SIGE::Bool`: use successive initial guess estimation
- `provide_reduced_orders::Bool`: provide reduced orders for the basis
"""
@with_kw mutable struct opt_settings
    verbose::Bool = false
    initial_guess::Bool = false
    max_iter::Int64 = 3000
    which_quad_term::String = "F"
    reproject::Bool = false
    SIGE::Bool = false  # Successive Initial Guess Estimation
end


"""
Least-Squares Operator Inference.

# Fields
- `method::String`: the name of the method
- `system::sys_struct`: the system structure
- `vars::vars`: the system variables
- `data::data`: the data
- `optim::opt_settings`: the optimization settings
- `λ::Real`: the Tikhonov regularization parameter
- `pinv_tol::Real`: the tolerance for the least square pseudo inverse
"""
@with_kw mutable struct LS_options <: Abstract_Options
    method::String = "LS"
    system::sys_struct = sys_struct()
    vars::vars = vars()
    data::data = data()
    optim::opt_settings = opt_settings()
    λ::Real = 0
    pinv_tol::Real = 1e-6
end


"""
Non-Constrained Operator Inference.

# Fields
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
Energy-Preserving Hard Equality Constraint Operator Inference.

# Fields
- `method::String`: the name of the method
- `system::sys_struct`: the system structure
- `vars::vars`: the system variables
- `data::data`: the data
- `optim::opt_settings`: the optimization settings
- `λ_lin::Real`: the Tikhonov regularization parameter for linear state operator
- `λ_quad::Real`: the Tikhonov regularization parameter for quadratic state operator
"""
@with_kw mutable struct EPHEC_options <: Abstract_Options
    method::String = "EPHEC"
    system::sys_struct = sys_struct()
    vars::vars = vars()
    data::data = data()
    optim::opt_settings = opt_settings()
    λ_lin::Real = 0
    λ_quad::Real = 0
end


"""
Energy-Preserving Soft Inequality Constraint Operator Inference.

# Fields
- `method::String`: the name of the method
- `system::sys_struct`: the system structure
- `vars::vars`: the system variables
- `data::data`: the data
- `optim::opt_settings`: the optimization settings
- `λ_lin::Real`: the Tikhonov regularization parameter for linear state operator
- `λ_quad::Real`: the Tikhonov regularization parameter for quadratic state operator
- `ϵ::Real`: soft constraint radius
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
end


"""
Energy-Preserving Penalty Operator Inference.

# Fields
- `method::String`: the name of the method
- `system::sys_struct`: the system structure
- `vars::vars`: the system variables
- `data::data`: the data
- `optim::opt_settings`: the optimization settings
- `λ_lin::Real`: the Tikhonov regularization parameter for linear state operator
- `λ_quad::Real`: the Tikhonov regularization parameter for quadratic state operator
- `α::Float64`: the weight for the energy-preserving term in the cost function
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
end
