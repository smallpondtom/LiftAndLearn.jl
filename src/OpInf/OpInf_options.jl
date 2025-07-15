export SystemStructure, VariableStructure, DataStructure, OptimizationSetting
export TikhonovParameter, LSOpInfOption, NCOpInfOption, EPHECOpInfOption, EPSICOpInfOption, EPPOpInfOption

""" 
$(TYPEDEF)

Structure of the given system.

## Fields
- `state::Union{Array{<:Int,1},Int}`: the state variables
- `control::Union{Array{<:Int,1},Int}`: the control variables
- `output::Union{Array{<:Int,1},Int}`: the output variables
- `coupled_input::Union{Array{<:Int,1},Int}`: the coupled input variables
- `coupled_output::Union{Array{<:Int,1},Int}`: the coupled output variables
- `constant::Int`: the constant variables
- `constant_output::Int`: the constant output variables

## Note
- The variables are represented as integers or arrays of integers. 

## Example
```julia
SystemStructure(state=[1, 2], control=1, output=1, coupled_input=1, constant=1, constant_output=1)
```

This represents a system that has:
- linear (=1) and quadratic (=2) state variables
- linear control variable
- linear output variable
- Bilinear input variable (coupling of state x input)
- constant variable in the state equation
- constant variable in the state-output equation
"""
@with_kw mutable struct SystemStructure
    state::Union{Array{<:Int,1},Int} = 0
    control::Union{Array{<:Int,1},Int} = 0
    output::Union{Array{<:Int,1},Int} = 0
    coupled_input::Union{Array{<:Int,1},Int} = 0
    coupled_output::Union{Array{<:Int,1},Int} = 0
    constant::Int = 0
    constant_output::Int = 0
end


"""
$(TYPEDEF)

Information about the system variables.

## Fields
- `N::Int64`: the number of system variables (default: `0`)
- `N_lift::Int64`: the number of lifted system variables (default: `N`)
"""
@with_kw mutable struct VariableStructure
    N::Int64 = 0
    N_lift::Int64 = N
end


"""
$(TYPEDEF)

Information about the data.

## Fields
- `Δt::Float64`: the time step or temporal discretization (default: `0.01`)
- `DS::Int64`: the downsampling rate (default: `0`)
- `deriv_type::String`: the derivative scheme, e.g. "F"orward "E"uler (default: `"FE"`)
"""
@with_kw mutable struct DataStructure
    Δt::Float64 = 0.01
    DS::Int64 = 0
    deriv_type::String = "FE"
end


"""
$(TYPEDEF)

Information about the optimization.

## Fields
- `verbose::Bool`: enable the verbose output for optimization (default: `false`)
- `initial_guess::Bool`: use initial guesses for optimization (default: `false`)
- `max_iter::Int64`: the maximum number of iterations for the optimization (default: `3000`)
- `nonredundant_operators::Bool`: use nonredundant operators for optimization (default: `true`)
- `reproject::Bool`: use reprojection method for derivative data (default: `false`)
- `SIGE::Bool`: use successive initial guess estimation (default: `false`)
- `with_bnds::Bool`: add bounds to the variables (default: `false`)
- `linear_solver::String`: the linear solver to use for optimization (default: `"none"`)
- `HSL_lib_path::String`: the path to the HSL library (default: `"none"`)
"""
@with_kw mutable struct OptimizationSetting
    verbose::Bool = false
    initial_guess::Bool = false
    max_iter::Int64 = 3000
    nonredundant_operators::Bool = true
    reproject::Bool = false
    SIGE::Bool = false  # Successive Initial Guess Estimation
    with_bnds::Bool = false  # add bounds to the variables
    linear_solver::String = "none"
    HSL_lib_path::String = "none"
end


"""
$(TYPEDEF)

Tikhonov regularization parameters.

## Fields
- `A::Union{Real, AbstractArray{Real}}`: the Tikhonov regularization parameter for the linear state operator
- `A2::Real`: the Tikhonov regularization parameter for the quadratic state operator
- `A3::Real`: the Tikhonov regularization parameter for the cubic state operator
- `A4::Real`: the Tikhonov regularization parameter for the quartic state operator
- `B::Real`: the Tikhonov regularization parameter for the linear input operator
- `N::Real`: the Tikhonov regularization parameter for the bilinear state-input operator
- `C::Real`: the Tikhonov regularization parameter for the constant operator
- `K::Real`: the Tikhonov regularization parameter for the constant output operator
"""
@with_kw mutable struct TikhonovParameter
    A::Union{Real, AbstractArray{Real}} = 0.0
    A2::Real = 0.0
    A3::Real = 0.0
    A4::Real = 0.0
    B::Real = 0.0
    N::Real = 0.0
    C::Real = 0.0
    K::Real = 0.0
end


"""
$(TYPEDEF)

Standard least-squares Operator Inference.

## Fields
- `method::Symbol`: the name of the method (default: `:LS`)
- `system::SystemStructure`: the system structure (default: `SystemStructure()`)
- `vars::VariableStructure`: the system variables (default: `VariableStructure()`)
- `data::DataStructure`: the data (default: `DataStructure()`)
- `optim::OptimizationSetting`: the optimization settings (default: `OptimizationSetting()`)
- `λ::TikhonovParameter`: the Tikhonov regularization parameters (default: `TikhonovParameter()`)
- `with_tol::Bool`: the option to use tolerance for the least square pseudo inverse (default: `false`)
- `with_reg::Bool`: the option to use Tikhonov regularization (default: `false`)
- `pinv_tol::Real`: the tolerance for the least square pseudo inverse (default: `1e-6`)
- `use_gpu::Bool`: the option to use GPU for least-squares solve (default: `false`)
- `use_backslash::Bool`: the option to use backslash for least-squares solve (default: `true`)
"""
@with_kw mutable struct LSOpInfOption <: AbstractOption
    method::Symbol = :LS
    system::SystemStructure = SystemStructure()
    vars::VariableStructure = VariableStructure()
    data::DataStructure = DataStructure()
    optim::OptimizationSetting = OptimizationSetting()
    λ::TikhonovParameter = TikhonovParameter()
    with_tol::Bool = false      # This options makes it way slower
    with_reg::Bool = false      # tikhonov regularization
    pinv_tol::Real = 1e-6       # tolerance for the least square pseudo inverse

    use_gpu::Bool = false                         # use GPU for least-squares solve
    use_normal_equations::Bool = false            # use normal equations method
    chunk_size::Int = 1000                        # batch size for LS memory management
    tolerance::Real = 1e-12                       # numerical tolerance for LS solve
    algorithm::Union{Function,Nothing} = nothing  # custom solver method
    use_backslash::Bool = true  # use backslash for least-squares solve
    estimate_memory::Bool = false   # estimate memory usage for the LS
end


"""
$(TYPEDEF)

Energy-Preserving Hard Equality Constraint Operator Inference.

## Fields
- `method::Symbol`: the name of the method (default: `:NC`)
- `system::SystemStructure`: the system structure (default: `SystemStructure()`)
- `vars::VariableStructure`: the system variables (default: `VariableStructure()`)
- `data::DataStructure`: the data (default: `DataStructure()`)
- `optim::OptimizationSetting`: the optimization settings (default: `OptimizationSetting()`)
- `λ_lin::Real`: the Tikhonov regularization parameter for linear state operator (default: `0`)
- `λ_quad::Real`: the Tikhonov regularization parameter for quadratic state operator (default: `0`)
- `linear_operator_bounds::Tuple{Float64, Float64}`: the bounds for the linear operator (default: `(0.0, 0.0)`)
- `quad_operator_bounds::Tuple{Float64, Float64}`: the bounds for the quadratic operator (default: `(0.0, 0.0)`)
"""
@with_kw mutable struct EPHECOpInfOption <: AbstractOption
    method::Symbol = :EPHEC
    system::SystemStructure = SystemStructure()
    vars::VariableStructure = VariableStructure()
    data::DataStructure = DataStructure()
    optim::OptimizationSetting = OptimizationSetting()
    λ_lin::Real = 0
    λ_quad::Real = 0
    linear_operator_bounds::Tuple{Float64, Float64} = (0.0, 0.0)
    quad_operator_bounds::Tuple{Float64, Float64} = (0.0, 0.0)
end


"""
$(TYPEDEF)

Energy-Preserving Soft Inequality Constraint Operator Inference.

## Fields
- `method::Symbol`: the name of the method (default: `:EPSIC`)
- `system::SystemStructure`: the system structure (default: `SystemStructure()`)
- `vars::VariableStructure`: the system variables (default: `VariableStructure()`)
- `data::DataStructure`: the data (default: `DataStructure()`)
- `optim::OptimizationSetting`: the optimization settings (default: `OptimizationSetting()`)
- `λ_lin::Real`: the Tikhonov regularization parameter for linear state operator (default: `0`)
- `λ_quad::Real`: the Tikhonov regularization parameter for quadratic state operator (default: `0`)
- `ϵ::Real`: soft constraint radius (default: `0.1`)
- `linear_operator_bounds::Tuple{Float64, Float64}`: the bounds for the linear operator (default: `(0.0, 0.0)`)
- `quad_operator_bounds::Tuple{Float64, Float64}`: the bounds for the quadratic operator (default: `(0.0, 0.0)`)
"""
@with_kw mutable struct EPSICOpInfOption <: AbstractOption
    method::Symbol = :EPSIC
    system::SystemStructure = SystemStructure()
    vars::VariableStructure = VariableStructure()
    data::DataStructure = DataStructure()
    optim::OptimizationSetting = OptimizationSetting()
    λ_lin::Real = 0
    λ_quad::Real = 0
    ϵ::Real = 0.1
    linear_operator_bounds::Tuple{Float64, Float64} = (0.0, 0.0)
    quad_operator_bounds::Tuple{Float64, Float64} = (0.0, 0.0)
end


"""
$(TYPEDEF)

Energy-Preserving Penalty Operator Inference.

## Fields
- `method::Symbol`: the name of the method (default: `:EPP`)
- `system::SystemStructure`: the system structure (default: `SystemStructure()`)
- `vars::VariableStructure`: the system variables (default: `VariableStructure()`)
- `data::DataStructure`: the data (default: `DataStructure()`)
- `optim::OptimizationSetting`: the optimization settings (default: `OptimizationSetting()`)
- `λ_lin::Real`: the Tikhonov regularization parameter for linear state operator (default: `0`)
- `λ_quad::Real`: the Tikhonov regularization parameter for quadratic state operator (default: `0`)
- `α::Float64`: the weight for the energy-preserving term in the cost function (default: `1.0`)
- `linear_operator_bounds::Tuple{Float64, Float64}`: the bounds for the linear operator (default: `(0.0, 0.0)`)
- `quad_operator_bounds::Tuple{Float64, Float64}`: the bounds for the quadratic operator (default: `(0.0, 0.0)`)
"""
@with_kw mutable struct EPPOpInfOption <: AbstractOption
    method::Symbol = :EPP
    system::SystemStructure = SystemStructure()
    vars::VariableStructure = VariableStructure()
    data::DataStructure = DataStructure()
    optim::OptimizationSetting = OptimizationSetting()
    λ_lin::Real = 0
    λ_quad::Real = 0
    α::Float64 = 1.0
    linear_operator_bounds::Tuple{Float64, Float64} = (0.0, 0.0)
    quad_operator_bounds::Tuple{Float64, Float64} = (0.0, 0.0)
end
