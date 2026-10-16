export SystemStructure, VariableStructure, DataStructure, OptimizationSetting
export TikhonovParameter, LSOpInfOption, EPHECOpInfOption, EPSICOpInfOption, EPPOpInfOption

"""
$(TYPEDEF)

Polynomial structure of the system whose operators are to be inferred.

Each field lists the polynomial orders present in the corresponding term of the model
(`0` means the term is absent). For example `state=[1, 2]` means the state equation has a
linear term ``\\mathbf{A}\\mathbf{x}`` and a quadratic term
``\\mathbf{A}_2(\\mathbf{x}\\otimes\\mathbf{x})``.

## Fields
- `state::Union{Array{<:Int,1},Int}=0`: polynomial orders of the state terms (e.g. `[1, 2]` for linear + quadratic).
- `control::Union{Array{<:Int,1},Int}=0`: polynomial orders of the input terms (`1` adds the linear input operator ``\\mathbf{B}``).
- `output::Union{Array{<:Int,1},Int}=0`: polynomial orders of the output terms (`1` adds the linear output operator ``\\mathbf{C}``).
- `coupled_input::Union{Array{<:Int,1},Int}=0`: polynomial orders of the state-input coupling terms (`1` adds the bilinear operator ``\\mathbf{N}``).
- `coupled_output::Union{Array{<:Int,1},Int}=0`: polynomial orders of the state-input coupling terms in the output equation.
- `constant::Int=0`: nonzero to include a constant term ``\\mathbf{K}`` in the state equation.
- `constant_output::Int=0`: nonzero to include a constant term in the output equation.

## Notes
- The orders are represented as integers or vectors of integers.
- `coupled_output` and `constant_output` are currently not used by the core OpInf routines.

## Examples
```julia
SystemStructure(state=[1, 2], control=1, output=1, coupled_input=1, constant=1, constant_output=1)
```

This represents a system that has:
- linear (=1) and quadratic (=2) state terms,
- a linear input term,
- a linear output term,
- a bilinear (state ``\\times`` input) coupling term,
- a constant term in the state equation,
- a constant term in the output equation.
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
- `N::Int64=0`: number of (unlifted) system variables, e.g. `2` for a system in ``(v, w)``.
- `N_lift::Int64=N`: number of system variables after lifting (used by Lift & Learn (LnL)).
"""
@with_kw mutable struct VariableStructure
    N::Int64 = 0
    N_lift::Int64 = N
end


"""
$(TYPEDEF)

Information about the snapshot data.

## Fields
- `Δt::Float64=0.01`: time step between consecutive snapshots.
- `DS::Int64=0`: downsampling rate (not used by the core OpInf routines; available for user scripts).
- `deriv_type::String="FE"`: finite-difference scheme used by `time_derivative_approx` to approximate the time derivative; one of `"FE"`, `"BE"`, `"SI"`, `"FE4"`, `"BE4"`, `"CTD4"`, `"FBCT4"`.
"""
@with_kw mutable struct DataStructure
    Δt::Float64 = 0.01
    DS::Int64 = 0
    deriv_type::String = "FE"
end


"""
$(TYPEDEF)

Settings for the regression / optimization problem.

## Fields
- `verbose::Bool=false`: print the solver output of the (EP-OpInf) optimization.
- `initial_guess::Bool=false`: warm-start the EP-OpInf optimization with the operators passed as `IG`.
- `max_iter::Int64=3000`: maximum number of Ipopt iterations for EP-OpInf.
- `nonredundant_operators::Bool=true`: use the non-redundant (unique Kronecker) form of the polynomial operators (e.g. `A2u`) instead of the redundant form (e.g. `A2`).
- `reproject::Bool=false`: use the reprojection scheme to generate the derivative data (Lift & Learn (LnL) `opinf` method).
- `SIGE::Bool=false`: successive initial guess estimation; the initial guess `IG` may be of lower dimension than the model being learned.
- `with_bnds::Bool=false`: add box bounds to the EP-OpInf optimization variables.
- `linear_solver::String="none"`: name of the HSL linear solver passed to Ipopt (`"none"` keeps the Ipopt default).
- `HSL_lib_path::String="none"`: path to the HSL library used when `linear_solver` is set.
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

Tikhonov regularization parameters for each operator.

These values populate the diagonal of the Tikhonov matrix ``\\boldsymbol{\\Gamma}`` (see
`tikhonov_matrix!`) used when `LSOpInfOption(with_reg=true)`. The redundant and
non-redundant polynomial operators share the same parameter (e.g. `A2` is used for both
`:A2` and `:A2u`).

## Fields
- `A::Union{Real, AbstractArray{Real}}=0.0`: parameter for the linear state operator ``\\mathbf{A}`` (an array assigns one value per column of ``\\mathbf{A}``).
- `A2::Real=0.0`: parameter for the quadratic state operator.
- `A3::Real=0.0`: parameter for the cubic state operator.
- `A4::Real=0.0`: parameter for the quartic state operator.
- `B::Real=0.0`: parameter for the linear input operator ``\\mathbf{B}``.
- `N::Real=0.0`: parameter for the bilinear state-input operator ``\\mathbf{N}``.
- `C::Real=0.0`: parameter for the linear output operator ``\\mathbf{C}`` (used when solving the output least-squares problem in `unpack_operators!`).
- `K::Real=0.0`: parameter for the constant operator ``\\mathbf{K}`` of the state equation.
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

Options for standard (least-squares) Operator Inference (OpInf), with optional Tikhonov
regularization.

When `with_reg=false` the regression is solved by `standard_least_squares`; when
`with_reg=true` it is solved by `tikhonov` with a diagonal Tikhonov matrix
``\\boldsymbol{\\Gamma}`` built from `λ`. The solver fields below are forwarded to those
functions.

## Fields
- `method::Symbol=:LS`: name of the method.
- `system::SystemStructure=SystemStructure()`: polynomial structure of the system.
- `vars::VariableStructure=VariableStructure()`: information about the system variables.
- `data::DataStructure=DataStructure()`: information about the snapshot data (time step, derivative scheme).
- `optim::OptimizationSetting=OptimizationSetting()`: regression settings (e.g. `nonredundant_operators`, `reproject`).
- `λ::TikhonovParameter=TikhonovParameter()`: Tikhonov regularization parameters (used only if `with_reg=true`).
- `with_reg::Bool=false`: use Tikhonov regularization.
- `use_gpu::Bool=false`: solve the regression on the GPU (CUDA.jl or Metal.jl) if one is functional; falls back to the CPU otherwise.
- `use_normal_equations::Bool=false`: solve via the normal equations (passed as `use_normal_equations` to `standard_least_squares` and as `use_normal_form` to `tikhonov`).
- `use_svd_truncation::Bool=false`: allow the SVD-truncation solver in `tikhonov` for ill-conditioned problems (regularized path only).
- `chunk_size::Int=1000`: number of right-hand-side columns processed per batch.
- `tolerance::Real=1e-12`: numerical tolerance (absolute tolerance of the iterative solvers, relative singular-value cutoff of the SVD-truncation solver).
- `algorithm::Union{Function,Nothing}=nothing`: custom LinearSolve.jl algorithm for the non-regularized solve (`nothing` uses the LinearSolve.jl default).
- `use_backslash::Bool=true`: solve with the backslash operator (`D \\ R` for least squares, augmented system ``[\\mathbf{D}; \\boldsymbol{\\Gamma}^{1/2}]`` for Tikhonov).
- `estimate_memory::Bool=false`: log an estimate of the memory needed by the solver.
- `max_iterations::Int=1000`: maximum number of iterations of the iterative solvers.
- `preconditioning::Bool=false`: request diagonal preconditioning in the Tikhonov solver.

## Notes
- `preconditioning` is stored in the Tikhonov solver configuration but is only used by
  `solve_iterative_normal_form`, which is not called by the default solve paths.
"""
@with_kw mutable struct LSOpInfOption <: AbstractOption
    method::Symbol = :LS
    system::SystemStructure = SystemStructure()
    vars::VariableStructure = VariableStructure()
    data::DataStructure = DataStructure()
    optim::OptimizationSetting = OptimizationSetting()
    λ::TikhonovParameter = TikhonovParameter()
    # with_tol::Bool = false      # This options makes it way slower
    with_reg::Bool = false      # tikhonov regularization
    # pinv_tol::Real = 1e-6       # tolerance for the least square pseudo inverse

    use_gpu::Bool = false                         # use GPU for least-squares solve
    use_normal_equations::Bool = false            # use normal equations method
    use_svd_truncation::Bool = false              # use SVD-based truncation
    chunk_size::Int = 1000                        # batch size for LS memory management
    tolerance::Real = 1e-12                       # numerical tolerance for LS solve
    algorithm::Union{Function,Nothing} = nothing  # custom solver method
    use_backslash::Bool = true  # use backslash for least-squares solve
    estimate_memory::Bool = false   # estimate memory usage for the LS
    max_iterations::Int = 1000  # maximum iterations for iterative methods
    preconditioning::Bool = false  # use preconditioning for the LS solve
end


"""
$(TYPEDEF)

Options for Energy-Preserving OpInf (EP-OpInf) with hard equality constraints (EPHEC).

## Fields
- `method::Symbol=:EPHEC`: name of the method (used by `epopinf` for dispatch).
- `system::SystemStructure=SystemStructure()`: polynomial structure of the system.
- `vars::VariableStructure=VariableStructure()`: information about the system variables.
- `data::DataStructure=DataStructure()`: information about the snapshot data.
- `optim::OptimizationSetting=OptimizationSetting()`: optimization settings.
- `λ_lin::Real=0`: Tikhonov regularization parameter for the linear state operator.
- `λ_quad::Real=0`: Tikhonov regularization parameter for the quadratic state operator.
- `linear_operator_bounds::Tuple{Float64, Float64}=(0.0, 0.0)`: lower/upper bounds of the linear operator entries (used if `optim.with_bnds=true`).
- `quad_operator_bounds::Tuple{Float64, Float64}=(0.0, 0.0)`: lower/upper bounds of the quadratic operator entries (used if `optim.with_bnds=true`).
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

Options for Energy-Preserving OpInf (EP-OpInf) with soft inequality constraints (EPSIC).

## Fields
- `method::Symbol=:EPSIC`: name of the method (used by `epopinf` for dispatch).
- `system::SystemStructure=SystemStructure()`: polynomial structure of the system.
- `vars::VariableStructure=VariableStructure()`: information about the system variables.
- `data::DataStructure=DataStructure()`: information about the snapshot data.
- `optim::OptimizationSetting=OptimizationSetting()`: optimization settings.
- `λ_lin::Real=0`: Tikhonov regularization parameter for the linear state operator.
- `λ_quad::Real=0`: Tikhonov regularization parameter for the quadratic state operator.
- `ϵ::Real=0.1`: radius of the soft constraint; the energy-preserving residuals are constrained to ``[-\\epsilon, \\epsilon]``.
- `linear_operator_bounds::Tuple{Float64, Float64}=(0.0, 0.0)`: lower/upper bounds of the linear operator entries (used if `optim.with_bnds=true`).
- `quad_operator_bounds::Tuple{Float64, Float64}=(0.0, 0.0)`: lower/upper bounds of the quadratic operator entries (used if `optim.with_bnds=true`).
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

Options for Energy-Preserving OpInf (EP-OpInf) with an energy-preserving penalty (EPP).

## Fields
- `method::Symbol=:EPP`: name of the method (used by `epopinf` for dispatch).
- `system::SystemStructure=SystemStructure()`: polynomial structure of the system.
- `vars::VariableStructure=VariableStructure()`: information about the system variables.
- `data::DataStructure=DataStructure()`: information about the snapshot data.
- `optim::OptimizationSetting=OptimizationSetting()`: optimization settings.
- `λ_lin::Real=0`: Tikhonov regularization parameter for the linear state operator.
- `λ_quad::Real=0`: Tikhonov regularization parameter for the quadratic state operator.
- `α::Float64=1.0`: weight of the energy-preserving penalty term in the cost function.
- `linear_operator_bounds::Tuple{Float64, Float64}=(0.0, 0.0)`: lower/upper bounds of the linear operator entries (used if `optim.with_bnds=true`).
- `quad_operator_bounds::Tuple{Float64, Float64}=(0.0, 0.0)`: lower/upper bounds of the quadratic operator entries (used if `optim.with_bnds=true`).
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
