"""
    define_dimensions!(Xhat::Matrix, U::Matrix, Y::VecOrMat, options::AbstractOption)

Define the dimensions of the system

## Arguments
- `Xhat::Matrix`: projected data matrix
- `U::Matrix`: input data matrix
- `Y::VecOrMat`: output data matrix
- `options::AbstractOption`: options for the operator inference set by the user
"""
function define_dimensions!(Xhat::Matrix, U::Matrix, Y::VecOrMat, options::AbstractOption)
    # Important dimensions
    n, K = size(Xhat)
    m = options.system.has_control ? size(U, 2) : 0  # make sure that the U-matrix is tall
    l = options.system.has_output ? size(Y, 1) : 0
    v2 = options.system.is_quad ? Int(n * n) : 0
    s2 = options.system.is_quad ? Int(n * (n + 1) / 2) : 0
    v3 = options.system.is_cubic ? Int(n * n * n) : 0
    s3 = options.system.is_cubic ? Int(n * (n + 1) * (n + 2) / 6) : 0
    w1 = options.system.is_bilin ? Int(n * m) : 0

    d = 0
    for (key, val) in options.system.dims
        if key != :K && key != :l && key != :d
            d += val
        end
    end

    options.system.dims = Dict(
        :n => n, :K => K, :m => m, :l => l, 
        :v2 => v2, :s2 => s2, :v3 => v3, :s3 => s3,
        :w1 => w1, :d => d
    )  # create dimension dict
    return nothing
end


# """
# $(SIGNATURES)
# """
# define_dimensions!(Xhat::Matrix, U::Matrix, options::AbstractOption) = define_dimensions!(Xhat, U, zeros(1,1), options)

# """
# $(SIGNATURES)
# """
# define_dimensions!(Xhat::Matrix, Y::VecOrMat, options::AbstractOption) = define_dimensions!(Xhat, zeros(1,1), Y, options)

# """
# $(SIGNATURES)
# """
# define_dimensions!(Xhat::Matrix, options::AbstractOption) = define_dimensions!(Xhat, zeros(1,1), zeros(1,1), options)


# """
#     define_dimensions!(Xhat::Matrix, U::Matrix, options::AbstractOption)

# Define the dimensions of the system (dispatch)

# ## Arguments
# - `Xhat::Matrix`: projected data matrix
# - `U::Matrix`: input data matrix
# - `options::AbstractOption`: options for the operator inference set by the user
# """
# function define_dimensions!(Xhat::Matrix, U::Matrix, options::AbstractOption)
#     # Important dimensions
#     n, m = size(Xhat)
#     p = options.system.has_control ? size(U, 2) : 0  # make sure that the U-matrix is tall
#     v2 = options.system.is_quad ? Int(n * n) : 0
#     s2 = options.system.is_quad ? Int(n * (n + 1) / 2) : 0
#     v3 = options.system.is_cubic ? Int(n * n * n) : 0
#     s3 = options.system.is_cubic ? Int(n * (n + 1) * (n + 2) / 6) : 0
#     w1 = options.system.is_bilin ? Int(n * p) : 0
#     options.system.dims = Dict(
#         :n => n, :m => m, :p => p,
#         :v2 => v2, :s2 => s2, :v3 => v3, :s3 => s3,
#         :w1 => w1
#     )  # create dimension dict
#     return nothing
# end


# """
#     define_dimensions!(Xhat::Matrix, Y::VecOrMat, options::AbstractOption)

# Define the dimensions of the system (dispatch)

# ## Arguments
# - `Xhat::Matrix`: projected data matrix
# - `Y::VecOrMat`: output data matrix
# - `options::AbstractOption`: options for the operator inference set by the user
# """
# function define_dimensions!(Xhat::Matrix, Y::VecOrMat, options::AbstractOption)
#     # Important dimensions
#     n, m = size(Xhat)
#     q = options.system.has_output ? size(Y, 1) : 0
#     v2 = options.system.is_quad ? Int(n * n) : 0
#     s2 = options.system.is_quad ? Int(n * (n + 1) / 2) : 0
#     v3 = options.system.is_cubic ? Int(n * n * n) : 0
#     s3 = options.system.is_cubic ? Int(n * (n + 1) * (n + 2) / 6) : 0
#     w1 = options.system.is_bilin ? Int(n * p) : 0
#     options.system.dims = Dict(
#         :n => n, :m => m, :q => q, 
#         :v2 => v2, :s2 => s2, :v3 => v3, :s3 => s3,
#         :w1 => w1
#     )  # create dimension dict
#     return nothing
# end


# """
#     define_dimensions!(Xhat::Matrix, options::AbstractOption)

# Define the dimensions of the system (dispatch)

# ## Arguments
# - `Xhat::Matrix`: projected data matrix
# - `options::AbstractOption`: options for the operator inference set by the user
# """
# function define_dimensions!(Xhat::Matrix, options::AbstractOption)
#     # Important dimensions
#     n, m = size(Xhat)
#     v2 = options.system.is_quad ? Int(n * n) : 0
#     s2 = options.system.is_quad ? Int(n * (n + 1) / 2) : 0
#     v3 = options.system.is_cubic ? Int(n * n * n) : 0
#     s3 = options.system.is_cubic ? Int(n * (n + 1) * (n + 2) / 6) : 0
#     w1 = options.system.is_bilin ? Int(n * p) : 0
#     options.system.dims = Dict(
#         :n => n, :m => m,
#         :v2 => v2, :s2 => s2, :v3 => v3, :s3 => s3,
#         :w1 => w1
#     )  # create dimension dict
#     return nothing
# end