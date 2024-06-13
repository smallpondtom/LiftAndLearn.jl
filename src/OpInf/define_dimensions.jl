"""
    define_dimensions!(Xhat::Matrix, U::Matrix, Y::VecOrMat, options::AbstractOption)

Define the dimensions of the system

## Arguments
- `Xhat::AbstractArray`: projected data matrix
- `U::AbstractArray`: input data matrix
- `Y::AbstractArray`: output data matrix
- `options::AbstractOption`: options for the operator inference set by the user
"""
function define_dimensions!(Xhat::AbstractArray, U::AbstractArray, Y::AbstractArray, options::AbstractOption)
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
            if key == :s2
                d += (options.optim.which_quad_term == "F") * val
            elseif key == :v2
                d += (options.optim.which_quad_term == "R") * val
            elseif key == :s3
                d += (options.optim.which_cubic_term == "E") * val
            elseif key == :v3
                d += (options.optim.which_cubic_term == "G") * val
            else
                d += val 
            end
        end
    end
    d += (options.system.has_const) * 1  # if has constant term

    options.system.dims = Dict(
        :n => n, :K => K, :m => m, :l => l, 
        :v2 => v2, :s2 => s2, :v3 => v3, :s3 => s3,
        :w1 => w1, :d => d
    )  # create dimension dict
    return nothing
end
