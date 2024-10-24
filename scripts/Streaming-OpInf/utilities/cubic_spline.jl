"""
    cubic_spline_coefficients(z::Vector, f::Vector)

Calculate the cubic spline coefficients for interpolation.

# Arguments
- `z::Vector`: The vector of parameter values.
- `f::Vector`: The vector of function values at each parameter.

# Returns
- A tuple of vectors containing the spline coefficients for each interval.
"""
function cubic_spline_coefficients(z::Vector, f::Vector)
    n = length(z) - 1
    h = diff(z)
    
    # Set up the system of equations to solve for second derivatives
    A = zeros(n+1, n+1)
    b = zeros(n+1)
    
    # Natural spline boundary conditions
    A[1, 1] = 1
    A[n+1, n+1] = 1
    
    # Fill in the tridiagonal system
    for i in 2:n
        A[i, i-1] = h[i-1] / 6
        A[i, i] = (h[i-1] + h[i]) / 3
        A[i, i+1] = h[i] / 6
        b[i] = (f[i+1] - f[i]) / h[i] - (f[i] - f[i-1]) / h[i-1]
    end
    
    # Solve for second derivatives
    M = A \ b
    
    return M
end

"""
    cubic_spline_interpolate(z::Vector, f::Vector, M::Vector, ẑ)

Interpolate the value at a given point using cubic spline interpolation.

# Arguments
- `z::Vector`: The vector of parameter values.
- `f::Vector`: The vector of function values at each parameter.
- `M::Vector`: The vector of second derivatives at each parameter.
- `ẑ`: The point at which to interpolate.

# Returns
- The interpolated value at `ẑ`.
"""
function cubic_spline_interpolate(z::Vector, f::Vector, M::Vector, ẑ)
    # Find the interval [z[i], z[i+1]] that contains ẑ
    i = findlast(z .<= ẑ)
    h = z[i+1] - z[i]
    
    # Cubic spline polynomial
    term1 = M[i] * (z[i+1] - ẑ)^3 / (6h)
    term2 = M[i+1] * (ẑ - z[i])^3 / (6h)
    term3 = (f[i] - M[i] * h^2 / 6) * (z[i+1] - ẑ) / h
    term4 = (f[i+1] - M[i+1] * h^2 / 6) * (ẑ - z[i]) / h
    
    return term1 + term2 + term3 + term4
end

"""
    interpolate_matrix_elements(z::Vector, matrices::Vector, ẑ)

Interpolate each element of a set of matrices at a given parameter value.

# Arguments
- `z::Vector`: The vector of parameter values.
- `matrices::Vector`: A vector of matrices corresponding to each parameter value.
- `ẑ`: The point at which to interpolate.

# Returns
- The interpolated matrix at `ẑ`.
"""
function interpolate_matrix_elements(z::Vector, matrices::Vector, ẑ)
    nrows, ncols = size(matrices[1])
    result = similar(matrices[1])
    
    # Interpolate each element of the matrices
    for row in 1:nrows
        for col in 1:ncols
            # Extract the values for the current element across all matrices
            f = [mat[row, col] for mat in matrices]
            
            # Compute spline coefficients
            M = cubic_spline_coefficients(z, f)
            
            # Interpolate the value at ẑ
            result[row, col] = cubic_spline_interpolate(z, f, M, ẑ)
        end
    end
    
    return result
end

# # Example usage
# using LinearAlgebra
# z = [1.0, 2.0, 3.0]
# matrices = [
#     [1.0 2.0; 3.0 4.0],
#     [1.5 2.5; 3.5 4.5],
#     [2.0 3.0; 4.0 5.0]
# ]
# ẑ = 2.5

# interpolated_matrix = interpolate_matrix_elements(z, matrices, ẑ)
# println("Interpolated Matrix at ẑ = $ẑ:\n$interpolated_matrix")