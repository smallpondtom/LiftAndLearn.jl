"""
    interpolate_matrix_elements(z::Vector, matrices::Vector, ẑ)

Interpolate each element of a set of matrices at a given parameter value.

# Arguments
- `z::Vector`: The vector of parameter values.
- `matrices::Vector`: A vector of matrices corresponding to each parameter value.
- `ẑ`: The point at which to interpolate.
- `order::Int`: The order of the spline interpolation.

# Returns
- The interpolated matrix at `ẑ`.
"""
function interpolate_matrix_elements(z::Vector, matrices::Vector, ẑ; order::Int=3)
    nrows, ncols = size(matrices[1])
    result = similar(matrices[1])
    @assert order ∈ [3, 4, 5] "Only cubic, quartic, and quintic splines are supported."
    
    # Interpolate each element of the matrices
    for row in 1:nrows
        for col in 1:ncols
            # Extract the values for the current element across all matrices
            f = [mat[row, col] for mat in matrices]
            
            # Compute spline coefficients
            if order == 3
                M = cubic_spline_coefficients(z, f)
            elseif order == 4
                M = quartic_spline_coefficients(z, f)
            elseif order == 5
                M = quintic_spline_coefficients(z, f)
            end
            
            # Interpolate the value at ẑ
            if order == 3
                result[row, col] = cubic_spline_interpolate(z, f, M, ẑ)
            elseif order == 4
                result[row, col] = quartic_spline_interpolate(z, M, ẑ)
            elseif order == 5
                result[row, col] = quintic_spline_interpolate(z, M, ẑ)
            end
        end
    end
    
    return result
end


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
    
    # Natural spline boundary conditions (second derivatives are zero at the boundaries)
    A[1, 1] = 1
    A[n+1, n+1] = 1
    
    # Fill in the tridiagonal system
    for i in 2:n
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 6 * ((f[i+1] - f[i]) / h[i] - (f[i] - f[i-1]) / h[i-1])
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
    
    # The interpolation formula based on cubic splines (consistent with the image):
    term1 = M[i] * (z[i+1] - ẑ)^3 / (6 * h)
    term2 = M[i+1] * (ẑ - z[i])^3 / (6 * h)
    term3 = (f[i] - M[i] * h^2 / 6) * (z[i+1] - ẑ) / h
    term4 = (f[i+1] - M[i+1] * h^2 / 6) * (ẑ - z[i]) / h
    
    return term1 + term2 + term3 + term4
end

"""
    quartic_spline_coefficients(x::Vector, y::Vector)

Calculate the quartic spline coefficients for interpolation.

# Arguments
- `x::Vector`: The vector of parameter values (knots).
- `y::Vector`: The vector of function values at each parameter.

# Returns
- A vector containing the spline coefficients for each interval.
"""
function quartic_spline_coefficients(x::Vector, y::Vector)
    n = length(x) - 1
    h = diff(x)  # Step sizes between each pair of points

    # Total unknowns: 5n
    # Initialize the matrix and RHS vector
    A = zeros(5n, 5n)
    b = zeros(5n)

    # Equation counter
    eq = 1

    # 1. Interpolation conditions at knots
    for i in 1:n
        idx = 5i - 4  # Starting index for coefficients of interval i

        # s_i(x_i) = y_i
        A[eq, idx] = 1  # Coefficient a_{i0}
        b[eq] = y[i]
        eq += 1

        # s_i(x_{i+1}) = y_{i+1}
        A[eq, idx:idx+4] = [1, h[i], h[i]^2, h[i]^3, h[i]^4]
        b[eq] = y[i+1]
        eq += 1
    end

    # 2. Continuity of first derivative at interior knots
    for i in 2:n
        idx_prev = 5(i - 1) - 4
        idx_curr = 5i - 4
        h_prev = h[i - 1]

        # s_{i-1}'(x_i) - s_i'(x_i) = 0
        A[eq, idx_prev+1:idx_prev+4] = [1, 2*h_prev, 3*h_prev^2, 4*h_prev^3]
        A[eq, idx_curr+1] = -1
        b[eq] = 0
        eq += 1
    end

    # 3. Continuity of second derivative at interior knots
    for i in 2:n
        idx_prev = 5(i - 1) - 4
        idx_curr = 5i - 4
        h_prev = h[i - 1]

        # s_{i-1}''(x_i) - s_i''(x_i) = 0
        A[eq, idx_prev+2:idx_prev+4] = [2, 6*h_prev, 12*h_prev^2]
        A[eq, idx_curr+2] = -2
        b[eq] = 0
        eq += 1
    end

    # 4. Continuity of third derivative at interior knots
    for i in 2:n
        idx_prev = 5(i - 1) - 4
        idx_curr = 5i - 4
        h_prev = h[i - 1]

        # s_{i-1}'''(x_i) - s_i'''(x_i) = 0
        A[eq, idx_prev+3:idx_prev+4] = [6, 24*h_prev]
        A[eq, idx_curr+3] = -6
        b[eq] = 0
        eq += 1
    end

    # 5. Boundary conditions
    # Second derivative at the first knot
    idx_first = 1  # Starting index for the first interval
    A[eq, idx_first+2] = 2  # Coefficient of a_{12}
    b[eq] = 0  # Set desired second derivative at x[1]
    eq += 1

    # Third derivative at the first knot
    A[eq, idx_first+3] = 6  # Coefficient of a_{13}
    b[eq] = 0  # Set desired third derivative at x[1]
    eq += 1

    # Second derivative at the last knot
    idx_last = 5n - 4
    h_last = h[end]

    A[eq, idx_last+2:idx_last+4] = [2, 6*h_last, 12*h_last^2]
    b[eq] = 0  # Set desired second derivative at x[end]
    eq += 1

    # Now eq should be equal to 5n + 1 (since we started from eq = 1)
    # Since we have 5n unknowns, eq - 1 should be equal to 5n
    if eq - 1 != 5n
        error("The number of equations does not match the number of unknowns.")
    end

    # Solve for the quartic spline coefficients
    coefficients = A \ b

    return coefficients
end


"""
    quartic_spline_interpolate(x::Vector, coefficients::Vector, x̂)

Interpolate the value at a given point using quartic spline interpolation.

# Arguments
- `x::Vector`: The vector of parameter values (knots).
- `coefficients::Vector`: The vector of quartic spline coefficients.
- `x̂`: The point at which to interpolate.

# Returns
- The interpolated value at `x̂`.
"""
function quartic_spline_interpolate(x::Vector, coefficients::Vector, x̂)
    n = length(x) - 1
    # Find the interval [x_i, x_{i+1}] that contains x̂
    i = findlast(x .<= x̂)
    if i == length(x)
        i -= 1  # Adjust index if x̂ is at the last knot
    end
    h = x̂ - x[i]
    idx = 5i - 4  # Starting index for interval i

    # Extract coefficients for interval i
    a0 = coefficients[idx]
    a1 = coefficients[idx+1]
    a2 = coefficients[idx+2]
    a3 = coefficients[idx+3]
    a4 = coefficients[idx+4]

    # Quartic interpolation formula
    return a0 + a1*h + a2*h^2 + a3*h^3 + a4*h^4
end


"""
    quintic_spline_coefficients(z::Vector, f::Vector)

Calculate the quintic spline coefficients for interpolation.

# Arguments
- `z::Vector`: The vector of parameter values.
- `f::Vector`: The vector of function values at each parameter.

# Returns
- A tuple of vectors containing the spline coefficients for each interval.
"""
function quintic_spline_coefficients(z::Vector, f::Vector)
    n = length(z) - 1
    h = diff(z)  # Step sizes between each pair of points

    # Updated size of the matrix and RHS vector
    A = zeros(6n, 6n)
    b = zeros(6n)

    # Fill in the conditions for function values at the points
    for i in 1:n
        idx = 6i - 5  # Starting index for interval i
        A[2i-1, idx] = 1  # s_i(z_i) = f_i
        A[2i, idx:idx+5] = [1, h[i], h[i]^2, h[i]^3, h[i]^4, h[i]^5]  # s_i(z_{i+1}) = f_{i+1}
        b[2i-1] = f[i]
        b[2i] = f[i+1]
    end

    # Enforce continuity of derivatives at interior points
    for i in 2:n
        row = 2n + 4*(i - 2) + 1
        h_im1 = h[i - 1]
        idx_prev = 6*(i - 1) - 5
        idx_curr = 6i - 5

        # First derivative continuity
        A[row, idx_prev:idx_prev+5] = [0, 1, 2*h_im1, 3*h_im1^2, 4*h_im1^3, 5*h_im1^4]
        A[row, idx_curr+1] = -1  # Subtract derivative from next interval

        # Second derivative continuity
        A[row+1, idx_prev:idx_prev+5] = [0, 0, 2, 6*h_im1, 12*h_im1^2, 20*h_im1^3]
        A[row+1, idx_curr+2] = -2  # Subtract second derivative from next interval

        # Third derivative continuity
        A[row+2, idx_prev:idx_prev+5] = [0, 0, 0, 6, 24*h_im1, 60*h_im1^2]
        A[row+2, idx_curr+3] = -6  # Subtract third derivative from next interval

        # Fourth derivative continuity
        A[row+3, idx_prev:idx_prev+5] = [0, 0, 0, 0, 24, 120*h_im1]
        A[row+3, idx_curr+4] = -24  # Subtract fourth derivative from next interval
    end

    # Add boundary conditions at the first point (z[1])
    idx_first = 1  # Starting index for the first interval
    # First derivative at z[1]
    A[6n - 3, idx_first+1] = 1  # Coefficient of a1 in the first interval
    b[6n - 3] = 0  # Set desired first derivative value at z[1]

    # Second derivative at z[1]
    A[6n - 2, idx_first+2] = 2  # Coefficient of 2a2
    b[6n - 2] = 0  # Set desired second derivative value at z[1]

    # Add boundary conditions at the last point (z[n+1])
    h_n = h[end]
    idx_last = 6n - 5  # Starting index for the last interval
    # First derivative at z[n+1]
    A[6n - 1, idx_last+1:idx_last+5] = [
        1,
        2*h_n,
        3*h_n^2,
        4*h_n^3,
        5*h_n^4
    ]
    b[6n - 1] = 0  # Set desired first derivative value at z[n+1]

    # Second derivative at z[n+1]
    A[6n, idx_last+2:idx_last+5] = [
        2,
        6*h_n,
        12*h_n^2,
        20*h_n^3
    ]
    b[6n] = 0  # Set desired second derivative value at z[n+1]

    # Solve for the quintic spline coefficients
    coefficients = A \ b

    return coefficients
end


"""
    quintic_spline_interpolate(z::Vector, coefficients::Vector, ẑ)

Interpolate the value at a given point using quintic spline interpolation.

# Arguments
- `z::Vector`: The vector of parameter values.
- `coefficients::Vector`: The vector of quintic spline coefficients.
- `ẑ`: The point at which to interpolate.

# Returns
- The interpolated value at `ẑ`.
"""
function quintic_spline_interpolate(z::Vector, coefficients::Vector, ẑ)
    n = length(z) - 1
    # Find the interval [z[i], z[i+1]] that contains ẑ
    i = findlast(z .<= ẑ)
    if i == length(z)
        i -= 1  # Adjust index if ẑ is at the last knot
    end
    h = ẑ - z[i]
    idx = 6i - 5  # Starting index for interval i

    # Extract coefficients for interval i
    a0 = coefficients[idx]
    a1 = coefficients[idx+1]
    a2 = coefficients[idx+2]
    a3 = coefficients[idx+3]
    a4 = coefficients[idx+4]
    a5 = coefficients[idx+5]

    # Quintic interpolation formula
    return a0 + a1*h + a2*h^2 + a3*h^3 + a4*h^4 + a5*h^5
end

# # Example usage
# z = [1.0, 2.0, 3.0]
# matrices = [
#     [1.0 2.0; 3.0 4.0],
#     [1.5 2.5; 3.5 4.5],
#     [2.0 3.0; 4.0 5.0]
# ]
# ẑ = 2.5

# interpolated_matrix = interpolate_matrix_elements(z, matrices, ẑ; order=3)
# println("Interpolated Matrix at ẑ = $ẑ:\n$interpolated_matrix")


# """
#     cubic_spline_coefficients(z::Vector, f::Vector)

# Calculate the cubic spline coefficients for interpolation.

# # Arguments
# - `z::Vector`: The vector of parameter values.
# - `f::Vector`: The vector of function values at each parameter.

# # Returns
# - A tuple of vectors containing the spline coefficients for each interval.
# """
# function cubic_spline_coefficients(z::Vector, f::Vector)
#     n = length(z) - 1
#     h = diff(z)
    
#     # Set up the system of equations to solve for second derivatives
#     A = zeros(n+1, n+1)
#     b = zeros(n+1)
    
#     # Natural spline boundary conditions
#     A[1, 1] = 1
#     A[n+1, n+1] = 1
    
#     # Fill in the tridiagonal system
#     for i in 2:n
#         A[i, i-1] = h[i-1] / 6
#         A[i, i] = (h[i-1] + h[i]) / 3
#         A[i, i+1] = h[i] / 6
#         b[i] = (f[i+1] - f[i]) / h[i] - (f[i] - f[i-1]) / h[i-1]
#     end
    
#     # Solve for second derivatives
#     M = A \ b
    
#     return M
# end

# """
#     cubic_spline_interpolate(z::Vector, f::Vector, M::Vector, ẑ)

# Interpolate the value at a given point using cubic spline interpolation.

# # Arguments
# - `z::Vector`: The vector of parameter values.
# - `f::Vector`: The vector of function values at each parameter.
# - `M::Vector`: The vector of second derivatives at each parameter.
# - `ẑ`: The point at which to interpolate.

# # Returns
# - The interpolated value at `ẑ`.
# """
# function cubic_spline_interpolate(z::Vector, f::Vector, M::Vector, ẑ)
#     # Find the interval [z[i], z[i+1]] that contains ẑ
#     i = findlast(z .<= ẑ)
#     h = z[i+1] - z[i]
    
#     # Cubic spline polynomial
#     term1 = M[i] * (z[i+1] - ẑ)^3 / (6h)
#     term2 = M[i+1] * (ẑ - z[i])^3 / (6h)
#     term3 = (f[i] - M[i] * h^2 / 6) * (z[i+1] - ẑ) / h
#     term4 = (f[i+1] - M[i+1] * h^2 / 6) * (ẑ - z[i]) / h
    
#     return term1 + term2 + term3 + term4
# end