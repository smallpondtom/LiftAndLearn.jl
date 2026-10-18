


# """
#     quintic_spline_coefficients(z::Vector, f::Vector)

# Calculate the quintic spline coefficients for interpolation.

# # Arguments
# - `z::Vector`: The vector of parameter values.
# - `f::Vector`: The vector of function values at each parameter.

# # Returns
# - A tuple of vectors containing the spline coefficients for each interval.
# """
# function quintic_spline_coefficients(z::Vector, f::Vector)
#     n = length(z) - 1
#     h = diff(z)  # Step sizes between each pair of points

#     # Updated size of the matrix and RHS vector
#     A = zeros(6n, 6n)
#     b = zeros(6n)

#     # Fill in the conditions for function values at the points
#     for i in 1:n
#         A[2i-1, 6i-5] = 1
#         A[2i, 6i-5:6i] = [1, h[i], h[i]^2, h[i]^3, h[i]^4, h[i]^5]
#         b[2i-1] = f[i]
#         b[2i] = f[i+1]
#     end

#     # Enforce continuity of first, second, third, and fourth derivatives at interior points
#     for i in 2:n
#         row = 2n + 4*(i-2) + 1
#         h_im1 = h[i-1]

#         # First derivative continuity
#         A[row, 6(i-1)-5:6(i-1)] = [0, 1, 2*h_im1, 3*h_im1^2, 4*h_im1^3, 5*h_im1^4]
#         A[row, 6i-5+1] = -1

#         # Second derivative continuity
#         A[row+1, 6(i-1)-5:6(i-1)] = [0, 0, 2, 6*h_im1, 12*h_im1^2, 20*h_im1^3]
#         A[row+1, 6i-5+2] = -2

#         # Third derivative continuity
#         A[row+2, 6(i-1)-5:6(i-1)] = [0, 0, 0, 6, 24*h_im1, 60*h_im1^2]
#         A[row+2, 6i-5+3] = -6

#         # Fourth derivative continuity
#         A[row+3, 6(i-1)-5:6(i-1)] = [0, 0, 0, 0, 24, 120*h_im1]
#         A[row+3, 6i-5+4] = -24
#     end

#     # Add boundary conditions (example: zero second derivatives at endpoints)
#     A[6n - 3, 3] = 2  # Second derivative at first point
#     b[6n - 3] = 0
#     A[6n - 2, end - 3] = 2
#     A[6n - 2, end - 2] = 6*h[end]
#     A[6n - 2, end - 1] = 12*h[end]^2
#     A[6n - 2, end] = 20*h[end]^3
#     b[6n - 2] = 0

#     # Solve for the quintic spline coefficients
#     coefficients = A \ b

#     return coefficients
# end


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



# """
#     quintic_spline_interpolate(z::Vector, f::Vector, coefficients::Vector, ẑ)

# Interpolate the value at a given point using quintic spline interpolation.

# # Arguments
# - `z::Vector`: The vector of parameter values.
# - `f::Vector`: The vector of function values at each parameter.
# - `coefficients::Vector`: The vector of quintic spline coefficients.
# - `ẑ`: The point at which to interpolate.

# # Returns
# - The interpolated value at `ẑ`.
# """
# function quintic_spline_interpolate(z::Vector, f::Vector, coefficients::Vector, ẑ)
#     n = length(z) - 1
#     # Find the interval [z[i], z[i+1]] that contains ẑ
#     i = findlast(z .<= ẑ)
#     h = z[i+1] - z[i]

#     a0 = coefficients[5*i-4]
#     a1 = coefficients[5*i-3]
#     a2 = coefficients[5*i-2]
#     a3 = coefficients[5*i-1]
#     a4 = coefficients[5*i]
#     a5 = coefficients[5*i+1]

#     # Quintic interpolation formula
#     return a0 + a1*(ẑ - z[i]) + a2*(ẑ - z[i])^2 + a3*(ẑ - z[i])^3 + a4*(ẑ - z[i])^4 + a5*(ẑ - z[i])^5
# end
