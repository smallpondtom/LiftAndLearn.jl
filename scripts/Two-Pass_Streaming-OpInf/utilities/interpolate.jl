"""
    interpolate_zero_columns!(mat)

Given a matrix `mat`, this function linearly interpolates any zero values in each
row between known (nonzero) values. The interpolation proceeds in-place.

**Example**:

If a row looks like `[1, 0, 0, 2, 0, 0, 4]`, then for the segment from `1` to `2`,
it will fill in columns 2 and 3 with values by linear interpolation between 1 and 2.
Then for the segment from `2` to `4`, it fills in columns 5 and 6 by linear
interpolation between 2 and 4.

This function assumes:
  - Each row begins and ends with nonzero entries (so you have "anchors" at the
    extremes).
  - Zeros in between should be replaced by linear interpolation from one
    nonzero column to the next.

The function modifies `mat` in-place and returns the same matrix for convenience.
"""
function interpolate_zero_columns!(mat::AbstractMatrix)
    nrows, ncols = size(mat)

    for r in 1:nrows
        # Identify all columns in row r that have nonzero entries
        nonzero_cols = findall(x -> x != 0, mat[r, :])

        # Interpolate between consecutive nonzero "anchors"
        for j in 1:(length(nonzero_cols) - 1)
            c_start = nonzero_cols[j]
            c_end   = nonzero_cols[j+1]

            y_start = mat[r, c_start]
            y_end   = mat[r, c_end]

            # Number of steps in between
            gap = c_end - c_start
            if gap > 1
                # Linear interpolation step
                slope = (y_end - y_start) / gap
                for c in (c_start+1):(c_end-1)
                    mat[r, c] = y_start + slope * (c - c_start)
                end
            end
        end
    end

    return mat
end

"""
    moving_average_interpolate!(mat; window=2, passes=2)

Fill zero entries in each row of `mat` using a simple moving-average scheme.

- For each row, we run `passes` passes.
- On each pass, we scan the row left-to-right, then right-to-left (or vice versa).
- When we see a zero, we replace it with the average of the known (non-zero or previously filled)
  values in a window of size ±`window` columns around that position.

Arguments:
- `mat`    : A matrix (preferably Float64) with some zero entries to fill.
- `window` : How many columns on each side to average over (default = 2).
- `passes` : How many passes to do over each row (default = 2).

Modifies `mat` in-place and returns `mat`.
"""
function moving_average_interpolate!(mat; window=2, passes=2)
    # Convert to Float so we can store fractional averages if needed
    mat .= float.(mat)
    nrows, ncols = size(mat)

    for r in 1:nrows
        
        for p in 1:passes
            #
            # -- Pass 1: left-to-right
            #
            for c in 1:ncols
                if mat[r, c] == 0.0
                    # find the window of indices around c
                    left_idx  = max(1, c - window)
                    right_idx = min(ncols, c + window)
                    # collect known (nonzero) values in that window
                    vals = mat[r, left_idx:right_idx]
                    known = vals[vals .!= 0.0]
                    if !isempty(known)
                        mat[r, c] = mean(known)
                    end
                end
            end

            #
            # -- Pass 2: right-to-left
            #
            for c in ncols:-1:1
                if mat[r, c] == 0.0
                    # find the window of indices around c
                    left_idx  = max(1, c - window)
                    right_idx = min(ncols, c + window)
                    # collect known (nonzero) values in that window
                    vals = mat[r, left_idx:right_idx]
                    known = vals[vals .!= 0.0]
                    if !isempty(known)
                        mat[r, c] = mean(known)
                    end
                end
            end
        end
    end

    return mat
end

"""
    interpolate_matrix_at_times(A::AbstractMatrix, v1::AbstractVector, v2::AbstractVector)

Interpolate the columns of matrix A at new time points.

Arguments:
- `A`: An m×n matrix where each column represents data at a specific time point
- `v1`: A vector of length n containing the time points for each column of A
- `v2`: A vector of length d containing the time points at which to interpolate

Returns:
- `B`: An m×d matrix where each column is the interpolated data at the 
       corresponding time point in v2

The function uses linear interpolation between columns of A to compute values at 
the time points specified in v2. For time points in v2 that are outside the range 
of v1, the function will use the nearest column from A.
"""
function interpolate_matrix_at_times(A::AbstractMatrix, v1::AbstractVector, v2::AbstractVector)
    m, n = size(A)
    d = length(v2)
    
    # Create output matrix
    B = zeros(eltype(A), m, d)
    
    # Check for empty inputs
    if n == 0 || d == 0
        return B
    end
    
    # Ensure v1 is sorted (if not, sort A and v1 together)
    if !issorted(v1)
        p = sortperm(v1)
        A = A[:, p]
        v1 = v1[p]
    end
    
    # For each target time point
    for j in 1:d
        t = v2[j]
        
        # Handle extrapolation cases first
        if t <= v1[1]
            # Before first time point, use first column
            B[:, j] = A[:, 1]
        elseif t >= v1[end]
            # After last time point, use last column
            B[:, j] = A[:, end]
        else
            # Find the index of the largest time in v1 that's <= t
            i = searchsortedlast(v1, t)
            
            # If exactly at a sample point
            if v1[i] == t
                B[:, j] = A[:, i]
            else
                # Linear interpolation between columns i and i+1
                t1, t2 = v1[i], v1[i+1]
                w2 = (t - t1) / (t2 - t1)  # Weight for the second point
                w1 = 1.0 - w2              # Weight for the first point
                
                # Weighted average of the two columns
                B[:, j] = w1 * A[:, i] + w2 * A[:, i+1]
            end
        end
    end
    
    return B
end

using Interpolations

"""
    cubic_interpolate_matrix(A::AbstractMatrix, v1::AbstractVector, v2::AbstractVector)

Interpolate the columns of matrix A using cubic spline interpolation at new time points.

Arguments:
- `A`: An m×n matrix where each column represents data at a specific time point
- `v1`: A vector of length n containing the time points for each column of A
- `v2`: A vector of length d containing the time points at which to interpolate

Returns:
- `B`: An m×d matrix where each column is the interpolated data at the 
       corresponding time point in v2

The function uses cubic spline interpolation for each row of the matrix.
For time points outside the range of v1, extrapolation is performed.
"""
function cubic_interpolate_matrix(A::AbstractMatrix, v1::AbstractVector, v2::AbstractVector)
    m, n = size(A)
    d = length(v2)
    
    # Create output matrix
    B = zeros(eltype(A), m, d)
    
    # Check for empty inputs
    if n == 0 || d == 0
        return B
    end
    
    # Ensure v1 is sorted (if not, sort A and v1 together)
    if !issorted(v1)
        p = sortperm(v1)
        A = A[:, p]
        v1 = v1[p]
    end
    
    # For small number of points, fall back to linear interpolation
    if n < 4
        # Linear interpolation
        for i in 1:m
            itp = linear_interpolation(v1, A[i,:], extrapolation_bc=Line())
            B[i,:] = itp.(v2)
        end
        return B
    end
    
    # For non-uniform knots (arbitrary time points), use Gridded interpolation
    # which works with arbitrary knot points
    for i in 1:m
        # Create cubic interpolation for this row
        itp = CubicSplineInterpolation(
            (range(0, 1, length=length(v1)),), # Normalize to [0,1] range
            A[i,:],
            extrapolation_bc=Line()
        )
        
        # Map the actual times to the normalized domain
        v1_min, v1_max = extrema(v1)
        v1_range = v1_max - v1_min
        
        # Evaluate at each target time point
        for j in 1:d
            # Map the target time to normalized [0,1] domain
            t_norm = (v2[j] - v1_min) / v1_range
            
            # Keep extrapolation within bounds to avoid errors
            t_norm = clamp(t_norm, 0.0, 1.0)
            
            # Evaluate the interpolation
            B[i,j] = itp(t_norm)
        end
    end
    
    return B
end

"""
    cubic_interpolate_matrix_pure(A::AbstractMatrix, v1::AbstractVector, v2::AbstractVector)

Interpolate the columns of matrix A using cubic interpolation at new time points.
Pure implementation without external packages.

Arguments:
- `A`: An m×n matrix where each column represents data at a specific time point
- `v1`: A vector of length n containing the time points for each column of A
- `v2`: A vector of length d containing the time points at which to interpolate

Returns:
- `B`: An m×d matrix where each column is the interpolated data at the 
       corresponding time point in v2
"""
function cubic_interpolate_matrix_pure(A::AbstractMatrix, v1::AbstractVector, v2::AbstractVector)
    m, n = size(A)
    d = length(v2)
    
    # Create output matrix
    B = zeros(eltype(A), m, d)
    
    # Sort inputs if needed
    if !issorted(v1)
        p = sortperm(v1)
        A = A[:, p]
        v1 = v1[p]
    end
    
    # Helper function for cubic interpolation between 4 points
    function cubic_interp(x, x_vals, y_vals)
        # Find which segment this x falls into
        idx = searchsortedlast(x_vals, x)
        
        # Handle boundary cases
        if idx < 2
            idx = 2
        elseif idx > length(x_vals) - 2
            idx = length(x_vals) - 2
        end
        
        # Get 4 points for cubic interpolation
        x0, x1, x2, x3 = x_vals[idx-1:idx+2]
        y0, y1, y2, y3 = y_vals[idx-1:idx+2]
        
        # Normalize to [0,1] interval for numerical stability
        t = (x - x1) / (x2 - x1)
        
        # Cubic Hermite spline coefficients
        h00 = 2t^3 - 3t^2 + 1
        h10 = t^3 - 2t^2 + t
        h01 = -2t^3 + 3t^2
        h11 = t^3 - t^2
        
        # Estimate derivatives using finite differences
        m1 = (y2 - y0) / (x2 - x0)
        m2 = (y3 - y1) / (x3 - x1)
        
        # Apply cubic Hermite interpolation
        return h00*y1 + h10*(x2-x1)*m1 + h01*y2 + h11*(x2-x1)*m2
    end
    
    # For each target time point
    for j in 1:d
        t = v2[j]
        
        # Handle extrapolation cases
        if t <= v1[1]
            B[:, j] = A[:, 1]  # Use first column
        elseif t >= v1[end]
            B[:, j] = A[:, end]  # Use last column
        elseif n < 4
            # Not enough points for cubic, use linear
            idx = searchsortedlast(v1, t)
            t1, t2 = v1[idx], v1[idx+1]
            w2 = (t - t1) / (t2 - t1)
            w1 = 1.0 - w2
            B[:, j] = w1 * A[:, idx] + w2 * A[:, idx+1]
        else
            # For each row, do cubic interpolation
            for i in 1:m
                B[i,j] = cubic_interp(t, v1, A[i,:])
            end
        end
    end
    
    return B
end