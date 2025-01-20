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